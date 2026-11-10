"""Public native-calibrated, streaming MoE quantization workflow."""
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def add_parser(sub):
    p = sub.add_parser('quantize-moe', help='original HF + native calibration GGUF -> streaming MACaQ VQ GGUF')
    p.add_argument('--hf', required=True, help='original local checkpoint or Hub model ID')
    p.add_argument('--revision', default='main', help='Hub revision, resolved to a commit before downloading')
    p.add_argument('--teacher', type=Path, required=True, help='matching original-basis GGUF used only to collect activations')
    p.add_argument('--calibration', type=Path, required=True, help='UTF-8 calibration text')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--work-dir', type=Path, required=True)
    p.add_argument('--bin-dir', type=Path, default=HERE / 'llama.cpp/build/bin')
    p.add_argument('--format', choices=('IQ4_XS', 'IQ4_NL'), default='IQ4_XS')
    p.add_argument('--embedding-format', choices=('auto', 'Q4_K', 'Q6_K', 'Q8_0', 'F16'), default='auto')
    p.add_argument('--gpus', default='0', help='physical GPU indices exposed to calibration and VQ, e.g. 0,1')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--ubatch', type=int, default=128)
    p.add_argument('--chunks', type=int, default=32)
    p.add_argument('--sample-rows', type=int, default=64, help='bounded uniform reservoir rows per expert/projection')
    p.add_argument('--sample-budget-gib', type=float, default=48)
    p.add_argument('--min-expert-tokens', type=int, default=32)
    p.add_argument('--coverage-policy', choices=('error', 'router-weighted'), default='router-weighted',
                   help='calibrate missed experts with soft router weights, or require complete native routing')
    p.add_argument('--vq-sweeps', type=int, default=2)
    p.add_argument('--alpha', type=float, default=0.75)
    p.add_argument('--threads', type=int, default=8, help='CPU encoder workers')
    p.add_argument('--resume', action='store_true', help='reuse verified calibration and completed tensor checkpoints')
    p.add_argument('--calibration-only', action='store_true', help='save verified activation samples before a later --resume export')
    return p


def quantize(args):
    from macaq import sha256, binary
    from moe_stream import converted_tensors, export
    from prepare_weight_only_gguf import source_paths
    import gguf
    sys.path.insert(0, str(HERE.parent / 'quantize'))
    from gguf_metadata import serializable, input_tokenizer_metadata, runtime_libraries
    from runtime_logging import run_logged
    devices = args.gpus.split(',')
    if not devices or any(not value.isdigit() for value in devices) or len(set(devices)) != len(devices):
        raise ValueError('gpus must contain unique, non-negative GPU indices')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.device == 'cuda':
        import torch
        if torch.cuda.device_count() != len(devices):
            raise ValueError('One or more selected GPUs are unavailable')
    if (args.context < 32 or args.chunks < 1 or not 1 <= args.ubatch <= args.context or
            not 1 <= args.sample_rows <= 4096 or args.min_expert_tokens < 1 or args.vq_sweeps < 1 or
            not math.isfinite(args.sample_budget_gib) or not 0 < args.sample_budget_gib <= 1024 or
            not 0 <= args.alpha <= 1 or args.threads < 1):
        raise ValueError('Invalid calibration, reservoir or VQ settings')
    if args.output.exists() or args.output.with_suffix('.recipe.json').exists():
        raise ValueError('The requested output or recipe already exists')
    if not args.resume and (args.output.with_suffix('.part.gguf').exists() or args.work_dir.exists()):
        raise ValueError('Choose fresh output/work paths, or use --resume for an unfinished run')
    invocation = {key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()
                  if key not in ('resume', 'action', 'calibration_only')}
    prior = None
    if args.resume:
        prior = json.loads((args.work_dir / 'recipe.json').read_text())
        if prior.get('completed'):
            raise ValueError('The previous run is already complete')
        changes = {key for key in invocation if prior.get('invocation', {}).get(key) != invocation[key]}
        allowed = {'format', 'embedding_format', 'vq_sweeps', 'min_expert_tokens', 'alpha', 'threads', 'device', 'gpus', 'sample_budget_gib', 'coverage_policy'}
        if (args.work_dir / 'export-state.json').exists() or changes - allowed:
            if changes:
                raise ValueError(f'Resume settings changed: {sorted(changes)}')
    if args.output.resolve().is_relative_to(args.work_dir.resolve()):
        raise ValueError('Output must be outside the work directory')
    if not args.calibration.is_file() or not args.calibration.read_text(encoding='utf-8').strip():
        raise ValueError('Calibration must be a nonempty UTF-8 file')
    teacher_paths = source_paths(args.teacher)
    if any(not p.is_file() for p in teacher_paths):
        raise ValueError('Calibration GGUF or one of its split parts is missing')
    if args.output.resolve() in {p.resolve() for p in [*teacher_paths, args.calibration]}:
        raise ValueError('Output must differ from every input')
    collector = binary(args.bin_dir, 'llama-imatrix')
    libraries = [args.bin_dir / name for name in ('libggml-base.so', 'libggml-base.dylib', 'ggml-base.dll')]
    library = next((p for p in libraries if p.is_file()), None)
    if library is None:
        raise ValueError('The native ggml base library is missing')
    if sys.byteorder != 'little':
        raise ValueError('Direct native encoding currently requires a little-endian host')
    args.work_dir.mkdir(parents=True, exist_ok=args.resume)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recipe = dict(completed=False, method='MACaQ native-teacher calibration + D1 + Hadamard-64 + product VQ',
                  source_hf=args.hf, teacher=str(args.teacher.resolve()), format=args.format,
                  gpus=args.gpus, device=args.device, context=args.context, chunks=args.chunks,
                  ubatch=args.ubatch, sample_rows=args.sample_rows, sample_budget_gib=args.sample_budget_gib,
                  minimum_seen=args.min_expert_tokens, vq_sweeps=args.vq_sweeps, alpha=args.alpha,
                  calibration_sha256=sha256(args.calibration), invocation=invocation, commands=[],
                  collector_sha256=sha256(Path(collector)), runtime_libraries=runtime_libraries(Path(collector)),
                  native_encoder_sha256=sha256(library), intermediate_full_precision_gguf=False)
    if prior:
        if prior['calibration_sha256'] != recipe['calibration_sha256']:
            raise ValueError('Calibration text changed since the unfinished run')
        recipe.update({key:prior[key] for key in ('hf_revision', 'source_files', 'teacher_files',
                       'samples_sha256', 'commands', 'calibration_gpus') if key in prior})
    def save():
        temporary = args.work_dir / 'recipe.tmp'
        temporary.write_text(json.dumps(recipe, indent=2) + '\n')
        temporary.replace(args.work_dir / 'recipe.json')
    def progress(message):
        print(message, flush=True)
        recipe['progress'] = message
        save()
    save()
    hf = Path(args.hf)
    if not hf.is_dir():
        from huggingface_hub import HfApi, snapshot_download
        info = HfApi().model_info(args.hf, revision=recipe.get('hf_revision', args.revision))
        recipe['hf_revision'] = info.sha
        progress(f'Downloading original checkpoint {args.hf} at {info.sha}')
        hf = Path(snapshot_download(args.hf, revision=info.sha, local_dir=args.work_dir / 'hf-source'))
    recipe['hf_directory'] = str(hf.resolve())
    if args.output.resolve().is_relative_to(hf.resolve()):
        raise ValueError('Output must be outside the original checkpoint directory')
    progress('Planning converted tensors and checking calibration-model compatibility')
    model, entries = converted_tensors(hf)
    teacher = gguf.GGUFReader(args.teacher)
    if 'macaq.hadamard_block' in teacher.fields:
        raise ValueError('The calibration GGUF must use the original activation basis, without MACaQ transforms')
    if teacher.fields['general.architecture'].contents() != model.gguf_writer.arch:
        raise ValueError('Calibration-model architecture differs from the HF checkpoint')
    metadata_file = args.work_dir / 'tokenizer-metadata.gguf'
    metadata = gguf.GGUFWriter(metadata_file, model.gguf_writer.arch)
    for key, value in model.gguf_writer.kv_data[0].items():
        if key != 'general.architecture':
            metadata.add_key_value(key, value.value, value.type, sub_type=value.sub_type)
    metadata.write_header_to_file()
    metadata.write_kv_data_to_file()
    metadata.close()
    reference = gguf.GGUFReader(metadata_file)
    def tokens(reader):
        return input_tokenizer_metadata({key:serializable(value.contents()) for key,value in reader.fields.items()
                                         if key.startswith('tokenizer.')})
    if tokens(teacher) != tokens(reference):
        raise ValueError('Calibration GGUF and HF export have different raw-text tokenizer metadata')
    cached_sources = {item['path']:item for item in recipe.get('source_files', [])}
    recipe['source_files'] = []
    for path in sorted(hf.glob('*.safetensors')):
        progress(f'Fingerprinting source {path.name}')
        old = cached_sources.get(str(path.resolve()))
        stat = path.stat()
        checksum = old['sha256'] if old and old.get('mtime_ns') == stat.st_mtime_ns and old['bytes'] == stat.st_size else sha256(path)
        recipe['source_files'].append(dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=checksum))
    cached_teachers = {item['path']:item for item in recipe.get('teacher_files', [])}
    recipe['teacher_files'] = []
    for path in teacher_paths:
        old = cached_teachers.get(str(path.resolve()))
        stat = path.stat()
        checksum = old['sha256'] if old and old.get('mtime_ns') == stat.st_mtime_ns and old['bytes'] == stat.st_size else sha256(path)
        recipe['teacher_files'].append(dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=checksum))
    recipe['hf_config_sha256'] = sha256(hf / 'config.json')
    if prior:
        for key in ('source_files', 'teacher_files'):
            current = {item['path']:item['sha256'] for item in recipe[key]}
            if any(current.get(item['path']) != item['sha256'] for item in prior.get(key, [])):
                raise ValueError(f'Resume input content changed: {key}')
        if prior.get('hf_config_sha256', recipe['hf_config_sha256']) != recipe['hf_config_sha256']:
            raise ValueError('HF configuration changed since the unfinished run')
    recipe['teacher_quantized'] = any(t.tensor_type not in (gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32)
        for path in teacher_paths for t in gguf.GGUFReader(path).tensors)
    samples = args.work_dir / 'calibration-samples.gguf'
    command = [collector, '-m', str(args.teacher), '-f', str(args.calibration), '-o', str(samples),
               '--chunks', str(args.chunks), '-c', str(args.context), '-b', str(args.context),
               '-ub', str(args.ubatch), '-ngl', str(args.gpu_layers), '--no-ppl', '--output-format', 'gguf',
               '--output-frequency', '2147483647']
    if len(devices) > 1:
        command += ['--tensor-split', ','.join('1' for _ in devices)]
    env = dict(os.environ, MACAQ_IMATRIX_SAMPLES=str(args.sample_rows),
               MACAQ_IMATRIX_SAMPLE_GIB=str(args.sample_budget_gib), MACAQ_TRITON_MODE='off')
    if prior and recipe.get('samples_sha256') and samples.is_file():
        if sha256(samples) != recipe['samples_sha256']:
            raise ValueError('Saved calibration samples changed')
        progress('Reusing verified original-basis calibration samples')
    else:
        recipe['commands'].append(command)
        progress(f'Collecting original-basis routed activations; log: {args.work_dir / "calibration.log"}')
        run_logged(command, args.work_dir / 'calibration.log', env=env)
        recipe['samples_sha256'] = sha256(samples)
        recipe['calibration_gpus'] = args.gpus
    if args.calibration_only:
        progress('Calibration samples verified; use --resume to export the model')
        return
    progress('Refining calibrated matrices and writing the final GGUF directly')
    quant_devices = tuple(f'cuda:{i}' for i in range(len(devices))) if args.device == 'cuda' else ('cpu',)
    refinement = export(hf, samples, args.output, library, format=args.format, sweeps=args.vq_sweeps,
        alpha=args.alpha, minimum=args.min_expert_tokens, devices=quant_devices,
        progress=progress, prepared=(model, entries), checkpoint=args.work_dir / 'export-state.json',
        resume=args.resume, encode_threads=args.threads, coverage_policy=args.coverage_policy,
        embedding_format=args.embedding_format)
    recipe['refinement'] = refinement
    recipe['output'] = dict(path=str(args.output.resolve()), bytes=args.output.stat().st_size,
                            sha256=sha256(args.output))
    recipe['completed'] = True
    save()
    args.output.with_suffix('.recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')
    print(f'Saved calibrated VQ model: {args.output}', flush=True)

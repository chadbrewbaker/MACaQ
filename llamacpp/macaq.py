"""Quantize and run text GGUFs with MACAQ vector-codebook refinement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def binary(directory, name):
    path = directory / name
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f'Missing executable {path}; build the vendored llama.cpp first')
    return str(path.resolve())


def inspect_source(model):
    from prepare_weight_only_gguf import source_paths
    import gguf
    paths = source_paths(model)
    for path in paths:
        if not path.is_file():
            raise ValueError(f'Missing source GGUF: {path}')
    readers = [gguf.GGUFReader(path) for path in paths]
    first = readers[0]
    architecture = first.fields['general.architecture'].contents()
    if architecture not in ('qwen3', 'qwen35') or 'macaq.hadamard_block' in first.fields:
        raise ValueError('Source must be an original dense Qwen3/Qwen3.5-family full-precision GGUF')
    full = {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16,
            gguf.GGMLQuantizationType.F32}
    for reader in readers:
        for tensor in reader.tensors:
            if len(tensor.shape) == 2 and tensor.tensor_type not in full:
                raise ValueError(f'Already quantized source tensor: {tensor.name}')
    return paths


def output_precision(paths, quant_format):
    """A GGUF without output.weight uses the embedding as its output head."""
    import gguf
    names = {t.name for path in paths for t in gguf.GGUFReader(path).tensors}
    shared = 'token_embd.weight' in names and 'output.weight' not in names
    head = 'q5_K' if quant_format == 'Q3_K_M' else 'q6_K'
    return (head if shared else 'q4_K'), head, shared


def verify_output_precision(path, embedding_type, output_type):
    import gguf
    types = {t.name: t.tensor_type.name for t in gguf.GGUFReader(path).tensors
             if t.name in ('token_embd.weight', 'output.weight')}
    actual_embedding = types.get('token_embd.weight')
    actual_output = types.get('output.weight', actual_embedding)
    if actual_embedding != embedding_type.upper() or actual_output != output_type.upper():
        raise ValueError(f'Unexpected embedding/output precision: {types}')
    return types


def quantize(args):
    from prepare_weight_only_gguf import prepare
    is_vq = args.format in ("IQ2_XXS", "IQ2_XS", "IQ3_S", "IQ3_XXS", "IQ4_NL", "IQ4_XS")
    if is_vq and (not args.calibration or not getattr(args, "hf", None)):
        raise ValueError("VQ requires --calibration and the original --hf checkpoint")
    paths = inspect_source(args.model)
    embedding, head, shared_output = output_precision(paths, args.format)
    quantizer = binary(args.bin_dir, 'llama-quantize')
    collector = binary(args.bin_dir, 'llama-imatrix') if args.calibration else None
    if (args.output.exists() or args.work_dir.exists() or
            args.output.with_suffix('.part.gguf').exists() or
            args.output.with_suffix('.recipe.json').exists()):
        raise ValueError('Output/work directory already exists; choose fresh paths')
    if args.output.resolve().is_relative_to(args.work_dir.resolve()):
        raise ValueError('Output must be outside the temporary work directory')
    if args.calibration and not args.calibration.read_text(encoding='utf-8').strip():
        raise ValueError('Calibration file is empty')
    if args.imatrix and not args.imatrix.is_file():
        raise ValueError('Imatrix file does not exist')
    inputs = paths + [args.calibration or args.imatrix]
    if args.output.resolve() in {p.resolve() for p in inputs}:
        raise ValueError('Output must differ from every input')
    hf = None
    if is_vq:
        import importlib.util
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.hf)
        text_config = getattr(config, 'text_config', config)
        if text_config.model_type not in ('qwen3', 'qwen3_5', 'qwen3_5_text'):
            raise ValueError('No VQ streaming adapter for this text architecture')
        if text_config.model_type != 'qwen3' and importlib.util.find_spec('transformers.models.qwen3_5') is None:
            raise ValueError('The Qwen3.5-family VQ adapter needs a compatible Transformers version')
        hf = Path(args.hf)
        if not hf.is_dir():
            from huggingface_hub import snapshot_download
            hf = Path(snapshot_download(args.hf))
    args.work_dir.mkdir(parents=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    recipe_path = args.work_dir / 'recipe.json'
    import importlib.metadata
    recipe = dict(completed=False, method=('MACAQ D1 + Hadamard + vector-codebook refinement' if is_vq
                          else 'MACaQ D1 + Hadamard + k-quants'),
                  vq=is_vq, format=args.format, gpu=args.gpu, gpu_layers=args.gpu_layers,
                  cwd=str(Path.cwd()), versions={name: importlib.metadata.version(name)
                                                for name in ('torch', 'numpy')},
                  binaries={str(p): sha256(Path(p)) for p in (quantizer, collector) if p},
                  device=args.device, context=args.context, chunks=args.chunks,
                  ubatch=args.ubatch, alpha=args.alpha, hadamard_block=args.had_block,
                  inputs=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in inputs],
                  commands=[])
    recipe['tensor_precision'] = dict(embedding_type=embedding.upper(),
                                    output_type=head.upper(), shared_output=shared_output)
    def save():
        recipe_path.write_text(json.dumps(recipe, indent=2))
    def run(name, command):
        recipe['commands'].append(command)
        save()
        print(f'{name}: log in {args.work_dir / (name + ".log")}', flush=True)
        with (args.work_dir / (name + '.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    save()
    stats = args.imatrix or args.work_dir / 'calibration.gguf'
    if args.calibration:
        run('calibration', [collector, '-m', str(args.model), '-f', str(args.calibration),
                           '-o', str(stats), '--chunks', str(args.chunks),
                           '-c', str(args.context), '-b', str(args.context),
                           '-ub', str(args.ubatch), '-ngl', str(args.gpu_layers),
                           '--no-ppl', '--output-format', 'gguf'])
    intermediate = args.work_dir / 'transformed-f16.gguf'
    transform = prepare(args.model, stats, intermediate, args.device, args.alpha, args.had_block)
    recipe['transform'] = transform
    recipe['imatrix_sha256'] = sha256(stats)
    partial = args.work_dir / 'quantized.part.gguf'
    quant_options = ['--pure'] if is_vq else []
    if args.format in ('IQ2_XXS', 'IQ2_XS', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS'):
        rotated_stats = args.work_dir / 'calibration-rotated.gguf'
        run('rotated-calibration', [collector, '-m', str(intermediate), '-f', str(args.calibration),
            '-o', str(rotated_stats), '--chunks', str(args.chunks), '-c', str(args.context),
            '-b', str(args.context), '-ub', str(args.ubatch), '-ngl', str(args.gpu_layers),
            '--no-ppl', '--output-format', 'gguf'])
        quant_options += ['--imatrix', str(rotated_stats)]
        recipe['rotated_imatrix_sha256'] = sha256(rotated_stats)
    run('quantization', [quantizer] + quant_options + ['--token-embedding-type', embedding,
                         '--output-tensor-type', head, str(intermediate), str(partial),
                         args.format, str(args.threads)])
    if is_vq:
        recipe['hf'] = str(hf.resolve())
        recipe['hf_config_sha256'] = sha256(hf / 'config.json')
        refined_dir = args.work_dir / 'vq'
        skip_types = ['Q3_K', 'Q4_K', 'Q5_K', 'Q6_K']
        skip_types += [kind for kind in ('IQ4_NL', 'IQ4_XS') if kind != args.format]
        run('vq-refinement', [sys.executable, str(HERE / 'refine/macaq_refine_gguf.py'),
            '--hf', str(hf), '--f16', str(intermediate), '--inputs', str(partial),
            '--out-dir', str(refined_dir), '--calib', str(args.calibration),
            '--n-calib-tokens', str(args.chunks * args.context),
            '--chunk-tokens', str(args.context), '--method', 'block',
            '--n-block-sweeps', str(args.vq_sweeps), '--device', args.device,
            '--no-head', '--skip-types', ','.join(skip_types)])
        refined = refined_dir / 'quantized.part-mrf.gguf'
        stats = json.loads((refined_dir / 'quantized.part-mrf.json').read_text())
        if not stats['tensors'] or not all(t['method'] == 'block' for t in stats['tensors'].values()):
            raise ValueError('No vector-codebook refinement was completed')
        recipe['vq_refinement'] = stats
        recipe['codebook_kind'] = 'product' if args.format in ('IQ4_NL', 'IQ4_XS') else 'lattice'
        partial = refined
    if not partial.is_file() or partial.stat().st_size == 0:
        raise ValueError('Quantizer did not produce an output')
    io_types = verify_output_precision(partial, embedding, head)
    recipe['output'] = dict(path=str(args.output.resolve()), bytes=partial.stat().st_size,
                           sha256=sha256(partial), embedding_type=embedding.upper(),
                           output_type=head.upper(), tensor_types=io_types,
                           shared_output='output.weight' not in io_types)
    # Copy between filesystems without exposing an incomplete destination.
    import shutil
    pending = args.output.with_suffix('.part.gguf')
    if pending.exists():
        raise ValueError(f'Incomplete output already exists: {pending}')
    shutil.copyfile(partial, pending)
    pending.replace(args.output)
    recipe['completed'] = True
    save()
    args.output.with_suffix('.recipe.json').write_text(json.dumps(recipe, indent=2))
    if not args.keep_intermediate:
        intermediate.unlink()
    partial.unlink()
    print(f'Saved {args.output}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    from moe_workflow import add_parser as add_moe_parser
    add_moe_parser(sub)
    q = sub.add_parser('quantize', help='original GGUF + original HF + calibration -> MACAQ VQ GGUF')
    q.add_argument('--model', type=Path, required=True, help='original GGUF or first split')
    q.add_argument('--calibration', type=Path, required=True, help='UTF-8 calibration text')
    q.set_defaults(imatrix=None)
    q.add_argument('--output', type=Path, required=True)
    q.add_argument('--work-dir', type=Path, required=True, help='fresh directory for logs/statistics')
    q.add_argument('--format', choices=('IQ2_XXS', 'IQ2_XS', 'IQ3_S', 'IQ3_XXS', 'IQ4_NL', 'IQ4_XS'), default='IQ3_S')
    q.add_argument('--hf', required=True, help='original HF model ID or local checkpoint for VQ calibration')
    q.add_argument('--vq-sweeps', type=int, default=2)
    q.add_argument('--device', choices=('cpu', 'cuda'), default='cpu', help='weight transform device')
    q.add_argument('--chunks', type=int, default=32)
    q.add_argument('--context', type=int, default=512)
    q.add_argument('--ubatch', type=int, default=128)
    q.add_argument('--threads', type=int, default=8)
    q.add_argument('--alpha', type=float, default=0.75)
    q.add_argument('--had-block', type=int, choices=(32, 64), default=32,
                   help='rotation block size; 64 can use the existing fast runtime transform')
    q.add_argument('--keep-intermediate', action='store_true')
    r = sub.add_parser('run', help='load a GGUF and generate; does not calibrate or quantize')
    r.add_argument('--model', type=Path, required=True)
    r.add_argument('--prompt', required=True)
    r.add_argument('--max-tokens', type=int, default=128)
    r.add_argument('--context', type=int, default=2048)
    r.add_argument('--thinking', action='store_true', help='enable thinking within the token limit')
    r.add_argument('--kernel-dir', type=Path, help='optional ahead-of-time compiled Triton preprocessing kernels')
    for parser in (q, r):
        parser.add_argument('--bin-dir', type=Path, default=HERE / 'llama.cpp/build/bin')
        parser.add_argument('--gpu', type=int, default=0, help='only physical GPU exposed to the process')
        parser.add_argument('--gpu-layers', type=int, default=0, help='offloaded layers; 0 uses CPU')
    args = ap.parse_args()
    if args.action == 'quantize-moe':
        try:
            from moe_workflow import quantize as quantize_moe
            quantize_moe(args)
        except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as error:
            ap.exit(1, f'{error}\n')
        return

    if args.gpu < 0 or args.gpu_layers < 0 or args.context < 32:
        ap.error('GPU/layers must be nonnegative and context must be >=32')
    if args.action == 'quantize' and (args.chunks < 1 or args.threads < 1 or
                                     args.vq_sweeps < 1 or not 1 <= args.ubatch <= args.context or not 0 <= args.alpha <= 1):
        ap.error('Invalid calibration, batch, thread or alpha setting')
    if args.action == 'run' and args.max_tokens < 1:
        ap.error('max-tokens must be positive')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.action == 'run' and args.kernel_dir:
        if not (args.kernel_dir / 'manifest.json').is_file():
            ap.error('kernel-dir must contain the manifest from compile_kernels.py')
        os.environ['MACAQ_TRITON_KERNEL_DIR'] = str(args.kernel_dir.resolve())
    try:
        if args.action == 'quantize':
            quantize(args)
        else:
            if not args.model.is_file():
                raise ValueError(f'Missing GGUF: {args.model}')
            subprocess.run([binary(args.bin_dir, 'llama-completion'), '-m', str(args.model),
                            '-ngl', str(args.gpu_layers), '-c', str(args.context),
                            '-n', str(args.max_tokens), '--temp', '0', '--seed', '42',
                            '-cnv', '--single-turn', '--jinja', '--reasoning',
                            'on' if args.thinking else 'off', '-p', args.prompt], check=True)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        ap.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()

"""MACaQ-R: refine native GGUF codes while preserving their inference layout."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from threading import Lock

HERE = Path(__file__).resolve().parent


def add_parser(sub):
    parser = sub.add_parser('refine', help='MACaQ-R refinement of a native K-quant MoE GGUF')
    parser.add_argument('--model', type=Path, required=True, help='single, native quantized GGUF')
    parser.add_argument('--hf', required=True, help='matching original HF checkpoint or Hub ID')
    parser.add_argument('--revision', default='main', help='HF source revision, resolved before downloading')
    parser.add_argument('--teacher', type=Path, required=True, help='matching original full-precision GGUF')
    parser.add_argument('--calibration', type=Path, required=True, help='representative UTF-8 text')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--bin-dir', type=Path, default=HERE / 'llama.cpp/build/bin')
    parser.add_argument('--gpus', default='0', help='comma-separated physical indices')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--gpu-layers', type=int, default=99)
    parser.add_argument('--context', type=int, default=512)
    parser.add_argument('--ubatch', type=int, default=128)
    parser.add_argument('--chunks', type=int, default=256)
    parser.add_argument('--sample-rows', type=int, default=128)
    parser.add_argument('--sample-budget-gib', type=float, default=32)
    parser.add_argument('--min-expert-tokens', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=64)
    parser.add_argument('--ridge', type=float, default=1.0)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--resume', action='store_true', help='continue a verified unfinished run')
    return parser


def _save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _plain(reader):
    if ('macaq.hadamard_block' in reader.fields
            or any(t.name.endswith('.pre_scale') for t in reader.tensors)):
        raise ValueError('MACaQ-R requires a GGUF in its original activation basis')
    if 'split.count' in reader.fields and int(reader.fields['split.count'].contents()) > 1:
        raise ValueError('Merge split input GGUFs before refinement')


def _tokenizer(reader):
    from gguf_metadata import input_tokenizer_metadata, serializable
    return input_tokenizer_metadata({key: serializable(value.contents())
        for key, value in reader.fields.items() if key.startswith('tokenizer.')})


def _digest_slice(fd, offset, size):
    digest = hashlib.sha256()
    while size:
        data = os.pread(fd, min(size, 8 * 1024**2), offset)
        if not data:
            raise ValueError('Incomplete tensor payload')
        digest.update(data)
        offset += len(data)
        size -= len(data)
    return digest.hexdigest()


def _verify(source, result, changed):
    import numpy as np
    import gguf
    a, b = gguf.GGUFReader(source), gguf.GGUFReader(result)
    if source.stat().st_size != result.stat().st_size or len(a.tensors) != len(b.tensors):
        raise ValueError('Refinement changed the native GGUF layout')
    with source.open('rb') as left, result.open('rb') as right:
        if left.read(a.data_offset) != right.read(b.data_offset):
            raise ValueError('Refinement changed GGUF metadata')
    for x, y in zip(a.tensors, b.tensors):
        if ((x.name, x.tensor_type, x.n_bytes, x.data_offset)
                != (y.name, y.tensor_type, y.n_bytes, y.data_offset)
                or not np.array_equal(x.shape, y.shape)):
            raise ValueError('Tensor layout changed: ' + x.name)
        if x.name not in changed:
            if not np.array_equal(x.data.view(np.uint8), y.data.view(np.uint8)):
                raise ValueError('Unselected tensor changed: ' + x.name)
        else:
            block = gguf.GGML_QUANT_SIZES[x.tensor_type][1]
            left, right = x.data.reshape(-1, block), y.data.reshape(-1, block)
            scales = slice(192, 210) if x.tensor_type == gguf.GGMLQuantizationType.Q6_K else slice(0, 16)
            if not np.array_equal(left[:, scales], right[:, scales]):
                raise ValueError('Native block scales changed: ' + x.name)


def _template_without_comments(template):
    from jinja2 import Environment
    # Map lexer positions back to the original line endings.
    characters = list(re.finditer(r'\r\n|.', template, flags=re.DOTALL))
    normalized = ''.join('\n' if m[0] in ('\r', '\r\n') else m[0] for m in characters)
    offsets = [m.start() for m in characters] + [len(template)]
    cursor, edits = 0, []
    for _, kind, text in Environment().lex(normalized):
        if not text:
            continue
        start = normalized.index(text, cursor)
        cursor = start + len(text)
        if kind == 'comment':
            edits.append((offsets[start], offsets[cursor]))
    for start, end in reversed(edits):
        template = template[:start] + ' ' + template[end:]
    return template


def _publish(partial, output):
    """Write refinement metadata and preserve native tensor payloads."""
    import gguf
    reader = gguf.GGUFReader(partial)
    arch = reader.fields['general.architecture'].contents()
    key = 'qwen35moe.nextn_predict_layers'
    normalize = (arch == 'qwen35moe' and key not in reader.fields
                 and not any('.nextn.' in t.name for t in reader.tensors))
    if normalize:
        blocks = int(reader.fields[arch + '.block_count'].contents())
        if any((match := re.match(r'blk\.(\d+)\.', t.name)) and int(match[1]) >= blocks
               for t in reader.tensors):
            raise ValueError('Declare the input prediction-layer metadata before refinement')
    extra = {'general.quantized_by': 'Huawei-CSL',
             'general.repo_url': 'https://github.com/huawei-csl/MACaQ',
             'macaq.method': 'MACaQ-R'}
    drop_keys = {name for name in reader.fields if name.startswith('quantize.imatrix.')}
    template_key = 'tokenizer.chat_template'
    if template_key in reader.fields:
        template = _template_without_comments(reader.fields[template_key].contents())
        if template != reader.fields[template_key].contents():
            extra[template_key] = template
    if normalize:
        extra[key] = 0
    from macaq_refine_gguf import write_refined_gguf
    pending = output.with_suffix('.normalized.gguf')
    write_refined_gguf(reader, str(pending), {}, extra, drop_keys=drop_keys)
    check = gguf.GGUFReader(pending)
    for name, value in extra.items():
        if name not in check.fields or check.fields[name].contents() != value:
            raise ValueError('Refinement metadata was not saved: ' + name)
    if drop_keys.intersection(check.fields):
        raise ValueError('Output retained obsolete calibration metadata')
    for name, field in reader.fields.items():
        if name in extra or name in drop_keys or name.startswith('GGUF.'):
            continue
        if (name not in check.fields or field.types != check.fields[name].types
                or field.contents() != check.fields[name].contents()):
            raise ValueError('Unrelated GGUF metadata changed: ' + name)
    if len(check.tensors) != len(reader.tensors):
        raise ValueError('Normalized GGUF has an incomplete tensor directory')
    # The writer copied native bytes directly; verify payload identity before publishing.
    with partial.open('rb') as left, pending.open('rb') as right:
        for a, b in zip(reader.tensors, check.tensors):
            if (a.name, a.tensor_type, a.n_bytes) != (b.name, b.tensor_type, b.n_bytes):
                raise ValueError('Normalized GGUF changed tensor layout')
            left.seek(a.data_offset)
            right.seek(b.data_offset)
            remaining = a.n_bytes
            while remaining:
                size = min(remaining, 8 * 1024**2)
                if left.read(size) != right.read(size):
                    raise ValueError('Normalized GGUF changed tensor data')
                remaining -= size
    del reader, check
    pending.replace(output)
    partial.unlink()
    return normalize


def refine(args):
    for directory in (HERE, HERE / 'llama.cpp', HERE / 'llama.cpp/gguf-py',
                      HERE / 'refine', HERE.parent / 'quantize'):
        sys.path.insert(0, str(directory))
    import gguf
    import numpy as np
    import torch
    from gguf.lazy import LazyBase
    from gguf_codecs import make_codec
    from gguf_metadata import sha256, runtime_libraries
    from macaq import binary
    from moe_stream import converted_tensors, check_samples, EXPERT, DENSE
    from moe_coverage import SoftCoverage
    from prepare_weight_only_gguf import source_paths
    from runtime_logging import run_logged
    from uniform_refine import refine_codes

    devices = args.gpus.split(',')
    if not all(value.isdigit() for value in devices) or len(set(devices)) != len(devices):
        raise ValueError('gpus must contain distinct, nonnegative indices')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.device == 'cuda' and torch.cuda.device_count() != len(devices):
        raise ValueError('One or more selected devices are unavailable')
    if (args.context < 32 or args.chunks < 1 or not 1 <= args.ubatch <= args.context
            or not 1 <= args.sample_rows <= 4096 or args.min_expert_tokens < 1
            or args.iterations < 1 or args.threads < 1 or args.gpu_layers < 0
            or not math.isfinite(args.ridge) or args.ridge < 0
            or not math.isfinite(args.sample_budget_gib) or not 0 < args.sample_budget_gib <= 1024):
        raise ValueError('Invalid calibration or refinement settings')
    if not args.model.is_file() or not args.calibration.is_file():
        raise ValueError('Model and calibration files must exist')
    if not args.calibration.read_text(encoding='utf-8').strip():
        raise ValueError('Calibration text is empty')
    partial = args.output.with_suffix('.part.gguf')
    if (args.output.exists() or args.output.with_suffix('.recipe.json').exists()
            or args.output.with_suffix('.normalized.gguf').exists()):
        raise ValueError('Choose a fresh output path')
    if not args.resume and (args.work_dir.exists() or partial.exists()):
        raise ValueError('Choose fresh work/output paths, or resume the unfinished run')
    if args.output.resolve().is_relative_to(args.work_dir.resolve()):
        raise ValueError('Output must be outside the work directory')
    teacher_paths = source_paths(args.teacher)
    if any(not path.is_file() for path in teacher_paths):
        raise ValueError('The teacher GGUF or one of its splits is missing')
    if args.output.resolve() in {path.resolve() for path in [args.model, args.calibration, *teacher_paths]}:
        raise ValueError('Output must differ from every input')
    if sys.byteorder != 'little':
        raise ValueError('Native code refinement requires a little-endian host')
    source = gguf.GGUFReader(args.model)
    teacher = gguf.GGUFReader(args.teacher)
    if source.endianess == gguf.GGUFEndian.BIG or teacher.endianess == gguf.GGUFEndian.BIG:
        raise ValueError('Use little-endian GGUF inputs for native code refinement')
    _plain(source)
    if len(teacher_paths) == 1:
        _plain(teacher)
    if 'macaq.hadamard_block' in teacher.fields:
        raise ValueError('The teacher must use the original activation basis')
    full = {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32}
    if any(t.tensor_type not in full for path in teacher_paths for t in gguf.GGUFReader(path).tensors):
        raise ValueError('Use an original full-precision teacher GGUF')
    collector = binary(args.bin_dir, 'llama-imatrix')
    args.work_dir.mkdir(parents=True, exist_ok=args.resume)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads((args.work_dir / 'recipe.json').read_text()) if args.resume else None
    if old and old.get('completed'):
        raise ValueError('The previous refinement is already complete')
    if old and old.get('requested_revision', args.revision) != args.revision:
        raise ValueError('Resume source revision changed')
    hf = Path(args.hf)
    revision = None
    if not hf.is_dir():
        from huggingface_hub import HfApi, snapshot_download
        revision = HfApi().model_info(args.hf, revision=(old or {}).get('source_revision') or args.revision).sha
        hf = Path(snapshot_download(args.hf, revision=revision,
            allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.jinja', 'LICENSE*'],
            local_dir=args.work_dir / 'hf-source'))
    if args.output.resolve().is_relative_to(hf.resolve()):
        raise ValueError('Output must be outside the original checkpoint directory')
    model, entries = converted_tensors(hf)
    arch = model.gguf_writer.arch
    if any(reader.fields['general.architecture'].contents() != arch for reader in (source, teacher)):
        raise ValueError('GGUF architecture differs from the original HF checkpoint')
    metadata = gguf.GGUFWriter(args.work_dir / 'tokenizer.gguf', arch)
    for key, value in model.gguf_writer.kv_data[0].items():
        if key != 'general.architecture':
            metadata.add_key_value(key, value.value, value.type, sub_type=value.sub_type)
    metadata.write_header_to_file()
    metadata.write_kv_data_to_file()
    metadata.close()
    expected_tokens = _tokenizer(gguf.GGUFReader(args.work_dir / 'tokenizer.gguf'))
    if any(_tokenizer(reader) != expected_tokens for reader in (source, teacher)):
        raise ValueError('GGUF and HF raw-text tokenizers differ')
    tensors = {t.name: t for t in source.tensors}
    calibrated = [name for name in entries if EXPERT.fullmatch(name) or DENSE.fullmatch(name)]
    kinds = {gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q5_K, gguf.GGMLQuantizationType.Q6_K}
    names = [name for name in calibrated if name in tensors and tensors[name].tensor_type in kinds]
    if not names or not any(EXPERT.fullmatch(name) for name in names):
        raise ValueError('No supported native K-quant MoE projections were found')
    for name in names:
        if tuple(tensors[name].shape[::-1]) != tuple(entries[name].shape):
            raise ValueError('Original and quantized tensor shapes differ: ' + name)
    def identity(path):
        return dict(file=path.name, bytes=path.stat().st_size, sha256=sha256(path))
    settings = {key: getattr(args, key) for key in ('context', 'chunks', 'ubatch', 'sample_rows',
        'sample_budget_gib', 'min_expert_tokens', 'iterations', 'ridge', 'threads', 'device', 'gpus', 'gpu_layers', 'revision')}
    linked = runtime_libraries(Path(collector)) or {}
    fingerprint = dict(model=identity(args.model), teacher=[identity(p) for p in teacher_paths],
        hf_config=identity(hf / 'config.json'), hf_weights=[identity(p) for p in sorted(hf.glob('*.safetensors'))],
        calibration=identity(args.calibration), collector=identity(Path(collector)), settings=settings,
        refinement_code=sha256(Path(__file__)), kernel_code=sha256(HERE / 'refine/uniform_refine.py'),
        codecs=sha256(HERE / 'refine/gguf_codecs.py'), coverage=sha256(HERE / 'moe_coverage.py'),
        converter=sha256(Path(sys.modules[type(model).__module__].__file__)),
        collector_libraries={name: item['sha256'] for name, item in linked.items()})
    if old and old['inputs'] != fingerprint:
        raise ValueError('Resume inputs or refinement settings changed')
    recipe = old or dict(completed=False, method='MACaQ-R', source_revision=revision,
                         requested_revision=args.revision, inputs=fingerprint, tensors={})
    recipe_file = args.work_dir / 'recipe.json'
    _save(recipe_file, recipe)
    samples = args.work_dir / 'calibration-samples.gguf'
    if recipe.get('samples_sha256'):
        if not samples.is_file() or sha256(samples) != recipe['samples_sha256']:
            raise ValueError('Saved calibration samples changed')
    else:
        command = [collector, '-m', str(args.teacher), '-f', str(args.calibration), '-o', str(samples),
            '--chunks', str(args.chunks), '-c', str(args.context), '-b', str(args.context),
            '-ub', str(args.ubatch), '-ngl', str(args.gpu_layers), '-t', str(args.threads),
            '-tb', str(args.threads), '--no-ppl', '--output-format', 'gguf', '--output-frequency', '2147483647']
        if len(devices) > 1 and args.device == 'cuda':
            command += ['--tensor-split', ','.join('1' for _ in devices)]
        print('Collecting original-basis calibration samples', flush=True)
        env = dict(os.environ, MACAQ_IMATRIX_SAMPLES=str(args.sample_rows),
            MACAQ_IMATRIX_SAMPLE_GIB=str(args.sample_budget_gib), MACAQ_TRITON_MODE='off')
        run_logged(command, args.work_dir / 'calibration.log', env)
        recipe['samples_sha256'] = sha256(samples)
        _save(recipe_file, recipe)
    sample_reader = gguf.GGUFReader(samples)
    banks, aliases = check_samples(calibrated, entries, sample_reader, args.min_expert_tokens, allow_uncovered=True)
    coverage = SoftCoverage(hf, banks, aliases)
    if not partial.exists():
        if recipe['tensors']:
            raise ValueError('Partial output is missing')
        shutil.copyfile(args.model, partial)
    if partial.stat().st_size != args.model.stat().st_size:
        raise ValueError('Partial output has the wrong size')
    fd = os.open(partial, os.O_RDWR)
    lock = Lock()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    quant_devices = [torch.device(f'cuda:{i}') for i in range(len(devices))] if args.device == 'cuda' else [torch.device('cpu')]

    def worker(rank):
        device = quant_devices[rank]
        context = torch.cuda.device(device) if device.type == 'cuda' else nullcontext()
        with context, torch.no_grad():
            for index, name in enumerate(names):
                if index % len(quant_devices) != rank:
                    continue
                tensor = tensors[name]
                if name in recipe['tensors']:
                    if _digest_slice(fd, tensor.data_offset, tensor.n_bytes) != recipe['tensors'][name]['sha256']:
                        raise ValueError('Completed tensor checksum differs: ' + name)
                    continue
                with lock:
                    data = entries.pop(name)
                    values = LazyBase.to_eager(data)
                shape = tuple(values.shape)
                experts, rows, width = shape if len(shape) == 3 else (1, *shape)
                raw = tensor.data.reshape(experts, rows, -1)
                key = aliases[name]
                detail = []
                for expert in range(experts):
                    seen = int(banks[key + '.sample_seen'][expert])
                    retained = min(seen, banks[key + '.samples'].shape[1])
                    mode = 'native-routed'
                    if seen < args.min_expert_tokens:
                        x, _ = coverage.inputs(name, expert, device)
                        mode = 'router-weighted-fallback'
                    else:
                        x = torch.from_numpy(np.array(banks[key + '.samples'][expert, :retained], copy=True)).to(device)
                    target = torch.from_numpy(np.array(values[expert] if len(shape) == 3 else values,
                                                       dtype=np.float32, copy=True)).to(device)
                    hessian = (x.T @ x / len(x)).contiguous()
                    if not torch.isfinite(hessian).all() or float(hessian.diagonal().sum()) <= 0:
                        raise ValueError('Invalid activation Hessian: ' + name)
                    hessian.diagonal().add_(args.ridge * hessian.diagonal().mean())
                    codec = make_codec(tensor.tensor_type, device)
                    state = codec.decode(np.array(raw[expert], copy=True))
                    initial = state.q.clone()
                    error = state.vals - target
                    before = float(((error @ hessian) * error).sum())
                    if not math.isfinite(before):
                        raise ValueError('Non-finite source or quantized weights: ' + name)
                    refine_codes(codec, state, target, hessian, args.iterations)
                    error = state.vals - target
                    after = float(((error @ hessian) * error).sum())
                    if not math.isfinite(after) or after > before * (1 + 1e-5) + 1e-10:
                        raise ValueError('Refinement increased the Hessian objective: ' + name)
                    encoded = codec.encode(state)
                    decoded = codec.decode(encoded)
                    if not torch.equal(decoded.vals, state.vals):
                        raise ValueError('Native code packing changed refined values: ' + name)
                    payload = encoded.tobytes()
                    if len(payload) != raw[expert].nbytes:
                        raise ValueError('Refinement changed native storage size')
                    if os.pwrite(fd, payload, tensor.data_offset + expert * len(payload)) != len(payload):
                        raise OSError('Incomplete tensor write')
                    detail.append(dict(expert=expert, seen=seen, rows=len(x), calibration_mode=mode,
                        loss_before=before, loss_after=after, changed_codes=int((initial != state.q).sum())))
                    del x, target, hessian, state, decoded, initial, error
                with lock:
                    os.fsync(fd)
                    recipe['tensors'][name] = dict(experts=detail,
                        sha256=_digest_slice(fd, tensor.data_offset, tensor.n_bytes))
                    _save(recipe_file, recipe)
                print(f'Refined {name}: {experts} matrices', flush=True)
                del values, data
    try:
        with ThreadPoolExecutor(max_workers=len(quant_devices)) as pool:
            list(pool.map(worker, range(len(quant_devices))))
    finally:
        os.close(fd)
        coverage.close()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_num_threads(previous_threads)
    _verify(args.model, partial, set(names))
    recipe['declared_absent_mtp'] = _publish(partial, args.output)
    recipe['output'] = identity(args.output)
    recipe['completed'] = True
    _save(recipe_file, recipe)
    _save(args.output.with_suffix('.recipe.json'), recipe)
    print(f'Saved MACaQ-R model: {args.output}', flush=True)

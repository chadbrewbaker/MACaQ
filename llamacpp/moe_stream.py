"""Expert-calibrated VQ export directly from HF tensors into one atomic GGUF."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (HERE / 'llama.cpp/gguf-py', HERE / 'llama.cpp', HERE / 'refine', ROOT / 'quantize'):
    sys.path.insert(0, str(p))

import numpy as np
import gguf

EXPERT = re.compile(r'blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight')
DENSE = re.compile(r'blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|attn_qkv|attn_gate|'
                   r'ssm_out|ffn_gate|ffn_up|ffn_down|'
                   r'ffn_gate_shexp|ffn_up_shexp|ffn_down_shexp)\.weight')


def linear_shape(shape):
    return shape if len(shape) == 3 else (1, *shape)


class NativeEncoder:
    def __init__(self, library):
        self.library = C.CDLL(str(library))
        self.fn = self.library.ggml_quantize_chunk
        self.fn.argtypes = [C.c_int, C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_int64, C.c_void_p]
        self.fn.restype = C.c_size_t

    def encode(self, values, qtype, importance=None):
        values = np.ascontiguousarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite source weights')
        if qtype in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
            return values.astype(np.float16 if qtype == gguf.GGMLQuantizationType.F16 else np.float32)
        block, size = gguf.GGML_QUANT_SIZES[qtype]
        rows, width = values.shape
        if width % block:
            raise ValueError(f'{qtype.name} cannot encode width {width}')
        result = np.empty((rows, width // block * size), dtype=np.uint8)
        importance = None if importance is None else np.ascontiguousarray(importance, dtype=np.float32)
        actual = self.fn(int(qtype), values.ctypes.data, result.ctypes.data, 0, rows, width,
                         importance.ctypes.data if importance is not None else None)
        if actual != result.nbytes:
            raise ValueError(f'Native encoder returned {actual} bytes, expected {result.nbytes}')
        return result


def converted_tensors(hf):
    """Use the included architecture adapters while retaining lazy tensor payloads."""
    from conversion import get_model_class
    config = json.loads((hf / 'config.json').read_text())
    if config.get('quantization_config'):
        raise ValueError('The original, unquantized HF checkpoint is required')
    architecture = config['architectures'][0]
    if architecture not in ('Qwen3MoeForCausalLM', 'Qwen3NextForCausalLM', 'OlmoeForCausalLM',
                            'Qwen3_5MoeForConditionalGeneration', 'Qwen3_5MoeForCausalLM',
                            'Qwen4ExpForConditionalGeneration', 'Qwen4ExpForCausalLM'):
        raise ValueError(f'No validated MoE export adapter for {architecture}')
    model_class = get_model_class(architecture)
    if getattr(model_class, 'supports_mtp_export', False):
        model_class.no_mtp = True
    model = model_class(hf, gguf.LlamaFileType.MOSTLY_F16, Path('unused.gguf'))
    entries = {}
    writer = model.gguf_writer

    def capture(name, data, raw_dtype=None, **kwargs):
        if name in entries:
            raise ValueError(f'Duplicate converted tensor {name}')
        entries[name] = data
        writer.add_tensor_info(name, data.shape, data.dtype, data.nbytes, raw_dtype=raw_dtype)

    writer.add_tensor = capture
    model.prepare_tensors()
    model.prepare_metadata(vocab_only=False)
    return model, entries


class DirectGGUF:
    """Reserve final layout once; write disjoint tensor slices without a weight spool."""
    def __init__(self, partial, source_writer, entries, qtypes, transformed, resume=False):
        self.partial = partial
        with tempfile.NamedTemporaryFile(prefix=partial.name + '.layout-', dir=partial.parent, delete=False) as temporary:
            header = Path(temporary.name)
        self.writer = gguf.GGUFWriter(header, source_writer.arch)
        for key, value in source_writer.kv_data[0].items():
            if key in ('general.architecture', 'general.file_type'):
                continue
            self.writer.add_key_value(key, value.value, value.type, sub_type=value.sub_type)
        self.writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_IQ4_XS if any(
            q == gguf.GGMLQuantizationType.IQ4_XS for q in qtypes.values()) else gguf.LlamaFileType.MOSTLY_IQ4_NL))
        self.writer.add_macaq_hadamard_block(64)
        self.writer.add_bool('macaq.moe_vq', True)
        self.expected = {}
        for name, data in entries.items():
            q = qtypes[name]
            shape = tuple(data.shape)
            rawshape = shape if q in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32) else gguf.quant_shape_to_byte_shape(shape, q)
            dtype = np.dtype(np.float16 if q == gguf.GGMLQuantizationType.F16 else np.float32 if q == gguf.GGMLQuantizationType.F32 else np.uint8)
            nbytes = math.prod(rawshape) * dtype.itemsize
            self.writer.add_tensor_info(name, rawshape, dtype, nbytes, raw_dtype=q)
            self.expected[name] = nbytes
            if name in transformed:
                experts, _, width = linear_shape(shape)
                ps = name.replace('.weight', '.pre_scale')
                self.writer.add_tensor_info(ps, (experts, width) if len(shape) == 3 else (width,), np.dtype(np.float32), experts * width * 4)
                self.expected[ps] = experts * width * 4
        self.writer.write_header_to_file()
        self.writer.write_kv_data_to_file()
        self.writer.write_ti_data_to_file()
        file = self.writer.fout[0]
        base = self.writer.ggml_pad(file.tell(), self.writer.data_alignment)
        self.offsets = {}
        position = base
        for name, info in self.writer.tensors[0].items():
            self.offsets[name] = position
            position += self.writer.ggml_pad(info.nbytes, self.writer.data_alignment)
        self.writer.write_padding(file, file.tell())
        self.writer.close()
        if partial.exists():
            if not resume or partial.stat().st_size != position:
                header.unlink()
                raise ValueError('Existing partial output has an incompatible layout or size')
            with header.open('rb') as expected, partial.open('rb') as existing:
                if expected.read() != existing.read(base):
                    header.unlink()
                    raise ValueError('Partial output metadata differs from the current export')
            header.unlink()
        else:
            if shutil.disk_usage(partial.parent).free < position + 1024**3:
                header.unlink()
                raise ValueError(f'Insufficient free disk space for {position / 1024**3:.1f} GiB output plus working margin')
            header.replace(partial)
        self.fd = os.open(partial, os.O_RDWR)
        try:
            os.posix_fallocate(self.fd, 0, position)
        except BaseException:
            os.close(self.fd)
            raise
        self.size = position
        self.written = {name: 0 for name in self.expected}

    def write(self, name, byte_offset, data):
        payload = memoryview(np.ascontiguousarray(data)).cast('B')
        if byte_offset < 0 or byte_offset + len(payload) > self.expected[name]:
            raise ValueError(f'Out-of-range write: {name}')
        position = self.offsets[name] + byte_offset
        done = 0
        while done < len(payload):
            count = os.pwrite(self.fd, payload[done:], position + done)
            if count <= 0:
                raise OSError(f'Incomplete write: {name}')
            done += count
        # Workers write disjoint expert slices; the caller accounts for completed bytes.
        return len(payload)

    def digest(self, name):
        digest = hashlib.sha256()
        position, remaining = self.offsets[name], self.expected[name]
        while remaining:
            block = os.pread(self.fd, min(remaining, 8 * 1024**2), position)
            if not block:
                raise ValueError(f'Cannot read completed tensor {name}')
            digest.update(block)
            position += len(block)
            remaining -= len(block)
        return digest.hexdigest()

    def finish(self):
        missing = {name: self.expected[name] - count for name, count in self.written.items()
                   if count != self.expected[name]}
        if missing:
            raise ValueError(f'Incomplete tensors: {missing}')
        os.fsync(self.fd)
        os.close(self.fd)
        self.fd = None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def sample_name(name, tensors):
    if name + '.samples' in tensors:
        return name
    alias = re.sub(r'ffn_(gate|up)_exps', 'ffn_gate_up_exps', name)
    if alias + '.samples' in tensors:
        return alias
    raise ValueError(f'Missing routed activation samples for {name}')


def check_samples(names, entries, reader, minimum, allow_uncovered=False):
    tensors = {t.name: t.data for t in reader.tensors}
    if 'macaq.sample_cap' not in reader.fields:
        raise ValueError('Calibration must include reservoir samples from the updated llama-imatrix')
    aliases = {}
    for name in names:
        key = sample_name(name, tensors)
        aliases[name] = key
        experts, _, width = linear_shape(entries[name].shape)
        cap = int(reader.fields['macaq.sample_cap'].contents())
        try:
            samples = tensors[key + '.samples'].reshape(experts, cap, width)
            diagonal = tensors[key + '.in_sum2'].reshape(experts, width)
        except ValueError as error:
            raise ValueError(f'Expert layout does not match the calibration model: {name}') from error
        tensors[key + '.samples'] = samples
        tensors[key + '.in_sum2'] = diagonal
        seen = tensors[key + '.sample_seen'].reshape(-1)
        counts = tensors[key + '.counts'].reshape(-1)
        if samples.shape[0] != experts or samples.shape[2] != width or diagonal.shape != (experts, width):
            raise ValueError(f'Expert layout does not match the calibration model: {name}')
        if len(seen) != experts or not np.array_equal(seen.astype(np.float32), counts):
            raise ValueError(f'Inconsistent expert coverage: {name}')
        missing = np.flatnonzero(seen < minimum)
        if len(missing) and (not allow_uncovered or not EXPERT.fullmatch(name)):
            raise ValueError(f'{name}: {len(missing)}/{experts} experts have fewer than {minimum} routed tokens; '
                             f'minimum={int(seen.min())}. Increase calibration coverage.')
    return tensors, aliases


def export(hf, samples, output, library, *, format='IQ4_XS', sweeps=2, alpha=0.75,
           minimum=32, devices=('cuda:0',), row_chunk=1024, progress=None, prepared=None,
           checkpoint=None, resume=False, encode_threads=8, coverage_policy='error', embedding_format='auto'):
    import torch
    from macaq_balance import balance_pre_scale_from_diag
    from macaq_math import macaq_transform_weight
    from macaq_refine_gguf import to_macaq_frame, hadamard, h_loss
    from gguf_codecs import make_codec
    from product_refine import product_gs_refine
    from gguf.lazy import LazyBase
    hf, output = Path(hf), Path(output)
    if output.exists() or (output.with_suffix('.part.gguf').exists() and not resume):
        raise ValueError('Choose a fresh output path')
    if sweeps < 1 or minimum < 1 or row_chunk < 1 or not 0 <= alpha <= 1 or not devices or encode_threads < 1:
        raise ValueError('Invalid VQ/calibration/export settings')
    if embedding_format not in ('auto', 'Q4_K', 'Q6_K', 'Q8_0', 'F16'):
        raise ValueError('unsupported embedding format')
    model, entries = prepared or converted_tensors(hf)
    transformed = [name for name in entries if EXPERT.fullmatch(name) or DENSE.fullmatch(name)]
    if not any(EXPERT.fullmatch(name) for name in transformed):
        raise ValueError('No routed expert projections were found')
    samples_reader = gguf.GGUFReader(samples)
    if coverage_policy not in ('error', 'router-weighted'):
        raise ValueError('coverage policy must be error or router-weighted')
    banks, aliases = check_samples(transformed, entries, samples_reader, minimum,
                                    allow_uncovered=coverage_policy == 'router-weighted')
    soft_coverage = None
    if coverage_policy == 'router-weighted':
        from moe_coverage import SoftCoverage
        soft_coverage = SoftCoverage(hf, banks, aliases)
    qtypes = {}
    for name, data in entries.items():
        width = data.shape[-1]
        if name in transformed:
            qtypes[name] = gguf.GGMLQuantizationType[format] if width % 256 == 0 else gguf.GGMLQuantizationType.IQ4_NL
            if width % 64:
                raise ValueError(f'Hadamard-64 cannot transform {name} with width {width}')
        elif name == 'output.weight':
            qtypes[name] = gguf.GGMLQuantizationType.Q6_K if width % 256 == 0 else gguf.GGMLQuantizationType.Q8_0
        elif name in ('token_embd.weight', gguf.TENSOR_NAMES[gguf.MODEL_TENSOR.PER_LAYER_TOKEN_EMBD] + '.weight'):
            qtypes[name] = (gguf.GGMLQuantizationType[embedding_format] if embedding_format != 'auto' else
                           gguf.GGMLQuantizationType.Q4_K if width % 256 == 0 else gguf.GGMLQuantizationType.IQ4_NL)
        else:
            qtypes[name] = model.gguf_writer.tensors[0][name].dtype
    if ('output.weight' not in entries and 'token_embd.weight' in entries
            and embedding_format == 'auto'):
        qtypes['token_embd.weight'] = (gguf.GGMLQuantizationType.Q6_K
            if entries['token_embd.weight'].shape[-1] % 256 == 0 else gguf.GGMLQuantizationType.Q8_0)
    for name, kind in qtypes.items():
        block, _ = gguf.GGML_QUANT_SIZES[kind]
        if entries[name].shape[-1] % block:
            raise ValueError(f'{kind.name} cannot encode {name} with width {entries[name].shape[-1]}')
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.part.gguf')
    checkpoint = Path(checkpoint) if checkpoint else None
    source_key = dict(output=str(output.resolve()), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), config=hashlib.sha256((hf / 'config.json').read_bytes()).hexdigest(),
        files=[(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(hf.glob('*.safetensors'))],
        samples=(str(Path(samples).resolve()), Path(samples).stat().st_size, Path(samples).stat().st_mtime_ns),
        format=format, sweeps=sweeps, alpha=alpha, minimum=minimum, devices=list(devices), coverage_policy=coverage_policy,
        embedding_format=embedding_format,
        encoder_sha256=hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (Path(__file__), HERE / 'refine/product_refine.py', HERE / 'refine/gguf_codecs.py',
               HERE / 'refine/product_kernels.py', HERE / 'moe_coverage.py',
               HERE / 'llama.cpp/macaq_math.py', ROOT / 'quantize/macaq_balance.py')})
    identity = hashlib.sha256(json.dumps(source_key, sort_keys=True).encode()).hexdigest()
    saved = None
    if resume and checkpoint and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        if saved.get('identity') != identity:
            raise ValueError('Resume inputs, settings, devices, encoder or refinement code changed')
        if not partial.is_file():
            raise ValueError('The partial GGUF belonging to this checkpoint is missing')
    elif resume and partial.exists():
        raise ValueError('A partial file without a matching export checkpoint cannot be resumed')
    writer = DirectGGUF(partial, model.gguf_writer, entries, qtypes, transformed, resume=bool(saved))
    encoder = NativeEncoder(library)
    stats = dict(completed=False, format=format, vq=True, hadamard=64, refined_tensors={},
                 source=str(hf.resolve()), calibration=str(Path(samples).resolve()),
                 quantization_types={name:q.name for name,q in qtypes.items()},
                 output_bytes=writer.size, intermediate_full_precision_gguf=False)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    device_list = [torch.device(device) for device in devices]
    completed = set(saved['completed']) if saved else set()
    checksums = dict(saved['checksums']) if saved else {}
    if saved:
        stats['refined_tensors'] = saved['refined_tensors']
        for name in completed:
            if name not in entries:
                raise ValueError(f'Unknown completed tensor in checkpoint: {name}')
            if writer.digest(name) != checksums.get(name):
                writer.close()
                raise ValueError(f'Completed tensor checksum differs: {name}')
            writer.written[name] = writer.expected[name]
            if name in transformed:
                ps = name.replace('.weight', '.pre_scale')
                if writer.digest(ps) != checksums.get(ps):
                    writer.close()
                    raise ValueError(f'Completed scale checksum differs: {ps}')
                writer.written[ps] = writer.expected[ps]

    def save_checkpoint():
        if checkpoint:
            os.fsync(writer.fd)
            for name in completed:
                for key in ([name, name.replace('.weight', '.pre_scale')] if name in transformed else [name]):
                    if key not in checksums:
                        checksums[key] = writer.digest(key)
            payload = dict(identity=identity, completed=sorted(completed), checksums=checksums, refined_tensors=stats['refined_tensors'])
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps(payload, indent=2) + '\n')
            temporary.replace(checkpoint)

    def prepare_expert(name, values, expert):
        started = time.perf_counter()
        device = device_list[expert % len(device_list)]
        key = aliases[name]
        seen = int(banks[key + '.sample_seen'][expert])
        rows = min(seen, banks[key + '.samples'].shape[1])
        x = torch.from_numpy(np.array(banks[key + '.samples'][expert, :rows], copy=True)).to(device)
        weight = torch.from_numpy(np.array(values[expert] if values.ndim == 3 else values, dtype=np.float32, copy=True)).to(device)
        diag = torch.from_numpy(np.array(banks[key + '.in_sum2'][expert], copy=True)).to(device)
        calibration_mode = 'native-routed'
        normalization_count = seen
        if seen < minimum:
            x, diag = soft_coverage.inputs(name, expert, device)
            rows = len(x)
            normalization_count = rows
            calibration_mode = 'router-weighted-fallback'
        if not torch.isfinite(x).all() or not torch.isfinite(diag).all() or (diag < 0).any():
            raise ValueError(f'Invalid activation statistics: {name} expert {expert}')
        scale = balance_pre_scale_from_diag(weight, diag, normalization_count, alpha).bfloat16().float()
        target = macaq_transform_weight(weight, scale, 64).half().float()
        original_h = x.T @ x / rows
        if original_h.diagonal().sum() <= 0:
            raise ValueError(f'Zero-energy calibration: {name} expert {expert}')
        h = to_macaq_frame(original_h, scale, hadamard(64, device), 1.0)
        raw = encoder.encode(target.cpu().numpy(), qtypes[name], h.diagonal().cpu().numpy())
        codec = make_codec(qtypes[name], device)
        state = codec.decode(raw)
        before = h_loss(state.vals, target, h)
        return dict(name=name, expert=expert, codec=codec, state=state, target=target,
                    h=h, scale=scale, before=before, seen=seen, rows=rows, started=started,
                    calibration_mode=calibration_mode)

    def finish_expert(item, result):
        name, expert = item['name'], item['expert']
        codec, state, target, h, scale = (item[key] for key in ('codec', 'state', 'target', 'h', 'scale'))
        after = h_loss(state.vals, target, h)
        before = item['before']
        if not math.isfinite(after) or after > before * (1 + 1e-5) + 1e-10:
            raise ValueError(f'VQ increased the Hessian objective: {name} expert {expert}')
        packed = codec.encode(state)
        byte_count = writer.write(name, expert * packed.nbytes, packed)
        ps_name = name.replace('.weight', '.pre_scale')
        ps = scale.cpu().numpy()
        ps_count = writer.write(ps_name, expert * ps.nbytes, ps)
        return expert, byte_count, ps_count, dict(seen=item['seen'], retained=item['rows'], loss_before=before,
            loss_after=after, moves=result['moves'], sweeps=result['sweeps'], method='product-vq',
            seconds=time.perf_counter() - item['started'], calibration_mode=item['calibration_mode'])

    def one_expert(name, values, expert):
        item = prepare_expert(name, values, expert)
        result = product_gs_refine(item['codec'], item['state'], item['target'], item['h'],
                                   n_sweeps=sweeps, seed=expert)
        return finish_expert(item, result)

    def expert_batch(name, values, indices):
        from refine.product_kernels import product_batch_refine
        items = list(cpu_pool.map(lambda expert: prepare_expert(name, values, expert), indices))
        result = product_batch_refine(items[0]['codec'], [item['state'] for item in items],
            [item['target'] for item in items], [item['h'] for item in items], sweeps=sweeps)
        return [finish_expert(item, stats) for item, stats in zip(items, result)]

    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with ThreadPoolExecutor(max_workers=len(device_list)) as pool, ThreadPoolExecutor(max_workers=encode_threads) as cpu_pool:
            for name in list(entries):
                data = entries.pop(name)
                if name in completed:
                    if progress:
                        progress(f'Resumed completed tensor {name}')
                    continue
                if name in transformed:
                    values = LazyBase.to_eager(data)
                    expert_stats = {}
                    n_experts = linear_shape(values.shape)[0]
                    if all(device.type == 'cuda' for device in device_list):
                        jobs = []
                        assignments = [list(range(index, n_experts, len(device_list)))
                                       for index in range(len(device_list))]
                        for start in range(0, max(map(len, assignments)), 16):
                            jobs.extend(assigned[start:start + 16] for assigned in assignments
                                        if start < len(assigned))
                        results = pool.map(lambda indices: expert_batch(name, values, indices), jobs)
                    else:
                        results = ([result] for result in pool.map(lambda e: one_expert(name, values, e), range(n_experts)))
                    for expert, count, ps_count, detail in (item for batch in results for item in batch):
                        writer.written[name] += count
                        writer.written[name.replace('.weight', '.pre_scale')] += ps_count
                        expert_stats[str(expert)] = detail
                        if progress and (expert % 32 == 0 or expert + 1 == n_experts):
                            progress(f'{name}: {expert + 1}/{n_experts} matrices refined')
                    stats['refined_tensors'][name] = dict(kind='routed' if EXPERT.fullmatch(name) else 'dense', experts=expert_stats)
                    del values
                else:
                    chunks = data.iter_chunks() if isinstance(data, gguf.LazyChunkedTensor) else (LazyBase.to_eager(data),)
                    position = 0
                    for chunk in chunks:
                        if chunk.ndim == 1:
                            position += writer.write(name, position, encoder.encode(chunk, qtypes[name]))
                        else:
                            matrix = chunk.reshape(-1, chunk.shape[-1])
                            batches = (matrix[start:start + row_chunk] for start in range(0, len(matrix), row_chunk))
                            for raw in cpu_pool.map(lambda rows: encoder.encode(rows, qtypes[name]), batches):
                                position += writer.write(name, position, raw)
                    writer.written[name] += position
                completed.add(name)
                save_checkpoint()
                if progress:
                    progress(f'Written {name}')
                del data
        writer.finish()
        # Parse the completed file before publishing it under the requested name.
        check = gguf.GGUFReader(partial)
        if len(check.tensors) != len(writer.expected):
            raise ValueError('Output tensor directory is incomplete')
        del check
        partial.rename(output)
        stats['completed'] = True
        return stats
    finally:
        if soft_coverage is not None:
            soft_coverage.close()
        writer.close()
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32

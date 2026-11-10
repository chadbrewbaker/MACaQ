"""Prepare MACAQ weight-only GGUF directly from full-precision GGUF + imatrix.

Uses the existing MACAQ D1 formula and weight transform. Calibration statistics
come from llama-imatrix on original inputs. The output is unquantized: run
llama-quantize afterwards. This does not produce an MXFP4/vLLM checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llamacpp/llama.cpp/gguf-py"))
sys.path.insert(0, str(ROOT / "llamacpp/llama.cpp"))
sys.path.insert(0, str(ROOT / "quantize"))


def source_paths(path):
    match = re.fullmatch(r"(.*)-00001-of-(\d{5})\.gguf", path.name)
    if not match:
        return [path]
    return [path.with_name(f"{match[1]}-{i:05d}-of-{match[2]}.gguf")
            for i in range(1, int(match[2]) + 1)]


def prepare(model, imatrix, output, device="cpu", alpha=0.75, had_block=32):
    import numpy as np
    import torch
    import gguf
    from macaq_math import macaq_transform_weight
    from macaq_balance import balance_pre_scale_from_diag
    if had_block < 1 or had_block & (had_block - 1) or not 0 <= alpha <= 1:
        raise ValueError("Hadamard block must be a power of two; alpha must lie in [0,1]")
    paths = source_paths(model)
    if output.resolve() in {path.resolve() for path in paths}:
        raise ValueError("output must differ from source")
    readers = [gguf.GGUFReader(path) for path in paths]
    first = readers[0]
    arch = first.fields["general.architecture"].contents()
    if arch not in ("qwen3", "qwen35"):
        raise ValueError(f"unsupported weight-only architecture: {arch}")
    if "macaq.hadamard_block" in first.fields:
        raise ValueError("source must be the original full-precision GGUF")
    tensors = [tensor for reader in readers for tensor in reader.tensors]
    if len({tensor.name for tensor in tensors}) != len(tensors):
        raise ValueError("duplicate tensors in split source")
    stats = {t.name: np.asarray(t.data).reshape(-1) for t in gguf.GGUFReader(imatrix).tensors}
    layer_count = int(first.fields[f"{arch}.block_count"].contents())
    mtp = first.fields.get(f"{arch}.nextn_predict_layers")
    main_layers = layer_count - (int(mtp.contents()) if mtp else 0)
    pattern = re.compile(r"blk\.(\d+)\.(attn_[qkv]|attn_output|attn_qkv|attn_gate|"
                         r"ssm_out|ssm_beta|ssm_alpha|ffn_gate|ffn_up|ffn_down)\.weight")
    selected = set()
    for tensor in tensors:
        match = pattern.fullmatch(tensor.name)
        if not match or int(match[1]) >= main_layers:
            continue
        if tensor.tensor_type not in (gguf.GGMLQuantizationType.F16,
                                     gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32):
            raise ValueError(f"original full-precision weights required: {tensor.name}")
        if len(tensor.shape) != 2 or int(tensor.shape[0]) % had_block:
            raise ValueError(f"unsupported dimensions: {tensor.name}")
        sum_x2 = stats.get(tensor.name + ".in_sum2")
        count = stats.get(tensor.name + ".counts")
        if sum_x2 is None or count is None or count.size != 1 or count[0] <= 0:
            raise ValueError(f"missing complete activation statistics: {tensor.name}")
        if sum_x2.size != int(tensor.shape[0]) or not np.isfinite(sum_x2).all() or (sum_x2 < 0).any():
            raise ValueError(f"invalid activation statistics: {tensor.name}")
        selected.add(tensor.name)
    if not selected:
        raise ValueError("no supported calibrated linears")
    output.parent.mkdir(parents=True, exist_ok=True)
    recipe = dict(completed=False, source_files=[str(path.resolve()) for path in paths],
                  imatrix_sha256=hashlib.sha256(imatrix.read_bytes()).hexdigest(),
                  architecture=arch, main_layers=main_layers, transformed_linears=len(selected),
                  hadamard_block=had_block, alpha=alpha, device=device,
                  method="MACAQ D1 + block-Hadamard; later llama.cpp weight quantization",
                  prescale_precision="BF16-rounded, stored as F32", layers=[])
    recipe_path = output.with_suffix(".recipe.json")
    recipe_path.write_text(json.dumps(recipe, indent=2))
    temporary = output.with_suffix(".part.gguf")
    writer = gguf.GGUFWriter(temporary, arch, endianess=first.endianess)
    for key, field in first.fields.items():
        if key.startswith(("GGUF.", "split.")) or key == "general.architecture":
            continue
        sub_type = field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(key, field.contents(), field.types[0], sub_type=sub_type)
    writer.add_macaq_hadamard_block(had_block)
    for tensor in tensors:
        if tensor.name in selected:
            shape = tuple(reversed(tensor.shape.tolist()))
            writer.add_tensor_info(tensor.name, shape, np.dtype("float16"), tensor.n_elements * 2,
                                   gguf.GGMLQuantizationType.F16)
            width = int(tensor.shape[0])
            writer.add_tensor_info(tensor.name.replace(".weight", ".pre_scale"), (width,),
                                   np.dtype("float32"), width * 4, gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                                   tensor.n_bytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    try:
        for tensor in tensors:
            if tensor.name not in selected:
                writer.write_tensor_data(tensor.data, tensor_endianess=first.endianess)
                continue
            raw = gguf.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
            weight = torch.from_numpy(raw).to(device)
            diagonal = torch.from_numpy(stats[tensor.name + ".in_sum2"].copy()).to(device)
            count = float(stats[tensor.name + ".counts"][0])
            pre = balance_pre_scale_from_diag(weight, diagonal, count, alpha).bfloat16().float()
            transformed = macaq_transform_weight(weight, pre, had_block).half()
            if not torch.isfinite(transformed).all():
                raise ValueError(f"nonfinite transformed weights: {tensor.name}")
            writer.write_tensor_data(transformed.cpu().numpy())
            writer.write_tensor_data(pre.cpu().numpy())
            recipe["layers"].append(dict(name=tensor.name, calibration_rows=count,
                                        input_width=int(tensor.shape[0])))
            recipe_path.write_text(json.dumps(recipe, indent=2))
            print(f"TRANSFORMED {tensor.name} ({len(recipe['layers'])}/{len(selected)})", flush=True)
            del raw, weight, diagonal, pre, transformed
    finally:
        writer.close()
    temporary.replace(output)
    recipe["completed"] = True
    recipe_path.write_text(json.dumps(recipe, indent=2))
    return recipe


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="original GGUF, or first split")
    ap.add_argument("--imatrix", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--had-block", type=int, default=32)
    args = ap.parse_args()
    prepare(args.model, args.imatrix, args.output, args.device, args.alpha, args.had_block)


if __name__ == "__main__":
    main()

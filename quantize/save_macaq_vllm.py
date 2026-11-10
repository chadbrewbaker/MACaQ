"""save_macaq_vllm.py , serialize a macaq-quantized (OCPDenseW4A4Linear) model into
a checkpoint our vLLM dense W4A4 or weight-only LinearMethod can load.

Per quantized Linear at HF name P (e.g. model.layers.0.self_attn.q_proj):
    P.weight_packed  (out, in//2) uint8     , MXFP4 nibbles (OCP convention)
    P.weight_scale   (out, in//32) uint8    , e8m0 (floor(log2 amax)−2+127)
    P.pre_scale      (in,) bf16              , D1, identity when absent
    P.bias           (out,) if present
The block-Hadamard B is NOT stored , vLLM rebuilds it from hadamard_block in
config.json (canonical Sylvester; the all-ones-row alignment was removed).
Plus all non-quantized tensors (embeddings, norms, lm_head) copied verbatim, and a
config.json carrying quantization_config so vLLM routes Linears to our method.

Keys use HF module names; vLLM's packed_modules_mapping fuses q/k/v→qkv_proj and
gate/up→gate_up_proj at load (output-dim shards), and our weight_loader stacks the
per-shard pre_scale.
"""
import json
import os
import torch
from safetensors.torch import save_file
from checkpoint_config import portable_config


def _is_real_mxfp4(m):
    return hasattr(m, "W_packed") and hasattr(m, "W_scales") and hasattr(m, "B_inv_T")


@torch.no_grad()
def save_macaq_checkpoint(model, tokenizer, save_dir, had_block,
                          source_model_name="", activation_precision="w4a4"):
    if activation_precision not in ("w4a4", "w4a16"):
        raise ValueError("activation precision must be w4a4 or w4a16")
    if os.path.isdir(save_dir) and os.listdir(save_dir):
        raise ValueError("Checkpoint directory is not empty; choose a fresh save_dir")
    os.makedirs(save_dir, exist_ok=True)
    tensors = {}
    shared_bh = None
    n_q = 0

    quant_names, unq_params = set(), set()
    for name, m in model.named_modules():
        if _is_real_mxfp4(m):
            quant_names.add(name)
            out_f, in_f = m.out_features, m.in_features
            tensors[f"{name}.weight_packed"] = m.W_packed.cpu().contiguous()
            # the packer pads W_scales to (ceil(out/128)*128, ceil(nblk/4)*4);
            # the kernel wants the unpadded (out, in//32) e8m0 bytes.
            tensors[f"{name}.weight_scale"] = (
                m.W_scales.view(torch.uint8)[:out_f, : in_f // 32].cpu().contiguous())
            if getattr(m, "pre_scale", None) is not None:
                tensors[f"{name}.pre_scale"] = m.pre_scale.bfloat16().cpu().contiguous()
            else:
                tensors[f"{name}.pre_scale"] = torch.ones(in_f, dtype=torch.bfloat16)
            if getattr(m, "bias", None) is not None:
                tensors[f"{name}.bias"] = m.bias.bfloat16().cpu().contiguous()
            if shared_bh is None:
                shared_bh = m.B_inv_T.bfloat16().cpu().contiguous()
            n_q += 1

    # Everything not inside a quantized Linear: copy verbatim (embed, norms, lm_head).
    for pname, p in model.state_dict().items():
        if any(pname.startswith(q + ".") for q in quant_names):
            continue
        tensors[pname] = p.bfloat16().cpu().contiguous() if p.is_floating_point() \
            else p.cpu().contiguous()

    assert shared_bh is not None, "no quantized (OCPDenseW4A4Linear) layer found"
    # NOTE: the block-Hadamard is NOT saved as a tensor , vLLM reconstructs it
    # deterministically from `hadamard_block` in config.json (canonical Sylvester),
    # so it never appears as an unmatched checkpoint key. (shared_bh validated.)

    save_file(tensors, os.path.join(save_dir, "model.safetensors"),
              metadata={"format": "pt"})

    cfg = portable_config(model.config.to_dict())
    cfg["quantization_config"] = {
        "quant_method": "mxfp4",                 # vLLM Mxfp4Config; plugin routes dense
        "macaq_dense_w4a4": True,                # our marker
        "macaq_activation_precision": activation_precision,
        "hadamard_block": int(had_block),        # BH = canonical Sylvester
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_dir)

    print(f"[save] {n_q} quantized Linears + shared BH{tuple(shared_bh.shape)} "
          f"-> {save_dir}  ({len(tensors)} tensors)")
    return save_dir

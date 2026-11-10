"""Calibrate Qwen3 MoE experts and export a self-contained vLLM checkpoint."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import torch
from torch import nn
from checkpoint_config import portable_config

ROOT = Path(__file__).resolve().parents[1]
from source_provenance import save_source_fingerprint
from mxfp4_helper import (QuantizedFusedExperts, ShardCache, load_weight_index,
                          load_block_from_safetensors, capture_layer_input)
from quantize_moe_qwen35 import quantize_block_moe, collect_bf16_block_tensors, ShardedSafetensorsWriter


class CalibrationMoE(nn.Module):
    """Preserve Qwen3 routing while exposing expert activations to calibration."""
    def __init__(self, source):
        super().__init__()
        self.return_router = hasattr(source, "top_k")
        router = source if self.return_router else source.gate
        self.top_k = router.top_k
        self.norm_topk_prob = router.norm_topk_prob
        self.gate = nn.Linear(source.gate.weight.shape[1], source.gate.weight.shape[0],
                              bias=False, device=source.gate.weight.device, dtype=source.gate.weight.dtype)
        self.gate.weight = source.gate.weight
        experts = QuantizedFusedExperts(len(source.experts), nn.functional.silu)
        for expert in source.experts:
            gate, up, down = expert.gate_proj, expert.up_proj, expert.down_proj
            combined = nn.Linear(gate.in_features, 2 * gate.out_features, bias=False,
                                 device=gate.weight.device, dtype=gate.weight.dtype)
            combined.weight = nn.Parameter(torch.cat((gate.weight, up.weight)), requires_grad=False)
            experts.gate_up_proj.append(combined)
            experts.down_proj.append(down)
        self.experts = experts

    def forward(self, hidden):
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])
        logits = self.gate(x)
        weights, ids = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float32), self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        out = self.experts(x, ids, weights.to(x.dtype))
        return (out.reshape(shape), logits) if self.return_router else out.reshape(shape)


class PackedExperts(nn.Module):
    """Replay the calibrated layer with the serving kernel instead of expert loops."""
    def __init__(self, tensors, device):
        super().__init__()
        for name, value in tensors.items():
            self.register_buffer(name, value.to(device))
        from macaq_vllm_mxfp4_dense import make_hadamard
        self.register_buffer("bh", make_hadamard(32, device).bfloat16())

    def forward(self, hidden, ids, weights):
        sys.path.insert(0, str(ROOT / "vllm"))
        from triton_mxfp4_moe import fused_mxfp4_moe
        return fused_mxfp4_moe(hidden, self.w13_weight, self.w13_weight_scale,
            self.w2_weight, self.w2_weight_scale, ids.int(), weights.float(),
            self.w13_pre_scale, self.w2_pre_scale, self.bh)


def checkpoint_config(config):
    result = portable_config(config.to_dict())
    # Both names are used by supported Transformers config versions.
    result["num_experts"] = config.num_experts
    result["num_local_experts"] = config.num_experts
    result["quantization_config"] = dict(quant_method="mxfp4", macaq_moe=True, rotation_block_size=32)
    result["architectures"] = ["Qwen3MoeForCausalLM"]
    return result


@torch.inference_mode()
def quantize(source, calibration, output, tokens=16384, chunk=512, flips=64, device="cuda:0"):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeForCausalLM, Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding)
    if output.exists() or min(tokens, chunk, flips) < 1:
        raise ValueError("choose a fresh output directory and positive settings")
    config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if config.model_type != "qwen3_moe" or getattr(config, "quantization_config", None):
        raise ValueError("an original Qwen3 MoE checkpoint is required")
    config._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    text = calibration.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("calibration must contain nonempty UTF-8 text")
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokens).input_ids.to(device)
    if ids.shape[1] < chunk:
        raise ValueError("calibration text is shorter than one chunk")
    chunks = [ids[:, i:i + chunk] for i in range(0, ids.shape[1], chunk) if ids.shape[1] - i >= 32]
    weights = load_weight_index(source)
    shards = ShardCache(source)
    def load(key):
        return shards.get(weights[key]).get_tensor(key).to(device, torch.bfloat16)
    with init_empty_weights():
        model = Qwen3MoeForCausalLM(config)
    model.eval()
    model.model.embed_tokens.weight = nn.Parameter(load("model.embed_tokens.weight"), requires_grad=False)
    model.model.rotary_emb = Qwen3MoeRotaryEmbedding(config=config).to(device)
    def layer(index):
        with init_empty_weights():
            block = Qwen3MoeDecoderLayer(config, layer_idx=index)
            # Original Qwen3 checkpoints store individual expert projections.
            # Recent Transformers versions construct fused expert parameters;
            # expose the source layout before loading the individual tensors.
            if not isinstance(block.mlp.experts, nn.ModuleList):
                experts = nn.ModuleList()
                for _ in range(config.num_experts):
                    expert = nn.Module()
                    expert.gate_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
                    expert.up_proj = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
                    expert.down_proj = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
                    experts.append(expert)
                block.mlp.experts = experts
        load_block_from_safetensors(block, index, weights, shards, "model.layers.")
        block = block.to(device).eval()
        block.mlp = CalibrationMoE(block.mlp).eval()
        return block
    first = layer(0)
    model.model.layers[0] = first
    cached = [list(capture_layer_input(model.model, row, layer_idx=0)) for row in chunks]
    model.model.layers[0] = None
    del first, ids, chunks
    gc.collect()
    torch.cuda.empty_cache()
    settings = dict(n_flip_iters=flips, ridge=1., flip=True, exp_screen=True, fast_flip=True,
                    exp_offsets=(1, 0, -1), pre_alpha=.75, calib_mode="router")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / "checkpoint"
        writer = ShardedSafetensorsWriter(pending, 2_000_000_000)
        for index in range(config.num_hidden_layers):
            block = layer(index)
            packed = quantize_block_moe(block, index, cached, settings, device)
            if set(packed) != {"mlp.experts"}:
                raise ValueError("unexpected expert inventory")
            for prefix, tensors in packed.items():
                for name, value in tensors.items():
                    writer.add(f"model.layers.{index}.{prefix}.{name}", value)
            for name, value in collect_bf16_block_tensors(block, index).items():
                writer.add(name, value)
            block.mlp.experts = PackedExperts(packed["mlp.experts"], device)
            for index_chunk, (args, kwargs) in enumerate(cached):
                result = block(*args, **kwargs)
                hidden = result[0] if isinstance(result, tuple) else result
                cached[index_chunk][0] = (hidden.detach().clone(),) + tuple(args[1:])
            del block, packed
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Quantized layer {index + 1}/{config.num_hidden_layers}", flush=True)
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            writer.add(name, load(name).cpu())
        shards.close()
        writer.finalize()
        result = checkpoint_config(config)
        (pending / "config.json").write_text(json.dumps(result, indent=2) + "\n")
        for path in source.iterdir():
            if path.is_file() and path.name.startswith(("tokenizer", "vocab", "merges", "special_tokens",
                                                       "added_tokens", "chat_template", "generation_config", "LICENSE")):
                shutil.copyfile(path, pending / path.name)
        save_source_fingerprint(source, pending)
        recipe = dict(model_type=config.model_type, expert_layers=config.num_hidden_layers,
                      experts_per_layer=config.num_experts, calibration_sha256=__import__("hashlib").sha256(text.encode()).hexdigest(),
                      tokens=sum(args[0].shape[1] for args, _ in cached), chunk=chunk, flips=flips,
                      method="router-weighted MACaQ", precision="w4a4")
        (pending / "macaq_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        pending.rename(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--flips", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    if a.gpu < 0:p.error("GPU index must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    torch.set_num_threads(4)
    quantize(a.model, a.calibration, a.output, a.tokens, a.chunk, a.flips)

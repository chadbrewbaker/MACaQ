"""Streaming text adapters for hybrid Qwen MoEs and OLMoE."""

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import tempfile
from contextlib import nullcontext

import torch
from torch import nn
from checkpoint_config import portable_config

from moe_text import PackedExperts
from mxfp4_helper import (QuantizedFusedExperts, ShardCache, load_weight_index,
                          load_block_from_safetensors, capture_layer_input,
                          unfuse_moe_experts, set_nested_attr)
from quantize_moe_qwen35 import quantize_block_moe, ShardedSafetensorsWriter


def architecture(config):
    text = getattr(config, "text_config", config)
    if text.model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForCausalLM as Model, Qwen3_5MoeDecoderLayer as Layer)
    elif text.model_type == "qwen3_next":
        from transformers.models.qwen3_next.modeling_qwen3_next import (
            Qwen3NextForCausalLM as Model, Qwen3NextDecoderLayer as Layer)
    elif text.model_type == "olmoe":
        from transformers.models.olmoe.modeling_olmoe import (
            OlmoeForCausalLM as Model, OlmoeDecoderLayer as Layer)
    elif text.model_type == "qwen4_exp_text":
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpForCausalLM as Model, Qwen4ExpTextDecoderLayer as Layer)
    else:
        raise ValueError(f"no streaming MoE adapter for {text.model_type}")
    return text, Model, Layer


def serving_config(source, text, architecture_name):
    # The calibration library renames indexed-attention layers. Preserve the
    # source schema understood by the serving runtime, including rotary fields.
    original = source.get('text_config', source)
    config = portable_config(original if text.model_type == 'qwen4_exp_text' else text.to_dict())
    config.update(architectures=[architecture_name], model_type=text.model_type,
                  num_experts=text.num_experts, num_local_experts=text.num_experts,
                  quantization_config=dict(quant_method='mxfp4', macaq_moe=True, rotation_block_size=32))
    return config


class CalibrationBlock(nn.Module):
    """Expose raw router logits while retaining shared-expert arithmetic."""
    def __init__(self, source, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.normalize = getattr(config, "norm_topk_prob", True)
        weight = source.gate.weight
        self.gate = nn.Linear.__new__(nn.Linear)
        nn.Module.__init__(self.gate)
        self.gate.in_features, self.gate.out_features = weight.shape[1], weight.shape[0]
        self.gate.weight, self.gate.bias = weight, None
        if isinstance(source.experts, QuantizedFusedExperts):
            self.experts = source.experts
        else:
            experts = QuantizedFusedExperts(len(source.experts), nn.functional.silu)
            for expert in source.experts:
                gate, up = expert.gate_proj, expert.up_proj
                combined = nn.Linear.__new__(nn.Linear)
                nn.Module.__init__(combined)
                combined.in_features, combined.out_features = gate.in_features, 2 * gate.out_features
                combined.weight = nn.Parameter(torch.cat((gate.weight, up.weight)), requires_grad=False)
                combined.bias = None
                experts.gate_up_proj.append(combined)
                experts.down_proj.append(expert.down_proj)
            self.experts = experts
        self.shared_expert = getattr(source, "shared_expert", None)
        self.shared_expert_gate = getattr(source, "shared_expert_gate", None)

    def forward(self, hidden):
        shape = hidden.shape
        x = hidden.reshape(-1, shape[-1])
        logits = self.gate(x)
        weights, ids = torch.softmax(logits, dim=-1, dtype=torch.float32).topk(self.top_k)
        if self.normalize:
            weights = weights / weights.sum(-1, keepdim=True)
        result = self.experts(x, ids, weights.to(x.dtype))
        if self.shared_expert is not None:
            shared = self.shared_expert(x)
            if self.shared_expert_gate is not None:
                shared = torch.sigmoid(self.shared_expert_gate(x)) * shared
            result = result + shared
        return result.reshape(shape)


def individual_expert_layout(block, config):
    width = getattr(config, "moe_intermediate_size", None) or config.intermediate_size
    experts = nn.ModuleList()
    for _ in range(config.num_experts):
        expert = nn.Module()
        expert.gate_proj = nn.Linear(config.hidden_size, width, bias=False)
        expert.up_proj = nn.Linear(config.hidden_size, width, bias=False)
        expert.down_proj = nn.Linear(width, config.hidden_size, bias=False)
        experts.append(expert)
    block.mlp.experts = experts


class HostShardedEmbedding(nn.Module):
    """Exact CPU lookup from mapped source shards without joining the full table."""
    def __init__(self, parts):
        super().__init__()
        if (not parts or any(p.device.type != 'cpu' or p.ndim != 2 or not len(p)
                             or p.shape[1] != parts[0].shape[1] or p.dtype != parts[0].dtype for p in parts)):
            raise ValueError('incompatible host embedding shards')
        self.parts = tuple(parts)
        # Transformers uses weight.device to select the embedding lookup device.
        self.weight = parts[0]
        self.ends = torch.tensor([len(p) for p in parts], dtype=torch.long, device='cpu').cumsum(0)
        self.num_embeddings = int(self.ends[-1])

    def forward(self, ids):
        flat = ids.to('cpu').reshape(-1)
        if flat.numel() and (flat.min() < 0 or flat.max() >= self.num_embeddings):
            raise IndexError('hashed embedding index is outside the source table')
        output = torch.empty((len(flat), self.weight.shape[1]), dtype=self.weight.dtype, device='cpu')
        assignments = torch.bucketize(flat, self.ends, right=True)
        for shard in assignments.unique().tolist():
            positions = torch.where(assignments == shard)[0]
            offset = int(self.ends[shard - 1]) if shard else 0
            output[positions] = self.parts[shard][flat[positions] - offset]
        return output.reshape(*ids.shape, self.weight.shape[1]).to(ids.device)


def load_flash_block(block, position, index, shards, prefix, device):
    """Keep the hashed embedding on the host while loading one decoder block."""
    base = prefix + f"layers.{position}."
    embedding_parts = []
    for name, shard in index.items():
        if not name.startswith(base):
            continue
        relative = name[len(base):]
        match = re.fullmatch(r"(.+ngram_embedding)\.shard_(\d+)\.weight", relative)
        if match:
            embedding_parts.append((int(match[2]), match[1], name, shard))
            continue
        value = shards.get(shard).get_tensor(name)
        if relative.endswith('.ngram_embedding.weight'):
            set_nested_attr(block, relative, value)
        else:
            set_nested_attr(block, relative, value.to(device))
    if embedding_parts:
        embedding_parts.sort()
        if (len({p[1] for p in embedding_parts}) != 1
                or [p[0] for p in embedding_parts] != list(range(len(embedding_parts)))):
            raise ValueError('hashed embedding checkpoint has inconsistent shard indices')
        embedding = block.get_submodule(embedding_parts[0][1])
        parts = [shards.get(shard).get_tensor(name) for _, _, name, shard in embedding_parts]
        lookup = HostShardedEmbedding(parts)
        if (lookup.num_embeddings, lookup.weight.shape[1]) != tuple(embedding.weight.shape):
            raise ValueError('hashed embedding checkpoint is incomplete')
        set_nested_attr(block, embedding_parts[0][1], lookup)
    if any(value.is_meta for value in block.state_dict().values()):
        raise ValueError('source checkpoint did not materialize every decoder tensor')


@torch.inference_mode()
def quantize(source, calibration, output, tokens=16384, chunk=512, flips=64):
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoTokenizer
    from source_provenance import save_source_fingerprint
    if output.exists() or min(tokens, chunk, flips) < 1:
        raise ValueError("choose a fresh output and positive calibration settings")
    raw_config = json.loads((source / 'config.json').read_text())
    if raw_config.get('model_type') in ('qwen4_exp', 'qwen4_exp_text'):
        # The serving plugin may register its compatibility config in the same
        # process. Calibration needs the native Transformers config instead.
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig, Qwen4ExpTextConfig
        cls = Qwen4ExpConfig if raw_config['model_type'] == 'qwen4_exp' else Qwen4ExpTextConfig
        config = cls.from_dict(raw_config)
    else:
        config = AutoConfig.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    if getattr(config, "quantization_config", None):
        raise ValueError("an original full-precision checkpoint is required")
    text, Model, Layer = architecture(config)
    text._attn_implementation = "sdpa"
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
    raw = calibration.read_text(encoding="utf-8")
    ids = tokenizer(raw, return_tensors="pt", truncation=True, max_length=tokens).input_ids.to(device)
    if ids.shape[1] < chunk:
        raise ValueError("calibration text is shorter than one chunk")
    ids = [ids[:, start:start + chunk] for start in range(0, ids.shape[1], chunk)
           if ids.shape[1] - start >= 32]
    index = load_weight_index(source)
    shards = ShardCache(source)
    prefix = "model.language_model." if "model.language_model.embed_tokens.weight" in index else "model."
    def load(name):
        return shards.get(index[name]).get_tensor(name).to(device, torch.bfloat16)
    # Factory allocations must also be meta for Flash's large embedding table,
    # before Accelerate can intercept parameter registration.
    with init_empty_weights(), (torch.device('meta') if text.model_type == 'qwen4_exp_text' else nullcontext()):
        model = Model(text)
    model.eval()
    model.model.embed_tokens.weight = nn.Parameter(load(prefix + "embed_tokens.weight"), requires_grad=False)
    model.model.rotary_emb = type(model.model.rotary_emb)(config=text).to(device)
    def layer(position):
        with init_empty_weights(), (torch.device('meta') if text.model_type == 'qwen4_exp_text' else nullcontext()):
            block = Layer(text, layer_idx=position)
            if prefix + f"layers.{position}.mlp.experts.0.gate_proj.weight" in index:
                individual_expert_layout(block, text)
        if text.model_type == 'qwen4_exp_text':
            load_flash_block(block, position, index, shards, prefix, device)
            block.eval()
        else:
            load_block_from_safetensors(block, position, index, shards, prefix + "layers.")
            block = block.to(device).eval()
        unfuse_moe_experts(block, verbose=False)
        block.mlp = CalibrationBlock(block.mlp, text)
        return block
    first = layer(0)
    model.model.layers[0] = first
    cached = [list(capture_layer_input(model.model, row)) for row in ids]
    total_tokens = sum(row.shape[1] for row in ids)
    model.model.layers[0] = None
    del first, ids
    model.model.embed_tokens.weight = nn.Parameter(torch.empty(0), requires_grad=False)
    settings = dict(n_flip_iters=flips, ridge=1., flip=True, exp_screen=True, fast_flip=True,
                    exp_offsets=(1, 0, -1), pre_alpha=.75, calib_mode="router")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / "checkpoint"
        writer = ShardedSafetensorsWriter(pending, 2_000_000_000)
        for position in range(text.num_hidden_layers):
            block = layer(position)
            packed = quantize_block_moe(block, position, cached, settings, device)
            if set(packed) != {"mlp.experts"}:
                raise ValueError("unexpected expert inventory")
            for name, tensor in packed["mlp.experts"].items():
                writer.add(f"model.layers.{position}.mlp.experts.{name}", tensor)
            original_prefix = prefix + f'layers.{position}.'
            for name in index:
                if name.startswith(original_prefix) and '.mlp.experts.' not in name:
                    writer.add('model.' + name[len(prefix):], shards.get(index[name]).get_tensor(name))
            block.mlp.experts = PackedExperts(packed["mlp.experts"], device)
            for i, (args, kwargs) in enumerate(cached):
                result = block(*args, **kwargs)
                hidden = result[0] if isinstance(result, tuple) else result
                cached[i][0] = (hidden.detach().clone(),) + tuple(args[1:])
            del block, packed
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Quantized layer {position + 1}/{text.num_hidden_layers}", flush=True)
        for name in index:
            if (name == 'lm_head.weight' or name.startswith(prefix)) and '.layers.' not in name and '.mtp.' not in name:
                saved_name = 'model.' + name[len(prefix):] if name.startswith(prefix) else name
                writer.add(saved_name, shards.get(index[name]).get_tensor(name))
        writer.finalize()
        cfg = serving_config(raw_config, text, Model.__name__)
        (pending / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        for path in source.iterdir():
            if path.is_file() and path.name.startswith(("tokenizer", "vocab", "merges", "special_tokens",
                                                       "added_tokens", "chat_template", "generation_config", "LICENSE")):
                shutil.copyfile(path, pending / path.name)
        save_source_fingerprint(source, pending)
        recipe = dict(model_type=text.model_type, expert_layers=text.num_hidden_layers,
                      experts_per_layer=text.num_experts, calibration_sha256=hashlib.sha256(raw.encode()).hexdigest(),
                      tokens=total_tokens, chunk=chunk, flips=flips, precision="w4a4", text_only=True)
        (pending / "macaq_recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
        pending.rename(output)
    shards.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tokens", type=int, default=16384)
    p.add_argument("--chunk", type=int, default=512)
    p.add_argument("--flips", type=int, default=64)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    if a.gpu < 0:p.error("GPU index must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    torch.set_num_threads(4)
    quantize(a.model, a.calibration, a.output, a.tokens, a.chunk, a.flips)

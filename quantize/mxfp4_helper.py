"""Shared infrastructure for MACaQ MXFP4 quantizers.

Includes perplexity evaluation, Hadamard construction, model-memory reporting,
MoE projection adapters and streaming safetensors loading.

`CACHE_DIR` follows `HF_HUB_CACHE`; otherwise datasets and transformers use
their default cache locations.
"""
import os
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors import safe_open

# HF cache dir; None => library default (~/.cache/huggingface).
CACHE_DIR = os.environ.get("HF_HUB_CACHE") or None

MXFP4_BLOCK = 32  # default microscale block size (32 elements per E8M0 scale)


# ── Hadamard ─────────────────────────────────────────────────────────────────
def make_hadamard(n: int, device="cpu") -> torch.Tensor:
    """Normalized n×n Sylvester Hadamard (orthogonal)."""
    H = torch.tensor(
        [[(-1) ** bin(i & j).count("1") for j in range(n)] for i in range(n)],
        dtype=torch.float32, device=device,
    ) / math.sqrt(n)
    return H


def largest_pow2_divisor(n: int) -> int:
    """Largest power of 2 that divides n (e.g. 5120 -> 1024)."""
    return n & (-n)


# ── Capture a decoder block's forward inputs (for streaming calibration) ──────
class _CapturedLayerInput(Exception):
    """Raised by the capture pre-hook to abort the forward and return inputs."""
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def _deep_detach_clone(x):
    """Detach + clone every tensor leaf in args/kwargs (handles tuples/lists/dicts)."""
    if torch.is_tensor(x):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(_deep_detach_clone(v) for v in x)
    if isinstance(x, list):
        return [_deep_detach_clone(v) for v in x]
    if isinstance(x, dict):
        return {k: _deep_detach_clone(v) for k, v in x.items()}
    return x


@torch.no_grad()
def capture_layer_input(text_model, ids, layer_idx: int = 0):
    """Run `text_model(ids)` only far enough to reach `layers[layer_idx]`, then
    return the `(args, kwargs)` that layer would have been called with. Aborts
    the forward via a pre-hook that raises, so nothing past that layer runs.

    The returned args/kwargs are detached clones, safe to keep across later
    forward passes. `use_cache=False` is forced so the per-layer kwargs don't
    carry a Cache that subsequent layer calls would mutate. This is how the MoE
    quantizer gets a block's full forward signature (position embeddings, GDN
    state, masks) without hand-reconstructing the hybrid plumbing."""
    def _pre_hook(module, args, kwargs):
        raise _CapturedLayerInput(_deep_detach_clone(args),
                                  _deep_detach_clone(kwargs))

    handle = text_model.layers[layer_idx].register_forward_pre_hook(
        _pre_hook, with_kwargs=True)
    try:
        text_model(ids, use_cache=False)
    except _CapturedLayerInput as e:
        return e.args, e.kwargs
    finally:
        handle.remove()
    raise RuntimeError(f"capture_layer_input: pre-hook on layer {layer_idx} never fired")


# ── Fused-MoE unfusion (Qwen3.5-MoE quantizer) ───────────────────────────────
class QuantizedFusedExperts(nn.Module):
    """Drop-in replacement for HF fused-expert modules. Holds per-expert weights
    as nn.Linear inside two ModuleLists so the nn.Linear-based quantization path
    sees each expert weight individually.

    Forward signature matches HF Qwen3_5MoeExperts.forward:
        (hidden_states, top_k_index, top_k_weights) -> hidden_states
    """
    def __init__(self, num_experts: int, act_fn):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = act_fn
        self.gate_up_proj = nn.ModuleList()
        self.down_proj = nn.ModuleList()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            e = int(expert_idx[0])
            if e == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            current_state = hidden_states[token_idx]
            gate, up = self.gate_up_proj[e](current_state).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = self.down_proj[e](current_hidden_states)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(
                0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
            )
        return final_hidden_states


def _make_linear_view(in_features: int, out_features: int,
                      weight: torch.Tensor) -> nn.Linear:
    """nn.Linear that uses `weight` directly (no copy, no placeholder alloc)."""
    lin = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(lin)
    lin.in_features = in_features
    lin.out_features = out_features
    lin.weight = nn.Parameter(weight, requires_grad=False)
    lin.bias = None
    return lin


def unfuse_moe_experts(model, verbose: bool = True) -> int:
    """Find HF fused-experts modules (3-D `gate_up_proj` / `down_proj`
    nn.Parameters) and replace them with QuantizedFusedExperts containing
    per-expert nn.Linear views into the original storage (no memory copy)."""
    targets = []
    for name, mod in model.named_modules():
        gu = getattr(mod, "gate_up_proj", None)
        dp = getattr(mod, "down_proj", None)
        if (isinstance(gu, nn.Parameter) and gu.dim() == 3
                and isinstance(dp, nn.Parameter) and dp.dim() == 3
                and gu.shape[0] == dp.shape[0]):
            targets.append((name, mod))

    n_per_expert = 0
    for name, mod in targets:
        gu = mod.gate_up_proj.data
        dp = mod.down_proj.data
        E, gu_out, gu_in = gu.shape
        _, dp_out, dp_in = dp.shape
        if gu_in != dp_out or gu_out != 2 * dp_in:
            print(f"  warning: skipping {name} , unexpected fused shapes "
                  f"gate_up={tuple(gu.shape)} down={tuple(dp.shape)}")
            continue
        act_fn = getattr(mod, "act_fn", nn.functional.silu)
        replacement = QuantizedFusedExperts(E, act_fn)
        for e in range(E):
            replacement.gate_up_proj.append(_make_linear_view(gu_in, gu_out, gu[e]))
            replacement.down_proj.append(_make_linear_view(dp_in, dp_out, dp[e]))
        n_per_expert += 2 * E
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent, child_name = model, name
        setattr(parent, child_name, replacement)

    if targets:
        gc.collect(); torch.cuda.empty_cache()
        if verbose:
            print(f"  unfused {len(targets)} MoE expert module(s) -> "
                  f"{n_per_expert} per-expert nn.Linear")
    return len(targets)


# ── WikiText-2 perplexity ─────────────────────────────────────────────────────
def evaluate_wikitext2(model, tokenizer, seq_len: int = 2048,
                       device: str = "cuda", batch_size: int = 1) -> float:
    """Non-overlapping-window WikiText-2 PPL. `batch_size > 1` groups several
    windows into a single forward pass (major speedup on MoE)."""
    from datasets import load_dataset
    gc.collect(); torch.cuda.empty_cache()
    # "Salesforce/wikitext", not the bare "wikitext": the legacy script-based id
    # was removed from the Hub and raises HfUriError on datasets >= 3.
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                        cache_dir=CACHE_DIR)
    enc = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids = enc.input_ids[0]
    total_len = input_ids.size(0)

    n_full = total_len // seq_len
    nlls, n_tokens = [], 0
    if n_full > 0:
        windows = input_ids[:n_full * seq_len].view(n_full, seq_len)
        for bs in tqdm(range(0, n_full, batch_size), desc="PPL", leave=False):
            batch = windows[bs:bs + batch_size].to(device)
            with torch.no_grad():
                loss = model(batch, labels=batch).loss
            n_tok = batch.numel()
            nlls.append(loss.float() * n_tok)
            n_tokens += n_tok

    tail_start = n_full * seq_len
    if total_len - tail_start >= 10:
        tail = input_ids[tail_start:].unsqueeze(0).to(device)
        with torch.no_grad():
            loss = model(tail, labels=tail).loss
        n_tok = tail.numel()
        nlls.append(loss.float() * n_tok)
        n_tokens += n_tok

    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()


# ── Memory accounting ─────────────────────────────────────────────────────────
def print_model_memory(model, label: str = "Model"):
    """Total parameter + buffer memory, broken down by dtype."""
    dtype_bytes = {}
    total = 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        nbytes = t.nelement() * t.element_size()
        dt = str(t.dtype)
        dtype_bytes[dt] = dtype_bytes.get(dt, 0) + nbytes
        total += nbytes
    print(f"\n  [{label}] Total model state: {total / 1e9:.3f} GB")
    for dt in sorted(dtype_bytes):
        print(f"    {dt:35s}: {dtype_bytes[dt] / 1e9:.3f} GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            if alloc > 0.001:
                peak = torch.cuda.max_memory_allocated(i) / 1e9
                print(f"    CUDA:{i} allocated: {alloc:.3f} GB  |  peak: {peak:.3f} GB")


# ── Streaming safetensors helpers (used by the per-block MoE quantizer) ───────
def find_snapshot_dir(repo_id: str, cache_dir: str = CACHE_DIR) -> Path:
    """Most-recent HF-cache snapshot directory for `repo_id`."""
    base = Path(cache_dir or os.path.expanduser("~/.cache/huggingface/hub")) \
        / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    snaps = [s for s in base.iterdir() if s.is_dir()] if base.exists() else []
    if not snaps:
        raise FileNotFoundError(f"No cached snapshot for {repo_id} under {base}")
    return sorted(snaps, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_weight_index(snapshot_dir: Path) -> dict:
    """Return {tensor_name -> shard_file_basename}. Handles single-shard models."""
    idx = snapshot_dir / "model.safetensors.index.json"
    if idx.exists():
        with open(idx) as f:
            return json.load(f)["weight_map"]
    sf = list(snapshot_dir.glob("model*.safetensors"))
    if len(sf) != 1:
        raise FileNotFoundError(f"No index and != 1 shards in {snapshot_dir}: {sf}")
    with safe_open(str(sf[0]), framework="pt") as f:
        return {k: sf[0].name for k in f.keys()}


class ShardCache:
    """Holds open safetensors handles so each shard is opened once."""
    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.handles = {}

    def get(self, shard_name: str):
        h = self.handles.get(shard_name)
        if h is None:
            h = safe_open(str(self.snapshot_dir / shard_name), framework="pt")
            h.__enter__()
            self.handles[shard_name] = h
        return h

    def close(self):
        for h in self.handles.values():
            h.__exit__(None, None, None)
        self.handles.clear()


def set_nested_attr(root: nn.Module, dotted: str, value):
    """Walk dotted attr path on `root` and assign `value` to the leaf."""
    *path, leaf = dotted.split(".")
    target = root
    for p in path:
        target = getattr(target, p)
    if isinstance(value, nn.Parameter):
        setattr(target, leaf, value)
    elif isinstance(value, torch.Tensor):
        existing = getattr(target, leaf, None)
        if isinstance(existing, nn.Parameter):
            setattr(target, leaf, nn.Parameter(value, requires_grad=False))
        else:
            try:
                target.register_buffer(leaf, value)
            except KeyError:
                setattr(target, leaf, value)
    else:
        setattr(target, leaf, value)


def load_block_from_safetensors(block: nn.Module, block_idx: int,
                                weight_map: dict, shards: ShardCache,
                                block_prefix: str,
                                dtype: torch.dtype = torch.bfloat16):
    """Copy weights for one transformer block into `block` (in place).
    `block_prefix` is the safetensors-key prefix BEFORE the layer index,
    e.g. "model.language_model.layers."."""
    prefix = f"{block_prefix}{block_idx}."
    block_keys = [k for k in weight_map if k.startswith(prefix)]
    by_shard = defaultdict(list)
    for k in block_keys:
        by_shard[weight_map[k]].append(k)
    for shard_name, names in by_shard.items():
        f = shards.get(shard_name)
        for full_name in names:
            t = f.get_tensor(full_name)
            rel = full_name[len(prefix):]
            set_nested_attr(block, rel, t.to(dtype))
            del t

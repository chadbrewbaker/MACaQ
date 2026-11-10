"""Batch-one attention over persistent speech KV buffers."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _gelu(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, offsets < N, 0).to(tl.float32)
    # Preserve each BF16 rounding point of the upstream GPT-2 activation.
    square = (x * x).to(tl.bfloat16).to(tl.float32)
    cube = (square * x).to(tl.bfloat16).to(tl.float32)
    inner = (cube * .044715).to(tl.bfloat16).to(tl.float32)
    inner = (x + inner).to(tl.bfloat16).to(tl.float32)
    inner = (inner * .7978845608028654).to(tl.bfloat16).to(tl.float32)
    tangent = libdevice.tanh(inner).to(tl.bfloat16).to(tl.float32)
    factor = (1. + tangent).to(tl.bfloat16).to(tl.float32)
    half = (.5 * x).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + offsets, half * factor, offsets < N)


class SpeechGELU(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.original = original

    def forward(self, x):
        if (torch.is_grad_enabled() or x.dtype != torch.bfloat16
                or not x.is_contiguous() or not x.is_cuda):
            return self.original(x)
        out = torch.empty_like(x)
        _gelu[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), 256, enable_fp_fusion=False)
        return out


def use_speech_activations(decoder):
    for layer in decoder.h:
        if layer.mlp.act.__class__.__name__ == "NewGELUActivation":
            layer.mlp.act = SpeechGELU(layer.mlp.act)


@triton.jit
def _attention_parts(Q, K, V, NewK, NewV, Position, Partial, Maximum, Sum,
                     CAP: tl.constexpr, D: tl.constexpr, PARTS: tl.constexpr,
                     SCALE: tl.constexpr, BLOCK: tl.constexpr, UPDATE: tl.constexpr):
    head, part = tl.program_id(0), tl.program_id(1)
    columns = tl.arange(0, D)
    positions = part * BLOCK + tl.arange(0, BLOCK)
    q = tl.load(Q + head * D + columns).to(tl.float32)
    position = tl.load(Position)
    valid = (positions < CAP) & (positions <= position)
    k = tl.load(K + head * CAP * D + positions[:, None] * D + columns[None, :],
                valid[:, None], 0).to(tl.float32)
    if UPDATE:
        fresh_k = tl.load(NewK + head * D + columns).to(tl.float32)
        k = tl.where(positions[:, None] == position, fresh_k[None, :], k)
        tl.store(K + head * CAP * D + positions[:, None] * D + columns[None, :],
                 fresh_k[None, :], (positions[:, None] == position) & (positions[:, None] < CAP))
    score = tl.sum(k * q[None, :], 1) * SCALE
    score = tl.where(valid, score, -float("inf"))
    maximum = tl.max(score, 0)
    safe_maximum = tl.where(maximum == -float("inf"), 0., maximum)
    weights = tl.exp(score - safe_maximum)
    values = tl.load(V + head * CAP * D + positions[:, None] * D + columns[None, :],
                     valid[:, None], 0).to(tl.float32)
    if UPDATE:
        fresh_v = tl.load(NewV + head * D + columns).to(tl.float32)
        values = tl.where(positions[:, None] == position, fresh_v[None, :], values)
        tl.store(V + head * CAP * D + positions[:, None] * D + columns[None, :],
                 fresh_v[None, :], (positions[:, None] == position) & (positions[:, None] < CAP))
    weighted = tl.sum(values * weights[:, None], 0)
    tl.store(Partial + (head * PARTS + part) * D + columns, weighted)
    tl.store(Maximum + head * PARTS + part, maximum)
    tl.store(Sum + head * PARTS + part, tl.sum(weights, 0))


@triton.jit
def _attention_merge(Partial, Maximum, Sum, Out, D: tl.constexpr,
                     PARTS: tl.constexpr, BP: tl.constexpr):
    head = tl.program_id(0)
    parts = tl.arange(0, BP)
    columns = tl.arange(0, D)
    maximum = tl.load(Maximum + head * PARTS + parts, parts < PARTS, -float("inf"))
    factors = tl.exp(maximum - tl.max(maximum, 0))
    denominator = tl.sum(tl.load(Sum + head * PARTS + parts, parts < PARTS, 0) * factors, 0)
    value = tl.load(Partial + (head * PARTS + parts[:, None]) * D + columns[None, :],
                    parts[:, None] < PARTS, 0)
    output = tl.sum(value * factors[:, None], 0) / denominator
    tl.store(Out + head * D + columns, output)


def decode_attention(query, key, value, position, scale, new_key=None, new_value=None):
    heads, capacity, dim = key.shape[1:]
    if (query.shape != (1, heads, 1, dim) or dim not in (32, 64, 128)
            or key.shape != value.shape or not key.is_contiguous() or not value.is_contiguous()
            or query.dtype != torch.bfloat16 or key.dtype != query.dtype
            or value.dtype != query.dtype or position.shape != (1,)
            or query.stride(-1) != 1 or query.stride(1) != dim):
        raise ValueError("unsupported speech attention layout")
    update = new_key is not None
    if update != (new_value is not None):
        raise ValueError("both new KV tensors are required")
    if update and any(x.shape != query.shape or x.dtype != query.dtype
                      or x.stride(-1) != 1 or x.stride(1) != dim for x in (new_key, new_value)):
        raise ValueError("unsupported new KV layout")
    block = 128
    parts = triton.cdiv(capacity, block)
    partial = torch.empty((heads, parts, dim), device=query.device, dtype=torch.float32)
    maximum = torch.empty((heads, parts), device=query.device, dtype=torch.float32)
    sums = torch.empty_like(maximum)
    out = torch.empty_like(query, memory_format=torch.contiguous_format)
    _attention_parts[(heads, parts)](query, key, value, new_key, new_value, position, partial, maximum, sums,
        capacity, dim, parts, scale, block, update, num_warps=4, enable_fp_fusion=False)
    _attention_merge[(heads,)](partial, maximum, sums, out, dim, parts,
        triton.next_power_of_2(parts), num_warps=4, enable_fp_fusion=False)
    return out


def use_decode_attention(decoder, cache):
    from functools import wraps
    for layer in decoder.h:
        attn = layer.attn
        if hasattr(attn, "_macaq_original_forward"):
            continue
        original = attn.forward
        def wrap(attn, original):
            @wraps(original)
            def forward(hidden_states, past_key_values=None, cache_position=None, **kwargs):
                if (hidden_states.shape[:2] != (1, 1) or attn.training or torch.is_grad_enabled()
                        or hidden_states.dtype != torch.bfloat16 or cache_position is None
                        or past_key_values is not cache
                        or kwargs.get("encoder_hidden_states") is not None
                        or kwargs.get("output_attentions") or attn.reorder_and_upcast_attn):
                    return original(hidden_states, past_key_values=past_key_values,
                                    cache_position=cache_position, **kwargs)
                q, k, v = attn.c_attn(hidden_states).split(attn.split_size, dim=2)
                shape = (1, 1, attn.num_heads, attn.head_dim)
                q, k, v = [x.view(shape).transpose(1, 2) for x in (q, k, v)]
                stored = past_key_values.layers[attn.layer_idx]
                scale = attn.head_dim ** -.5 if attn.scale_attn_weights else 1.
                if attn.scale_attn_by_inverse_layer_idx:
                    scale /= attn.layer_idx + 1
                out = decode_attention(q, stored.keys, stored.values, cache_position, scale, k, v)
                return attn.c_proj(out.reshape(1, 1, attn.embed_dim)), None
            return forward
        attn._macaq_original_forward = original
        attn.forward = wrap(attn, original)

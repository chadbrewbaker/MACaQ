"""Small-batch W4A4 speech projections using Triton device code."""

import sys
from pathlib import Path
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diffusers"))
from kernels import FusedW4A4Linear


class SpeechW4A4Linear(FusedW4A4Linear):
    """Fuse rotation, activation packing, FP4 multiplication and output rounding.

    A narrow output tile supplies enough independent blocks for single-token
    decode. Multi-token prefill retains the existing dense kernel.
    """
    def forward(self, x):
        if (x.numel() != self.in_features or x.dtype != torch.bfloat16
                or not x.is_contiguous() or self.activation_precision != "w4a4"
                or self.in_features % 1024 or self.B_inv_T.shape != (32, 32)
                or self.pre_scale is None):
            return super().forward(x)
        out = torch.empty((*x.shape[:-1], self.out_features), device=x.device, dtype=x.dtype)
        _decode_native[(triton.cdiv(self.out_features, 32),)](
            x, self.pre_scale, self.B_inv_T, self.W_packed, self.W_scales,
            self.bias, out, self.out_features, self.in_features, self.bias is not None,
            BN=32, BK=1024, num_warps=4, num_stages=2)
        return out


def use_speech_kernels(model):
    for name, module in list(model.named_modules()):
        if isinstance(module, FusedW4A4Linear):
            module.__class__ = SpeechW4A4Linear


@triton.jit
def _decode_native(X, Pre, BH, W, WS, Bias, Y, N: tl.constexpr, K: tl.constexpr,
                   HAS_BIAS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    lane = tl.arange(0, 32)
    groups = tl.arange(0, BK // 32)
    rows = tl.program_id(0) * BN + tl.arange(0, BN)
    kb = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 32)
    bh = tl.load(BH + lane[:, None] * 32 + lane[None, :])
    acc = tl.zeros((16, BN), tl.float32)
    for start in range(triton.cdiv(K, BK)):
        offsets = start * BK + groups[:, None] * 32 + lane[None, :]
        x = tl.load(X + offsets, offsets < K, 0).to(tl.float32)
        pre = tl.load(Pre + offsets, offsets < K, 1).to(tl.float32)
        scaled = (x * pre).to(tl.bfloat16)
        r = tl.dot(scaled, bh, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        lo, hi = tl.split(tl.reshape(r, (BK // 32, 16, 2)))
        maximum = tl.maximum(tl.maximum(tl.max(tl.abs(lo), 1), tl.max(tl.abs(hi), 1)), 1e-20)
        e = tl.minimum(tl.maximum(tl.floor(tl.log2(maximum)) - 2.0, -127.0), 127.0)
        scale = tl.exp2(e)[:, None]
        al, ah = tl.abs(lo) / scale, tl.abs(hi) / scale
        cl = ((al >= .25).to(tl.int32) + (al >= .75) + (al >= 1.25) + (al >= 1.75)
              + (al >= 2.5) + (al >= 3.5) + (al >= 5.0)) | ((lo < 0).to(tl.int32) << 3)
        ch = ((ah >= .25).to(tl.int32) + (ah >= .75) + (ah >= 1.25) + (ah >= 1.75)
              + (ah >= 2.5) + (ah >= 3.5) + (ah >= 5.0)) | ((hi < 0).to(tl.int32) << 3)
        packed = tl.reshape((cl | (ch << 4)).to(tl.uint8), (1, BK // 2))
        asc = tl.reshape((e + 127.0).to(tl.uint8), (1, BK // 32))
        a = tl.broadcast_to(packed, (16, BK // 2))
        asc = tl.broadcast_to(asc, (16, BK // 32))
        w = tl.load(W + rows[:, None] * (K // 2) + start * (BK // 2) + kb[None, :], (rows[:, None] < N) & (start * (BK // 2) + kb[None, :] < K // 2), 0)
        ws = tl.load(WS + rows[:, None] * (K // 32) + start * (BK // 32) + ks[None, :], (rows[:, None] < N) & (start * (BK // 32) + ks[None, :] < K // 32), 127)
        acc = tl.dot_scaled(a, asc, 'e2m1', w.T, ws, 'e2m1', acc)
    rowid = tl.arange(0, 16)
    value = tl.sum(tl.where(rowid[:, None] == 0, acc, 0.), 0).to(tl.bfloat16)
    if HAS_BIAS:
        value = (value.to(tl.float32) + tl.load(Bias + rows, rows < N, 0).to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + rows, value, rows < N)


@triton.jit
def _rms(X, Weight, Y, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    values = tl.load(X + row * N + columns, columns < N, 0).to(tl.float32)
    variance = tl.sum(values * values, 0) / N
    normalized = (values * tl.rsqrt(variance + EPS)).to(tl.bfloat16).to(tl.float32)
    weight = tl.load(Weight + columns, columns < N, 0).to(tl.float32)
    tl.store(Y + row * N + columns, normalized * weight, columns < N)


class SpeechRMSNorm(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.weight = source.weight
        self.variance_epsilon = source.variance_epsilon
        self.original = source

    def forward(self, x):
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            return self.original(x)
        out = torch.empty_like(x)
        n = x.shape[-1]
        _rms[(x.numel() // n,)](x, self.weight, out, n, self.variance_epsilon,
                              BLOCK=triton.next_power_of_2(n), enable_fp_fusion=False)
        return out


def use_speech_norms(model):
    for name, module in list(model.named_modules()):
        if module.__class__.__name__ == 'Qwen3TTSRMSNorm':
            parent, _, leaf = name.rpartition('.')
            setattr(model.get_submodule(parent), leaf, SpeechRMSNorm(module))

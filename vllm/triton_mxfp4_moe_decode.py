"""Fused small-batch grouped W4A4 expert decode without token sorting."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _activation(X, Pre, BH, row, expert, start, K: tl.constexpr, BK: tl.constexpr):
    lane = tl.arange(0, 32)
    groups = tl.arange(0, BK // 32)
    offsets = start + groups[:, None] * 32 + lane[None, :]
    x = tl.load(X + row * K + offsets, offsets < K, 0).to(tl.float32)
    p = tl.load(Pre + expert * K + offsets, offsets < K, 1).to(tl.float32)
    bh = tl.load(BH + lane[:, None] * 32 + lane[None, :])
    rotated = tl.dot((x * p).to(tl.bfloat16), bh, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
    lo, hi = tl.split(tl.reshape(rotated, (BK // 32, 16, 2)))
    maximum = tl.maximum(tl.maximum(tl.max(tl.abs(lo), 1), tl.max(tl.abs(hi), 1)), 1.e-20)
    exponent = tl.minimum(tl.maximum(tl.floor(tl.log2(maximum)) - 2., -127.), 127.)
    scale = tl.exp2(exponent)[:, None]
    a, b = tl.abs(lo) / scale, tl.abs(hi) / scale
    ca = ((a >= .25).to(tl.int32) + (a >= .75) + (a >= 1.25) + (a >= 1.75)
          + (a >= 2.5) + (a >= 3.5) + (a >= 5.)) | ((lo < 0).to(tl.int32) << 3)
    cb = ((b >= .25).to(tl.int32) + (b >= .75) + (b >= 1.25) + (b >= 1.75)
          + (b >= 2.5) + (b >= 3.5) + (b >= 5.)) | ((hi < 0).to(tl.int32) << 3)
    packed = tl.reshape((ca | (cb << 4)).to(tl.uint8), (1, BK // 2))
    scales = tl.reshape((exponent + 127.).to(tl.uint8), (1, BK // 32))
    return tl.broadcast_to(packed, (16, BK // 2)), tl.broadcast_to(scales, (16, BK // 32))


@triton.jit
def _gate_up(X, Pre, BH, W, WS, Ids, Out, H: tl.constexpr, I: tl.constexpr,
             TOP: tl.constexpr, E: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(Ids + route).to(tl.int64)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    if (expert < 0) | (expert >= E):
        tl.store(Out + route * I + rows, 0., rows < I)
        return
    kb = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 32)
    gate = tl.zeros((16, BN), tl.float32)
    up = tl.zeros((16, BN), tl.float32)
    for block in range(triton.cdiv(H, BK)):
        activation, asc = _activation(X, Pre, BH, route // TOP, expert, block * BK, H, BK)
        offset = expert * (2 * I) * (H // 2) + rows[:, None] * (H // 2) + block * (BK // 2) + kb[None, :]
        soffset = expert * (2 * I) * (H // 32) + rows[:, None] * (H // 32) + block * (BK // 32) + ks[None, :]
        mask = (rows[:, None] < I) & (block * (BK // 2) + kb[None, :] < H // 2)
        smask = (rows[:, None] < I) & (block * (BK // 32) + ks[None, :] < H // 32)
        wg = tl.load(W + offset, mask, 0)
        wu = tl.load(W + offset + I * (H // 2), mask, 0)
        sg = tl.load(WS + soffset, smask, 127)
        su = tl.load(WS + soffset + I * (H // 32), smask, 127)
        gate = tl.dot_scaled(activation, asc, 'e2m1', wg.T, sg, 'e2m1', gate)
        up = tl.dot_scaled(activation, asc, 'e2m1', wu.T, su, 'e2m1', up)
    first = tl.arange(0, 16) == 0
    g = tl.sum(tl.where(first[:, None], gate, 0.), 0).to(tl.bfloat16).to(tl.float32)
    u = tl.sum(tl.where(first[:, None], up, 0.), 0).to(tl.bfloat16).to(tl.float32)
    sigmoid = (1. / (1. + libdevice.exp(-g))).to(tl.bfloat16).to(tl.float32)
    inter = (g * sigmoid).to(tl.bfloat16).to(tl.float32) * u
    tl.store(Out + route * I + rows, inter, rows < I)


@triton.jit
def _down(X, Pre, BH, W, WS, Ids, Weights, Out, H: tl.constexpr, I: tl.constexpr,
          E: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    route = tl.program_id(0)
    expert = tl.load(Ids + route).to(tl.int64)
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    if (expert < 0) | (expert >= E):
        tl.store(Out + route * H + rows, 0., rows < H)
        return
    kb = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 32)
    accumulator = tl.zeros((16, BN), tl.float32)
    for block in range(triton.cdiv(I, BK)):
        activation, asc = _activation(X, Pre, BH, route, expert, block * BK, I, BK)
        offset = expert * H * (I // 2) + rows[:, None] * (I // 2) + block * (BK // 2) + kb[None, :]
        soffset = expert * H * (I // 32) + rows[:, None] * (I // 32) + block * (BK // 32) + ks[None, :]
        w = tl.load(W + offset, (rows[:, None] < H) & (block * (BK // 2) + kb[None, :] < I // 2), 0)
        ws = tl.load(WS + soffset, (rows[:, None] < H) & (block * (BK // 32) + ks[None, :] < I // 32), 127)
        accumulator = tl.dot_scaled(activation, asc, 'e2m1', w.T, ws, 'e2m1', accumulator)
    first = tl.arange(0, 16) == 0
    value = tl.sum(tl.where(first[:, None], accumulator, 0.), 0)
    value *= tl.load(Weights + route).to(tl.float32)
    tl.store(Out + route * H + rows, value, rows < H)


def fused_moe_decode(hidden, w13, w13_scales, w2, w2_scales, ids, weights, pre13, pre2, bh):
    tokens, width = hidden.shape
    top = ids.shape[1]
    intermediate = w13.shape[1] // 2
    hidden = hidden.contiguous()
    ids, weights = ids.contiguous(), weights.contiguous()
    inter = torch.empty((tokens * top, intermediate), dtype=torch.bfloat16, device=hidden.device)
    partial = torch.empty((tokens * top, width), dtype=torch.bfloat16, device=hidden.device)
    _gate_up[(tokens * top, triton.cdiv(intermediate, 32))](hidden, pre13, bh, w13, w13_scales,
        ids, inter, width, intermediate, top, w13.shape[0], 32, 512, num_warps=4, enable_fp_fusion=False)
    _down[(tokens * top, triton.cdiv(width, 32))](inter, pre2, bh, w2, w2_scales, ids, weights,
        partial, width, intermediate, w13.shape[0], 32, 512, num_warps=4, enable_fp_fusion=False)
    return partial.view(tokens, top, width).sum(dim=1, dtype=torch.float32).to(torch.bfloat16)

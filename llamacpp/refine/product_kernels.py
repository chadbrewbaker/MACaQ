"""Batched joint IQ4 product-codebook refinement with Triton."""

import torch
import triton
import triton.language as tl
from gguf_codecs import QT, TableCodec


@triton.jit
def _sweep(Values, Codes, Scales, Residual, Hessian, Table, Order, Moves,
           K: tl.constexpr, ROWS: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    expert = row // ROWS
    columns = tl.arange(0, BK)
    offset = row * K + columns
    values = tl.load(Values + offset, columns < K, 0)
    codes = tl.load(Codes + offset, columns < K, 0).to(tl.int32)
    residual = tl.load(Residual + offset, columns < K, 0)
    hbase = expert * K * K
    choices = tl.arange(0, 256)
    word0 = tl.load(Table + choices // 16)
    word1 = tl.load(Table + choices % 16)
    moves = tl.full((), 0, tl.int64)
    for step in range(K // 2):
        first = 2 * tl.load(Order + step)
        current0 = tl.sum(tl.where(columns == first, values, 0.), 0)
        current1 = tl.sum(tl.where(columns == first + 1, values, 0.), 0)
        r0 = tl.sum(tl.where(columns == first, residual, 0.), 0)
        r1 = tl.sum(tl.where(columns == first + 1, residual, 0.), 0)
        scale = tl.load(Scales + row * K + first)
        h00 = tl.load(Hessian + hbase + first * K + first)
        h01 = tl.load(Hessian + hbase + first * K + first + 1)
        h10 = tl.load(Hessian + hbase + (first + 1) * K + first)
        h11 = tl.load(Hessian + hbase + (first + 1) * K + first + 1)
        linear0 = current0 * h00 + current1 * h10 - r0
        linear1 = current0 * h01 + current1 * h11 - r1
        quadratic = word0 * word0 * h00 + word0 * word1 * h01 + word1 * word0 * h10 + word1 * word1 * h11
        cost = scale * scale * quadratic - 2 * scale * (linear0 * word0 + linear1 * word1)
        best = tl.min(cost, 0)
        chosen = tl.min(tl.where(cost == best, choices, 256), 0)
        current_cost = current0 * (current0 * h00 + current1 * h10) + current1 * (current0 * h01 + current1 * h11)
        current_cost -= 2 * (current0 * linear0 + current1 * linear1)
        accept = current_cost - best > 1.e-6 * (tl.abs(current_cost) + tl.abs(best) + 1.e-30)
        replacement0 = scale * tl.load(Table + chosen // 16)
        replacement1 = scale * tl.load(Table + chosen % 16)
        delta0 = tl.where(accept, replacement0 - current0, 0.)
        delta1 = tl.where(accept, replacement1 - current1, 0.)
        values = tl.where((columns == first) & accept, replacement0, values)
        values = tl.where((columns == first + 1) & accept, replacement1, values)
        codes = tl.where((columns == first) & accept, chosen // 16, codes)
        codes = tl.where((columns == first + 1) & accept, chosen % 16, codes)
        h0 = tl.load(Hessian + hbase + first * K + columns, columns < K, 0)
        h1 = tl.load(Hessian + hbase + (first + 1) * K + columns, columns < K, 0)
        residual += delta0 * h0 + delta1 * h1
        moves += accept.to(tl.int64)
    tl.store(Values + offset, values, columns < K)
    tl.store(Codes + offset, codes, columns < K)
    tl.store(Moves + row, tl.load(Moves + row) + moves)


@torch.no_grad()
def product_batch_refine(codec, states, targets, hessians, sweeps=2, seed=0):
    """Use a shared pair order across experts; every assignment searches 256 words."""
    if not isinstance(codec, TableCodec) or codec.qtype not in (QT.IQ4_NL, QT.IQ4_XS):
        raise ValueError("product VQ supports native IQ4_NL and IQ4_XS only")
    if not states or sweeps < 1:
        raise ValueError("states and positive sweeps are required")
    rows, columns = states[0].vals.shape
    if any(state.vals.shape != (rows, columns) or not torch.equal(state.se[:, ::2], state.se[:, 1::2])
           or torch.count_nonzero(state.me) for state in states):
        raise ValueError("joint IQ4 pairs require equal shapes, shared pair scales and zero offset")
    values = torch.stack([state.vals for state in states])
    codes = torch.stack([state.q for state in states])
    scales = torch.stack([state.se for state in states])
    target = torch.stack(targets)
    hessian = torch.stack(hessians)
    moves = torch.zeros((len(states), rows), device=values.device, dtype=torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for sweep in range(sweeps):
            residual = torch.bmm(values - target, hessian)
            before = moves.sum().item()
            order = torch.randperm(columns // 2, generator=generator).to(device=values.device, dtype=torch.int32)
            with torch.cuda.device(values.device):
                _sweep[(len(states) * rows,)](values, codes, scales, residual, hessian, codec.table, order, moves,
                    columns, rows, triton.next_power_of_2(columns), num_warps=4, enable_fp_fusion=False)
            if moves.sum().item() == before:
                break
        counts = moves.sum(dim=1).cpu().tolist()
        for index, state in enumerate(states):
            state.vals.copy_(values[index])
            state.q.copy_(codes[index])
        return [dict(moves=count, sweeps=sweep + 1) for count in counts]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

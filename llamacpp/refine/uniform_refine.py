"""Fixed-scale greedy refinement for native K-quant weight codes."""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None


if triton is not None:
    @triton.jit
    def _greedy(q_ptr, scale_ptr, residual_ptr, h_ptr,
                N: tl.constexpr, LEVELS: tl.constexpr,
                STEPS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        valid = columns < N
        codes = tl.load(q_ptr + row * N + columns, valid, 0).to(tl.int32)
        scale = tl.load(scale_ptr + row * N + columns, valid, 0)
        residual = tl.load(residual_ptr + row * N + columns, valid, 0)
        diagonal = tl.load(h_ptr + columns * N + columns, valid, 0)
        step = 0
        active = True
        while (step < STEPS) & active:
            up = scale * scale * diagonal + 2 * scale * residual
            down = scale * scale * diagonal - 2 * scale * residual
            up = tl.where(valid & (codes < LEVELS - 1), up, float('inf'))
            down = tl.where(valid & (codes > 0), down, float('inf'))
            best_up = tl.min(up, 0)
            best_down = tl.min(down, 0)
            increase = best_up <= best_down
            best = tl.minimum(best_up, best_down)
            active = best < -1.e-12
            costs = tl.where(increase, up, down)
            selected = tl.min(tl.where(costs == best, columns, 2147483647), 0)
            selected = tl.where(active, selected, 0)
            selected_scale = tl.sum(tl.where(columns == selected, scale, 0), 0)
            delta = tl.where(active, tl.where(increase, selected_scale, -selected_scale), 0)
            h_row = tl.load(h_ptr + selected * N + columns, valid & active, 0)
            residual += delta * h_row
            codes += tl.where((columns == selected) & active, tl.where(increase, 1, -1), 0)
            step += 1
        tl.store(q_ptr + row * N + columns, codes, valid)


@torch.no_grad()
def refine_codes(codec, state, target, hessian, iterations):
    """Update codes in place, retaining every native scale and offset."""
    from gguf_codecs import QT, TableCodec

    if not isinstance(codec, TableCodec) or codec.qtype not in (QT.Q4_K, QT.Q5_K, QT.Q6_K):
        raise ValueError('MACaQ-R supports Q4_K, Q5_K and Q6_K weight codes')
    if iterations < 1 or state.vals.shape != target.shape:
        raise ValueError('Invalid refinement iterations or target shape')
    rows, width = state.vals.shape
    if hessian.shape != (width, width):
        raise ValueError('Hessian shape differs from the input width')
    if state.vals.is_cuda:
        if triton is None:
            raise ValueError('Install the llama.cpp quantization dependencies for device refinement')
        state.q = state.q.contiguous()
        state.se = state.se.contiguous()
        residual = ((state.vals - target) @ hessian).contiguous()
        _greedy[(rows,)](
            state.q, state.se, residual, hessian.contiguous(), width,
            codec.n_levels, iterations, triton.next_power_of_2(width),
            num_warps=4, enable_fp_fusion=False,
        )
        state.vals = state.se * codec.table[state.q.long()] - state.me
    else:
        from macaq_refine_gguf import flip_refine
        flip_refine(codec, state, target, hessian, iterations)

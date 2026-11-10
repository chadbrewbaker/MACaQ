"""Joint two-dimensional VQ assignment over the native IQ4 product codebook.

Storage remains IQ4_NL/IQ4_XS. Assignment enumerates all 16 x 16 signed-level
pairs against the full Hessian objective; it is not coordinate-wise rounding.
"""

import time
import torch
from gguf_codecs import QT, TableCodec


@torch.no_grad()
def product_gs_refine(codec, st, target, hessian, n_sweeps=2, seed=0, log=None):
    if not isinstance(codec, TableCodec) or codec.qtype not in (QT.IQ4_NL, QT.IQ4_XS):
        raise ValueError("product VQ supports native IQ4_NL and IQ4_XS only")
    rows, columns = st.vals.shape
    if (
        columns % 2
        or not torch.equal(st.se[:, ::2], st.se[:, 1::2])
        or torch.count_nonzero(st.me)
    ):
        raise ValueError("IQ4 product pairs need a shared scale and zero offset")
    if n_sweeps < 1:
        raise ValueError("VQ sweeps must be positive")
    device = st.vals.device
    indices = torch.cartesian_prod(
        torch.arange(16, device=device), torch.arange(16, device=device)
    )
    words = codec.table[indices]
    squares = (words[:, :, None] * words[:, None, :]).reshape(256, 4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    stats = dict(
        code_moves=0,
        sign_moves=0,
        scale_moves=0,
        d_moves=0,
        sweeps=0,
        per_sweep=[],
        codebook_kind="product",
        vector_size=2,
        codebook_size=256,
        refit_scales=False,
    )
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:

        def loss():
            error = st.vals - target
            return float(((error @ hessian) * error).sum())

        stats["loss_start"] = loss()
        for sweep in range(n_sweeps):
            start = time.time()
            residual = (st.vals - target) @ hessian
            moves = 0
            for pair in torch.randperm(columns // 2, generator=generator).tolist():
                first = pair * 2
                block = hessian[first : first + 2, first : first + 2]
                current = st.vals[:, first : first + 2]
                scale = st.se[:, first]
                linear = current @ block - residual[:, first : first + 2]
                cost = scale.square()[:, None] * (squares @ block.reshape(4))[None, :]
                cost = cost - 2 * scale[:, None] * (linear @ words.T)
                current_cost = (current * (current @ block)).sum(1) - 2 * (
                    current * linear
                ).sum(1)
                best, codes = cost.min(1)
                accept = (current_cost - best) > 1e-6 * (
                    current_cost.abs() + best.abs() + 1e-30
                )
                if accept.any():
                    accepted = accept.nonzero().squeeze(1)
                    replacement = scale[accepted, None] * words[codes[accepted]]
                    delta = replacement - current[accepted]
                    st.q[accepted, first : first + 2] = indices[codes[accepted]].to(
                        st.q.dtype
                    )
                    st.vals[accepted, first : first + 2] = replacement
                    residual.index_add_(
                        0, accepted, delta @ hessian[first : first + 2, :]
                    )
                    moves += len(accepted)
            stats["sweeps"] = sweep + 1
            stats["code_moves"] += moves
            value = loss()
            stats["per_sweep"].append(
                dict(
                    loss=value,
                    seconds=time.time() - start,
                    code=moves,
                    sign=0,
                    scale=0,
                    d=0,
                )
            )
            if log:
                log(
                    f"      product sweep {sweep + 1}: loss {value:.6e}, joint pairs {moves}"
                )
            if not moves:
                break
        stats["loss_end"] = loss()
        stats["moves"] = stats["code_moves"]
        return stats
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

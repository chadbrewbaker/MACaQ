"""Block Gauss-Seidel refinement for native IQ2/IQ3 lattice codebooks.

IQ2 assigns eight weights per grid codeword; IQ3 assigns four. For each
vector block, candidate u minimizes u^T H_bb u - 2 u^T(H_bb v - residual_b).
The residual is updated across the complete row after every accepted move.

IQ3_S jointly searches grid points and free signs. IQ2 and IQ3_XXS alternate
whole-vector grid search with the 128 allowed even-parity sign patterns per
eight weights. Optional scale refinement uses the codec's 16/32-weight
sub-blocks and a rounded F16 scale for each 256-weight super-block.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from gguf_codecs import QT, GridCodec, State  # noqa: F401

REL_TOL = 1e-6   # accept a move only if it lowers the block objective by more than this, relative


def sign_patterns4(device) -> torch.Tensor:
    """(16, 4) of +-1; pattern p has coordinate k negative iff bit k of p is set."""
    p = torch.arange(16, device=device)
    bits = (p[:, None] >> torch.arange(4, device=device)[None, :]) & 1
    return torch.where(bits.bool(), -1.0, 1.0)


def signed_codebook(grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """grid (G, 4) -> (CW (G*16, 4), SP (16, 4)); codeword c = g*16 + p  ->  grid[g] * SP[p]."""
    SP = sign_patterns4(grid.device)
    CW = (grid[:, None, :] * SP[None, :, :]).reshape(-1, 4)
    return CW, SP


def parity_patterns8(ksigns: np.ndarray, device) -> torch.Tensor:
    """(128, 8) of +-1: the sign patterns IQ3_XXS can store (bit 7 = parity of bits 0..6)."""
    ks = torch.as_tensor(ksigns.astype(np.int64), device=device)
    bits = (ks[:, None] >> torch.arange(8, device=device)[None, :]) & 1
    return torch.where(bits.bool(), -1.0, 1.0)


def _accept(best: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    return (cur - best) > REL_TOL * (cur.abs() + best.abs() + 1e-30)


@torch.no_grad()
def block_gs_refine(codec: GridCodec, st: State, W_t: torch.Tensor, H: torch.Tensor,
                    n_sweeps: int = 6, refit_scales: bool = True, seed: int = 0, log=None) -> dict:
    assert isinstance(codec, GridCodec)
    R, N = st.vals.shape
    dev = st.vals.device
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False          # the block deltas must be exact
    try:
        return _block_gs_refine(codec, st, W_t, H, n_sweeps, refit_scales, seed, log, R, N, dev)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32


def _block_gs_refine(codec, st, W_t, H, n_sweeps, refit_scales, seed, log, R, N, dev):
    joint = codec.qtype == QT.IQ3_S
    grid = codec.grid                                        # (G, 4)
    if joint:
        CW, SP = signed_codebook(grid)                       # (8192, 4), (16, 4)
    else:
        CW = grid                                            # codec-sized vectors; signs in stage B
        P8 = parity_patterns8(codec.ksigns, dev)             # (128, 8)
        PP = (P8[:, :, None] * P8[:, None, :]).reshape(128, 64)
    width = codec.vector_size
    scale_group = codec.scale_group_size
    scale_groups_per_super = 256 // scale_group
    CW2 = (CW[:, :, None] * CW[:, None, :]).reshape(CW.shape[0], width * width)     # (C, 16)
    G = N // width
    nsub = N // scale_group
    nsup = N // 256
    gen = torch.Generator(device="cpu").manual_seed(seed)
    stats = dict(code_moves=0, sign_moves=0, scale_moves=0, d_moves=0, sweeps=0, per_sweep=[])

    def mags(cols0, ncols):
        """Grid magnitude per weight, using the codec vector width."""
        b0 = cols0 // width
        return grid[st.idx[:, b0:b0 + ncols // width]].reshape(R, ncols)

    def h_loss():
        E = st.vals - W_t
        return float(((E @ H) * E).sum())

    l_start = h_loss()
    for sweep in range(n_sweeps):
        t0 = time.time()
        HE = (st.vals - W_t) @ H                             # (R, N), refreshed each sweep against drift
        moved = dict(code=0, sign=0, scale=0, d=0)

        # ---- stage A: codewords per vector block -------------------------------------------------
        for b in torch.randperm(G, generator=gen).tolist():
            c0 = width * b
            Hbb = H[c0:c0 + width, c0:c0 + width]
            v = st.vals[:, c0:c0 + width]
            s = st.se[:, c0]                                                 # (R,)
            q = v @ Hbb - HE[:, c0:c0 + width]                                   # (R, 4) = (H_bb v - r_b)
            if joint:
                quad = CW2 @ Hbb.reshape(width * width)                                 # (C,)
                lin = q @ CW.T                                               # (R, C)
                delta = (s * s)[:, None] * quad[None, :] - 2 * s[:, None] * lin
            else:
                sg = st.sgn[:, c0:c0 + width]                                    # (R, 4)
                M = (sg[:, :, None] * sg[:, None, :]) * Hbb[None]            # (R, 4, 4)
                quad = M.reshape(R, width * width) @ CW2.T                              # (R, C)
                lin = (sg * q) @ CW.T                                        # (R, C)
                delta = (s * s)[:, None] * quad - 2 * s[:, None] * lin
            dcur = (v * (v @ Hbb)).sum(1) - 2 * (v * q).sum(1)
            best, c = delta.min(1)
            acc = _accept(best, dcur)
            if acc.any():
                rows = acc.nonzero().squeeze(1)
                cc = c[rows]
                if joint:
                    st.idx[rows, b] = cc // 16
                    st.sgn[rows, c0:c0 + width] = SP[cc % 16]
                    u = s[rows, None] * CW[cc]
                else:
                    st.idx[rows, b] = cc
                    u = s[rows, None] * sg[rows] * CW[cc]
                d = u - v[rows]
                st.vals[rows, c0:c0 + width] = u
                HE.index_add_(0, rows, d @ H[c0:c0 + width, :])
                moved["code"] += rows.numel()

        # ---- stage B: parity-constrained signs per 8-group ---------------------------------------
        if not joint:
            for t in torch.randperm(N // 8, generator=gen).tolist():
                c0 = 8 * t
                Hgg = H[c0:c0 + 8, c0:c0 + 8]
                v = st.vals[:, c0:c0 + 8]
                m = v.abs()                                                  # (R, 8)
                q = v @ Hgg - HE[:, c0:c0 + 8]
                A = (m[:, :, None] * m[:, None, :] * Hgg[None]).reshape(R, 64)
                delta = A @ PP.T - 2 * (m * q) @ P8.T                        # (R, 128)
                dcur = (v * (v @ Hgg)).sum(1) - 2 * (v * q).sum(1)
                best, p = delta.min(1)
                acc = _accept(best, dcur)
                if acc.any():
                    rows = acc.nonzero().squeeze(1)
                    sg = P8[p[rows]]                                         # (na, 8)
                    u = sg * m[rows]
                    d = u - v[rows]
                    # Store signs in codec space, including a possible negative base scale.
                    scale_sign = torch.where(st.se[rows, c0:c0 + 8] < 0, -1.0, 1.0)
                    st.sgn[rows, c0:c0 + 8] = sg * scale_sign
                    st.vals[rows, c0:c0 + 8] = u
                    HE.index_add_(0, rows, d @ H[c0:c0 + 8, :])
                    moved["sign"] += rows.numel()

        if refit_scales:
            # ---- stage C: 4-bit sub-block scale codes ------------------------------------------
            fac16 = codec.scale_factor(torch.arange(16, device=dev))         # (16,)
            for k in torch.randperm(nsub, generator=gen).tolist():
                c0 = scale_group * k
                Hss = H[c0:c0 + scale_group, c0:c0 + scale_group]
                v = st.vals[:, c0:c0 + scale_group]
                z = mags(c0, scale_group) * st.sgn[:, c0:c0 + scale_group]                     # (R, 32) direction
                q = v @ Hss - HE[:, c0:c0 + scale_group]
                A = ((z @ Hss) * z).sum(1)                                   # (R,)
                B = (z * q).sum(1)
                dsup = st.d[:, k // scale_groups_per_super]                                        # (R,)
                a_cand = dsup[:, None] * fac16[None, :]                      # (R, 16)
                delta = a_cand * a_cand * A[:, None] - 2 * a_cand * B[:, None]
                a_cur = st.se[:, c0]
                dcur = a_cur * a_cur * A - 2 * a_cur * B
                best, j = delta.min(1)
                acc = _accept(best, dcur)
                if acc.any():
                    rows = acc.nonzero().squeeze(1)
                    a_new = a_cand[rows, j[rows]]
                    st.sc[rows, k] = j[rows]
                    st.se[rows, c0:c0 + scale_group] = a_new[:, None]
                    u = a_new[:, None] * z[rows]
                    d = u - v[rows]
                    st.vals[rows, c0:c0 + scale_group] = u
                    HE.index_add_(0, rows, d @ H[c0:c0 + scale_group, :])
                    moved["scale"] += rows.numel()

            # ---- stage D: f16 super-block scale, closed form -----------------------------------
            for k in torch.randperm(nsup, generator=gen).tolist():
                c0 = 256 * k
                Hkk = H[c0:c0 + 256, c0:c0 + 256]
                v = st.vals[:, c0:c0 + 256]
                fac = torch.repeat_interleave(codec.scale_factor(st.sc[:, scale_groups_per_super * k:scale_groups_per_super * (k + 1)]), scale_group, dim=1)
                z = fac * mags(c0, 256) * st.sgn[:, c0:c0 + 256]              # (R, 256): vals = d * z
                q = v @ Hkk - HE[:, c0:c0 + 256]
                A = ((z @ Hkk) * z).sum(1)
                B = (z * q).sum(1)
                d_cur = st.d[:, k]
                d_new = torch.where(A > 0, B / A.clamp(min=1e-30), d_cur).half().float()
                dcur = d_cur * d_cur * A - 2 * d_cur * B
                dnew = d_new * d_new * A - 2 * d_new * B
                acc = _accept(dnew, dcur) & torch.isfinite(d_new)
                if acc.any():
                    rows = acc.nonzero().squeeze(1)
                    st.d[rows, k] = d_new[rows]
                    se_new = torch.repeat_interleave(d_new[rows, None] * codec.scale_factor(st.sc[rows, scale_groups_per_super * k:scale_groups_per_super * (k + 1)]), scale_group, dim=1)
                    st.se[rows, c0:c0 + 256] = se_new
                    u = d_new[rows, None] * z[rows]
                    d = u - v[rows]
                    st.vals[rows, c0:c0 + 256] = u
                    HE.index_add_(0, rows, d @ H[c0:c0 + 256, :])
                    moved["d"] += rows.numel()

        stats["sweeps"] = sweep + 1
        for a, b in (("code_moves", "code"), ("sign_moves", "sign"), ("scale_moves", "scale"), ("d_moves", "d")):
            stats[a] += moved[b]
        l_now = h_loss()
        stats["per_sweep"].append(dict(loss=l_now, seconds=time.time() - t0, **moved))
        if log:
            log(f"      sweep {sweep + 1}: loss {l_now:.6e}  moves code {moved['code']} sign {moved['sign']} "
                f"scale {moved['scale']} d {moved['d']}  {time.time() - t0:.1f}s")
        if sum(moved.values()) == 0:
            break
    stats["loss_start"] = l_start
    stats["loss_end"] = h_loss()
    stats["moves"] = stats["code_moves"] + stats["sign_moves"] + stats["scale_moves"] + stats["d_moves"]
    return stats

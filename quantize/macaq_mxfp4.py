"""Dense W4A4 quantization with OCP MXFP4 weights and activations.

Applies activation-aware channel balancing, a block-Hadamard rotation,
Hessian-weighted exponent selection and greedy code refinement. Weights and
activations share the OCP convention: value = code * 2**(e8 - 127).
"""
import os
# HF cache honours the standard env vars (HF_HUB_CACHE / HF_HOME); leave unset
# to use the default ~/.cache/huggingface.

import time
import torch
import torch.nn as nn
from tqdm import tqdm
from macaq_balance import balance_pre_scale_from_diag

from mxfp4_helper import (
    CACHE_DIR, make_hadamard, print_model_memory,
    evaluate_wikitext2,
)

# Standalone dense kernel shared with the serving integration.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from macaq_vllm_mxfp4_dense import (mxfp4_w4a4_linear, dequantize_mxfp4 as _ocp_dequant,
                                   rotate_blocks)

MXFP4_BLOCK = 32


# ── OCP dense-W4A4 eval module ───────────────────────────────────────────────
class OCPDenseW4A4Linear(nn.Module):
    """Holds OCP MXFP4 weights, with W4A4 or weight-only BF16 forward.
    Exposes the attribute names
    save_macaq_vllm.save_macaq_checkpoint reads (W_packed, W_scales, B_inv_T,
    pre_scale, bias, out/in_features)."""

    def __init__(self, W_packed, W_scales, BH, bias, block_size,
                 out_features, in_features, pre_scale=None, activation_precision="w4a4"):
        super().__init__()
        self.out_features, self.in_features = out_features, in_features
        self.block_size = block_size
        if activation_precision not in ("w4a4", "w4a16"):
            raise ValueError("activation precision must be w4a4 or w4a16")
        self.activation_precision = activation_precision
        # pack returns padded e8m0 scales; the kernel and the saver both want the
        # unpadded (out, in//block) e8m0 bytes.
        ws = W_scales.view(torch.uint8)[:out_features, : in_features // block_size]
        self.register_buffer("W_packed", W_packed.contiguous())
        self.register_buffer("W_scales", ws.contiguous())
        self.register_buffer("B_inv_T", BH.bfloat16().contiguous())
        self.register_buffer("pre_scale",
                             None if pre_scale is None else pre_scale.bfloat16().contiguous())
        self.register_buffer("bias",
                             None if bias is None else bias.bfloat16().contiguous())

    def forward(self, x):
        if self.activation_precision == "w4a16":
            from macaq_vllm_mxfp4_dense import mxfp4_w4a16_linear
            return mxfp4_w4a16_linear(
                x, self.W_packed, self.W_scales, pre_scale=self.pre_scale,
                BH=self.B_inv_T, bias=self.bias).to(x.dtype)
        out = mxfp4_w4a4_linear(
            x.to(torch.bfloat16), self.W_packed, self.W_scales,
            pre_scale=self.pre_scale, BH=self.B_inv_T, bias=self.bias,
            out_scale=1.0)
        return out.to(x.dtype)        # match input dtype (fp16 model) like RealMXFP4Linear

# ── Signed E2M1 grid (15 distinct levels; 0 shared) ──────────────────────────
_SG = torch.tensor([-6., -4., -3., -2., -1.5, -1., -0.5,
                    0., 0.5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
_MAG_LUT  = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8)
_SIGN_LUT = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint8)


def _device_const(t, device):
    return t.to(device) if t.device != torch.device(device) else t


# ── Fixed block-Hadamard rotation ────────────────────────────────────────────
_BH_CACHE: dict = {}


def get_bh(b, device):
    key = (b, str(device))
    H = _BH_CACHE.get(key)
    if H is None:
        H = make_hadamard(b, device=device).bfloat16()
        _BH_CACHE[key] = H
    return H


def balance_pre_scale(W, H, n_tokens, alpha=0.75):
    """MACAQ D1, the activation-aware pre-rotation diagonal:

        p_j = w_std_j^α / act_rms_j^(1−α),   act_rms_j = sqrt(H_jj / T)

    Free, since diag(H) is already accumulated. Folds into x (·p) and W (/p)
    before the rotation and cancels in the GEMM. The same definition
    is used by calibration and inference."""
    return balance_pre_scale_from_diag(W, H.diagonal(), n_tokens, alpha)


# ── 2. MXFP4 RTN → signed-level codes (now with an exponent offset) ──────────
@torch.no_grad()
def mxfp4_rtn_signed_index(M_W, block_size, exp_offset=0):
    """RTN in the OCP MXFP4 convention , IDENTICAL to the activation quantizer
    (macaq_vllm_mxfp4_dense.quantize_mxfp4): dequant value = code·2^(e8−127), so
    the saved weights load natively (kernel out_scale=1.0).

    Per-block exponent = floor(log2(amax)) − 2 + exp_offset.  The −2 is the
    exponent of the largest E2M1 magnitude (6.0 = 1.5·2²), per the OCP spec.
    e8m0 stores only powers of two, so the base (2^(floor−2)=2^floor/4) is finer
    but CLIPS block-maxima with amax∈(1.5,2)·2^floor; offset +1 (2^floor/2) is
    coarser but never clips.  The exponent screen brackets both (default offsets
    1,0,−1 → floor−1/−2/−3) and picks the H-best per block."""
    out, inp = M_W.shape
    nblk = inp // block_size
    M_g = M_W.float().view(out, nblk, block_size)
    amax = M_g.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    log2_amax = torch.floor(torch.log2(amax)) - 2.0 + float(exp_offset)
    scale = torch.exp2(log2_amax)
    abs_s = (M_g / scale).abs()
    code = torch.where(abs_s > 5.0, 7,
           torch.where(abs_s > 3.5, 6,
           torch.where(abs_s > 2.5, 5,
           torch.where(abs_s > 1.75, 4,
           torch.where(abs_s > 1.25, 3,
           torch.where(abs_s > 0.75, 2,
           torch.where(abs_s > 0.25, 1,
           torch.zeros_like(abs_s, dtype=torch.long)))))))).long()
    neg = (M_g < 0)
    sidx = torch.where(code == 0, torch.full_like(code, 7),
                       torch.where(neg, 7 - code, 7 + code)).view(out, inp)
    scale_elem = scale.expand(out, nblk, block_size).reshape(out, inp).contiguous()
    return sidx, log2_amax.squeeze(-1), scale_elem


# ── (v2.4) H-weighted per-(row,block) exponent screen ─────────────────────────
@torch.no_grad()
def screen_block_exponents(M_W, Hd, block_size, offsets=(1, 0, -1)):
    """RTN every (row, block) at each exponent offset and keep the one with the
    lowest diag(Hf)-weighted SE:  err[r,blk] = Σ_{j∈blk} Hd_j (M_W − deq)²_{rj}.
    The diagonal metric makes the screen separable per block (the flip handles
    the off-diagonal coupling afterwards, with the chosen exponents frozen)."""
    out, inp = M_W.shape
    nblk = inp // block_size
    dev = M_W.device
    sg = _device_const(_SG, dev)
    Hd_b = Hd.float().view(1, nblk, block_size)

    best_err = best_sidx = best_l2a = best_sel = None
    for off in offsets:
        sidx, l2a, sel = mxfp4_rtn_signed_index(M_W, block_size, off)
        deq = sg[sidx] * sel
        err = ((M_W.float() - deq).pow(2).view(out, nblk, block_size) * Hd_b).sum(-1)
        if best_err is None:
            best_err, best_sidx, best_l2a, best_sel = err, sidx, l2a, sel
        else:
            better = err < best_err                                   # (out,nblk)
            be = better.unsqueeze(-1)
            best_err = torch.where(better, err, best_err)
            best_l2a = torch.where(better, l2a, best_l2a)
            best_sidx = torch.where(be, sidx.view(out, nblk, block_size),
                                    best_sidx.view(out, nblk, block_size)).view(out, inp)
            best_sel = torch.where(be, sel.view(out, nblk, block_size),
                                   best_sel.view(out, nblk, block_size)).reshape(out, inp)
    return best_sidx.contiguous(), best_l2a, best_sel.contiguous()


@torch.no_grad()
def pack_mxfp4_from_signed_index(sidx, log2_amax, block_size):
    """Pack signed-grid indices + held exponents → (W_packed, W_scales).
    log2_amax may include screen offsets; no abs_max re-derivation, so neither
    the screen choice nor flip moves can shift a block's exponent afterwards."""
    out, inp = sidx.shape
    nblk = inp // block_size
    dev = sidx.device
    mag = _device_const(_MAG_LUT, dev)[sidx]
    sgn = _device_const(_SIGN_LUT, dev)[sidx]
    nibbles = (sgn.to(torch.uint8) << 3) | mag.to(torch.uint8)
    data_packed = (nibbles[:, 1::2] << 4) | nibbles[:, 0::2]
    biased_exp = (log2_amax + 127).clamp(0, 254).to(torch.int32)
    padded_rows = ((out + 127) // 128) * 128
    padded_cols = ((nblk + 3) // 4) * 4
    scales_padded = torch.zeros(padded_rows, padded_cols, dtype=torch.int32, device=dev)
    scales_padded[:out, :nblk] = biased_exp
    scales = scales_padded.to(torch.uint8).view(torch.float8_e8m0fnu)
    return data_packed, scales


# ── 3. Two-sided greedy flip-refinement (fixed H) , active-set masked ──────────
def _flip_sweep_fast(sidx, scale_elem, XTyfp, XTyq, active, sg, INF, Hd, H):
    """One sweep of the active-set flip-refinement (single iteration body).
    Modifies sidx and XTyq in place.  Returns new_active (rows that improved)
    or empty tensor if converged.  Extracted so torch.compile can fuse it."""
    dev = sidx.device
    na = active.shape[0]
    si = sidx[active]
    se = scale_elem[active]
    g = (XTyfp[:, active] - XTyq[:, active]).T
    up_ok = si < 14
    c_up = (sg[torch.where(up_ok, si + 1, si)] - sg[si]) * se
    dl_up = torch.where(up_ok, c_up * c_up * Hd - 2 * c_up * g, INF)
    dn_ok = si > 0
    c_dn = (sg[torch.where(dn_ok, si - 1, si)] - sg[si]) * se
    dl_dn = torch.where(dn_ok, c_dn * c_dn * Hd - 2 * c_dn * g, INF)
    up_better = dl_up <= dl_dn
    dl = torch.where(up_better, dl_up, dl_dn)
    best_dL, best_j = dl.min(dim=1)
    improving = best_dL < -1e-12
    if not improving.any():
        return torch.empty(0, dtype=torch.long, device=dev)
    lr = torch.arange(na, device=dev)
    up_better_sel = up_better[lr, best_j]
    sel_up = up_better_sel & improving
    sel_dn = (~up_better_sel) & improving
    c_sel = torch.where(up_better_sel, c_up[lr, best_j], c_dn[lr, best_j])
    sidx[active[sel_up], best_j[sel_up]] += 1
    sidx[active[sel_dn], best_j[sel_dn]] -= 1
    glob_imp = active[improving]
    XTyq[:, glob_imp] = (XTyq[:, glob_imp]
                         + H.index_select(1, best_j[improving])
                         * c_sel[improving].unsqueeze(0))
    return glob_imp


def _flip_sweep_dense(sidx, scale_elem, XTyfp, XTyq, rows, sg, INF, Hd, H):
    """One sweep of the dense flip-refinement (always all rows).
    Modifies sidx and XTyq in place.  Returns improving bool mask."""
    g = (XTyfp - XTyq).T
    up_ok = sidx < 14
    c_up = (sg[torch.where(up_ok, sidx + 1, sidx)] - sg[sidx]) * scale_elem
    dl_up = torch.where(up_ok, c_up * c_up * Hd - 2 * c_up * g, INF)
    dn_ok = sidx > 0
    c_dn = (sg[torch.where(dn_ok, sidx - 1, sidx)] - sg[sidx]) * scale_elem
    dl_dn = torch.where(dn_ok, c_dn * c_dn * Hd - 2 * c_dn * g, INF)
    up_better = dl_up <= dl_dn
    dl = torch.where(up_better, dl_up, dl_dn)
    best_dL, best_j = dl.min(dim=1)
    improving = best_dL < -1e-12
    if not improving.any():
        return improving
    sel_up = up_better[rows, best_j] & improving
    sel_dn = (~up_better[rows, best_j]) & improving
    c_sel = torch.where(up_better[rows, best_j], c_up[rows, best_j], c_dn[rows, best_j])
    c_sel = torch.where(improving, c_sel, torch.zeros_like(c_sel))
    sidx[rows[sel_up], best_j[sel_up]] += 1
    sidx[rows[sel_dn], best_j[sel_dn]] -= 1
    XTyq.add_(H.index_select(1, best_j) * c_sel.unsqueeze(0))
    return improving


@torch.no_grad()
def flip_refine_levels(sidx, scale_elem, M_W_target, H, n_iters, fast=True,
                   compile_flip=False):
    """Greedy fixed-H refinement of the e2m1 level codes (sidx mutated in place).
    Δℓ = c²·H_ii − 2c·g, g=H·e, c=signed value change.

    fast=True (active-set masking): output rows are fully decoupled here ,
    XTyq = H @ W_hat.T has column r = H @ (row r of W_hat), and row r's gradient
    reads only that column, so the rank-1 update for row r touches nothing else.
    A row with no improving single-flip move therefore stays converged forever
    (its gradient never changes again).  We keep an `active` working set and drop
    such rows; this changes neither which flips are applied nor their order, so
    `sidx` is bit-identical to the dense sweep (fast=False) , only the per-sweep
    work shrinks as rows converge.

    compile_flip=True: wrap the per-sweep body with torch.compile(dynamic=True)
    to fuse the elementwise ops.  First call pays a compile cost; subsequent
    Linear layers of the same shape reuse the cached graph."""
    out, inp = sidx.shape
    dev = sidx.device
    sg = _device_const(_SG, dev)
    Hd = H.diagonal()
    INF = torch.tensor(float("inf"), device=dev)
    W_hat = sg[sidx] * scale_elem
    XTyfp = H @ M_W_target.T                        # (in, out)
    XTyq = H @ W_hat.T                              # (in, out)

    if compile_flip:
        if fast:
            sweep_fn = torch.compile(_flip_sweep_fast, dynamic=True)
            active = torch.arange(out, device=dev)
            for _ in range(n_iters):
                active = sweep_fn(sidx, scale_elem, XTyfp, XTyq, active,
                                  sg, INF, Hd, H)
                if active.numel() == 0:
                    break
            return
        else:
            sweep_fn = torch.compile(_flip_sweep_dense, dynamic=True)
            rows = torch.arange(out, device=dev)
            for _ in range(n_iters):
                improving = sweep_fn(sidx, scale_elem, XTyfp, XTyq, rows,
                                     sg, INF, Hd, H)
                if not improving.any():
                    break
            return

    if not fast:                                    # original dense sweep
        rows = torch.arange(out, device=dev)
        for _ in range(n_iters):
            g = (XTyfp - XTyq).T
            up_ok = sidx < 14
            c_up = (sg[torch.where(up_ok, sidx + 1, sidx)] - sg[sidx]) * scale_elem
            dl_up = torch.where(up_ok, c_up * c_up * Hd - 2 * c_up * g, INF)
            dn_ok = sidx > 0
            c_dn = (sg[torch.where(dn_ok, sidx - 1, sidx)] - sg[sidx]) * scale_elem
            dl_dn = torch.where(dn_ok, c_dn * c_dn * Hd - 2 * c_dn * g, INF)
            up_better = dl_up <= dl_dn
            dl = torch.where(up_better, dl_up, dl_dn)
            best_dL, best_j = dl.min(dim=1)
            improving = best_dL < -1e-12
            if not improving.any():
                break
            sel_up = up_better[rows, best_j] & improving
            sel_dn = (~up_better[rows, best_j]) & improving
            c_sel = torch.where(up_better[rows, best_j], c_up[rows, best_j], c_dn[rows, best_j])
            c_sel = torch.where(improving, c_sel, torch.zeros_like(c_sel))
            sidx[rows[sel_up], best_j[sel_up]] += 1
            sidx[rows[sel_dn], best_j[sel_dn]] -= 1
            XTyq = XTyq + H.index_select(1, best_j) * c_sel.unsqueeze(0)
        return

    active = torch.arange(out, device=dev)          # global indices of live rows
    for _ in range(n_iters):
        si = sidx[active]                           # (na, in)
        se = scale_elem[active]
        g = (XTyfp[:, active] - XTyq[:, active]).T
        up_ok = si < 14
        c_up = (sg[torch.where(up_ok, si + 1, si)] - sg[si]) * se
        dl_up = torch.where(up_ok, c_up * c_up * Hd - 2 * c_up * g, INF)
        dn_ok = si > 0
        c_dn = (sg[torch.where(dn_ok, si - 1, si)] - sg[si]) * se
        dl_dn = torch.where(dn_ok, c_dn * c_dn * Hd - 2 * c_dn * g, INF)
        up_better = dl_up <= dl_dn
        dl = torch.where(up_better, dl_up, dl_dn)
        best_dL, best_j = dl.min(dim=1)             # (na,)
        improving = best_dL < -1e-12
        if not improving.any():
            break
        lr = torch.arange(active.shape[0], device=dev)
        up_better_sel = up_better[lr, best_j]
        sel_up = up_better_sel & improving
        sel_dn = (~up_better_sel) & improving
        c_sel = torch.where(up_better_sel, c_up[lr, best_j], c_dn[lr, best_j])
        sidx[active[sel_up], best_j[sel_up]] += 1
        sidx[active[sel_dn], best_j[sel_dn]] -= 1
        glob_imp = active[improving]                # rows that moved this sweep
        XTyq[:, glob_imp] = (XTyq[:, glob_imp]
                             + H.index_select(1, best_j[improving])
                             * c_sel[improving].unsqueeze(0))
        active = glob_imp                           # drop converged rows forever


# ── Per-Linear real-quant (D1 → BH → exp screen → flip) ──────────────────────
@torch.no_grad()
def mxfp4_realquant_h(W, H, n_tokens, had_block,
                      n_flip_iters, ridge, do_flip=True,
                      pre_mode="balance", pre_alpha=0.5,
                      exp_screen=True, exp_offsets=(1, 0, -1),
                      return_ref=False, fast_flip=True,
                      compile_flip=False, pre_scale_override=None):
    """W [out,in] fp, H [in,in]=ΣXᵀX fp32 over n_tokens.
    Returns the artifacts dict for an OCPDenseW4A4Linear.

    `pre_scale_override` [in] (optional): use this fixed pre_scale p instead of
    deriving it per-Linear from w_std/act_rms , supply it here and the function
    still does W/p, H·p·pᵀ, the rotation, screen and flip in the scaled+rotated
    frame. (General hook; the MoE and dense paths both derive their own balance
    pre_scale per Linear and do not pass this.)"""
    dev = W.device
    out, in_f = W.shape
    BH = get_bh(had_block, dev)

    # 0. (v2.2) pre-rotation diagonal D1 (= pre_scale p), activation-aware:
    #    p = w_std^α / act_rms^(1−α),  act_rms_j = sqrt(H_jj / T)  (free).
    #    α=0.5: symmetric geometric balance (hot activation channels shrunk,
    #    heavy weight columns flattened, in equal measure). α=1.0: weight-std only.
    #    Folds into x (·p) and W (/p) before the rotation, cancels in the GEMM.
    #    A caller-supplied `pre_scale_override` wins.
    p = None
    if pre_scale_override is not None:
        p = pre_scale_override.to(dev).float().clamp(1e-4, 1e4)
        W = W / p.unsqueeze(0)
        H = H * p.unsqueeze(0) * p.unsqueeze(1)
    elif pre_mode != "none":
        p = balance_pre_scale(W, H, n_tokens, alpha=pre_alpha)
        W = W / p.unsqueeze(0)
        H = H * p.unsqueeze(0) * p.unsqueeze(1)

    # Rotated frame.
    BHf = BH.float()
    Wr = rotate_blocks(W.float(), BHf)
    Hr = rotate_blocks(rotate_blocks(H.float(), BHf).T, BHf).T          # BH·H·BH

    # GEMM-faithful metric for the screen + flip: the rotated Hessian Hr, plus a
    # Uniform ridge on the diagonal.
    Hf = None
    if do_flip or exp_screen:
        Hf = Hr.clone()
        if ridge > 0:
            Hf.diagonal().add_(ridge * Hf.diagonal().mean())

    # 2. RTN in the rotated frame , with the exponent screen (v2.4).
    M_W = Wr
    if exp_screen and Hf is not None and len(exp_offsets) > 1:
        sidx, log2_amax, scale_elem = screen_block_exponents(
            M_W, Hf.diagonal(), MXFP4_BLOCK, exp_offsets)
    else:
        sidx, log2_amax, scale_elem = mxfp4_rtn_signed_index(M_W, MXFP4_BLOCK)

    # 3. greedy flip in the (ridge-regularized) H-metric, exponents frozen.
    if do_flip:
        flip_refine_levels(sidx, scale_elem, M_W, Hf, n_flip_iters, fast=fast_flip,
                       compile_flip=compile_flip)

    # 4. assemble
    W_packed, W_scales = pack_mxfp4_from_signed_index(sidx, log2_amax, MXFP4_BLOCK)
    art = dict(W_packed=W_packed, W_scales=W_scales, BH=BH)
    if p is not None:
        art["pre_scale"] = p.bfloat16().contiguous()
    if return_ref:
        # exact roundtrip reference: dequant of the FINAL codes (post screen+flip)
        sg = _device_const(_SG, dev)
        art["deq_ref"] = (sg[sidx] * scale_elem).contiguous()
    return art


# ── Streaming per-block quantization (CPU-resident model, GPU hidden) ─────────
def _linears_in_block(block):
    return [(n, m) for n, m in block.named_modules() if isinstance(m, nn.Linear)]


def _set_module(root, name, new):
    parent = root
    for p in name.split(".")[:-1]:
        parent = getattr(parent, p)
    setattr(parent, name.split(".")[-1], new)


def _capture_hooks(block, dev):
    """Hooks that accumulate Hᵢ = Σ XᵀX per Linear , one forward pass;
    shared-input Linears share the XᵀX via the data_ptr cache."""
    H, handles = {}, []
    xtx_cache: dict = {}
    for name, m in _linears_in_block(block):
        din = m.weight.shape[1]
        H[name] = torch.zeros(din, din, dtype=torch.float32, device=dev)

        def _hook(_mod, inp, _out, n=name):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            key = x.data_ptr()
            cached = xtx_cache.get(key)
            if cached is None:
                xf = x.float()
                xtx = xf.T @ xf
                xtx_cache[key] = (xtx, x)
            else:
                xtx = cached[0]
            H[n].add_(xtx)
        handles.append(m.register_forward_hook(_hook))
    return H, handles, xtx_cache


def streaming_quantize(model, tokenizer, calib_text, had_block,
                       n_calib_tokens=100_000, chunk_tokens=2048,
                       n_flip_iters=256, ridge=0.0, do_flip=True,
                       pre_mode="balance", pre_alpha=0.5,
                       exp_screen=True, exp_offsets=(1, 0, -1),
                       check_equiv=True, fast_flip=True,
                       compile_flip=False, activation_precision="w4a4"):
    """Quantize every transformer block sequentially, replacing each Linear with a
    OCPDenseW4A4Linear. Each block is calibrated on the already-quantized upstream
    output (block-level error propagation, as in v1)."""
    dev = "cuda"
    model.cpu()

    ids = tokenizer(calib_text, return_tensors="pt", truncation=True,
                    max_length=n_calib_tokens).input_ids.to(dev)
    T = ids.shape[1]
    print(f"  calibration: {T} tokens ({(T + chunk_tokens - 1)//chunk_tokens} chunks)")

    torch.cuda.synchronize(); t0 = time.perf_counter()
    embed = model.model.embed_tokens.to(dev)
    H_dim = embed.weight.shape[1]
    hidden = torch.empty(1, T, H_dim, dtype=torch.float16, device=dev)
    with torch.no_grad():
        for i in range(0, T, chunk_tokens):
            hidden[:, i:i + chunk_tokens] = embed(ids[:, i:i + chunk_tokens])
    embed.cpu(); del ids; torch.cuda.empty_cache()

    rotary = model.model.rotary_emb.to(dev)
    pos_full = torch.arange(T, device=dev).unsqueeze(0)
    pos_embs = []
    with torch.no_grad():
        for i in range(0, T, chunk_tokens):
            j = min(i + chunk_tokens, T)
            pos_embs.append(rotary(hidden[:, i:j], pos_full[:, i:j]))
    rotary.cpu()
    torch.cuda.synchronize()
    print(f"  [embed + rotary setup: {time.perf_counter() - t0:.1f}s]")

    def run_block_chunked(block, hidden):
        with torch.no_grad():
            for k, i in enumerate(range(0, T, chunk_tokens)):
                j = min(i + chunk_tokens, T)
                hidden[:, i:j] = block(hidden[:, i:j],
                                       position_embeddings=pos_embs[k],
                                       position_ids=pos_full[:, i:j])[0]

    layers = model.model.layers
    t_calib = t_quant = t_fwd = 0.0
    did_equiv = False
    total_q = 0
    pbar = tqdm(range(len(layers)), desc="blocks")
    for li in pbar:
        block = layers[li].to(dev)

        # (a) Calibration pass.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        H, handles, xtx_cache = _capture_hooks(block, dev)
        with torch.no_grad():
            for k, i in enumerate(range(0, T, chunk_tokens)):
                j = min(i + chunk_tokens, T)
                xtx_cache.clear()
                _ = block(hidden[:, i:j],
                          position_embeddings=pos_embs[k],
                          position_ids=pos_full[:, i:j])[0]
        xtx_cache.clear()
        for h in handles:
            h.remove()
        torch.cuda.synchronize(); t_calib += time.perf_counter() - t0

        # (b) Quantize every Linear → OCPDenseW4A4Linear.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for name, m in _linears_in_block(block):
            if m.in_features % MXFP4_BLOCK != 0 or m.in_features % had_block != 0:
                H[name] = None
                continue
            want_ref = check_equiv and not did_equiv
            art = mxfp4_realquant_h(m.weight.data.float(), H[name], n_tokens=T,
                                    had_block=had_block, n_flip_iters=n_flip_iters,
                                    ridge=ridge, do_flip=do_flip,
                                    pre_mode=pre_mode, pre_alpha=pre_alpha,
                                    exp_screen=exp_screen, exp_offsets=exp_offsets,
                                    return_ref=want_ref, fast_flip=fast_flip,
                                    compile_flip=compile_flip)
            bias = m.bias.data.clone() if m.bias is not None else None
            wq = OCPDenseW4A4Linear(art["W_packed"], art["W_scales"], art["BH"], bias,
                                    MXFP4_BLOCK, m.out_features, m.in_features,
                                    pre_scale=art.get("pre_scale"),
                                    activation_precision=activation_precision).to(dev)
            if want_ref:
                _verify_equiv(wq, art["deq_ref"], li, name)
                did_equiv = True
            _set_module(block, name, wq)
            H[name] = None
            total_q += 1
        del H
        torch.cuda.empty_cache()
        torch.cuda.synchronize(); t_quant += time.perf_counter() - t0

        # (c) Re-forward through the now-quantized block.
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_block_chunked(block, hidden)
        torch.cuda.synchronize(); t_fwd += time.perf_counter() - t0

        block.cpu(); torch.cuda.empty_cache()
        pbar.set_postfix(calib=f"{t_calib:.0f}s", quant=f"{t_quant:.0f}s",
                         fwd=f"{t_fwd:.0f}s", q=total_q)
    print(f"  quantized {total_q} linear(s) | "
          f"[calib {t_calib:.1f}s | quant {t_quant:.1f}s | re-forward {t_fwd:.1f}s]")


@torch.no_grad()
def _verify_equiv(wq, deq_ref, li, name):
    """Exact pack/unpack roundtrip check. `deq_ref` is the OCP dequant of the
    FINAL codes (post screen + flip): value = code·2^(e8−127).  Unpacking the
    stored bytes through the OCP dequantizer must reproduce it to fp rounding."""
    dev = deq_ref.device
    M_Wq = _ocp_dequant(wq.W_packed, wq.W_scales, wq.in_features).float()
    rel = (M_Wq - deq_ref).abs().max() / deq_ref.abs().max().clamp_min(1e-6)
    y = wq(torch.randn(4, wq.in_features, device=dev, dtype=torch.bfloat16))
    print(f"  [equiv block {li}/{name}] roundtrip rel {rel:.2e} (expect ~0) | "
          f"forward ok {tuple(y.shape)}")


# ── Memory accounting ────────────────────────────────────────────────────────
def model_resident_bytes(model):
    b = 0
    for p in model.parameters():
        b += p.numel() * p.element_size()
    for buf in model.buffers():
        b += buf.numel() * buf.element_size()
    return b


def parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description="Streaming macaq MXFP4 dense real-quant "
                    "(D1 balance → BH → exp screen → greedy flip; OCP convention).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B",
                   help="HF model id or local path (the default is small, "
                        "so `python macaq_mxfp4.py` alone is a working smoke test)")
    p.add_argument("--n_calib_tokens", type=int, default=2_000_000) #2_000_000
    p.add_argument('--revision', help='source checkpoint revision')
    p.add_argument('--trust-remote-code', action='store_true', help='allow reviewed custom model/tokenizer code; remote sources require an immutable revision')
    p.add_argument("--calib-file", type=str,
                   help="UTF-8 calibration text supplied by you; replaces WikiText train")
    p.add_argument("--skip-eval", action="store_true",
                   help="quantize and save without downloading or evaluating WikiText")
    p.add_argument("--chunk_tokens", type=int, default=2048)
    p.add_argument("--had_block", type=int, default=32,
                   help="block-Hadamard rotation size (power of 2, divides in_features)")
    # --- greedy flip ---
    p.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n_flip_iters", type=int, default=512, # 512
                   help="max greedy flip sweeps (early-stops at convergence)")
    p.add_argument("--fast_flip", action=argparse.BooleanOptionalAction, default=False,
                   help="active-set masking in the flip-refinement: drop rows once they "
                        "have no improving move (rows are decoupled, so this is "
                        "bit-identical to the dense sweep, just faster). "
                        "--no-fast_flip = original dense sweep.")
    p.add_argument("--compile_flip", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the per-sweep flip body (first use incurs "
                        "a compile cost; subsequent layers of the same shape reuse "
                        "the cached graph)")
    p.add_argument("--ridge", type=float, default=1.0,
                   help="uniform ridge added to the flip-Hessian diagonal "
                        "(λ·mean(diag)); 0 disables the ridge")
    # --- per-block exponent screen ---
    p.add_argument("--exp_screen", action=argparse.BooleanOptionalAction, default=True,
                   help="H-weighted per-(row,block) screen over exponent offsets "
                        "before the flip (clip-vs-resolution choice per block)")
    p.add_argument("--exp_offsets", type=str, default="1,0,-1",
                   help="comma-separated exponent offsets to screen (OCP base "
                        "floor−2; +1=floor−1 no-clip, 0=floor−2 OCP-exact, "
                        "−1=floor−3 fine , the screen picks the H-best per block)")
    # --- pre-rotation diagonal D1 (= pre_scale, free at inference) ---
    p.add_argument("--pre_mode", choices=["none", "balance"], default="balance",
                   help="'balance': p = w_std^α / act_rms^(1−α) with act_rms from "
                        "diag(H) , activation-aware smoothing folded into pre_scale.")
    p.add_argument("--pre_alpha", type=float, default=0.75,
                   help="D1 balance exponent α (0.5 = symmetric geometric mean, "
                        "1.0 = weight-std-only)")
    p.add_argument("--save_dir", default=None,
                   help="if set, write a vLLM-loadable dense checkpoint here "
                        "(per-Linear packed weights + scales + pre_scale + "
                        "shared block-Hadamard) right after quantization")
    p.add_argument("--activation-precision", choices=("w4a4", "w4a16"), default="w4a4",
                   help="activation precision used in calibration propagation and saved vLLM inference")
    p.add_argument("--check_equiv", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    args = parse_args()
    if args.save_dir and os.path.isdir(args.save_dir) and os.listdir(args.save_dir):
        raise ValueError("Checkpoint directory is not empty; choose a fresh save_dir")
    if args.calib_file:
        from pathlib import Path
        calib_text = Path(args.calib_file).read_text(encoding="utf-8")
        if not calib_text.strip():
            raise ValueError("calibration file is empty")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    exp_offsets = tuple(int(x) for x in args.exp_offsets.split(","))

    print(f"macaq-MXFP4 {args.activation_precision.upper()} dense (OCP) "
          f"(D1[{args.pre_mode},a={args.pre_alpha}] → BH "
          f"→ screen{exp_offsets if args.exp_screen else '=off'} → flip) on {args.model}\n"
          f"  had_block={args.had_block} flip={'on' if args.flip else 'off'} "
          f"ridge={args.ridge} compile_flip={'on' if args.compile_flip else 'off'} "
          f"tf32={'on' if args.tf32 else 'off'}")

    t0 = time.time()
    from model_loading import loading_options
    source_options = loading_options(args.model, args.revision, args.trust_remote_code)
    tok = AutoTokenizer.from_pretrained(args.model, cache_dir=CACHE_DIR, **source_options)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, cache_dir=CACHE_DIR, **source_options)
    model.eval()
    fp16_bytes = model_resident_bytes(model)

    if not args.calib_file:
        train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                             cache_dir=os.environ.get("HF_DATASETS_CACHE") or None)
        calib_text = "\n\n".join(t for t in train["text"] if t.strip())
    print(f"  loaded model + calib in {time.time() - t0:.1f}s")

    t0 = time.time()
    streaming_quantize(model, tok, calib_text, had_block=args.had_block,
                       n_calib_tokens=args.n_calib_tokens, chunk_tokens=args.chunk_tokens,
                       n_flip_iters=args.n_flip_iters, ridge=args.ridge,
                       do_flip=args.flip, pre_mode=args.pre_mode, pre_alpha=args.pre_alpha,
                       exp_screen=args.exp_screen, exp_offsets=exp_offsets,
                       check_equiv=args.check_equiv, fast_flip=args.fast_flip,
                       compile_flip=args.compile_flip, activation_precision=args.activation_precision)
    t_quant = time.time() - t0
    quant_bytes = model_resident_bytes(model)

    if args.save_dir:
        from save_macaq_vllm import save_macaq_checkpoint
        save_macaq_checkpoint(model, tok, args.save_dir, had_block=args.had_block,
                              source_model_name=args.model,
                              activation_precision=args.activation_precision)
        import hashlib
        import importlib.metadata
        import json
        from pathlib import Path
        metadata = {
            "arguments": vars(args),
            "calibration_source": args.calib_file or "Salesforce/wikitext:wikitext-2-raw-v1:train",
            "calibration_sha256": hashlib.sha256(calib_text.encode("utf-8")).hexdigest(),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "triton", "transformers", "safetensors")},
        }
        Path(args.save_dir, "macaq_run.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8")
        import sys
        from source_provenance import save_source_fingerprint
        source_path = Path(args.model)
        if not source_path.is_dir():
            from huggingface_hub import snapshot_download
            source_path = Path(snapshot_download(args.model, revision=getattr(model.config, '_commit_hash', None) or args.revision,
                cache_dir=CACHE_DIR, allow_patterns=['*.safetensors','*.json','*.txt','*.model','*.jinja']))
        save_source_fingerprint(source_path, args.save_dir)

    if args.skip_eval:
        GB = 1024 ** 3
        print(f"quantize {t_quant:.1f}s | resident {quant_bytes/GB:.3f} GB "
              f"({fp16_bytes/quant_bytes:.2f}x smaller); evaluation skipped")
        raise SystemExit(0)

    print("Loading full quantized model to GPU for eval...")
    model.cuda(); torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    print_model_memory(model, "macaq-MXFP4 W4A4 dense-simple v2")

    GB = 1024 ** 3
    tag = (f"  macaq-MXFP4-{MXFP4_BLOCK} W4A4 dense-simple v2 "
           f"(flip={'on' if args.flip else 'off'})")

    t0 = time.time()
    ppl = evaluate_wikitext2(model, tok, 2048, "cuda", batch_size=1)
    t_eval = time.time() - t0
    # Print the WikiText-2 result before the (slower) C4 eval starts.
    print(f"\n{'='*64}")
    print(tag)
    print(f"    WikiText-2 PPL: {ppl:.3f}  [eval {t_eval:.1f}s]")


    print(f"  quantize {t_quant:.1f}s")
    print(f"  resident: fp16 {fp16_bytes/GB:.3f} GB → {quant_bytes/GB:.3f} GB "
          f"({fp16_bytes/quant_bytes:.2f}x smaller)")
    print(f"{'='*64}")

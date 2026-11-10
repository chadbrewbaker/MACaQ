"""
triton_mxfp4_moe.py : grouped W4A4 MXFP4 MoE built on tl.dot_scaled.

Verified path: tl.dot_scaled(a,"e2m1", b,"e2m1") lowers to the native FP4
`mma.sync...kind::mxf4...ue8m0` op on the target GPU. Here we turn that into a
real grouped MoE:

  hidden (T,H) bf16
    -> quantize to MXFP4 (e2m1 packed + e8m0 per-32 scales)
    -> grouped GEMM1 (gate_up)  [token-sorted by expert, vLLM moe_align layout]
    -> SiLU(gate)*up  (plain, NO gpt-oss clamp : we own the epilogue)
    -> quantize intermediate to MXFP4
    -> grouped GEMM2 (down), multiply routed weights
    -> sum over top_k

Weights are the raw MXFP4 export: w13 (E, 2I, H/2) u8 + w13_s (E, 2I, H/32) u8,
w2 (E, H, I/2) u8 + w2_s (E, H, I/32) u8. Same nibble/e8m0 convention as the
activation quantizer below (low nibble = even K index; e8m0 byte = exp+127).
"""
import os
import torch
import triton
import triton.language as tl
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 128

_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ── MXFP4 quantization ──────────────────────────────────────────────────────
@triton.jit
def _quant_mxfp4_kernel(x_ptr, pk_ptr, sc_ptr, K,
                        stride_xr, stride_pr, stride_sr,
                        BLOCK_G: tl.constexpr):
    """One program = BLOCK_G scale-groups (of 32) of one row. Single pass over x:
    read 32 bf16/group, emit 16 packed bytes + 1 e8m0 byte. Round-to-nearest on
    the E2M1 grid via boundary sums (matches the torch argmin to within ties)."""
    row = tl.program_id(0)
    g0 = tl.program_id(1) * BLOCK_G
    gid = g0 + tl.arange(0, BLOCK_G)                         # (G,) group indices
    half = tl.arange(0, 16)                                  # 16 byte slots / group
    # even (low-nibble) and odd (high-nibble) element columns within each group
    col_lo = gid[:, None] * 32 + half[None, :] * 2          # (G,16)
    col_hi = col_lo + 1
    xl = tl.load(x_ptr + row * stride_xr + col_lo, col_lo < K, 0).to(tl.float32)
    xh = tl.load(x_ptr + row * stride_xr + col_hi, col_hi < K, 0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(xl), axis=1), tl.max(tl.abs(xh), axis=1))  # (G,)
    amax = tl.maximum(amax, 1e-20)
    e = tl.floor(tl.log2(amax)) - 2.0
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    scale = tl.exp2(e)[:, None]

    al = tl.abs(xl) / scale
    clo = ((al >= 0.25).to(tl.int32) + (al >= 0.75) + (al >= 1.25) + (al >= 1.75)
           + (al >= 2.5) + (al >= 3.5) + (al >= 5.0))
    nlo = ((xl < 0).to(tl.int32) << 3) | clo
    ah = tl.abs(xh) / scale
    chi = ((ah >= 0.25).to(tl.int32) + (ah >= 0.75) + (ah >= 1.25) + (ah >= 1.75)
           + (ah >= 2.5) + (ah >= 3.5) + (ah >= 5.0))
    nhi = ((xh < 0).to(tl.int32) << 3) | chi
    packed = (nlo | (nhi << 4)).to(tl.uint8)                # (G,16)

    pk_off = row * stride_pr + gid[:, None] * 16 + half[None, :]
    tl.store(pk_ptr + pk_off, packed, gid[:, None] < K // 32)
    sc_byte = (e + 127.0).to(tl.uint8)                      # (G,)
    tl.store(sc_ptr + row * stride_sr + gid, sc_byte, gid < K // 32)


def quantize_mxfp4(x: torch.Tensor):
    """x (T,K) -> (packed u8 (T,K//2), e8m0 u8 (T,K//32)). Triton single-pass."""
    T, K = x.shape
    assert K % 32 == 0
    Kg = K // 32
    packed = torch.empty(T, K // 2, dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty(T, Kg, dtype=torch.uint8, device=x.device)
    BLOCK_G = 16
    grid = (T, triton.cdiv(Kg, BLOCK_G))
    _quant_mxfp4_kernel[grid](
        x, packed, e8m0, K,
        x.stride(0), packed.stride(0), e8m0.stride(0),
        BLOCK_G=BLOCK_G,
    )
    return packed, e8m0


def dequantize_mxfp4(packed: torch.Tensor, e8m0: torch.Tensor, K: int) -> torch.Tensor:
    """Inverse, for reference checks. packed (T,K//2), e8m0 (T,K//32) -> (T,K)."""
    T = packed.shape[0]
    lo = (packed & 0xF)
    hi = (packed >> 4) & 0xF
    code = torch.stack([lo, hi], dim=-1).reshape(T, K).long()
    sign = torch.where((code >> 3) & 1 == 1, -1.0, 1.0)
    val = sign * _E2M1_MAG.to(packed.device)[code & 7]
    scale = torch.exp2((e8m0.float() - 127)).repeat_interleave(32, dim=1)
    return val * scale


# ── Per-expert fused (D1 · block-Hadamard · MXFP4 quantize) ──────────────────
# Each routed (token, expert) pair gets THIS expert's diagonal pre_scale and the
# shared 32×32 block-Hadamard folded in before quantization, exactly mirroring the
# dense kernel's rot_scale_quantize : but the pre_scale is gathered per expert
# (by topk_ids) and the output is written to the expanded (T*top_k, K) layout.
@triton.jit
def _moe_rot_scale_quant_kernel(
    x_ptr, ps_ptr, bh_ptr, tk_ptr, pk_ptr, sc_ptr, weights_ptr, clean_ids_ptr, clean_weights_ptr,
    K, top_k, num_experts,
    stride_xr, stride_psr, stride_pkr, stride_scr, stride_tkr,
    GATHER: tl.constexpr, CLEAN_ROUTES: tl.constexpr, BLOCK_G: tl.constexpr):
    """One program = BLOCK_G groups (of 32) of one EXPANDED row j∈[0,T*top_k).
    j → token = j//top_k, slot = j%top_k, expert e = topk_ids[token, slot].
    GATHER: source activation row is `token` (gate_up input, shape (T,K)); else it
    is `j` itself (the already-expanded intermediate, shape (T*top_k,K))."""
    j = tl.program_id(0)
    g0 = tl.program_id(1) * BLOCK_G
    tok = j // top_k
    slot = j - tok * top_k
    e = tl.load(tk_ptr + tok * stride_tkr + slot).to(tl.int64)
    valid_expert = (e >= 0) & (e < num_experts)
    if CLEAN_ROUTES:
        if g0 == 0:
            tl.store(clean_ids_ptr + j, tl.where(valid_expert, e, 0))
            probability = tl.load(weights_ptr + j, valid_expert, 0.)
            tl.store(clean_weights_ptr + j, probability)
    src = tok if GATHER else j
    gid = g0 + tl.arange(0, BLOCK_G)                         # (G,)
    cols = tl.arange(0, 32)
    off = gid[:, None] * 32 + cols[None, :]                  # (G,32)
    x = tl.load(x_ptr + src * stride_xr + off, (off < K) & valid_expert, 0).to(tl.float32)
    ps = tl.load(ps_ptr + e * stride_psr + off, (off < K) & valid_expert, 1).to(tl.float32)
    xs = (x * ps).to(tl.bfloat16)
    bh = tl.load(bh_ptr + cols[:, None] * 32 + cols[None, :]).to(tl.bfloat16)  # (32,32)
    r = tl.dot(xs, bh, out_dtype=tl.float32)                 # rotated (G,32)
    r = r.to(tl.bfloat16).to(tl.float32)                     # match rotate_blocks bf16
    rl, rh = tl.split(tl.reshape(r, (BLOCK_G, 16, 2)))       # even/odd cols (G,16)
    amax = tl.maximum(tl.max(tl.abs(rl), axis=1), tl.max(tl.abs(rh), axis=1))
    amax = tl.maximum(amax, 1e-20)
    ex = tl.floor(tl.log2(amax)) - 2.0
    ex = tl.minimum(tl.maximum(ex, -127.0), 127.0)
    scale = tl.exp2(ex)[:, None]
    al = tl.abs(rl) / scale
    clo = ((al >= 0.25).to(tl.int32) + (al >= 0.75) + (al >= 1.25) + (al >= 1.75)
           + (al >= 2.5) + (al >= 3.5) + (al >= 5.0))
    nlo = ((rl < 0).to(tl.int32) << 3) | clo
    ah = tl.abs(rh) / scale
    chi = ((ah >= 0.25).to(tl.int32) + (ah >= 0.75) + (ah >= 1.25) + (ah >= 1.75)
           + (ah >= 2.5) + (ah >= 3.5) + (ah >= 5.0))
    nhi = ((rh < 0).to(tl.int32) << 3) | chi
    packed = (nlo | (nhi << 4)).to(tl.uint8)                 # (G,16)
    half = tl.arange(0, 16)
    pk_off = j * stride_pkr + gid[:, None] * 16 + half[None, :]
    tl.store(pk_ptr + pk_off, packed, gid[:, None] < K // 32)
    tl.store(sc_ptr + j * stride_scr + gid, (ex + 127.0).to(tl.uint8), gid < K // 32)


def moe_rot_scale_quantize(x, pre_scale_E, bh, topk_ids, gather, route_weights=None):
    """Per-expert (x·pre_scale[e]) @ blkdiag(bh) → MXFP4, in the expanded
    (T*top_k, K) layout. pre_scale_E (E,K) bf16; bh (32,32); topk_ids (T,top_k).
    gather=True: x is (T,K) (source row = token); False: x is (T*top_k,K)."""
    T, top_k = topk_ids.shape
    K = x.shape[-1]
    assert K % 32 == 0 and bh.shape[0] == 32
    R, Kg = T * top_k, K // 32
    packed = torch.empty(R, K // 2, dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty(R, Kg, dtype=torch.uint8, device=x.device)
    BLOCK_G = 16
    grid = (R, triton.cdiv(Kg, BLOCK_G))
    x = x.contiguous()
    pre_scale_E = pre_scale_E.contiguous()
    topk_ids = topk_ids.to(torch.int32).contiguous()
    clean_ids = torch.empty_like(topk_ids) if route_weights is not None else topk_ids
    clean_weights = torch.empty_like(route_weights) if route_weights is not None else e8m0
    _moe_rot_scale_quant_kernel[grid](
        x, pre_scale_E, bh.contiguous(), topk_ids, packed, e8m0,
        route_weights if route_weights is not None else packed, clean_ids, clean_weights,
        K, top_k, pre_scale_E.shape[0],
        x.stride(0), pre_scale_E.stride(0), packed.stride(0), e8m0.stride(0),
        topk_ids.stride(0),
        GATHER=gather, CLEAN_ROUTES=route_weights is not None, BLOCK_G=BLOCK_G)
    if route_weights is not None:
        return packed, e8m0, clean_ids, clean_weights
    return packed, e8m0


@triton.jit
def _moe_mxfp4_gemm(
    a_ptr, b_ptr, asc_ptr, bsc_ptr, c_ptr, topk_weights_ptr,
    sorted_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, num_valid_tokens, top_k,
    stride_am, stride_be, stride_bn, stride_asm, stride_bse, stride_bsn,
    stride_cm, stride_cn,
    MUL_WEIGHTS: tl.constexpr, OUT_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    npp = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= npp:
        return
    offs_tid = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_token = tl.load(sorted_ids_ptr + offs_tid).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_e = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_e == -1:
        return

    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    offs_kb = tl.arange(0, BLOCK_K // 2)        # packed bytes along K
    offs_ks = tl.arange(0, BLOCK_K // 32)       # scale groups along K
    a_row = offs_token // top_k                  # source-token row in (T,*)
    a_ptrs = a_ptr + a_row[:, None] * stride_am + offs_kb[None, :]
    b_ptrs = b_ptr + off_e * stride_be + offs_n[:, None] * stride_bn + offs_kb[None, :]
    asc_ptrs = asc_ptr + a_row[:, None] * stride_asm + offs_ks[None, :]
    bsc_ptrs = bsc_ptr + off_e * stride_bse + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block in range(0, tl.cdiv(K, BLOCK_K)):
        valid_kb = block * (BLOCK_K // 2) + offs_kb < K // 2
        valid_ks = block * (BLOCK_K // 32) + offs_ks < K // 32
        a = tl.load(a_ptrs, mask=token_mask[:, None] & valid_kb[None, :], other=0)
        b = tl.load(b_ptrs, mask=(offs_n[:, None] < N) & valid_kb[None, :], other=0)
        asc = tl.load(asc_ptrs, mask=token_mask[:, None] & valid_ks[None, :], other=127)
        bsc = tl.load(bsc_ptrs, mask=(offs_n[:, None] < N) & valid_ks[None, :], other=127)
        acc = tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1", acc)
        a_ptrs += BLOCK_K // 2
        b_ptrs += BLOCK_K // 2
        asc_ptrs += BLOCK_K // 32
        bsc_ptrs += BLOCK_K // 32

    if MUL_WEIGHTS:
        w = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        acc = acc * w[:, None]

    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_n[None, :] * stride_cn
    val = acc.to(tl.bfloat16) if OUT_BF16 else acc      # bf16 epilogue (fast path)
    tl.store(c_ptrs, val, mask=token_mask[:, None] & (offs_n[None, :] < N))


def _grouped_gemm(a_pk, a_sc, b_pk, b_sc, sorted_ids, expert_ids, npp,
                  N, K, num_valid, top_k, out, topk_weights, mul_weights,
                  block_m=BLOCK_M, block_n=BLOCK_N, block_k=BLOCK_K,
                  num_warps=4, num_stages=3, out_bf16=False):
    grid = (triton.cdiv(sorted_ids.numel(), block_m) * triton.cdiv(N, block_n),)
    _moe_mxfp4_gemm[grid](
        a_pk, b_pk, a_sc, b_sc, out,
        topk_weights if topk_weights is not None else a_pk,  # dummy ptr if unused
        sorted_ids, expert_ids, npp,
        N, K, num_valid, top_k,
        a_pk.stride(0), b_pk.stride(0), b_pk.stride(1),
        a_sc.stride(0), b_sc.stride(0), b_sc.stride(1),
        out.stride(0), out.stride(1),
        MUL_WEIGHTS=mul_weights, OUT_BF16=out_bf16,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )


def fused_mxfp4_moe(hidden, w13, w13_s, w2, w2_s, topk_ids, topk_weights,
                    w13_pre, w2_pre, bh):
    """hidden (T,H) bf16; w13 (E,2I,H//2) u8, w13_s (E,2I,H//32) u8;
    w2 (E,H,I//2) u8, w2_s (E,H,I//32) u8. w13_pre (E,H) bf16, w2_pre (E,I) bf16
    are the per-expert MACAQ diagonals; bh is the 32×32 block-Hadamard. Each
    expert's (pre_scale·Hadamard) is folded into its routed activation inside the
    fused quantizer : no shared transform. Returns (T,H) bf16."""
    T, H = hidden.shape
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    E, twoI, _ = w13.shape
    I = twoI // 2
    top_k = topk_ids.shape[1]
    num_valid = topk_ids.numel()

    if (T <= 4 and hidden.dtype == torch.bfloat16 and os.environ.get('MACAQ_MOE_DECODE', '1') != '0' and
            os.environ.get('MACAQ_FAST', '1') not in ('', '0', 'false', 'False')):
        from triton_mxfp4_moe_decode import fused_moe_decode
        return fused_moe_decode(hidden, w13, w13_s, w2, w2_s, topk_ids, topk_weights,
                                w13_pre, w2_pre, bh)

    # Tuned configs for this GPU class. gemm1 & gemm2 share BLOCK_M
    # (= moe_align block size); they differ in N/warps/stages.
    if T <= 64:                       # decode: tiny M, minimize wasted tiles
        bm = 16
        c1 = dict(block_n=128, block_k=256, num_warps=4, num_stages=2)
        c2 = dict(block_n=256, block_k=128, num_warps=4, num_stages=2)
    else:                             # prefill: bigger tiles, more warps
        bm = 64
        c1 = dict(block_n=256, block_k=256, num_warps=8, num_stages=2)
        c2 = dict(block_n=128, block_k=256, num_warps=8, num_stages=3)

    # GEMM1: gate_up. A = per-expert (x·w13_pre[e])·BH, MXFP4-quantized into the
    # expanded (T*top_k, H) layout : so the activation is indexed 1:1 by the
    # expanded token id (a_row = offs_token, hence top_k=1 to the GEMM).
    # Default: bf16 epilogue (GEMM stores bf16, no fp32 intermediate). gemm1 feeds
    # SiLU then is re-quantized to MXFP4, so bf16 there is exact-equivalent; gemm2's
    # per-expert terms are bf16 but the top_k reduction stays fp32 below. Verified
    # E2E (WikiText-2 PPL unchanged). Set MACAQ_FAST=0 for the fp32 reference path.
    fast = os.environ.get("MACAQ_FAST", "1") not in ("", "0", "false", "False")
    odt = torch.bfloat16 if fast else torch.float32
    # Clean capture-padding routes in the activation transform, before sorting.
    a_pk, a_sc, topk_ids, topk_weights = moe_rot_scale_quantize(
        hidden, w13_pre, bh, topk_ids, gather=True, route_weights=topk_weights)
    sorted_ids, expert_ids, npp = moe_align_block_size(topk_ids, bm, E)
    out1 = torch.empty(T * top_k, twoI, device=hidden.device, dtype=odt)
    _grouped_gemm(a_pk, a_sc, w13, w13_s, sorted_ids, expert_ids, npp,
                  twoI, H, num_valid, 1, out1, None, False, block_m=bm,
                  out_bf16=fast, **c1)

    # SiLU(gate)*up : plain, no clamp.
    gate, up = out1[:, :I], out1[:, I:]
    inter = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16)

    # GEMM2: down. A = per-expert (inter·w2_pre[e])·BH, MXFP4-quantized (inter is
    # already expanded → gather=False). Multiply routed weights in the epilogue.
    b_pk, b_sc = moe_rot_scale_quantize(inter, w2_pre, bh, topk_ids, gather=False)
    out2 = torch.empty(T * top_k, H, device=hidden.device, dtype=odt)
    _grouped_gemm(b_pk, b_sc, w2, w2_s, sorted_ids, expert_ids, npp,
                  H, I, num_valid, 1, out2, topk_weights, True, block_m=bm,
                  out_bf16=fast, **c2)

    # top_k reduction in fp32 (works for fp32 or bf16 out2) → bf16 residual.
    return out2.view(T, top_k, H).sum(dim=1, dtype=torch.float32).to(torch.bfloat16)

"""triton_mxfp4_dense.py , dense W4A4 MXFP4 Linear for macaq-quantized checkpoints
(Qwen3-14B etc.). Same tl.dot_scaled native-mxf4 path as the MoE kernel, minus the
token-sort/expert indexing , just a tiled quantized GEMM.

The activation chain  M_A = (x·pre_scale)·BH  is MXFP4-quantized at runtime, and
M_A @ M_Wᵀ = x @ Wᵀ  (pre_scale cancels, BH orthogonal).

Convention: weights and activations are both OCP MXFP4 (value = code·2^(e8−127)),
so the GEMM needs no correction , out_scale = 1.0.

Forward:  x' = (x·pre)·BH  →  MXFP4-quantize  →  dot_scaled(x', W)  + bias
"""
import math
import torch
import triton
import triton.language as tl
from triton_mxfp4_moe import quantize_mxfp4, dequantize_mxfp4, _E2M1_MAG


def make_hadamard(b: int, device="cpu") -> torch.Tensor:
    """Normalized b×b Sylvester Hadamard (orthogonal)."""
    H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(b)]
                      for i in range(b)], dtype=torch.float32, device=device)
    return (H / math.sqrt(b))


def rotate_blocks(x: torch.Tensor, BH: torch.Tensor) -> torch.Tensor:
    """Right-multiply x by block-diagonal Hadamard on the last axis (x @ blkdiag(BH))."""
    b = BH.shape[0]
    lead, K = x.shape[:-1], x.shape[-1]
    return (x.reshape(*lead, K // b, b) @ BH.to(x.dtype)).reshape(*lead, K)


# ── Fused (pre_scale · block-Hadamard · MXFP4 quantize) , one launch ──────────
@triton.jit
def _rot_scale_quant_kernel(x_ptr, ps_ptr, bh_ptr, pk_ptr, sc_ptr, K,
                            stride_xr, stride_pr, stride_sr,
                            BLOCK_G: tl.constexpr,
                            BLOCKED: tl.constexpr, NCB: tl.constexpr):
    """One program = BLOCK_G groups (of 32) of one row. Loads the full 32-elt
    group, multiplies by pre_scale (per-channel), applies the 32x32 block
    Hadamard (x @ BH), then RTN-quantizes the rotated group to MXFP4 (16 packed
    bytes + 1 e8m0). Fuses what were 3 separate ops/launches + 2 global tensors.

    BLOCKED writes the e8m0 scales straight into the 128x4 swizzle the block-
    scaled tensor-core GEMMs want, instead of row-major. That saves a whole
    separate permute pass per forward; NCB is the number of 4-wide column
    blocks (ceil(K/32/4)) and is only read when BLOCKED."""
    row = tl.program_id(0)
    g0 = tl.program_id(1) * BLOCK_G
    gid = g0 + tl.arange(0, BLOCK_G)                         # (G,)
    cols = tl.arange(0, 32)                                  # within-group cols
    off = gid[:, None] * 32 + cols[None, :]                  # (G,32)
    x = tl.load(x_ptr + row * stride_xr + off, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
    ps = tl.load(ps_ptr + off, mask=gid[:, None] < K // 32, other=1).to(tl.float32)               # per-channel pre_scale
    xs = (x * ps).to(tl.bfloat16)                            # scaled (G,32)
    bh = tl.load(bh_ptr + cols[:, None] * 32 + cols[None, :]).to(tl.bfloat16)  # (32,32)
    r = tl.dot(xs, bh, out_dtype=tl.float32)                # rotated (G,32)
    r = r.to(tl.bfloat16).to(tl.float32)                    # match rotate_blocks' bf16 rounding
    # split into even (lo nibble) / odd (hi nibble) columns
    rl, rh = tl.split(tl.reshape(r, (BLOCK_G, 16, 2)))      # each (G,16)
    amax = tl.maximum(tl.max(tl.abs(rl), axis=1), tl.max(tl.abs(rh), axis=1))
    amax = tl.maximum(amax, 1e-20)
    e = tl.floor(tl.log2(amax)) - 2.0
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    scale = tl.exp2(e)[:, None]
    al = tl.abs(rl) / scale
    clo = ((al >= 0.25).to(tl.int32) + (al >= 0.75) + (al >= 1.25) + (al >= 1.75)
           + (al >= 2.5) + (al >= 3.5) + (al >= 5.0))
    nlo = ((rl < 0).to(tl.int32) << 3) | clo
    ah = tl.abs(rh) / scale
    chi = ((ah >= 0.25).to(tl.int32) + (ah >= 0.75) + (ah >= 1.25) + (ah >= 1.75)
           + (ah >= 2.5) + (ah >= 3.5) + (ah >= 5.0))
    nhi = ((rh < 0).to(tl.int32) << 3) | chi
    packed = (nlo | (nhi << 4)).to(tl.uint8)                # (G,16)
    half = tl.arange(0, 16)
    pk_off = row * stride_pr + gid[:, None] * 16 + half[None, :]
    tl.store(pk_ptr + pk_off, packed, mask=gid[:, None] < K // 32)
    sc = (e + 127.0).to(tl.uint8)
    if BLOCKED:
        # 128x4 tiles; within a tile the 128 rows land at (r%32)*16 + (r//32)*4.
        r = row % 128
        off_s = ((row // 128) * (512 * NCB) + (gid // 4) * 512
                 + (r % 32) * 16 + (r // 32) * 4 + (gid % 4))
        tl.store(sc_ptr + off_s, sc, mask=gid < K // 32)
    else:
        tl.store(sc_ptr + row * stride_sr + gid, sc, mask=gid < K // 32)


def blocked_scale_shape(rows: int, cols: int):
    """Shape of the 128x4-swizzled e8m0 buffer holding a (rows, cols) scale grid."""
    return ((rows + 127) // 128 * 128, (cols + 3) // 4 * 4)


def rot_scale_quantize_mxfp4(x, pre_scale, BH, blocked=False):
    """Fused (x·pre_scale) @ blkdiag(BH) -> MXFP4. BH must be 32x32 (= MXFP4 grp).
    x (T,K) bf16, pre_scale (K,), BH (32,32). Returns (packed u8, e8m0 u8).

    blocked=True emits the scales in the 128x4 swizzle (padded, see
    blocked_scale_shape) that the cuDNN/CUTLASS block-scaled GEMMs read; the
    default row-major layout is what the Triton `dot_scaled` kernel wants."""
    T, K = x.shape
    assert K % 32 == 0 and BH.shape[0] == 32
    Kg = K // 32
    packed = torch.empty(T, K // 2, dtype=torch.uint8, device=x.device)
    if blocked:
        pr, pc = blocked_scale_shape(T, Kg)
        if Kg % 4:
            e8m0 = torch.zeros(pr, pc, dtype=torch.uint8, device=x.device)
        else:
            e8m0 = torch.empty(pr, pc, dtype=torch.uint8, device=x.device)
            if T % 128:
                e8m0.view(-1)[(T // 128) * 512 * (pc // 4):] = 0
    else:
        e8m0 = torch.empty(T, Kg, dtype=torch.uint8, device=x.device)
    ps = (pre_scale if pre_scale is not None
          else torch.ones(K, device=x.device, dtype=x.dtype)).contiguous()
    BLOCK_G = 16
    grid = (T, triton.cdiv(Kg, BLOCK_G))
    _rot_scale_quant_kernel[grid](
        x, ps, BH.contiguous(), packed, e8m0, K,
        x.stride(0), packed.stride(0), e8m0.stride(0), BLOCK_G=BLOCK_G,
        BLOCKED=blocked, NCB=(e8m0.shape[1] // 4 if blocked else 0))
    return packed, e8m0


@triton.jit
def _dense_w4a4_kernel(
    a_ptr, b_ptr, asc_ptr, bsc_ptr, c_ptr, M, N, K,
    stride_am, stride_bn, stride_asm, stride_bsn, stride_cm,
    OUT_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_kb = tl.arange(0, BLOCK_K // 2)        # packed bytes along K
    offs_ks = tl.arange(0, BLOCK_K // 32)       # scale groups along K
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_kb[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_kb[None, :]
    asc_ptrs = asc_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :]
    bsc_ptrs = bsc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0)
        asc = tl.load(asc_ptrs, mask=m_mask[:, None], other=127)
        bsc = tl.load(bsc_ptrs, mask=n_mask[:, None], other=127)
        acc = tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1", acc)
        a_ptrs += BLOCK_K // 2
        b_ptrs += BLOCK_K // 2
        asc_ptrs += BLOCK_K // 32
        bsc_ptrs += BLOCK_K // 32

    acc = acc * OUT_SCALE
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


def mxfp4_w4a4_linear(x, W_packed, W_scales, pre_scale=None,
                      BH=None, bias=None, out_scale=1.0,
                      block_m=64, block_n=128, block_k=256,
                      num_warps=8, num_stages=3):
    """x (..., K) bf16; W_packed (N, K//2) u8, W_scales (N, K//32) e8m0 u8 (macaq).
    pre_scale (K,) bf16; BH (b,b). Returns (..., N) bf16 ≈ x @ W_origᵀ + bias."""
    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K)
    # Fused (pre_scale · block-Hadamard · MXFP4 quant) in one launch when BH is
    # 32x32 (the MXFP4 group). Else fall back to the separate ops.
    if BH is not None and BH.shape[0] == 32:
        a_pk, a_sc = rot_scale_quantize_mxfp4(x2.contiguous().to(torch.bfloat16),
                                              pre_scale, BH)
    else:
        if pre_scale is not None:
            x2 = x2 * pre_scale.to(x2.dtype)
        if BH is not None:
            x2 = rotate_blocks(x2, BH)
        a_pk, a_sc = quantize_mxfp4(x2.contiguous())     # OCP activation quant
    M = x2.shape[0]
    N = W_packed.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
    _dense_w4a4_kernel[grid](
        a_pk, W_packed, a_sc, W_scales, out, M, N, K,
        a_pk.stride(0), W_packed.stride(0), a_sc.stride(0), W_scales.stride(0),
        out.stride(0),
        OUT_SCALE=out_scale,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        num_warps=num_warps, num_stages=num_stages,
    )
    out = out.to(torch.bfloat16)
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.reshape(*lead, N)


# ── Decode-tuned fast variant (bf16 epilogue + shape-aware tiles) ─────────────
# Numerically identical to mxfp4_w4a4_linear above , SAME single-pass activation
# transform/quant; only the GEMM epilogue (stores bf16, folds bias/out_scale) and
# the tile selection differ. This is the DEFAULT kernel used by the vLLM plugin;
# set MACAQ_FAST=0 to fall back to the reference mxfp4_w4a4_linear.
@triton.jit
def _dense_w4a4_bf16_kernel(
    a_ptr, b_ptr, asc_ptr, bsc_ptr, c_ptr, bias_ptr, M, N, K,
    stride_am, stride_bn, stride_asm, stride_bsn, stride_cm,
    OUT_SCALE: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Same native-FP4 `dot_scaled` GEMM as _dense_w4a4_kernel, but stores bf16 and
    folds bias/out_scale into the epilogue (no separate fp32 buffer + cast)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
    m_mask = offs_m < M
    n_mask = offs_n < N
    offs_kb = tl.arange(0, BLOCK_K // 2)
    offs_ks = tl.arange(0, BLOCK_K // 32)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_kb[None, :]
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_kb[None, :]
    asc_ptrs = asc_ptr + offs_m[:, None] * stride_asm + offs_ks[None, :]
    bsc_ptrs = bsc_ptr + offs_n[:, None] * stride_bsn + offs_ks[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
        b = tl.load(b_ptrs, mask=n_mask[:, None], other=0)
        asc = tl.load(asc_ptrs, mask=m_mask[:, None], other=127)
        bsc = tl.load(bsc_ptrs, mask=n_mask[:, None], other=127)
        acc = tl.dot_scaled(a, asc, "e2m1", b.T, bsc, "e2m1", acc)
        a_ptrs += BLOCK_K // 2
        b_ptrs += BLOCK_K // 2
        asc_ptrs += BLOCK_K // 32
        bsc_ptrs += BLOCK_K // 32

    acc = acc * OUT_SCALE
    if HAS_BIAS:
        acc = acc + tl.load(bias_ptr + offs_n, mask=n_mask,
                            other=0.0).to(tl.float32)[None, :]
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def _largest_bk(K: int, prefer):
    """Largest candidate BLOCK_K that divides K (kernel has no K-mask). K is a
    multiple of 32 (MXFP4 group), so 32 always divides → this never fails."""
    for bk in prefer:
        if K % bk == 0:
            return bk
    return 32


def _pick_tiles(M: int, K: int):
    """Decode-aware launch config. Small M → skinny tiles + many CTAs (HBM-bound);
    large M → the reference compute-bound tiles. BLOCK_K must divide K."""
    if M <= 64:                                   # decode / small-batch
        return dict(block_m=16, block_n=64, block_k=_largest_bk(K, (512, 256, 128, 64)),
                    num_warps=4, num_stages=3)
    return dict(block_m=64, block_n=128, block_k=_largest_bk(K, (256, 128, 64)),
                num_warps=8, num_stages=3)


def mxfp4_w4a4_linear_fast(x, W_packed, W_scales, pre_scale=None, BH=None,
                           bias=None, out_scale=1.0, tiles=None):
    """Decode-tuned MACaQ W4A4 linear. Same signature/semantics as the reference
    mxfp4_w4a4_linear; returns (..., N) bf16. Numerics identical (shared
    single-pass quant); only the GEMM epilogue + tile sizes differ. `tiles` may
    override the auto-selected launch config."""
    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K)
    if BH is not None and BH.shape[0] == 32:
        a_pk, a_sc = rot_scale_quantize_mxfp4(x2.contiguous().to(torch.bfloat16),
                                              pre_scale, BH)
    else:
        if pre_scale is not None:
            x2 = x2 * pre_scale.to(x2.dtype)
        if BH is not None:
            x2 = rotate_blocks(x2, BH)
        a_pk, a_sc = quantize_mxfp4(x2.contiguous())
    M, N = x2.shape[0], W_packed.shape[0]
    cfg = tiles or _pick_tiles(M, K)
    has_bias = bias is not None
    bias_t = bias.contiguous() if has_bias else W_scales      # dummy ptr when unused
    out = torch.empty(M, N, device=x.device, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, cfg["block_m"]), triton.cdiv(N, cfg["block_n"]))
    _dense_w4a4_bf16_kernel[grid](
        a_pk, W_packed, a_sc, W_scales, out, bias_t, M, N, K,
        a_pk.stride(0), W_packed.stride(0), a_sc.stride(0), W_scales.stride(0),
        out.stride(0),
        OUT_SCALE=out_scale, HAS_BIAS=has_bias,
        BLOCK_M=cfg["block_m"], BLOCK_N=cfg["block_n"], BLOCK_K=cfg["block_k"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return out.reshape(*lead, N)

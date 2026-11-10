"""macaq_vllm_mxfp4_dense.py , STANDALONE dense W4A4 MXFP4 Linear kernel built on
the native `tl.dot_scaled` FP4 MMA path. No imports from triton_mxfp4_moe.py or
the vLLM plugin (vllm-free), so it loads in any env; the macaq reference eval and
the vLLM plugin both use this same kernel.

────────────────────────────────────────────────────────────────────────────
FORWARD FORMULATION (macaq, true weight+activation MXFP4)

A macaq-quantized Linear reconstructs  y = x @ Wᵀ  via a transform pair that
cancels on the contraction axis:

    M_A = ( x · d1 ) · B            ← built & MXFP4-quantized at runtime
    M_W = ( W · d1⁻¹) · B           ← baked offline by the macaq save step

    y   = M_A @ M_Wᵀ

    M_A @ M_Wᵀ = x (d1 B)(Bᵀ d1⁻¹) Wᵀ = x Wᵀ      (B Bᵀ = I, d1·d1⁻¹ = I)

  d1 = pre_scale  (diagonal, per-input-channel balance)
  B  = block-diagonal Hadamard (32-wide blocks)
  Both M_A and M_W are quantized to MXFP4 , the transform flattens the two
  operands so 4-bit weights AND activations stay accurate.

CONVENTION:
  Weights and activations are both packed in OCP MXFP4 (value = code·2^(e8−127)),
  so the GEMM needs no correction: out_scale = 1.0.

Pipeline:  x' = (x·pre)·BH → MXFP4-quantize → dot_scaled(x', W) + bias
────────────────────────────────────────────────────────────────────────────
"""
import math
import torch
import triton
import triton.language as tl

# E2M1 magnitudes for the 8 unsigned codes (sign carried separately in bit 3).
_E2M1_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ── MXFP4 activation quantizer (OCP: value = code · 2^(e-127)) ────────────────
@triton.jit
def _quant_mxfp4_kernel(x_ptr, pk_ptr, sc_ptr, K,
                        stride_xr, stride_pr, stride_sr,
                        BLOCK_G: tl.constexpr):
    """One program = BLOCK_G scale-groups (of 32) of one row. Single pass over x:
    read 32 bf16/group, emit 16 packed bytes + 1 e8m0 byte. Round-to-nearest on
    the E2M1 grid via boundary sums (matches a torch argmin to within ties)."""
    row = tl.program_id(0)
    g0 = tl.program_id(1) * BLOCK_G
    gid = g0 + tl.arange(0, BLOCK_G)                         # (G,) group indices
    half = tl.arange(0, 16)                                  # 16 byte slots / group
    # even (low-nibble) and odd (high-nibble) element columns within each group
    col_lo = gid[:, None] * 32 + half[None, :] * 2          # (G,16)
    col_hi = col_lo + 1
    xl = tl.load(x_ptr + row * stride_xr + col_lo, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
    xh = tl.load(x_ptr + row * stride_xr + col_hi, mask=gid[:, None] < K // 32, other=0).to(tl.float32)
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
    tl.store(pk_ptr + pk_off, packed, mask=gid[:, None] < K // 32)
    sc_byte = (e + 127.0).to(tl.uint8)                      # (G,)
    tl.store(sc_ptr + row * stride_sr + gid, sc_byte, mask=gid < K // 32)


def quantize_mxfp4(x: torch.Tensor):
    """x (T,K) -> (packed u8 (T,K//2), e8m0 u8 (T,K//32)). Triton single-pass."""
    T, K = x.shape
    assert K % 32 == 0
    Kg = K // 32
    packed = torch.empty(T, K // 2, dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty(T, Kg, dtype=torch.uint8, device=x.device)
    BLOCK_G = min(triton.next_power_of_2(Kg), 16)
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


# ── Hadamard rotation ─────────────────────────────────────────────────────────
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
                            BLOCK_G: tl.constexpr):
    """One program = BLOCK_G groups (of 32) of one row. Loads the full 32-elt
    group, multiplies by pre_scale (per-channel), applies the 32x32 block
    Hadamard (x @ BH), then RTN-quantizes the rotated group to MXFP4 (16 packed
    bytes + 1 e8m0). Fuses what were 3 separate ops/launches + 2 global tensors."""
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
    tl.store(sc_ptr + row * stride_sr + gid, (e + 127.0).to(tl.uint8), mask=gid < K // 32)


def rot_scale_quantize_mxfp4(x, pre_scale, BH):
    """Fused (x·pre_scale) @ blkdiag(BH) -> MXFP4. BH must be 32x32 (= MXFP4 grp).
    x (T,K) bf16, pre_scale (K,), BH (32,32). Returns (packed u8, e8m0 u8)."""
    T, K = x.shape
    assert K % 32 == 0 and BH.shape[0] == 32
    Kg = K // 32
    packed = torch.empty(T, K // 2, dtype=torch.uint8, device=x.device)
    e8m0 = torch.empty(T, Kg, dtype=torch.uint8, device=x.device)
    ps = (pre_scale if pre_scale is not None
          else torch.ones(K, device=x.device, dtype=x.dtype)).contiguous()
    BLOCK_G = min(triton.next_power_of_2(Kg), 16)
    grid = (T, triton.cdiv(Kg, BLOCK_G))
    _rot_scale_quant_kernel[grid](
        x, ps, BH.contiguous(), packed, e8m0, K,
        x.stride(0), packed.stride(0), e8m0.stride(0), BLOCK_G=BLOCK_G)
    return packed, e8m0


# ── Dense W4A4 GEMM (native FP4 mxf4 via tl.dot_scaled) ───────────────────────
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


# ── W4A8: MXFP4 weights (4-bit in memory) x per-token FP8 activations ─────────────
# Dequantize weights to FP8, fuse activation scaling and rotation, then use
# a row-wise FP8 GEMM through torch._scaled_mm.
@triton.jit
def _dequant_w4_to_fp8_kernel(pk_ptr, sc_ptr, emax_ptr, out_ptr, K, BLOCK_G: tl.constexpr):
    """Row n of packed MXFP4 -> e4m3 bytes with ONE per-row scale 2^(emax_n-127-6):
    element = mag(code) * 2^(e_block - emax_n + 6). Exact for E2M1 values unless a
    block's exponent is > 11 below the row max (then it rounds/underflows)."""
    row = tl.program_id(0)
    gid = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    half = tl.arange(0, 16)
    Kg = K // 32
    m = (gid < Kg)[:, None]
    pk = tl.load(pk_ptr + row * (K // 2) + gid[:, None] * 16 + half[None, :], mask=m, other=0)
    e = tl.load(sc_ptr + row * Kg + gid, mask=gid < Kg, other=0).to(tl.float32)
    emax = tl.load(emax_ptr + row).to(tl.float32)
    lo = (pk & 0xF).to(tl.int32)
    hi = ((pk >> 4) & 0xF).to(tl.int32)
    mlo = lo & 7
    mhi = hi & 7
    maglo = tl.where(mlo < 4, mlo.to(tl.float32) * 0.5, tl.where(mlo == 4, 2.0, tl.where(mlo == 5, 3.0, tl.where(mlo == 6, 4.0, 6.0))))
    maghi = tl.where(mhi < 4, mhi.to(tl.float32) * 0.5, tl.where(mhi == 4, 2.0, tl.where(mhi == 5, 3.0, tl.where(mhi == 6, 4.0, 6.0))))
    f = tl.exp2(e - emax + 6.0)[:, None]
    vlo = tl.where((lo >> 3) == 1, -1.0, 1.0) * maglo * f
    vhi = tl.where((hi >> 3) == 1, -1.0, 1.0) * maghi * f
    base = out_ptr + row * K + gid[:, None] * 32 + half[None, :] * 2
    tl.store(base, vlo.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m)
    tl.store(base + 1, vhi.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m)


def dequant_w4_to_fp8(W_packed, W_scales, emax):
    """(N,K//2) u8 + (N,K//32) e8m0 u8 + per-row max exponent (N,) -> e4m3 (N,K).
    Pair with row scales 2^(emax-127-6) (see w8_row_scales)."""
    N, Kh = W_packed.shape
    K = Kh * 2
    out = torch.empty(N, K, dtype=torch.uint8, device=W_packed.device)
    BLOCK_G = 16
    _dequant_w4_to_fp8_kernel[(N, triton.cdiv(K // 32, BLOCK_G))](
        W_packed, W_scales, emax, out, K, BLOCK_G=BLOCK_G)
    return out.view(torch.float8_e4m3fn)


def _weight_only_ops():
    import sys
    from pathlib import Path
    directory = str(Path(__file__).resolve().parents[1] / "vllm")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import macaq_weight_only
    return macaq_weight_only


def dequant_w4_to_bf16(W_packed, W_scales):
    return _weight_only_ops().dequant_w4_to_bf16(W_packed, W_scales)


def mxfp4_w4a16_linear(x, W_packed, W_scales, pre_scale=None, BH=None, bias=None):
    return _weight_only_ops().mxfp4_w4a16_linear(
        x, W_packed, W_scales, pre_scale=pre_scale, BH=BH, bias=bias)


def w8_row_scales(W_scales):
    """Per-row fp32 scale (1,N) for the dequantized FP8 weight (+ the emax it uses)."""
    emax = W_scales.amax(dim=1).to(torch.int32).contiguous()
    return torch.exp2(emax.float() - 127.0 - 6.0).unsqueeze(0).contiguous(), emax


@triton.jit
def _rot_scale_quant_fp8_rows_kernel(x_ptr, ps_ptr, bh_ptr, q_ptr, sc_ptr, M, K,
                                     stride_xr, stride_qr,
                                     BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr, HAS_BH: tl.constexpr):
    """Per-token (row) FP8 of (x·p)·blkdiag(BH). Program = BLOCK_R rows x full K in
    BLOCK_C-column steps; the rotation is one tl.dot with the (BLOCK_C x BLOCK_C)
    block-diagonal Hadamard. Pass 1 tracks row amax, pass 2 quantizes (recompute,
    no intermediate tensor). Scale = amax/448 exactly as fp8_linear.quant_rows."""
    r0 = tl.program_id(0) * BLOCK_R
    rows = r0 + tl.arange(0, BLOCK_R)
    rmask = rows < M
    cols = tl.arange(0, BLOCK_C)
    if HAS_BH:
        bh = tl.load(bh_ptr + cols[:, None] * BLOCK_C + cols[None, :]).to(tl.bfloat16)
    amax = tl.zeros((BLOCK_R,), dtype=tl.float32)
    for c0 in range(0, K, BLOCK_C):
        off = c0 + cols
        x = tl.load(x_ptr + rows[:, None] * stride_xr + off[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        ps = tl.load(ps_ptr + off).to(tl.float32)
        xs = (x * ps[None, :]).to(tl.bfloat16)
        if HAS_BH:
            r = tl.dot(xs, bh, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        else:
            r = xs.to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(r), axis=1))
    sc = tl.maximum(amax, 1e-12) / 448.0
    for c0 in range(0, K, BLOCK_C):
        off = c0 + cols
        x = tl.load(x_ptr + rows[:, None] * stride_xr + off[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        ps = tl.load(ps_ptr + off).to(tl.float32)
        xs = (x * ps[None, :]).to(tl.bfloat16)
        if HAS_BH:
            r = tl.dot(xs, bh, out_dtype=tl.float32).to(tl.bfloat16).to(tl.float32)
        else:
            r = xs.to(tl.float32)
        q = (r / sc[:, None]).to(tl.float8e4nv)
        tl.store(q_ptr + rows[:, None] * stride_qr + off[None, :], q.to(tl.uint8, bitcast=True), mask=rmask[:, None])
    tl.store(sc_ptr + rows, sc, mask=rmask)


_BHBLK_CACHE: dict = {}


def _blockdiag_bh(BH, width):
    """Cache by live transform identity and mutation version, never by width alone."""
    import weakref
    try:
        version = BH._version
    except RuntimeError:
        # Inference tensors have no mutation counter, so rebuilding is required.
        return torch.block_diag(*([BH.bfloat16()] * (width // 32))).contiguous()
    key = (id(BH), width)
    signature = (version, BH.data_ptr(), BH.dtype, BH.device)
    cached = _BHBLK_CACHE.get(key)
    if cached is not None and cached[0]() is BH and cached[1] == signature:
        return cached[2]
    def expired(reference):
        if key in _BHBLK_CACHE and _BHBLK_CACHE[key][0] is reference:
            del _BHBLK_CACHE[key]
    result = torch.block_diag(*([BH.bfloat16()] * (width // 32))).contiguous()
    _BHBLK_CACHE[key] = (weakref.ref(BH, expired), signature, result)
    return result


def rot_scale_quantize_fp8_rows(x, pre_scale, BH, block_r=32, block_c=128, num_warps=8, num_stages=3):
    """x (T,K) bf16 -> (e4m3 (T,K), fp32 row scales (T,1)) of (x·pre_scale) @ blkdiag(BH)."""
    T, K = x.shape
    assert K % block_c == 0
    q = torch.empty(T, K, dtype=torch.uint8, device=x.device)
    sc = torch.empty(T, 1, dtype=torch.float32, device=x.device)
    ps = (pre_scale if pre_scale is not None else torch.ones(K, device=x.device, dtype=x.dtype)).contiguous()
    has_bh = BH is not None
    bh = _blockdiag_bh(BH, block_c) if has_bh else ps
    _rot_scale_quant_fp8_rows_kernel[(triton.cdiv(T, block_r),)](
        x, ps, bh, q, sc, T, K, x.stride(0), q.stride(0), BLOCK_R=block_r, BLOCK_C=block_c, HAS_BH=has_bh,
        num_warps=num_warps, num_stages=num_stages)
    return q.view(torch.float8_e4m3fn), sc


def mxfp4_w4a8_linear_fast(x, W_packed, W_scales, emax, w_scale, pre_scale=None, BH=None, bias=None):
    """W4A8 with 4-bit weights in memory: dequant -> FP8 (exact, per-row 2^k scale),
    per-token FP8 activations of x·p·B (one fused launch), cuBLAS FP8 GEMM. Same
    GEMM inputs as the unfused W4A8Linear reference, so its quality numbers hold."""
    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K).contiguous().to(torch.bfloat16)
    M = x2.shape[0]
    pad = (-M) % 16
    if pad:
        x2 = torch.nn.functional.pad(x2, (0, 0, 0, pad))
    xq, xs = rot_scale_quantize_fp8_rows(x2, pre_scale, BH)
    W8 = dequant_w4_to_fp8(W_packed, W_scales, emax)
    out = torch._scaled_mm(xq, W8.t(), scale_a=xs, scale_b=w_scale, bias=bias, out_dtype=torch.bfloat16)
    if pad:
        out = out[:M]
    return out.reshape(*lead, W_packed.shape[0])

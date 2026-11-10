"""Packed MACaQ MXFP4 weights with BF16 activations and GEMM.

Weights remain packed between calls. Each linear dequantizes a temporary BF16
matrix, applies D1 and block-Hadamard to BF16 inputs, then uses a BF16 GEMM.
This is a functional weight-only path, not a fused low-bit GEMM.
"""
import torch
import triton
import triton.language as tl


def rotate_blocks(x, bh):
    b = bh.shape[0]
    return (x.reshape(*x.shape[:-1], x.shape[-1] // b, b) @ bh.to(x.dtype)).reshape_as(x)


@triton.jit
def _dequant_w4_to_bf16_kernel(pk_ptr, sc_ptr, out_ptr, K, BLOCK_G: tl.constexpr):
    """Packed MXFP4 row -> bf16: element = mag(code) * 2^(e_block - 127). Exact
    (E2M1 has <= 2 significant bits). Uses only integer and floating-point operations."""
    row = tl.program_id(0)
    gid = tl.program_id(1) * BLOCK_G + tl.arange(0, BLOCK_G)
    half = tl.arange(0, 16)
    Kg = K // 32
    m = (gid < Kg)[:, None]
    pk = tl.load(pk_ptr + row * (K // 2) + gid[:, None] * 16 + half[None, :], mask=m, other=0)
    e = tl.load(sc_ptr + row * Kg + gid, mask=gid < Kg, other=127).to(tl.float32)
    lo = (pk & 0xF).to(tl.int32)
    hi = ((pk >> 4) & 0xF).to(tl.int32)
    mlo = lo & 7
    mhi = hi & 7
    maglo = tl.where(mlo < 4, mlo.to(tl.float32) * 0.5, tl.where(mlo == 4, 2.0, tl.where(mlo == 5, 3.0, tl.where(mlo == 6, 4.0, 6.0))))
    maghi = tl.where(mhi < 4, mhi.to(tl.float32) * 0.5, tl.where(mhi == 4, 2.0, tl.where(mhi == 5, 3.0, tl.where(mhi == 6, 4.0, 6.0))))
    f = tl.exp2(e - 127.0)[:, None]
    vlo = tl.where((lo >> 3) == 1, -1.0, 1.0) * maglo * f
    vhi = tl.where((hi >> 3) == 1, -1.0, 1.0) * maghi * f
    base = out_ptr + row * K + gid[:, None] * 32 + half[None, :] * 2
    tl.store(base, vlo.to(tl.bfloat16), mask=m)
    tl.store(base + 1, vhi.to(tl.bfloat16), mask=m)


def dequant_w4_to_bf16(W_packed, W_scales):
    """(N,K//2) u8 + (N,K//32) e8m0 u8 -> exact bf16 (N,K) weights (rotated frame)."""
    if W_packed.dtype != torch.uint8 or W_packed.ndim != 2 or W_scales.element_size() != 1:
        raise ValueError('MXFP4 weights and scales must contain packed one-byte values')
    N, Kh = W_packed.shape
    K = Kh * 2
    if K % 32 or W_scales.ndim != 2 or W_scales.shape[0] < N or W_scales.shape[1] < K // 32:
        raise ValueError('MXFP4 requires 32-wide groups and matching scale dimensions')
    W_packed = W_packed.contiguous()
    # Quantizer buffers can include padded rows/columns and E8M0-typed bytes.
    W_scales = W_scales.view(torch.uint8)[:N, :K // 32].contiguous()
    out = torch.empty(N, K, dtype=torch.bfloat16, device=W_packed.device)
    BLOCK_G = 16
    _dequant_w4_to_bf16_kernel[(N, triton.cdiv(K // 32, BLOCK_G))](W_packed, W_scales, out, K, BLOCK_G=BLOCK_G)
    return out


def mxfp4_w4a16_linear(x, W_packed, W_scales, pre_scale=None, BH=None, bias=None):
    """Packed 4-bit weights, BF16 activations; no activation quantization."""
    xb = x.to(torch.bfloat16)
    if pre_scale is not None:
        xb = xb * pre_scale
    if BH is not None:
        xb = rotate_blocks(xb, BH)
    W = dequant_w4_to_bf16(W_packed, W_scales)
    return torch.nn.functional.linear(xb, W, bias)

"""cudnn_mxfp4_dense.py , the same MACAQ dense W4A4 linear, with the GEMM handed
to a vendor block-scaled MXFP4 kernel (FlashInfer's `mm_fp4`, cuDNN backend)
instead of the in-house Triton `dot_scaled` one.

Only the GEMM changes. The activation chain is untouched: the fused Triton
kernel still does (x·pre_scale) @ blkdiag(BH) → MXFP4 in one launch, just
emitting its e8m0 scales in the 128x4 swizzle the vendor kernel reads
(`rot_scale_quantize_mxfp4(..., blocked=True)`) so no extra permute pass is
needed. Weights are consumed byte-for-byte from the checkpoint , same even/odd
nibble packing, same OCP e8m0 convention , with only a one-time lossless
permutation of the scale bytes at load (`blocked_weight_scales`). Nothing is
requantized, and MACAQ's out_scale of 1.0 means the GEMM needs no correction.

Measured on one GPU, Qwen3-32B linear shapes, per
decoder layer (qkv + o_proj + gate_up + down), against the Triton path:

    M=128  0.95x    M=512  1.65x    M=2048  1.80x
    M=256  1.15x    M=1024 1.85x    M=4096  1.90x   M=8192  2.00x

so the win is prefill / large-batch only, and `should_use` keeps everything
below MIN_M on Triton. That floor is not a tuning knob: the cuDNN kernel does
not merely underperform below 128 rows, it raises an illegal instruction and
takes the CUDA context with it, so MIN_M is clamped to >= 128.

Output is bit-identical to the Triton path (verified across all four Qwen3-32B
shapes at M in {256, 512, 2048, 4096}): both accumulate in fp32 and round to
bf16, whose 8-bit mantissa absorbs the difference in accumulation order.
"""
import os

import torch

from triton_mxfp4_dense import (
    blocked_scale_shape, quantize_mxfp4, rot_scale_quantize_mxfp4)

# Hard floor: below this the vendor kernel kills the CUDA context (not a
# graceful error). The default dispatch threshold is well above it anyway.
_HARD_MIN_M = 128
MIN_M = max(_HARD_MIN_M, int(os.environ.get("MACAQ_CUDNN_MIN_M", "256")))

_ENABLED = os.environ.get("MACAQ_CUDNN", "1") not in ("", "0", "false", "False")
_alpha_cache: dict = {}
_present = None


def backend_present() -> bool:
    """True if this build/device could run the vendor MXFP4 GEMM at all.

    Deliberately ignores MACAQ_CUDNN. Callers use this to decide what to
    ALLOCATE, and it must depend only on the machine and the model , never on a
    flag the user flips between runs. The blocked weight scales are op arguments,
    so their presence is baked into the torch.compile artifact, and vLLM's
    compile-cache key does not include our env vars: if MACAQ_CUDNN changed the
    argument list, a cached graph from one setting would be reused under the
    other and fail while unpacking its inputs."""
    global _present
    if _present is None:
        _present = False
        if torch.cuda.is_available():
            try:
                import flashinfer  # noqa: F401
                # Require native block-scaled FP4 support.
                _present = torch.cuda.get_device_capability()[0] >= 10
            except Exception:
                _present = False
    return _present


def available() -> bool:
    """True if the vendor MXFP4 GEMM should actually be dispatched to. This is
    the runtime switch (MACAQ_CUDNN) and is only ever read INSIDE the opaque
    custom op, where it cannot reach the traced graph."""
    return _ENABLED and backend_present()


def should_use(M: int, has_bias: bool = False) -> bool:
    """Dispatch predicate. M is tokens in the batch, not sequences , a decode
    step has M = concurrency, which is normally far below MIN_M.

    A bias sends the layer back to Triton so that bit-identity holds without
    caveat: the Triton epilogue folds bias into the fp32 accumulator, while the
    vendor GEMM only returns bf16, so adding it afterwards rounds twice. The
    vLLM plugin adds bias outside the op anyway (and Qwen3's linears carry
    none), so this costs nothing in practice."""
    return available() and M >= MIN_M and not has_bias


def blocked_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """(N, K/32) row-major e8m0 -> the 128x4 swizzle, padded. Pure permutation of
    the checkpoint's scale bytes; call once at load, never in the forward."""
    N, Kg = scales.shape
    pr, pc = blocked_scale_shape(N, Kg)
    padded = torch.zeros(pr, pc, dtype=scales.dtype, device=scales.device)
    padded[:N, :Kg] = scales
    blocks = padded.view(pr // 128, 128, pc // 4, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).view(pr, pc)


def _alpha(device, value: float) -> torch.Tensor:
    key = (device, float(value))
    if key not in _alpha_cache:
        _alpha_cache[key] = torch.tensor(value, device=device, dtype=torch.float32)
    return _alpha_cache[key]


def mxfp4_w4a4_linear_cudnn(x, W_packed, W_scales_blocked, pre_scale=None,
                            BH=None, bias=None, out_scale=1.0):
    """Same signature/semantics as mxfp4_w4a4_linear_fast, except the weight
    scales arrive pre-swizzled (blocked_weight_scales) rather than row-major.
    Returns (..., N) bf16.

    Bit-identical to the Triton path for bias=None. With a bias it differs by up
    to one bf16 ulp, because the Triton epilogue adds it to the fp32 accumulator
    and this can only add it to the rounded bf16 output , hence `should_use`
    declines biased layers."""
    import flashinfer

    lead, K = x.shape[:-1], x.shape[-1]
    x2 = x.reshape(-1, K).contiguous().to(torch.bfloat16)
    if pre_scale is None and BH is None:
        # Used by the speed-only SR ablation. The activation quantizer is
        # deliberately unfused here because there is no online transform.
        a_pk, a_sc = quantize_mxfp4(x2)
        a_scb = blocked_weight_scales(a_sc)
    else:
        a_pk, a_scb = rot_scale_quantize_mxfp4(x2, pre_scale, BH, blocked=True)
    out = flashinfer.mm_fp4(
        a_pk, W_packed.t(), a_scb, W_scales_blocked.t(),
        _alpha(x.device, out_scale), out_dtype=torch.bfloat16,
        block_size=32, use_nvfp4=False, backend="cudnn")
    if bias is not None:
        out = out + bias.to(out.dtype)
    return out.reshape(*lead, W_packed.shape[0])

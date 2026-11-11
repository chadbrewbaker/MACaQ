"""Diffusion W4A4 GEMM with BF16 conversion and bias fused into its epilogue."""

import torch
import fnmatch
import triton
import triton.language as tl
from macaq_mxfp4 import OCPDenseW4A4Linear
from macaq_vllm_mxfp4_dense import rot_scale_quantize_mxfp4


@triton.jit
def _gemm(A, W, AS, WS, Bias, Out, M, N: tl.constexpr, K: tl.constexpr,
          HAS_BIAS: tl.constexpr, BM: tl.constexpr = 64,
          BN: tl.constexpr = 128, BK: tl.constexpr = 256):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kb = tl.arange(0, BK // 2)
    ks = tl.arange(0, BK // 32)
    ap = A + rows[:, None] * (K // 2) + kb[None, :]
    wp = W + cols[:, None] * (K // 2) + kb[None, :]
    asp = AS + rows[:, None] * (K // 32) + ks[None, :]
    wsp = WS + cols[:, None] * (K // 32) + ks[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(K // BK):
        a = tl.load(ap, rows[:, None] < M, other=0)
        w = tl.load(wp, cols[:, None] < N, other=0)
        asc = tl.load(asp, rows[:, None] < M, other=127)
        wsc = tl.load(wsp, cols[:, None] < N, other=127)
        acc = tl.dot_scaled(a, asc, "e2m1", w.T, wsc, "e2m1", acc)
        ap += BK // 2
        wp += BK // 2
        asp += BK // 32
        wsp += BK // 32
    # Preserve the reference's two rounding points: GEMM -> BF16, then bias
    # addition -> BF16. Changing this order can change the denoising trajectory.
    value = acc.to(tl.bfloat16)
    if HAS_BIAS:
        bias = tl.load(Bias + cols, cols < N, other=0).to(tl.bfloat16)
        value = (value.to(tl.float32) + bias.to(tl.float32)).to(tl.bfloat16)
    tl.store(Out + rows[:, None] * N + cols[None, :], value,
             (rows[:, None] < M) & (cols[None, :] < N))


class FusedW4A4Linear(OCPDenseW4A4Linear):
    def forward(self, x):
        if self.activation_precision != "w4a4":
            return super().forward(x)
        k = self.in_features
        if k % 256 or self.B_inv_T.shape != (32, 32):
            return super().forward(x)
        flat = x.reshape(-1, k).to(torch.bfloat16).contiguous()
        packed, scales = rot_scale_quantize_mxfp4(flat, self.pre_scale, self.B_inv_T)
        out = torch.empty((flat.shape[0], self.out_features), device=x.device, dtype=torch.bfloat16)
        _gemm[(triton.cdiv(flat.shape[0], 64), triton.cdiv(self.out_features, 128))](
            packed, self.W_packed, scales, self.W_scales, self.bias, out,
            flat.shape[0], self.out_features, k, self.bias is not None,
            num_warps=8, num_stages=3)
        return out.reshape(*x.shape[:-1], self.out_features).to(x.dtype)


class W4A8Linear(OCPDenseW4A4Linear):
    """Same packed weights, higher-precision activations for sensitive layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from macaq_vllm_mxfp4_dense import w8_row_scales
        scale, emax = w8_row_scales(self.W_scales)
        self.register_buffer("fp8_weight_scale", scale, persistent=False)
        self.register_buffer("fp8_emax", emax, persistent=False)

    def forward(self, x):
        from macaq_vllm_mxfp4_dense import mxfp4_w4a8_linear_fast
        return mxfp4_w4a8_linear_fast(
            x, self.W_packed, self.W_scales, self.fp8_emax, self.fp8_weight_scale,
            pre_scale=self.pre_scale, BH=self.B_inv_T, bias=self.bias).to(x.dtype)


def set_kernel(transformer, mode, fp8_layers=()):
    if mode not in ("reference", "fused"):
        raise ValueError("kernel must be reference or fused")
    cls = FusedW4A4Linear if mode == "fused" else OCPDenseW4A4Linear
    names = [name for name, module in transformer.named_modules()
             if isinstance(module, OCPDenseW4A4Linear)]
    if any(not any(fnmatch.fnmatchcase(name, pattern) for name in names) for pattern in fp8_layers):
        raise ValueError("each --fp8-layers pattern must match a quantized layer")
    for name, module in list(transformer.named_modules()):
        if not isinstance(module, OCPDenseW4A4Linear):
            continue
        selected = W4A8Linear if any(fnmatch.fnmatchcase(name, pattern) for pattern in fp8_layers) else cls
        replacement = selected(module.W_packed, module.W_scales, module.B_inv_T,
                          module.bias, 32, module.out_features, module.in_features,
                          pre_scale=module.pre_scale,
                          activation_precision=module.activation_precision)
        parent, _, leaf = name.rpartition(".")
        setattr(transformer.get_submodule(parent), leaf, replacement)

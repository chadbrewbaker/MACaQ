"""vLLM integration for MACaQ dense and MoE MXFP4 checkpoints.

Dense projections load saved packed weights and input transforms. Projections
with identical transforms share a GEMM. Qwen3.5-family text classes also expose
the hybrid cache and M-RoPE interfaces required by the runtime.

MoE experts retain separate diagonals, applied inside the grouped W4A4 kernel.
Their transform cannot be represented by the Marlin weight-only backend.

Native Gated DeltaNet kernels are the default. MACAQ_GDN_REFERENCE=1 explicitly
selects a slower PyTorch implementation for diagnosis.
"""

import math
import os
import torch
from torch import nn
from macaq_hybrid import register_hybrid_ops

register_hybrid_ops()


# ── Register Qwen3_5MoeForCausalLM (text-only) in vLLM's model registry ─────
from vllm.model_executor.models.registry import ModelRegistry as _ModelRegistry

if "Qwen3_5MoeForCausalLM" not in _ModelRegistry.models:
    _ModelRegistry.register_model(
        "Qwen3_5MoeForCausalLM",
        "vllm.model_executor.models.qwen3_5:Qwen3_5MoeForCausalLM",
    )


from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM as _Qwen3_5MoeForCausalLM,
    Qwen3_5ForConditionalGeneration as _Qwen3_5ForConditionalGeneration,
)

# Flat text checkpoints use the same hybrid cache and position interfaces.
if "Qwen3_5ForCausalLM" not in _ModelRegistry.models:
    _ModelRegistry.register_model("Qwen3_5ForCausalLM", "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM")
if not getattr(_Qwen3_5ForCausalLM, "_macaq_hybrid_patched", False):
    _Qwen3_5ForCausalLM.is_hybrid = True
    for name in ("get_mamba_state_shape_from_config", "get_mamba_state_dtype_from_config", "get_mamba_state_copy_func"):
        setattr(_Qwen3_5ForCausalLM, name, getattr(_Qwen3_5ForConditionalGeneration, name))
    _Qwen3_5ForCausalLM.supports_mrope = True
    def _dense_text_mrope(self, input_tokens, mm_features):
        positions = torch.arange(len(input_tokens), dtype=torch.long).unsqueeze(0).expand(3, -1).contiguous()
        return positions, 0
    _Qwen3_5ForCausalLM.get_mrope_input_positions = _dense_text_mrope
    _Qwen3_5ForCausalLM._macaq_hybrid_patched = True

# Advertise hybrid + carry the GDN-state classmethods. Without these vLLM's
# `_align_hybrid_block_size` is skipped (`model_config.is_hybrid == False`),
# `cache_config.mamba_block_size` never gets set, and the GDN
# `get_kv_cache_spec` assertion fails.
if not getattr(_Qwen3_5MoeForCausalLM, "_macaq_hybrid_patched", False):
    _Qwen3_5MoeForCausalLM.is_hybrid = True
    _Qwen3_5MoeForCausalLM.get_mamba_state_shape_from_config = (
        _Qwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config
    )
    _Qwen3_5MoeForCausalLM.get_mamba_state_dtype_from_config = (
        _Qwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config
    )
    _Qwen3_5MoeForCausalLM.get_mamba_state_copy_func = (
        _Qwen3_5ForConditionalGeneration.get_mamba_state_copy_func
    )
    _Qwen3_5MoeForCausalLM._macaq_hybrid_patched = True

# Add M-RoPE support , Qwen3.5 configs carry `mrope_section` and vLLM checks
# `uses_mrope(hf_config)`; the text-only class doesn't satisfy `SupportsMRoPE`
# on its own.
if not getattr(_Qwen3_5MoeForCausalLM, "_macaq_mrope_patched", False):
    _Qwen3_5MoeForCausalLM.supports_mrope = True

    def _text_only_get_mrope_input_positions(self, input_tokens, mm_features):
        """Text-only M-RoPE positions: same scalar arange replicated across T/H/W."""
        n = len(input_tokens)
        llm_positions = torch.arange(n, dtype=torch.long).unsqueeze(0).expand(3, n).contiguous()
        return llm_positions, 0

    _Qwen3_5MoeForCausalLM.get_mrope_input_positions = _text_only_get_mrope_input_positions
    _Qwen3_5MoeForCausalLM._macaq_mrope_patched = True


# Strip Qwen3_5VL checkpoint prefixes at load time so the same vLLM
# `Qwen3_5MoeForCausalLM` class can consume both:
#   - our pre-flattened MXFP4 export   (already `model.layers.X...`)
#   - the original HF VL checkpoint    (`model.language_model.layers.X...`
#     plus `model.visual.*` and `mtp.*` keys that we drop)
# No-op on the MXFP4 export so it's safe to enable unconditionally.
if not getattr(_Qwen3_5MoeForCausalLM, "_macaq_textonly_loadw_patched", False):
    _orig_load_weights = _Qwen3_5MoeForCausalLM.load_weights

    def _macaq_textonly_load_weights(self, weights):
        def _remap(it):
            for name, tensor in it:
                if name.startswith("model.visual.") \
                        or name.startswith("mtp.") \
                        or name.startswith("model.mtp."):
                    continue
                if name.startswith("model.language_model."):
                    name = "model." + name[len("model.language_model."):]
                yield name, tensor
        return _orig_load_weights(self, _remap(weights))

    _Qwen3_5MoeForCausalLM.load_weights = _macaq_textonly_load_weights
    _Qwen3_5MoeForCausalLM._macaq_textonly_loadw_patched = True


# ── Pure-PyTorch GDN forward fallback ──────────────────────────────────────
@torch.no_grad()
def _pure_torch_chunk_gated_delta_rule(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    g: torch.Tensor, beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
):
    """
    Recurrent gated-delta-rule forward, in plain PyTorch.

    Conventions match `vllm/.../fla/ops/chunk.py::chunk_gated_delta_rule`:
      q, k        : [B, T, H, K]        bf16/fp16
      v           : [B, T, HV, V]       bf16/fp16     (HV may differ from H; GVA)
      g           : [B, T, H]           bf16/fp16  , log-space decay
      beta        : [B, T, H]           bf16/fp16
      initial_state : [N, HV, V, K]     fp32       (N==B when no cu_seqlens)
      cu_seqlens  : [N+1] int32          , packed variable-length seqs

    Returns:
      o           : [B, T, HV, V]       same dtype as q
      final_state : [N, HV, V, K] fp32 if output_final_state else None
    """
    if use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k = torch.nn.functional.normalize(k.float(), p=2, dim=-1).to(k.dtype)

    in_dtype = q.dtype
    q32 = q.float()
    k32 = k.float()
    v32 = v.float()
    g32 = g.float()
    beta32 = beta.float()

    B, T, H_kq, K = q32.shape
    HV = v32.shape[-2]
    V = v32.shape[-1]
    assert HV % H_kq == 0, f"HV={HV} must be divisible by H_kq={H_kq}"
    gva_ratio = HV // H_kq
    if gva_ratio > 1:
        # Grouped Value Attention: replicate q/k along the head axis so each
        # value head has a matching key/query.
        q32 = q32.repeat_interleave(gva_ratio, dim=-2)
        k32 = k32.repeat_interleave(gva_ratio, dim=-2)
    H = HV
    scale = 1.0 / (K ** 0.5)

    if cu_seqlens is None:
        N = B
        starts = torch.arange(0, B * T + 1, T, device=q.device, dtype=torch.int64)
        q32 = q32.reshape(B * T, H, K)
        k32 = k32.reshape(B * T, H, K)
        v32 = v32.reshape(B * T, H, V)
        g32 = g32.reshape(B * T, H)
        beta32 = beta32.reshape(B * T, H)
    else:
        assert B == 1, "cu_seqlens path expects B == 1 (packed batch)"
        starts = cu_seqlens.to(torch.int64)
        N = starts.numel() - 1
        q32 = q32.squeeze(0)
        k32 = k32.squeeze(0)
        v32 = v32.squeeze(0)
        g32 = g32.squeeze(0)
        beta32 = beta32.squeeze(0)

    o = torch.empty_like(v32)

    if initial_state is None:
        S = torch.zeros(N, H, V, K, dtype=torch.float32, device=q.device)
    else:
        S = initial_state.to(torch.float32).clone()

    for n in range(N):
        s0 = int(starts[n].item())
        s1 = int(starts[n + 1].item())
        if s1 <= s0:
            continue
        S_n = S[n]
        decay = g32[s0:s1].exp()
        beta_n = beta32[s0:s1]
        q_n = q32[s0:s1] * scale
        k_n = k32[s0:s1]
        v_n = v32[s0:s1]
        T_n = s1 - s0
        for t in range(T_n):
            S_n = decay[t][:, None, None] * S_n
            v_pred = torch.einsum('hvk,hk->hv', S_n, k_n[t])
            r = v_n[t] - v_pred
            S_n = S_n + beta_n[t][:, None, None] * torch.einsum('hv,hk->hvk', r, k_n[t])
            o[s0 + t] = torch.einsum('hvk,hk->hv', S_n, q_n[t])
        S[n] = S_n

    if cu_seqlens is None:
        o = o.view(B, T, H, V)
    else:
        o = o.unsqueeze(0)

    o = o.to(in_dtype).contiguous()
    final_state = S if output_final_state else None
    return o, final_state


from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule as _CGDR
if os.environ.get("MACAQ_GDN_REFERENCE") == "1" and not getattr(_CGDR, "_macaq_gdn_patched", False):
    def _patched_forward(self, q, k, v, g, beta, initial_state, output_final_state,
                          cu_seqlens=None, chunk_indices=None, chunk_offsets=None,
                          use_qk_l2norm_in_kernel=True):
        return _pure_torch_chunk_gated_delta_rule(
            q, k, v, g, beta, initial_state, output_final_state,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    _CGDR.forward_native = _patched_forward
    _CGDR.forward_cuda = _patched_forward
    _CGDR._macaq_gdn_patched = True


# ── FusedMoE.weight_loader direct-copy fallback ─────────────────────────────
# vLLM's Qwen3.5 load_weights remaps HF-format MoE tensors (e.g.
# `experts.<e>.gate_up_proj`) via a per-expert mapping into vLLM's
# `w13_weight` slot. Our saved checkpoint already carries pre-fused
# `experts.w13_weight` / `w2_weight` / `_scale` keys, which fall through
# to `qwen3_5.py`'s direct loader: `weight_loader(param, loaded_weight)`
# (2 args). For FusedMoE-owned parameters that loader is
# `FusedMoE.weight_loader` with a 5-arg real signature , TypeError. We
# add a thin fallback: if called with 2 args, do a direct shape-matched copy.
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _FusedMoE

if not getattr(_FusedMoE, "_macaq_patched_weight_loader", False):
    _orig_fmoe_weight_loader = _FusedMoE.weight_loader

    def _macaq_patched_weight_loader(
        self,
        param,
        loaded_weight,
        weight_name=None,
        shard_id=None,
        expert_id=None,
        return_success: bool = False,
    ):
        if weight_name is None and shard_id is None and expert_id is None:
            if param.shape != loaded_weight.shape:
                from macaq_expert_layout import partition_expert_tensor
                from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
                name = next((name for name in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale",
                                               "w13_pre_scale", "w2_pre_scale")
                             if getattr(self, name, None) is param), None)
                loaded_weight = partition_expert_tensor(name, loaded_weight, param.shape,
                    get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
            assert param.shape == loaded_weight.shape, (
                f"shape mismatch in direct-copy fallback: "
                f"param={tuple(param.shape)} loaded={tuple(loaded_weight.shape)}")
            param.data.copy_(loaded_weight.to(param.dtype))
            return True if return_success else None
        return _orig_fmoe_weight_loader(
            self, param, loaded_weight, weight_name, shard_id, expert_id,
            return_success=return_success,
        )

    _FusedMoE.weight_loader = _macaq_patched_weight_loader
    _FusedMoE._macaq_patched_weight_loader = True


# ── Hadamard rotation injected into the Qwen3-Next MoE block forward ────────
# The block-diagonal Hadamard with block size 8 is shared across all
# experts in all blocks. `quantize_and_save.py` folds it into each
# `gate_up_proj` weight; here we apply the same rotation to the
# hidden_states going into the experts. By orthogonality the two cancel,
# so gate_up_proj sees its original input up to MXFP4 noise.
#
# Importantly, the *router* (`self.gate`) sees the unrotated input , its
# expert-assignment logits are unchanged.
#
# The Hadamard is registered as a per-module `register_buffer` at module
# __init__ time so `torch.compile` / cudagraph capture treats it as a
# stable graph input (instead of a CPU constant that needs copying).
_HADAMARD_B8_BLOCK = 8       # legacy default block size when config omits it


def _make_hadamard(n: int) -> torch.Tensor:
    """Build the n×n orthogonal (Sylvester) Hadamard matrix."""
    H = torch.tensor(
        [[(-1) ** bin(i & j).count('1') for j in range(n)] for i in range(n)],
        dtype=torch.float32,
    ) / math.sqrt(n)
    return H.bfloat16().contiguous()


from vllm.model_executor.models.qwen3_next import (
    Qwen3NextSparseMoeBlock as _Qwen3NextSparseMoeBlock,
)
from vllm.model_executor.models.utils import (
    sequence_parallel_chunk as _sequence_parallel_chunk,
)
from vllm.distributed import (
    tensor_model_parallel_all_gather as _tensor_model_parallel_all_gather,
)

import os as _os
_DISABLE_HADAMARD = bool(_os.environ.get("MACAQ_DISABLE_HADAMARD"))

# Per-(device, block) cache of the Hadamard matrix for the in-kernel rotation.
_HADAMARD_GPU: dict[tuple, torch.Tensor] = {}


def _hadamard_for(device: torch.device, dtype: torch.dtype, b: int) -> torch.Tensor:
    H = _HADAMARD_GPU.get((device, b))
    if H is None:
        H = _make_hadamard(b).to(device=device)
        _HADAMARD_GPU[(device, b)] = H
    return H.to(dtype)


def _rotate_hidden(x: torch.Tensor, layer=None) -> torch.Tensor:
    """Legacy per-layer learned-rotation path (`rotation_mode="perlayer"`): a
    block-diagonal rotation applied to the expert input so the quantize-time fold
    cancels. The MACAQ per-expert MoE checkpoint does NOT use this , its
    per-expert diagonal + Hadamard are folded inside the W4A4 kernel. A per-layer
    learned rotation (`layer.w13_rotation`) takes precedence; otherwise a block
    Hadamard at `layer.macaq_rot_block`.
    """
    K = x.shape[-1]
    R = getattr(layer, "w13_rotation", None) if layer is not None else None
    if R is not None:
        n_B, n, _ = R.shape
        assert n_B * n == K, f"per-layer rotation: K={K} != n_B*n={n_B}*{n}"
        return torch.einsum(
            "ibn,bnk->ibk", x.view(-1, n_B, n), R.to(x.dtype)
        ).reshape(-1, K)
    b = getattr(layer, "macaq_rot_block", _HADAMARD_B8_BLOCK) if layer is not None \
        else _HADAMARD_B8_BLOCK
    H = _hadamard_for(x.device, x.dtype, b)
    return (x.view(-1, K // b, b) @ H).view(-1, K)


# Inject the Hadamard rotation INSIDE GptOssMxfp4MoEMethod.apply / apply_monolithic
# so only the routed-experts kernel sees rotated x. Router (self.gate) and
# shared_expert see the original unrotated hidden_states , matching the
# HF reference's per-linear local rotation pattern.
from vllm.model_executor.layers.quantization.mxfp4 import (
    GptOssMxfp4MoEMethod as _GptOssMxfp4MoEMethod_apply,
)
if not getattr(_GptOssMxfp4MoEMethod_apply, "_macaq_apply_rot", False):
    _orig_method_apply = _GptOssMxfp4MoEMethod_apply.apply
    _orig_method_apply_mono = _GptOssMxfp4MoEMethod_apply.apply_monolithic

    def _patched_method_apply(self, layer, x, topk_weights, topk_ids,
                                shared_experts_input):
        if getattr(layer, "w13_pre_scale", None) is not None:
            raise RuntimeError(
                "MACAQ per-expert MoE checkpoints require the true W4A4 path; "
                "re-run with MACAQ_TRITON_W4A4=1 (the Marlin W4A16 backend cannot "
                "fold the per-expert diagonal + Hadamard).")
        if not _DISABLE_HADAMARD:
            x = _rotate_hidden(x, layer=layer)
        return _orig_method_apply(self, layer, x, topk_weights, topk_ids,
                                    shared_experts_input)

    def _patched_method_apply_mono(self, layer, x, router_logits, input_ids=None):
        if getattr(layer, "w13_pre_scale", None) is not None:
            raise RuntimeError(
                "MACAQ per-expert MoE checkpoints require the true W4A4 path; "
                "re-run with MACAQ_TRITON_W4A4=1 (the Marlin W4A16 backend cannot "
                "fold the per-expert diagonal + Hadamard).")
        if not _DISABLE_HADAMARD:
            x = _rotate_hidden(x, layer=layer)
        return _orig_method_apply_mono(self, layer, x, router_logits, input_ids=input_ids)

    _GptOssMxfp4MoEMethod_apply.apply = _patched_method_apply
    _GptOssMxfp4MoEMethod_apply.apply_monolithic = _patched_method_apply_mono
    _GptOssMxfp4MoEMethod_apply._macaq_apply_rot = True


# ── Disable GPT-OSS-specific SwiGLU clamping for non-GPT-OSS MXFP4 models ────
# `GptOssMxfp4MoEMethod.get_fused_moe_quant_config` hardcodes
# `swiglu_limit=7.0` and `gemm1_alpha=1.702`. Those are GPT-OSS's SwiGLU
# variant; Qwen3.5 uses plain `silu(gate) * up` with no clamp. With plain
# MXFP4 the clamp rarely fires (gate values stay well below 7), but with
# the Hadamard rotation the per-token gate distribution shifts just
# enough that the ±7 clamp starts destroying signal , explaining why
# Hadamard runs scrambled outputs even though the underlying math
# (verified by an isolated Marlin GEMM test) cancels cleanly.
from vllm.model_executor.layers.quantization.mxfp4 import (
    GptOssMxfp4MoEMethod as _GptOssMxfp4MoEMethod,
)
if not getattr(_GptOssMxfp4MoEMethod, "_macaq_no_swiglu_clamp", False):
    _orig_get_fused = _GptOssMxfp4MoEMethod.get_fused_moe_quant_config

    def _patched_get_fused(self, layer):
        cfg = _orig_get_fused(self, layer)
        # Strip GPT-OSS SwiGLU variants , Qwen3.5 uses plain silu(gate) * up
        # with no clamp / no alpha. Leaving these in scrambles the model.
        if cfg is not None:
            cfg.gemm1_clamp_limit = None
            cfg.gemm1_alpha = None
            cfg.gemm1_beta = None
        return cfg

    _GptOssMxfp4MoEMethod.get_fused_moe_quant_config = _patched_get_fused
    _GptOssMxfp4MoEMethod._macaq_no_swiglu_clamp = True


# ── Honor `ignored_layers` for FusedMoE (vLLM only checks LinearBase) ───────
# Stock `Mxfp4Config.get_quant_method` ignores its own `ignored_layers` list
# when the layer is a FusedMoE , it always returns GptOssMxfp4MoEMethod. To
# keep specific MoE blocks (e.g. layers 0 and 47) in bf16, patch the method
# so a matched prefix falls back to UnquantizedFusedMoEMethod. Because the
# Hadamard-rotation patch above lives on `GptOssMxfp4MoEMethod.apply`, the
# bf16 layers automatically skip the rotation too , they use a different
# class with no monkey-patched apply.
from vllm.model_executor.layers.quantization.mxfp4 import (
    Mxfp4Config as _Mxfp4Config,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE as _FusedMoE_cls
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod as _UnqFusedMoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped as _is_layer_skipped,
)
if not getattr(_Mxfp4Config, "_macaq_moe_ignored", False):
    _orig_get_quant_method = _Mxfp4Config.get_quant_method
    _orig_from_config = _Mxfp4Config.from_config

    def _patched_get_quant_method(self, layer, prefix):
        if isinstance(layer, _FusedMoE_cls) and self.ignored_layers:
            if _is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return _UnqFusedMoEMethod(layer.moe_config)
        method = _orig_get_quant_method(self, layer, prefix)
        # Attach rotation mode to the method instance so create_weights
        # can read it. The module-level global doesn't propagate across
        # vLLM's parent/child process boundary, but the Mxfp4Config
        # instance is pickled with `rotation_mode` set.
        if method is not None:
            method._macaq_rotation_mode = getattr(self, "rotation_mode", None)
            method._macaq_rotation_block_size = int(
                getattr(self, "rotation_block_size", 8)
            )
            method._macaq_moe = bool(getattr(self, "macaq_moe", False))
        return method

    @classmethod
    def _patched_from_config(cls, config):
        # Stock from_config drops everything; we need ignored_layers to flow
        # through so per-layer bf16 fallback works. Also surface
        # rotation_mode / rotation_block_size / macaq_moe for the MACAQ MoE
        # checkpoint (shared per-layer D1 pre_scale + a wider block-Hadamard).
        cfg = cls(ignored_layers=config.get("ignored_layers"))
        cfg.rotation_mode = config.get("rotation_mode")
        cfg.macaq_moe = bool(config.get("macaq_moe", False))
        # MACAQ MoE rotates at the MXFP4 group (32) unless told otherwise.
        default_block = 32 if cfg.macaq_moe else 8
        cfg.rotation_block_size = int(config.get("rotation_block_size", default_block))
        return cfg

    _Mxfp4Config.get_quant_method = _patched_get_quant_method
    _Mxfp4Config.from_config = _patched_from_config
    _Mxfp4Config._macaq_moe_ignored = True


# ── Register w13_rotation parameter when rotation_mode == "perlayer" ────────
# vLLM's GptOssMxfp4MoEMethod.create_weights doesn't know about our learned
# block-diagonal rotation. When the loaded config says rotation_mode=perlayer
# we extend create_weights to also register `w13_rotation [n_B, n, n] bf16`
# on each FusedMoE layer; the standard weight loader (FusedMoE.weight_loader
# direct-copy fallback patched above) fills it from the safetensors.
from vllm.model_executor.utils import set_weight_attrs as _set_weight_attrs
if not getattr(_GptOssMxfp4MoEMethod, "_macaq_perlayer_create_patched", False):
    _orig_cw = _GptOssMxfp4MoEMethod.create_weights

    def _patched_create_weights(self, layer, num_experts, hidden_size,
                                  intermediate_size_per_partition,
                                  params_dtype, **extra_weight_attrs):
        _orig_cw(self, layer, num_experts, hidden_size,
                 intermediate_size_per_partition, params_dtype,
                 **extra_weight_attrs)
        # Read mode/flags from this method instance (set by the patched
        # get_quant_method); module globals aren't shared across vLLM's
        # parent/child process boundary.
        n = int(getattr(self, "_macaq_rotation_block_size", 8))

        # MACAQ MoE: each expert is quantized like an independent dense linear ,
        # its OWN diagonal D1 (w13_pre_scale for gate_up, w2_pre_scale for down)
        # plus the shared 32×32 block-Hadamard, both folded into the routed
        # activation per-expert inside the grouped W4A4 kernel. No sharing.
        if getattr(self, "_macaq_moe", False):
            layer.macaq_rot_block = n
            E = num_experts
            I = intermediate_size_per_partition
            w13p = torch.nn.Parameter(
                torch.empty(E, hidden_size, dtype=torch.bfloat16), requires_grad=False)
            w2p = torch.nn.Parameter(
                torch.empty(E, I, dtype=torch.bfloat16), requires_grad=False)
            _set_weight_attrs(w13p, extra_weight_attrs)
            _set_weight_attrs(w2p, extra_weight_attrs)
            layer.register_parameter("w13_pre_scale", w13p)
            layer.register_parameter("w2_pre_scale", w2p)

        # Per-layer learned rotation variant (rotation_mode=perlayer).
        if getattr(self, "_macaq_rotation_mode", None) == "perlayer":
            assert hidden_size % n == 0, (
                f"perlayer rotation: hidden_size={hidden_size} not divisible by n={n}")
            rot = torch.nn.Parameter(
                torch.empty(hidden_size // n, n, n, dtype=torch.bfloat16),
                requires_grad=False,
            )
            _set_weight_attrs(rot, extra_weight_attrs)
            layer.register_parameter("w13_rotation", rot)

    _GptOssMxfp4MoEMethod.create_weights = _patched_create_weights
    _GptOssMxfp4MoEMethod._macaq_perlayer_create_patched = True


# ── Triton W4A4 MXFP4 path (MACAQ_TRITON_W4A4=1) ─────────────────────────────
# Bypass Marlin (W4A16) entirely: keep the raw packed MXFP4 weights on the
# layer (skip prepare_moe_mxfp4_layer_for_marlin) and run our grouped
# tl.dot_scaled W4A4 kernel in apply(). The kernel's nibble/e8m0 convention is
# byte-identical to quantize_moe_qwen35.py::pack_mxfp4_ocp, so the stored
# w13_weight/_scale + w2_weight/_scale feed straight in with no repack.
# Verified native FP4 mxf4 MMA; see triton_mxfp4_moe.py.
_TRITON_W4A4 = bool(_os.environ.get("MACAQ_TRITON_W4A4"))

if _TRITON_W4A4 and not getattr(_GptOssMxfp4MoEMethod, "_macaq_triton_w4a4", False):
    # Make our repo dir importable from vLLM worker subprocesses.
    import sys as _sys
    _repo = _os.path.dirname(_os.path.abspath(__file__))
    if _repo not in _sys.path:
        _sys.path.insert(0, _repo)

    def _keep_raw_pwal(self, layer):
        """No Marlin repack , keep raw packed MXFP4 weights for the Triton kernel,
        and precompute the 32×32 block-Hadamard once on the weight device (the
        forward never builds it with Python bin())."""
        layer.w13_weight.data = layer.w13_weight.data.contiguous()
        layer.w2_weight.data = layer.w2_weight.data.contiguous()
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.contiguous()
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.contiguous()
        b = int(getattr(layer, "macaq_rot_block", 32))
        layer.macaq_bh = _make_hadamard(b).to(layer.w13_weight.device)
        self.moe_kernel = None  # bypassed by our apply()

    def _triton_w4a4_apply(self, layer, x, topk_weights, topk_ids,
                            shared_experts_input):
        # The fused MoE kernel folds each expert's diagonal (w13_pre_scale /
        # w2_pre_scale) and the block-Hadamard into its ROUTED activation, and
        # returns the ROUTED experts' contribution only. The always-on SHARED
        # expert is NOT added here: vLLM's MoERunner computes it separately
        # (`layer.shared_experts._layer`, an unquantized bf16 gated MLP) and adds
        # it in `forward` (result = shared_output + fused_output). Adding it here
        # too would double-count the shared expert on every token of every layer
        # (verified at runtime: moe_kernel is None -> mk_owns_shared_expert False
        # -> the runner owns the shared add).
        from triton_mxfp4_moe import fused_mxfp4_moe
        out = fused_mxfp4_moe(
            x,
            layer.w13_weight, layer.w13_weight_scale,
            layer.w2_weight, layer.w2_weight_scale,
            topk_ids.to(torch.int32), topk_weights.to(torch.float32),
            layer.w13_pre_scale, layer.w2_pre_scale, layer.macaq_bh,
        )
        return out

    _GptOssMxfp4MoEMethod.process_weights_after_loading = _keep_raw_pwal
    _GptOssMxfp4MoEMethod.apply = _triton_w4a4_apply
    _GptOssMxfp4MoEMethod._macaq_triton_w4a4 = True
    # vLLM still logs its own "Using '<X>' Mxfp4 MoE backend." pick at init;
    # we replaced apply/process_weights above, so flag the real path.
    print("[macaq-mxfp4] MACAQ_TRITON_W4A4=1: FusedMoE on the true grouped W4A4 "
          "kernel (vLLM's 'Mxfp4 MoE backend' line is overridden).", flush=True)


# ── Dense W4A4 MXFP4 LinearMethod for macaq-quantized DENSE models ───────────
# (Qwen3-14B etc.: Qwen3ForCausalLM, no MoE). The checkpoint (save_macaq_vllm.py)
# carries per-Linear weight_packed/weight_scale (OCP convention) + pre_scale(D1),
# with a quantization_config marking macaq_dense_w4a4=True. vLLM merges
# q/k/v->qkv_proj & gate/up->gate_up_proj (output-dim shards); q/k/v carry
# DISTINCT pre_scale (D1 depends on each weight), so we store it per-shard and run
# a per-shard transform+GEMM in apply via triton_mxfp4_dense.mxfp4_w4a4_linear.
# The block-Hadamard is rebuilt from config (canonical Sylvester), not loaded.
from vllm.model_executor.layers.linear import (
    LinearBase as _LinearBase, LinearMethodBase as _LinearMethodBase,
)

_SHARD_IDX = {"q": 0, "k": 1, "v": 2, 0: 0, 1: 1, "gate": 0, "up": 1}


def _make_macaq_bh(b, device, dtype):
    H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(b)]
                      for i in range(b)], dtype=torch.float32) / math.sqrt(b)
    return H.to(device=device, dtype=dtype)


# The per-shard (transform → MXFP4-quantize → dot_scaled GEMM) is wrapped as a
# registered custom op so it is a single OPAQUE node to torch.compile/Dynamo
# (raw triton.jit launches inside mxfp4_w4a4_linear can't be traced otherwise →
# graph break → cudagraph capture fails). With this op + a precomputed Hadamard
# (no Python bin() in the forward), the macaq dense path runs in graph mode.
def _macaq_dense_w4a4_op(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_blocked: list[torch.Tensor],
    pre_scale: torch.Tensor,
    bh: torch.Tensor,
    shard_offs: list[int],
    out_scale: float,
) -> torch.Tensor:
    # Speed-only ablation: skip both online balancing and rotation. The stored
    # weights remain unchanged; this intentionally measures only the removed
    # activation-side work.
    disable_sr = _os.environ.get("MACAQ_DISABLE_SR", "") not in (
        "", "0", "false", "False")
    if disable_sr:
        pre_scale = None
        bh = None
    pre_arg = None if disable_sr else pre_scale
    if _os.environ.get("MACAQ_FAST", "1") not in ("", "0", "false", "False"):
        from triton_mxfp4_dense import mxfp4_w4a4_linear_fast as mxfp4_w4a4_linear
    else:
        from triton_mxfp4_dense import mxfp4_w4a4_linear
    # The M branch lives INSIDE the op, which is opaque to Dynamo , putting it in
    # apply() would guard on the (dynamic) batch dim and force recompiles.
    import cudnn_mxfp4_dense as _cudnn
    use_cudnn = bool(weight_scale_blocked) and _cudnn.should_use(
        x.numel() // x.shape[-1])
    outs = []
    for s in range(len(shard_offs) - 1):
        o0, o1 = shard_offs[s], shard_offs[s + 1]
        if use_cudnn:
            outs.append(_cudnn.mxfp4_w4a4_linear_cudnn(
                x, weight_packed[o0:o1], weight_scale_blocked[s],
                pre_scale=pre_arg[s] if pre_arg is not None else None,
                BH=bh, bias=None, out_scale=out_scale))
        else:
            outs.append(mxfp4_w4a4_linear(
                x, weight_packed[o0:o1], weight_scale[o0:o1],
                pre_scale=pre_arg[s] if pre_arg is not None else None,
                BH=bh, bias=None, out_scale=out_scale))
    return outs[0] if len(outs) == 1 else torch.cat(outs, dim=-1)


def _macaq_dense_w4a4_fake(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_blocked: list[torch.Tensor],
    pre_scale: torch.Tensor,
    bh: torch.Tensor,
    shard_offs: list[int],
    out_scale: float,
) -> torch.Tensor:
    N = weight_packed.shape[0]                       # sum of output-dim shards
    return x.new_empty((*x.shape[:-1], N), dtype=torch.bfloat16)


if not hasattr(torch.ops.vllm, "macaq_dense_w4a4"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        "macaq_dense_w4a4", _macaq_dense_w4a4_op,
        mutates_args=[], fake_impl=_macaq_dense_w4a4_fake)


class _MacaqDenseW4A4Method(_LinearMethodBase):
    def __init__(self, hadamard_block):
        self.b = int(hadamard_block)
        # Weights are OCP MXFP4 (value = code·2^(e8−127)) → no GEMM correction.
        self.out_scale = 1.0

    def create_weights(self, layer, input_size_per_partition,
                        output_partition_sizes, input_size, output_size,
                        params_dtype, **extra):
        K = input_size_per_partition
        sizes = list(output_partition_sizes)
        N, ns = sum(sizes), len(sizes)
        offs = [0]
        for s in sizes:
            offs.append(offs[-1] + s)
        layer.macaq_shard_offs = offs
        wp = nn.Parameter(torch.empty(N, K // 2, dtype=torch.uint8), requires_grad=False)
        ws = nn.Parameter(torch.empty(N, K // 32, dtype=torch.uint8), requires_grad=False)
        pre = nn.Parameter(torch.empty(ns, K, dtype=torch.bfloat16), requires_grad=False)

        def _loader(kind):
            def load(param, loaded, shard_id=None):
                if isinstance(shard_id, tuple):
                    indices = list(shard_id)
                else:
                    indices = [_SHARD_IDX.get(shard_id, 0 if shard_id is None else shard_id)]
                if (not indices or not all(isinstance(i, int) for i in indices)
                        or indices[0] < 0 or indices[-1] >= ns
                        or indices != list(range(indices[0], indices[-1] + 1))):
                    raise ValueError("invalid packed projection shard")
                i = indices[0]
                d = param.data
                lw = loaded.view(d.dtype) if loaded.dtype != d.dtype and \
                    loaded.element_size() == d.element_size() else loaded.to(d.dtype)
                if kind == "row":                      # output-dim sharded
                    o0 = offs[i]
                    d[o0:o0 + lw.shape[0]].copy_(lw)
                else:                                  # pre_scale: per-shard vector
                    for index in indices:
                        d[index].copy_(lw.reshape(-1))
            return load

        for name, p, kind in (("weight_packed", wp, "row"),
                              ("weight_scale", ws, "row"),
                              ("pre_scale", pre, "vec")):
            layer.register_parameter(name, p)
            _set_weight_attrs(p, {"weight_loader": _loader(kind)})

    def process_weights_after_loading(self, layer):
        # Precompute the block-Hadamard once on the weight device so the forward
        # never constructs it with Python bin() (untraceable by Dynamo).
        layer.macaq_bh = _make_macaq_bh(
            self.b, layer.weight_packed.device, torch.bfloat16)
        # Fused GDN q/k/v projections share one saved input transform. Merge
        # adjacent equal transforms so they use one GEMM instead of three.
        offsets = layer.macaq_shard_offs
        starts, selected = [offsets[0]], [0]
        for shard in range(1, len(offsets) - 1):
            if not torch.equal(layer.pre_scale[shard], layer.pre_scale[shard - 1]):
                starts.append(offsets[shard])
                selected.append(shard)
        layer.macaq_shard_offs = starts + [offsets[-1]]
        layer.pre_scale.data = layer.pre_scale.data[selected].contiguous()
        # Vendor block-scaled GEMM (prefill only): it reads the e8m0 scales in a
        # 128x4 swizzle. Permute the checkpoint's scale bytes once, per shard ,
        # the swizzle of a row range is not the row range of the swizzle. The
        # weights themselves are never touched, and nothing is requantized.
        # Costs ~1/16 of the packed-weight bytes in VRAM.
        # Only for a 32-wide rotation: the vendor GEMM itself is agnostic (MXFP4
        # groups are 32 regardless), but the fused kernel that emits swizzled
        # activation scales is the 32-wide one, and a wider rotation takes the
        # unfused route whose scales come out row-major. Those stay on Triton.
        # Gated on backend_present(), NOT on MACAQ_CUDNN: this list is an op
        # argument, so its length reaches the compiled graph, and vLLM's compile
        # cache does not key on our env vars. MACAQ_CUDNN switches the dispatch
        # inside the op instead, leaving the traced signature alone.
        layer.macaq_ws_blocked = []
        import cudnn_mxfp4_dense as _cudnn
        if _cudnn.backend_present() and self.b == 32:
            offs = layer.macaq_shard_offs
            layer.macaq_ws_blocked = [
                _cudnn.blocked_weight_scales(
                    layer.weight_scale[offs[s]:offs[s + 1]].contiguous())
                for s in range(len(offs) - 1)]

    def apply(self, layer, x, bias=None):
        out = torch.ops.vllm.macaq_dense_w4a4(
            x, layer.weight_packed, layer.weight_scale, layer.macaq_ws_blocked,
            layer.pre_scale, layer.macaq_bh, layer.macaq_shard_offs,
            self.out_scale)
        if bias is not None:
            out = out + bias
        return out


def _macaq_dense_w4a16_op(
    x: torch.Tensor, weight_packed: torch.Tensor, weight_scale: torch.Tensor,
    pre_scale: torch.Tensor, bh: torch.Tensor, shard_offs: list[int],
) -> torch.Tensor:
    from macaq_weight_only import mxfp4_w4a16_linear
    outputs = [mxfp4_w4a16_linear(
        x, weight_packed[lo:hi], weight_scale[lo:hi],
        pre_scale=pre_scale[i], BH=bh)
        for i, (lo, hi) in enumerate(zip(shard_offs, shard_offs[1:]))]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)


def _macaq_dense_w4a16_fake(
    x: torch.Tensor, weight_packed: torch.Tensor, weight_scale: torch.Tensor,
    pre_scale: torch.Tensor, bh: torch.Tensor, shard_offs: list[int],
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight_packed.shape[0]), dtype=torch.bfloat16)


if not hasattr(torch.ops.vllm, "macaq_dense_w4a16"):
    from vllm.utils.torch_utils import direct_register_custom_op
    direct_register_custom_op(
        "macaq_dense_w4a16", _macaq_dense_w4a16_op,
        mutates_args=[], fake_impl=_macaq_dense_w4a16_fake)


class _MacaqDenseW4A16Method(_MacaqDenseW4A4Method):
    """Same checkpoint loader, BF16 activations and temporary weight dequantization."""

    def process_weights_after_loading(self, layer):
        layer.macaq_bh = _make_macaq_bh(self.b, layer.weight_packed.device, torch.bfloat16)

    def apply(self, layer, x, bias=None):
        out = torch.ops.vllm.macaq_dense_w4a16(
            x, layer.weight_packed, layer.weight_scale, layer.pre_scale,
            layer.macaq_bh, layer.macaq_shard_offs)
        return out if bias is None else out + bias


# Route dense LinearBase layers to the macaq method when the checkpoint marks it.
if not getattr(_Mxfp4Config, "_macaq_macaq_dense", False):
    import sys as _sys
    _repo2 = _os.path.dirname(_os.path.abspath(__file__))
    if _repo2 not in _sys.path:
        _sys.path.insert(0, _repo2)

    _prev_from_config = _Mxfp4Config.from_config
    _prev_get_qm = _Mxfp4Config.get_quant_method

    @classmethod
    def _macaq_from_config(cls, config):
        cfg = _prev_from_config(config)          # prev is the bound classmethod
        cfg.macaq_dense_w4a4 = bool(config.get("macaq_dense_w4a4", False))
        cfg.hadamard_block = int(config.get("hadamard_block", 32))
        cfg.macaq_unquantized_suffixes = tuple(config.get("macaq_unquantized_suffixes", ()))
        cfg.macaq_activation_precision = config.get("macaq_activation_precision", "w4a4")
        if cfg.macaq_dense_w4a4 and cfg.macaq_activation_precision not in ("w4a4", "w4a16"):
            raise ValueError("MACAQ dense activation precision must be w4a4 or w4a16")
        return cfg

    def _macaq_get_qm(self, layer, prefix):
        if getattr(self, "macaq_dense_w4a4", False) and isinstance(layer, _LinearBase):
            if prefix.endswith(getattr(self, "macaq_unquantized_suffixes", ())):
                from vllm.model_executor.layers.linear import UnquantizedLinearMethod
                return UnquantizedLinearMethod()
            if not (self.ignored_layers and _is_layer_skipped(
                    prefix=prefix, ignored_layers=self.ignored_layers,
                    fused_mapping=self.packed_modules_mapping)):
                method = (_MacaqDenseW4A16Method if self.macaq_activation_precision == "w4a16"
                          else _MacaqDenseW4A4Method)
                return method(self.hadamard_block)
        return _prev_get_qm(self, layer, prefix)

    _Mxfp4Config.from_config = _macaq_from_config
    _Mxfp4Config.get_quant_method = _macaq_get_qm
    _Mxfp4Config._macaq_macaq_dense = True


def register():
    """Entry-point hook for `vllm.general_plugins`.

    Everything above ran at import time; vLLM calls this after
    `plugin.load()` and we don't need to do anything more. Idempotent.
    """
    return None

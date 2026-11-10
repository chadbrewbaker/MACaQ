"""Graph-compatible wrappers for the hybrid text runtime."""

import torch


def register_hybrid_ops():
    from vllm.utils.torch_utils import direct_register_custom_op
    from vllm.model_executor.layers.layernorm import RMSNormGated

    def gated_norm(x: torch.Tensor, weight: torch.Tensor,
                   bias: torch.Tensor | None, gate: torch.Tensor | None,
                   eps: float, group_size: int | None, before_gate: bool,
                   activation: str) -> torch.Tensor:
        from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
        return rmsnorm_fn(x, weight, bias, z=gate, eps=eps, group_size=group_size,
                          norm_before_gate=before_gate, activation=activation)

    def fake(x: torch.Tensor, weight: torch.Tensor,
             bias: torch.Tensor | None, gate: torch.Tensor | None,
             eps: float, group_size: int | None, before_gate: bool,
             activation: str) -> torch.Tensor:
        return torch.empty_like(x)

    if not hasattr(torch.ops.vllm, 'macaq_gated_norm'):
        direct_register_custom_op('macaq_gated_norm', gated_norm, mutates_args=[], fake_impl=fake)
    if not getattr(RMSNormGated, '_macaq_graph_wrapper', False):
        def forward(self, x, z=None):
            return torch.ops.vllm.macaq_gated_norm(
                x, self.weight, self.bias, z, self.eps, self.group_size,
                self.norm_before_gate, self.activation)
        RMSNormGated.forward_cuda = forward
        RMSNormGated._macaq_graph_wrapper = True

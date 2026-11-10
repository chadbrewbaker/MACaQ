"""Packed expert integration for the vLLM routed-expert API."""

import torch


def register():
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm.model_executor.utils import set_weight_attrs
    if getattr(Mxfp4Config, '_macaq_modern_registered', False):
        return
    from transformers import AutoConfig
    from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig
    from vllm import ModelRegistry
    AutoConfig.register('qwen4_exp_text', Qwen4ExpTextConfig, exist_ok=True)
    if 'Qwen4ExpForCausalLM' not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model('Qwen4ExpForCausalLM', 'vllm.models.qwen4_exp:Qwen4ExpForCausalLM')
    from importlib import import_module
    from vllm.models.qwen4_exp import Qwen4ExpForCausalLM
    # Follow the runtime's platform dispatch and keep its embedding storage
    # implementation. Only expert projections are packed by this adapter.
    embedding_module = import_module('.ngram_embedding',
        Qwen4ExpForCausalLM.__module__.rsplit('.', 1)[0])
    embedding_method = embedding_module.Qwen4ExpPLEEmbeddingMethod
    original_embedding = embedding_method.from_quant_config

    def embedding_from_config(config, prefix, embedding_dtype=None):
        if getattr(config, 'macaq_moe', False):
            return original_embedding(None, prefix, None)
        return original_embedding(config, prefix, embedding_dtype)

    embedding_method.from_quant_config = staticmethod(embedding_from_config)

    class MacaqMoEMethod(FusedMoEMethodBase):
        _macaq_moe = True
        _macaq_triton_w4a4 = True

        def maybe_roundup_sizes(self, hidden_size, intermediate_size_per_partition,
                               act_dtype, moe_parallel_config):
            if hidden_size % 32 or intermediate_size_per_partition % 32:
                raise ValueError('MACaQ expert dimensions must be divisible by 32')
            return hidden_size, intermediate_size_per_partition

        @property
        def skip_forward_padding(self):
            return True

        def create_weights(self, layer, num_experts, hidden_size,
                           intermediate_size_per_partition, params_dtype, **attrs):
            h, i, e = hidden_size, intermediate_size_per_partition, num_experts
            shapes = dict(w13_weight=(e, 2 * i, h // 2), w13_weight_scale=(e, 2 * i, h // 32),
                          w2_weight=(e, h, i // 2), w2_weight_scale=(e, h, i // 32),
                          w13_pre_scale=(e, h), w2_pre_scale=(e, i))
            for name, shape in shapes.items():
                dtype = torch.bfloat16 if name.endswith('pre_scale') else torch.uint8
                parameter = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
                set_weight_attrs(parameter, attrs)
                def load(param, value, *args, field=name, **kwargs):
                    from macaq_expert_layout import partition_expert_tensor
                    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
                    selected = partition_expert_tensor(field, value, param.shape,
                        get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
                    param.data.copy_(selected.to(param.device, param.dtype))
                    return True
                parameter.weight_loader = load
                layer.register_parameter(name, parameter)

        def get_fused_moe_quant_config(self, layer):
            return None

        def process_weights_after_loading(self, layer):
            expected = {'w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale',
                        'w13_pre_scale', 'w2_pre_scale'}
            if getattr(layer, '_macaq_loaded_fields', set()) != expected:
                raise ValueError('MACaQ checkpoint is missing packed expert tensors')
            from triton_mxfp4_dense import make_hadamard
            layer.macaq_bh = make_hadamard(32, layer.w13_weight.device).bfloat16()

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
            from triton_mxfp4_moe import fused_mxfp4_moe
            return fused_mxfp4_moe(x, layer.w13_weight, layer.w13_weight_scale,
                layer.w2_weight, layer.w2_weight_scale, topk_ids.int(), topk_weights.float(),
                layer.w13_pre_scale, layer.w2_pre_scale, layer.macaq_bh)

    original_from = Mxfp4Config.from_config.__func__
    original_method = Mxfp4Config.get_quant_method
    original_load = RoutedExperts.load_weights

    def load_weights(self, weights):
        if not getattr(self.quant_method, '_macaq_moe', False):
            yield from original_load(self, weights)
            return
        # Packed checkpoints already contain every expert in its final layout.
        # The standard loader maps individual source projections and does not
        # recognize these complete packed tensors.
        fields = {'w13_weight', 'w13_weight_scale', 'w2_weight', 'w2_weight_scale',
                  'w13_pre_scale', 'w2_pre_scale'}
        for name, tensor in weights:
            field = name.removeprefix('routed_experts.')
            if field not in fields:
                raise ValueError(f'unrecognized MACaQ expert tensor: {name}')
            parameter = getattr(self, field)
            parameter.weight_loader(parameter, tensor)
            if not hasattr(self, '_macaq_loaded_fields'):
                self._macaq_loaded_fields = set()
            self._macaq_loaded_fields.add(field)
            yield field

    @classmethod
    def from_config(cls, config):
        if config.get('macaq_dense_w4a4'):
            raise ValueError('MACaQ dense checkpoints require the documented vLLM 0.20.2 environment')
        result = original_from(cls, config)
        result.macaq_moe = bool(config.get('macaq_moe'))
        return result

    def get_quant_method(self, layer, prefix):
        if getattr(self, 'macaq_moe', False) and isinstance(layer, RoutedExperts):
            return MacaqMoEMethod(layer.moe_config)
        return original_method(self, layer, prefix)

    Mxfp4Config.from_config = from_config
    Mxfp4Config.get_quant_method = get_quant_method
    RoutedExperts.load_weights = load_weights
    Mxfp4Config._macaq_modern_registered = True

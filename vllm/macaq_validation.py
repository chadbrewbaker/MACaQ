"""Checkpoint validation for text serving workers."""
from collections import Counter
import os

if os.environ.get("MACAQ_VERIFY_WORKER") == "1":
    from macaq_plugin import register
    register()


class DenseValidationWorkerExtension:
    def macaq_inspect_methods(self, require_macaq=True, activation_precision=None):
        methods = Counter()
        n_macaq = 0
        n_moe = 0
        for module in self.model_runner.model.modules():
            method = getattr(module, "quant_method", None)
            if method is not None:
                methods[type(method).__name__] += 1
                n_macaq += type(method).__name__ in ('_MacaqDenseW4A4Method', '_MacaqDenseW4A16Method')
                if getattr(method, "_macaq_moe", False) and hasattr(module, "w13_pre_scale"):
                    if not getattr(method, "_macaq_triton_w4a4", False):
                        raise AssertionError("MoE checkpoint did not select the grouped W4A4 method")
                    n_moe += 1
        if require_macaq and not (n_macaq or n_moe):
            raise AssertionError("worker did not load a MACaQ quantization method")
        if n_moe and activation_precision not in (None, "w4a4"):
            raise AssertionError("the grouped MoE path requires W4A4 activations")
        if require_macaq and n_macaq and activation_precision is not None:
            expected = {'w4a4': '_MacaqDenseW4A4Method', 'w4a16': '_MacaqDenseW4A16Method'}
            if activation_precision not in expected or methods[expected[activation_precision]] != n_macaq:
                raise AssertionError(f"worker did not load the expected MACAQ {activation_precision} method")
        return dict(methods)

    def macaq_validate_saved_text(self, checkpoint):
        """Validate packed expert tensors, or delegate to dense validation."""
        from pathlib import Path
        import json
        import torch
        from safetensors import safe_open
        root = Path(checkpoint)
        if not json.loads((root / "config.json").read_text()).get("quantization_config", {}).get("macaq_moe"):
            return self.macaq_validate_saved_dense(checkpoint)
        index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
        handles = {}
        results = []
        try:
            for name, layer in self.model_runner.model.named_modules():
                if (not getattr(getattr(layer, "quant_method", None), "_macaq_moe", False)
                        or not hasattr(layer, "w13_pre_scale")):
                    continue
                for field in ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale",
                              "w13_pre_scale", "w2_pre_scale"):
                    key = name + "." + field
                    # Newer runtimes put the parameter owner below the runner;
                    # checkpoint names retain the public expert-module layout.
                    if key not in index:
                        key = key.replace(".experts.routed_experts.", ".experts.")
                    if key not in index:
                        raise AssertionError(f"saved expert tensor missing: {key}")
                    shard = index[key]
                    if shard not in handles:
                        handles[shard] = safe_open(str(root / shard), framework="pt")
                        handles[shard].__enter__()
                    expected = handles[shard].get_tensor(key)
                    actual = getattr(layer, field).detach().cpu()
                    if expected.shape != actual.shape:
                        from macaq_expert_layout import partition_expert_tensor
                        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
                        expected = partition_expert_tensor(field, expected, actual.shape,
                            get_tensor_model_parallel_rank(), get_tensor_model_parallel_world_size())
                    if actual.shape != expected.shape or not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                        raise AssertionError(f"loaded expert tensor differs: {key}")
                results.append(dict(layer=name, experts=layer.w13_weight.shape[0], exact=True))
        finally:
            for handle in handles.values():
                handle.__exit__(None, None, None)
        if not results:
            raise AssertionError("no packed experts verified")
        return results

    def macaq_validate_saved_dense(self, checkpoint):
        """Compare every loaded dense packed projection with the saved tensors."""
        from pathlib import Path
        import torch
        from safetensors import safe_open
        from vllm_macaq_mxfp4_plugin import _MacaqDenseW4A4Method, _MacaqDenseW4A16Method
        mappings = dict(qkv_proj=('q_proj','k_proj','v_proj'),
                        gate_up_proj=('gate_proj','up_proj'),
                        in_proj_qkvz=('in_proj_qkv','in_proj_z'))
        import re
        results = []
        with safe_open(str(Path(checkpoint)/'model.safetensors'),framework='pt') as source:
            keys = set(source.keys())
            for name, layer in self.model_runner.model.named_modules():
                if not isinstance(getattr(layer,'quant_method',None),_MacaqDenseW4A4Method):
                    continue
                match = re.search(r'(?:^|\.)layers\.(\d+)\.(.+)$',name)
                if not match:
                    raise ValueError(f'unrecognized dense layer {name}')
                index,tail = match.groups()
                parent,_,leaf = tail.rpartition('.')
                bases = [f'model.language_model.layers.{index}.{parent}.{part}'
                         for part in mappings.get(leaf,(leaf,))]
                if not all(base+'.weight_packed' in keys for base in bases):
                    bases = [base.replace('model.language_model.','model.',1) for base in bases]
                for field,actual in [('weight_packed',layer.weight_packed),('weight_scale',layer.weight_scale)]:
                    expected = torch.cat([source.get_tensor(base+'.'+field) for base in bases])
                    if not torch.equal(actual.cpu(),expected):
                        raise AssertionError(f'saved {field} differs at {name}')
                transforms = [source.get_tensor(base+'.pre_scale') for base in bases]
                if leaf == 'in_proj_qkvz':
                    transforms = [transforms[0]] * 3 + transforms[1:]
                unique = [transforms[0]]
                for previous,current in zip(transforms,transforms[1:]):
                    if isinstance(layer.quant_method, _MacaqDenseW4A16Method) or not torch.equal(previous,current):
                        unique.append(current)
                if not torch.equal(layer.pre_scale.cpu(),torch.stack(unique)):
                    raise AssertionError(f'saved input transform differs at {name}')
                results.append(dict(layer=name,packed_shape=list(layer.weight_packed.shape),
                                    transform_groups=len(unique),exact=True))
        if not results:
            raise AssertionError('no packed projections verified')
        return results

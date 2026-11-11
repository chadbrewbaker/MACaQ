"""MACaQ W4A4 for Diffusers transformers, with safetensors save and reload."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quantize"))
from checkpoint_config import portable_config, portable_model_source


def config_hash(transformer):
    config = portable_config(dict(transformer.config))
    for key in list(config):
        if key.startswith("_"):
            config.pop(key)
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()


def targets(transformer, family):
    import torch

    if family == "flux2-klein":
        excluded = ("norm", "modulation", "embed", "time_", "proj_out")
        def select(name, module):
            return (isinstance(module, torch.nn.Linear)
                    and any(block in name for block in (
                        "transformer_blocks", "single_transformer_blocks",
                        "double_blocks", "single_blocks"))
                    and not any(part in name.lower() for part in excluded))
        expected = "Flux2Transformer2DModel"
    elif family == "wan":
        select = lambda n, m: isinstance(m, torch.nn.Linear) and n.startswith("blocks.")
        expected = "WanTransformer3DModel"
    elif family == "stable-audio":
        select = lambda n, m: isinstance(m, torch.nn.Linear) and n.startswith("transformer_blocks.")
        expected = "StableAudioDiTModel"
    else:
        raise ValueError("family must be flux2-klein, wan or stable-audio")
    if type(transformer).__name__ != expected:
        raise ValueError(
            f"{family} requires {expected}, got {type(transformer).__name__}"
        )
    result = [(n, m) for n, m in transformer.named_modules() if select(n, m)]
    if not result:
        raise ValueError("No supported transformer linear layers found")
    if any(m.in_features % 256 for _, m in result):
        raise ValueError(
            "native W4A4 currently requires every input width divisible by 256"
        )
    return result


def replace(transformer, name, module):
    parent, _, leaf = name.rpartition(".")
    setattr(transformer.get_submodule(parent), leaf, module)


def quantize(transformer, run_calibration, family, flips=512, max_rows=1024, progress=None,
             keep_layers=()):
    import fnmatch
    import torch
    from macaq_mxfp4 import mxfp4_realquant_h, OCPDenseW4A4Linear

    if flips < 1 or max_rows < 1:
        raise ValueError("flips and max_rows must be positive")
    inventory = targets(transformer, family)
    for pattern in keep_layers:
        if not any(fnmatch.fnmatchcase(name, pattern) for name, _ in inventory):
            raise ValueError(f"keep-layers pattern matches no selected layer: {pattern}")
    retained = [name for name, _ in inventory
                if any(fnmatch.fnmatchcase(name, pattern) for pattern in keep_layers)]
    inventory = [(name, module) for name, module in inventory if name not in retained]
    if not inventory:
        raise ValueError("keep-layers must leave at least one quantized layer")
    stats, counts, handles = {}, {}, []

    def hook(name):
        def capture(module, inputs):
            x = inputs[0].detach().reshape(-1, module.in_features)
            if not torch.isfinite(x).all():
                raise ValueError(f"nonfinite calibration input at {name}")
            n = min(len(x), max_rows)
            ix = torch.linspace(0, len(x) - 1, n, device=x.device).long()
            x = x[ix].float()
            h = x.T @ x
            stats[name] = h if name not in stats else stats[name] + h
            counts[name] = counts.get(name, 0) + n

        return capture

    try:
        for name, module in inventory:
            handles.append(module.register_forward_pre_hook(hook(name)))
        with torch.inference_mode():
            run_calibration()
    finally:
        for handle in handles:
            handle.remove()
    if any(name not in stats for name, _ in inventory):
        raise ValueError("calibration did not cover every selected linear")
    with torch.no_grad():
        for index, (name, module) in enumerate(inventory):
            if progress is not None:
                progress(index, len(inventory), name)
            art = mxfp4_realquant_h(
                module.weight,
                stats.pop(name),
                counts[name],
                had_block=32,
                n_flip_iters=flips,
                ridge=1.0,
                do_flip=True,
                pre_mode="balance",
                pre_alpha=0.75,
                exp_screen=True,
                exp_offsets=(1, 0, -1),
                fast_flip=True,
                compile_flip=False,
            )
            packed = OCPDenseW4A4Linear(
                art["W_packed"],
                art["W_scales"],
                art["BH"],
                module.bias,
                32,
                module.out_features,
                module.in_features,
                pre_scale=art.get("pre_scale"),
            )
            replace(transformer, name, packed)
    return dict(
        family=family,
        precision="w4a4",
        retained_source_layers=retained,
        vq=False,
        flips=flips,
        max_rows_per_call=max_rows,
        rows=counts,
        hadamard_block=32,
    )


def save_checkpoint(transformer, output, recipe):
    import torch
    from safetensors.torch import save_file
    from macaq_mxfp4 import OCPDenseW4A4Linear

    recipe = portable_config(recipe)
    if isinstance(recipe.get('source'), dict):
        recipe['source'] = portable_model_source(recipe['source'])

    output = Path(output)
    if output.exists():
        raise ValueError("choose a fresh checkpoint directory")
    layers = {
        n: dict(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=m.bias is not None,
        )
        for n, m in transformer.named_modules()
        if isinstance(m, OCPDenseW4A4Linear)
    }
    if not layers:
        raise ValueError("no MACaQ linear layers to save")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        pending = Path(directory) / "checkpoint"
        pending.mkdir()
        tensors = {
            k: v.detach().cpu().contiguous().clone()
            for k, v in transformer.state_dict().items()
        }
        # Wan's RoPE tables are nonpersistent buffers. Rebuilding them at the
        # constructor's default precision can change attention after reload.
        nonpersistent = {}
        for name, value in transformer.named_buffers():
            if name not in tensors:
                # Runtime caches of derived FP8 weight scales are rebuilt when
                # selecting the activation policy; they are not source buffers.
                if any(name.startswith(layer + ".") for layer in layers):
                    continue
                key = "__macaq_nonpersistent__." + name
                tensors[key] = value.detach().cpu().contiguous().clone()
                nonpersistent[name] = dict(shape=list(value.shape), dtype=str(value.dtype))
        for name, spec in layers.items():
            if name + ".pre_scale" not in tensors:
                tensors[name + ".pre_scale"] = torch.ones(
                    spec["in_features"], dtype=torch.bfloat16
                )
        save_file(tensors, pending / "model.safetensors")
        manifest = dict(
            format="macaq-diffusers-v1",
            config_sha256=config_hash(transformer),
            transformer_class=type(transformer).__name__,
            transformer_config=portable_config(dict(transformer.config)),
            layers=layers,
            nonpersistent_buffers=nonpersistent,
            recipe=recipe,
            weights_sha256=file_hash(pending / "model.safetensors"),
        )
        (pending / "macaq.json").write_text(json.dumps(manifest, indent=2) + "\n")
        pending.rename(output)
    return manifest


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(transformer, checkpoint):
    import torch
    from safetensors.torch import load_file
    from macaq_mxfp4 import OCPDenseW4A4Linear

    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "macaq.json").read_text())
    if manifest["format"] != "macaq-diffusers-v1" or manifest[
        "config_sha256"
    ] != config_hash(transformer):
        raise ValueError("checkpoint format or transformer config mismatch")
    if manifest["transformer_class"] != type(transformer).__name__:
        raise ValueError("transformer class mismatch")
    if file_hash(checkpoint / "model.safetensors") != manifest["weights_sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    state = load_file(checkpoint / "model.safetensors")
    restored_buffers = {}
    buffers = dict(transformer.named_buffers())
    for name, spec in manifest.get("nonpersistent_buffers", {}).items():
        value = state.pop("__macaq_nonpersistent__." + name, None)
        if (name not in buffers or value is None or list(value.shape) != spec["shape"]
                or tuple(value.shape) != tuple(buffers[name].shape) or str(value.dtype) != spec["dtype"]):
            raise ValueError(f"invalid nonpersistent buffer {name}")
        restored_buffers[name] = value
    if not manifest["layers"]:
        raise ValueError("checkpoint contains no quantized layers")
    for name, spec in manifest["layers"].items():
        rows, cols = spec["out_features"], spec["in_features"]
        expected = {
            "W_packed": ((rows, cols // 2), torch.uint8),
            "W_scales": ((rows, cols // 32), torch.uint8),
            "B_inv_T": ((32, 32), torch.bfloat16),
            "pre_scale": ((cols,), torch.bfloat16),
        }
        if cols % 256:
            raise ValueError(f"unsupported native input width at {name}")
        if spec["bias"]:
            expected["bias"] = ((rows,), torch.bfloat16)
        for key, (shape, dtype) in expected.items():
            tensor = state.get(name + "." + key)
            if tensor is None or tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"invalid packed tensor {name}.{key}")
    for name, spec in manifest["layers"].items():
        old = transformer.get_submodule(name)
        if not isinstance(old, torch.nn.Linear) or (
            old.in_features,
            old.out_features,
        ) != (spec["in_features"], spec["out_features"]):
            raise ValueError(f"linear architecture mismatch at {name}")
    for name, spec in manifest["layers"].items():
        get = lambda key: state.get(name + "." + key)
        packed = OCPDenseW4A4Linear(
            get("W_packed"),
            get("W_scales"),
            get("B_inv_T"),
            get("bias"),
            32,
            spec["out_features"],
            spec["in_features"],
            pre_scale=get("pre_scale"),
        )
        replace(transformer, name, packed)
    transformer.load_state_dict(state, strict=True, assign=True)
    for name, value in restored_buffers.items():
        parent, _, leaf = name.rpartition(".")
        transformer.get_submodule(parent)._buffers[leaf] = value
    return manifest


def load_transformer(checkpoint):
    """Build only from config, restoring saved tensors without original DiT weights."""
    from accelerate import init_empty_weights
    from diffusers import Flux2Transformer2DModel, WanTransformer3DModel, StableAudioDiTModel

    manifest = json.loads((Path(checkpoint) / "macaq.json").read_text())
    classes = {
        "Flux2Transformer2DModel": Flux2Transformer2DModel,
        "WanTransformer3DModel": WanTransformer3DModel,
        "StableAudioDiTModel": StableAudioDiTModel,
    }
    if (
        manifest.get("transformer_class") not in classes
        or "transformer_config" not in manifest
    ):
        raise ValueError(
            "checkpoint needs a supported transformer class and saved config"
        )
    with init_empty_weights():
        transformer = classes[manifest["transformer_class"]].from_config(
            manifest["transformer_config"]
        )
    load_checkpoint(transformer, checkpoint)
    return transformer.eval()

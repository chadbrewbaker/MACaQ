"""Quantize Qwen3-TTS CustomVoice and generate paired speech samples with PyTorch."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quantize"))
TARGET = re.compile(r"^talker(?:\.code_predictor)?\.model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)$")
GENERATION = dict(do_sample=True, top_k=50, top_p=1.0, temperature=0.9,
                  repetition_penalty=1.05, subtalker_dosample=True,
                  subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    pending.replace(path)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_prompts(path):
    import unicodedata
    normalize = lambda s: " ".join(unicodedata.normalize("NFKC", s).casefold().split())
    data = json.loads(path.read_text())
    sets = []
    for split in ("calibration", "evaluation"):
        if not data.get(split):
            raise ValueError(f"empty {split} split")
        ids = set()
        for row in data[split]:
            for key in ("id", "text", "language", "speaker", "instruct", "seed"):
                if key not in row:
                    raise ValueError(f"missing prompt field {key}")
            source = row.get("source", {})
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("prompt text must be nonempty")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", row["id"]) or row["id"] in ids:
                raise ValueError("prompt IDs must be unique safe filenames")
            ids.add(row["id"])
            if not isinstance(row["seed"], int) or not 0 <= row["seed"] < 2**32:
                raise ValueError("prompt seed must be an integer in [0, 2**32)")
            if not isinstance(source, dict):
                raise ValueError("prompt source must be a provenance object")
        sets.append({normalize(row["text"]) for row in data[split]})
    if not sets[0].isdisjoint(sets[1]):
        raise ValueError("calibration and evaluation text overlap")
    return data


def target_layers(model):
    import torch
    targets = {name: module for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and TARGET.fullmatch(name)}
    if not targets or any(layer.in_features % 256 for layer in targets.values()):
        raise ValueError("unsupported TTS projection inventory or input width")
    return targets


def group_name(name):
    name = re.sub(r"\.self_attn\.[kv]_proj$", ".self_attn.q_proj", name)
    return re.sub(r"\.mlp\.up_proj$", ".mlp.gate_proj", name)


def replace(model, name, value):
    parent, _, leaf = name.rpartition(".")
    setattr(model.get_submodule(parent), leaf, value)


def synthesize(tts, prompt, max_tokens):
    seed_all(prompt["seed"])
    return tts.generate_custom_voice(
        text=prompt["text"], language=prompt["language"], speaker=prompt["speaker"],
        instruct=prompt["instruct"], **GENERATION, max_new_tokens=max_tokens)


def calibrate(tts, prompts, targets, directory, rows_per_prompt):
    import torch
    from safetensors.torch import save_file
    representatives = {group_name(name): targets[group_name(name)] for name in targets}
    chunks = {name: [] for name in representatives}
    state = {}
    handles = []

    def hook(name):
        def capture(module, args):
            item = state[name]
            item["calls"] += 1
            stride = 7 if ".code_predictor." in name else 1
            if (item["calls"] - 1) % stride or item["rows"] >= rows_per_prompt:
                return
            x = args[0].detach().reshape(-1, module.in_features)
            n = min(len(x), rows_per_prompt - item["rows"])
            index = torch.linspace(0, len(x) - 1, n, device=x.device).long()
            item["parts"].append(x[index].clone())
            item["rows"] += n
        return capture

    for name, layer in representatives.items():
        handles.append(layer.register_forward_pre_hook(hook(name)))
    try:
        with torch.inference_mode():
            for i, prompt in enumerate(prompts):
                state.clear()
                state.update({name: dict(calls=0, rows=0, parts=[]) for name in representatives})
                synthesize(tts, prompt, 256)
                for name, item in state.items():
                    if not item["rows"]:
                        raise ValueError(f"uncovered projection {name}")
                    rows = torch.cat(item["parts"])
                    if not torch.isfinite(rows).all():
                        raise ValueError(f"nonfinite activation at {name}")
                    chunks[name].append(rows.cpu())
                print(f"Calibration {i + 1}/{len(prompts)}: {prompt['id']}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    directory.mkdir()
    counts = {}
    for name, parts in chunks.items():
        rows = torch.cat(parts).contiguous()
        save_file({"samples": rows}, directory / (name + ".safetensors"))
        counts[name] = len(rows)
    return counts


def save_checkpoint(tts, source, output, recipe):
    from safetensors.torch import save_file
    from macaq_mxfp4 import OCPDenseW4A4Linear
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        pending = Path(temp) / "checkpoint"
        pending.mkdir()
        # Carry tokenizer/codec assets so reload does not need the original
        # full-precision talker checkpoint or network access.
        for name in ("config.json", "generation_config.json", "preprocessor_config.json",
                     "tokenizer_config.json", "vocab.json", "merges.txt", "tokenizer.json",
                     "added_tokens.json", "special_tokens_map.json"):
            entry = source / name
            if entry.is_file():
                shutil.copyfile(entry, pending / entry.name)
        shutil.copytree(source / "speech_tokenizer", pending / "speech_tokenizer")
        model = tts.model
        state = {key: value.detach().cpu().contiguous().clone()
                 for key, value in model.state_dict().items()}
        buffers = {}
        for name, value in model.named_buffers():
            if name not in state:
                state["__macaq_buffers__." + name] = value.detach().cpu().contiguous().clone()
                buffers[name] = dict(shape=list(value.shape), dtype=str(value.dtype))
        layers = {name: dict(in_features=m.in_features, out_features=m.out_features,
                             bias=m.bias is not None)
                  for name, m in model.named_modules() if isinstance(m, OCPDenseW4A4Linear)}
        save_file(state, pending / "model.safetensors")
        manifest = dict(format="macaq-qwen3-tts-v1", config=model.config.to_dict(),
                        layers=layers, nonpersistent_buffers=buffers, recipe=recipe,
                        assets={str(p.relative_to(pending)): sha(p)
                                for p in pending.rglob("*") if p.is_file()})
        write_json(pending / "macaq.json", manifest)
        pending.rename(output)
    return manifest


def check_assets(path, assets):
    for name, expected in assets.items():
        target = (path / name).resolve()
        if not target.is_relative_to(path.resolve()):
            raise ValueError("checkpoint asset escapes its directory")
        if sha(target) != expected:
            raise ValueError(f"checkpoint checksum mismatch: {name}")


def load_checkpoint(path):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
    from macaq_mxfp4 import OCPDenseW4A4Linear

    manifest = json.loads((path / "macaq.json").read_text())
    if manifest["format"] != "macaq-qwen3-tts-v1":
        raise ValueError("unsupported TTS checkpoint format")
    check_assets(path, manifest["assets"])
    # qwen-tts 0.1.1 adds runtime dtype fields to the speaker config whose
    # constructor does not accept them. Reuse the hashed source configuration.
    config = Qwen3TTSConfig.from_pretrained(str(path), local_files_only=True)
    config._attn_implementation = "sdpa"
    config.talker_config._attn_implementation = "sdpa"
    config.talker_config.code_predictor_config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = Qwen3TTSForConditionalGeneration(config)
    if set(manifest["layers"]) != set(target_layers(model)):
        raise ValueError("checkpoint does not cover the supported speech projections")
    state = load_file(path / "model.safetensors")
    for name, spec in manifest["layers"].items():
        module = model.get_submodule(name)
        if not TARGET.fullmatch(name) or (module.in_features, module.out_features) != (spec["in_features"], spec["out_features"]):
            raise ValueError(f"projection mismatch: {name}")
        get = lambda key: state.get(name + "." + key)
        n, k = spec["out_features"], spec["in_features"]
        expected = dict(W_packed=((n, k // 2), torch.uint8),
                        W_scales=((n, k // 32), torch.uint8),
                        B_inv_T=((32, 32), torch.bfloat16), pre_scale=((k,), torch.bfloat16))
        if spec["bias"]:
            expected["bias"] = ((n,), torch.bfloat16)
        for key, (shape, dtype) in expected.items():
            value = get(key)
            if value is None or tuple(value.shape) != shape or value.dtype != dtype:
                raise ValueError(f"invalid packed projection {name}.{key}")
        replace(model, name, OCPDenseW4A4Linear(
            get("W_packed"), get("W_scales"), get("B_inv_T"), get("bias"),
            32, spec["out_features"], spec["in_features"], pre_scale=get("pre_scale")))
    buffers = {name: state.pop("__macaq_buffers__." + name)
               for name in manifest["nonpersistent_buffers"]}
    original_buffers = dict(model.named_buffers())
    for name, value in buffers.items():
        spec = manifest["nonpersistent_buffers"][name]
        if (name not in original_buffers or tuple(value.shape) != tuple(original_buffers[name].shape)
                or list(value.shape) != spec["shape"] or str(value.dtype) != spec["dtype"]):
            raise ValueError(f"invalid source buffer {name}")
    model.load_state_dict(state, strict=True, assign=True)
    for name, value in buffers.items():
        parent, _, leaf = name.rpartition(".")
        model.get_submodule(parent)._buffers[leaf] = value
    model = model.eval().to("cuda")
    model.load_speech_tokenizer(Qwen3TTSTokenizer.from_pretrained(
        str(path / "speech_tokenizer"), device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True))
    defaults = json.loads((path / "generation_config.json").read_text())
    model.load_generate_config(defaults)
    processor = Qwen3TTSProcessor.from_pretrained(str(path), local_files_only=True, fix_mistral_regex=True)
    return Qwen3TTSModel(model, processor, defaults)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    quant = sub.add_parser("quantize")
    quant.add_argument("--model", type=Path, required=True)
    quant.add_argument("--work-dir", type=Path, required=True)
    quant.add_argument("--flips", type=int, default=512)
    quant.add_argument("--rows-per-prompt", type=int, default=160)
    generate = sub.add_parser("generate")
    choice = generate.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", type=Path)
    choice.add_argument("--checkpoint", type=Path)
    generate.add_argument("--kernel", choices=("reference", "fused"), default="fused")
    generate.add_argument("--runtime", choices=("eager", "fast"), default="fast",
                          help="fast uses speech kernels and graph replay; eager keeps the original runtime")
    generate.add_argument("--max-tokens", type=int, default=384)
    generate.add_argument("--limit", type=int)
    for p in (quant, generate):
        p.add_argument("--prompts", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.gpu < 0:
        parser.error("use a fresh output and a valid GPU")
    if args.action == "quantize" and (args.work_dir.exists() or min(args.flips, args.rows_per_prompt) < 1):
        parser.error("use a fresh work directory and positive effort settings")
    if args.action == "generate" and (args.max_tokens < 1 or (args.limit is not None and args.limit < 1)):
        parser.error("generation counts must be positive")
    prompts = read_prompts(args.prompts)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        parser.error("W4A4 execution is unavailable in this environment")
    tts = (load_checkpoint(args.checkpoint) if args.action == "generate" and args.checkpoint
           else Qwen3TTSModel.from_pretrained(str(args.model), device_map="cuda:0",
                dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True))
    tts.model.eval()
    if tts.model.tts_model_type != "custom_voice":
        raise ValueError("this example supports Qwen3-TTS CustomVoice models")
    if args.action == "quantize":
        from safetensors.torch import load_file
        from macaq_mxfp4 import mxfp4_realquant_h, OCPDenseW4A4Linear
        args.work_dir.mkdir(parents=True)
        targets = target_layers(tts.model)
        print(f"Quantizing {len(targets)} speech projections", flush=True)
        counts = calibrate(tts, prompts["calibration"], targets,
                           args.work_dir / "calibration", args.rows_per_prompt)
        groups = {}
        for name in targets:
            groups.setdefault(group_name(name), []).append(name)
        with torch.inference_mode():
            index = 0
            for representative, names in groups.items():
                x = load_file(args.work_dir / "calibration" / (representative + ".safetensors"))["samples"].cuda()
                h = x.float().T @ x.float()
                for name in names:
                    old = targets.pop(name)
                    art = mxfp4_realquant_h(old.weight.float(), h, len(x), had_block=32,
                        n_flip_iters=args.flips, ridge=1.0, pre_mode="balance", pre_alpha=0.75,
                        do_flip=True, exp_screen=True, exp_offsets=(1, 0, -1),
                        fast_flip=True, compile_flip=False)
                    replace(tts.model, name, OCPDenseW4A4Linear(
                        art["W_packed"], art["W_scales"], art["BH"], old.bias,
                        32, old.out_features, old.in_features, pre_scale=art.get("pre_scale")))
                    del old, art
                    index += 1
                    print(f"Quantized {index}: {name}", flush=True)
                del x, h
        recipe = dict(source=str(args.model.resolve()), source_weights_sha256=sha(args.model / "model.safetensors"),
                      prompts_sha256=sha(args.prompts), calibration=prompts["calibration"],
                      precision="w4a4", flips=args.flips, rows_per_prompt=args.rows_per_prompt,
                      rows=counts, generation=GENERATION, calibration_max_tokens=256)
        manifest = save_checkpoint(tts, args.model, args.output, recipe)
        print(f"Saved {len(manifest['layers'])} projections to {args.output}", flush=True)
    else:
        if args.checkpoint:
            sys.path.insert(0, str(ROOT / "diffusers"))
            from kernels import set_kernel
            set_kernel(tts.model, args.kernel)
        if args.runtime == "fast":
            from tts_runtime import configure_fast_runtime
            configure_fast_runtime(tts.model, packed=bool(args.checkpoint and args.kernel == "fused"))
        args.output.mkdir(parents=True)
        report = dict(runtime=args.runtime, max_tokens=args.max_tokens,
                      model=str(args.model or args.checkpoint), prompts_sha256=sha(args.prompts),
                      mode="w4a4" if args.checkpoint else "bf16", generation=GENERATION,
                      kernel=args.kernel if args.checkpoint else None, samples=[])
        with torch.inference_mode():
            for prompt in prompts["evaluation"][:args.limit]:
                counts = []
                original = tts.model.generate
                def capture(*a, **kw):
                    output = original(*a, **kw)
                    counts.extend(len(c) for c in output[0])
                    return output
                tts.model.generate = capture
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                try:
                    waves, rate = synthesize(tts, prompt, args.max_tokens)
                finally:
                    tts.model.generate = original
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                wave = np.asarray(waves[0], dtype=np.float32)
                if not len(wave) or not np.isfinite(wave).all():
                    raise ValueError("invalid generated speech")
                path = args.output / (prompt["id"] + ".wav")
                sf.write(path, wave, rate, subtype="PCM_16")
                item = dict(prompt, file=path.name, sha256=sha(path), sample_rate=rate,
                            duration_seconds=len(wave) / rate, generation_seconds=elapsed,
                            codec_tokens=counts[0], reached_limit=counts[0] >= args.max_tokens - 1,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            rms=float(np.sqrt(np.mean(wave**2))))
                report["samples"].append(item)
                write_json(args.output / "report.json", report)
                print(f"{report['mode']} {prompt['id']}: {len(wave)/rate:.2f}s audio, {elapsed:.2f}s generation, capped={item['reached_limit']}", flush=True)


if __name__ == "__main__":
    main()

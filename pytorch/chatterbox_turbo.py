"""Quantize, save and run Chatterbox Turbo speech with MACaQ."""

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

os.environ.setdefault("TQDM_DISABLE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "quantize"))
from qwen3_tts import sha, write_json, seed_all, replace, check_assets

MODEL = "ResembleAI/chatterbox-turbo"
REVISION = "749d1c1a46eb10492095d68fbcf55691ccf137cd"
TARGET = re.compile(r"^tfmr\.h\.\d+\.(?:attn\.(?:c_attn|c_proj)|mlp\.(?:c_fc|c_proj))$")
ASSETS = ("s3gen_meanflow.safetensors", "ve.safetensors", "conds.pt",
          "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
          "vocab.json", "merges.txt")


def read_prompts(path):
    from qwen3_tts import read_prompts as read
    result = read(path)
    if any(row["language"] != "English" for split in ("calibration", "evaluation") for row in result[split]):
        raise ValueError("Chatterbox Turbo supports English prompts")
    return result


def projections(model):
    from transformers.pytorch_utils import Conv1D
    result = {n: m for n, m in model.named_modules() if TARGET.fullmatch(n)}
    if len(result) != 4 * len(model.tfmr.h) or any(not isinstance(m, Conv1D) for m in result.values()):
        raise ValueError("unsupported Chatterbox projection layout")
    if any(m.weight.shape[0] % 256 for m in result.values()):
        raise ValueError("unsupported projection input width")
    return result


def load_source(path, device="cuda"):
    import torch
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    tts = ChatterboxTurboTTS.from_local(path, device)
    if tts.conds is None:
        raise ValueError("the source checkpoint must include its built-in voice conditionals")
    tts.t3.to(torch.bfloat16).eval()
    tts.conds.t3.to(dtype=torch.bfloat16)
    tts.t3.tfmr.set_attn_implementation("sdpa")
    return monitor_generation(tts)


def monitor_generation(tts):
    original = tts.t3.inference_turbo
    @wraps(original)
    def inference(*args, **kwargs):
        tokens = original(*args, **kwargs)
        tts._macaq_last_frames = tokens.numel()
        tts._macaq_last_capped = tokens.numel() > kwargs.get("max_gen_len", 1000)
        return tokens
    tts.t3.inference_turbo = inference
    return tts


def source_identity(path):
    from source_provenance import digest
    files = {name: sha(path / name) for name in (*ASSETS, "t3_turbo_v1.safetensors")}
    return dict(files=files, digest=digest(files))


def save_checkpoint(tts, source, output, recipe):
    from safetensors.torch import save_file
    from macaq_mxfp4 import OCPDenseW4A4Linear
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("choose a fresh checkpoint directory")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        pending = Path(tmp) / "checkpoint"
        pending.mkdir()
        for name in ASSETS:
            shutil.copyfile(source / name, pending / name)
        state = {k: v.detach().cpu().contiguous().clone() for k, v in tts.t3.state_dict().items()}
        save_file(state, pending / "t3.safetensors")
        layers = {name: dict(in_features=m.in_features, out_features=m.out_features)
                  for name, m in tts.t3.named_modules() if isinstance(m, OCPDenseW4A4Linear)}
        manifest = dict(format="macaq-chatterbox-turbo-v1", layers=layers,
                        recipe=recipe, source=source_identity(source),
                        assets={p.name: sha(p) for p in pending.iterdir()})
        write_json(pending / "macaq.json", manifest)
        pending.rename(output)
    return manifest


def load_checkpoint(path, device="cuda"):
    import torch
    from accelerate import init_empty_weights
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
    from chatterbox.models.t3 import T3
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.voice_encoder import VoiceEncoder
    from chatterbox.models.s3gen import S3Gen
    from macaq_mxfp4 import OCPDenseW4A4Linear
    manifest = json.loads((path / "macaq.json").read_text())
    if manifest["format"] != "macaq-chatterbox-turbo-v1":
        raise ValueError("unsupported speech checkpoint")
    if set(manifest["assets"]) != {*ASSETS, "t3.safetensors"}:
        raise ValueError("checkpoint asset inventory mismatch")
    check_assets(path, manifest["assets"])
    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = "GPT2_medium"
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = 375
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    with init_empty_weights():
        t3 = T3(hp)
    del t3.tfmr.wte
    expected = projections(t3)
    if set(expected) != set(manifest["layers"]):
        raise ValueError("checkpoint projection inventory mismatch")
    state = load_file(path / "t3.safetensors")
    for name, spec in manifest["layers"].items():
        k, n = expected[name].weight.shape
        if spec != dict(in_features=k, out_features=n):
            raise ValueError("checkpoint projection dimensions mismatch")
        fields = {key: state.get(name + "." + key) for key in
                  ("W_packed", "W_scales", "B_inv_T", "bias", "pre_scale")}
        shapes = dict(W_packed=((n, k // 2), torch.uint8), W_scales=((n, k // 32), torch.uint8),
                      B_inv_T=((32, 32), torch.bfloat16), bias=((n,), torch.bfloat16),
                      pre_scale=((k,), torch.bfloat16))
        for key, (shape, dtype) in shapes.items():
            v = fields[key]
            if v is None or tuple(v.shape) != shape or v.dtype != dtype:
                raise ValueError(f"invalid packed projection {name}.{key}")
        replace(t3, name, OCPDenseW4A4Linear(fields["W_packed"], fields["W_scales"],
            fields["B_inv_T"], fields["bias"], 32, n, k, pre_scale=fields["pre_scale"]))
    t3.load_state_dict(state, strict=True, assign=True)
    t3 = t3.eval().to(device)
    t3.tfmr.set_attn_implementation("sdpa")
    ve = VoiceEncoder()
    ve.load_state_dict(load_file(path / "ve.safetensors"))
    s3gen = S3Gen(meanflow=True)
    s3gen.load_state_dict(load_file(path / "s3gen_meanflow.safetensors"), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    conds = Conditionals.load(path / "conds.pt", map_location="cpu").to(device)
    conds.t3.to(dtype=torch.bfloat16)
    return monitor_generation(ChatterboxTurboTTS(
        t3, s3gen.eval().to(device), ve.eval().to(device), tokenizer, device, conds))


def synthesize(tts, prompt):
    import torch
    seed_all(prompt["seed"])
    # Keep the codec at its original precision for both controls. Only the
    # autoregressive speech generator uses BF16 or packed MACaQ projections.
    with torch.inference_mode():
        wave = tts.generate(prompt["text"])
        if getattr(tts, "_macaq_last_capped", False):
            raise ValueError("speech generation reached its length limit; split the text into shorter passages")
        return wave


def quantize(tts, prompts, source, output, work, rows_per_prompt, flips):
    import torch
    from safetensors.torch import save_file, load_file
    from macaq_mxfp4 import OCPDenseW4A4Linear, mxfp4_realquant_h
    targets = projections(tts.t3)
    chunks = {name: [] for name in targets}
    handles = []
    counts = {}
    def hook(name):
        def capture(module, args):
            if counts.get(name, 0) >= rows_per_prompt:
                return
            x = args[0].detach().reshape(-1, module.weight.shape[0])
            n = min(len(x), rows_per_prompt - counts.get(name, 0))
            idx = torch.linspace(0, len(x) - 1, n, device=x.device).long()
            chunks[name].append(x[idx].cpu())
            counts[name] = counts.get(name, 0) + n
        return capture
    work.mkdir(parents=True)
    for name, module in targets.items():
        handles.append(module.register_forward_pre_hook(hook(name)))
    try:
        for i, row in enumerate(prompts):
            counts.clear()
            synthesize(tts, row)
            if set(counts) != set(targets):
                raise ValueError("calibration did not cover every speech projection")
            print(f"Calibration {i + 1}/{len(prompts)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    coverage = {}
    for name, parts in chunks.items():
        x = torch.cat(parts).contiguous()
        if not torch.isfinite(x).all():
            raise ValueError("nonfinite calibration activation")
        coverage[name] = len(x)
        save_file({"samples": x}, work / (name + ".safetensors"))
    del chunks
    with torch.inference_mode():
        for i, (name, module) in enumerate(targets.items()):
            x = load_file(work / (name + ".safetensors"))["samples"].cuda()
            h = x.float().T @ x.float()
            k, n = module.weight.shape
            art = mxfp4_realquant_h(module.weight.T.float().contiguous(), h, len(x),
                had_block=32, n_flip_iters=flips, ridge=1.0, pre_mode="balance", pre_alpha=0.75,
                do_flip=True, exp_screen=True, exp_offsets=(1, 0, -1), fast_flip=True, compile_flip=False)
            replace(tts.t3, name, OCPDenseW4A4Linear(art["W_packed"], art["W_scales"], art["BH"],
                module.bias, 32, n, k, pre_scale=art["pre_scale"]))
            print(f"Quantized {i + 1}/{len(targets)}: {name}", flush=True)
    return save_checkpoint(tts, source, output, dict(precision="w4a4", flips=flips,
        rows_per_prompt=rows_per_prompt, calibration_rows=coverage,
        calibration_prompts_sha256=__import__("source_provenance").digest(prompts),
        decoder_precision="float32", backbone_precision="bfloat16"))


def configure_runtime(tts, runtime):
    sys.path.insert(0, str(ROOT / "diffusers"))
    from kernels import set_kernel
    if runtime == "fast":
        from tts_kernels import use_speech_kernels
        set_kernel(tts.t3, "fused")
        use_speech_kernels(tts.t3)
        from chatterbox_runtime import use_decoder_graphs
        use_decoder_graphs(tts.t3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    d = sub.add_parser("download")
    d.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--model", type=Path, required=True)
    q.add_argument("--work-dir", type=Path, required=True)
    q.add_argument("--flips", type=int, default=256)
    q.add_argument("--rows-per-prompt", type=int, default=128)
    g = sub.add_parser("generate")
    choice = g.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", type=Path)
    choice.add_argument("--checkpoint", type=Path)
    g.add_argument("--runtime", choices=("eager", "fast"), default="fast")
    g.add_argument("--limit", type=int)
    for parser in (q, g):
        parser.add_argument("--prompts", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    if a.output.exists():
        p.error("choose a fresh output directory")
    if a.action == "download":
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL, revision=REVISION, local_dir=a.output,
                          allow_patterns=[*ASSETS, "t3_turbo_v1.safetensors", "LICENSE*"])
        return
    if a.gpu < 0 or (a.action == "quantize" and (a.work_dir.exists() or min(a.flips, a.rows_per_prompt) < 1)):
        p.error("choose a fresh work directory and positive settings")
    if a.action == "generate" and a.limit is not None and a.limit < 1:
        p.error("limit must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    sys.path.insert(0, str(ROOT / "diffusers"))
    import soundfile as sf
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    prompts = read_prompts(a.prompts)
    tts = load_checkpoint(a.checkpoint) if a.action == "generate" and a.checkpoint else load_source(a.model)
    if a.action == "quantize":
        quantize(tts, prompts["calibration"], a.model, a.output, a.work_dir, a.rows_per_prompt, a.flips)
        return
    configure_runtime(tts, a.runtime)
    a.output.mkdir(parents=True)
    report = dict(model=MODEL, mode="w4a4" if a.checkpoint else "bf16", runtime=a.runtime,
                  prompts_sha256=sha(a.prompts), source=(json.loads((a.checkpoint / "macaq.json").read_text())["source"]
                  if a.checkpoint else source_identity(a.model)), samples=[])
    # Warm the full path before recording samples; loading and compilation are
    # excluded from subsequent generation timings.
    synthesize(tts, prompts["evaluation"][0])
    for row in prompts["evaluation"][:a.limit]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        wav = synthesize(tts, row).squeeze().float()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if wav.ndim != 1 or not wav.numel() or not torch.isfinite(wav).all():
            raise ValueError("invalid generated audio")
        path = a.output / (row["id"] + ".wav")
        sf.write(path, wav.numpy(), tts.sr, subtype="PCM_16")
        report["samples"].append(dict(row, file=path.name, sha256=sha(path), sample_rate=tts.sr,
            duration_seconds=len(wav) / tts.sr, generation_seconds=elapsed,
            codec_frames=tts._macaq_last_frames, reached_limit=False))
        write_json(a.output / "report.json", report)
        print(f"Generated {row['id']}", flush=True)


if __name__ == "__main__":
    main()

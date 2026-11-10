"""Quantize and use MACaQ diffusion transformers through Hugging Face Diffusers."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--family", choices=("flux2-klein", "wan", "stable-audio"), required=True)
    calibration = q.add_mutually_exclusive_group(required=True)
    calibration.add_argument(
        "--calibration",
        type=Path,
        help="calibration prompts, one per line",
    )
    calibration.add_argument("--calibration-jsonl", type=Path,
                             help="prompt and optional image recipe")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--flips", type=int, default=512)
    q.add_argument("--max-rows", type=int, default=1024)
    q.add_argument("--keep-layers", action="append", default=[],
                   help="layer-name glob retaining source precision for sensitive layers")
    r = sub.add_parser("run")
    r.add_argument("--checkpoint", type=Path, required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--fp8-layers", action="append", default=[],
                   help="layer-name glob using FP8 activations; packed weights stay 4-bit")
    for p in (q, r):
        p.add_argument(
            "--model",
            required=p is q,
            help="original Diffusers model ID or local directory",
        )
        p.add_argument("--revision", help="pin remote model revision")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--steps", type=int)
        p.add_argument("--height", type=int)
        p.add_argument("--width", type=int)
        p.add_argument("--guidance", type=float)
        p.add_argument("--seed", type=int)
        p.add_argument("--kernel", choices=("reference", "fused"), default="fused")
        p.add_argument(
            "--image", type=Path, help="licensed conditioning image for Wan I2V"
        )
        p.add_argument("--frames", type=int)
        p.add_argument("--audio-seconds", type=float)
        p.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()
    saved = (json.loads((args.checkpoint / "macaq.json").read_text())
             if args.action == "run" else None)
    family_hint = args.family if saved is None else saved["recipe"]["family"]
    defaults = dict(steps=4, height=512, width=512, guidance=1.0, frames=17, audio_seconds=10.0, seed=42)
    if family_hint == "wan":
        defaults.update(steps=30, width=896, guidance=5.0, frames=33)
    elif family_hint == "stable-audio":
        defaults.update(steps=100, guidance=7.0)
    if saved is not None:
        defaults.update({key: value for key, value in saved["recipe"].get("settings", {}).items()
                         if key in defaults})
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if (
        args.gpu < 0
        or args.steps < 1
        or args.height < 16
        or args.width < 16
        or args.frames < 1
        or args.audio_seconds <= 0
        or args.fps < 1
    ):
        parser.error("invalid GPU, step count or dimensions")
    if args.output.exists():
        parser.error("--output must be a fresh path")
    if args.image and not args.image.is_file():
        parser.error("--image must exist")
    if args.action == "quantize":
        if args.flips < 1 or args.max_rows < 1:
            parser.error("flips and max-rows must be positive")
        if args.calibration_jsonl:
            from calibration import read_examples
            if args.image:
                parser.error("use recipe images instead of --image with --calibration-jsonl")
            examples = read_examples(args.calibration_jsonl)
        else:
            examples = [dict(prompt=line.strip(), seed=args.seed + i,
                             image=str(args.image) if args.image else None)
                        for i, line in enumerate(args.calibration.read_text(encoding="utf-8").splitlines())
                        if line.strip()]
        if not examples:
            parser.error("calibration prompts are empty")
        family = args.family
        image_conditioned = bool(examples[0].get("image"))
        if image_conditioned and family != "wan":
            parser.error("--image is currently supported for Wan I2V only")
    else:
        manifest = saved
        family = manifest["recipe"]["family"]
        source = manifest["recipe"].get("source", {})
        args.model = args.model or source.get("model")
        args.revision = args.revision or source.get("revision")
        if not args.model:
            parser.error("checkpoint has no pipeline source; provide --model")
        if "image_conditioned" in source and bool(args.image) != bool(
            source["image_conditioned"]
        ):
            parser.error("match the saved text-to-video/image-to-video pipeline mode")
        image_conditioned = bool(args.image)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    import macaq_diffusers as macaq
    from diffusers import DiffusionPipeline
    from pipelines import select_pipeline, load_pipeline
    from diffusers.utils import load_image, export_to_video

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        parser.error("W4A4 execution is unavailable in this environment")
    if args.action == "quantize" and not Path(args.model).exists():
        from huggingface_hub import HfApi
        args.revision = HfApi().model_info(args.model, revision=args.revision).sha
    pipeline_config = DiffusionPipeline.load_config(args.model, revision=args.revision)
    try:
        select_pipeline(pipeline_config, family, image_conditioned)
    except ValueError as error:
        parser.error(str(error))
    transformer = macaq.load_transformer(args.checkpoint) if args.action == "run" else None
    pipe = load_pipeline(args.model, family, image_conditioned, args.revision, transformer)
    if getattr(pipe, "transformer_2", None) is not None:
        parser.error(
            "dual-transformer diffusion models require a separate adapter; use the single-transformer Wan variant"
        )
    pipe.to("cuda")
    if args.action == "run":
        from kernels import set_kernel
        set_kernel(pipe.transformer, args.kernel, args.fp8_layers)
    image = (
        load_image(str(args.image)).resize((args.width, args.height))
        if args.image
        else None
    )

    def generate(prompt, seed, calibration=False, conditioning_image=None):
        kwargs = dict(
            prompt=prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=args.height,
            width=args.width,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        if family == "stable-audio":
            kwargs.pop("height")
            kwargs.pop("width")
            kwargs["audio_end_in_s"] = args.audio_seconds
            kwargs["output_type"] = "np"
        if calibration:
            kwargs["output_type"] = "latent"
        if family == "wan":
            kwargs["num_frames"] = args.frames
            selected_image = conditioning_image if conditioning_image is not None else image
            if selected_image is not None:
                kwargs["image"] = selected_image
        return pipe(**kwargs)

    if args.action == "quantize":
        def calibrate():
            # Release each decoded video before generating the next one.
            for i, entry in enumerate(examples):
                conditioning = (load_image(entry["image"]).resize((args.width, args.height))
                                if entry.get("image") else None)
                generate(entry["prompt"], entry.get("seed", args.seed + i),
                         calibration=True, conditioning_image=conditioning)

        recipe = macaq.quantize(
            pipe.transformer,
            calibrate,
            family,
            args.flips,
            args.max_rows,
            keep_layers=args.keep_layers,
            progress=lambda index, total, name: print(
                f"Quantizing {index + 1}/{total}: {name}", flush=True
            ) if index % 10 == 0 or index == total - 1 else None,
        )
        recipe.update(
            source=dict(
                model=args.model,
                revision=args.revision,
                image_conditioned=image_conditioned,
            ),
            calibration_sha256=macaq.file_hash(args.calibration_jsonl or args.calibration),
            prompts=len(examples),
            prompt_sha256=[hashlib.sha256(entry["prompt"].strip().encode()).hexdigest()
                           for entry in examples],
            conditioning_images=[dict(sha256=macaq.file_hash(entry["image"]),
                                      source=entry.get("source", {}))
                                 for entry in examples if entry.get("image")],
            settings={
                k: getattr(args, k)
                for k in ("steps", "height", "width", "guidance", "seed", "frames", "audio_seconds")
            },
            vae_dtype=str(pipe.vae.dtype),
        )
        if args.image:
            recipe["image_sha256"] = macaq.file_hash(args.image)
        macaq.save_checkpoint(pipe.transformer, args.output, recipe)
        print(f"Saved {args.output}")
    else:
        with torch.inference_mode():
            result = generate(args.prompt, args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if family == "flux2-klein":
            result.images[0].save(args.output)
        elif family == "stable-audio":
            import soundfile as sf
            sf.write(args.output, result.audios[0].T, pipe.vae.config.sampling_rate)
        else:
            export_to_video(result.frames[0], str(args.output), fps=args.fps)
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

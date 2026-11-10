"""MACaQ text-only vLLM: quantize, run or serve a saved checkpoint."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    q = sub.add_parser("quantize")
    q.add_argument("--model", required=True)
    q.add_argument("--calibration", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--precision", choices=("w4a4", "w4a16"), default="w4a4")
    q.add_argument("--tokens", type=int, default=16384)
    q.add_argument("--chunk", type=int, default=512)
    q.add_argument("--flips", type=int, help="refinement iterations; defaults to 512 for dense and 64 for MoE")
    q.add_argument('--revision', help='pin a remote source checkpoint revision')
    q.add_argument('--native-recipe', type=Path, help='verified native calibration recipe for Flash-Next')
    q.add_argument('--pool-rows', type=int, default=2048)
    r = sub.add_parser("run")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--prompt", required=True)
    r.add_argument("--max-tokens", type=int, default=128)
    s = sub.add_parser("serve")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    for p in (q, r, s):
        p.add_argument("--gpu", type=int, default=0)
    for p in (r, s):
        p.add_argument('--gpus', help='comma-separated physical GPU indices for tensor parallelism')
        p.add_argument('--tensor-parallel-size', type=int, default=1)
        p.add_argument('--cpu-offload-gib', type=float, default=0,
                       help='optional host offload budget per GPU')
        p.add_argument('--gpu-memory-utilization', type=float, default=.85,
                       help='startup memory fraction; the KV budget is set separately')
        p.add_argument(
            "--profile",
            choices=("fast", "stable"),
            default="fast",
            help="fast: compilation/graphs and prefix caching; stable: eager serial diagnostics",
        )
        p.add_argument("--context", type=int, default=2048)
        p.add_argument("--kv-cache-mib", type=int, default=1024)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.action != 'quantize':
        visible = args.gpus or str(args.gpu)
        if not math.isfinite(args.cpu_offload_gib) or args.cpu_offload_gib < 0:
            parser.error('host offload budget must be finite and nonnegative')
        if not 0 < args.gpu_memory_utilization <= 1:
            parser.error('GPU memory utilization must be greater than zero and at most one')
        if (args.tensor_parallel_size < 1 or any(not x.isdigit() for x in visible.split(','))
                or len(set(visible.split(','))) != len(visible.split(','))
                or len(visible.split(',')) < args.tensor_parallel_size):
            parser.error('supply distinct GPU indices covering the tensor parallel size')
        os.environ['CUDA_VISIBLE_DEVICES'] = visible
    if args.action == "quantize":
        if args.tokens < 1 or args.chunk < 1 or (args.flips is not None and args.flips < 1):
            parser.error("token, chunk and flip counts must be positive")
        if not args.calibration.is_file() or not args.calibration.read_text().strip():
            parser.error("--calibration must be a nonempty local UTF-8 file")
        if args.output.exists():
            parser.error("--output must be a fresh directory")
        if args.native_recipe:
            if args.precision != 'w4a4' or args.pool_rows < 1:
                parser.error('native expert export requires W4A4 and positive pool rows')
            from huggingface_hub import HfApi, snapshot_download
            source = Path(args.model)
            if not source.is_dir():
                revision = HfApi().model_info(args.model, revision=args.revision).sha
                source = Path(snapshot_download(args.model, revision=revision,
                    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', 'LICENSE*']))
            subprocess.run([sys.executable, str(ROOT / 'quantize/native_moe.py'),
                '--model', str(source), '--calibration', str(args.calibration),
                '--native-recipe', str(args.native_recipe), '--output', str(args.output),
                '--flips', str(args.flips or 64), '--pool-rows', str(args.pool_rows), '--gpu', str(args.gpu)], check=True)
            return
        from transformers import AutoConfig

        try:
            cfg = AutoConfig.from_pretrained(args.model, revision=args.revision)
        except ValueError as error:
            parser.error(f"cannot load this model configuration: {error}; choose the model environment documented in vllm/README.md")
        text_config = getattr(cfg, "text_config", cfg)
        moe_type = text_config.model_type in ("qwen3_moe", "qwen3_next", "olmoe", "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp_text")
        if args.flips is None:
            args.flips = 64 if moe_type else 512
        if moe_type:
            if args.precision != "w4a4":
                parser.error("the MoE adapters currently support W4A4")
            from huggingface_hub import snapshot_download
            source = Path(args.model)
            if not source.is_dir():
                source = Path(snapshot_download(args.model, revision=args.revision or getattr(cfg, "_commit_hash", None),
                    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "LICENSE*"]))
            adapter = "moe_text.py" if cfg.model_type == "qwen3_moe" else "moe_families.py"
            subprocess.run([sys.executable, str(ROOT / "quantize" / adapter),
                "--model", str(source), "--calibration", str(args.calibration), "--output", str(args.output),
                "--tokens", str(args.tokens), "--chunk", str(args.chunk), "--flips", str(args.flips),
                "--gpu", str(args.gpu)], check=True)
            return
        if cfg.model_type in ("qwen3_5", "qwen3_5_text"):
            from huggingface_hub import snapshot_download
            source = Path(args.model)
            if not source.is_dir():
                source = Path(snapshot_download(args.model, revision=args.revision or getattr(cfg, "_commit_hash", None),
                    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "LICENSE", "README.md"]))
            command = [sys.executable, str(ROOT / "quantize/hybrid_text.py"),
                       "--model", str(source), "--calibration", str(args.calibration),
                       "--output", str(args.output), "--precision", args.precision,
                       "--tokens", str(args.tokens), "--chunk", str(args.chunk),
                       "--flips", str(args.flips)]
            subprocess.run(command, check=True)
            return
        if cfg.model_type != "qwen3":
            parser.error("supported text adapters are listed in vllm/README.md")
        command = [
            sys.executable,
            str(ROOT / "quantize/mxfp4_macaq_v3.py"),
            "--model",
            args.model,
            "--calib-file",
            str(args.calibration),
            "--skip-eval",
            "--save_dir",
            str(args.output),
            "--activation-precision",
            args.precision,
            "--n_calib_tokens",
            str(args.tokens),
            "--chunk_tokens",
            str(args.chunk),
            "--n_flip_iters",
            str(args.flips),
            "--no-compile_flip",
        ]
        subprocess.run(command, check=True)
        return
    if args.context < 32 or args.kv_cache_mib < 1:
        parser.error("context must be >=32 and KV cache positive")
    config = json.loads((args.model / "config.json").read_text())
    qc = config.get("quantization_config", {})
    if not (qc.get("macaq_dense_w4a4") or qc.get("macaq_moe")) or qc.get("quant_method") != "mxfp4":
        parser.error("use a saved MACaQ text checkpoint")
    if qc.get("macaq_moe"):
        os.environ["MACAQ_TRITON_W4A4"] = "1"
    if qc.get("macaq_activation_precision", "w4a4") not in ("w4a4", "w4a16"):
        parser.error("unsupported saved activation precision")
    from macaq_env import setup_cuda_env
    from macaq_plugin import register

    setup_cuda_env()
    register()
    # Also bootstrap explicitly in spawned workers. This catches stale plugin
    # entry points after an editable installation has been updated.
    os.environ['MACAQ_VERIFY_WORKER'] = '1'
    if args.action == "serve":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                str(args.model),
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--served-model-name",
                "macaq",
                "--worker-extension-cls",
                "macaq_validation.DenseValidationWorkerExtension",
                "--dtype",
                "bfloat16",
                "--tensor-parallel-size",
                str(args.tensor_parallel_size),
                "--cpu-offload-gb",
                str(args.cpu_offload_gib),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
                "--max-model-len",
                str(args.context),
                "--max-num-seqs",
                "1",
                "--max-num-batched-tokens",
                str(args.context),
                "--kv-cache-memory-bytes",
                str(args.kv_cache_mib * 1024**2),
                "--enable-prefix-caching"
                if args.profile == "fast"
                else "--no-enable-prefix-caching",
                "--no-async-scheduling",
                "--kernel-config",
                '{"ir_op_priority":{"rms_norm":["vllm_c"]}}',
                "--compilation-config",
                '{"custom_ops":["all"]}',
            ]
            + (["--language-model-only"] if config.get("model_type") == "qwen3_5" else [])
            + (["--enforce-eager"] if args.profile == "stable" else []),
            check=True,
        )
        return
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    from vllm import LLM, SamplingParams

    model = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tensor_parallel_size,
        cpu_offload_gb=args.cpu_offload_gib,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=args.context,
        kv_cache_memory_bytes=args.kv_cache_mib * 1024**2,
        enforce_eager=args.profile == "stable",
        max_num_seqs=1,
        max_num_batched_tokens=args.context,
        enable_prefix_caching=args.profile == "fast",
        async_scheduling=False,
        kernel_config={"ir_op_priority": {"rms_norm": ["vllm_c"]}},
        compilation_config={"custom_ops": ["all"]},
        language_model_only=config.get("model_type") == "qwen3_5",
        worker_extension_cls="macaq_validation.DenseValidationWorkerExtension",
    )
    model.collective_rpc('macaq_inspect_methods',
        args=(True, qc.get('macaq_activation_precision', 'w4a4')))
    tokenizer = model.get_tokenizer()
    prompt = (tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    ) if getattr(tokenizer, 'chat_template', None) else args.prompt)
    print(
        model.generate(
            [prompt],
            SamplingParams(temperature=0, max_tokens=args.max_tokens),
            use_tqdm=False,
        )[0]
        .outputs[0]
        .text
    )


if __name__ == "__main__":
    main()

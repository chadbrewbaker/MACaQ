#!/usr/bin/env python3
"""convert_macaq_to_gguf.py , MACAQ MXFP4 W4A4 checkpoint -> GGUF for llama.cpp.

Takes a directory written by `quantize/macaq_mxfp4.py --save_dir ...` (i.e. by
`save_macaq_vllm.save_macaq_checkpoint`) and produces a .gguf that the vendored
llama.cpp in ./llama.cpp can run.

The quantized weights carry over byte-for-byte apart from a nibble permutation ,
MACAQ's OCP MXFP4 and ggml's `block_mxfp4` share the e8m0 convention
(floor(log2 amax) - 2 + 127) and the E2M1 code table. What llama.cpp gains here is
the per-linear `pre_scale` (D1) and the `macaq.hadamard_block` key; the rotation
matrix itself is rebuilt at run time, exactly as vLLM does.

    python convert_macaq_to_gguf.py ../quantize/qwen3-32b-macaq-mxfp4-w4a4

Everything after the checkpoint path is forwarded to llama.cpp's own
convert_hf_to_gguf.py, so its flags (--outfile, --outtype, --split-max-size, ...)
all work.
"""
import argparse
import json
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LLAMA_CPP = HERE / "llama.cpp"


def check_macaq_checkpoint(path: Path) -> dict:
    """Fail early, with a useful message, on anything that is not a MACAQ dir."""
    if not path.is_dir():
        raise SystemExit(f"{path} is not a directory")

    cfg_path = path / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"{cfg_path} not found , is this a MACAQ save_dir?")

    cfg = json.loads(cfg_path.read_text())
    quant = cfg.get("quantization_config") or {}

    if not quant.get("macaq_dense_w4a4"):
        raise SystemExit(
            f"{cfg_path} has no `quantization_config.macaq_dense_w4a4` marker.\n"
            "This converter accepts dense checkpoints only. "
            "For MoE GGUFs, use llamacpp/macaq.py quantize-moe.")

    arch = (cfg.get("architectures") or ["?"])[0]
    if arch not in ("Qwen3ForCausalLM", "Qwen3Model"):
        raise SystemExit(
            f"architecture {arch!r} is not wired up for MACAQ conversion yet.\n"
            "Mix conversion.macaq.MacaqModel into its class in llama.cpp/conversion/ "
            "to add it , the per-linear format is architecture-agnostic.")

    if not list(path.glob("*.safetensors")):
        raise SystemExit(f"no *.safetensors in {path}")

    if int(quant.get("hadamard_block", 32)) == 32:
        print("[macaq] note: 32-wide rotation. This runs fine; on some platforms a\n"
              "        checkpoint quantized with --had_block 64 is faster, since llama.cpp's\n"
              "        fast Hadamard transform starts at 64. Measure first: it is\n"
              "        model-dependent, and the vLLM path wants 32.",
              file=sys.stderr)

    return quant


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path, help="MACAQ checkpoint directory")
    ap.add_argument("--outtype", default="f16",
                    help="dtype for the tensors MACAQ left unquantized (embeddings, "
                         "norms, lm_head). Default f16; the quantized linears are "
                         "MXFP4 regardless (or --outtype in weight-only mode).")
    ap.add_argument("--weight-only", metavar="BASE_MODEL", type=Path, default=None,
                    help="emit a weight-only GGUF instead of MXFP4 W4A4: re-apply "
                         "MACAQ's transform to BASE_MODEL's original weights at full "
                         "precision and leave them unquantized, so `llama-quantize` "
                         "can pick the level scheme (Q4_K_M, IQ4_NL, Q3_K_M, IQ3_S, "
                         "...). BASE_MODEL must be the checkpoint the MACAQ run "
                         "started from. Activation precision is selected by the "
                         "backend; this export does not force FP4 activations.")
    args, passthrough = ap.parse_known_args()

    quant = check_macaq_checkpoint(args.model)
    print(f"[macaq] {args.model}  hadamard_block={quant.get('hadamard_block', 32)}")

    if not (LLAMA_CPP / "convert_hf_to_gguf.py").is_file():
        raise SystemExit(f"vendored llama.cpp not found at {LLAMA_CPP}")

    sys.path.insert(0, str(LLAMA_CPP))

    if args.weight_only is not None:
        if not args.weight_only.is_dir():
            raise SystemExit(f"--weight-only: {args.weight_only} is not a directory")
        from conversion.macaq import set_weight_only_base
        set_weight_only_base(args.weight_only)
        print(f"[macaq] weight-only: transforming {args.weight_only} at full precision.\n"
              f"        The result is UNQUANTIZED , quantize it next, e.g.\n"
              f"          llama.cpp/build/bin/llama-quantize out.gguf out-q4.gguf Q4_K_M")

    sys.argv = ["convert_hf_to_gguf.py", str(args.model),
                "--outtype", args.outtype, *passthrough]
    runpy.run_path(str(LLAMA_CPP / "convert_hf_to_gguf.py"), run_name="__main__")


if __name__ == "__main__":
    main()

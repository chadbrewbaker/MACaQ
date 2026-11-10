# vLLM: text generation and serving

**Contents**

- [Install](#install)
- [Quantize and save](#quantize-and-save)
- [Hybrid text models](#hybrid-text-models)
- [Mixture of experts](#mixture-of-experts)
- [Load and generate](#load-and-generate)
- [Serve an HTTP API](#serve-an-http-api)
- [Weight-only mode](#weight-only-mode)

Supports dense Qwen3 and Qwen3.5-family text models, including Qwen3.8-27B,
and [text MoEs](../llamacpp/README.md#mixture-of-experts).
**W4A4 is the default.** Run all commands from the repository root.


## Install

Use Linux and Python 3.12.

```bash
python3.12 -m venv .venv-vllm
.venv-vllm/bin/python -m pip install -r requirements/vllm.txt
.venv-vllm/bin/python -m pip install -r requirements/build.txt
.venv-vllm/bin/python -m pip install --no-build-isolation --no-deps -e .
```

Installation registers the MACaQ plugin in worker processes.

## Hybrid text models

For Qwen3.8-27B and hybrid MoEs, use the hybrid dependencies:

```bash
.venv-vllm/bin/python -m pip install -r requirements/vllm-qwen35.txt
```

The same quantize, run and serve commands below accept these model IDs.

## Quantize and save

Provide representative UTF-8 text. For an integration trial:

```bash
python3.12 examples/make_smoke_calibration.py --output data/text-calibration.txt
.venv-vllm/bin/python vllm/cli.py quantize \
  --model Qwen/Qwen3-4B --calibration data/text-calibration.txt \
  --output models/qwen3-macaq --gpu 0
```

`--model` accepts a Hub ID or local directory. Use a fresh output directory.
`--tokens`, `--chunk` and `--flips` control calibration effort. The saved
checkpoint contains packed weights, scales, transforms, tokenizer and source
provenance. The public CLI uses your supplied text.

## Mixture of experts

The same `quantize`, `run` and `serve` commands support the listed
[Qwen and OLMoE models](../llamacpp/README.md#mixture-of-experts).
Routed expert projections use W4A4 with grouped kernels and a fused small-batch decode
path. Attention, shared experts, routers, norms, embeddings and the output head
retain their original precision.
The MoE adapter currently supports W4A4 only.

Prepare text with `examples/make_moe_calibration.py`, then choose a source
from the [supported MoE list](../llamacpp/README.md#mixture-of-experts).
For multiple GPUs, add `--gpus 0,1 --tensor-parallel-size 2` to `run` or `serve`.

<details>
<summary><strong>Flash-Next environment</strong></summary>

```bash
python3.12 -m venv .venv-vllm-flash
.venv-vllm-flash/bin/python -m pip install -r requirements/vllm-flash.txt
.venv-vllm-flash/bin/python -m pip install -r requirements/build.txt
.venv-vllm-flash/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv-vllm-flash/bin/python vllm/cli.py quantize \
  --model Qwen/Qwen3.8-Flash-Next --calibration data/calibration.txt \
  --output models/flash-w4a4 --tokens 16384 --chunk 512 --flips 64 --gpu 0
.venv-vllm-flash/bin/python vllm/cli.py run \
  --model models/flash-w4a4 --gpu 0 --context 1024 --kv-cache-mib 512 \
  --prompt 'Write a Python function that returns the square of an integer.'
```

The original source uses about 360 GB of disk space and the serving checkpoint
about 178 GB. The large embedding table stays on the CPU and requires about
102 GB before framework and loading overhead.

</details>

## Load and generate

```bash
.venv-vllm/bin/python vllm/cli.py run --model models/qwen3-macaq \
  --prompt 'Explain how a rainbow forms.' --gpu 0
```

The default **fast** profile enables compilation, device graphs and prefix
caching. The first load includes warmup. Use `--profile stable` for eager
execution. `--context` and `--kv-cache-mib` control context and cache memory;
weights must fit alongside the cache. The current CLI processes one sequence
at a time. Loading or serving a checkpoint does not repeat quantization.

## Serve an HTTP API

```bash
.venv-vllm/bin/python vllm/cli.py serve --model models/qwen3-macaq --gpu 0
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"macaq","messages":[{"role":"user","content":"What is 17 plus 26?"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

Use `--host` and `--port` to change the bind address.

## Weight-only mode

Add `--precision w4a16` when quantizing, or reuse existing packed codes:

```bash
.venv-vllm/bin/python examples/select_dense_precision.py \
  --model models/qwen3-macaq --output models/qwen3-weight-only --precision w4a16
```

Run or serve the new directory normally. This fallback expands weights for
BF16 GEMM and is currently slower than W4A4. Keep shared checkpoint weights
immutable. Checkpoints from other quantization implementations are not
interchangeable with MACaQ checkpoints.

[Main README](../README.md)

# llama.cpp: text models

**Contents**

- [Install](#install)
- [Prepare a model](#prepare-a-model)
- [Quantize and save](#quantize-and-save)
- [Hybrid text models](#hybrid-text-models)
- [Mixture of experts](#mixture-of-experts)
- [Load and generate](#load-and-generate)
- [Faster GPU inference](#faster-gpu-inference)
- [Formats](#formats)

Supports dense Qwen3, Qwen3.8-27B and the listed
[Qwen and OLMoE text MoEs](#mixture-of-experts). Use the included llama.cpp
fork for MACaQ GGUFs.

Run all commands from the repository root.

## Install

```bash
python3.12 -m venv .venv-llamacpp
.venv-llamacpp/bin/python -m pip install -r requirements/llamacpp.txt
MACAQ_TOOLKIT=/usr/local/cuda
cmake -S llamacpp/llama.cpp -B llamacpp/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$MACAQ_TOOLKIT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$MACAQ_TOOLKIT" -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build llamacpp/llama.cpp/build \
  --target llama-imatrix llama-quantize llama-completion llama-perplexity llama-bench -j 4
```

Set `MACAQ_TOOLKIT` to the installed toolkit. Use a fresh build directory when
changing toolkits. `--bin-dir` selects an existing runtime build.

<details>
<summary><strong>CPU-only installation</strong></summary>

For CPU execution, use this instead of the GPU installation above:

```bash
python3.12 -m venv .venv-llamacpp
.venv-llamacpp/bin/python -m pip install -r requirements/llamacpp-cpu.txt
.venv-llamacpp/bin/python -m pip install -r requirements/llamacpp.txt
cmake -S llamacpp/llama.cpp -B llamacpp/llama.cpp/build-cpu \
  -DGGML_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build llamacpp/llama.cpp/build-cpu \
  --target llama-imatrix llama-quantize llama-completion llama-perplexity llama-bench -j 4
```

Use `--bin-dir llamacpp/llama.cpp/build-cpu/bin`, `--device cpu` and
`--gpu-layers 0` when quantizing. For generation, use the same binary directory
and `--gpu-layers 0`. CPU quantization takes longer.

</details>

## Prepare a model

Start with an original full-precision GGUF and its matching HF checkpoint.
Download and convert the same source copy if needed:

```bash
.venv-llamacpp/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-4B", local_dir="models/Qwen3-4B")
PY
.venv-llamacpp/bin/python llamacpp/llama.cpp/convert_hf_to_gguf.py \
  models/Qwen3-4B --outtype f16 --outfile models/qwen3-4b-f16.gguf
```

## Quantize and save

Supply representative UTF-8 text. To try the workflow with generated example text:

```bash
python3.12 examples/make_smoke_calibration.py --output data/text-calibration.txt
.venv-llamacpp/bin/python llamacpp/macaq.py quantize \
  --model models/qwen3-4b-f16.gguf --hf models/Qwen3-4B \
  --calibration data/text-calibration.txt --format IQ4_XS --had-block 64 \
  --device cuda --gpu 0 --gpu-layers 99 \
  --output models/qwen3-4b-macaq.gguf --work-dir work/qwen3-4b
```

Choose fresh output and work paths. The command collects calibration statistics
and writes the GGUF with a `.recipe.json` sidecar. Allow disk space for an
intermediate full-precision GGUF. Keep evaluation text separate from calibration.
A short integration run can use `--context 128 --ubatch 128 --chunks 2`.

## Hybrid text models

For Qwen3.8-27B, install the hybrid model dependencies:

```bash
.venv-llamacpp/bin/python -m pip install -r requirements/llamacpp-qwen35.txt
```

Download `Qwen/Qwen3.8-27B` instead of the dense example above, then use the
same conversion, quantization and generation commands with its local paths.

## Mixture of experts

The native MoE adapter supports IQ4_XS and IQ4_NL:

| Model | Source checkpoint | Weight license |
|:---|:---|:---|
| Qwen3-30B-A3B | [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | Apache-2.0 |
| Qwen3-Coder-30B-A3B | [Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct) | Apache-2.0 |
| Qwen3.6-35B-A3B | [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | Apache-2.0 |
| OLMoE-1B-7B | [allenai/OLMoE-1B-7B-0125-Instruct](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct) | Apache-2.0 |
| Qwen3-Next-80B-A3B | [Qwen/Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) | Apache-2.0 |
| Qwen3.8-Flash-Next | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | Qwen Community 1.0 |


Install `requirements/llamacpp-qwen35.txt` in the native environment. Download
an original checkpoint from the table with `snapshot_download`, using
`models/moe-source` as the local directory. Prepare calibration text:

```bash
python3.12 examples/make_moe_calibration.py --output data/calibration.txt
```

Convert the same local source into the calibration GGUF:

```bash
.venv-llamacpp/bin/python llamacpp/llama.cpp/convert_hf_to_gguf.py \
  models/moe-source --outtype f16 --outfile models/moe-f16.gguf

.venv-llamacpp/bin/python llamacpp/macaq.py quantize-moe \
  --hf models/moe-source --teacher models/moe-f16.gguf \
  --calibration data/calibration.txt --output models/moe-macaq.gguf \
  --work-dir work/moe --format IQ4_XS --gpus 0 --gpu-layers 99 \
  --context 512 --ubatch 128 --chunks 64 --sample-rows 32 \
  --sample-budget-gib 8 --min-expert-tokens 8 --vq-sweeps 1
```

For Qwen3.6 and Qwen3-Next, add `--no-mtp` to the HF-to-GGUF conversion. The MACaQ text
export excludes auxiliary prediction blocks.

Choose fresh output and work paths. Add `--resume` to the same command after an
interruption. Resume verifies input fingerprints and completed tensor checksums.
The default coverage policy uses router-weighted inputs for experts missed by
hard routing. Use `--coverage-policy error` to require native observations for
every expert, then extend calibration if coverage is insufficient.

Compile grouped preprocessing and generate from the saved GGUF:

```bash
.venv-llamacpp/bin/python llamacpp/compile_kernels.py \
  --output work/moe-kernels --gpu 0 --moe --pdl

.venv-llamacpp/bin/python llamacpp/macaq.py run \
  --model models/moe-macaq.gguf --kernel-dir work/moe-kernels \
  --gpu 0 --gpu-layers 99 --prompt 'What is 17 plus 26? Return only the result.'
```

See [kernel setup](../llamacpp/README.md#faster-gpu-inference) for `--pdl`
compatibility. `--gpus 0,1` splits native calibration and expert refinement
across two GPUs when needed. The final GGUF loads without the original HF
weights, calibration GGUF or reservoir file.

<details>
<summary><strong>Flash-Next quantization</strong></summary>

Download the original source first:

```python
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.8-Flash-Next", local_dir="models/flash-source")
```

For llama.cpp, use the native environment and a compatible calibration GGUF of
the same base model. A compatible released calibration model can be downloaded as follows:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "unsloth/Qwen3.8-Flash-Next-GGUF",
    revision="766911a6b7369840a91dbcd95f9f997acaab6cd6",
    allow_patterns=["UD-IQ4_XS/*.gguf", "LICENSE*"],
    local_dir="models/flash-teacher",
)
```

```bash
.venv-llamacpp/bin/python llamacpp/macaq.py quantize-moe \
  --hf models/flash-source \
  --teacher models/flash-teacher/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
  --calibration data/calibration.txt --output models/flash-macaq.gguf \
  --work-dir work/flash-native --format IQ4_XS --embedding-format Q8_0 \
  --gpus 0,1 --gpu-layers 99 --context 512 --ubatch 128 --chunks 32 \
  --sample-rows 16 --sample-budget-gib 12 --min-expert-tokens 8 --vq-sweeps 1
```

The resulting GGUF is about 125 GB. Its large embedding table uses Q8_0 and
reads rows from CPU memory as needed. Generate with the common native `run` command, substituting
`models/flash-macaq.gguf`. The teacher and original HF weights are needed for
quantization; the saved GGUF is sufficient for inference.

</details>

## Load and generate

```bash
.venv-llamacpp/bin/python llamacpp/macaq.py run \
  --model models/qwen3-4b-macaq.gguf --gpu 0 --gpu-layers 99 \
  --prompt 'Explain how a rainbow forms.'
```

Loading does not repeat calibration. Generation uses the saved chat template
and greedy sampling; add `--thinking` to enable thinking. Compatible published
GGUFs can also be loaded with `run`.

## Faster GPU inference

For a model quantized with `--had-block 64`, compile preprocessing kernels
once for the inference device:

```bash
.venv-llamacpp/bin/python llamacpp/compile_kernels.py \
  --output work/kernels --gpu 0 --pdl
.venv-llamacpp/bin/python llamacpp/macaq.py run \
  --model models/qwen3-4b-macaq.gguf --kernel-dir work/kernels \
  --gpu 0 --gpu-layers 99 --prompt 'Explain how a rainbow forms.'
```

Use `--pdl` only with a compatible compiler and device; omit it below compute
capability 9.0. Compile a new directory when changing device architecture.
Do not edit a saved model's block-size metadata to enable fusion; changing the
transform requires quantizing again.

## Formats

| Weight bits | `--format` |
|:---|:---|
| 2 | `IQ2_XXS`, `IQ2_XS` |
| 3 | `IQ3_XXS`, `IQ3_S` |
| 4 | `IQ4_NL`, `IQ4_XS` |

These are weight codecs; activation arithmetic is selected by the runtime.
The fast decode path uses native Q8 activations. Embedding/head precision and
other tensor types are recorded in the saved recipe.

[Main README](../README.md)

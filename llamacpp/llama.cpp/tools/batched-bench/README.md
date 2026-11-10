# llama.cpp/example/batched-bench

Benchmark the batched decoding performance of `llama.cpp`

## Usage

There are 2 modes of operation:

- `prompt not shared` - each batch has a separate prompt of size `PP` (i.e. `N_KV = B*(PP + TG)`)
- `prompt is shared` - there is a common prompt of size `PP` used by all batches (i.e. `N_KV = PP + B*TG`)

```bash
./llama-batched-bench -m model.gguf -c 2048 -b 2048 -ub 512 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32 [-pps]

# LLaMA 7B, F16, N_KV_MAX = 16384 (8GB), prompt not shared
./llama-batched-bench -m ./models/llama-7b/ggml-model-f16.gguf -c 16384 -b 2048 -ub 512 -ngl 99

# LLaMA 7B, Q8_0, N_KV_MAX = 16384 (8GB), prompt is shared
./llama-batched-bench -m ./models/llama-7b/ggml-model-q8_0.gguf -c 16384 -b 2048 -ub 512 -ngl 99 -pps

# custom set of batches
./llama-batched-bench -m ./models/llama-7b/ggml-model-q8_0.gguf -c 2048 -b 512 -ub 512 -ngl 999 -npp 128,256,512 -ntg 128,256 -npl 1,2,4,8,16,32
```

## Sample results

- `PP` - prompt tokens per batch
- `TG` - generated tokens per batch
- `B` - number of batches
- `N_KV` - required KV cache size
- `T_PP` - prompt processing time (i.e. time to first token)
- `S_PP` - prompt processing speed (`(B*PP)/T_PP` or `PP/T_PP`)
- `T_TG` - time to generate all batches
- `S_TG` - text generation speed (`(B*TG)/T_TG`)
- `T` - total time
- `S` - total speed (i.e. all tokens / total time)

|    PP |     TG |    B |   N_KV |   T_PP s |   T_TG s |      T s |
|-------|--------|------|--------|----------|----------|----------|
|   128 |    128 |    1 |    256 |    0.108 |    3.079 |    3.187 |
|   128 |    128 |    2 |    512 |    0.198 |    5.029 |    5.227 |
|   128 |    128 |    4 |   1024 |    0.373 |    6.878 |    7.251 |
|   128 |    128 |    8 |   2048 |    0.751 |    7.344 |    8.095 |
|   128 |    128 |   16 |   4096 |    1.570 |    8.455 |   10.024 |
|   128 |    128 |   32 |   8192 |    3.408 |    8.801 |   12.209 |
|   128 |    256 |    1 |    384 |    0.107 |    6.329 |    6.436 |
|   128 |    256 |    2 |    768 |    0.194 |   10.239 |   10.433 |
|   128 |    256 |    4 |   1536 |    0.366 |   13.960 |   14.326 |
|   128 |    256 |    8 |   3072 |    0.751 |   15.110 |   15.861 |
|   128 |    256 |   16 |   6144 |    1.569 |   18.073 |   19.642 |
|   128 |    256 |   32 |  12288 |    3.409 |   19.223 |   22.633 |

### JSONL output

Pass `--output-format jsonl` to output JSONL instead of Markdown, á la

```json lines
{"n_kv_max": 2048, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "is_pp_shared": 0, "n_gpu_layers": 99, "n_threads": 8, "n_threads_batch": 8, "pp": 128, "tg": 128, "pl": 1, "n_kv": 256, "t_pp": 0.233810, "speed_pp": 547.453064, "t_tg": 3.503684, "speed_tg": 36.532974, "t": 3.737494, "speed": 68.495094}
{"n_kv_max": 2048, "n_batch": 2048, "n_ubatch": 512, "flash_attn": 0, "is_pp_shared": 0, "n_gpu_layers": 99, "n_threads": 8, "n_threads_batch": 8, "pp": 128, "tg": 128, "pl": 2, "n_kv": 512, "t_pp": 0.422602, "speed_pp": 605.770935, "t_tg": 11.106112, "speed_tg": 23.050371, "t": 11.528713, "speed": 44.410854}
```

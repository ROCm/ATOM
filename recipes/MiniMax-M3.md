# MiniMax-M3 MXFP4/MXFP8 Usage Guide

[MiniMax-M3-MXFP4](https://huggingface.co/amd/MiniMax-M3-MXFP4) and [MiniMax-M3-MXFP8](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8) are supported by the native ATOM OpenAI-compatible server path.

## Preparing Environment

Pull the latest development image:

```bash
docker pull rocm/atom-dev:latest
```

## MXFP4 on 4xMI355 GPUs

### Launching Server

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' 2>&1 | tee "${run_name}-server.log"
```

## MXFP8 on 4xMI355 GPUs

### Launching Server

For the MXFP8 model, online quant is used to convert the linear weights in attention module and first 3 dense MLP layers to PTPC FP8 format, which are originally equipped with 1*32 block scale.
The MoE weights keep unchanged. Check **--online_quant_config** in the script below for more details.

```bash
model_path=${model_path:-MiniMaxAI/MiniMax-M3-MXFP8}
run_name=${run_name:-m3-mxfp8}
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --max-num-batched-tokens 32768 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' 2>&1 | tee "${run_name}-server.log"
```


### Accuracy Test

Run GSM8K 5-shot with `lm_eval`:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
run_name=${run_name:-m3-mxfp4}
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn 2>&1 | tee "${run_name}-bs65-accuracy.log"
```

Validated MXFP4 GSM8K result:

```text
local-chat-completions ({'model': 'amd/MiniMax-M3-MXFP4', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9363|±  |0.0067|
|     |       |strict-match    |     5|exact_match|↑  |0.9371|±  |0.0067|
```

Validated MXFP8 GSM8K result:

```text
local-chat-completions ({'model': 'MiniMaxAI/MiniMax-M3-MXFP8', 'base_url': 'http://127.0.0.1:8000/v1/chat/completions', 'num_concurrent': 32, 'max_gen_toks': 16384}), gen_kwargs: ({}), limit: None, num_fewshot: 5, batch_size: 65
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.9484|±  |0.0061|
|     |       |strict-match    |     5|exact_match|↑  |0.9477|±  |0.0061|
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and
latency:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
  --model="$model_path" \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len="${ISL}" \
  --random-output-len="${OSL}" \
  --random-range-ratio=0.8 \
  --num-prompts=$(( CONC * 10 )) \
  --max-concurrency="${CONC}" \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

Reference MXFP4 results from the validated run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 73.27 | 260.77 | 791.33 | 7.50 | 8.33 | 502.35 | 4515.86 |
| 8 | 80 | 85.64 | 295.52 | 1144.91 | 8.78 | 9.29 | 864.87 | 7693.44 |
| 16 | 160 | 114.35 | 383.04 | 2200.03 | 11.73 | 12.84 | 1280.47 | 11555.95 |
| 32 | 320 | 163.86 | 512.32 | 4477.16 | 16.74 | 19.12 | 1807.32 | 16161.65 |
| 64 | 640 | 242.49 | 831.98 | 8566.28 | 25.00 | 29.83 | 2432.75 | 21928.25 |

Reference MXFP8 results from the validated run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 82.00 | 268.02 | 564.13 | 8.43 | 8.66 | 448.82 | 4034.60 |
| 8 | 80 | 103.52 | 323.33 | 1284.59 | 10.67 | 11.31 | 715.51 | 6364.77 |
| 16 | 160 | 143.25 | 414.95 | 2411.41 | 14.80 | 16.44 | 1022.17 | 9224.81 |
| 32 | 320 | 208.34 | 565.02 | 4936.02 | 21.42 | 24.16 | 1421.47 | 12711.25 |
| 64 | 640 | 305.81 | 893.93 | 9610.43 | 31.69 | 37.31 | 1929.04 | 17387.94 |

## EAGLE3 Speculative Decoding

EAGLE3 runs a small single-layer draft model alongside the MiniMax-M3 target to
propose multiple tokens per step, which the target then verifies. It is lossless
with respect to the target's greedy output. The draft checkpoint is
[`Inferact/MiniMax-M3-EAGLE3`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3).
Enable it by adding three flags to any of the server commands above:

- `--method eagle3`
- `--draft-model Inferact/MiniMax-M3-EAGLE3`
- `--num-speculative-tokens 3`

### Launching Server

The following starts the MXFP4 target with the EAGLE3 draft on 4xMI355 (the FP4
server command above plus the three speculative-decoding flags):

```bash
model_path=amd/MiniMax-M3-MXFP4
draft_path=Inferact/MiniMax-M3-EAGLE3

export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4

python -m atom.entrypoints.openai_server \
  --model "$model_path" \
  --tensor-parallel-size 4 \
  --server-port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --block-size 128 \
  --max-model-len 32768 \
  --max-num-seqs 256 \
  --kv_cache_dtype fp8 \
  --max-num-batched-tokens 32768 \
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
  --no-enable_prefix_caching \
  --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
  --method eagle3 \
  --draft-model "$draft_path" \
  --num-speculative-tokens 3 2>&1 | tee m3-mxfp4-eagle3-server.log
```

### Accuracy Test

Run GSM8K 5-shot with `lm_eval` (identical to the non-speculative test):

```bash
model_path=amd/MiniMax-M3-MXFP4
model_path=MiniMaxAI/MiniMax-M3-MXFP8
BS=65

lm_eval \
  --model local-chat-completions \
  --model_args "model=$model_path,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=32,max_gen_toks=16384" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --batch_size "${BS}" \
  --apply_chat_template \
  --fewshot_as_multiturn 2>&1 | tee m3-mxfp4-eagle3-bs65-accuracy.log
```

Validated MXFP4+EAGLE GSM8K result:

```text
| Case | ATOM Commit | GSM8K flexible-extract | GSM8K strict-match | Accept ratio | Avg toks/fwd | Accepted / Total Draft |
|---|---:|---:|---:|---:|---:|---:|
| `fp4_eagle_tp4` | `9fc48338` | `0.9469 ± 0.0062` | `0.9477 ± 0.0061` | `73.36%` | `3.20` | `90229 / 123000` |

MiniMax-M3 Eagle accepted tokens distribution:
`{0: 14.40%, 1: 12.00%, 2: 12.73%, 3: 60.87%}`
```

### Serving Benchmark

The following script can be used to benchmark online serving throughput and latency:

```bash
model_path=${model_path:-amd/MiniMax-M3-MXFP4}
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
  --model="$model_path" \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len="${ISL}" \
  --random-output-len="${OSL}" \
  --random-range-ratio=0.8 \
  --num-prompts=$(( CONC * 10 )) \
  --max-concurrency="${CONC}" \
  --request-rate=inf \
  --ignore-eos \
  --save-result \
  --use-chat-template \
  --percentile-metrics="ttft,tpot,itl,e2el"
```

Reference MXFP4 EAGLE3 results from our run on 4xMI355 GPUs:

| CONC | Requests | Duration (s) | Mean TTFT (ms) | P99 TTFT (ms) | Mean TPOT (ms) | P99 TPOT (ms) | Output tok/s | Total tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 40 | 43.38 | 287.09 | 755.46 | 4.27 | 7.78 | 850.53 | 7653.56 |
| 8 | 80 | 59.31 | 343.81 | 1516.38 | 5.93 | 10.85 | 1251.08 | 11146.00 |
| 16 | 160 | 78.17 | 430.34 | 2680.95 | 7.91 | 15.58 | 1876.30 | 16928.43 |
| 32 | 320 | 125.69 | 609.24 | 5304.23 | 12.60 | 23.81 | 2355.93 | 21132.49 |
| 64 | 640 | 198.58 | 966.20 | 10476.78 | 19.97 | 40.44 | 2973.94 | 26857.80 |

## PD Disaggregation

For PD-disaggregated serving (1P+1D, 2P+1D DPA, with/without EAGLE3), see
[recipes/mesh/MiniMax-M3.md](mesh/MiniMax-M3.md).

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

## PD Disaggregation (Single-Node 1P+1D)

Run prefill and decode as separate processes on the same node, each using 4 GPUs
(TP=4). KV cache transfer via Mooncake RDMA, routed through atomesh.

### Setup

Start the container on the node with the RDMA-aware docker script:

```bash
DOCKER_IMAGE=rocm/atom-dev:latest bash atom/mesh/scripts/docker_start.sh
docker exec -it atom_sglang_mesh bash
```

All commands below run **inside the container**.

### Start Prefill Server (GPU 0-3)

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee prefill.log
```

### Start Decode Server (GPU 4-7)

In a separate shell inside the same container:

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

export HIP_VISIBLE_DEVICES=4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee decode.log
```

### Start Router (atomesh)

Wait for both servers to show `Application startup complete`, then in a third shell:

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${NODE_IP}:8010" \
    --decode  "http://${NODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

### Verify

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"amd/MiniMax-M3-MXFP4","prompt":"Hello","max_tokens":32,"temperature":0}'
```

### PD + EAGLE3

Add EAGLE3 speculative decoding to the PD setup by appending these flags to
**both** the prefill and decode server commands:

```
--method eagle3 \
--draft-model Inferact/MiniMax-M3-EAGLE3 \
--num-speculative-tokens 3
```

The router command stays the same.

### GSM8K Accuracy (via Router)

```bash
lm_eval --model local-chat-completions \
    --model_args "model=amd/MiniMax-M3-MXFP4,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=3,max_gen_toks=16384" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 65 \
    --apply_chat_template \
    --fewshot_as_multiturn
```

### Serving Benchmark (via Router)

```bash
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
    --model=amd/MiniMax-M3-MXFP4 \
    --backend=vllm \
    --base-url=http://127.0.0.1:8000 \
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

### Online Quant Config Reference

The `--online_quant_config` flag converts attention and dense MLP linear weights
to PTPC FP8 at load time. It applies to all modes (standalone, PD, DPA).
The `exclude_layer` list differs by model variant and parallelism mode:

| Mode | Model | `exclude_layer` |
|------|-------|-----------------|
| TP-only or DPA | MXFP4 | `"lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"` |
| TP-only | MXFP8 | same as MXFP4 above |
| DPA | MXFP8 | `"lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*.gate.*", "*.block_sparse_moe.experts*"` |

The DPA + MXFP8 variant uses a broader exclude pattern (`*.gate.*`,
`*.block_sparse_moe.experts*`) to avoid quantizing the MoE gate and expert
weights that are already in FP8 format.

## PD Disaggregation — Multi-Node 2P+1D with DPA + TBO

For high-concurrency workloads, scale out to **2 prefill instances + 1 decode
instance** across 3 nodes. Each instance uses TP=4 with Data-Parallel Attention
(DPA) and Token-Budget Optimization (TBO) on prefill.

### Topology

```
Node 0 (prefill-1)  ─┐
                      ├──▶  atomesh router (:8000) ──▶ Client
Node 1 (prefill-2)  ─┤
                      │
Node 2 (decode)     ──┘
```

- Prefill instances: TP=4, `--enable-dp-attention --enable-tbo prefill`
- Decode instance: TP=4, `--enable-dp-attention`, higher `--max-num-seqs` (1024)
- Each node runs its own container (avoids ATOM port 29500 conflicts)

### Setup

Start a container on **each of the 3 nodes** using the RDMA-aware docker script:

```bash
DOCKER_IMAGE=rocm/atom-dev:latest bash atom/mesh/scripts/docker_start.sh
docker exec -it atom_sglang_mesh bash
```

### Start Prefill Server (Node 0 — prefill-1)

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --enable-dp-attention \
    --enable-tbo prefill \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee prefill.log
```

Repeat on **Node 1 (prefill-2)** with the same command.

### Start Decode Server (Node 2)

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_FORCE_ATTN_TRITON=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model amd/MiniMax-M3-MXFP4 \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --enable-dp-attention \
    --block-size 128 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --max-num-seqs 1024 \
    --max-num-batched-tokens 32768 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --no-enable_prefix_caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    2>&1 | tee decode.log
```

Key differences from prefill:
- `kv_role: kv_consumer`
- `--enable-dp-attention` without `--enable-tbo` (TBO is prefill-only)
- `--max-num-seqs 1024` — higher batch capacity for decode throughput
- `--cudagraph-capture-sizes` — pre-captures graphs for common batch sizes

### Start Router (atomesh)

Wait for all 3 servers to show `Application startup complete`, then on any node:

```bash
export PREFILL_IP_1=<prefill-node-0-ip>
export PREFILL_IP_2=<prefill-node-1-ip>
export DECODE_IP=<decode-node-2-ip>

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP_1}:8010" \
    --prefill "http://${PREFILL_IP_2}:8010" \
    --decode  "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

Note the **two `--prefill` flags** — atomesh round-robins requests across both
prefill instances.

### GSM8K Accuracy (via Router)

```bash
lm_eval --model local-chat-completions \
    --model_args "model=amd/MiniMax-M3-MXFP4,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=256,max_retries=3,max_gen_toks=16384" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 65 \
    --apply_chat_template \
    --fewshot_as_multiturn
```

### Serving Benchmark (via Router)

```bash
ISL=8192
OSL=1024
CONC=256

python -m atom.benchmarks.benchmark_serving \
    --model=amd/MiniMax-M3-MXFP4 \
    --backend=vllm \
    --base-url=http://127.0.0.1:8000 \
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

Typical concurrency sweep: `CONC=256,512,768,1024` to find the throughput ceiling.

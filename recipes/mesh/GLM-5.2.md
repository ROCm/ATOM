# GLM-5.2 PD Disaggregation with atomesh

PD-disaggregated serving for GLM-5.2 (MXFP4) using the ATOM native backend,
Mooncake RDMA KV transfer, and atomesh routing. Prefill uses Chunked Pipeline
Parallelism (PP4×TP1) for higher throughput; decode uses TP4 with full
CUDAGraph.

## Prerequisites

- AMD MI355X GPUs (8 GPUs per instance)
- RDMA network connectivity (RoCE or InfiniBand) for KV cache transfer
- Model weights accessible at the same path on all nodes
- Checkpoint: `GLM-5.2-MXFP4`

## Quick Reference

| Topology | Nodes | GPUs | Prefill flags | Decode flags | Typical CONC |
|----------|------:|-----:|---------------|--------------|-------------|
| 1P+1D (single-node) | 1 | 8 | PP=4, TP=1 | TP=4, CUDAGraph FULL | 1–256 |

## Environment Setup

Start a container with the RDMA-aware docker script:

```bash
DOCKER_IMAGE=rocm/atom-dev:latest bash atom/mesh/scripts/docker_start.sh
docker exec -it atom_mesh bash
```

All commands below run **inside the container**.

### Common Env Vars

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true
```

## Online Quant Config

```json
{
  "global_quant_config": "ptpc_fp8",
  "exclude_layer": [
    "lm_head",
    "model.embed_tokens",
    "*.mlp.gate",
    "*expert*"
  ]
}
```

## PP Layer Partition

GLM-5.2 has 78 transformer layers. The default even split (PP=4 → 19/20/19/20)
causes OOM on the embedding-heavy rank 0. Use a front-light partition:

```bash
export VLLM_PP_LAYER_PARTITION=18,20,20,20
```

---

## 1P+1D — Single-Node (PP4×TP1 Prefill + TP4 Decode)

Prefill on GPU 0-3 (PP4×TP1), decode on GPU 4-7 (TP4), router on port 8000.

### Prefill Server (GPU 0-3)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3
export VLLM_PP_LAYER_PARTITION=18,20,20,20

python3 -m atom.entrypoints.openai_server \
    --model GLM-5.2-MXFP4 \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --pipeline-parallel-size 4 --tensor-parallel-size 1 \
    --level 3 --enforce-eager \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 8192 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}' \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301,"proxy_ip":"127.0.0.1"}' \
    2>&1 | tee prefill.log
```

Key flags:
- `--pipeline-parallel-size 4 --tensor-parallel-size 1` — Chunked PP splits
  the model across 4 GPUs with no tensor-parallel communication, maximizing
  prefill throughput
- `--enforce-eager` — PP prefill uses eager mode (CUDAGraph not supported with PP)
- `--level 3` — enables piecewise compilation for individual ops
- `--max-num-batched-tokens 8192` — chunked-prefill batch budget

### Decode Server (GPU 4-7)

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model GLM-5.2-MXFP4 \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --level 3 \
    --cudagraph-mode FULL \
    --cudagraph-capture-sizes "[1,2,4,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,200,208,216,224,232,240,248,256]" \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "*.mlp.gate", "*expert*"]}' \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301,"proxy_ip":"127.0.0.1"}' \
    2>&1 | tee decode.log
```

Key differences from prefill:
- `kv_role: kv_consumer`
- `--tensor-parallel-size 4` — TP4 for low-latency decode
- `--cudagraph-mode FULL` — full CUDAGraph capture for decode batches
- No `--pipeline-parallel-size` or `--enforce-eager`

### Router

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${NODE_IP}:8010" 6301 \
    --decode  "http://${NODE_IP}:8020" \
    --backend atom \
    --log-level info
```

The `6301` after the prefill URL is the mooncake handshake port.

---

## Verify

After the router is up, send a test request:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"GLM-5.2-MXFP4","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}'
```

## GSM8K Accuracy (via Router)

GLM-5.2 uses `local-chat-completions` with 5-shot:

```bash
lm_eval --model local-chat-completions \
    --model_args "model=GLM-5.2-MXFP4,base_url=http://127.0.0.1:8000/v1/chat/completions,num_concurrent=64,max_retries=3,max_gen_toks=16384" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 65 \
    --apply_chat_template \
    --fewshot_as_multiturn
```

Expected: `exact_match (flexible) ≥ 0.93`

## Serving Benchmark (via Router)

```bash
ISL=8192
OSL=1024
CONC=128

python -m atom.benchmarks.benchmark_serving \
    --model=GLM-5.2-MXFP4 \
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

### Validated Results (2026-07-21)

**Accuracy:** GSM8K full (1319 samples, fewshot=5)
- `exact_match (flexible) = 0.9348 ± 0.0068` (threshold ≥ 0.90 PASS)

**Performance:** ISL=8192, OSL=1024, range-ratio=0.8

| Metric | conc=128 | conc=256 |
|--------|----------|----------|
| Successful requests | 1280/1280 | 2560/2560 |
| Output tok/s | 3,451 | 5,094 |
| Total tok/s | 31,157 | 45,836 |
| Mean / Median TTFT (ms) | 2250/1501 | 4009/2124 |
| P99 TTFT (ms) | 16,703 | 32,282 |
| Mean / Median TPOT (ms) | 33.5/33.7 | 44.4/44.9 |
| P99 TPOT (ms) | 34.1 | 48.6 |
| Mean / Median ITL (ms) | 33.5/33.5 | 44.4/44.3 |
| Mean / Median E2EL (ms) | 33019/32565 | 44895/43895 |

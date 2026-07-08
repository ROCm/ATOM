# DeepSeek-V4-Pro PD Disaggregation with atomesh

PD-disaggregated serving for DeepSeek-V4-Pro (FP8 native weights) using the
ATOM native backend, Mooncake RDMA KV transfer, and atomesh routing. Covers
two topologies (1P+1D pure TP, 2P+1D DPA), each with optional MTP
(multi-token prediction) speculative decoding.

## Prerequisites

- AMD MI355X GPUs (8 GPUs per instance, TP=8)
- RDMA network connectivity (RoCE or InfiniBand) for KV cache transfer
- Model weights accessible at the same path on all nodes
- Model: [`DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

## Quick Reference

| Topology | Nodes | Prefill flags | Decode flags | MTP | Typical CONC |
|----------|------:|---------------|--------------|-----|-------------|
| 1P+1D TP | 2 | TP=8 | TP=8 | -- | 1–256 |
| 1P+1D TP + MTP | 2 | TP=8, mtp | TP=8, mtp | `--num-speculative-tokens 3` | 1–256 |
| 2P+1D DPA | 3 | TP=8, DPA+TBO | TP=8, DPA | -- | 512–2048 |
| 2P+1D DPA + MTP | 3 | TP=8, DPA+TBO, mtp | TP=8, DPA, mtp | `--num-speculative-tokens 1` | 512–2048 |

## Environment Setup

Start container(s) with the RDMA-aware docker script:

```bash
DOCKER_IMAGE=rocm/atom-dev:latest bash atom/mesh/scripts/docker_start.sh
docker exec -it atom_mesh bash
```

For multi-node topologies, start a container on **each node** (separate
containers avoid ATOM port 29500 conflicts).

All commands below run **inside the container**.

### Common Env Vars

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true
```

DPA prefill instances additionally set:

```bash
export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5
```

---

## 1P+1D — Pure TP (2 Nodes)

Prefill on Node 0 (8 GPUs), decode on Node 1 (8 GPUs), router on port 8000.

### Prefill Server (Node 0)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model DeepSeek-V4-Pro \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    2>&1 | tee prefill.log
```

### Decode Server (Node 1)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model DeepSeek-V4-Pro \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    2>&1 | tee decode.log
```

### Router

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode  "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

### Adding MTP

Append these two flags to **both** the prefill and decode server commands:

```
--method mtp \
--num-speculative-tokens 3
```

The router command stays the same.

---

## 2P+1D DPA — Multi-Node (3 Nodes)

Two prefill instances + one decode instance across 3 nodes. Each instance uses
TP=8 with Data-Parallel Attention (DPA). Prefill instances additionally enable
Token-Budget Optimization (TBO).

### Topology

```
Node 0 (prefill-1)  ─┐
                      ├──▶  atomesh router (:8000) ──▶ Client
Node 1 (prefill-2)  ─┤
                      │
Node 2 (decode)     ──┘
```

### Prefill Server (Node 0 — prefill-1)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5

python3 -m atom.entrypoints.openai_server \
    --model DeepSeek-V4-Pro \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-dp-attention \
    --enable-tbo \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    2>&1 | tee prefill.log
```

Repeat on **Node 1 (prefill-2)** with the same command.

### Decode Server (Node 2)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model DeepSeek-V4-Pro \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --enable-dp-attention \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 256 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    2>&1 | tee decode.log
```

Key differences from prefill:
- `kv_role: kv_consumer`
- `--enable-dp-attention` without `--enable-tbo` (TBO is prefill-only)
- `--cudagraph-capture-sizes` — pre-captures graphs for common batch sizes

### Router

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

Two `--prefill` flags — atomesh round-robins requests across both prefill
instances.

### Adding MTP

Append these two flags to **both** the prefill and decode server commands:

```
--method mtp \
--num-speculative-tokens 1
```

The router command stays the same.

---

## Verify

After the router is up, send a test request:

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"DeepSeek-V4-Pro","prompt":"The capital of France is","max_tokens":32,"temperature":0}'
```

## GSM8K Accuracy (via Router)

DeepSeek-V4-Pro uses `local-completions` (not chat) with 3-shot:

```bash
lm_eval --model local-completions \
    --model_args "model=DeepSeek-V4-Pro,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3
```

For 2P+1D DPA topologies, increase `num_concurrent` to `768` or higher.

## Serving Benchmark (via Router)

```bash
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
    --model=DeepSeek-V4-Pro \
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

For 2P+1D DPA topologies, sweep higher concurrency: `CONC=512,768,1024,2048`.

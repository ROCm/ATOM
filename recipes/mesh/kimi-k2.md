# Kimi-K2.5 PD Disaggregation with atomesh

PD-disaggregated serving for Kimi-K2.5 (MXFP4, text-only backbone) using the
ATOM native backend, Mooncake RDMA KV transfer, and atomesh routing. Single-node
1P+1D topology.

## Prerequisites

- AMD MI355X GPUs (8 GPUs per instance, TP=4)
- RDMA network connectivity (RoCE or InfiniBand) for KV cache transfer
- Model weights accessible at the same path on all nodes
- Checkpoint: [`amd/Kimi-K2.5-MXFP4`](https://huggingface.co/amd/Kimi-K2.5-MXFP4)

## Quick Reference

| Topology | Nodes | GPUs | Prefill flags | Decode flags | Typical CONC |
|----------|------:|-----:|---------------|--------------|-------------|
| 1P+1D (single-node) | 1 | 8 | TP=4 | TP=4 | 1–256 |

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
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))")/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
rm -rf /root/.cache/atom/* 2>/dev/null || true
```

## 1P+1D — Single-Node (MXFP4)

Prefill on GPU 0-3, decode on GPU 4-7, router on port 8000.

### Prefill Server (GPU 0-3)

```bash
export HIP_VISIBLE_DEVICES=0,1,2,3

HSA_NO_SCRATCH_RECLAIM=1 python3 -m atom.entrypoints.openai_server \
    --model amd/Kimi-K2.5-MXFP4 \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}' \
    --no-enable_prefix_caching \
    2>&1 | tee prefill.log
```

### Decode Server (GPU 4-7)

```bash
export HIP_VISIBLE_DEVICES=4,5,6,7

HSA_NO_SCRATCH_RECLAIM=1 python3 -m atom.entrypoints.openai_server \
    --model amd/Kimi-K2.5-MXFP4 \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 32768 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    --no-enable_prefix_caching \
    2>&1 | tee decode.log
```

### Router

```bash
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

---

## Verify

After the router is up, send a test request:

```bash
curl -sS http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"amd/Kimi-K2.5-MXFP4","prompt":"Hello","max_tokens":32,"temperature":0}'
```

## GSM8K Accuracy (via Router)

```bash
lm_eval --model local-completions \
    --model_args "model=amd/Kimi-K2.5-MXFP4,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3
```

## Serving Benchmark (via Router)

```bash
ISL=8192
OSL=1024
CONC=16

python -m atom.benchmarks.benchmark_serving \
    --model=amd/Kimi-K2.5-MXFP4 \
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

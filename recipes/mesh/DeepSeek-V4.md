# DeepSeek-V4-Pro 1P+1D TP8 PD Disaggregation with ATOMesh

PD-disaggregated serving for DeepSeek-V4-Pro (FP8 native weights) using the ATOM native backend, Mooncake RDMA KV transfer, and ATOMesh routing. Covers four 2-node nightly configurations: pure TP, DPA, TP with MTP3, and DPA with MTP1.

## Prerequisites

- AMD MI355X GPUs (8 GPUs per instance, TP=8)
- RDMA network connectivity (RoCE or InfiniBand) for KV cache transfer
- Model weights accessible at the same path on all nodes
- Model: [`DeepSeek-V4-Pro`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)

## Quick Reference

| Configuration | Nodes | Prefill flags | Decode flags | MAX_NUM_SEQS | CONC |
|---------------|------:|---------------|--------------|-------------:|------|
| 1P+1D TP | 2 | TP=8 | TP=8 | 512 | 1–128 |
| 1P+1D DPA | 2 | TP=8, DPA+TBO | TP=8, DPA | 512 | 256, 512 |
| 1P+1D TP MTP3 | 2 | TP=8, MTP3 | TP=8, MTP3 | 512 | 1–128 |
| 1P+1D DPA MTP1 | 2 | TP=8, DPA+TBO, MTP1 | TP=8, DPA, MTP1 | 512 | 256, 512 |

All four configurations use:

- prefill port `8010`
- decode port `8020`
- atomesh router port `8000`
- Mooncake handshake port `6301`
- FP8 KV cache with block size `16`
- GPU memory utilization `0.85`
- prefix caching disabled

For multi-node topologies, start a container on **each node** (separate
containers avoid ATOM port 29500 conflicts).

All commands below run **inside the container**.

### Common Env Vars

The common YAML and backend configuration provide:

```bash
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_PD_RANK_MAPPING_POLICY=none
export ATOM_HOST_IP=${NODE_IP}
```

DPA prefill instances additionally set:

```bash
export GPU_MAX_HW_QUEUES=5
export ATOM_NUMA_BIND=1
```
---

## 1P+1D — Pure TP (2 Nodes)

Prefill runs on Node 0 with 8 GPUs, decode runs on Node 1 with 8 GPUs, and the router runs on Node 0.

### Topology

```text
Node 0: prefill TP8 :8010 ──┐
                            ├──▶ atomesh router :8000 ──▶ Client
Node 1: decode  TP8 :8020 ──┘
```

### Prefill Server (Node 0)

```bash
export NODE_IP=<prefill-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'", "handshake_port":6301}' \
    2>&1 | tee prefill.log
```

### Decode Server (Node 1)

```bash
export NODE_IP=<decode-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model deepseek-ai/DeepSeek-V4-Pro \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    2>&1 | tee decode.log
```

### Router

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

/usr/local/bin/atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

The default serving benchmark sweeps concurrency `1,2,4,8,16,32,64,128`.

---

## 1P+1D DPA — Multi-Node (2 Nodes)

Prefill and decode each use TP=8 with Data-Parallel Attention and Token-Budget Optimization enabled on prefill. Decode enables DPA without TBO.

### Topology

```text
Node 0: prefill TP8, DPA+TBO :8010 ──┐
                                     ├──▶ ATOMesh router :8000 ──▶ Client
Node 1: decode  TP8, DPA     :8020 ──┘
```

### Prefill Server (Node 0)

```bash
export GPU_MAX_HW_QUEUES=5
export ATOM_NUMA_BIND=1
export NODE_IP=<prefill-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --enable-dp-attention \
    --enable-tbo \
    2>&1 | tee prefill.log
```

### Decode Server (Node 1)

```bash
export NODE_IP=<decode-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    --enable-dp-attention \
    2>&1 | tee decode.log
```

### Router

The DPA topology uses the same router command as pure TP:

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

/usr/local/bin/atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-circuit-breaker \
    --prometheus-port 29100
```
Key differences from pure TP:

- Prefill receives `--enable-dp-attention --enable-tbo`.
- Decode receives `--enable-dp-attention` only.
- Default benchmark concurrency changes to `256,512`.
- Mooncake producer/consumer roles and router configuration remain unchanged.

---

## 1P+1D TP MTP3

This variant keeps the pure TP topology and concurrency sweep. Add the following flags to both prefill and decode commands: `--method mtp --num-speculative-tokens 3`

### Prefill Server (Node 0)

```bash
export NODE_IP=<prefill-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --method mtp \
    --num-speculative-tokens 3 \
    2>&1 | tee prefill-mtp3.log
```

### Decode Server (Node 1)

```bash
export NODE_IP=<decode-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    --method mtp \
    --num-speculative-tokens 3 \
    2>&1 | tee decode-mtp3.log
```

## 1P+1D DPA MTP1

This variant keeps the DPA concurrency sweep. Use these role-specific flags: 

For prefill: `--method mtp --num-speculative-tokens 1 --enable-dp-attention --enable-tbo`

For decode: `--method mtp --num-speculative-tokens 1 --enable-dp-attention`

### Prefill Server (Node 0)

```bash
export GPU_MAX_HW_QUEUES=5
export ATOM_NUMA_BIND=1
export NODE_IP=<prefill-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --method mtp \
    --num-speculative-tokens 1 \
    --enable-dp-attention \
    --enable-tbo \
    2>&1 | tee prefill-dpa-mtp1.log
```

### Decode Server (Node 1)

```bash
export NODE_IP=<decode-node-ip>
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    --no-enable_prefix_caching \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 512 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}' \
    --cudagraph-capture-sizes "[1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128,132,136,140,144,148,152,156,160,164,168,172,176,180,184,188,192,196,200,204,208,212,216,220,224,228,232,236,240,244,248,252,256]" \
    --method mtp \
    --num-speculative-tokens 1 \
    --enable-dp-attention \
    2>&1 | tee decode-dpa-mtp1.log
```

### Router

Both MTP variants use the same router command:

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

/usr/local/bin/atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-circuit-breaker \
    --prometheus-port 29100
```

Both MTP variants retain `--max-num-seqs 512`, FP8 KV cache, block size `16`, GPU memory utilization `0.85`, and the same Mooncake and router settings.

---

## Verify

First wait for the two worker health endpoints and the router model endpoint:

```bash
curl -f http://<prefill-node-ip>:8010/health
curl -f http://<decode-node-ip>:8020/health
curl -f http://<prefill-node-ip>:8000/v1/models
```

Then send a completion request through the router. To verify manually:

```bash
curl -sS http://<prefill-node-ip>:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "/mnt/models/DeepSeek-V4-Pro/",
      "prompt": "The capital of France is",
      "max_tokens": 32,
      "temperature": 0
    }'
```

DeepSeek-V4-Pro initialization includes weight loading, warmup, and CUDA graph capture.

## GSM8K Accuracy (via Router)

The generated GSM8K script uses `local-completions` with 3-shot prompting. `run_eval` is enabled for the pure TP and TP MTP3 nightly cells:

```bash
lm_eval --model local-completions \
    --model_args "model=deepseek-ai/DeepSeek-V4-Pro,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3
```

## Serving Benchmark (via Router)

Clone [InferenceX](https://github.com/SemiAnalysisAI/InferenceX) and execute its serving benchmark:

```bash
ISL=8192
OSL=1024
CONC=16
MODEL_PATH=deepseek-ai/DeepSeek-V4-Pro

python InferenceX/utils/bench_serving/benchmark_serving.py \
    --model="${MODEL_PATH}" \
    --backend=vllm \
    --base-url=http://127.0.0.1:8000 \
    --dataset-name=random \
    --random-input-len="${ISL}" \
    --random-output-len="${OSL}" \
    --random-range-ratio=0.8 \
    --num-prompts=$(( CONC * 10 )) \
    --max-concurrency="${CONC}" \
    --trust-remote-code \
    --num-warmups=$(( 2 * CONC )) \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics="ttft,tpot,itl,e2el"
```

Pure TP and TP MTP3 sweep `CONC=1,2,4,8,16,32,64,128`; DPA and DPA MTP1 sweep `CONC=256,512`.

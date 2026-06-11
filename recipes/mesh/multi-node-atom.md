# Multi-Node PD Disaggregation with ATOM Native Backend

Two-node Prefill-Decode disaggregation using the ATOM native inference engine and atomesh router for DeepSeek-V4-Pro. KV cache transfer via Mooncake RDMA.

## Prerequisites

- Two nodes with AMD MI355X GPUs (8 GPUs each)
- RDMA network connectivity between nodes (RoCE or InfiniBand)
- Shared filesystem (NFS) mounting model weights at the same path on both nodes
- Model: `DeepSeek-V4-Pro` (FP8 native weights)

## Step 1: Pull Docker Image

On both nodes:

```bash
docker pull rocm/atom-dev:latest
```

## Step 2: Start Docker Containers

On **each node**, start a container:

```bash
docker run -d --name atomesh \
    --network host --ipc host --privileged \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --cap-add IPC_LOCK --cap-add NET_ADMIN \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
    --shm-size 128G \
    -v /mnt:/mnt \
    rocm/atom-dev:latest sleep infinity
```

## Step 3: Start Prefill Server (Node 1)

Enter the container on the prefill node:

```bash
docker exec -it atomesh bash
```

Find the node IP and launch:

```bash
export PREFILL_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

# Clear stale ATOM compile cache from prior runs
rm -rf /root/.cache/atom/* 2>/dev/null || true

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=${PREFILL_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"'"${PREFILL_IP}"'","handshake_port":6301}' \
    2>&1 | tee /workspace/logs/prefill.log
```

Key parameters:
- `ATOM_HOST_IP` — must be set to the node's routable IP
- `AITER_BF16_FP8_MOE_BOUND=0` and `ATOM_MOE_GU_ITLV=1` — required for V4-Pro MoE in PD mode
- `--kv-transfer-config` — JSON with `kv_role: kv_producer` and Mooncake handshake port
- `-tp 8` — TP=8 across all 8 GPUs

## Step 4: Start Decode Server (Node 2)

Enter the container on the decode node:

```bash
docker exec -it atomesh bash
```

```bash
export DECODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

rm -rf /root/.cache/atom/* 2>/dev/null || true

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=${DECODE_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}

python3 -m atom.entrypoints.openai_server \
    --model /mnt/models/DeepSeek-V4-Pro/ \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    -tp 8 \
    --kv_cache_dtype fp8 \
    --block-size 16 \
    --gpu-memory-utilization 0.85 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"'"${DECODE_IP}"'","handshake_port":6301}' \
    2>&1 | tee /workspace/logs/decode.log
```

Key differences from prefill:
- `kv_role: kv_consumer` — receives KV cache from the prefill node
- `proxy_ip` — set to the decode node's own IP

## Step 5: Verify KV Transfer Info

Before starting the router, verify both servers report correct KV roles:

```bash
# From any node with network access:
curl -s http://<PREFILL_IP>:8010/kv_transfer_info | python3 -m json.tool
# Should show: "kv_role": "kv_producer"

curl -s http://<DECODE_IP>:8020/kv_transfer_info | python3 -m json.tool
# Should show: "kv_role": "kv_consumer"
```

## Step 6: Start Mesh Router (on Prefill Node)

Inside the prefill node container:

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode  "http://<DECODE_IP>:8020" \
    --policy random \
    --backend atom \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
```

ATOM backend does **not** require a bootstrap port after the `--prefill` URL (unlike SGLang/vLLM). The mesh router learns worker topology via the `/kv_transfer_info` HTTP endpoint.

## Step 7: Smoke Test

Verify the full pipeline with a quick completion:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/mnt/models/DeepSeek-V4-Pro/","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

## Step 8: Performance Benchmark

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=/mnt/models/DeepSeek-V4-Pro/ \
    --backend=vllm \
    --base-url=http://127.0.0.1:8000 \
    --dataset-name=random \
    --random-input-len=8192 \
    --random-output-len=1024 \
    --random-range-ratio 0.8 \
    --num-prompts=160 \
    --max-concurrency=16 \
    --trust-remote-code \
    --num-warmups=32 \
    --request-rate=inf \
    --ignore-eos \
    --save-result \
    --percentile-metrics='ttft,tpot,itl,e2el' \
    --result-dir=/workspace/benchmark_results \
    --result-filename=pd-atom-v4-mesh-8192-1024-16.json
```

The benchmark client uses `--backend=vllm` because the mesh router exposes OpenAI-compatible `/v1/completions` regardless of the upstream backend.

## Step 9: Accuracy Validation (GSM8K)

```bash
pip install 'lm-eval[api]'

lm_eval --model local-completions \
    --model_args "model=/mnt/models/DeepSeek-V4-Pro/,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3
```

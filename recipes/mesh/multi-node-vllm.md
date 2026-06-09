# Multi-Node PD Disaggregation with vLLM Backend

Two-node Prefill-Decode disaggregation using the vLLM backend with MooncakeConnector and atomesh router.

## Prerequisites

- Two nodes with AMD MI300X / MI325X / MI355X GPUs (8 GPUs each)
- RDMA network connectivity between nodes (RoCE or InfiniBand)
- Shared filesystem (NFS) mounting model weights at the same path on both nodes
- Model: `amd/DeepSeek-R1-0528-MXFP4-V2` (or any supported checkpoint)

## Step 1: Pull Docker Image

On both nodes:

```bash
docker pull rocm/atom-dev:vllm-latest
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
    -v /it-share:/it-share \
    rocm/atom-dev:vllm-latest sleep infinity
```

## Step 3: Start Prefill Server (Node 1)

Enter the container on the prefill node:

```bash
docker exec -it atomesh bash
```

```bash
export PREFILL_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_CACHE=/mnt/hf_hub_cache \
SAFETENSORS_FAST_GPU=1 \
VLLM_RPC_TIMEOUT=1800000 \
VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
vllm serve /mnt/models/DeepSeek-R1-0528-MXFP4-V2 \
    --host 0.0.0.0 --port 8010 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"}' \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": ['"$(seq -s, 1 256)"']}' \
    --load-format fastsafetensors \
    2>&1 | tee /workspace/logs/prefill.log
```

Key parameters:
- `VLLM_MOONCAKE_BOOTSTRAP_PORT=8998` — Mooncake bootstrap port (env var, not CLI flag)
- `--kv-transfer-config` — JSON with `MooncakeConnector` and `kv_role: kv_producer`
- `--async-scheduling` + `--compilation-config` — enables async scheduling with CUDA Graph

For eager mode (debugging), replace the `--async-scheduling --compilation-config ...` line with `--enforce-eager`.

## Step 4: Start Decode Server (Node 2)

Enter the container on the decode node:

```bash
docker exec -it atomesh bash
```

```bash
export DECODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_CACHE=/mnt/hf_hub_cache \
SAFETENSORS_FAST_GPU=1 \
VLLM_RPC_TIMEOUT=1800000 \
LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
vllm serve /mnt/models/DeepSeek-R1-0528-MXFP4-V2 \
    --host 0.0.0.0 --port 8020 \
    --trust-remote-code \
    --tensor-parallel-size 8 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --max-num-batched-tokens 16384 \
    --max-model-len 16384 \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"}' \
    --async-scheduling \
    --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": ['"$(seq -s, 1 256)"']}' \
    --load-format fastsafetensors \
    2>&1 | tee /workspace/logs/decode.log
```

Key differences from prefill:
- `kv_role: kv_consumer` — receives KV cache
- No `VLLM_MOONCAKE_BOOTSTRAP_PORT` env var on the decode side

## Step 5: Start Mesh Router (on Prefill Node)

Wait for both servers to report ready (`/v1/models` returns 200), then launch the router:

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" 8998 \
    --decode  "http://<DECODE_IP>:8020" \
    --policy random \
    --backend vllm \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
```

The `--prefill` flag takes the server URL followed by the bootstrap port (8998). Replace `<DECODE_IP>` with the decode node's IP.

## Step 6: Smoke Test

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/mnt/models/DeepSeek-R1-0528-MXFP4-V2","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

## Step 7: Performance Benchmark

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=/mnt/models/DeepSeek-R1-0528-MXFP4-V2 \
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
    --result-filename=pd-vllm-mesh-8192-1024-16.json
```

## Step 8: Accuracy Validation (GSM8K)

```bash
pip install 'lm-eval[api]'

lm_eval --model local-completions \
    --model_args "model=/mnt/models/DeepSeek-R1-0528-MXFP4-V2,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False" \
    --tasks gsm8k \
    --num_fewshot 3
```

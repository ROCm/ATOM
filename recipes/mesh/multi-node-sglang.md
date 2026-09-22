# Multi-Node PD Disaggregation with SGLang Backend

Two-node Prefill-Decode disaggregation using the SGLang-ATOM backend and atomesh router. KV cache transfer via Mooncake RDMA.

## Prerequisites

- Two nodes with AMD MI300X / MI325X / MI355X GPUs (8 GPUs each)
- RDMA network connectivity between nodes (RoCE or InfiniBand)
- Shared filesystem (NFS) mounting model weights at the same path on both nodes
- Model: `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (or any supported checkpoint)

## Step 1: Pull Docker Image

On both nodes:

```bash
docker pull rocm/atom-dev:sglang-latest
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
    rocm/atom-dev:sglang-latest sleep infinity
```

## Step 3: Start Prefill Server (Node 1)

Enter the container on the prefill node:

```bash
docker exec -it atomesh bash
```

Find the node IP and launch:

```bash
export PREFILL_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_CACHE=/mnt/hf_hub_cache \
SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
SGLANG_USE_AITER=1 \
SGLANG_AITER_FP8_PREFILL_ATTN=0 \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1 \
SGLANG_HOST_IP=${PREFILL_IP} \
SGLANG_MOONCAKE_SEND_AUX_TCP=1 \
MC_TCP_ENABLE_CONNECTION_POOL=true \
LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
python3 -m sglang.launch_server \
    --model-path /mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host 0.0.0.0 --port 8010 \
    --grpc-mode \
    --trust-remote-code \
    --tp-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --page-size 1 \
    --max-running-requests 128 \
    --disable-radix-cache \
    --log-level info \
    --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
    2>&1 | tee /workspace/logs/prefill.log
```

Key parameters:
- `--grpc-mode` — enables gRPC protocol (mesh router connects via `grpc://`)
- `SGLANG_HOST_IP` — must be set to the node's routable IP
- `--disaggregation-ib-device` — comma-separated RDMA device names

## Step 4: Start Decode Server (Node 2)

Enter the container on the decode node:

```bash
docker exec -it atomesh bash
```

```bash
export DECODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_CACHE=/mnt/hf_hub_cache \
SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
SGLANG_USE_AITER=1 \
SGLANG_AITER_FP8_PREFILL_ATTN=0 \
AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
ATOM_ENABLE_DS_QKNORM_QUANT_FUSION=1 \
SGLANG_HOST_IP=${DECODE_IP} \
SGLANG_MOONCAKE_SEND_AUX_TCP=1 \
MC_TCP_ENABLE_CONNECTION_POOL=true \
LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path /mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host 0.0.0.0 --port 8020 \
    --grpc-mode \
    --trust-remote-code \
    --tp-size 8 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --page-size 1 \
    --max-running-requests 128 \
    --cuda-graph-bs $(seq 1 256) \
    --disable-radix-cache \
    --log-level info \
    --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --disaggregation-ib-device rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7 \
    2>&1 | tee /workspace/logs/decode.log
```

Key differences from prefill:
- `--disaggregation-mode decode` — KV consumer
- `--cuda-graph-bs $(seq 1 256)` — CUDA Graph batch sizes for decode optimization
- `TORCHINDUCTOR_COMPILE_THREADS=128` — speeds up TorchInductor compilation

## Step 5: Start Mesh Router (on Prefill Node)

Wait for both gRPC servers to become healthy, then launch the router:

```bash
export HF_HUB_CACHE=/mnt/hf_hub_cache

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "grpc://${PREFILL_IP}:8010" 8998 \
    --decode  "grpc://<DECODE_IP>:8020" \
    --policy random \
    --backend sglang \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
```

Key differences from other backends:
- URL scheme is `grpc://` (not `http://`) because SGLang is launched with `--grpc-mode`
- The `--prefill` flag takes the server URL followed by the bootstrap port (8998)
- Replace `<DECODE_IP>` with the decode node's IP

## Step 6: Smoke Test

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"/mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

## Step 7: Performance Benchmark

```bash
git clone --depth 1 https://github.com/kimbochen/bench_serving.git /tmp/bench_serving

python3 /tmp/bench_serving/benchmark_serving.py \
    --model=/mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --backend=sglang \
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
    --result-filename=pd-sglang-mesh-8192-1024-16.json
```

## Step 8: Accuracy Validation (GSM8K)

```bash
pip install 'lm-eval[api]'

lm_eval --model local-completions \
    --model_args "model=/mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=65,max_retries=1,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3
```

# Single-Node PD Disaggregation with SGLang Backend

Prefill-Decode disaggregation on a single machine using the SGLang backend and atomesh router. Split GPUs between prefill and decode instances (xPyD topology).

## Prerequisites

- AMD MI300X / MI325X / MI355X node with 8 GPUs
- RDMA network (RoCE or InfiniBand) for Mooncake KV transfer
- Model: `amd/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4` (or any supported checkpoint)

## Step 1: Pull Docker Image

```bash
docker pull rocm/atom-dev:sglang-latest
```

## Step 2: Start Docker Container

```bash
docker run -d --name atomesh \
    --network host --ipc host --privileged \
    --device /dev/kfd --device /dev/dri \
    --group-add video \
    --cap-add IPC_LOCK --cap-add NET_ADMIN \
    --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288 \
    -v /mnt:/mnt -v /it-share:/it-share \
    rocm/atom-dev:sglang-latest sleep infinity
```

Enter the container:

```bash
docker exec -it atomesh bash
```

## Step 3: Start Prefill Server

Find the local node IP and IB devices:

```bash
export NODE_IP=$(ip route get 1.1.1.1 | awk '/src/ {print $7; exit}')

# Auto-detect IB devices (or set manually, e.g. rdma0,rdma1,...)
export IB_DEVICE=$(ls /sys/class/infiniband/ 2>/dev/null | paste -sd,)
```

Launch the prefill instance:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0 \
SGLANG_HOST_IP=${NODE_IP} \
SGLANG_MOONCAKE_SEND_AUX_TCP=1 \
MC_TCP_ENABLE_CONNECTION_POOL=true \
LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
python3 -m sglang.launch_server \
    --model-path /mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host 0.0.0.0 --port 8010 \
    --trust-remote-code \
    --tp-size 4 \
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
    --disaggregation-ib-device "${IB_DEVICE}" \
    2>&1 | tee /workspace/logs/prefill.log
```

Key parameters:
- `HIP_VISIBLE_DEVICES=0,1,2,3` — GPUs for this prefill instance (adjust per topology)
- `--tp-size 4` — tensor parallelism matching GPU count
- `--disaggregation-mode prefill` — marks this instance as KV producer
- `--disaggregation-bootstrap-port 8998` — Mooncake handshake port

## Step 4: Start Decode Server

In a separate terminal inside the same container:

```bash
HIP_VISIBLE_DEVICES=4,5,6,7 \
SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models \
ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=0 \
SGLANG_HOST_IP=${NODE_IP} \
SGLANG_MOONCAKE_SEND_AUX_TCP=1 \
MC_TCP_ENABLE_CONNECTION_POOL=true \
LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-} \
TORCHINDUCTOR_COMPILE_THREADS=128 \
python3 -m sglang.launch_server \
    --model-path /mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4 \
    --host 0.0.0.0 --port 8020 \
    --trust-remote-code \
    --tp-size 4 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --page-size 1 \
    --max-running-requests 128 \
    --cuda-graph-bs $(seq 1 64) \
    --disable-radix-cache \
    --log-level info \
    --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 9098 \
    --disaggregation-ib-device "${IB_DEVICE}" \
    2>&1 | tee /workspace/logs/decode.log
```

Key differences from prefill:
- `HIP_VISIBLE_DEVICES=4,5,6,7` — separate GPU set
- `--disaggregation-mode decode` — marks this as KV consumer
- `--cuda-graph-bs` — CUDA Graph batch sizes for decode optimization
- Different `--disaggregation-bootstrap-port` (9098 vs 8998)

## Step 5: Start Mesh Router

Wait for both servers to report ready, then launch the router:

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${NODE_IP}:8010" 8998 \
    --decode  "http://${NODE_IP}:8020" \
    --policy random \
    --backend sglang \
    --log-dir /workspace/logs \
    --log-level info \
    --disable-health-check \
    --prometheus-port 29100 \
    2>&1 | tee /workspace/logs/router.log
```

The router exposes an OpenAI-compatible API on port 8000. The `--prefill` flag takes the server URL followed by the bootstrap port.

## Step 6: Performance Benchmark

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
    --result-filename=pd-mesh-8192-1024-16.json
```

## Step 7: Accuracy Validation (GSM8K)

```bash
pip install 'lm-eval[api]'

lm_eval --model local-completions \
    --model_args "model=/mnt/models/DeepSeek-R1-0528-MXFP4-MTP-MoEFP4,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --tasks gsm8k \
    --num_fewshot 3
```

## Topology Variations

The single-node script supports flexible xPyD topologies:
- **1P1D (TP=4)**: `PREFILL_GPUS=0,1,2,3 DECODE_GPUS=4,5,6,7` (default above)
- **2P1D (TP=2)**: `PREFILL_GPUS=0,1,2,3 DECODE_GPUS=4,5 PREFILL_TP=2 DECODE_TP=2`
- **1P2D (TP=2)**: `PREFILL_GPUS=0,1 DECODE_GPUS=2,3,4,5 PREFILL_TP=2 DECODE_TP=2`

For multi-instance topologies, launch additional prefill/decode servers with different GPU sets and ports, and add corresponding `--prefill` / `--decode` flags to the router command.

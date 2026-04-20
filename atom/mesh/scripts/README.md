# ATOM Mesh PD Disaggregation Scripts

End-to-end guide for building, deploying, and benchmarking the ATOM Mesh prefill-decode (PD) disaggregation setup.

## Prerequisites

- Two nodes with AMD GPUs (one for prefill, one for decode)
- RDMA network between nodes
- Docker installed on both nodes

## 1. Build Docker Image

Run from the **ATOM repository root** on a machine with GPU access:

```bash
# Stage 1: build base image with ATOM + atom-mesh binary
DOCKER_BUILDKIT=1 docker build \
    --target atom_image \
    --build-arg MAX_JOBS=$(nproc) \
    -t atom-mesh-$(date +%Y%m%d)-v0.1:latest \
    -f docker/Dockerfile .

# Stage 2: build SGLang image on top of base
DOCKER_BUILDKIT=1 docker build \
    --target atom_sglang \
    --build-arg SGLANG_BASE_IMAGE=atom-mesh-$(date +%Y%m%d)-v0.1:latest \
    -t sglang-atom-mesh-$(date +%Y%m%d)-v0.1:latest \
    -f docker/Dockerfile .
```

Distribute the final image `sglang-atom-mesh-YYYYMMDD-v0.1:latest` to both nodes.

## 2. Start Docker Container

Run on **each node** (prefill and decode):

> **NOTE:** The default image in `docker_start.sh` is hardcoded to `sglang-atom-mesh-20260420-v0.1:latest`. If your built image has a different tag (e.g. a different date), you **must** pass it via the `DOCKER_IMAGE` environment variable, otherwise the container will fail to start with an image-not-found error.

```bash
# Adjust DOCKER_IMAGE to match your built image tag
DOCKER_IMAGE=sglang-atom-mesh-20260420-v0.1:latest bash docker_start.sh
```

Then enter the container:

```bash
docker exec -it atom_sglang_mesh bash
```

All remaining scripts are run **inside the container**.

## 3. Launch Prefill Server

On the **prefill node** container:

```bash
PREFILL_IP=<prefill_node_ip> \
MODEL_PATH=/mnt/models/deepseek-ai/DeepSeek-R1 \
PREFILL_TP=4 \
bash start_prefill.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFILL_IP` | *required* | Prefill node IP address |
| `MODEL_PATH` | *required* | Model path |
| `PREFILL_TP` | *required* | Tensor parallel size |
| `PREFILL_PORT` | `8010` | Server port |
| `BOOTSTRAP_PORT` | `8998` | Disaggregation bootstrap port |
| `MEM_FRACTION` | `0.85` | GPU memory fraction |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV cache data type |
| `CHUNKED_PREFILL_SIZE` | `16384` | Chunked prefill size |
| `MAX_RUNNING_REQUESTS` | `128` | Max concurrent requests |
| `IB_DEVICE` | `rdma0,...,rdma7` | RDMA devices |

## 4. Launch Decode Server

On the **decode node** container:

```bash
DECODE_IP=<decode_node_ip> \
MODEL_PATH=/mnt/models/deepseek-ai/DeepSeek-R1 \
DECODE_TP=8 \
bash start_decode.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DECODE_IP` | *required* | Decode node IP address |
| `MODEL_PATH` | *required* | Model path |
| `DECODE_TP` | *required* | Tensor parallel size |
| `DECODE_PORT` | `8020` | Server port |
| `CUDA_GRAPH_BS_START` | `1` | CUDA graph batch size start |
| `CUDA_GRAPH_BS_END` | `64` | CUDA graph batch size end |

Other optional variables are the same as prefill (`BOOTSTRAP_PORT`, `MEM_FRACTION`, etc.).

## 5. Launch Router

On either node's container (typically the **prefill node**):

```bash
PREFILL_IP=<prefill_node_ip> \
DECODE_IP=<decode_node_ip> \
bash start_router.sh
```

The script waits for both prefill and decode servers to be ready before starting the router (default timeout: 900s).

| Variable | Default | Description |
|----------|---------|-------------|
| `PREFILL_IP` | *required* | Prefill node IP |
| `DECODE_IP` | *required* | Decode node IP |
| `PREFILL_PORT` | `8010` | Prefill server port |
| `DECODE_PORT` | `8020` | Decode server port |
| `ROUTER_PORT` | `8000` | Router listening port |
| `POLICY` | `random` | Routing policy |
| `MESH_BIN` | `/usr/local/bin/atom-mesh` | Path to atom-mesh binary |
| `WAIT_TIMEOUT` | `900` | Timeout waiting for prefill/decode (seconds) |

## 6. Verify

Quick health check:

```bash
# Check router
curl http://127.0.0.1:8000/v1/models

# Send a test request
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "/mnt/models/deepseek-ai/DeepSeek-R1", "prompt": "Hello", "max_tokens": 32}'
```

### GSM8K Accuracy Evaluation

```bash
MODEL_PATH=/mnt/models/deepseek-ai/DeepSeek-R1 bash run_gsm8k.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *required* | Model path |
| `ROUTER_PORT` | `8000` | Router port |
| `LM_EVAL_TASK` | `gsm8k` | Evaluation task |
| `LM_EVAL_NUM_FEWSHOT` | `3` | Number of few-shot examples |
| `LM_EVAL_NUM_CONCURRENT` | `65` | Concurrent requests |
| `RESULT_DIR` | `/workspace/gsm8k_results` | Results directory |

### Performance Benchmark

```bash
MODEL_PATH=/mnt/models/deepseek-ai/DeepSeek-R1 \
ISL_LIST="1024,8192" \
OSL=1024 \
CONC_LIST="16,32,64" \
bash run_benchmark.sh
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | *required* | Model path |
| `ROUTER_PORT` | `8000` | Router port |
| `ISL_LIST` | `1024,8192` | Input sequence lengths (comma-separated) |
| `OSL` | `1024` | Output sequence length |
| `CONC_LIST` | `16,32,64` | Concurrency levels (comma-separated) |
| `RANDOM_RANGE_RATIO` | `0.8` | Random range ratio |
| `RESULT_DIR` | `/workspace/benchmark_results` | Results directory |
| `BACKEND` | `sglang` | Benchmark backend |

## Script Summary

| Script | Where to Run | Purpose |
|--------|-------------|---------|
| `docker_start.sh` | Host | Start Docker container |
| `start_prefill.sh` | Container | Launch prefill server |
| `start_decode.sh` | Container | Launch decode server |
| `start_router.sh` | Container | Launch mesh router (waits for prefill/decode) |
| `run_gsm8k.sh` | Container | Run GSM8K accuracy evaluation |
| `run_benchmark.sh` | Container | Run performance benchmark |

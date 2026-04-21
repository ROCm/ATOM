# ATOM Mesh PD Disaggregation Scripts

End-to-end guide for building, deploying, and benchmarking the ATOM Mesh prefill-decode (PD) disaggregation setup.

## Prerequisites

- Two nodes with AMD GPUs (one for prefill, one for decode)
- RDMA network between nodes
- Docker installed on both nodes

## 1. Start Docker Container

Pre-built images are available at `rocm/atom-dev:mesh-sglang-latest`. Run on **each node** (prefill and decode):

```bash
bash docker_start.sh
```

Then enter the container:

```bash
docker exec -it atom_sglang_mesh bash
```

All remaining scripts are run **inside the container**.

## 2. Launch Prefill Server

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

## 3. Launch Decode Server

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

## 4. Launch Router

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

## 5. Verify

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

## One-Shot SLURM Automation

For two-node SLURM clusters, `ds_fp8_1p_tp4_1d_tp8_slurm.sh` runs the entire flow
(pre-cleanup → containers → prefill + decode servers → router → GSM8K accuracy →
performance benchmark → cleanup) in a single `sbatch` submission. It is
**self-contained** — does not depend on any of the other scripts above.

### Submit

```bash
mkdir -p /it-share/yajizhan/slurm_logs
sbatch atom/mesh/scripts/ds_fp8_1p_tp4_1d_tp8_slurm.sh
```

### Override defaults

```bash
# Fast smoke test with random weights (skips weight load + GSM8K)
sbatch --export=ALL,LOAD_DUMMY=1 atom/mesh/scripts/ds_fp8_1p_tp4_1d_tp8_slurm.sh

# Custom workload (multiple ISL, custom concurrency)
sbatch --export=ALL,ISL_LIST="1024,4096,8192",CONC_LIST="32,64,128" \
    atom/mesh/scripts/ds_fp8_1p_tp4_1d_tp8_slurm.sh

# Skip GSM8K even with real weights (just run perf benchmark)
sbatch --export=ALL,RUN_GSM8K=0 atom/mesh/scripts/ds_fp8_1p_tp4_1d_tp8_slurm.sh
```

### Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/mnt/models/deepseek-ai/DeepSeek-R1` | Model path |
| `DOCKER_IMAGE` | `rocm/atom-dev:mesh-sglang-latest` | Container image |
| `PREFILL_TP` / `DECODE_TP` | `4` / `8` | Tensor parallel sizes |
| `LOAD_DUMMY` | `<empty>` | Load real weights by default. Set `LOAD_DUMMY=1` to skip weight loading for fast smoke-test |
| `ISL_LIST` | `8192` | Input sequence lengths (comma-separated) |
| `OSL` | `1024` | Output sequence length |
| `CONC_LIST` | `16,32,64` | Concurrency levels |
| `RUN_GSM8K` | `auto` | Run GSM8K accuracy eval before perf benchmark. `auto` = on iff real weights; `0` to force-skip; `1` to force-run |
| `GSM8K_LIMIT` | `100` | Number of GSM8K samples |
| `GSM8K_NUM_FEWSHOT` | `3` | Few-shot examples |
| `GSM8K_NUM_CONCURRENT` | `65` | Concurrent eval requests |
| `WAIT_SERVER_TIMEOUT` | `1800` | Per-server cold-start timeout (s) |

SBATCH directives (edit in script if your cluster differs):
- partition / account: `amd-frameworks`
- nodelist: `mia1-p02-g42,mia1-p02-g44`
- 2 nodes × 8 GPUs, exclusive, 4-hour walltime

### Outputs

```
/it-share/yajizhan/slurm_logs/
├── ds_fp8_1p_tp4_1d_tp8-<jobid>.out      # sbatch stdout (orchestration log)
├── ds_fp8_1p_tp4_1d_tp8-<jobid>.err      # sbatch stderr
└── <MMDD>_ds_fp8_1p_tp4_1d_tp8_<jobid>/
    ├── prefill/prefill.log               # sglang prefill server
    ├── decode/decode.log                 # sglang decode server
    ├── router/                           # atom-mesh router
    ├── gsm8k/<timestamp>_gsm8k/          # lm_eval GSM8K results (if enabled)
    ├── bench/pd-mesh-<isl>-<osl>-<conc>-<ratio>.json
    └── scripts/                          # generated in-container scripts
```

### Cancel / Cleanup

```bash
scancel <jobid>
```

`scancel` does not propagate into docker containers, so the script's pre-cleanup
step force-stops all stale containers on both nodes at the start of every run.
Residual GPU memory from a prior failed run is cleared automatically.

## Script Summary

| Script | Where to Run | Purpose |
|--------|-------------|---------|
| `docker_start.sh` | Host | Start Docker container |
| `start_prefill.sh` | Container | Launch prefill server |
| `start_decode.sh` | Container | Launch decode server |
| `start_router.sh` | Container | Launch mesh router (waits for prefill/decode) |
| `run_gsm8k.sh` | Container | Run GSM8K accuracy evaluation |
| `run_benchmark.sh` | Container | Run performance benchmark |
| `ds_fp8_1p_tp4_1d_tp8_slurm.sh` | SLURM head node | One-shot end-to-end PD benchmark on 2 nodes |

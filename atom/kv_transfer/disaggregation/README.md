# KV Cache Disaggregation (Prefill/Decode Separation)

Prefill/Decode (P/D) disaggregation runs the prefill and decode phases on
separate GPU instances. The prefill node computes KV caches and transfers
them to the decode node via RDMA, so the decode node can skip prefill
entirely and start generating tokens immediately.

Routing between clients and the P/D instances is handled by **atomesh**, a
lightweight Rust router.

## MORI (Modular RDMA Interface)

The underlying KV cache transfer is powered by
[**MORI**](https://github.com/ROCm/mori) — a modular, high-performance
RDMA framework for GPU-centric communication on AMD platforms.

Specifically, this module uses **MORI-IO**, the point-to-point communication
library within MORI. MORI-IO provides:

- **GPU-direct RDMA** — data moves directly between GPU VRAM across nodes
  without staging through host memory, minimizing latency and CPU overhead.
- **IBGDA (InfiniBand GPUDirect Async)** — RDMA operations are issued
  directly from GPU kernels, bypassing the CPU entirely for the data path.
- **Session-based transfers** — MORI-IO pre-builds RDMA sessions (QP pairs,
  memory registrations) during a one-time handshake. Subsequent transfers
  reuse these sessions with near-zero setup cost.
- **Hardware support** — works with AMD MI300X/MI325X/MI355X GPUs and
  ConnectX-7, Broadcom Thor2, and AMD Pollara (AINIC) NICs.

In the P/D disaggregation flow, the decode node uses MORI-IO to issue
RDMA READs against the prefill node's KV cache blocks. Each TP rank
independently reads its own KV slice, so the transfer is fully parallel
across the tensor-parallel group.

```
  Client ──▶ Router (atomesh :8000)
                │
                ▼
         Prefill Node (kv_producer)     # 1. compute KV caches
                │
                ▼
         Decode Node (kv_consumer)      # 2. RDMA read KV, generate tokens
                │
                ▼
             Router ──▶ Client          # 3. stream response back
```

## How to Run

### TP-only Mode (Tensor Parallelism)

#### 1. Start the Prefill Node

```bash
python -m atom.entrypoints.openai_server \
  --kv_cache_dtype fp8 \
  --model /path/to/model \
  --block-size 16 \
  -tp 8 \
  --server-port 8010 \
  --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}'
```

#### 2. Start the Decode Node

```bash
python -m atom.entrypoints.openai_server \
  --kv_cache_dtype fp8 \
  --model /path/to/model \
  --block-size 16 \
  -tp 8 \
  --server-port 8020 \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}'
```

#### 3. Start the Router (atomesh)

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://<PREFILL_IP>:8010" \
    --decode  "http://<DECODE_IP>:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker
```

#### 4. Send Requests (to the Router)

```bash
curl -s http://<ROUTER_IP>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"1 2 3 4 5","max_tokens":10,"temperature":0}'
```

### DP + TP Mode (Data Parallelism + Tensor Parallelism)

When running MoE models (e.g. DeepSeek-V3/R1), you can enable data parallelism
with expert parallelism for higher throughput. Each DP rank runs a full TP
group, and MoE all-to-all is handled by MORI.

Key differences from TP-only mode:

- Add `--enable-dp-attention --enable-expert-parallel` to both prefill and
  decode nodes.
- Set `MORI_SHMEM_MODE=ISOLATION` to separate MoRI (MoE all-to-all) and
  MORI-IO (KV transfer) symmetric heap memory pools — without this, the two
  subsystems compete for the same memory and cause OOM during warmup.
- Each decode DP rank binds MORI-IO sessions to **all** prefill DP ranks
  (not just its own), because any prefill DP rank may have processed the
  request.

#### 1. Start the Prefill Node

```bash
export MORI_SHMEM_MODE=ISOLATION

python -m atom.entrypoints.openai_server \
  --kv_cache_dtype fp8 \
  --model /path/to/model \
  --block-size 16 \
  -tp 8 \
  --server-port 8010 \
  --enable-dp-attention \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}'
```

#### 2. Start the Decode Node

```bash
export MORI_SHMEM_MODE=ISOLATION

python -m atom.entrypoints.openai_server \
  --kv_cache_dtype fp8 \
  --model /path/to/model \
  --block-size 16 \
  -tp 8 \
  --server-port 8020 \
  --enable-dp-attention \
  --enable-expert-parallel \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301}'
```

#### 3. Start the Router (atomesh)

```bash
atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://<PREFILL_IP>:8010" \
    --decode  "http://<DECODE_IP>:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker
```

#### 4. Send Requests

Same as TP-only — requests go through the router:

```bash
curl -s http://<ROUTER_IP>:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"1 2 3 4 5","max_tokens":10,"temperature":0}'
```


## RDMA Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ATOM_HOST_IP` | RDMA IP of this machine (overrides auto-detect) | `<RDMA_IP>` |
| `MORI_RDMA_SL` | RDMA Service Level (must match PFC no-drop priority) | `3` |
| `MORI_RDMA_TC` | RDMA Traffic Class (DSCP × 4) | `104` |
| `MORI_RDMA_DEVICES` | Restrict to specific RDMA device(s) | `rdma5` or `rdma0,rdma5` |

## RDMA Backend Tuning (`kv-transfer-config`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `qp_per_transfer` | 4 | QPs per RDMA transfer (more = higher parallelism) |
| `num_worker_threads` | 4 | RDMA I/O worker threads |
| `post_batch_size` | -1 (auto) | WRs posted per batch |
| `max_send_wr` | 0 (auto) | QP send queue depth |
| `max_cqe_num` | 0 (auto) | Max CQ entries |
| `max_msg_sge` | 0 (auto) | Max scatter/gather per WR |
| `enable_notification` | false | Transfer completion notification |

Example with tuning:

```bash
--kv-transfer-config '{
  "kv_role": "kv_consumer",
  "kv_connector": "moriio",
  "handshake_port": 6301,
  "qp_per_transfer": 8,
  "num_worker_threads": 8
}'
```

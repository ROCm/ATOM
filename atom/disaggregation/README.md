# KV Cache Disaggregation (Prefill/Decode Separation)

Prefill/Decode (P/D) disaggregation runs the prefill and decode phases on
separate GPU instances. The prefill node computes KV caches and transfers
them to the decode node via RDMA, so the decode node can skip prefill
entirely and start generating tokens immediately.

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
  Client ──▶ Proxy (:10001)
                │
                ▼
         Prefill Node (kv_producer)     # 1. compute KV caches
                │
                ▼
             Proxy                      # 2. receive block metadata
                │
                ▼
         Decode Node (kv_consumer)      # 3. RDMA read KV, generate tokens
                │
                ▼
             Proxy ──▶ Client           # 4. stream response back
```

## How to Run

### 1. Start the Proxy

```bash
python -m atom.disaggregation.proxy
# or with custom port:
python -m atom.disaggregation.proxy --port 10001
```

### 2. Start the Prefill Node

```bash
python -m atom.entrypoints.openai_server \
  --kv_cache_dtype fp8 \
  --model /path/to/model \
  --block-size 16 \
  -tp 8 \
  --kv-transfer-config '{"kv_role":"kv_producer","proxy_ip":"<PROXY_IP>","proxy_ping_port":36367,"http_prt":8000}'
```

### 3. Start the Decode Node

```bash
python -m atom.entrypoints.openai_server \
  --kv_cache_dtype fp8 \
  --model /path/to/model \
  --block-size 16 \
  -tp 8 \
  --kv-transfer-config '{"kv_role":"kv_consumer","proxy_ip":"<PROXY_IP>","proxy_ping_port":36367,"http_prt":8000}'
```

### 4. Send Requests (to the Proxy)

```bash
curl -s http://<PROXY_IP>:10001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"1 2 3 4 5","max_tokens":10,"temperature":0}'
```

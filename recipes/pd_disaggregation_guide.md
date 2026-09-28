# PD Disaggregation with Mooncake (RDMA)

Prefill-Decode disaggregation splits inference into two stages on separate nodes:
- **Producer** (prefill): runs prompt prefill, pushes KV cache via RDMA
- **Consumer** (decode): receives KV cache, runs autoregressive decode

Routing between clients and the P/D instances is handled by **atomesh**, a
lightweight Rust router that replaces the legacy Python proxy.

## Prerequisites

- Two nodes with AMD MI300X GPUs (8 GPUs each for TP=8)
- RDMA network connectivity between nodes (RoCE or InfiniBand)
- Docker installed on both nodes
- Producer and consumer should be in the **same network partition** for best accuracy

## Setup

### 1. Pull Docker Image

Pre-built images include ATOM, Mooncake, atomesh, and all RDMA dependencies:

```bash
docker pull rocm/atom-dev:latest
```

### 2. Start Container

On **each node**, use the provided script which auto-detects the RDMA NIC type
(bnxt/ionic/mlx5) and mounts the correct host ibverbs provider libraries:

```bash
DOCKER_IMAGE=rocm/atom-dev:latest bash atom/mesh/scripts/docker_start.sh
```

Then enter the container:

```bash
docker exec -it atom_mesh bash
```

All remaining commands are run **inside the container**.

## RDMA rails and HCA registration

By default, each Mooncake worker registers GPU memory on its GPU-local HCA.
Keep this default when the producer and consumer use matching, mutually
reachable rails, as in a two-node deployment with same-index GPU rank mapping.

For cross-rail transfers, such as prefill on physical GPUs `0,1` and decode on
`4,5` on a rail-separated fabric, enable alternate HCA registration on the
**decode** workers. Add these fields to the consumer's Mooncake
`--kv-transfer-config`:

```json
{
  "kv_role": "kv_consumer",
  "kv_connector": "mooncake",
  "handshake_port": 6301,
  "ib_enable_alternate_hca": true,
  "ib_hca_count": 8
}
```

Set `MC_ENABLE_DEST_DEVICE_AFFINITY=1` in both server environments before
starting them, as shown below. It lets the Mooncake initiator select a
destination HCA reachable from its local rail. The router does not register
GPU memory and does not need this setting.

- `ib_enable_alternate_hca` defaults to `false`. When enabled, ATOM keeps the
  GPU-local HCA first and adds available HCAs from indices `0` through
  `ib_hca_count - 1`, skipping missing devices and duplicates.
- `ib_hca_count` defaults to `8` and must be positive when automatic alternate
  HCA selection is enabled. It is an index range to scan, not a guarantee that
  eight HCAs will be selected. Automatic selection checks `rdmaN`, then
  `ionic_N`, for each physical GPU/HCA index; GPU masking is resolved through
  `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`.
- An explicit `ib_device` in the connector config, or
  `ATOM_MOONCAKE_IB_DEVICE` in the environment, overrides automatic selection.
  Comma-separated lists are trimmed and deduplicated; the first device is used
  for RDMA-local IP selection. Use device names and a GPU/HCA mapping appropriate
  to your fabric.
- Alternate registration requires GPU memory to be registrable on the selected
  HCAs. It does not fix driver-level GPU memory registration failures.
  `protocol: "tcp"` bypasses RDMA device selection entirely.

## Quick Start

### Step 1: Start Producer (prefill node)

```bash
AITER_LOG_LEVEL=WARNING \
MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-R1/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8010 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "handshake_port": 6301
  }' \
  2>&1 | tee producer.log
```

### Step 2: Start Consumer (decode node)

```bash
AITER_LOG_LEVEL=WARNING \
MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-R1/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8020 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "handshake_port": 6301
  }' \
  2>&1 | tee consumer.log
```

### Step 3: Start Router (atomesh)

Once both servers are healthy, start the atomesh router on either node:

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode  "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker
```

Send requests to the router at `http://<router-ip>:8000`.

## DeepSeek V4-Pro

V4-Pro requires additional env vars for its hash-routed MoE to work correctly in PD mode.

### Step 1: Start Producer (prefill node)

```bash
export LOCAL_IP=<this-node-ip>

AITER_BF16_FP8_MOE_BOUND=0 \
ATOM_MOE_GU_ITLV=1 \
AITER_LOG_LEVEL=WARNING \
MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8010 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "handshake_port": 6301
  }' \
  2>&1 | tee producer.log
```

### Step 2: Start Consumer (decode node)

```bash
AITER_BF16_FP8_MOE_BOUND=0 \
ATOM_MOE_GU_ITLV=1 \
AITER_LOG_LEVEL=WARNING \
MC_ENABLE_DEST_DEVICE_AFFINITY=1 \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8020 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "handshake_port": 6301
  }' \
  2>&1 | tee consumer.log
```

### Step 3: Start Router (atomesh)

```bash
export PREFILL_IP=<prefill-node-ip>
export DECODE_IP=<decode-node-ip>

atomesh launch \
    --host 0.0.0.0 --port 8000 \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:8010" \
    --decode  "http://${DECODE_IP}:8020" \
    --policy random \
    --backend atom \
    --log-level info \
    --disable-health-check \
    --disable-circuit-breaker
```

V4-specific env vars:
- `AITER_BF16_FP8_MOE_BOUND=0` — disables the BF16↔FP8 MoE boundary optimization (required for PD correctness)
- `ATOM_MOE_GU_ITLV=1` — enables MoE gate-up interleaving for V4's hash-routed expert selection

---

## Single-Node PD: NUMA Binding

When prefill and decode run on the **same** node — each a separate process pinned
to a GPU subset via `HIP_VISIBLE_DEVICES` — bind every worker's CPU threads
(especially mooncake's native RDMA threads) and its memory to the GPU's **local
NUMA node**. On a 2-socket box this removes the cross-socket GPU launch bubbles
that otherwise dominate prefill.

### Mechanism

`ATOM_NUMA_BIND` runs at the top of `AsyncIOProc.__init__` — before any large
allocation or native (mooncake) thread spawn — calling `sched_setaffinity` +
libnuma `numa_set_preferred`. Child threads inherit the mask and Linux
first-touch lands memory on the local node. See `atom/utils/numa_utils.py` and
`atom/model_engine/async_proc.py`. This replaces hand-rolled `taskset` / the old
NUMA-blind `ATOM_CPU_AFFINITY` linear slice.

### Env vars

| Variable | Default | Meaning |
|---|---|---|
| `ATOM_NUMA_BIND` | `0` (off) | Master switch; `=1` enables binding |
| `ATOM_NUMA_NODE` | empty (auto) | Explicit node id(s); empty = auto-detect (amdsmi → sysfs) |
| `ATOM_AUTO_NUMA_BIND` | `1` | Auto-detect toggle (rarely changed) |
| `ATOM_CRASH_ON_NUMA_BIND_FAILURE` | `0` | Raise instead of warn on bind failure |

### Check the topology first

```bash
# GPU -> NUMA node (amdsmi if present, else sysfs)
for d in /sys/class/drm/card*/device; do
  [ -e "$d/numa_node" ] && echo "$(basename $(realpath $d)) node=$(cat $d/numa_node)"
done | sort
# cpus per node
for n in /sys/devices/system/node/node*/cpulist; do echo "$n: $(cat $n)"; done
```

Example (2-socket, 8-GPU box): physical GPU 0–3 → node 0, 4–7 → node 1.

### Per-process configuration

Set a single node id per process (it broadcasts to all `tp` ranks in that
process):

| Process | `HIP_VISIBLE_DEVICES` | node | Config |
|---|---|---|---|
| prefill #1 | `0,1` | 0 | `ATOM_NUMA_BIND=1 ATOM_NUMA_NODE="0"` |
| prefill #2 | `2,3` | 0 | `ATOM_NUMA_BIND=1 ATOM_NUMA_NODE="0"` |
| decode #1 | `4,5` | 1 | `ATOM_NUMA_BIND=1 ATOM_NUMA_NODE="1"` |
| decode #2 | `6,7` | 1 | `ATOM_NUMA_BIND=1 ATOM_NUMA_NODE="1"` |

```bash
export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE="0"          # node of this process's GPUs
export HIP_VISIBLE_DEVICES=0,1
python -m atom.entrypoints.openai_server ... -tp 2 ...
```

### Full launch scripts (1P1D example)

Set `NODE_IP` to this node's address. Prefill is the `kv_producer`, decode the
`kv_consumer`; they coordinate through mooncake on the shared `handshake_port`.

The decode example enables alternate HCA registration for the cross-rail
GPU `0,1` → GPU `4,5` layout described in
[RDMA rails and HCA registration](#rdma-rails-and-hca-registration).
If the GPU-local HCAs are already mutually reachable, omit
`ib_enable_alternate_hca` and `ib_hca_count` to keep local-only registration.
NUMA binding controls CPU and memory locality; it does not select RDMA rails.

**Prefill (GPU 0,1 → node 0):**

```bash
export NODE_IP=<this-node-ip>

export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE="0"
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export HIP_VISIBLE_DEVICES=0,1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export PYTHONUNBUFFERED=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export ATOM_DISABLE_MMAP=true
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model /data/models/MiniMax-M2.7/ \
    --host 0.0.0.0 --server-port 8010 \
    --trust-remote-code \
    -tp 2 \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.75 \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","handshake_port":6301}'
```

**Decode (GPU 4,5 → node 1):**

```bash
export NODE_IP=<this-node-ip>

export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE="1"
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export HIP_VISIBLE_DEVICES=4,5
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export PYTHONUNBUFFERED=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export ATOM_DISABLE_MMAP=true
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model /data/models/MiniMax-M2.7/ \
    --host 0.0.0.0 --server-port 8020 \
    --trust-remote-code \
    -tp 2 \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.75 \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","handshake_port":6301,"ib_enable_alternate_hca":true,"ib_hca_count":8}'
```

**Router (atomesh):**

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
    --disable-circuit-breaker
```

> Each process pins to the NUMA node local to its GPUs (`0,1`/`2,3` → node 0;
> `4,5`/`6,7` → node 1). For a 2P1D / 1P2D mesh, bump every deterministic id
> (`server-port`, `handshake_port`) per extra process so they don't collide.
> Send all client requests to the **router** endpoint (`http://<node-ip>:8000`).

### Indexing rule (important under HIP_VISIBLE_DEVICES masking)

- **auto** (no `ATOM_NUMA_NODE`): resolves through `_physical_index()`, mapping
  the process-local rank back to the real physical GPU via `HIP_VISIBLE_DEVICES`
  before querying its node. Masking is handled correctly — **prefer auto when
  amdsmi is available; it is portable and zero-config.**
- **explicit** `ATOM_NUMA_NODE`: does **not** go through `_physical_index`; it is
  indexed by **process-local rank** (`0..tp-1`). Write the node(s) of the GPUs
  visible to *this* process, in local-rank order. A single value applies to all
  ranks — convenient when a process's GPUs are all on one node.

> If amdsmi is not installed, auto falls back to a sysfs scan (works when ROCm
> enumerates GPUs in PCI-BDF order). When the node layout is fixed and known,
> explicit `ATOM_NUMA_NODE` is the most deterministic choice.

### Verify

Each process logs one line per worker:

```
NUMA bind (ModelRunner0/2): gpu=0 -> node 0 (64 cores)
```

`tp2` → 2 lines with the expected node ids. A failure logs
`NUMA bind ... failed ...` — in Docker add `--cap-add SYS_NICE`, or set
`ATOM_NUMA_NODE` explicitly.

### CI / portability

When the target topology is unknown (e.g. a CI runner), set **only**
`ATOM_NUMA_BIND=1` and let auto-detect pick the node per machine — do not
hardcode `ATOM_NUMA_NODE`.

---

## Accuracy Validation

### DeepSeek-R1

### Step 4: Validate Accuracy

Run GSM8K evaluation against the **router** endpoint:

```bash
lm_eval --model local-chat-completions \
  --model_args "model=DeepSeek-R1,base_url=http://${ROUTER_IP}:8000/v1,tokenizer_backend=huggingface,pretrained=/data/models/DeepSeek-R1/" \
  --tasks gsm8k_cot \
  --batch_size 1 \
  --limit 100 \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --predict_only \
  --log_samples \
  --output_path results/ \
  --gen_kwargs "max_tokens=8192,temperature=0.6"
```

Expected accuracy: ~0.95-0.96 (matching non-PD baseline).
```
ewshot: None, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value|   |Stderr|
|-----|------:|----------------|-----:|-----------|---|----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  | 0.96|±  | 0.028|
|     |       |strict-match    |     5|exact_match|↑  | 0.96|±  | 0.028|
```

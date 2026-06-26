# PD Disaggregation with Mooncake (RDMA)

Prefill-Decode disaggregation splits inference into two stages on separate nodes:
- **Producer** (prefill): runs prompt prefill, pushes KV cache via RDMA
- **Consumer** (decode): receives KV cache, runs autoregressive decode

## Prerequisites

- Two nodes with AMD MI300X GPUs (8 GPUs each for TP=8)
- RDMA network connectivity between nodes (RoCE or InfiniBand)
- Mooncake package installed (`pip install mooncake`)
- Producer and consumer should be in the **same network partition** for best accuracy

## Building Mooncake for ROCm

If Mooncake is not pre-installed in your Docker image, build from source:

### 1. System dependencies

```bash
apt update && apt install -y \
    zip unzip wget gcc make libtool autoconf cmake \
    librdmacm-dev rdmacm-utils infiniband-diags ibverbs-utils perftest ethtool \
    libibverbs-dev rdma-core \
    openssh-server openmpi-bin openmpi-common libopenmpi-dev
```

### 2. Install Go 1.22.2

Mooncake's etcd wrapper requires Go. **Must use Go 1.22.2** — Go 1.23+ causes `mallocHeaderSize redeclared` errors.

```bash
apt remove -y golang golang-go 2>/dev/null || true
rm -rf /usr/local/go
wget https://go.dev/dl/go1.22.2.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.22.2.linux-amd64.tar.gz
rm go1.22.2.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
go version  # expect: go1.22.2
```

### 3. Clone and build

Use the SGLang-validated commit:

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git checkout b6a841dc78c707ec655a563453277d969fb8f38d
git submodule update --init --recursive

# Install C++ dependencies (etcd, protobuf, gRPC, abseil)
bash dependencies.sh -y

# CMake build
mkdir -p build && cd build
cmake .. -DUSE_HIP=ON -DUSE_ETCD=ON
make -j$(nproc)
make install
```

Key flags: `-DUSE_HIP=ON` for ROCm (default is CUDA), `-DUSE_ETCD=ON` for metadata store.

### 4. Install Python package

```bash
cd ../mooncake-wheel
pip install .

# Copy compiled .so to Python package
MOONCAKE_DIR=$(python -c "import mooncake; print(mooncake.__path__[0])")
cp ../build/mooncake-integration/engine.cpython-*-linux-gnu.so "$MOONCAKE_DIR/"

ldconfig
```

### 5. Verify

```bash
python -c "from mooncake.engine import TransferEngine; print('Mooncake ROCm OK')"
```

If you see `libglog.so` or other missing library errors, install them (`apt install -y libgoogle-glog-dev`) and re-run `ldconfig`.

> **Important**: Producer and consumer nodes must use the **same Mooncake build version**. Mismatched versions cause `Corrupted segment descriptor` errors due to incompatible metadata serialization formats.

## Quick Start

### Step 0: Find local IP

On each node, find the network IP (not loopback):

```bash
export LOCAL_IP=$(ip addr show | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | cut -d/ -f1 | head -1)
echo "Local IP: ${LOCAL_IP}"
```

### Step 1: Start Proxy (on producer node)

The proxy handles routing between producer and consumer:

```bash
python -m atom.kv_transfer.disaggregation.proxy --port 10001
```

### Step 2: Start Producer (prefill node)

```bash
ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=lo \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-R1/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8003 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${LOCAL_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8003
  }' \
  2>&1 | tee producer.log
```

### Step 3: Start Consumer (decode node)

Replace `PRODUCER_IP` with the producer node's IP:

```bash
export PRODUCER_IP=<producer-node-ip>

ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=lo \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-R1/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8004 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${PRODUCER_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8004
  }' \
  2>&1 | tee consumer.log
```

## DeepSeek V4-Pro

V4-Pro requires additional env vars for its hash-routed MoE to work correctly in PD mode.

### Step 1: Start Proxy

```bash
python -m atom.kv_transfer.disaggregation.proxy --port 10001
```

### Step 2: Start Producer (prefill node)

```bash
export LOCAL_IP=<this-node-ip>

AITER_BF16_FP8_MOE_BOUND=0 \
ATOM_MOE_GU_ITLV=1 \
ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=lo \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8003 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${LOCAL_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8003
  }' \
  2>&1 | tee producer.log
```

### Step 3: Start Consumer (decode node)

```bash
export PRODUCER_IP=<producer-node-ip>

AITER_BF16_FP8_MOE_BOUND=0 \
ATOM_MOE_GU_ITLV=1 \
ATOM_DISABLE_MMAP=true \
NCCL_SOCKET_IFNAME=eno0 \
AITER_LOG_LEVEL=WARNING \
python -m atom.entrypoints.openai_server \
  --model /data/models/DeepSeek-V4-Pro/ \
  --kv_cache_dtype fp8 \
  -tp 8 \
  --server-port 8004 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "'"${PRODUCER_IP}"'",
    "proxy_ping_port": 36367,
    "http_port": 8004
  }' \
  2>&1 | tee consumer.log
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

**Prefill (GPU 0,1 → node 0):**

```bash
export NODE_IP=<this-node-ip>

export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE="0"
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export HIP_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export ATOM_DISABLE_MMAP=true
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model /data/models/MiniMax-M2.7/ \
    --host 0.0.0.0 --server-port 8030 \
    --trust-remote-code \
    -tp 2 \
    --port 8006 \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.75 \
    --torch-profiler-dir /it-share/lirzhang/trace/prefill \
    --kv-transfer-config '{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6301}'
```

**Decode (GPU 4,5 → node 1):**

```bash
export NODE_IP=<this-node-ip>

export ATOM_NUMA_BIND=1
export ATOM_NUMA_NODE="1"
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1
export HIP_VISIBLE_DEVICES=4,5
export PYTHONUNBUFFERED=1
export ATOM_HOST_IP=${NODE_IP}
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/mooncake:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export ATOM_DISABLE_MMAP=true
rm -rf /root/.cache/atom/* 2>/dev/null || true

python3 -m atom.entrypoints.openai_server \
    --model /data/models/MiniMax-M2.7/ \
    --host 0.0.0.0 --server-port 8031 \
    --trust-remote-code \
    -tp 2 \
    --port 8007 \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.75 \
    --torch-profiler-dir /it-share/lirzhang/trace/decode \
    --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"'"${NODE_IP}"'","handshake_port":6302,"http_port":8041}'
```

> Each process pins to the NUMA node local to its GPUs (`0,1`/`2,3` → node 0;
> `4,5`/`6,7` → node 1). For a 2P1D / 1P2D mesh, bump every deterministic id
> (`server-port`, `--port`, `handshake_port`, `http_port`) per extra process so
> they don't collide.

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

Run GSM8K evaluation against the consumer endpoint:

```bash
lm_eval --model local-chat-completions \
  --model_args "model=DeepSeek-R1,base_url=http://${CONSUMER_IP}:8004/v1,tokenizer_backend=huggingface,pretrained=/data/models/DeepSeek-R1/" \
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
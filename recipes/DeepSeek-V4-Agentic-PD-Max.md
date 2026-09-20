# DeepSeek-V4-Pro agentic on ATOM, PD-disaggregated — max throughput

The two-node companion to
[`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md). That
file is one node, no disaggregation, and is the right starting point. This one
splits prefill from decode across two nodes and adds a CPU KV offload tier,
which is what the workload needs at high concurrency.

- Hardware: MI355X ×8 per node, **two nodes**, prefill/decode disaggregated (1P1D)
- Model: `DeepSeek-V4-Pro-0813`, FP4 weights, FP8 KV, FP4 index cache;
  `dspark` with 3 speculative tokens and synthetic acceptance length 3.01
  (benchmark-only; see [Forced acceptance length](../docs/forced_acceptance_length.md))
- Transport: Mooncake RDMA, GID index 1
- Scenario: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Router: `atomesh`, PD mode, independent P/D rank selection; cache-aware on
  both sides for DP, round-robin for TP

Pick the section by the concurrency you are running:

| concurrency | section |
|---|---|
| 1 – 32 | [TP](#tp--concurrency-1--32) |
| 64 – 128 | [DP attention](#dp-attention--concurrency-64--128) |
| 256 | [DP attention with CPU offload](#dp-attention-with-cpu-offload--concurrency-256) |

Each concurrency section contains complete prefill and decode server commands,
including environment variables and KV transfer configuration. Replace the model
path and node IPs in the chosen section, and use the same request concurrency on
both server nodes. Each command runs from its own shell.

`MODEL_PATH` must contain both the 0813 model and its matching tokenizer files;
the server loads its tokenizer from `--model`. Each router/client command sets
`TOKENIZER_PATH` to a copy of the same tokenizer accessible on that host.

One note before you start: **below about 32 concurrency, a single node without PD
gives roughly twice this per-chip throughput** (1,484 against 737 tok/s/chip at
c=1). PD earns its keep from 64 up, where it buys 2.6–3.6× the per-user output
rate. Use the TP section if your deployment is already PD-disaggregated, not as
a reason to split two nodes for low concurrency.

## RDMA rail configuration

`ATOM_MOONCAKE_MATCHED_RAILS` is an opt-in setting for rail-isolated RDMA
fabrics, where a NIC can reach its corresponding NIC on another node but not
every remote NIC. The examples below enable it to support independent P/D GPU
ranks on that topology. If the default GPU-local HCA pairs are already mutually
reachable for every allowed P/D pairing, omit this export.

For matched-rail mode, set `ATOM_MOONCAKE_MATCHED_RAILS=auto` on both
prefill and decode, and leave `ATOM_MOONCAKE_IB_DEVICE` unset to select each
GPU's primary HCA. Auto mode discovers ACTIVE HCAs in that primary's numbered
name family and logs the resolved list. For example, an `ionic_2` primary
selects active `ionic_*` devices without including an unrelated `mlx5_0`.

The same HCA name must identify mutually reachable rails on both nodes.
Auto mode discovers local names and link state; it does not test cross-node
reachability. An explicit comma-separated allowlist remains available for custom
naming or restricting the selected rails. See
[Matched RDMA rails](../docs/mooncake_matched_rails.md) for details.

These settings control HCA selection. The GPU memory
registration workaround described in [If the servers OOM at
startup](#if-the-servers-oom-at-startup) addresses a separate driver-level issue.

## TP — concurrency 1 – 32

Set `CONC` to the target request concurrency in 1–32 on both nodes. Decode
captures every graph size from 1 through `min(64, CONC * 2)`; prefill uses
default graph sizes. TBO is off on both nodes.

### Prefill node

```bash
export CONC="<request concurrency>"
if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set CONC to a positive integer before starting the server." >&2
  exit 1
fi

export MODEL_PATH="<DeepSeek-V4-Pro-0813 checkpoint path or model ID>"
export TOKENIZER_PATH="$MODEL_PATH"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP="<PREFILL_IP>"
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
# For rail-isolated fabrics; omit if all default P/D HCA pairs are reachable.
export ATOM_MOONCAKE_MATCHED_RAILS=auto
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export ATOM_NUMA_BIND=1
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8010 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "<PREFILL_IP>",
    "handshake_port": 6301,
    "protocol": "rdma"
  }'
```

### Decode node

```bash
export CONC="<request concurrency>"
if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set CONC to a positive integer before starting the server." >&2
  exit 1
fi

export MODEL_PATH="<DeepSeek-V4-Pro-0813 checkpoint path or model ID>"
export TOKENIZER_PATH="$MODEL_PATH"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP="<DECODE_IP>"
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
# For rail-isolated fabrics; omit if all default P/D HCA pairs are reachable.
export ATOM_MOONCAKE_MATCHED_RAILS=auto
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export ATOM_NUMA_BIND=1
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

DENSE_CAPTURE_SIZES="$(python3 - "$CONC" <<'PY'
import json, sys
print(json.dumps(list(range(1, min(64, int(sys.argv[1]) * 2) + 1))))
PY
)"

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8020 \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$DENSE_CAPTURE_SIZES" \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "<DECODE_IP>",
    "handshake_port": 6301,
    "protocol": "rdma"
  }'
```

### Router

```bash
# Matching 0813 tokenizer files must be accessible on this host.
export TOKENIZER_PATH="<local 0813 tokenizer path or model ID>"

atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --prefill-policy round_robin --decode-policy round_robin \
  --atom-pd-rank-mapping-policy none \
  --backend atom --model-path "$TOKENIZER_PATH" \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## DP attention — concurrency 64 – 128

Set `CONC` to 64 or 128 on both nodes. Prefill uses TBO with
`GPU_MAX_HW_QUEUES=5`; decode keeps TBO off. Decode captures every per-rank
batch size from 1 through `CONC / 4`: 1–16 at 64c and 1–32 at 128c.
This gives twice the average request count per rank across the eight DP ranks.
Prefill uses default graph sizes. This configuration has no CPU offload tier.

With `--dp-aware`, the cache-aware router selects P/D ranks independently and
sends explicit rank hints. These take priority over engine-local session
affinity and load balancing, so `ATOM_DP_SESSION_AFFINITY` and
`ATOM_DP_LB_REQ_EQUIV` are not needed for this routing path.

### Prefill node

```bash
export CONC="<request concurrency>"
if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set CONC to a positive integer before starting the server." >&2
  exit 1
fi

export MODEL_PATH="<DeepSeek-V4-Pro-0813 checkpoint path or model ID>"
export TOKENIZER_PATH="$MODEL_PATH"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP="<PREFILL_IP>"
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
# For rail-isolated fabrics; omit if all default P/D HCA pairs are reachable.
export ATOM_MOONCAKE_MATCHED_RAILS=auto
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export ATOM_NUMA_BIND=1
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

export ATOM_ENABLE_PREFILL_DELAYER=0       # Disable cross-DP prefill coalescing
export GPU_MAX_HW_QUEUES=5

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8010 \
  --tensor-parallel-size 8 \
  --enable-dp-attention --enable-tbo \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{
    "kv_role": "kv_producer",
    "kv_connector": "mooncake",
    "proxy_ip": "<PREFILL_IP>",
    "handshake_port": 6301,
    "protocol": "rdma"
  }'
```

### Decode node

```bash
export CONC="<request concurrency>"
if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set CONC to a positive integer before starting the server." >&2
  exit 1
fi

export MODEL_PATH="<DeepSeek-V4-Pro-0813 checkpoint path or model ID>"
export TOKENIZER_PATH="$MODEL_PATH"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP="<DECODE_IP>"
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
# For rail-isolated fabrics; omit if all default P/D HCA pairs are reachable.
export ATOM_MOONCAKE_MATCHED_RAILS=auto
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export ATOM_NUMA_BIND=1
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

export ATOM_ENABLE_PREFILL_DELAYER=0       # Disable cross-DP prefill coalescing

DENSE_CAPTURE_SIZES="$(python3 - "$CONC" <<'PY'
import json, sys
print(json.dumps(list(range(1, int(sys.argv[1]) // 4 + 1))))
PY
)"

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8020 \
  --tensor-parallel-size 8 \
  --enable-dp-attention \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$DENSE_CAPTURE_SIZES" \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "<DECODE_IP>",
    "handshake_port": 6301,
    "protocol": "rdma"
  }'
```

### Router

```bash
# Matching 0813 tokenizer files must be accessible on this host.
export TOKENIZER_PATH="<local 0813 tokenizer path or model ID>"

atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --dp-aware --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 0.8 --balance-abs-threshold 20 --balance-rel-threshold 2.0 \
  --eviction-interval 300 --atom-pd-rank-mapping-policy none \
  --backend atom --model-path "$TOKENIZER_PATH" \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## DP attention with CPU offload — concurrency 256

The commands below use concurrency 256 on both server nodes and the client.
Replace `10.0.0.1` and `10.0.0.2` with the prefill and decode IPs
in both the server and router commands. Prefill uses TBO and the tuned CPU
offload tier; decode uses plain Mooncake without TBO or CPU offload.
Decode captures every per-rank batch size from 1 through `CONC / 4`
(1–64 at concurrency 256); prefill uses default graph sizes.

`lmcache.max_local_cpu_size` is **per worker**: 8 workers × 128 GiB = 1,024 GiB
for the CPU cache. Ensure at least **1,280 GiB of available host memory** before
startup, including 256 GiB of headroom. Check `psutil.virtual_memory().available`
against `(8 * size + 256) * 1024**3` bytes, where `size` is the per-worker cache
size in GiB.

### Prefill node

```bash
export CONC=256
if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set CONC to a positive integer before starting the server." >&2
  exit 1
fi

export MODEL_PATH="<DeepSeek-V4-Pro-0813 checkpoint path or model ID>"
export TOKENIZER_PATH="$MODEL_PATH"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP="10.0.0.1"
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
# For rail-isolated fabrics; omit if all default P/D HCA pairs are reachable.
export ATOM_MOONCAKE_MATCHED_RAILS=auto
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export ATOM_NUMA_BIND=1
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

export ATOM_ENABLE_PREFILL_DELAYER=0       # Disable cross-DP prefill coalescing
export GPU_MAX_HW_QUEUES=5

export PYTHONHASHSEED=0                  # Consistent LMCache hashes across processes
export OFFLOAD_COPY_WORKERS=1
export OFFLOAD_MIN_LOAD_TOKENS=8192
export OFFLOAD_SLOT_STAGING_SLOTS=4

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8010 \
  --tensor-parallel-size 8 \
  --enable-dp-attention --enable-tbo \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{
    "kv_connector": "multi",
    "connectors": [
      {
        "kv_role": "kv_producer",
        "kv_connector": "mooncake",
        "proxy_ip": "10.0.0.1",
        "handshake_port": 6301,
        "protocol": "rdma"
      },
      {
        "kv_connector": "lmcache_offload",
        "kv_role": "offload",
        "offload_layout": "hybrid",
        "max_pending_saves": 8,
        "slot_sidecar_staging_slots": 4,
        "lmcache.local_cpu": true,
        "lmcache.max_local_cpu_size": 128,
        "lmcache.local_disk": null,
        "lmcache.max_local_disk_size": 0,
        "lmcache.remote_url": null,
        "lmcache.chunk_size": 256,
        "lmcache.cache_policy": "LRU",
        "lmcache.lookup_server_worker_ids": [],
        "lmcache.store_location": "LocalCPUBackend",
        "lmcache.retrieve_locations": [
          "LocalCPUBackend"
        ]
      }
    ]
  }'
```

### Decode node

```bash
export CONC=256
if [[ ! "$CONC" =~ ^[1-9][0-9]*$ ]]; then
  echo "Set CONC to a positive integer before starting the server." >&2
  exit 1
fi

export MODEL_PATH="<DeepSeek-V4-Pro-0813 checkpoint path or model ID>"
export TOKENIZER_PATH="$MODEL_PATH"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export AITER_LOG_LEVEL=WARNING
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP="10.0.0.2"
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
# For rail-isolated fabrics; omit if all default P/D HCA pairs are reachable.
export ATOM_MOONCAKE_MATCHED_RAILS=auto
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export ATOM_NUMA_BIND=1
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

export ATOM_ENABLE_PREFILL_DELAYER=0       # Disable cross-DP prefill coalescing

DENSE_CAPTURE_SIZES="$(python3 - "$CONC" <<'PY'
import json, sys
print(json.dumps(list(range(1, int(sys.argv[1]) // 4 + 1))))
PY
)"

python3 -u -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8020 \
  --tensor-parallel-size 8 \
  --enable-dp-attention \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$DENSE_CAPTURE_SIZES" \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{
    "kv_role": "kv_consumer",
    "kv_connector": "mooncake",
    "proxy_ip": "10.0.0.2",
    "handshake_port": 6301,
    "protocol": "rdma"
  }'
```

### Router, then client

```bash
# Matching 0813 tokenizer files must be accessible on this host.
export TOKENIZER_PATH="<local 0813 tokenizer path or model ID>"

atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://10.0.0.1:8010 --decode http://10.0.0.2:8020 \
  --dp-aware --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 0.8 --balance-abs-threshold 20 --balance-rel-threshold 2.0 \
  --eviction-interval 300 --atom-pd-rank-mapping-policy none \
  --backend atom --model-path "$TOKENIZER_PATH" \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --tokenizer "$TOKENIZER_PATH" --tokenizer-trust-remote-code \
  --concurrency 256 --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126
```

Host memory: the eight workers use 1,024 GiB for the CPU cache. The prefill
node needs at least **1,280 GiB of available host memory** before startup,
including 256 GiB of headroom.

## The offload settings that matter

Both default to values this workload cannot live with, and neither is set by the
reference scripts. They are the two knobs that moved the needle most.

### `slot_sidecar_staging_slots` — default 1, use 4

A SLOT sidecar save snapshots the sliding-window ring into a connector-owned
staging row before the D2H copy. With one row per rank, ~200 saves/minute
contend for it, and the load path holds that same row for a whole batch. Losing
the race raises `SLOT snapshot was not acquired successfully`, and a failed
sidecar is **not retried** — `_sidecar_save_candidate` skips any boundary already
in `_failed_sidecar_saves`.

At c=256 with the default, **26% of sidecar saves failed**. With 4 rows, **1.5%**.
Each row costs `slot_bytes` = 27,142,400 B ≈ 25.9 MiB, so four rows is ~78 MiB
per rank — negligible against the KV pool.

Set it in both places; the code reads the config key first and falls back to the
environment:

```python
configured = extra.get("slot_sidecar_staging_slots")
if configured is None:
    configured = os.environ.get("OFFLOAD_SLOT_STAGING_SLOTS", "1")
```

### `max_pending_saves` — default `max(2, 2 × OFFLOAD_COPY_WORKERS)`, use 8

Save admission is a non-blocking semaphore. With the default of 2 it saturates
constantly at high concurrency: **6,758 rejections** in one 3,600 s run at
c=256, against **160** at 8.

A rejection is safe — the scheduler rolls the saved watermark back and re-emits
the range on the next step — but only on builds that carry that rollback. On
older builds a rejected save left a **permanent hole** in the persisted prefix,
which reads downstream as SLOT publication timing out forever. If
`page_visibility_timeout` appears in the prefill log at all, stop and check the
build before tuning this knob upward.

### What they bought, at c=256

| | defaults (1 / 2) | tuned (4 / 8) |
|---|---|---|
| tok/s/chip | 21,599 | **30,686** (+42%) |
| TTFT p90 | 192.8 s | **36.0 s** (−81%) |
| cache hit | 91.0% | **94.6%** |
| sidecar failure rate | 26.0% | **1.5%** |
| save rejections | 6,758 | **160** |
| prefill `requests_waiting` | 146.5 | **27.3** |

The prefill queue is the mechanism. At the defaults, 57% of a 256-request
concurrency sat waiting to start, because the offload save path was contending
with prefill rather than serving it. Output-per-user drops from 56.2 to 35.1
tok/s over the same move — this trades interactivity for throughput rather than
being free.

## Client

```bash
# Matching 0813 tokenizer files must be accessible on this host.
export TOKENIZER_PATH="<local 0813 tokenizer path or model ID>"

export CONC="<request concurrency>"

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --tokenizer "$TOKENIZER_PATH" --tokenizer-trust-remote-code \
  --concurrency $CONC --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126
```

`--benchmark-duration` has a floor of **900** for this scenario; AIPerf refuses
anything shorter unless you pass `--unsafe-override`, which marks the run
`submission_valid=false`.

Budget warmup separately: mandatory primers plus
`--warmup-requests-per-lane × lanes` are not covered by
`--benchmark-duration`. The current setting is 10 per lane; compare runs with
the same warmup setting.

## What to watch in the prefill log

```bash
grep -c 'page_visibility_timeout'    server.log   # must be 0
grep -c 'SLOT sidecar save failed'   server.log   # /(failed+published) under ~2%
grep -c 'save rejected'              server.log   # hundreds, not thousands
grep -c 'SLOT sidecar load restored' server.log   # must be non-zero
```

`SLOT sidecar load restored` is the only line that proves a full PAGE+SLOT round
trip through the CPU tier; it is emitted only after both succeed, and
deliberately excludes a PAGE-only hit or an HBM prefix-cache hit.

## Measured

16 chips, 3,600 s measurement per cell in the historical table below.
`tok/s/chip` is `(ΣISL + ΣOSL) / duration / 16`, using successful profiling
requests and the span from their first start to last completion. It counts input
tokens, so it is dominated by prefix-cache reads rather than compute. `x` is
AIPerf's `Output Token Throughput Per User` at p90.

The historical rows used the earlier model/MTP and router configuration,
with memory fraction 0.75 prefill / 0.70 decode.

With 0813 + `dspark` at c=128, independent P/D ranks, P cache-aware,
and **prefill TBO disabled** (the historical configuration):
**28,026.535 tok/s/chip** was measured over 3,600 s with D round-robin
(cache hit 95.0%); **~28,600 tok/s/chip** is the estimated 3,600 s throughput
with D cache-aware. The latter is an extrapolation, not a completed 3,600 s run.

| conc | mode | offload | tok/s/chip | x (tok/s/user) | ITL p90 | TTFT p90 | cache hit |
|---|---|---|---|---|---|---|---|
| 1 | TP | — | 737 | 149.3 | 7.5 ms | 1.9 s | 96.8% |
| 2 | TP | — | 789 | 145.8 | 7.7 ms | 1.6 s | 95.6% |
| 8 | TP | — | 2,512 | 138.9 | 9.2 ms | 1.6 s | 97.1% |
| 16 | TP | — | 4,442 | 123.1 | 11.8 ms | 2.0 s | 96.6% |
| 64 | DP | — | 15,137 | 69.1 | 19.4 ms | 7.8 s | 96.1% |
| 128 | DP | — | 21,652 | 57.5 | 29.6 ms | 15.0 s | 94.7% |
| 256 | DP | defaults | 21,599 | 56.2 | 31.0 ms | 192.8 s | 91.0% |
| 256 | DP | **tuned** | **30,686** | 35.1 | 34.3 ms | **36.0 s** | **94.6%** |

The two c=256 rows are the same run with and without the offload settings in
this recipe: `defaults` is `max_pending_saves=2` and
`slot_sidecar_staging_slots=1`, `tuned` is 8 and 4.

The c=64 and c=128 rows carry no offload tier at all, so the offload settings
section does not apply to them.

Three caveats on that pair. The tuned run sampled a longer trace
(`isl` p50 84,176 against 71,141), and `tok/s/chip` counts input tokens, so
perhaps a fifth of the +42% is the workload rather than the settings — the TTFT
and cache-hit moves are not affected by this. Output-per-user falls 56.2 → 35.1,
so this buys throughput with interactivity rather than for free. And both c=256
rows ran with `--max-num-seqs 256`; neither measures the updated
0813 + `dspark` configuration.

Two more things to read. **c=128 is the knee at default settings** — c=256
doubles the concurrency for no throughput and 13× the TTFT, and it is the tuned
settings that break that ceiling rather than the concurrency. And a 1P2D variant
measured at c=256 — 24 chips, 15,774 tok/s/chip, x=75.3, ITL p90 18.9 ms — shows
that adding *decode* capacity buys 34% interactivity and gives up 27% per-chip
throughput, because decode was never the constraint.

## Extending past 1P1D

Both `--prefill` and `--decode` accept repeats, so the router line extends to
xPyD. Keep `--dp-aware`, both cache-aware policies, thresholds 20/2, and
`--atom-pd-rank-mapping-policy none` for additional DP workers.

Each prefill node owns a private `LocalCPUBackend`; a prefix saved on one is
invisible to the others. Cache-aware routing preserves locality when load
permits. Verify matched-rail reachability across every P/D node pair.

## V4 cache and graph flags

The server commands use the following V4 cache and graph flags:

| flag | behavior |
|---|---|
| `--index-cache-dtype fp4` | Selects the FP4 index cache, including its data and scale regions in PD transfer. |
| `--cudagraph-mode FULL` | Pins full CUDA graph capture explicitly. |

## If the servers OOM at startup

The server commands use the default memory utilization. If startup runs out of
memory, set `--gpu-memory-utilization` for the affected node; registration and
workspace requirements vary by cluster.

The Crusoe MI355X cluster these numbers came from also needed a GPU-memory
registration workaround:

**An `LD_PRELOAD` shim** intercepting `ibv_reg_mr_iova2`. The native path
returns `EFAULT`/`EINVAL` on GPU memory, so the shim exports a dma-buf fd with
`hipMemGetHandleForAddressRange` and registers through `ibv_reg_dmabuf_mr`
instead. Where the native path works it succeeds first and the fallback never
runs, which is why the shim is not part of the recipe. It is what prints
`[hip-dmabuf-mr] registered GPU range ...`.

**A lower memory fraction.** At 0.9 the KV pool allocates and Mooncake registers
it, and then the first barrier in `allocate_kv_cache` cannot get 32 MiB — with
~43 GiB per chip still nominally free at 0.85, so this is not a budget overrun.
Cluster IT's guidance is `≤ 0.65`. For DP, 0.75 prefill / 0.70 decode completed a
3,600 s run. The current TP prefill failed at 0.75 on the first barrier and
started at 0.70. A prior 0.80 run passed smoke and then died in MoE stage-2:
starting is not evidence a value is safe.

Both are properties of that cluster, not of PD or DeepSeek-V4.

## Related

- [`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md) —
  single node, no disaggregation. Start there.
- [`DeepSeek-V4-Agentic-Benchmark.md`](DeepSeek-V4-Agentic-Benchmark.md) —
  cross-engine head-to-head.
- [`MiniMax-M3-Agentic-InferenceX.md`](MiniMax-M3-Agentic-InferenceX.md) — the
  offload tier's cache-policy knobs on a different model, under
  [Cache policies](MiniMax-M3-Agentic-InferenceX.md#cache-policies).

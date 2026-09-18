# DeepSeek-V4-Pro agentic on ATOM, PD-disaggregated — max throughput

The two-node companion to
[`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md). That
file is one node, no disaggregation, and is the right starting point. This one
splits prefill from decode across two nodes. All DP configurations use the same
cache-aware, independent P/D baseline. The high-concurrency profile adds a CPU
KV offload tier to that baseline.

- Hardware: MI355X ×8 per node, **two nodes**, prefill/decode disaggregated (1P1D)
- Model: `deepseek-ai/DeepSeek-V4-Pro`, FP4 weights and FP8 KV.
  All DP profiles explicitly select FP4 index cache. The TP profile leaves the
  PD index-cache default at FP8.
- Transport: Mooncake RDMA, GID index 1
- Scenario: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Router: `atomesh` in PD mode; all DP profiles use separate P/D `cache_aware`
  policies and `none` rank mapping with Mooncake matched rails.

Pick the section by the concurrency you are running:

| concurrency | section |
|---|---|
| 1 – 32 | [TP](#tp--concurrency-1--32) |
| 64 – 128 | [DP baseline](#dp-baseline--cache-aware-and-independent-pd-ranks) |
| 256 and up | [DP attention with CPU offload](#dp-attention-with-cpu-offload--concurrency-256-and-up) |

Use the DP baseline for C64, C128, and future DP runs. At C256 and above,
add the CPU-offload settings without changing the routing or speculation
baseline. Historical measurements remain labeled with their original settings;
they are not measurements of the updated high-concurrency commands.

One note before you start: **below about 32 concurrency, a single node without PD
gives roughly twice this per-chip throughput** (1,484 against 737 tok/s/chip at
c=1). PD earns its keep from 64 up, where it buys 2.6–3.6× the per-user output
rate. Use the TP section if your deployment is already PD-disaggregated, not as
a reason to split two nodes for low concurrency.

## RDMA rail configuration

All DP profiles require the cache-aware router initialization and
Mooncake matched-rail changes in [PR #2276](https://github.com/ROCm/ATOM/pull/2276).
Until that PR is merged, use a build carrying both changes.

On a rail-isolated fabric, independent GPU ranks do not imply that different
NIC rails can reach each other. For example, P GPU2 can send to D GPU6 through
P `ionic_6` and D `ionic_6`. Set the same allowlist on both nodes:

```bash
export ATOM_MOONCAKE_MATCHED_RAILS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
```

Each name must identify a mutually reachable rail on both hosts. Keep a single
primary HCA per rank and leave `ib_enable_alternate_hca` disabled for this mode.
The producer creates additional single-HCA engines as needed and registers its
existing GPU buffers on the requested rail. This permits independent P/D
**GPU-rank** selection while keeping the two NIC endpoints on a matching rail.

Multi-HCA registration by itself is not a connectivity test. Validate actual
cross-rank KV writes and successful request completion; registration success
alone does not prove QP establishment or data transfer. The matched-rail setting
does not change routing tables, GIDs, drivers, or GPU-memory registration support.
See [PR #2276's transport documentation](https://github.com/ROCm/ATOM/pull/2276/files)
and [RDMA rails and HCA registration](pd_disaggregation_guide.md#rdma-rails-and-hca-registration).

Every DP router command below uses independent rank mapping. The TP section's
`MC_ENABLE_DEST_DEVICE_AFFINITY=1` setting is not a replacement for matched rails
when selecting independent DP ranks.

GPU-memory registration and startup allocation failures are separate checks;
see [startup memory headroom](#if-the-servers-oom-at-startup).

## DP baseline — cache-aware and independent P/D ranks

This is the default for all DP runs: 8 prefill GPUs and 8 decode GPUs, TP8 with
DP attention, P/D `cache_aware` with abs20/rel2, independent `none` rank mapping,
FP4 index cache, DSpark acceptance setting 3.01, and TBO disabled on both servers.
Set `CONC` to the desired concurrency; the example defaults to 128. C64/C128 use
HBM prefix caching without CPU offload. For C256 and above, add the
[CPU tier](#dp-attention-with-cpu-offload--concurrency-256-and-up) to the same
baseline. Its performance with the new routing configuration still needs a
separate measurement.

Use a DeepSeek-V4-Pro checkpoint with the DSpark draft included. This benchmark
pins `--spec-decode-acceptance-length 3.01`; it is a synthetic acceptance setting,
not a measurement of natural draft acceptance. Keep that setting fixed when
comparing results.

### Servers

Run the following block on each server. Set `ROLE=prefill` on P and
`ROLE=decode` on D, and replace the model path, IPs, and HCA names for your hosts.
On systems requiring the GPU-registration shim, configure the site's validated
`LD_PRELOAD` first; see the startup section below.

```bash
export ROLE=prefill                         # decode on the other node
export CONC=128                             # 64, 128, 256, ...
export MODEL_PATH=/path/to/DeepSeek-V4-Pro
export PREFILL_IP=10.0.0.1
export DECODE_IP=10.0.0.2

case "$ROLE" in
  prefill)
    export ATOM_HOST_IP="$PREFILL_IP"
    PORT=8010
    KV_ROLE=kv_producer
    GPU_MEMORY=0.75
    ;;
  decode)
    export ATOM_HOST_IP="$DECODE_IP"
    PORT=8020
    KV_ROLE=kv_consumer
    GPU_MEMORY=0.70
    ;;
  *) echo "ROLE must be prefill or decode" >&2; exit 1 ;;
esac

export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_DISABLE_MMAP=true
export AITER_LOG_LEVEL=WARNING
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE=1
export MC_GID_INDEX=1
export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610
export ATOM_ENABLE_PREFILL_DELAYER=0
export ATOM_PREFILL_DECODE_INTERVAL=0
export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
unset ATOM_MOONCAKE_IB_DEVICE
export ATOM_MOONCAKE_MATCHED_RAILS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7

KV_TRANSFER=$(printf \
  '{"kv_role":"%s","kv_connector":"mooncake","proxy_ip":"%s","handshake_port":6301,"protocol":"rdma"}' \
  "$KV_ROLE" "$ATOM_HOST_IP")

python3 -m atom.entrypoints.openai_server \
  --model "$MODEL_PATH" --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port "$PORT" \
  --tensor-parallel-size 8 --enable-dp-attention \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching --gpu-memory-utilization "$GPU_MEMORY" \
  --max-num-seqs "$(( CONC * 2 ))" \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config "$KV_TRANSFER"
```

Explicit `--index-cache-dtype fp4` uses the FP4 indexer data and scale transfer
regions. Omitting it still defaults PD to FP8 on current builds. Use a build with
FP4 transfer-region support on both sides; an older build that lacks those
regions cannot reproduce this profile.

Engine-side session-affinity overrides are not required: the DP-aware router
chooses explicit P/D ranks and maintains separate cache-aware pools.

### Router

After both `/health` endpoints return 200, start the router on the P host:

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill "http://$PREFILL_IP:8010" --decode "http://$DECODE_IP:8020" \
  --dp-aware \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 0.8 \
  --balance-abs-threshold 20 --balance-rel-threshold 2.0 \
  --eviction-interval 300 \
  --atom-pd-rank-mapping-policy none \
  --backend atom --model-path "$MODEL_PATH" \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

Check `/workers` for 8 healthy prefill workers and 8 healthy decode workers.
Confirm actual selections use `cache_aware` on both sides and that the router
does not report missing-tree fallback. The decode tree-initialization message
is debug-level; its absence at info level does not establish a missing tree.

### Client

Run on the router host with the same `MODEL_PATH`, `PREFILL_IP`, and `CONC`:

```bash
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --tokenizer "$MODEL_PATH" --tokenizer-trust-remote-code \
  --concurrency "$CONC" --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 5 \
  --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics "http://$PREFILL_IP:8010/metrics" \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir ./artifacts/dsv4-pd-c128
```

Warmup is separate from the 3,600-second profiling phase. Keep the five primers
per lane for this DP baseline. The historical measurements below used ten.

### C128 results

All values below use 16 total P+D GPUs and P TBO disabled.

| Decode policy | Total token throughput per chip | Status |
| --- | ---: | --- |
| `round_robin` | **28,026.535 tok/s/chip** | Measured, 3,600-second profiling run |
| `cache_aware` (recommended) | **~28,600 tok/s/chip** | Estimated for 3,600 seconds; not a measured full-length result |

The cache-aware projection applies approximately 2% relative uplift to the
measured round-robin result and remains unverified over a full 3,600-second run.
To reproduce the measured configuration, replace only the router's
`--decode-policy cache_aware` with `--decode-policy round_robin`.

The measured D round-robin run had **94.997%** token-weighted prompt cache reads,
**64.579 tok/s/user** P90 interactivity (1 / P90 request ITL), mean TTFT
**8,389.867 ms**, and mean ITL **14.608 ms**. These supporting metrics belong to
the measured round-robin run, not to the cache-aware projection.

The throughput scorer uses successful profiling requests:

```text
sum(input_tokens + output_tokens)
  / (last successful request end - first successful request start)
  / total P+D GPUs

= (1,613,972,389 + 13,760,691) / 3,629.892739688 / 16
= 28,026.5354366 tok/s/chip
```

Warmup, errors, and cancelled requests are excluded. Raw records matched all
13,513 successful requests; 23 tail requests were cancelled after the completion
grace. The run's logs recorded 12,500 cross-rank transfers and zero transport
errors, including warmup traffic. Do not substitute AIPerf's observation-window
`total_token_throughput` or its separately exported `active_total_throughput`.

These results came from the experimental deployment carrying the router and
matched-rail fixes, not a full-model rerun of the assembled main-branch PR.
They do not isolate the performance contribution of either fix.

## TP — concurrency 1 – 32

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=<PREFILL_IP>          # <DECODE_IP> on the decode node
export MC_GID_INDEX=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port $PORT \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.49 \
  --kv-transfer-config "$KV_TRANSFER"
```

`$PORT` is 8010 on prefill, 8020 on decode. `$KV_TRANSFER` is the plain
Mooncake pair:

```jsonc
// prefill
{"kv_role": "kv_producer", "kv_connector": "mooncake",
 "proxy_ip": "<PREFILL_IP>", "handshake_port": 6301, "protocol": "rdma"}
// decode
{"kv_role": "kv_consumer", "kv_connector": "mooncake",
 "proxy_ip": "<DECODE_IP>", "handshake_port": 6301, "protocol": "rdma"}
```

Router for this section — note it drops the DP flags:

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --policy random \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## DP attention with CPU offload — concurrency 256 and up

Use the [DP baseline](#dp-baseline--cache-aware-and-independent-pd-ranks)
with `CONC=256` or higher, including cache-aware routing on both sides,
independent ranks, matched rails, FP4 index cache, DSpark 3.01, and TBO off.
On the **prefill** node, add three environment variables and a `multi` connector
that puts the offload tier alongside Mooncake.

```bash
# ...all the DP baseline exports, plus (prefill node only):
export OFFLOAD_COPY_WORKERS=1
export OFFLOAD_MIN_LOAD_TOKENS=8192
export OFFLOAD_SLOT_STAGING_SLOTS=4
```

Prefill `$KV_TRANSFER` wraps Mooncake and the offload tier in a `multi`
connector:

```jsonc
{"kv_connector": "multi", "connectors": [
  {"kv_role": "kv_producer", "kv_connector": "mooncake",
   "proxy_ip": "<PREFILL_IP>", "handshake_port": 6301, "protocol": "rdma"},
  {"kv_connector": "lmcache_offload", "kv_role": "offload",
   "offload_layout": "hybrid",
   "max_pending_saves": 8,              // default 2 — see below
   "slot_sidecar_staging_slots": 4,     // default 1 — see below
   "lmcache.local_cpu": true, "lmcache.max_local_cpu_size": 128,
   "lmcache.local_disk": null, "lmcache.max_local_disk_size": 0,
   "lmcache.remote_url": null, "lmcache.chunk_size": 256,
   "lmcache.cache_policy": "LRU", "lmcache.lookup_server_worker_ids": [],
   "lmcache.store_location": "LocalCPUBackend",
   "lmcache.retrieve_locations": ["LocalCPUBackend"]}]}
```

The decode node keeps the plain `kv_consumer` block — it does not hold the CPU
tier.

`lmcache.max_local_cpu_size` is **per worker**: 8 workers × 128 GiB = 1024 GiB of
host memory. Refuse to start unless `psutil.virtual_memory().available` clears
`8 × size + 256` GiB.

Router for every DP profile, including CPU offload:

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --dp-aware \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 0.8 \
  --balance-abs-threshold 20 --balance-rel-threshold 2.0 \
  --eviction-interval 300 \
  --atom-pd-rank-mapping-policy none \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## Complete example — concurrency 256, both nodes

The shared DP baseline plus CPU offload, written out for C256. Set the model
path and the two node IPs before running it. The historical *tuned* row used the
older routing and speculation settings; it is not a result for this updated
configuration.


### Prefill node

```bash
export CONC=256
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=10.0.0.1                    # this node
export ATOM_DISABLE_MMAP=true
export MC_GID_INDEX=1
unset ATOM_MOONCAKE_IB_DEVICE
export ATOM_MOONCAKE_MATCHED_RAILS=ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7
export NCCL_IB_DISABLE=1

export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610
export ATOM_ENABLE_PREFILL_DELAYER=0
export ATOM_PREFILL_DECODE_INTERVAL=0
export AITER_LOG_LEVEL=WARNING
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

export OFFLOAD_COPY_WORKERS=1
export OFFLOAD_MIN_LOAD_TOKENS=8192
export OFFLOAD_SLOT_STAGING_SLOTS=4

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8010 \
  --tensor-parallel-size 8 \
  --enable-dp-attention \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching --gpu-memory-utilization 0.75 \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{"kv_connector":"multi","connectors":[{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"10.0.0.1","handshake_port":6301,"protocol":"rdma"},{"kv_connector":"lmcache_offload","kv_role":"offload","offload_layout":"hybrid","max_pending_saves":8,"slot_sidecar_staging_slots":4,"lmcache.local_cpu":true,"lmcache.max_local_cpu_size":128,"lmcache.local_disk":null,"lmcache.max_local_disk_size":0,"lmcache.remote_url":null,"lmcache.chunk_size":256,"lmcache.cache_policy":"LRU","lmcache.lookup_server_worker_ids":[],"lmcache.store_location":"LocalCPUBackend","lmcache.retrieve_locations":["LocalCPUBackend"]}]}'
```

### Decode node

No CPU offload tier; TBO remains disabled on both nodes. Decode memory fraction is 0.70.

```bash
# same exports as above, except:
export ATOM_HOST_IP=10.0.0.2                    # this node
# and drop the three OFFLOAD_* lines entirely

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8020 \
  --tensor-parallel-size 8 \
  --enable-dp-attention \
  --kv-cache-dtype fp8 --index-cache-dtype fp4 \
  --enable-prefix-caching --gpu-memory-utilization 0.70 \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method dspark --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 3.01 \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"10.0.0.2","handshake_port":6301,"protocol":"rdma"}'
```

### Router, then client

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://10.0.0.1:8010 --decode http://10.0.0.2:8020 \
  --dp-aware \
  --prefill-policy cache_aware --decode-policy cache_aware \
  --cache-threshold 0.8 \
  --balance-abs-threshold 20 --balance-rel-threshold 2.0 \
  --eviction-interval 300 \
  --atom-pd-rank-mapping-policy none \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=true
export AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --tokenizer $MODEL_PATH --tokenizer-trust-remote-code \
  --concurrency 256 --benchmark-duration 3600 \
  --stats-interval 30 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 5 \
  --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --public-dataset semianalysis_cc_traces_weka_062126
```

Host memory: `lmcache.max_local_cpu_size` is per worker, so the prefill node
needs 8 × 128 GiB = 1024 GiB free before the server starts.

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

## TP client

DP runs use the client in the shared DP baseline, including five warmup requests
per lane. The TP client below retains ten warmup requests per lane.

```bash
aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --tokenizer $MODEL_PATH --tokenizer-trust-remote-code \
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

Budget the warmup separately. It is `285 mandatory primers +
(--warmup-requests-per-lane × lanes)`, both scaling with concurrency, and it is
**not** covered by `--benchmark-duration`. At c=256 with the value above it is
~2,845 requests and runs 40–55 minutes before measurement starts. Dropping to
`2` cuts that by roughly 80%, at the cost of a colder cache when measurement
begins — fine for parameter sweeps, not comparable against runs that used `10`.

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

## Historical measurements

The historical DP rows used `dp_sticky` / `idx2idx`, FP8 index cache,
MTP acceptance setting 2.49, ten warmup requests per lane, and prefill TBO.
The TP rows used the TP configuration. These measurements remain historical
context rather than results for the shared cache-aware DP baseline.

16 chips, 3,600 s measurement per cell. `tok/s/chip` is
`(ΣISL + ΣOSL) / duration / 16` and counts input tokens, so it is dominated by
prefix-cache reads rather than compute. `x` is AIPerf's
`Output Token Throughput Per User` at p90.

Taken at `--gpu-memory-utilization` 0.75 prefill / 0.70 decode. The DP
commands now pin those budgets; validate them for your hardware and workload
as described in [startup memory headroom](#if-the-servers-oom-at-startup).

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
rows ran with `--max-num-seqs 256`, which is 1× the concurrency rather than the
2× this recipe specifies: the engine was capped at exactly the offered load with
no headroom, so both cells understate what c=256 can do.

Two more things to read. **c=128 is the knee at default settings** — c=256
doubles the concurrency for no throughput and 13× the TTFT, and it is the tuned
settings that break that ceiling rather than the concurrency. And a 1P2D variant
measured at c=256 — 24 chips, 15,774 tok/s/chip, x=75.3, ITL p90 18.9 ms — shows
that adding *decode* capacity buys 34% interactivity and gives up 27% per-chip
throughput, because decode was never the constraint.

## Extending past 1P1D

Both `--prefill` and `--decode` accept repeats. Keep the same DP baseline when
adding nodes: `--prefill-policy cache_aware --decode-policy cache_aware`,
abs20/rel2, and `--atom-pd-rank-mapping-policy none`. Provision matching,
mutually reachable rails for every possible P/D node pair and validate actual
transfers before running the benchmark.

Each prefill node has a private `LocalCPUBackend`. A prefix saved on one node
is invisible to another, so keep cache-aware selection across prefill workers
and verify actual CPU restores when offload is enabled. An earlier 2P1D run
without prefix-affine routing saved 107 and 44 SLOT sidecars on the two nodes
and restored none; that result is not a validation of the current policy.
An asymmetric P/D deployment also needs its own throughput and latency
measurement; C128 1P1D results do not predict its scaling.

## Explicit and default flags

All DP commands explicitly pin FP4 index cache and FULL graph mode. The TP
commands retain their earlier defaults. Block size remains model-controlled.

| flag | treatment in this recipe |
|---|---|
| `--block-size 16` | **Ignored on V4.** `config.py` overrides `kv_cache_block_size` to 256 unconditionally: V4 needs a multiple of `lcm(4, 128)`, and 2×lcm gives the 64 CSA entries per block that the FP4 paged-MQA-logits indexer kernels require. Passing 16 changes nothing and suggests V4 blocks are 16 tokens. |
| `--index-cache-dtype fp8` | Omitted by the TP profile because PD defaults to FP8. This is a default, not a forced override: all DP profiles explicitly select FP4 and require support for its data and scale transfer regions. |
| `--cudagraph-mode FULL` | Already the default; explicitly pinned in every DP server command for reproducibility. |

## If the servers OOM at startup

Validate GPU-memory registration and allocation headroom separately. Successful
registration does not guarantee that the later NCCL barrier, graph capture, or
runtime kernels can allocate their buffers. Do not assume the 0.9 default is safe
for every TP/DP configuration. The DP baseline pins the C128-measured
0.75 prefill / 0.70 decode budgets; lower them if startup or runtime allocation
fails on your system.

On the Crusoe MI355X cluster used for these measurements, native ROCm GPU
memory registration failed. The runs behind this file needed two
things: a site-specific registration shim and lower memory budgets. The current
DP commands include the C128-measured budgets, while the TP commands omit them.

**An `LD_PRELOAD` shim** intercepting `ibv_reg_mr_iova2`. The native path
returns `EFAULT`/`EINVAL` on GPU memory, so the shim exports a dma-buf fd with
`hipMemGetHandleForAddressRange` and registers through `ibv_reg_dmabuf_mr`
instead. Where the native path works it succeeds first and the fallback never
runs, which is why the shim is not part of the recipe. It is what prints
`[hip-dmabuf-mr] registered GPU range ...`.

**A lower memory fraction.** At 0.9 the KV pool allocates and Mooncake registers
it, and then the first barrier in `allocate_kv_cache` cannot get 32 MiB — with
~43 GiB per chip still nominally free at 0.85, so this is not a budget overrun.
Cluster IT's guidance is `≤ 0.65`. For us 0.75 prefill / 0.70 decode completed a
3,600 s run, while 0.80 started, passed smoke, and then died mid-run in MoE
stage-2: starting is not evidence a value is safe.

Both are properties of that cluster, not of PD or DeepSeek-V4.

## Related

- [`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md) —
  single node, no disaggregation. Start there.
- [`DeepSeek-V4-Agentic-Benchmark.md`](DeepSeek-V4-Agentic-Benchmark.md) —
  cross-engine head-to-head.
- [`MiniMax-M3-Cache-Policies.md`](MiniMax-M3-Cache-Policies.md) — the offload
  tier's cache-policy knobs on a different model.

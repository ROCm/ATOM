# DeepSeek-V4-Pro agentic on ATOM, PD-disaggregated — max throughput

The two-node companion to
[`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md). That
file is one node, no disaggregation, and is the right starting point. This one
splits prefill from decode across two nodes and adds a CPU KV offload tier,
which is what the workload needs at high concurrency.

- Hardware: MI355X ×8 per node, **two nodes**, prefill/decode disaggregated (1P1D)
- Model: `deepseek-ai/DeepSeek-V4-Pro`, FP4 weights, FP8 KV, FP8 index cache
  (FP8 is forced under PD — the single-node recipe's FP4 indexer has no
  Mooncake staging layout, and `atom/config.py` rewrites it)
- Transport: Mooncake RDMA, GID index 1
- Scenario: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Router: `atomesh`, PD mode, `idx2idx` rank mapping

Pick the section by the concurrency you are running:

| concurrency | section |
|---|---|
| 1 – 32 | [TP](#tp--concurrency-1--32) |
| 64 – 128 | [DP attention](#dp-attention--concurrency-64--128) |
| 256 and up | [DP attention with CPU offload](#dp-attention-with-cpu-offload--concurrency-256-and-up) |

Each section is complete on its own — server commands for both nodes, the router
line, and the client. Nothing carries over between them except the model path
and the two node IPs.

One note before you start: **below about 32 concurrency, a single node without PD
gives roughly twice this per-chip throughput** (1,484 against 737 tok/s/chip at
c=1). PD earns its keep from 64 up, where it buys 2.6–3.6× the per-user output
rate. Use the TP section if your deployment is already PD-disaggregated, not as
a reason to split two nodes for low concurrency.

## RDMA rail configuration

The two-node TP8 examples assume matching, mutually reachable GPU-local RDMA
rails; the DP examples use `idx2idx` rank mapping. Keep the default local-HCA
registration for this topology: the Mooncake configs below intentionally omit
`ib_enable_alternate_hca` and `ib_hca_count`.

Set `MC_ENABLE_DEST_DEVICE_AFFINITY=1` on both server nodes, including the
prefill server using the `multi` connector. The exports below include it so
Mooncake can select a destination HCA reachable from the initiator's local rail.

If you adapt this recipe to a cross-rail layout, add
`"ib_enable_alternate_hca": true` and `"ib_hca_count": 8` to the **decode
Mooncake connector** config. If decode uses a `multi` connector, put these
fields inside its Mooncake entry. Check the actual HCA names and GPU/rail
mapping first; see [RDMA rails and HCA registration](pd_disaggregation_guide.md#rdma-rails-and-hca-registration)
for defaults, explicit-device overrides, and a complete consumer example.

These settings control HCA selection and reachability. The GPU memory
registration workaround described in [If the servers OOM at
startup](#if-the-servers-oom-at-startup) addresses a separate driver-level issue.

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

## DP attention — concurrency 64 – 128

Everything in the TP section, plus four environment variables, two flags, and a
higher memory fraction. No offload tier: at these concurrencies the HBM prefix
cache carries the reuse on its own (measured hit 96.1% at c=64, 94.7% at c=128).

```bash
# ...the TP exports above, plus:
export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5
export ATOM_DP_SESSION_AFFINITY=1
export ATOM_DP_LB_REQ_EQUIV=512
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port $PORT \
  --tensor-parallel-size 8 \
  --enable-dp-attention --enable-tbo \
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

`--enable-tbo` (two-batch overlap) goes on the **prefill node only**. The decode
node runs DP attention without it.

`ATOM_DP_SESSION_AFFINITY=1` is the load-bearing one of those four. It keeps a
trajectory on one rank, which is what the agentic scenario's prefix reuse
depends on; without it the DP path loses most of its cache hit.

`$KV_TRANSFER` is the same plain Mooncake pair as the TP section.

## DP attention with CPU offload — concurrency 256 and up

Identical server flags to the DP section above. The only change is on the
**prefill** node: three more environment variables, and a `multi` connector that
puts the offload tier alongside Mooncake.

```bash
# ...all the DP exports above, plus (prefill node only):
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

Router for both DP sections (same line for either):

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --dp-aware --policy dp_sticky \
  --atom-pd-rank-mapping-policy idx2idx \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## Complete example — concurrency 256, both nodes

The config the *Measured* tuned row was produced with, written out in full so it
can be copied without resolving any of the `$VAR` above. Substitute only the two
IPs and `$MODEL_PATH`.


### Prefill node

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=10.0.0.1                    # this node
export ATOM_DISABLE_MMAP=true
export MC_GID_INDEX=1
export MC_ENABLE_DEST_DEVICE_AFFINITY=1
export NCCL_IB_DISABLE=1

export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5
export ATOM_DP_SESSION_AFFINITY=1
export ATOM_DP_LB_REQ_EQUIV=512
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

export OFFLOAD_COPY_WORKERS=1
export OFFLOAD_MIN_LOAD_TOKENS=8192
export OFFLOAD_SLOT_STAGING_SLOTS=4

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8010 \
  --tensor-parallel-size 8 \
  --enable-dp-attention --enable-tbo \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.49 \
  --kv-transfer-config '{"kv_connector":"multi","connectors":[{"kv_role":"kv_producer","kv_connector":"mooncake","proxy_ip":"10.0.0.1","handshake_port":6301,"protocol":"rdma"},{"kv_connector":"lmcache_offload","kv_role":"offload","offload_layout":"hybrid","max_pending_saves":8,"slot_sidecar_staging_slots":4,"lmcache.local_cpu":true,"lmcache.max_local_cpu_size":128,"lmcache.local_disk":null,"lmcache.max_local_disk_size":0,"lmcache.remote_url":null,"lmcache.chunk_size":256,"lmcache.cache_policy":"LRU","lmcache.lookup_server_worker_ids":[],"lmcache.store_location":"LocalCPUBackend","lmcache.retrieve_locations":["LocalCPUBackend"]}]}'
```

### Decode node

No offload tier, no `--enable-tbo`, memory fraction 0.70.

```bash
# same exports as above, except:
export ATOM_HOST_IP=10.0.0.2                    # this node
# and drop the three OFFLOAD_* lines entirely

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port 8020 \
  --tensor-parallel-size 8 \
  --enable-dp-attention \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.49 \
  --kv-transfer-config '{"kv_role":"kv_consumer","kv_connector":"mooncake","proxy_ip":"10.0.0.2","handshake_port":6301,"protocol":"rdma"}'
```

### Router, then client

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://10.0.0.1:8010 --decode http://10.0.0.2:8020 \
  --dp-aware --policy dp_sticky \
  --atom-pd-rank-mapping-policy idx2idx \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8000 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --tokenizer $MODEL_PATH --tokenizer-trust-remote-code \
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

## Client

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

## Measured

16 chips, 3,600 s measurement per cell. `tok/s/chip` is
`(ΣISL + ΣOSL) / duration / 16` and counts input tokens, so it is dominated by
prefix-cache reads rather than compute. `x` is AIPerf's
`Output Token Throughput Per User` at p90.

Taken at `--gpu-memory-utilization` 0.75 prefill / 0.70 decode, which the
commands above no longer set — see [If the servers OOM at
startup](#if-the-servers-oom-at-startup).

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

Both `--prefill` and `--decode` accept repeats, so the router line extends to
xPyD. Two traps:

- `idx2idx` maps prefill DP ranks to decode DP ranks one to one. It survives an
  asymmetric count in practice (verified at 8 prefill ranks against 16 decode
  ranks), but that is not what the flag describes.
- **Multiple prefill nodes need prefix-affine routing.** Each prefill node owns a
  private `LocalCPUBackend`; a prefix saved on one is invisible to the other.
  With the default policy a follow-up request lands on the wrong node and the
  offload tier reads back nothing. Measured on a 2P1D attempt: 107 and 44 SLOT
  sidecars saved across the two nodes, **zero** restored. Set
  `--prefill-policy prefix_hash` (or `cache_aware`) before running more than one
  prefill node.

## Three flags deliberately not passed

Everything else in the commands is pinned on purpose, including values that
happen to match ATOM's current defaults. These three are left out because
passing them is either a no-op or actively misleading:

| flag | why it is gone |
|---|---|
| `--block-size 16` | **Ignored on V4.** `config.py` overrides `kv_cache_block_size` to 256 unconditionally: V4 needs a multiple of `lcm(4, 128)`, and 2×lcm gives the 64 CSA entries per block that the FP4 paged-MQA-logits indexer kernels require. Passing 16 changes nothing and suggests V4 blocks are 16 tokens. |
| `--index-cache-dtype fp8` | Forced anyway. `config.py` sets `fp8` whenever `kv_transfer_config` is set, which is every PD launch — Mooncake's producer-consumer staging has no layout for FP4's separate indexer scale pool. This is why PD cannot use the single-node recipe's `fp4`. |
| `--cudagraph-mode FULL` | Already the default, and unlike the pinned flags above it was never varied in any run here. |

## If the servers OOM at startup

It should not happen where RDMA memory registration works — the commands above
are then the whole configuration, and `--gpu-memory-utilization` is left at
ATOM's 0.9 default, the same value the single-node recipe uses. PD does not need
a lower one.

The Crusoe MI355X cluster these numbers came from is not such a place: ROCm's
GPU memory registration fails there, and the runs behind this file needed two
things that are deliberately not in the commands.

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

# DeepSeek-V4-Pro agentic on ATOM, PD-disaggregated — max throughput

The two-node companion to
[`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md). That
file is one node, no disaggregation, and is the right starting point. This one
splits prefill from decode across two nodes and adds a CPU KV offload tier,
which is what the workload needs at high concurrency.

- Hardware: MI355X ×8 per node, **two nodes**, prefill/decode disaggregated (1P1D)
- Model: `deepseek-ai/DeepSeek-V4-Pro`, FP4 weights, FP8 KV, FP8 index cache
- Transport: Mooncake RDMA, GID index 1, HIP DMA-BUF compatibility preload
- Scenario: `inferencex-agentx-mvp`, dataset `semianalysis_cc_traces_weka_062126`
- Router: `atomesh`, PD mode, `idx2idx` rank mapping

Two variants, split at concurrency 64:

| variant | concurrency | CPU offload |
|---|---|---|
| TP | 1 – 48 | not needed |
| DP attention | 64 and up | required from 256 |

The bands are contiguous. They differ by two flags, four environment variables,
and two sizing values — listed under *What changes between TP and DPA* below.

The PD sweep in *Measured* covers TP at 1–16 and DPA at 64–256; the 17–48 part
of the TP band follows the single-node recipe, where TP at c=48 is measured and
sits just under DP at c=64 (20,308 against 21,888 tok/s/chip). If you are
running between 16 and 48 here, that crossover is inherited rather than
re-measured on two nodes.

## Server — TP (concurrency 1 – 48)

```bash
export AITER_BF16_FP8_MOE_BOUND=0
export ATOM_MOE_GU_ITLV=1
export ATOM_HOST_IP=<PREFILL_IP>          # <DECODE_IP> on the decode node
export MC_GID_INDEX=1
export NCCL_IB_DISABLE=1
export ATOM_DISABLE_MMAP=true
export LD_PRELOAD=<path>/rdma_compat/libhip_dmabuf_mr.so

export ATOM_PREFIX_CACHE_POLICY=lru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port $PORT \
  --tensor-parallel-size 8 \
  --kv-cache-dtype fp8 --index-cache-dtype fp8 \
  --enable-prefix-caching --block-size 16 \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs $(( CONC * 2 )) \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.49 \
  --kv-transfer-config "$KV_TRANSFER"
```

`$PORT` is 8010 on prefill, 8020 on decode. `--gpu-memory-utilization` is 0.65
on prefill and 0.70 on decode. `$KV_TRANSFER` is the plain Mooncake pair:

```jsonc
// prefill
{"kv_role": "kv_producer", "kv_connector": "mooncake",
 "proxy_ip": "<PREFILL_IP>", "handshake_port": 6301, "protocol": "rdma"}
// decode
{"kv_role": "kv_consumer", "kv_connector": "mooncake",
 "proxy_ip": "<DECODE_IP>", "handshake_port": 6301, "protocol": "rdma"}
```

Router for this band drops the DP flags:

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --policy random \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## Server — DP attention (concurrency 64 and up)

Everything above, plus four environment variables, two flags, and different
sizing. On the **prefill** node add the CPU offload tier as well.

```bash
# ...the TP exports above, plus:
export ATOM_NUMA_BIND=1
export GPU_MAX_HW_QUEUES=5
export ATOM_DP_SESSION_AFFINITY=1
export ATOM_DP_LB_REQ_EQUIV=512
export ATOM_DP_MASTER_PORT=29510
export ATOM_DP_BASE_PORT=29610

# CPU offload — prefill node only. See "The offload settings that matter".
export OFFLOAD_COPY_WORKERS=1
export OFFLOAD_MIN_LOAD_TOKENS=8192
export OFFLOAD_SLOT_STAGING_SLOTS=4

python3 -m atom.entrypoints.openai_server \
  --model $MODEL_PATH --served-model-name deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 --server-port $PORT \
  --tensor-parallel-size 8 \
  --enable-dp-attention --enable-tbo \
  --kv-cache-dtype fp8 --index-cache-dtype fp8 \
  --enable-prefix-caching --block-size 16 \
  --gpu-memory-utilization 0.75 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 \
  --state-checkpoint-interval-tokens 8192 \
  --level 3 --cudagraph-mode FULL \
  --method mtp --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.49 \
  --kv-transfer-config "$KV_TRANSFER"
```

`--enable-tbo` (two-batch overlap) goes on the **prefill node only**. The decode
node runs DP attention without it, and at `--gpu-memory-utilization 0.70`.

Prefill `$KV_TRANSFER` for this band wraps Mooncake and the offload tier in a
`multi` connector:

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

Router for this band:

```bash
atomesh launch --host 0.0.0.0 --port 8000 --pd-disaggregation \
  --prefill http://<PREFILL_IP>:8010 --decode http://<DECODE_IP>:8020 \
  --dp-aware --policy dp_sticky \
  --atom-pd-rank-mapping-policy idx2idx \
  --backend atom --model-path $MODEL_PATH \
  --disable-circuit-breaker --prometheus-port 29100 \
  --request-timeout-secs 1800
```

## What changes between TP and DPA

Everything else in the two commands is identical. This is the whole delta:

| | TP | DPA |
|---|---|---|
| `--enable-dp-attention` | absent | **present** |
| `--enable-tbo` | absent | **present** (prefill only) |
| `--max-num-seqs` | `2 × CONC` | **256** |
| `--gpu-memory-utilization` (prefill) | 0.65 | **0.75** |
| `ATOM_NUMA_BIND` | unset | **1** |
| `GPU_MAX_HW_QUEUES` | unset | **5** |
| `ATOM_DP_SESSION_AFFINITY` | unset | **1** |
| `ATOM_DP_LB_REQ_EQUIV` | unset | **512** |
| router | `--policy random` | `--dp-aware --policy dp_sticky --atom-pd-rank-mapping-policy idx2idx` |

`ATOM_DP_SESSION_AFFINITY=1` is what makes the DP router keep a trajectory on
one rank, which is what the agentic scenario's prefix reuse depends on. Without
it the DP band loses most of its cache hit.

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

### What both of them actually bought

Beyond the failure counts, raising the two together unblocked the prefill queue:

| | defaults (1 / 2) | tuned (4 / 8) |
|---|---|---|
| SLOT sidecar failure rate | 26.0% | **1.5%** |
| save rejections | 6,758 | **160** |
| prefill `requests_waiting` | 146.5 | **27.3** |
| decode `requests_running` | 68.4 | **158.1** |
| warmup wall clock | ~3,400 s | **~2,280 s** |

The prefill queue is the point. At the defaults, 57% of a 256-request
concurrency sat waiting to start; the offload save path was contending with
prefill rather than serving it.

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

| conc | mode | offload | tok/s/chip | x (tok/s/user) | ITL p90 | TTFT p90 | cache hit |
|---|---|---|---|---|---|---|---|
| 1 | TP | — | 737 | 149.3 | 7.5 ms | 1.9 s | 96.8% |
| 2 | TP | — | 789 | 145.8 | 7.7 ms | 1.6 s | 95.6% |
| 8 | TP | — | 2,512 | 138.9 | 9.2 ms | 1.6 s | 97.1% |
| 16 | TP | — | 4,442 | 123.1 | 11.8 ms | 2.0 s | 96.6% |
| 64 | DPA | — | 15,137 | 69.1 | 19.4 ms | 7.8 s | 96.1% |
| **128** | DPA | — | **21,652** | 57.5 | 29.6 ms | 15.0 s | 94.7% |
| 256 | DPA | 1024 GiB | 21,599 | 56.2 | 31.0 ms | 192.8 s | 91.0% |

Every row above ran with the offload **defaults** (`max_pending_saves=2`,
`slot_sidecar_staging_slots=1`), which is why c=256 gains nothing over c=128 and
carries a 193 s TTFT. With the tuned values in this recipe the c=256 row changes
substantially; re-measure before quoting it.

Read three things carefully. **c=128 is the knee at default settings** — c=256
doubles the concurrency for no throughput and 13× the TTFT. **Cache hit falls
with concurrency** (96.8% → 91.0%), and every lost point is prefill compute the
offload tier failed to save. And a 1P2D variant measured at c=256 — 24 chips,
15,774 tok/s/chip, x=75.3, ITL p90 18.9 ms — shows that adding *decode* capacity
buys 34% interactivity and gives up 27% per-chip throughput, because decode was
never the constraint.

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

## Related

- [`DeepSeek-V4-Agentic-InferenceX.md`](DeepSeek-V4-Agentic-InferenceX.md) —
  single node, no disaggregation. Start there.
- [`DeepSeek-V4-Agentic-Benchmark.md`](DeepSeek-V4-Agentic-Benchmark.md) —
  cross-engine head-to-head.
- [`MiniMax-M3-Cache-Policies.md`](MiniMax-M3-Cache-Policies.md) — the offload
  tier's cache-policy knobs on a different model.

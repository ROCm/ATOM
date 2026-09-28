# MiniMax-M3 — Agentic serving recipe (InferenceX)

TP4 + EAGLE3, MXFP4 weights, concurrency 1–48. This is the configuration the SemiAnalysis InferenceX reference benchmarks for this model (`configs/amd-master.yaml`: `{tp: 4, spec-decoding: mtp}`); CONC 40/48 add the LMCache CPU tier.

MiniMax-M3's KV is **uncompressed** — 22.9 KB per token per rank — so the GPU pool stops holding reusable prefixes well before it runs out of room. Throughput flattens at CONC=32 while interactivity keeps falling; 40c is the best point on the ladder, and 48c buys 3.9% more throughput for 38.5% less interactivity. See [Measured](#measured).

Turn the CPU tier on at **40 and 48 only**. At 32 and below it does not pay: throughput stays inside noise and p90 interactivity drops 12%. See [Concurrency 40/48 — CPU offload](#concurrency-4048--cpu-offload).

One server launch per concurrency point — `--max-num-seqs` is sized off concurrency, so it cannot be reused across points. It is set at **2 × concurrency**: a value at or below the offered load silently becomes the bottleneck under test, and 2× leaves room for the trajectories that are mid-decode while others turn around.

```bash
FP4_TARGET=/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4   # or /mnt/m2m_nobackup/models/MiniMax-M3-MXFP4
DRAFT=/shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3-GQA
```

## Server

Concurrency ladder — run every point:

```bash
CONC=1    # then 2, 4, 5, 8, 10, 12, 15, 20, 24, 28, 32, 40, 48
```

```bash
env \
  NCCL_IB_DISABLE=1 \
  RCCL_IB_DISABLE=1 \
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  ATOM_FORCE_ATTN_TRITON=1 \
  AITER_LOG_LEVEL=WARNING \
  ATOM_GC_THRESHOLD=20000,50,50 \
  ATOM_PA_FLYDSL=1 \
  python3 -u -m atom.entrypoints.openai_server \
    --model "$FP4_TARGET" --served-model-name "$FP4_TARGET" \
    --host 0.0.0.0 --port 8896 --server-port 8890 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.95 \
    --block-size 128 \
    --max-num-batched-tokens 32768 \
    --attn-prefill-chunk-size 16384 \
    --max-num-seqs $((2 * CONC)) \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --default-chat-template-kwargs '{"thinking_mode": "enabled"}' \
    --enable-prefix-caching \
    --method eagle3 \
    --draft-model "$DRAFT" \
    --num-speculative-tokens 3 \
    --spec-decode-acceptance-rate 0.5933 \
  > server_c${CONC}.log 2>&1 &
```

## FlyDSL paged decode

`ATOM_PA_FLYDSL=1` is opt-in and off by default. It routes the paged decode to aiter's FlyDSL kernel instead of gluon, and brings aiter #5546's GPU work planner with it (`ATOM_PA_FLYDSL_PLAN`, on by default, and inert without `ATOM_PA_FLYDSL=1`).

gluon splits every request in a batch the same way. An agentic decode batch is not uniform — measured over 30,400 steps at CONC=32, the step-internal `max/min` context ratio is p50 2.49, p90 22.2, and 31.2% of steps exceed 4× — so one long request owns the critical path. The planner sizes each request's partition count from its real context length instead, under a workgroup budget.

Shapes outside FlyDSL's domain fall back to gluon on their own, so turning this on cannot make a working configuration raise. Measured against the gluon path on the same node, p90 interactivity: −0.0% at CONC=1, +4.0% at 10, +11.8% at 15, +20.5% at 20. The gain needs a batch with something to rebalance, which is why CONC=1 is flat.

## Indexer-only decode context parallelism

`ATOM_M3_INDEXER_CP=1` is opt-in and off by default. M3's lightning indexer is MQA: four index query heads share **one** index-K head (`index_kv_cache` is `[num_blocks, 128, 128]`, no head axis). TP splits the four query heads one per rank, but the single K head cannot be split — it is replicated, and every rank reads all of it. CP instead gives each rank all four heads over `1/P` of the blocks, so the replicated cache is read once per step instead of `P` times, then exchanges **candidates, not scores** (`world × topk` keys per token, independent of context length). Selection is bit-identical to the TP path — verified 9/9 points on four ranks with the real collective, contexts to 315K, comparing `topk_idx / sparse_bt / sparse_ctx`.

Enable it from **CONC ≥ 15** by adding one line to the server's `env \` block:

```bash
  ATOM_M3_INDEXER_CP=1 \
```

Requires `--tensor-parallel-size 4` (ATOM asserts `tp_size == sparse_num_index_heads`, 4 for M3) and `decode_context_parallel_size == 1`. Measured, same-commit A/B, 1800 s per arm:

| CONC | interactivity p90 | ISL-normalised throughput | prefix hit, TP vs CP |
| ---: | ---: | ---: | --- |
| 15 | +3.7% | +0.67% | 96.7% / 96.7% |
| 24 | +12.3% | +0.24% | 97.5% / 97.5% |
| 28 | **+16.7%** | +2.7% | 97.2% / 97.3% |

Monotone and never negative: throughput is neutral everywhere while interactivity rises with the running batch, which is what CP actually shortens. Identical prefix-hit rates within each pair are the control — CP changes the indexer's parallel axis and nothing in the KV pool. Below CONC 15 the exchange+merge tail — a fixed 10–14 µs per sparse layer, independent of batch and context — is not amortised, and at batch ≈ 2 the kernel chain measures 1.55× slower than the TP path.

> Arm verification does **not** use the `Engine kwargs` line — it prints the pre-override value and shows `indexer_dcp_only=False` on both arms. Use the `dcp_config.indexer_dcp_only enabled` startup line, the absence of a `disabled` line, and a `peak_torch` delta of ~0.13 GB (the widened `index_q` projection).

## Concurrency 40/48 — CPU offload

Run the 40c and 48c points with LMCache CPU offload and the SLRU cache policy. Keep the server command above and add these to its `env \` block:

```bash
  ROCR_VISIBLE_DEVICES=0,1,4,5 \
  PYTHONHASHSEED=0 \
  LMCACHE_LOCAL_CPU=True \
  LMCACHE_MAX_LOCAL_CPU_SIZE=256 \
  LMCACHE_CHUNK_SIZE=256 \
  ATOM_PREFIX_CACHE_POLICY=slru \
  ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5 \
  LMCACHE_CACHE_POLICY=ATOM_SLRU \
  LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3 \
```

### `ROCR_VISIBLE_DEVICES=0,1,4,5` is a prerequisite, not a tuning knob

GPUs 0–3 sit on NUMA node 0 and 4–7 on node 1, and ranks pinning host memory on one node starve each other. On the 8×MI355X nodes measured here this is not a slowdown but a failure: with all four TP4 ranks on node 0 **the server did not come up** — the fourth rank never finished pinning its 256 GB and the ranks that did timed out on the `allocate_kv_cache` barrier after 600 s. Split two per socket, the same 1 TB pins in **24 s**. (TP2 on one node is merely slow rather than fatal: 45 min against 21 s split.)

Splitting puts the TP all-reduce across sockets, which costs nothing measurable: a no-offload CONC=32 arm on `0,1,4,5` against the same arm on `0,1,2,3` is +0.43% throughput, −1.08% p90 interactivity, +0.50% p50 — all at the 0.6% noise floor.

`PYTHONHASHSEED=0` is not optional either: without it each TP rank hashes the same prompt to a different cache key and the offload hit rate goes to zero.

and this to its args:

```bash
    --kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
```

### Cache policies

`LMCACHE_LOCAL_CPU` / `LMCACHE_MAX_LOCAL_CPU_SIZE` / `LMCACHE_CHUNK_SIZE` turn the CPU tier on and size it. The remaining four variables are what make it worth having — two policies plus a fix that has no knob. Leave them out and HBM and CPU both fall back to plain LRU, and lookup defaults to rank 0.

- **`LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3` — all-rank lookup.** One id per TP4 rank. Each rank updates its own cache recency and pins its own KV shard, and the **minimum** hit length across ranks is the common restorable prefix — so a shard another rank has already evicted is never trusted.
- **`LMCACHE_CACHE_POLICY=ATOM_SLRU` + `ATOM_PREFIX_CACHE_POLICY=slru` — segmented LRU.** New data enters probation, reused data is promoted to protection; probationary data is evicted first, demoting older protected entries once the limit is exceeded. `ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5` caps HBM protection at that fraction of total pool blocks; CPU protection targets half the resident chunks. Protected data stays evictable — referenced/pinned data does not. Separate CPU queues keep probationary eviction from scanning the protected segment, though pinned entries may still require a scan.
- **Pin lifecycle** (no knob, always on): lookup pins are released on misses, errors, skipped loads and cancellation, including aborts before HBM allocation, while pins for pending loads are retained. This is independent of SLRU.

Only **synchronous** lookup is supported; enabling `LMCACHE_ENABLE_ASYNC_LOADING` is rejected at startup.

Lookup scope is the biggest of the three by a wide margin. Measured at 48c, 1800 s, 256 GiB CPU cache per rank, no NVMe, FP8 KV, synthetic acceptance 0.5933 — **both arms had SLRU and the pin fix on, only the lookup scope changed**:

| | rank 0 only | all-rank |
|---|---:|---:|
| Total prompt cache hit rate | 82.86% | **95.48%** |
| Throughput (tok/s/chip) | 21,692.64 | **42,843.46** |
| P90 ITL (ms) | 66.94 | **24.35** |

> This pair isolates lookup scope, **not SLRU's independent contribution** — nobody has run SLRU on/off with everything else held. It also predates the `--gpu-memory-utilization 0.95`, cudagraph-size and indexer-CP changes, so its absolute numbers are **not** comparable with the [Measured](#measured) table above; the 48c row there reads 55,516 tok/s/chip at a P90 ITL of ~22 ms on the newer basis.

## aiperf

```bash
ART=./results_FP4/c${CONC}_$(date +%m%d_%H%M)
mkdir -p "$ART"

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10
export AIPERF_WARMUP_REQUESTS_PER_LANE=10
export AIPERF_BENCHMARK_GRACE_PERIOD=30
export AIPERF_SERVER_METRICS_URLS="http://localhost:8890/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

aiperf profile --scenario inferencex-agentx-mvp \
  --url http://localhost:8890 --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming \
  --model "$FP4_TARGET" --tokenizer "$FP4_TARGET" \
  --tokenizer-trust-remote-code \
  --apply-chat-template \
  --concurrency $CONC --benchmark-duration 3600 --stats-interval 30 \
  --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
  --agentic-warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --server-metrics http://localhost:8890/metrics \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --output-artifact-dir "$ART"
```

## Measured

MI355X, TP4 + EAGLE3, MXFP4, `--gpu-memory-utilization 0.95`, explicit cudagraph sizes. **3600 s per point**, one node, one container image. Indexer CP is on from CONC 15 and the CPU tier from CONC 40, exactly as prescribed above.

| conc | CP | offload | tok/s/chip | intvty_p90 | engine hit | recompute | realised hit |
| ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 1 | — | — | 5,137 | 297.5 | 97.7% | 1.5 M | 98.0% |
| 2 | — | — | 5,485 | 274.9 | 96.9% | 2.2 M | 97.2% |
| 8 | — | — | 14,502 | 213.4 | 97.6% | 5.2 M | 97.5% |
| 10 | — | — | 16,458 | 206.6 | 97.9% | 6.3 M | 97.3% |
| 15 | ✅ | — | 22,234 | 172.8 | 97.1% | 10.6 M | 96.7% |
| 20 | ✅ | — | 34,236 | 133.0 | 98.5% | 12.8 M | 97.4% |
| 24 | ✅ | — | 40,925 | 115.8 | 98.1% | 15.6 M | 97.4% |
| 28 | ✅ | — | 44,609 | 100.3 | 95.9% | 23.0 M | 96.4% |
| 32 | ✅ | — | 45,072 | 74.4 | 92.2% | 36.6 M | 94.4% |
| 40 | ✅ | ✅ | 53,437 | 73.0 | 83.1% | 22.5 M | 97.1% |
| 48 | ✅ | ✅ | 55,516 | 44.9 | 70.2% | 41.0 M | 94.9% |

> The 1c and 2c rows complete only 270 and 431 requests in the hour, so their input-length mix is drawn from a much smaller sample than the rest of the ladder and skews long (ISL 272K and 182K against ~150K above 8c). Read them as operating points, not as points on the same corpus.

Peak throughput is at 48c, but 40c is the point worth quoting: **+18.6% throughput over 32c at essentially the same interactivity** (73.0 against 74.4). Past it, 48c buys 3.9% more throughput for 38.5% less interactivity.

> **Read the last column, not `engine hit`.** The engine's `Prefix cache hit rate` counts the GPU tier only, so a chunk served from LMCache scores as a miss — which is why it appears to collapse to 70.2% at 48c. `realised hit` is derived independently, from the prompt tokens the server actually ran prefill on (`Avg prompt throughput` integrated over the profiling window). Without offload the two agree within 0.5–2.2 pp; with offload they diverge by 14–25 pp, and that divergence *is* the CPU tier's contribution.

The 1c–32c ladder this file carried until 2026-09-17 has been removed rather than re-labelled. Those numbers were taken with `--hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}'`, which reuses the previous layer's top-k on 42 of the 57 sparse layers; the flag was dropped from the command on 2026-09-10 (`e885fc6f4`) but the table was never re-measured, so it reported a configuration this recipe no longer runs and read roughly 14% fast. Re-run the low-concurrency points on the current command if you need them.

## Notes

- `--model` / `--tokenizer` on the client must equal the server's `--served-model-name` exactly, or every request 404s.
- `--benchmark-duration` must be ≥ 900 (scenario floor). Shorter needs `--unsafe-override`, which stamps `submission_valid=false`.
- `--server-port` is the HTTP listener (8890); `--port` is the internal engine port (8896).
- Leave `--max-model-len` unset. Warmup primers replay contexts up to ~996k tokens; any cap below ~1M makes the server 400 them and the run aborts.
- The model auto-probe in the launcher scripts resolves MXFP8 first — pass `MODEL=` explicitly to both server and client, or the run is off-target.
- `ATOM_FORCE_ATTN_TRITON=1` is not optional. Without it decode routes to the ASM paged-attention kernel, whose `qlen × gqa ≤ 16` constraint M3 violates (gqa=16, qlen=4 → 64), and the server dies at startup.
- `NCCL_IB_DISABLE=1` / `RCCL_IB_DISABLE=1` are for single-node TP4. The container bind-mounts the host ionic RDMA provider over its own rdma-core and RCCL GP-faults inside `libibverbs`→`libionic` when it probes IB devices at worker init — every ModelRunner dies with `exitcode=-11` right after the Gloo handshake, before any weight load. TP4 is intra-node XGMI only, so nothing is lost.
- `--enable-prefix-caching` is on, a deliberate deviation from CI's `--no-enable_prefix_caching`. CI's random ISL/OSL workloads do not reuse prefixes; agentic replay is built on prefix reuse.
- `--gpu-memory-utilization 0.95` was verified under load, not assumed: five TP4 points each ran a full 3600 s replay with zero OOM, at 64,933–65,067 blocks against 59,951 at 0.90. The 40c and 48c points carry offload + EAGLE3 + indexer CP on top.
- `--cudagraph-capture-sizes '[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,24,26,28,30,32,34,36,40,48,56,64]'` is optional and worth adding: **+9.2% throughput, +50.7% interactivity** at TP4 CONC=24, with `cudagraph_est` unchanged at 0.99 GB. Size the list from the **running batch, not CONC** — only 22–41% of the offered concurrency is in decode at any instant here, so replays land in roughly 1..34 and the stock `[1,2,4,8,16,32,48,64,128,256,512]` pads a batch of 9 up to 16 and 17 up to 32, wasting 44–47% of the rows. `ModelRunner` drops the declared sizes a band cannot schedule, so one list serves every point.
- `--spec-decode-acceptance-rate` is performance-only: the draft and the verify still run, but which draft tokens commit is fixed to the target rate instead of the draft head's real agreement. It invalidates accuracy, and the acceptance figure in the server log becomes a restatement of the constant rather than a measurement.

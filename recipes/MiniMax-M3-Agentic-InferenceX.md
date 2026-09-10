# MiniMax-M3 — Agentic serving recipe (InferenceX)

TP4 + EAGLE3, no KV offload, MXFP4 weights, concurrency 1–32. This is the configuration the
SemiAnalysis InferenceX reference benchmarks for this model
(`configs/amd-master.yaml`: `{tp: 4, kv-offloading: none, spec-decoding: mtp}`).

MiniMax-M3's KV is **uncompressed** — 2,995,200 bytes per 128-token block = **22.9 KB per token
per rank**, against DeepSeek-V4's MLA which compresses ~10–20×. The GPU KV pool is 59,853 blocks
× 128 = **7.66M tokens**, and the agentic traces average ~150K tokens of context (~1,170 blocks
per in-flight request), so the running set fills the pool near 50 concurrent requests. At ≤24
there is still headroom, reusable multi-turn prefixes survive in HBM, and a CPU/NVMe offload tier
would only spend lookups recovering tokens HBM never lost — which is why offload is off on the
1–32 ladder. Past that knee — 48c and up — the running set exceeds the pool, so CPU offload with
the SLRU cache policy earns its keep; see [Concurrency 48 — CPU offload](#concurrency-48--cpu-offload).

One server launch per concurrency point — `--max-num-seqs` is sized off concurrency, so it
cannot be reused across points. It is set at **2 × concurrency**: a value at or below the offered
load silently becomes the bottleneck under test, and 2× leaves room for the trajectories that are
mid-decode while others turn around.

```bash
FP4_TARGET=/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4   # or /mnt/m2m_nobackup/models/MiniMax-M3-MXFP4
DRAFT=/shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3-GQA
```

## Server

Concurrency ladder — run every point:

```bash
CONC=1    # then 2, 4, 5, 8, 10, 12, 15, 20, 24, 28, 32
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
  python3 -u -m atom.entrypoints.openai_server \
    --model "$FP4_TARGET" --served-model-name "$FP4_TARGET" \
    --host 0.0.0.0 --port 8896 --server-port 8890 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
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

## Concurrency 48 — CPU offload

Beyond the ~32c knee the running set no longer fits the HBM pool, so the offload
tier stops being redundant. Run a 48c point by enabling LMCache CPU offload with
the SLRU cache policy — full rationale in `MiniMax-M3-Cache-Policies.md`. Keep the
server command above (`CONC=48`, so `--max-num-seqs` is 96) and add these to its
`env \` block:

```bash
  LMCACHE_LOCAL_CPU=True \
  LMCACHE_MAX_LOCAL_CPU_SIZE=256 \
  LMCACHE_CHUNK_SIZE=256 \
  ATOM_PREFIX_CACHE_POLICY=slru \
  ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5 \
  LMCACHE_CACHE_POLICY=ATOM_SLRU \
  LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3 \
```

and this to its args:

```bash
    --kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
```

`LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3` is the all-rank lookup scope — one id per
TP4 rank. Each rank pins its own KV shard and the minimum hit length across ranks
is the common restorable prefix, so a shard another rank already evicted is never
trusted. At 48c this took the total prompt cache hit rate from 82.86% (rank-0-only
lookup) to 95.48%. Only synchronous lookup is supported; enabling
`LMCACHE_ENABLE_ASYNC_LOADING` is rejected at startup.

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

MI355X, TP4 + EAGLE3, MXFP4, no offload.

| conc | tok/s/chip | intvty_p90 | prefix hit |
| ---: | ---: | ---: | ---: |
| 1 | 4,845 | 256.5 | 97.6% |
| 2 | 5,050 | 257.6 | 96.8% |
| 8 | 14,047 | 198.8 | 97.5% |
| 10 | 16,756 | 188.6 | 97.8% |
| 15 | 22,025 | 163.1 | 97.1% |
| 20 | 32,870 | 121.2 | 98.4% |
| 24 | 39,680 | 106.9 | 97.5% |
| 32 | 42,476 | **58.0** | **91.8%** |

Throughput still climbs to 32c, but that is where interactivity halves (106.9 → 58.0) and the
prefix hit finally breaks below 97% (91.8%) — the knee of this configuration.

## Notes

- `--model` / `--tokenizer` on the client must equal the server's `--served-model-name` exactly,
  or every request 404s.
- `--benchmark-duration` must be ≥ 900 (scenario floor). Shorter needs `--unsafe-override`, which
  stamps `submission_valid=false`.
- `--server-port` is the HTTP listener (8890); `--port` is the internal engine port (8896).
- Leave `--max-model-len` unset. Warmup primers replay contexts up to ~996k tokens; any cap below
  ~1M makes the server 400 them and the run aborts.
- The model auto-probe in the launcher scripts resolves MXFP8 first — pass `MODEL=` explicitly to
  both server and client, or the run is off-target.
- `ATOM_FORCE_ATTN_TRITON=1` is not optional. Without it decode routes to the ASM paged-attention
  kernel, whose `qlen × gqa ≤ 16` constraint M3 violates (gqa=16, qlen=4 → 64), and the server
  dies at startup.
- `NCCL_IB_DISABLE=1` / `RCCL_IB_DISABLE=1` are for single-node TP4. The container bind-mounts the
  host ionic RDMA provider over its own rdma-core and RCCL GP-faults inside `libibverbs`→`libionic`
  when it probes IB devices at worker init — every ModelRunner dies with `exitcode=-11` right after
  the Gloo handshake, before any weight load. TP4 is intra-node XGMI only, so nothing is lost.
- `--enable-prefix-caching` is on, a deliberate deviation from CI's `--no-enable_prefix_caching`.
  CI's random ISL/OSL workloads do not reuse prefixes; agentic replay is built on prefix reuse.
- `--spec-decode-acceptance-rate` is performance-only: the draft and the verify still run, but
  which draft tokens commit is fixed to the target rate instead of the draft head's real
  agreement. It invalidates accuracy, and the acceptance figure in the server log becomes a
  restatement of the constant rather than a measurement.

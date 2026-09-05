# MiniMax-M3 — Agentic serving recipe (KV offload + DP2 by concurrency)

Serving-layer only — model code (`atom/model_ops/minimax_m3/*`) is untouched; every knob below
is a server flag or an environment variable.

MiniMax-M3's KV is **uncompressed** (2,961,408 bytes / 128-token block = **22.6 KB/token/rank**,
vs DeepSeek-V4's MLA which compresses ~10–20×). HBM holds only ~4.66M tokens/rank, so the
prefix cache that agentic multi-turn replay lives on evicts fast as concurrency climbs. That
single fact drives the whole ladder: **how much of the reusable prefix survives in HBM, and
where the rest has to live.**

---

## TL;DR — pick by offered concurrency

| Concurrency | Config | Offload | GPUs | Ranks | Why |
|:-----------:|:-------|:-------:|:----:|:-----:|:----|
| **1 – 16c** | TP4 + EAGLE3 | off | 4 | 4 | Working set fits HBM; the HBM prefix cache alone hits ~100%. Offload only adds lookup + CPU-staging overhead for zero recovered tokens. |
| **32 – 48c** | TP4 + EAGLE3 | on | 4 | 4 | HBM prefix cache starts evicting reusable prefixes; the CPU (L2) + NVMe (L3) offload tier restores what HBM dropped. Total cache-read stays ~96%. 256 GB/rank pool (1 TB total) holds through 48c. |
| **≥ 64c** | DP2 × TP4 | on | 8 | 8 | A single TP4's offload pool saturates at 64c (per-rank working set × concurrency > pool → alloc-fail thrash, run stalls). DP2 = two independent TP4 replicas: each sees ~conc/2, **total CPU pool doubles (8 ranks), restore bandwidth doubles** → back in the unsaturated regime. |

Rule of thumb for the in-between points (24c, etc.): offload earns its keep the moment HBM
prefix hit starts dropping below ~100%; add DP2 the moment a single-TP4 offload pool starts
logging LMCache alloc-fails. Don't run DP2 below ~64c — splitting the batch halves each
replica's spec-decode/batch efficiency for no cache benefit while HBM is still coping.

Notes on the commands below:
- `--port` is the internal engine port; `--server-port` is the HTTP listener (8890). Passing
  only `--port` leaves nothing listening on 8890.
- `--max-num-seqs` is sized at **2 × offered concurrency** (32 for 16c, 96 for 48c, 192 for
  96c). A cap below the offered load silently becomes the bottleneck.
- `--model` points at the local checkpoint; adjust the path to your mount.

---

## A. Low concurrency (1–16c) — TP4 + EAGLE3, no offload

Server:
```bash
env \
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  ATOM_FORCE_ATTN_TRITON=1 \
  AITER_LOG_LEVEL=WARNING \
  ATOM_GC_THRESHOLD=20000,50,50 \
  python3 -u -m atom.entrypoints.openai_server \
    --model /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
    --served-model-name /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
    --host 0.0.0.0 --port 8896 --server-port 8890 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --attn-prefill-chunk-size 16384 \
    --state-checkpoint-interval-tokens 32768 \
    --max-num-seqs 32 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --enable-prefix-caching \
    --method eagle3 \
    --draft-model /shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3 \
    --num-speculative-tokens 3 \
    --spec-decode-acceptance-rate 0.7336
```
No `--kv-transfer-config` and no LMCache env: single TP4, EAGLE3 spec-decode, HBM prefix cache
only. Then run the client (§ Client) with `--concurrency 16`.

---

## B. Mid concurrency (32–48c) — TP4 + EAGLE3 + offload

Same as A plus the LMCache env and `--kv-transfer-config`:
```bash
mkdir -p /mnt/m2m_nobackup/lmcache_disk   # NVMe (L3) tier dir

env \
  PYTHONHASHSEED=0 \
  LMCACHE_LOCAL_CPU=True \
  LMCACHE_MAX_LOCAL_CPU_SIZE=256 \
  LMCACHE_CHUNK_SIZE=256 \
  LMCACHE_LOCAL_DISK=file:///mnt/m2m_nobackup/lmcache_disk \
  LMCACHE_MAX_LOCAL_DISK_SIZE=500 \
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  ATOM_FORCE_ATTN_TRITON=1 \
  AITER_LOG_LEVEL=WARNING \
  ATOM_GC_THRESHOLD=20000,50,50 \
  python3 -u -m atom.entrypoints.openai_server \
    --model /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
    --served-model-name /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
    --host 0.0.0.0 --port 8896 --server-port 8890 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --attn-prefill-chunk-size 16384 \
    --state-checkpoint-interval-tokens 32768 \
    --max-num-seqs 96 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
    --method eagle3 \
    --draft-model /shared_nfs/huggingface_models/Inferact/MiniMax-M3-EAGLE3 \
    --num-speculative-tokens 3 \
    --spec-decode-acceptance-rate 0.7336
```
`LMCACHE_MAX_LOCAL_CPU_SIZE` (GB) and `LMCACHE_MAX_LOCAL_DISK_SIZE` (GB) are the CPU (L2) and
NVMe (L3) tier sizes, **per rank**. 256 GB/rank × 4 ranks = **1 TB** CPU pool, which carries
48c. `PYTHONHASHSEED=0` is mandatory — without it each TP rank keys the same prompt differently
and the offload hit ratio collapses to 0. Then run the client with `--concurrency 48`.

---

## C. High concurrency (≥64c) — DP2 × TP4 + offload

`--data-parallel-size 2` (two independent TP4 replicas across all 8 GPUs), no EAGLE3, session
affinity on:
```bash
mkdir -p /mnt/m2m_nobackup/lmcache_disk   # NVMe (L3) tier dir

env \
  ATOM_DP_SESSION_AFFINITY=1 \
  PYTHONHASHSEED=0 \
  LMCACHE_LOCAL_CPU=True \
  LMCACHE_MAX_LOCAL_CPU_SIZE=256 \
  LMCACHE_CHUNK_SIZE=256 \
  LMCACHE_LOCAL_DISK=file:///mnt/m2m_nobackup/lmcache_disk \
  LMCACHE_MAX_LOCAL_DISK_SIZE=500 \
  AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
  AITER_QUICK_REDUCE_CAST_BF16_TO_FP16=0 \
  ATOM_FORCE_ATTN_TRITON=1 \
  AITER_LOG_LEVEL=WARNING \
  ATOM_GC_THRESHOLD=20000,50,50 \
  python3 -u -m atom.entrypoints.openai_server \
    --model /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
    --served-model-name /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
    --host 0.0.0.0 --port 8896 --server-port 8890 \
    --tensor-parallel-size 4 --data-parallel-size 2 --dp-load-balance least_requests \
    --trust-remote-code \
    --kv_cache_dtype fp8 \
    --gpu-memory-utilization 0.9 \
    --block-size 128 \
    --max-num-batched-tokens 16384 \
    --attn-prefill-chunk-size 16384 \
    --state-checkpoint-interval-tokens 32768 \
    --max-num-seqs 192 \
    --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}' \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4}' \
    --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}' \
    --index-cache-dtype fp8
```
`LMCACHE_MAX_LOCAL_CPU_SIZE` is **per rank** and DP2 has **8 ranks** (2 dp × 4 tp), so the host
CPU total is size × 8. 256 → **2 TB** (fits a 2.5 TB node); drop to 192 (→ 1.5 TB) on a tighter
box. `ATOM_DP_SESSION_AFFINITY=1` is mandatory — it pins each multi-turn session to the replica
whose local prefix cache owns it, so a later turn is a warm hit rather than a cold prefill on a
rank that never saw the prefix. The DP2 configuration runs without spec-decode (no draft model).
Then run the client with `--concurrency 96`.

---

## Client (aiperf agentic replay)

Identical across all three bands — **only `--concurrency` changes** to match the server's
`--max-num-seqs / 2`. Start it after the server logs `READY http=200`:
```bash
ART=./results_FP8/c96_$(date +%m%d_%H%M)   # set concurrency in the name to taste
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
  --model /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
  --tokenizer /mnt/m2m_nobackup/models/MiniMax-M3-MXFP8 \
  --tokenizer-trust-remote-code \
  --concurrency 96 --benchmark-duration 3600 --stats-interval 30 \
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
`--tokenizer` / `--model` must match the server's `--served-model-name` exactly or every request
404s. `--benchmark-duration` must be ≥ 900 (the scenario floor); anything shorter needs
`--unsafe-override`, which stamps `submission_valid=false` (smoke test only). The DP2 server in
§C is plain data parallelism (not DP-attention), so the client needs no session-id headers —
affinity is keyed server-side.

---

## Why these thresholds (the mechanism)

- **≤16c — HBM suffices.** Concurrency × per-session KV stays well under 4.66M tokens/rank,
  so evicted-then-reused prefixes are rare. The offload connector would spend lookups and
  CPU→HBM restores to recover tokens HBM never lost. Net negative. Keep it off.

- **32–48c — HBM leaks, offload catches.** The working set crosses HBM's ceiling and the
  prefix cache begins evicting reusable multi-turn prefixes. The **two tiers are disjoint**
  (HBM prefix cache checked first; the PAGE offload connector restores only what HBM missed),
  so as HBM hit falls the offload tier picks up the exact shift and the *total* holds. Measured:

  | Concurrency | Offload hit (LMCache) | Prefix hit (HBM) | Total cache-read |
  |:-----------:|:---------------------:|:----------------:|:----------------:|
  | 32c | 44.1% | 52.6% | **96.7%** |
  | 48c | 82.5% | 13.1% | **95.5%** |

  (% of prompt tokens, server Prometheus counters. Cross-checked vs aiperf Overall Prompt
  Cache Read %: 32c=94.87%, 48c=95.76% — within ~0.3%.) HBM collapses 52.6%→13.1% while
  offload rises 44.1%→82.5%; true prefill stays ~3–5%. **Without offload that HBM collapse
  falls straight through to recompute.**

- **≥64c — single pool saturates, split it.** At 64c the per-rank working set × concurrency
  exceeds a 4-rank / 1 TB CPU pool; LMCache logs mass allocation failures (≈778k observed at
  c64 single-TP4) and the run stalls. DP2 fixes it three ways at once: **(1)** each replica
  serves ~conc/2, halving per-rank working set; **(2)** 8 ranks instead of 4 double the total
  CPU pool; **(3)** 8 ranks double the CPU→HBM restore parallelism. The same 256 GB/rank that
  thrashes at c64-on-4-ranks runs clean at c64-on-8-ranks.

---

## Mandatory env (do not drop)

- `ATOM_FORCE_ATTN_TRITON=1` — always. Without it decode routes to the ASM paged-attention
  kernel whose `qlen*gqa ≤ 16` constraint M3 violates (gqa=16, qlen=4 → 64); server dies at
  startup.
- `PYTHONHASHSEED=0` — whenever offload is on. Without a fixed hash seed each TP rank derives a
  different cache key for the same prompt and the offload hit ratio collapses to 0.
- `ATOM_DP_SESSION_AFFINITY=1` — for the DP2 configuration. Without it a session's turns scatter
  across replicas and every turn is a cold prefill.
- Offload requires the **HIP source build of LMCache** (`/opt/LMCache`, `backend: lmcache.c_ops`).
  The PyPI wheel is CUDA-linked and silently falls back to a slow python path that negates the win.

## Notes / non-knobs on M3

- `--state-checkpoint-interval-tokens` (V4's 32768) is **inert on M3**: the runtime reports
  zero state slots ("state 0/0", "Checkpoint Fates kept: 0"). Set for flag-parity only; it
  changes nothing.
- `--max-model-len` — leave uncapped (model-native 1,048,576). AgentX warmup primers replay each
  trace's full accumulated context (up to ~996k tokens); any cap below ~1M makes the server 400
  those primers and the run aborts before profiling. A cap also can't grow the KV pool, so it
  can't help hit rate anyway.
- `--enable-prefix-caching` is on here, a deliberate deviation from CI's `--no-enable_prefix_caching`
  — CI's random ISL/OSL workloads don't reuse prefixes; agentic replay is built on prefix reuse.

# ATOM Environment Variables

This document describes the environment variables used in the ATOM project.

## OpenTelemetry tracing

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_OTEL_TRACE** | bool | false | Enable standalone request tracing with `1` or `true` (case insensitive). Requires the optional `tracing` dependencies and an explicit OTLP/HTTP traces endpoint. Read at process startup; restart to change. |
| **ATOM_TRACE_VERSION** | str | `unknown` | Exact build version attached to each tracing resource as `atom.version`. Set before startup. |

The standard `OTEL_SERVICE_NAME` and `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`
variables configure service identity and delivery. See [tracing setup and
limitations](observability/README.md).

## Metadata H2D

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_H2D_BACKEND** | str | `packed` | `packed` combines forward metadata into one H2D and GPU scatter per consumer group. `direct` copies each member separately. Both preserve source reuse gates and full cudagraph padding. Set before starting the runner. See [metadata publication](h2d_publication.md). |

## Data parallelism

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DP_RANK** | int | 0 | The rank ID for the current process in data parallelism. |
| **ATOM_DP_RANK_LOCAL** | int | 0 | The local rank ID for the current process (used in SPMD mode). |
| **ATOM_DP_SIZE** | int | 1 | Total number of data parallel ranks. |
| **ATOM_DP_MASTER_IP** | str | 127.0.0.1 | Master IP address for DP ranks coordination. |
| **ATOM_DP_MASTER_PORT** | int | 29500 | Master port for DP ranks coordination. |
| **ATOM_DP_LB_REQ_EQUIV** | int | 512 | Token-equivalent decode pressure assigned to each in-flight request by `least_tokens` routing. |
| **ATOM_DP_SESSION_AFFINITY** | bool | false | Load-place each new session, then keep later turns on the same prefix-cache owner. Reads `X-Dynamo-Session-ID`, then `X-Session-ID` (MLPerf Endpoints client), then `X-Correlation-ID`. |
| **ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT** | int | 360000 | DP prefill cost balancing (always on under DP attention with more than one rank; needs the prefill delayer): each lockstep prefill step aims at min(costliest, mean x (1 + slack)) of the ranks' estimated chunk costs, and a costlier rank takes fewer tokens. This is the per-token term of the chunk cost, in attended-context tokens: cost = new tokens x (weight + prior context + new / 2). |
| **ATOM_DP_PREFILL_BALANCE_SLACK** | float | 0.1 | The target cost is at most mean x (1 + slack) over the ranks that prefill. |
| **ATOM_DP_PREFILL_BALANCE_MIN_TOKENS** | int | 4096 | Fewest tokens a shrunk rank still prefills in a step. |

## Prefill delayer (TP/DCP and DP attention)

Coalesces waiting prefills while decode continues. DP attention enables it by
default through `ATOM_ENABLE_PREFILL_DELAYER`. For a single scheduler (DP=1,
PP=1), including TP/DCP, it is opt-in: set `ATOM_PREFILL_DECODE_INTERVAL` above
zero and keep the master switch enabled. Interval 0 leaves TP scheduling
unchanged; setting only the master switch does not enable TP coalescing.
On TP, the interval and coalescer are enabled together.
This applies to the standard scheduler, including connector-based P/D roles.
RapidServe's dedicated `PrefillScheduler`/`DecodeScheduler` do not use the delayer.

After each executed prefill, the decode interval runs before all coalescing
bounds. Once it expires, fill, queue age, KV pressure, partial-prefill and stall
bounds decide when to release. `MAX_QUEUE_MS` stops extra coalescing after that
interval; it does not guarantee end-to-end TTFT. DP decisions reduce local
signals across ranks to keep their phases aligned.

The local fill signal discounts HBM cache hits and uses the admission path's
chunk limits. To bound CPU work, it stops probing fresh requests after their
total prompt length reaches one batch budget. It reports only work found so
far; it does not treat unseen requests as a full batch. A deep, cache-heavy
queue can therefore release through the stall or hold bounds before reaching
the fill target. If parked transfers exhaust the unreserved slots, a fresh
request may signal possible work with zero estimated tokens until admission
resolves its connector match. This is an estimate, not a reservation: checkpoint dependencies,
connector results and resource changes during admission can still reduce a batch.
Repeated probes reuse immutable prompt hashes while rechecking pool contents
and resource fit. During decode protection, only the existence of queued or
partial work is checked; partial-prefill hold bounds start after the interval.

Local hybrid models with state checkpointing can wait for an in-flight
producer's reusable prompt-end checkpoint. The expected prefix must exceed both
the HBM hit and any offload match by at least one prefill budget. P/D transfers
and already-started offload loads keep their own progress paths. Deferred
requests retain their relative order, while at most 16 later queue entries are
examined for independent work each pass. Each request's wait expires after
`TTFT_MAX_TICKS` scheduler passes from its first dependency wait; bypassing it
does not restart that deadline. `MAX_QUEUE_MS` bounds coalescing, not this
dependency wait: time spent queued before a producer becomes runnable does not
make duplicate prefill useful. Pure-attention models do not use checkpoint waits.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_PREFILL_DELAYER** | bool | true | Master switch for the prefill coalescer. |
| **ATOM_PREFILL_DELAYER_TARGET_FILL** | float | 0.9 | Release once accumulated pending tokens reach `target_fill × max_num_batched_tokens` (averaged across prefillable ranks). In (0, 1]; higher = fewer, larger prefills at some TTFT cost. Clamped to (0, 1]. |
| **ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS** | int | 200 | Max consecutive scheduler ticks a held prefill waits before force-release. Values `< 1` clamped to 1. |
| **ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS** | int | 100 | Tighter bound for a held mid-chunked-prefill (it holds allocated KV). Values `< 1` clamped to 1. |
| **ATOM_PREFILL_DELAYER_STALL_TICKS** | int | 10 | After this many consecutive non-growing ticks, release (burst ended, more won't come), including while some DP rank has no prefill. Values `< 1` clamped to 1. |
| **ATOM_PREFILL_DELAYER_COST_GATE** | bool | true | With DP prefill cost balancing (DP attention, DP > 1): while some DP rank has no prefill, also release when a held decode step is not expected to pay off (request arrival rate on the ranks without prefill × the decode stall of one unaligned prefill step < waiting requests), or once the held requests have waited (in measured decode steps) as long as that stall. Step times and arrival rates are measured per rank and combined over the DP group; until every rank has timed both steps, only the stall bound releases. |
| **ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK** | float | 0.9 | At/above this KV usage a prefillable rank force-releases (can't accumulate a bigger batch anyway). |
| **ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK** | float\|"" | "" (None) | If set, a prefillable rank below this KV usage force-releases (GPU starving). |
| **ATOM_PREFILL_DELAYER_MAX_QUEUE_MS** | float\|"" | "" (None) | After decode protection, release coalescing when the oldest schedulable waiting prefill reaches this age since arrival. Empty disables the age guard. Checkpoint dependency waits use `TTFT_MAX_TICKS`. This is not a hard TTFT limit. |
| **ATOM_PREFILL_DECODE_INTERVAL** | int | 0 | Protect this many scheduler passes after an executed prefill. On DP=1, PP=1, a positive value also enables local coalescing when the master switch is on; `0` leaves TP scheduling unchanged. On DP>1, `0` disables only the interval. |
| **ATOM_PREFILL_DELAYER_DEBUG** | bool | false | Per-tick FIRE/HOLD debug logging. |
| **ATOM_PREFILL_DELAYER_LOG_EVERY** | int | 1000 | Emit aggregate stats (per-exit fire counts + hold rate) every N decisions (0 disables). |

## Dynamic chunking

Interpolates pipeline-parallel chunk sizes between the configured batch budget
and the equal-latency size solved from the calibrated chunk model. Used when
neither `--dynamic-chunking-smooth-factor` nor `Config.dynamic_chunking_smooth_factor`
is given, and read only when dynamic chunking is enabled, so a malformed value
cannot affect a server that does not use the feature.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DYNAMIC_CHUNKING_SMOOTH_FACTOR** | float | 0.75 | Blend from the initial chunk (`0`) to the equal-latency chunk (`1`). Must be in `[0, 1]` when dynamic chunking is enabled. |

## Model loading

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DISABLE_MMAP** | bool | false | If set to `true`, disable memory-mapped file loading for model weights. Useful in containerized environments where mmap may cause issues. |
| **ATOM_LOADER_NUM_THREADS** | int | 16 | Worker threads for weight loading. `>1` (default `16`) enables the batched parallel loader (routed expert weights staged in a CPU buffer, flushed with a single H2D copy when every routed expert of that parameter has arrived) with that many threads; set to `1` to fall back to the original sequential per-expert path. Raise on high-core hosts if loading is CPU-bound. |
| **ATOM_TOKENIZE_CACHE_MB** | int | 512 | Host memory, in MB, the API server keeps for reusing prompt tokenization across requests. A chat template separates messages with added (special) tokens, and the text between two of them tokenizes the same on its own. Each prompt is kept with its ids up to its last added token, so the next turn of the conversation reuses them up to the last added token both share; the rest is split at added tokens and each segment's ids are cached by content, the missing ones encoded in one parallel batch. Both are exact: they are used only when the tokenizer's added tokens, normalizer and pre-tokenizer allow it, and the first prompts are checked against the plain encoder (any difference turns the cache off). Prompts under 100K characters are encoded whole. One budget covers segment ids and stored prompts, counted by the bytes they retain (ids, text, per-entry overhead). The result still covers the whole prompt, so a request costs time linear in its length; what is saved is re-encoding text already encoded. Lowers TTFT on long multi-turn conversations; ITL and throughput are unaffected. `0` turns it off. |
| **ATOM_LOADER_STRICT_COVERAGE** | bool | `true` | Fail loading when a fused MoE parameter does not receive every routed expert from the checkpoint. Set to `false` to downgrade to a warning and load anyway, leaving those expert slots at their init values — useful when bringing up a checkpoint known to be partial, misleading otherwise (the symptom is an accuracy drop much later). |
| **ATOM_LOADER_PREFETCH** | bool | `true` | Warm the page cache by reading this rank's share of the checkpoint sequentially on a background thread, instead of leaving it to demand faults through the mmap. The fault pattern sustains ~3.2 GB/s on a local NVMe that a single sequential reader drives at 6.06 GB/s, so this is an access-pattern fix, not a queue-depth one. Measured on DeepSeek-R1 MXFP4 (350 GiB, TP=4): cold load 154s → 69s. Set to `false` to restore demand faulting. Also applies with `ATOM_DISABLE_MMAP=true`, whose whole-file reads go through the same page cache (DeepSeek-V4.1-Flash, TP=4, cold: 282–315s without it, 169–207s with it). |
| **ATOM_LOADER_PREFETCH_THREADS** | int | 4 | Concurrent sequential readers per rank used by the prefetcher. The device saturates at ~2 streams, so raising this mostly adds contention with the loader; `0` is clamped to `1` (use `ATOM_LOADER_PREFETCH=false` to switch prefetching off). |
| **ATOM_LOADER_PREFETCH_BLOCK_MB** | int | 16 | Read block size for the prefetcher, in MiB. |
| **ATOM_LOADER_FADVISE** | bool | `false` | Issue `posix_fadvise(SEQUENTIAL\|WILLNEED)` per shard before reading it. Off by default and ignored while `ATOM_LOADER_PREFETCH` is on: `WILLNEED` is a hint the kernel drops for most of a 350 GiB checkpoint, and running both makes the kernel read ahead over random-ish ranges while the prefetcher streams the same files, so the two compete for the device. Only useful with prefetching disabled. |
| **ATOM_ONLINE_QUANT_STREAMING** | bool | `false` | Opt in to quantizing eligible online-quant modules as soon as their checkpoint weights are complete, then release source storage to reduce load-time peak memory. Only active with a valid online quantization config. See the [streaming online quantization guide](./online_quantization_streaming_guide.md). |
| **ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING** | bool | `true` | Assemble streamed module weights in CPU storage before one H2D transfer. Keeps the checkpoint walk parallel; disabling it buffers loader calls and forces the checkpoint walk to one thread. |
| **ATOM_ONLINE_QUANT_STREAMING_THREADS** | int | `4` | Tail workers for H2D, per-module quantization, and source release. More workers increase overlap and in-flight memory; `0` runs finalization inline. |
| **ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4** | bool | 0 (false) | Fuse an MXFP8 shared expert into MXFP4 routed experts (the Quark MXFP4 export of MiniMax-M3.1), so the MoE runs as one fused MoE with the shared expert as one more expert (129 experts, top-5). The shared expert is staged in MXFP8 while loading and re-quantized to MXFP4 after it, so it carries MXFP4 rounding error. Not supported with EPLB under all2all EP, with an `--online_quant_config` that re-quantizes the MoE, or across a weight reload. Off keeps the shared expert a separate MXFP8 MLP. Part of the compile cache key. |

## Plugin mode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DISABLE_VLLM_PLUGIN** | bool | 0 (false) | If set to `1`, disable the vLLM plugin registration entirely. |
| **ATOM_V4_CG_INSTR_DIR** | str | — (off) | vLLM plugin, DeepSeek-V4 diagnostic: directory for per-process cudagraph dispatch counters (dispatched modes, steps whose uniform decode was overridden for a still-prefilling row, guard trips), one `cg_<pid>.json` per process, rewritten on every step. |

## Kernel / backend selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_USE_TRITON_GEMM** | bool | 0 (false) | If set to `1`, use AITER Triton FP4 weight preshuffled GEMM. Otherwise use AITER ASM FP4 weight preshuffled GEMM. |
| **ATOM_FP8_BLOCKSCALE_USE_E8M0_SCALE** | bool | unset (per checkpoint) | E8M0 rather than FP32 128x128 FP8 block scales, for the weight and the activation quantized for it; on gfx950 E8M0 operands take the AITER FlyDSL GEMM. Unset: E8M0 on every arch but gfx942 when the checkpoint declares `scale_fmt: ue8m0` (DeepSeek-V4), whose power-of-two scales E8M0 restates exactly; FP32 otherwise. `1`/`0` force it on/off; forcing it on for a checkpoint with non-power-of-two FP32 scales rounds them. |
| **ATOM_GROUP32_WEIGHT_PRESHUFFLE** | bool | 1 (true) | On gfx950, (16, 16)-shuffle native FP8 32x32 group32 weights (DeepSeek-V4.1) at load and after a weight sync, so AITER's preshuffled group32 GEMM (FlyDSL) reads them; that GEMM emits BF16 only. Set to `0` to keep checkpoint bytes and the row-major group32 GEMM. V4.1's grouped `wo_a` is shuffled either way. |
| **ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM** | bool | 0 (false) | If set to `1`, use AITER Triton FP4 GEMM with non-shuffled weights. Takes precedence over the FP4 preshuffled GEMM path selected by `ATOM_USE_TRITON_GEMM`. |
| **ATOM_FP4_MXSCALE_USE_32X4_SCALE_SHUFFLE** | bool | 0 (false) | On gfx1250, if set to `1`, keep the 1x32 E8M0 weight scale of every 2D MXFP4 (`per_1x32`) linear with K % 128 == 0 in AITER's 32x4 layout, quantize the activation into the matching layout (or move the scale of one a fused norm/activation already quantized into it) and call AITER `gemm_a4w4`, which picks FlyDSL or ASM per shape. The other layers keep the Gluon FP4 preshuffled GEMM, as do those whose scale other kernels read directly (MLA `kv_b_proj`; DeepSeek `fused_qkv_a_proj` under `ATOM_USE_TRITON_GEMM=1`). No effect when disabled, on other archs, or when `ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM=1`. |
| **ATOM_MHC_USE_BF16** | bool | 1 (true) | Use AITER BF16 hi/lo mHC computation for attention, FFN and head. After loading, replace FP32 fn storage with `mhc_shuffle_fn` output; no FP32 copy is retained. Set to `0` for FP32 mHC. Takes effect at model load; restart to change modes. On gfx1250, AITER enables shuffled residuals only while its runtime `mhc_fused_post_pre` policy remains fused (`M < 1024`); larger M uses ordinary residual layout and the standalone post/pre fallback. |
| **ATOM_DSV41_MHC_BACKEND** | str | `auto` | V4.1 delayed post/pre/RMSNorm backend, selected after checkpoint loading. `auto` packs each FP32 MHC weight once for HIP on gfx950 with `hc_mult=4`, H=4096/5120/6144/7168 and AITER `mega_mhc_shuffle_fn` support; otherwise it uses Triton. `hip` requires those conditions; `triton` forces Triton. Activations/norm weights and outputs are BF16. Each MHC fn keeps one packed copy shared across shapes and streams; AITER reuses weight-independent scratch per device/stream. Warm up each capture stream with sufficient scratch capacity; calls and graphs sharing that scratch must execute serially. Set the backend before loading. Reload repacks weights and requires graph recapture. Original FP32 weights remain available for mono decode's own kernels. |
| **ATOM_USE_TRITON_MXFP4_BMM** | bool | 0 (false) | If set to `1`, use FP4 BMM in MLA attention module. |
| **ATOM_USE_FLYDSL_GATHER_KV_B_PROJ** | bool | 1 (true) | Use the FlyDSL fused gather + `kv_b_proj` GEMM for MLA's cached-prefix path. Covers page_size-1 fp8 (e4m3) KV with an fp8 weight on gfx950 — i.e. Kimi-K3 / DeepSeek MLA under `--kv-cache-dtype fp8`. Unavailable imports or unsupported tensor configurations use Triton; kernel execution errors propagate. Set `0` to force Triton. |
| **ATOM_USE_FLYDSL_FP8_PREFILL_ATTN** | bool | 0 (false) | Enable FlyDSL FP8 MLA prefill on gfx950, including fused K concatenation with QKV quantization and direct FP8 output from FlyDSL gather where supported. AITER capability checks are cached per layer. Missing FP8 attention support or unsupported device/model configurations select BF16 attention before quantization and emit a warning once per process. FP16 activations and synthetic RoPE-padding configurations also use the normal attention path. Cached K/V reuse descales `max(new_token_descale, 1e-6) * 2`; larger outliers saturate. Unsupported FP8 gather configurations use BF16 gather followed by fused K/V dynamic quantization when FP8 attention is supported. Kernel execution errors propagate without retry. Added 2026-09-10. |
| **ATOM_PA_FLYDSL** | bool | 0 (false) | Route the MHA paged decode to aiter's FlyDSL kernel, with its work planner, instead of gluon. Shapes outside FlyDSL's domain, calls over 4096 sequences and calls whose context bound is below the query window fall back to gluon. Applies to the vLLM bridge too. **Needs aiter with #5809**; upgrade ATOM and aiter together. Plans use aiter's default workgroup budget (2×CU) and partition ceiling (CU count), except at head_dim 256 (Qwen3.5), where they use 64 partitions and a 256-workgroup budget to keep aiter's head-256 reducer on its fast path (until aiter #5949). In native ATOM, captured batch sizes use a plan built by the metadata builder (shared by target and EAGLE draft); other calls, and every vLLM bridge call, get a runtime plan, and a new row count compiles the planner once. Other native dense plans use aiter's default partition ceiling; MiniMax-M3 sparse and the vLLM bridge keep their static split count. MiniMax-M3 sparse decode refreshes its plan once per step. `ATOM_PA_FLYDSL_PLAN` was removed and is ignored with a warning. |
| **ATOM_V4_HCA_PERSIST** | bool | 1 (true) | DeepSeek-V4 HCA (`compress_ratio` 128) fp8 decode through aiter's persistent V4-NM kernel (`aiter.mla.mla_decode_fwd_v4_nm_ps`): one launch that splits the KV on the GPU and merges in-kernel, instead of the decode ASM plus a host-built split plan. Taken only with fp8 KV, 128 local heads, gfx950, row-dense KV pools and `ATOM_V4_HCA_PERSIST_MIN_ROWS <= rows <= 32768`; everything else stays on the ASM path. An aiter without the kernel keeps the ASM path (logged once). One workspace per device (about 64 MiB) is allocated at model load, before KV sizing and CUDA-graph capture. Set `0` to force the ASM path. |
| **ATOM_V4_HCA_PERSIST_MIN_ROWS** | int | 15 | Smallest decode call (q rows) that uses `ATOM_V4_HCA_PERSIST`; smaller calls stay on the ASM path. |
| **ATOM_DCP_SPARSE_PREFILL_NONPS** | bool | 1 (true) | DSA models under DCP: run sparse prefill attention on the non-persistent fp8 gqa64 MLA kernel, which needs no work metadata. Applies with fp8 q/KV, 64 gathered heads (tp4/dcp4, tp8/dcp8), gfx950, `ATOM_MLA_PAGE_SIZE=1` and `--max-num-batched-tokens` <= 32768 (each prefill token is one kernel row; a larger step budget logs a warning and keeps the default kernel). Needs an aiter that ships the kernel (ROCm/aiter#6132); with an older aiter set `0`, or the first sparse prefill past `index_topk` raises. |
| **ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB** | int | 2048 | Sparse (DSA) indexer dense fp32 logits budget, in MiB. Prefill scores query rows in bands within it. It also sizes the decode logits workspace (`LogitsWorkspace`, as wide as a request of `max_model_len` sees -- a pooled indexer's pools of it -- allocated once and counted in the KV budget) and so how many bands a decode step is scored in: lowering it saves memory but adds launches. `0` keeps only aiter's hard 2 GiB buffer-descriptor cap. |

### Qwen3.8-Flash-Next (QSA)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_QSA_STACKED_DECODE_SCORER** | bool | 1 (true) | Score decode-sized QSA index calls (at most 256 query rows: the target's MTP verify and the draft) with the MFMA kernel that stacks every row's index heads into the M dimension, so each key tile is read once per 4 rows instead of once per row; BF16, FP8 and MXFP4 index keys. Needs power-of-two index heads (at least 4) and head dim; other shapes keep the CUDA-core scorer. Set `0` to force the CUDA-core scorer. |
| **ATOM_QSA_DRAFT_KV_BF16** | bool | 1 (true) | With `--kv_cache_dtype fp8` or `--index-cache-dtype fp8/fp4`, keep the MTP draft layer's K/V and index caches in BF16, in a row space (`qsa_draft`) of their own, while the target's 12 QSA layers use the quantized caches. On Qwen3.8-Flash-Next an FP8 draft KV lowers the real acceptance length by about 3.5% (3.05 vs 3.17 tokens a forward at MTP 3); a BF16 one keeps it, at about 6% of the KV pool. Set `0` to quantize the draft's caches like the target's. |

### GLM-5.3

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_GLM5_KPOOL** | bool | 1 (true) | Enable the pooled sparse indexer. Setting `0` is an exact token-granular A/B only at or below `index_topk`; longer requests are refused. |
| **ATOM_GLM5_FORCE_DENSE_MLA** | bool | 0 (false) | Disable sparse MLA for short-context bring-up comparisons. |
| **ATOM_GLM5_DISABLE_FUSED_MHC** | bool | 0 (false) | Force the PyTorch mHC reference path instead of AITER's fused kernels. |

### Native model decode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_MONO_ENABLE** | bool | 1 (true) | Enable fused decode for supported GLM-5.2, Kimi-K3, MiniMax-M3, and DeepSeek-V4.1 deployments and step geometries. Unsupported configurations and forwards keep the production path; `0` disables every Mono runner. |

### MiniMax-M3

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_MONO_CHECK** | bool | 0 (false) | Debug aid for `ATOM_MONO_ENABLE`. MiniMax-M3: the dense layers run mono and then the original modules, compared where they hand the first sparse layer its input; every sparse layer also runs the original decoder layer on the same input (after the mono layer, so mono reads only the cache entries it inserted) and rank 0 logs the per-layer, per-token difference of the residual and of the reduced output. DeepSeek-V4.1: K2a and K2b are launched apart and every layer's kernel outputs are compared with the original modules run on the same inputs (K1: seam, query, window row, indexer query / weights; the indexer's selection and candidate blocks against the original scorer; K2a: its inverse-RoPE quant, `wo_a`, residual and FFN seam; K2b: router logits and the reduced MoE output, the reference also run twice to show its own run-to-run change), logged on rank 0 as the fraction of differing elements, the largest difference and the relative error (`atom/mono/runtime/compare.py`; MiniMax-M3 logs the same three). Run it with `--enforce-eager`. |
| **ATOM_MONO_TRACE** | path | unset | Debug aid for `ATOM_MONO_ENABLE`: rank 0 appends every mono step's input tokens, positions, top-2 logits and greedy pick to this file (one JSON line per step), with a fingerprint of every stage (dense layers, each sparse layer's output, the cache history it reads). Two runs of the same prompt then align token by token. Run it with `--enforce-eager`. |
| **ATOM_MONO_DEBUG** | bool | 0 (false) | Debug aid for `ATOM_MONO_ENABLE`: the mono kernels are built with bounded mailbox waits. A wait still unmet after 10 s gives up and records its region, pair index, expected and seen tags and CTA; at the end of the step the runner raises with every record (a region recorded by its `region_id`, the CRC-32 of its name, and reported by name: the kernels' hand-off tables; DeepSeek-V4.1 adds its kernels' scratch regions, the peer regions and K2's readiness flags) instead of the GPU hanging on a peer or stage that never delivers; the step's peer fence (`atom/mono/runtime/step_begin.py`, after each zeroing of the peer buffer) likewise gives up and names the ranks that never arrived. Run it with `--enforce-eager`: the records are read after a synchronize, which a graph replay does not run. |
| **ATOM_MONO_TIMELINE** | path prefix | unset | Debug aid for `ATOM_MONO_ENABLE`: the mono layer kernels are built with their per-phase `s_memrealtime` stamps, each sparse layer into its own buffer, and after 20 warm-up steps of every decode token count S the next 5 steps are saved per rank as `<prefix>_r<rank>_s<S>_<i>.pt` (int64 [layer][CTA][stamp], 100 MHz ticks, 0 = not reached); DeepSeek-V4.1 saves its K1 and K2 stamps apart, as `<prefix>_k1_r...` and `<prefix>_k2_r...` (K2a's points, then K2b's; with `ATOM_MONO_CHECK` `<prefix>_k2a_r...` and `<prefix>_k2b_r...`). Run it with `--enforce-eager`: a graph replay runs no Python. |
| **ATOM_GLM52_FUSED_INDEXER** | bool | 0 (false) | GLM-5.2 only. Moves the sparse indexer *inside* the mono decode launch on the 18 "full" layers that actually select: index Q/K/W projection, the paged FP8 index-cache update, scoring and the exact top-2048 all run in one kernel instead of `_refresh_indexer`'s separate pass. Off leaves the external-indexer mono path untouched. Requires `--index_cache_dtype fp8` (the default when `--kv_cache_dtype fp8`), `--block-size 16` exactly (the index plane shares `block_tables` with the KV cache), and `--max-model-len` at or under `FUSED_INDEX_MAX_SEQ`, which is **1048576** tokens. 16384 is not a ceiling but the build granularity: `fused_index_capacity()` rounds each deployment UP to a whole 16384-token leaf and builds at that, because the scoring work map is enumerated from the compiled capacity rather than from the request — a 16 k deployment built against a 1 M capacity inflates that table 64x to do identical work (measured: +11.14 % against the external indexer at a 1 M build, -0.09 % at a 16 k build). Anything else is refused by name in `atom.models.glm52.mono.config.fused_indexer_refusal` and falls back to the external indexer unchanged; "shared" layers (57 of 78) never use this kernel either way. |
| **ATOM_GLM52_INDEXER_CP** | bool | 0 (false) | GLM-5.2 only, and strictly narrower than `ATOM_GLM52_FUSED_INDEXER` — on its own it does nothing. Shards index scoring and selection across the TP ranks inside the fused indexer: each rank scores a contiguous 64-token-aligned shard of the query's live context and the ranks exchange score shards through in-kernel peer mailboxes. Selection stays exact and the index caches stay replicated — CP distributes computation, not storage. Refusals in `atom.models.glm52.mono.config.indexer_cp_refusal`; a refusal leaves the fused indexer running exactly as it does with CP off. |

#### Mono decode routing

**DeepSeek-V4.1** (`atom/models/deepseek_v41/mono/`): two kernels a layer -- K1 (mHC seam, attention norm, q / kv projections, the indexer query) and K2, one launch of K2a (the indexer's scores and top-k at any context length, the sparse decode attention, the output projections and their all-reduce, the FFN seam) then K2b (the MoE and its all-reduce) -- with the compressor and Engram left to the original modules. It routes a decode step of up to 48 rows, however they split into requests (a DSpark verify of any length a request, confidence-scheduled or ragged included, or one token a request without a drafter; each row's window from its own batch id; one kernel build per width) and the DSpark draft's block backbone of up to 8 requests at the full block width (`draft_runner.py`; after a prefill as after a decode), the draft's LM head and Markov sampler unchanged. It needs TP2, TP4 or TP8 without EP (the routed experts sharded across TP; every per-rank width from `kernels/dims.py`; TP8's kernels are compile-checked only, since the original path does not start at TP8), DP / PP / CP 1, no TBO, no plugin mode, FULL cudagraph or eager, fp8 (MXFP8) or bf16 KV and an fp8 or fp4 index cache (fp4: K1 quantizes the indexer query to E2M1 and K2a scores with `index_score_fp4`), no speculative decoding, or DSpark with a request's verify (speculative tokens + 1) of at most 48 rows (another drafter's hooks on the target layers would not fire under mono), `ATOM_DSV41_SIDE_STREAMS=0`, routed experts in A8W4 (MXFP4 weights, MXFP8 activations, gate / up interleaved: checked when the weights bind), at most 1024 attention keys a token (sliding window + index top-k) and the indexer shape the kernels are written for (top-512, 2048 candidate blocks of 8 rows, 32 index heads of 128).

**MiniMax-M3** (`atom/models/minimax_m3/mono/`): decode steps of up to 16 tokens (requests x speculative query tokens; a verify's tokens one row each), dense and sparse layers, Eagle3 aux hidden states included. It needs TP4, ptpc-FP8 attention linears, fp8 KV and index cache, KV block size 128, `max_model_len` up to 1M, no TBO / DP / PP / plugin mode, and FULL cudagraph or eager. A layer that reuses an earlier layer's selection is served, as long as the first sparse layer selects. The same entry class serves both paths.

**GLM-5.2** (`atom/models/glm52/mono/`): the MoE suffix of the layers, at decode widths of 1 token (no speculation) or an MTP verify of 4, 5 or 6, batched 1 or 2. It needs TP4, ptpc-FP8 attention linears, fp8 KV, `ATOM_MLA_PAGE_SIZE=1`, and no DP attention / plugin mode / shuffled MLA cache. Unlike DeepSeek-V4.1 and MiniMax-M3 it does **not** refuse TR_HYBRID: it routes inside the *uncompiled* outer `GlmMoeDsaForCausalLM` and bypasses the compiled inner `DeepseekV2Model` entirely, so a token graph of that module never runs underneath mono and cannot conflict with it (`common_refusals(..., refuse_tr_hybrid=False)`). TR_HYBRID — which a config with no `cudagraph_mode` resolves to at level 3 (`Config.__post_init__`; the CLI itself defaults `--cudagraph-mode` to FULL) — is accepted, as is FULL and eager. By default the indexer runs outside the kernel (`_refresh_indexer`); `ATOM_GLM52_FUSED_INDEXER=1` moves it inside the launch on the 18 full layers, with the extra requirements listed in the table above, and `ATOM_GLM52_INDEXER_CP=1` additionally shards its scoring across the four ranks. Which one a width ended up on is named in the `GLM-5.2 MonoKernel on: S=.. chunk=.. indexer=..` line, not inferred.

## MoE all2all (MoRI) wire format

Both are opt-in and default to off; they only apply with DP attention + expert
parallelism. They are *not* symmetric — FP4 dispatch only moves a quantization
the MoE GEMM was going to perform anyway (it consumes FP4 activations either
way, and `per_1x32` is per-row, so it does not matter which rank runs it), while
FP8 combine adds a quantization that would not otherwise happen, since the
expert output is bf16. Treat the dispatch knob as format matching and the
combine knob as a quality/throughput tradeoff.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_MORI_FP4_DISPATCH** | bool | 0 (false) | If set to `1`, quantize activations to packed FP4 (E2M1, `per_1x32`) before the MoE all2all instead of sending bf16 — a quarter of the bytes on the dispatch wire — which selects `EpDispatchIntraNodeKernel_fp4`. MoRI picks its dispatch kernel from the dtype of the tensor handed to `dispatch()` but sizes its staging buffers from the config built at init, so this also switches `scale_dim` to `hidden_dim/32` and the scale type to e8m0. All three are resolved together by `routed_rows.resolve_routed_rows_wire()`; never set one without the others, as a mismatch strides the staging scale buffer wrong and faults on the first real batch instead of erroring cleanly. |
| **ATOM_MEGA_STAGE2_FUSED** | bool | 1 (true) | Under `ATOM_MORI_V2=1`, run the MoE layer on aiter's `MegaMoEGfx1250` (MegaMoE): the gemm2 epilogue P2P-writes each weighted route into the peers' combine staging, so combine only barriers and sums. `0` falls back to mori's v2 op-layer with plain gather combine. Formerly `ATOM_MORI_V2_FUSED`. |
| **ATOM_MEGA_STAGE1_FUSED** | bool | 1 (true) | MegaMoE compact stage 1 (aiter `stage1_fused`): routing/layout planning fused with the flydsl TDM dispatch, which writes the grouped GEMM's per-expert rows directly. Only takes effect with `ATOM_MEGA_STAGE2_FUSED=1` and `MEGA_DISPATCH_WIRE=fp8`/`fp4`; a bf16 wire keeps the token-major path. Not supported together with the Triton experts. |
| **ATOM_MEGA_COMBINE_WIRE** | str | `bf16` | MegaMoE (`ATOM_MEGA_STAGE2_FUSED=1`) combine wire: `bf16`, `fp8` (mxfp8) or `fp4` (mxfp4). Prefill-only: decode steps always combine in bf16, and the choice is DP-agreed so every rank reduces in the same format. |
| **ATOM_MEGA_DECODE_MTPR** | int | 128 | Capacity (rows per rank) of the small MegaMoEV2 instance that DP-unified decode / verify / draft passes reuse (`ATOM_MEGA_DECODE_FAST_PATH=1`); one of 128, 256, 512, 1024. A pass with more rows falls back to the instance sized for `max_num_batched_tokens`. Size it to cover the DP-padded decode graphs (`running_bs * (num_speculative_tokens + 1)` rows); above 128 the instance stays on aiter's fixed-slot path only if `AITER_MEGA_FIXED_SLOT_MAX_MTPR` admits it (511 / 1023 / 2047), otherwise it runs compact. |
| **ATOM_MORI_COMBINE_QUANT** | str | `none` | Combine-side codec passed into the MoRI config. `none` returns bf16; `fp8_blockwise` selects `EpCombineIntraNodeKernel_*_fp8bwq_*`; MoRI also accepts `fp8_direct_cast`. |

## EPLB

Applies with `--enable-eplb`; the schedule itself is set by `--eplb-config`
(`load_window_size`, `rebalance_interval`, `rebalance_min_balancedness`, ...).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_EPLB_MAX_REBALANCES** | int | 0 (no limit) | Stop periodic rebalancing once this many rebalances have run, and serve with that expert placement for the rest of the process: balance on early traffic, then pay no further migrations. Only rebalances that run count; an interval the balancedness gate skips does not. The limit is checked after a rebalance's last migration chunk commits, on every rank in lockstep; after it, the manager also stops reducing the per-step has-prefill flag. Load recording continues (its buffers are fixed for CUDA graphs) but is no longer read. `0` or a negative value keeps rebalancing for the life of the process. Read when the EPLB manager is built, so set it before the server starts. |

## Fusion passes

### RMSNorm

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_USE_MODEL_SENSITIVE_RMSNORM** | bool | 0 (false) | If set to `1`, use AITER's model-sensitive RMSNorm rounding mode. This can change numerical results and prefill performance, so it is opt-in. |

### TP AllReduce fusion

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_ALLREDUCE_RMSNORM_FUSION** | bool | 1 (true) | If set to `1`, fuse allreduce with RMSNorm in tensor parallel mode. |

### DeepSeek-style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_DS_INPUT_RMSNORM_QUANT_FUSION** | bool | 1 (true) | If set to `1`, fuse RMSNorm with quantization. |
| **ATOM_ENABLE_DS_QKNORM_FUSION** | bool | 1 (true) | If set to `1`, use the fused Q/K RMSNorm path (`fused_qk_rmsnorm`) in the DeepSeek MLA attention module when Q-LoRA is enabled and QK norm+quant fusion is not used. If set to `0`, apply separate RMSNorm for the Q and KV branches instead. |
| **ATOM_ENABLE_DS_QKNORM_QUANT_FUSION** | bool | 1 (true) | If set to `1`, fuse QK norm with quantization in MLA attention module. |
| **ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD** | int | 1024 | Upper bound on MoE token count (`num_tokens` in the MoE forward) for using the dual-stream path: shared experts on a secondary CUDA stream while routed experts run on the default stream. If `num_tokens` exceeds this value, that forward uses single-stream MoE instead. Set to `0` to disable dual-stream setup entirely (no alt stream, no `maybe_dual_stream_forward` registration). The fork is taken in eager runs, FULL capture and token-graph segments alike; only a plugin frontend's own PIECEWISE graphs stay single-stream. |

### DSpark block sampling

DSpark drafts a `num_speculative_tokens`-wide block in one backbone pass, then
samples it left-to-right with a low-rank first-order Markov head
(`logits_k = base_logits_k + W1[x_{k-1}] @ W2ᵀ`, `x_k = argmax(logits_k)`). The
unfused loop casts the whole `[V, r]` `W2` table to fp32 on every iteration and
materializes two `[B, V]` fp32 tensors that only an `argmax` reads. See
`atom/model_ops/dspark_markov_sample.py`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DSPARK_FUSED_MARKOV_SAMPLE** | bool | 1 (true) | Sample the DSpark block with a fused Triton kernel that computes the rank-`r` bias GEMV, adds the base logits in the GEMM epilogue and reduces to token ids in registers — so `W2` stays bf16 and is read exactly once per block position, and no `[B, V]` intermediate exists. Covers both native DSpark block samplers, Kimi-K3 (`r=256`) and DeepSeek-V4 (`r=512`); the op is shape-generic and hands anything it cannot index back to the reference, but only K3 has been run on hardware. Tie-breaking matches `torch.argmax` (lowest index). The bias moves from an fp32 matmul to bf16 MFMA with an fp32 accumulator: every product is exact in fp32 either way, so the result is equal to the reference up to accumulation order. Measured on Kimi-K3 (MI355X, TP8, fp8 KV, full GSM8K 5-shot at 64 concurrency): acceptance 87.08% against 87.06% unfused with the accept-length distribution equal to within 0.1pp, and flexible-extract inside the run-to-run band. Saves 145 µs per drafting step at B=1 and ~235 µs at B=64. Set to `0` to force the reference spelling if an acceptance-rate regression is suspected — the two paths are not bit-identical by construction, so this is the fastest way to rule the kernel in or out. Read at Markov-head construction, so set it before the server starts. |

### Kimi-K3 DFlash draft

The Kimi-K3 DFlash draft is a standalone SpecForge-trained block drafter with a
GQA backbone, so it owns a KV pool of its own and reads it non-causally. See
`atom/models/kimi_k3_dflash.py` and `atom/spec_decode/draft_kv.py`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_K3_DRAFT_SGLANG_ATTN** | bool | 1 (true) | Take the DFlash draft's block attention through the SGLang two-stage verify kernel vendored under `atom/model_ops/attentions/sglang_port/`, which splits the block's one attention into a prefix pass over the paged pool and an in-register pass over the block's own freshly computed K/V, merged by LSE inside the kernel. Set to `0` to fall back to the aiter `mha_v3.flash_attn_with_kvcache(causal=False)` read, which attends prefix and block together out of the pool. The fallback requires a bf16 draft KV pool — it cannot read the fp8 one — and the draft pool inherits the target's dtype, so it needs the target run at `--kv_cache_dtype bf16` (ATOM registers that flag in underscore form only). It is the same pool either way (`fa_views` is entry-granular, which is the page the aiter read's `page_table` names); the two differ only in how the block's attention is decomposed, which makes this the fastest way to rule the vendored kernel in or out if the draft is suspected. Affects only a DFlash draft; no other model reads it. Read once when the draft attention and the proposer are built, so set it before the server starts. |

### Qwen3 style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION** | bool | 0 (false) | If set to `1`, fuse QK norm, RoPE, and cache quantization into one kernel for Qwen3 dense and MoE models. |
| **ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS** | int | 8192 | With the fusion above on, Gemma-norm models (Qwen3-Next, Qwen3.5) run the q/k norm, RoPE and cache write as separate kernels when a batch has more tokens than this; the fused kernel is slower at prefill sizes. |

### Llama-style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_RMSNORM_QUANT** | bool | 1 (true) | If set to `1`, use Triton kernel to fuse RMSNorm with quantization. |
| **ATOM_LLAMA_ENABLE_AITER_TRITON_FUSED_SILU_MUL_QUANT** | bool | 1 (true) | If set to `1`, use Triton kernel to fuse SiLU and mul with quantization in MLP module. |

### Draft CUDAGraphs (all drafter flavors)

A drafter declares its forward passes as `DraftGraph`s (`atom/spec_decode/drafter.py`).
At the end of CUDAGraph capture the runner runs each one once per captured batch
size, so the per-shape JIT — aiter's flydsl builds an hgemm per tile config,
in-process — is paid at startup instead of stalling a serving step. At serve
time a pass runs at the batch the target just ran, which `ForwardMode.decide`
picks out of those same `capture_sizes` — that is what makes a warmed shape and a
reachable shape one set rather than two lists that drift. A pass must support capture; the switch below then decides whether its warmup also records. V4.1 DSpark currently warms and drafts eagerly because its request windows and expert dispatch are outside the whole-block capture contract.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DRAFT_CUDAGRAPH** | bool | 1 (true) | Capture each declared draft pass into a per-`capture_sizes` CUDAGraph as it is warmed, so a draft pass replays instead of relaunching every kernel. `0` keeps the warmup (and therefore the JIT saving) but drafts eagerly. Only declared passes with capture support are recorded; every drafter ATOM ships declares at least one pass, including the separate-draft Kimi-K3 path, whose block pass builds its paged metadata at warmup so nothing host-side is left inside the recording. EPLB no longer declines the padding: the target pads on every cudagraph decode step and its rows reach the same expert-load recorder, so declining on the draft protected nothing. A DP-sync dummy DOES replay, in lockstep with the ranks holding work — `is_dummy_run` is per-rank, so gating on it splits one DP group across two collectives. Measured on V4-Flash-DSpark tp1: GSM8K 0.9527 / acceptance 65.25% captured against 0.9497 / 65.21% eager, i.e. indistinguishable; on tp4 with the LM head inside the capture, draft kernel launches went 30 → 0 per pass and draft wall time 915.8 → 118.9 µs. Read per pass at warmup time, so set it before the server starts. Grep a trace for a trailing ` graph` in a `propose_*` label to confirm which passes replayed. |

### DSpark drafting

The Kimi-K3 DSpark draft writes the target's context rows into its own paged MLA
cache once per draft layer per drafting step; the switch below shortens that
path. The first write of each process logs which path it took, and logs again if
that ever changes, so a fusion left inert by an unrecognised layout says so.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DSPARK_FUSED_CTX_KV** | bool | 1 (true) | Write the context rows with one Triton kernel (RMSNorm + RoPE + concat + paged store) instead of four launches plus a throwaway `empty_like` for the RoPE's query side. Falls back per call when the cache layout or the RoPE is not the plain one the kernel understands (seg / shuffled-KV layouts keep their own write kernels), and until the RoPE's cos/sin cache has reached the device. Measured on Kimi-K3 (MI355X, TP8, fp8 KV): one 4.65 µs kernel replaces a 14 µs three-kernel chain, saving ~39 µs per drafting step at B=1 and ~36 µs at B=64. Set to `0` to force the per-op chain; that chain is the fallback above rather than debug code, so it stays reachable either way (it runs the first write of every layer). |
| **ATOM_DSPARK_DISABLE_COMPILE** | bool | 0 (false) | Run the DSpark draft eager while the target stays compiled. Prefer it over `--level 0`, which drops compilation for both models; `--enforce-eager` does not reach it, because `support_torch_compile` keys off `compilation_config.level` alone. Flips the decorator's own bypass rather than handing the draft a cloned config, so the shared `static_forward_context` registry stays one object. |

### Speculative acceptance

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_RELAXED_MTP** | bool | 0 (false) | Accept a draft token when it lands in the target's top 10 within 0.6 of the top logit, instead of requiring the argmax. Intended for quantized MTP heads, whose drafts are right about the region and wrong about the exact winner often enough that strict acceptance throws away usable tokens. Read once at `rejection_sampler` import, so it must be set before the server starts. |

## Engram (DeepSeek-V4.1)

The n-gram tables are per-layer and large enough that where they live, and
whether they are rebuilt, both show up at startup. Both switches below are
all-or-nothing on purpose: a half-registered set would keep the host path for
some layers and the device path for others, which is the confusing state.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENGRAM_UVA** | bool | 1 (true) | Page-lock this rank's shard of the hash tables (copied into page-locked memory by default, see `ATOM_ENGRAM_UVA_COPY`) and let a device kernel read the rows it needs across the bus, dequantizing there. No HBM for the table. `0` falls back to gathering the rows on the host, which returns the same rows but costs ~50 ms of CPU per decode step with the GPU idle behind it. Anything that would make the device path unsafe — no CUDA, more TP ranks than hash heads, a registration that will not fit — falls back on its own, so the switch is for taking the host path deliberately. The fallback is the whole TP group's: the lookup ends in an all-gather, so one rank that cannot register turns every rank around rather than leaving the others in a collective it never enters. |
| **ATOM_ENGRAM_UVA_COPY** | bool | 1 (true) | How the UVA path page-locks this rank's shard. `1` copies the rows into page-locked memory of exactly their size (`hipHostMalloc`, on base pages: with transparent huge pages it compacts memory once the page cache has fragmented it, ~197 s per TP2 rank instead of ~12 s): one `copy_` below 512 MiB, otherwise one thread per 128 MiB but at least 8, at most 16 and the CPUs this process may run on (two or three threads lose to one `copy_`); each copied chunk of the checkpoint is dropped from the process (`madvise(DONTNEED)`; it stays in the page cache, reclaimable), so the shard is resident once and the peak stays near one shard. `0` registers the checkpoint mapping in place (`hipHostRegister`), which copies each page on its write fault on one thread while holding the HIP runtime lock; the shard is resident once as private anonymous pages. The page-locked memory is owned by the tensors the kernel reads and released with the last of them. Needs the HIP runtime; without it the shard is registered in place. `tools/engram/bench_uva_shard.py` reproduces per-rank time, memory and gather latency for either mode. |
| **ATOM_ENGRAM_CACHE_DIR** | path | `~/.cache/atom/engram` | Where the compressed-vocab table is cached between runs. The table is reproducible from the tokenizer, so this only trades startup time for disk; point it at shared storage to let several servers build it once. A truncated or stale cache is rebuilt rather than raised. |

## Attention side streams (DeepSeek-V4.1)

A layer's compressor reads the hidden row and its own arena state, and its
indexer reads the normed query latent; neither reads what the query projection
and the fused rope/window launch produce, so the three are branches of one
dependency graph that a single stream serializes.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DSV41_SIDE_STREAMS** | int (0/1/2) | 0 | How many of a layer's branches leave the main stream. **0** — none. **1** — the compressor, on the MoE's `alt_stream`, waited at the scorer rather than at attention, because `visible` counts the index row a boundary crossed in this same forward writes, so the scorer is its first reader a whole top-k chain earlier. It borrows that stream because the MoE joins it inside its own forward, a sublayer after the compressor was joined, so the two are never live at once. **2** — the same, plus the indexer on a stream of its own; it is the only branch live beside both others, so it is the only one worth a queue. A value outside 0–2 raises rather than rounding. |
| **ATOM_TOKEN_GRAPH_SIZES** | str | "256,512,1024,2048,3072" | Prefill token buckets to record as TR graphs (`atom/utils/region_graph.py`), comma-separated. Each bucket's forward is recorded once at startup as graph segments broken at the model's eager regions (indices, sparse attention, window write, compressor, indexer, Engram), and a prefill padded to the smallest bucket that holds it (under DP attention, every prefilling rank's to the one holding the group's largest) replays them instead of launching every kernel; a longer prefill runs eagerly. Empty records nothing. Only a backend with `supports_token_graph` (DeepSeek-V4.1) uses it, and a disaggregated decode instance records none. The default stops at 3072 because the graph pool grows with the largest bucket: on V4.1 TP4 per GPU, FULL decode alone takes 1.15 GB, the five default buckets 3.1 GB, buckets up to 16384 10.8 GB. Below 3072 a prefill step is CPU-bound, so a graph cuts its TTFT by 30–40 ms and padding to the next bucket costs about nothing; longer prefills are GPU-bound. The KV budget's graph estimate does not include these buckets: they come out of the safety margin. Under DP attention a rank with no prefill of its own runs a dummy prefill while a peer prefills, instead of decoding, and every rank of the step replays one bucket, the smallest holding the group's largest row count. Without the prefill delayer (`ATOM_ENABLE_PREFILL_DELAYER=0`) a rank may decode beside a peer's prefill; it pads its decode rows to that bucket and replays the same prefill recording. |
| **ATOM_TOKEN_GRAPH_DP_MAX_TOKENS** | int | 512 | Under DP attention with a drafter, the largest decode token bucket recorded as a TR_HYBRID token graph; larger `bs × q` buckets are skipped at capture and left out of the graph memory estimate. Such buckets never run under DP, and recording them only grows the graph pool. The same skip on every DP rank keeps their capture loops in step. |
| **ATOM_DECODER_GRAPH_SIZES** | str | "1,2,3,4,8,16" | The `running_bs` at which DeepSeek-V4.1's decoder (layers 20..39) on a bounded-replay prefill's tail is recorded as a request graph (`models/deepseek_v41/decoder_graph.py`), comma-separated. The tail is laid out for the step's `running_bs` requests, a window each, at one set of addresses per `running_bs`, so a recording replays the whole decoder in one launch; any other `running_bs` runs the same layout eagerly, and empty records nothing. Not on an FP8 index plane. Under DP attention each size also records the tail of padding a rank finishing no prompt runs beside its peers. On a 16k-token prefill step the decoder in one graph cuts the step's CPU time from 84 to about 45 ms; the six default sizes add about 0.33 GB to the graph pool on V4.1 TP4. |

**Level 0 is the default because forking measured slower, not faster.** Median
steady-state layer period, MI355X TP4 bf16 KV DSpark-5, 1024/1024 at
concurrency 64, ~10k sampled layers per trace:

| level | layer period | vs 0 |
|-------|--------------|------|
| 0 | 636.40 µs (repeat: 636.72) | — |
| 1 | 645.76 µs | +1.47% |
| 2 | 640.32 µs | +0.62% |

The anchor is the gap between consecutive `topk_gating` launches, so a branch
that moves to another stream cannot drop out of the sample. The two level-0
traces were taken either side of the other two, which puts the floor — session
drift included — at 0.05%, making those deltas 12× and 29× the noise; the
unfiltered medians (604.0/605.9 against 618.8 and 607.6) order the levels the
same way. End-to-end throughput cannot resolve this: one wall-clock number per
run carries 2–6% spread, which is why an earlier A/B called the same
arrangement a wash. Whoever revisits it should start from these numbers and
from `GPU_MAX_HW_QUEUES`, not from where the forks sit.

## V4 attention backend (Migration)

Selects between the legacy per-seq Python dispatch path in `atom/models/deepseek_v4.py`
and the new batched `V4AttentionBackend` (`atom/model_ops/v4_attention_backend.py`).
The new backend removes ~256 GPU→CPU `.item()` syncs per forward and is required
to enable CUDAGraph capture for V4. Legacy stays available during PR-A migration
for byte-equal A/B verification via dump-bisect; it is removed once all phases
land. See `atom/model_ops/v4_backend_gate.py` for the selector.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_V4_BACKEND** | str | `legacy` | `legacy` keeps the per-seq dispatch loop. `new` routes through `V4AttentionBackend`. Layer-restricted by `ATOM_V4_BACKEND_LAYERS` if set. |
| **ATOM_V4_BACKEND_LAYERS** | csv int | "" (= all) | Comma-separated layer ids that use the new backend (others stay legacy). Empty means: apply `ATOM_V4_BACKEND` uniformly. Used for layer-by-layer bisect during migration (e.g. `0,3,15,30`). |

## Prefix cache lookup

These options default on and reduce repeated prefix work or move it off the scheduler thread.
`ATOM_FRONTEND_PREFIX_HASH=0` and `ATOM_PREFIX_WALK_MEMO=0` give the same hits.
`ATOM_PREFIX_FAST_WALK=0` does not always: the fast walk compares tokens only on the last hit block, so a
64-bit xxhash chained-hash collision on an interior block reuses another
prompt's KV with no error (wrong output). `0` compares every hit block's tokens
and catches it; set it when chasing a corruption that does not reproduce.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_TOKENIZE_CACHE_BLOCK_HASHES** | bool | true | Keep the prompt block-hash chain that frontend prefix hashing ships to the engine (`ATOM_FRONTEND_PREFIX_HASH`) with each prompt the tokenizer cache stores, and on the next turn hash only the blocks after the tokenizer's matched prefix instead of the whole prompt. Content hashes only: whether a block's KV is still resident is decided by BlockManager, so this cannot turn into a KV prefix hit. The hashes count toward `ATOM_TOKENIZE_CACHE_MB`. Needs prefix caching, `ATOM_FRONTEND_PREFIX_HASH=1` and an active tokenizer cache; multimodal and token-ID requests hash as before. `0` hashes every prompt in full. |
| **ATOM_FRONTEND_PREFIX_HASH** | bool | true | Hash the prompt's block chain while the request is built (API server preprocessing executor) and ship it to the engine with a fingerprint of the prompt tokens, beside the request in the ADD message (never inside a pickled `Sequence`; with `DisaggCoreManager` only the decode engine, which allocates, gets it). The engine adopts it if block size, seed, length and fingerprint match and its first, middle and last links rehash equal, then uses it for the prefix walk, `allocate` and publishing the prompt blocks after prefill, so prefill hashes no prompt block on the scheduler thread (it hashes the prompt bytes once for the fingerprint, about 26 us at 186k tokens; the LMCache/offload and PD-transfer publish paths still hash their blocks). Any mismatch falls back to engine-side hashing. Read in the API server process. |
| **ATOM_PREFIX_FAST_WALK** | bool | true | Look the prompt's chain up with C-speed `map` rounds (the first block alone, then 16, 64, ... blocks, so a cold prompt costs one lookup), compare tokens on the last hit block only (all blocks if that one differs), and let `allocate` claim the blocks `can_allocate` found instead of walking the prompt again; any block that changed in between sends it back to the block-by-block walk. `0` walks block by block with a token compare per block. |
| **ATOM_PREFIX_WALK_MEMO** | bool | true | Reuse a waiting request's prefix walk across the probes that run every scheduler tick: the delayer's head check and probe, and the DP chunk-cost estimate (`BlockManager.can_allocate_memoized`). It walks again when a hash the walk read is evicted, remapped to another block or revoked by a republish of its block, the hash at its first miss is published, `clear_cache` runs, more than 8192 such changes happened since its last check, or the request's token count or seed changed. The fit, state gates and replay are re-asked on every tick, so the answer equals the exact probe; admission still probes exactly. `0` re-walks every tick. |

## State checkpoints

For models carrying per-request recurrent state (GDN: Qwen3-Next / Qwen3.5;
Kimi-K3's KDA; DeepSeek-V4's compressor ring), a checkpoint lets a later prefix
hit resume mid-prompt instead of recomputing from zero. *Where* they are placed
is a policy, set by `--state-checkpoint-interval-tokens` (three regimes carried
by the sign — see the [configuration guide](configuration_guide.md)) and the
flag below. Details in the state-checkpoint section of the
[scheduling & KV cache guide](scheduling_kv_cache_guide.md).

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_STATE_CHECKPOINT_DEMAND** | bool | 1 (true) | Set to `0` to stop a prefix hit that was refused for want of a checkpoint from placing a rung of its own, leaving the prompt-end anchor as the only placement. Overrides `--state-checkpoint-demand`, so the policy can be A/B'd without editing a launch script. The rung is most of the checkpoint write traffic and little of the read-back, and every write evicts something — `StateSlotPool.mark_speculative` carries the measurement. |
| **ATOM_STATE_CHECKPOINT_END_MARGIN** | int | 0 | Minimum number of prompt tokens left after the prompt-end anchor. The anchor goes on the last hash-block boundary at least `max(successor room, min final prefill rows, margin)` tokens before the prompt end: the largest of the three, not their sum. A chat template can re-render the tail of a prompt when the next user turn is appended: Qwen3.5's generation prompt ends in `<think>\n`, which the next turn drops, so the next prompt diverges at P-2 and an anchor whose last block covers P-2 (every prompt with P % 16 == 1) is never matched again. Set it to the template's generation-prompt length (`5` for Qwen3.5) to keep the anchor before that suffix. `0` keeps the old placement. Cost when the next turn does resend the prompt verbatim: the anchor moves back by up to `ceil((margin - room) / block)` hash blocks of recompute, where room is the larger of the other two terms: at most one block while the margin exceeds the room by no more than one block (Qwen3.5: margin 5, PAGE room 0, block 16), several for a larger margin. A prompt shorter than `margin + one block` keeps no anchor at all. |
| **ATOM_DSV4_STATE_REPLAY** | bool | 0 (false) | DeepSeek-V4: keep a prefix hit that reaches past the nearest state checkpoint by more than the window's receptive field (7,748 tokens on V4-Pro, 7,875 with MTP or DSpark) and recompute only that field below it, which rebuilds the state exactly, instead of resuming at the checkpoint (`Config.state_replay_tokens`). Off on anything but the DeepSeek-V4 trunk with compress ratios 4 / 128, and with context parallelism, a speculative method other than MTP or the model's own DSpark layers (not a separate `--draft-model`), prefix caching off, or a KV connector other than LMCache MP offload alone. With that one LMCache saves carry the PAGE KV alone, no state image (the native layout registers the PAGE group alone, under a namespace of its own, so a stock LMCache server's hit is how far that KV reaches), and a hit whose PAGE KV LMCache holds past the HBM prefix loads that KV alone and replays below it. Every other model, and DeepSeek-V4 without the replay, keeps storing images. |
| **ATOM_STATE_CHECKPOINT_INTERVAL_TOKENS** | int | — (the model's native interval) | vLLM plugin (DeepSeek-V4 state checkpoints): overrides the checkpoint interval in tokens, snapped down to a multiple of the block size. `>0` a rung every N tokens plus the prompt-end anchor and the demand rung; `-1` no rungs, anchor and demand only; `0` no state checkpoint is published, so there is no prefix hit at all (vLLM reports a hit only when every state checkpoint group is cached; the state checkpoint groups still allocate their working blocks). |
| **ATOM_STATE_CHECKPOINT_CHECK_BLOCKS** | bool | 0 (false) | vLLM plugin (state checkpoints) diagnostic: after every schedule, raise if a request holds more than two blocks in any state checkpoint group. |
| **ATOM_GDN_STATE_MIDSTEP** | bool | 0 (false) | Set to `1` to take Qwen3-Next / Qwen3.5 GDN state checkpoints inside a prefill forward, from the chunk kernel's per-64-token states, instead of cutting the prefill into several forwards that end on them. PAGE checkpoint path only. Checkpoint positions move back onto the forward's 64-token grid, so a resume recomputes up to 48 more tokens. |

### LMCache offload tier

ATOM's own offload knobs are defined in `atom/utils/envs.py`; their behavior is
documented in `atom/kv_transfer/offload/README.md`. Where a
`kv_connector_extra_config` key also exists, it takes precedence over the env
var. `LMCACHE_EC_PIN_TIMEOUT_SEC` belongs to LMCache and is read only to
derive a bound.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_KV_OFFLOAD** | str | "" (off) | Enables KV offload without `--kv-transfer-config`, so a launcher that owns that flag for P/D can still add offload. `lmcache` selects the in-process `lmcache_offload` connector, `lmcache_mp` the standalone-server `lmcache_mp` connector (start `lmcache server` first), `mooncake_store` the `mooncake_store` connector, which puts and gets KV chunks straight to a Mooncake Store with no LMCache (start the Store's master and owners first; dense layout only). With a P/D connector in `--kv-transfer-config`, both run behind a `multi` connector. Setting it while `--kv-transfer-config` already names an offload connector is an error. |
| **ATOM_KV_OFFLOAD_EXTRA_CONFIG** | JSON object | "" | The offload connector's `kv_connector_extra_config`: `lmcache.<field>` overrides (e.g. `{"lmcache.chunk_size": 256}`), for `lmcache_mp` `lmcache.mp.*` options (e.g. `{"lmcache.mp.port": 5556}`), and for `mooncake_store` the `mooncake_store.*` settings (e.g. `{"mooncake_store.master": "10.0.0.1:26051", "mooncake_store.metadata": "http://10.0.0.1:26080/metadata"}`; all of them are listed in `atom/kv_transfer/offload/README.md`). Requires `ATOM_KV_OFFLOAD`. |
| **OFFLOAD_COPY_WORKERS** | int | 1 | Save executor threads per offload worker. Also scales the default `OFFLOAD_MAX_PENDING_SAVES`. |
| **OFFLOAD_LOAD_WORKERS** | int | 1 | Load executor threads per offload worker. DSV4's in-process path ignores it (its SLOT load path needs a serial load executor). |
| **OFFLOAD_MAX_PENDING_SAVES** | int | unset: `max(2, 2 × OFFLOAD_COPY_WORKERS)` for connectors, **2** for the scheduler's state tier | Bound on running-plus-queued saves. KV and state saves share it because both pin the same pool. A non-integer raises on the connector path and warns (using 2) on the state-tier path. Overridden by `max_pending_saves` in `kv_connector_extra_config`. While both are unset, `mooncake_store` leaves its saves unbounded (one per request in flight at most). |
| **OFFLOAD_MIN_LOAD_TOKENS** | int | 8192 | Smallest external-tier hit worth loading; shorter hits are recomputed. Negative values clamp to 0. |
| **OFFLOAD_MIN_SAVE_TOKENS** | int | 8192 | Shortest prefix native `lmcache_mp` stores, as an absolute boundary for normal and late saves. With the default equal to `OFFLOAD_MIN_LOAD_TOKENS`, a shorter prefix could never be loaded back. Other connectors ignore it. |
| **OFFLOAD_LOOKUP_MEMO_STEPS** | int | 32 | Scheduler steps a memoised tier-lookup answer is replayed before the tier is asked again. |
| **OFFLOAD_LOOKUP_RETRY_STEPS** | int | 32 | Scheduler steps a failed tier lookup suppresses the next attempt. |
| **OFFLOAD_ASYNC_LOOKUP_DEPTH** | int | 16 | `lmcache_mp` sends tier lookups for the head of the waiting queue before admission, as non-blocking requests, and admission consumes the answer; a request whose answer has not arrived is passed over (in arrival order) instead of blocking the scheduler. This is how many waiting requests per scheduling pass get a prefetched lookup. |
| **OFFLOAD_LOOKUP_DEFER_S** | float | 2.0 | How long `lmcache_mp` admission may pass over a request whose prefetched lookup is still in flight before it waits for the answer (finite, ≥ 0). |
| **OFFLOAD_SKIP_LOOKUP_ON_HBM_HIT** | bool | 1 (true) | On by default; set to `0` to always wait for the lookup. Admit a prefill without waiting for its external-tier lookup when the lookup cannot change what is loaded: the HBM prefix leaves fewer than `OFFLOAD_MIN_LOAD_TOKENS` tokens uncached, so every load path would refuse the tier hit. Requires an asynchronous lookup (LMCache MP) and is not applied with joint KV/state offload. The lookup is still sent, because it refreshes the prefix's L1 LRU recency and opens the MP session whose end touches the request's chunks; its answer is not used as a load hit. The save floor starts at the HBM prefix and is lowered to the tier's contiguous hit when the answer arrives (or to 0 if it never does), so chunks the tier evicted are stored again. Read locks are released when the answer arrives; the MP session ends after that, and an unanswered lookup is given up after 120 s, leaving its locks to the read TTL. |
| **OFFLOAD_PROFILE** | bool | 0 | Emit `[OFFLOAD-SAVE-PROF]` / `[OFFLOAD-LOAD-PROF]` per-transfer records, and once a minute an `[OFFLOAD-PERF]` line per scheduler with lookup/save/load-skip counters and per-step timings. An empty value reads as off. |
| **OFFLOAD_SINGLE_STREAM** | bool | 0 | Experimental: run the staging pack and copy legs on one stream. |
| **OFFLOAD_GPU_STAGING_CHUNKS** | int | derived from KV geometry | GPU staging buffer size in LMCache chunks (≥ 1). |
| **OFFLOAD_GPU_STAGING_MAX_BYTES** | int | unset | Upper bound on the GPU staging buffer in bytes; must hold at least one chunk. Unset, `mooncake_store` bounds it at 48 MiB, which it holds back from the KV budget per save and load thread. |
| **OFFLOAD_RELEASE_GPU_STAGING_AFTER_TRANSFER** | bool | 0 | Free the GPU staging buffer after each transfer instead of keeping it. |
| **OFFLOAD_SLOT_STAGING_SLOTS** | int | 1 | DSV4 in-process SLOT sidecar staging rows. Overridden by `slot_sidecar_staging_slots`. |
| **OFFLOAD_COMMITTED_SIDECAR_CAPACITY** | int | 65536 | DSV4 in-process committed SLOT sidecar index capacity. Overridden by `committed_sidecar_index_capacity`. |
| **OFFLOAD_PUBLICATION_TIMEOUT_S** | float | 5.0 | DSV4 in-process wait for a saved SLOT sidecar to become visible (finite, ≥ 0). |
| **OFFLOAD_PUBLICATION_POLL_INTERVAL_S** | float | 0.01 | Poll period of that wait (finite, > 0). |
| **LMCACHE_MP_TRANSFER_MODE** | str | auto | `lmcache_mp` transfer mode: `auto` or `lmcache_driven` (`engine_driven` is rejected). Overridden by `lmcache.mp.mp_transfer_mode`. |
| **LMCACHE_EC_PIN_TIMEOUT_SEC** | float | LMCache's own (300) | LMCache's source-pin timeout. ATOM reads it only to derive the engine's save-abandon window (`pin + 30s`), so the two stay ordered: a lost store report is reclaimed only after LMCache would already have force-unpinned its source. Non-positive disables ATOM's reclamation. ATOM sets no default of its own and, when unset, assumes LMCache's. |

## KV cache events

The scheduler can publish prefix-cache changes (`BlockStored`, `BlockRemoved`,
`AllBlocksCleared`, and `BlockStored(medium=REMOTE)` for KV received from a
PD producer) over ZMQ so external routers and cache managers can mirror what
each engine holds. Every batch carries a monotonic 8-byte sequence number;
a consumer that sees a gap can ask for the missed batches over the optional
replay socket. Events are advisory and never stall inference: the in-process
queue drops the oldest batch when full. These variables are the only way to
configure the feature today: they build `KVEventsConfig` (see `atom/config.py`)
and there is no CLI flag.

Endpoint rules: each publisher binds its PUB and replay endpoints offset by its
data-parallel rank (see `ATOM_KV_EVENTS_ENDPOINT`), so with `tcp://` the two
configured ports must be at least `data_parallel_size` apart or rank N's PUB
lands on rank N-1's replay port. Under pipeline parallelism only the head stage
publishes; downstream stages bind nothing. Prefill/decode disaggregation runs
separate engine processes, and only decode publishes; if other engines share
the host, give each its own endpoints.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_KV_EVENTS_ENABLE** | bool | 0 (false) | Set to `1` to publish KV cache events. |
| **ATOM_KV_EVENTS_PUBLISHER** | str | `zmq` | `zmq` or `null` (accepts events and discards them). |
| **ATOM_KV_EVENTS_ENDPOINT** | str | `tcp://127.0.0.1:5557` | ZMQ PUB bind address. Under data parallelism every rank binds its own socket: `tcp://` endpoints get the DP rank added to the port, `ipc://`/`inproc://` endpoints get a `_dp<rank>` suffix. Rank 0 uses the configured value unchanged. |
| **ATOM_KV_EVENTS_TOPIC** | str | `""` | Subscription topic prefix sent as the first frame of every message. |
| **ATOM_KV_EVENTS_HWM** | int | 0 | ZMQ high-water mark on the PUB socket (0 = unlimited). |
| **ATOM_KV_EVENTS_BUFFER_STEPS** | int | 10000 | Depth of the in-process queue between the scheduler and the sender thread. When full, the oldest batch is dropped and counted in the publisher's `dropped` stat; the dropped batch still consumes a sequence number, so subscribers see the loss as a gap. |
| **ATOM_KV_EVENTS_REPLAY_ENDPOINT** | str | `""` | ZMQ ROUTER bind address for replay requests. Empty disables replay (PUB-only). A consumer sends an 8-byte big-endian start sequence and receives every retained batch with `seq >= start`, followed by a `REPLAY_DONE` terminal frame carrying the retained `[oldest, latest]` window. Offset per DP rank the same way as the PUB endpoint. Replay is serviced on the sender thread with non-blocking sends: a client that stops reading has its replay abandoned (counted in `replay_aborted`) rather than stalling live publication. |
| **ATOM_KV_EVENTS_REPLAY_BUFFER_STEPS** | int | 10000 | Number of most recently *sent* batches retained for replay. Independent of `ATOM_KV_EVENTS_BUFFER_STEPS`; each entry holds an encoded payload including token ids, so size it against the event rate and memory budget. Must be >= 1 when replay is enabled. |

## Shutdown

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_SHUTDOWN_TIMEOUT_S** | float | 5 | Seconds that shutdown waits for child processes to exit on their own before terminating them (SIGTERM, then SIGKILL): the server process waits this long for its EngineCores, and each EngineCore for its ModelRunner workers. All the children of one process share the same deadline. Must be finite and positive. Raise it when workers need longer to finish on exit, for example to write a profiler's output (`rocprofv3` writing a thread trace can take minutes). Set it before starting the server; every process reads it. |

## Profiling & debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_METRICS_UPDATE_INTERVAL_S** | float | 1.0 | Shared interval in seconds for ordinary/DP/PP engine metrics pushes and API snapshot refresh. Must be finite and positive; read when each loop starts, so set it before starting every service process. Prometheus scraping is configured independently. Does not cache rendered `/metrics` responses or change when histogram observations are recorded. |
| **ATOM_ENABLE_METRICS_DEVICE_TIMER** | bool | 0 (false) | Set to `1` before starting the service to collect GPU forward duration and cumulative request prefill GPU time. Uses CUDA/HIP events, a reusable pool capped at 256 pending pairs, and FIFO polling that stops at the first incomplete event. Adds event recording and query overhead; disabled services emit no GPU timing samples. Agentic dashboard CI explicitly enables it. |
| **ATOM_TORCH_PROFILER_DIR** | str | — | When set, enables PyTorch profiler and writes traces to this directory. Create subdirectories per rank (e.g., `rank_0`, `dp0_tp0`). |
| **ATOM_PROFILER_MORE** | bool | 0 (false) | When `ATOM_TORCH_PROFILER_DIR` is set and this is `1`, enables detailed profiling: `record_shapes`, `with_stack`, and `profile_memory`. Applies to both the run-phase profiler and the CUDA-graph capture profiler. Each option can be overridden individually by the per-option flags below. |
| **ATOM_PROFILER_RECORD_SHAPES** | bool | `ATOM_PROFILER_MORE` | Set to `1`/`0` to enable/disable `record_shapes`. Takes precedence over `ATOM_PROFILER_MORE`; unset or empty falls back to it. |
| **ATOM_PROFILER_WITH_STACK** | bool | `ATOM_PROFILER_MORE` | Set to `1`/`0` to enable/disable `with_stack` (Python stack capture). Takes precedence over `ATOM_PROFILER_MORE`; unset or empty falls back to it. |
| **ATOM_PROFILER_PROFILE_MEMORY** | bool | `ATOM_PROFILER_MORE` | Set to `1`/`0` to enable/disable `profile_memory`. Takes precedence over `ATOM_PROFILER_MORE`; unset or empty falls back to it. |
| **ATOM_ENABLE_DETAILED_ANNOTATION** | bool | 0 (false) | When profiling is active, appends detailed attention aggregates to the `prefill[]`/`decode[]` trace labels: `sqsq` (Σ N_Q²), `sqsk` (Σ N_Q·N_KV), and `sk` (Σ N_KV), where N_Q is the scheduled query tokens and N_KV the KV length per request. Used to estimate attention FLOPs for downstream roofline analysis. |
| **ATOM_LOG_MORE** | bool | 0 (false) | If set to `1`, use verbose logging format (includes process name, PID, path, line number, function name). |

## Garbage collection

CPython's generation-2 pass is stop-the-world and walks every tracked
container, so its cost tracks the live heap — which in a serving process is
almost entirely startup state (model, compiled graph, tokenizer, KV block
pool) that is never garbage. Measured on DeepSeek-V4-Flash-DSpark tp1: 242.8 ms
in the EngineCore, up to 596 ms in a ModelRunner worker, while reclaiming zero
objects once startup was done. See `atom/utils/gc_utils.py`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_GC_FREEZE** | bool | 1 (true) | Move the startup heap into CPython's permanent generation once warmup is done, so collections stop scanning it. Applied in every process that outlives startup — the API server, the atomesh frontend, every EngineCore and every ModelRunner worker; undone on engine shutdown so an in-process teardown does not leak. Set `0` to keep the pre-freeze behaviour. |
| **ATOM_GC_DEBUG** | bool | 0 (false) | Log every collection: generation, duration, objects reclaimed, objects tracked. Costly — counting the tracked set on every pass added ~90s of startup on a V4-Flash tp1 — but the only way to see these pauses, since a stall in the EngineCore idles the workers with no event in their torch trace. |
| **ATOM_GC_THRESHOLD** | csv int | "" (= CPython default 700,10,10) | `t0,t1,t2` for `gc.set_threshold()`. Thresholds are per-interpreter, so each process reads it independently; anything that is not three integers is logged and ignored, applying nothing. Raising these does not make a pass cheaper, it makes passes rarer — the same total scan lands in fewer, longer stop-the-world pauses, which is a trade against tail latency and not measured here. It is also **not uniform across processes**: at concurrency 4096 the API server's collector ran 13,956 times in twenty minutes over a set that grew to 688,646 objects and reclaimed **zero**, while each ModelRunner worker reclaimed thousands per pass, where spacing collections out defers real work. Read `atom:gc_collected_total` for the process you mean to tune before setting this — and note that only the API server exports it, so a worker has to be read with `ATOM_GC_DEBUG=1`. |

Raising a process's thresholds is free only while its collector keeps finding
nothing to free, which is a property of that one process, so it is exported
rather than assumed. All three are wired into the API server only; the engine
and worker processes serve no `/metrics`, so `ATOM_GC_DEBUG=1` is what reads
them there.

- **`atom:gc_collected_total`** (`/metrics`, per generation — `prometheus_client`
  appends the `_total`) is the invariant as a series. Flat after startup is the
  expected shape; a rising line means the process has started building reference
  cycles, and spacing its collections out would defer real work into a growing
  heap. `atom:gc_collections_total`, `atom:gc_uncollectable_total` and
  `atom:gc_threshold` sit beside it for context. All are O(1) reads taken at
  scrape time, which is a bound and not a preference: `/metrics` renders on the
  loop that delivers every stream. The frozen count is deliberately **not**
  here — `gc.get_freeze_count()` walks the permanent generation (11.9 ms at
  430k frozen, the cost freezing exists to remove) for a number that changes
  twice in a process's life. The startup log has it, and so does the census.
- **`reclaim_watch`** logs one warning — once, not per check — if that line
  ever rises, because a counter nobody looks at is not a safeguard. It sees
  only what a collection reclaimed, so cyclic garbage that reaches gen-2 before
  it dies is invisible to it where gen-2 passes are rare; the gen-2 size in
  `/debug/gc_census` is what shows that.
- **`GET /debug/gc_census`** breaks the scanned set down by type and by owning
  library. Unlike the metrics it walks every tracked object (~1s at a million),
  so it is asked for, never scraped, and it runs in a worker thread rather than
  on the loop that delivers the streams. It reports counts only: naming what a
  container *holds* would mean serialising the keys of parsed request bodies
  into an unauthenticated response. `top` and `types_per_owner` bound the two
  breakdowns.

## Incremental detokenizer

Streaming decodes each delta from two `tokenizer.decode` calls that share a
window start, so subtracting one from the other isolates the new text without
emitting a half-formed UTF-8 character. These two settings govern the shortcut
that removes one of those calls. Nothing here is related to the KV prefix
cache, which is what "cache" means everywhere else in this repository.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DETOKENIZER_DELTA_REUSE** | auto \| on \| off | auto | Whether the incremental detokenizer may reuse the delta it last emitted in place of one of its two `tokenizer.decode` calls per update. The two decodes share a window start so that subtracting one from the other isolates the new text; the first one only ever yields a length, and its span is what the previous call already emitted. Measured on DeepSeek-V4-Pro: **1.74x** at one token per update, **1.40x** at sixty-four. Holds only where decoding a token span does not depend on where the window started — true for the byte-level BPE tokenizers measured (DeepSeek, Qwen3.5, GLM-5.2), not for a SentencePiece-style decoder that adds or strips a leading space by position. `auto` therefore verifies it at startup (~15 ms) instead of assuming it from a class name, and leaves it off on any failure. Case-insensitive; an unrecognised value is logged and leaves reuse off, since off is what someone spelling this setting is reaching for. Unrelated to the KV prefix cache. |
| **ATOM_DETOKENIZER_AUDIT_EVERY** | int | 1000 | How often a reused delta is checked against a real decode once reuse is on. A mismatch answers that call from the decode, then turns reuse off for the process and logs once. Only audited calls are checked, so at the default up to 999 deltas can ship between a tokenizer starting to disagree and the audit that notices it; `1` checks every update, which is what the startup probe runs at. An unusable value — 0, negative, or not a number — is logged and the default used; this is not how reuse is turned off, `ATOM_DETOKENIZER_DELTA_REUSE=off` is. |

### Debug dump (`atom.utils.debug_helper`)

Env-gated dump / compare / monkey-patch primitives for forward bisect &
batch invariance investigation. All entries are **no-op when their
controlling `*_DIR` is unset**, so they are safe to leave wired into
production paths. See `.claude/skills/dump-bisect-debug.md` for the
methodology and `atom/utils/debug_helper/` for the implementation.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_FWD_DUMP_DIR** | str | — | Enables `install_block_forward_hooks`. Per-Block hidden state is saved to `{DIR}/layer{LL}_{Cls}_rank{R}[_call{NNN}].pt`. |
| **ATOM_FWD_DUMP_LAYERS** | csv int | "" (= all) | Comma-separated layer ids to dump (e.g. `0,5,15,30`). Empty string means dump every layer. |
| **ATOM_FWD_DUMP_BLOCK_CLASS** | csv str | `Block` | Module class names to hook. Multiple values supported (e.g. `Block,DeepseekV4Attention,MoE,Compressor,Indexer`) for sub-stage bisect. Override per model. |
| **ATOM_FWD_DUMP_LAYER_ATTR** | str | `layer_id` | Attribute name on the block carrying its index. Some non-DeepSeek models use `layer_idx`. |
| **ATOM_FWD_DUMP_ONE_SHOT** | bool | 1 (true) | When `1`, only the first call per layer is dumped (typical: warmup). Set to `0` to enumerate every call (`_call000.pt`, `_call001.pt`, …) — required when bisecting per-seq dispatch loops. |
| **ATOM_WEIGHT_DUMP_DIR** | str | — | Enables `maybe_dump_weights_and_exit`. Per-rank params + buffers for selected layers dumped to `{DIR}/weight_rank{R}_layer{L}.pt`. Skips `.experts.*` (FP4 packed). |
| **ATOM_WEIGHT_DUMP_LAYERS** | csv int | `0` | Comma-separated layer ids to dump weights for. |
| **ATOM_WEIGHT_DUMP_EXIT** | bool | 1 (true) | When `1` (default), call `sys.exit(0)` after dumping. Set to `0` to continue inference after dump. |
| **ATOM_DEBUG_TOPK** | int | 0 | Set to `K > 0` to log top-K logits per row from `Sampler.forward` via `maybe_log_topk()`. Only rank 0 writes. |
| **ATOM_DEBUG_TOPK_PATH** | str | — | Optional output file for top-K logs. Writes to stderr if unset. |

CLI for comparing dumps:

```bash
python -m atom.utils.debug_helper.compare slot-invariance --dir DIR --n-slots 4
python -m atom.utils.debug_helper.compare ref-vs-target  --dir DIR
python -m atom.utils.debug_helper.compare layer-bisect   --dir DIR --threshold 0.99
python -m atom.utils.debug_helper.compare schema --a A.pt --b B.pt
```

## Benchmarks (optional)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **OPENAI_API_KEY** | str | — | API key for OpenAI-compatible benchmark requests. |
| **VLLM_USE_MODELSCOPE** | bool | false | If set to `true`, use ModelScope for model downloads in benchmarks. |
| **SAVE_TO_PYTORCH_BENCHMARK_FORMAT** | bool | false | If set, save benchmark results in PyTorch benchmark format. |

## Internal / Set by ATOM

The following variables are set internally by ATOM; users typically do not need to configure them:

| Variable | Description |
|----------|-------------|
| **AITER_QUICK_REDUCE_QUANTIZATION** | Set to `INT4` for Llama models with bf16/fp16. |
| **TORCHINDUCTOR_CACHE_DIR** | Set by compiler interface for inductor cache. |
| **TRITON_CACHE_DIR** | Set by compiler interface for Triton cache. |

## Reference

Environment variables are defined and accessed via `atom.utils.envs`:

```python
from atom.utils import envs

# Example: check data parallel size
dp_size = envs.ATOM_DP_SIZE
```

See `atom/utils/envs.py` for the full list of lazy-evaluated environment variables.

## Mooncake PD matched rails

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_MOONCAKE_MATCHED_RAILS** | str | "" | Use `auto` to discover ACTIVE local HCAs in the primary HCA's numbered name family, or set a comma-separated allowlist. Enables a lazy single-HCA engine pool on P, selected by D's advertised HCA name. Requires RDMA, a single primary HCA, and corresponding same-name rails across hosts. Unset preserves existing behavior. |

See [Mooncake matched rails](mooncake_matched_rails.md) for independent P/D
rank configuration, deployment requirements, and registration lifetime.

## Mooncake PD MLA staging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_PD_MLA_STAGING** | bool | 1 | Prefill (`kv_producer`, plain `mooncake` connector) only. When the decode side runs DCP, gather the MLA tokens each decode rank owns into a GPU staging slot laid out as its destination pages, then RDMA one descriptor per run of adjacent destination pages instead of one per 576-byte token. Destination bytes are identical. The pool holds one 8 MiB slot per send worker (`num_worker_threads`, default 16), capped at 256 MiB per producer GPU, and is held back from the KV cache budget. A prefill learns the decode's DCP size only per request, so every Mooncake producer of a plain MLA model (not DeepSeek-V4 or an MLA hybrid such as Kimi) without producer DCP reserves the pool while this is on; set 0 where no decode runs DCP. 0 restores the per-token path. |
| **ATOM_PD_FAILED_WRITE_HOLD_S** | float | 60 | Mooncake P/D. Mooncake gives up on a sync RDMA write at `MC_TRANSFER_TIMEOUT` without cancelling the work it posted, so a failed write may still land afterwards; a prefill therefore never retries one. A decode holds a failed receive's KV pages (and its staging row) this many seconds before reporting the failure; the scheduler then aborts the request and frees its pages, which no late write can reach any more. A prefill keeps a landing slot whose write failed out of use for as long. 60 s covers a write Mooncake keeps re-posting while the network is down (each of its 9 tries waits at most a 5 s connect); behind a decode whose handshake hangs it can take about 10 minutes. 0 releases at once. MoRIIO P/D holds a failed receive's blocks the same way; a MoRIIO decode in write mode refuses 0, since the hold is also what lets a push started just before its deadline finish. |

## Mooncake PD MLA landing

Set on the decode side; the prefill side follows what each write request offers and needs MLA staging enabled.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_PD_MLA_LANDING** | bool | 1 | Decode (`kv_consumer`, plain `mooncake` connector, DCP > 1, a plain MLA model) only. Allocate a GPU landing pool per rank and give prefill stages partitions of it. A stage writes a rank's MLA rows packed in rank order into a landing slot with one RDMA descriptor per slot (instead of one per destination page); the decode rank scatters the slot into its paged KV cache on a side stream and returns the slot. A request completes only after its slots are scattered. Destination bytes are identical to the staged path. The pool is 256 MiB of 8 MiB slots per decode rank; it and the scatter's segment table (about 12 MiB at block size 1, less for larger blocks) are held back from the KV cache budget. Only a non-DCP MLA prefill with `ATOM_PD_MLA_STAGING=1` lands, so set 0 where none does (for example symmetric DCP P/D) to give the pool back to the KV cache. Partitions go first come, first served: a prefill stage endpoint gets `32 // its PP size` slots on first contact and keeps them while the decode runs, so with several prefill instances per decode rank a later one may get fewer slots, or none and keep the staged path. A stage that reports it cannot land returns its partition at its first write-done and is not offered slots again. A prefill sends a transfer needing fewer than 2 slots, or the rest of one after waiting 10 ms for a free slot, through the staged per-page path. 0 keeps the staged per-page writes. |

## MoRIIO PD

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_MORIIO_MODE** | str | "" | Decode (`kv_consumer`) only; a prefill serves both modes. `read` pulls the KV with RDMA reads; `write` sends the prefill the decode's block ids and has it push the KV with RDMA writes. Unset uses `moriio_mode` from `--kv-transfer-config`, else `read`. |
| **ATOM_MORIIO_HANDSHAKE_TIMEOUT_S** | float | 300 | How long a request waits for the handshake with its peer (the prefill, in read mode; the decode, for a prefill in write mode) before it fails. A single handshake gives up after at most 30 s without a reply, so a peer that is gone cannot hold up the others, which share one thread. |
| **ATOM_MORIIO_WRITE_TIMEOUT_S** | float | 600 | Write mode. A prefill starts no push this many seconds after the decode asked for it and reports it failed instead. A decode that never hears back fails the request `ATOM_PD_FAILED_WRITE_HOLD_S` after that deadline and holds its blocks until then, so a push started just in time can finish. The deadline is wall-clock, so prefill and decode clocks must agree to well within that hold. |
| **ATOM_MORIIO_VERIFY** | bool | 0 | Debugging. Digest each layer's destination blocks before and after every receive and log, per request, the layers the transfer left unchanged. Costs a device sync per request, and allocates an int64 weight per destination byte (about 150 MB for a 32k-token MLA fp8 prompt) on the worker loop, so it is not meant for runs under load. |

A failed receive, in either mode, holds its blocks `ATOM_PD_FAILED_WRITE_HOLD_S` before the failure is reported: MoRI marks a batch failed on its first failed work request while the rest of it may still be in flight.

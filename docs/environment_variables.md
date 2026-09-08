# ATOM Environment Variables

This document describes the environment variables used in the ATOM project.

## Data parallelism

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DP_RANK** | int | 0 | The rank ID for the current process in data parallelism. |
| **ATOM_DP_RANK_LOCAL** | int | 0 | The local rank ID for the current process (used in SPMD mode). |
| **ATOM_DP_SIZE** | int | 1 | Total number of data parallel ranks. |
| **ATOM_DP_MASTER_IP** | str | 127.0.0.1 | Master IP address for DP ranks coordination. |
| **ATOM_DP_MASTER_PORT** | int | 29500 | Master port for DP ranks coordination. |
| **ATOM_DP_LB_REQ_EQUIV** | int | 512 | Token-equivalent decode pressure assigned to each in-flight request by `least_tokens` routing. |
| **ATOM_DP_SESSION_AFFINITY** | bool | false | Load-place each new session, then keep later turns on the same prefix-cache owner. Reads `X-Dynamo-Session-ID`, falling back to `X-Correlation-ID`. |

## Prefill delayer (DP attention)

Prefill **coalescer** for DP-attention + EP-MoE serving. Holds back prefill
admission until the accumulated prefill (fresh waiting tokens + resumable
partials' remaining tokens) fills a worthwhile forward, so fragmented
short-input prefills / small partial tail chunks batch into one forward instead
of firing many tiny ones. Releases when the fill target is reached, when a
must-fire bound trips (no decode to hide behind, KV pressure/starvation, TTFT
deadline, partial deadline), or when the queue stops growing. Preserves
cross-rank phase alignment (releases only when every rank is prefill-ready,
unless a bound forces it). All timing is tick-based (deterministic across ranks —
no wall-clock skew). See `atom/model_engine/prefill_delayer.py`. Active only when
`data_parallel_size > 1`.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_PREFILL_DELAYER** | bool | true | Master switch for the prefill coalescer. |
| **ATOM_PREFILL_DELAYER_TARGET_FILL** | float | 0.9 | Release once accumulated pending tokens reach `target_fill × max_num_batched_tokens` (averaged across prefillable ranks). In (0, 1]; higher = fewer, larger prefills at some TTFT cost. Clamped to (0, 1]. |
| **ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS** | int | 200 | Max consecutive scheduler ticks a held prefill waits before force-release. Values `< 1` clamped to 1. |
| **ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS** | int | 100 | Tighter bound for a held mid-chunked-prefill (it holds allocated KV). Values `< 1` clamped to 1. |
| **ATOM_PREFILL_DELAYER_STALL_TICKS** | int | 10 | After this many consecutive non-growing ticks, release (burst ended, more won't come). Values `< 1` clamped to 1. |
| **ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK** | float | 0.9 | At/above this KV usage a prefillable rank force-releases (can't accumulate a bigger batch anyway). |
| **ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK** | float\|"" | "" (None) | If set, a prefillable rank below this KV usage force-releases (GPU starving). |
| **ATOM_PREFILL_DELAYER_MAX_QUEUE_MS** | float\|"" | "" (None) | TTFT SLA guard: if any rank's oldest schedulable waiting prefill has queued (since arrival) ≥ this many ms, force-release regardless of the fill target. Measures true end-to-end wait (backlog + coalescer holds), unlike the tick-based TTFT bound which only caps one hold episode. Empty = disabled; set to your TTFT budget (a small value under heavy backlog fires every tick and defeats coalescing). |
| **ATOM_PREFILL_DECODE_INTERVAL** | int | 0 | After an executed prefill forward, protect this many scheduler passes for decode before admitting another prefill. `0` disables the interval. |
| **ATOM_PREFILL_DELAYER_DEBUG** | bool | false | Per-tick FIRE/HOLD debug logging. |
| **ATOM_PREFILL_DELAYER_LOG_EVERY** | int | 1000 | Emit aggregate stats (per-exit fire counts + hold rate) every N decisions (0 disables). |

## Model loading

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DISABLE_MMAP** | bool | false | If set to `true`, disable memory-mapped file loading for model weights. Useful in containerized environments where mmap may cause issues. |
| **ATOM_LOADER_NUM_THREADS** | int | 16 | Worker threads for weight loading. `>1` (default `16`) enables the batched parallel loader (routed expert weights staged in a CPU buffer, flushed with a single H2D copy when every routed expert of that parameter has arrived) with that many threads; set to `1` to fall back to the original sequential per-expert path. Raise on high-core hosts if loading is CPU-bound. |
| **ATOM_LOADER_STRICT_COVERAGE** | bool | `true` | Fail loading when a fused MoE parameter does not receive every routed expert from the checkpoint. Set to `false` to downgrade to a warning and load anyway, leaving those expert slots at their init values — useful when bringing up a checkpoint known to be partial, misleading otherwise (the symptom is an accuracy drop much later). |
| **ATOM_LOADER_PREFETCH** | bool | `true` | Warm the page cache by reading this rank's share of the checkpoint sequentially on a background thread, instead of leaving it to demand faults through the mmap. The fault pattern sustains ~3.2 GB/s on a local NVMe that a single sequential reader drives at 6.06 GB/s, so this is an access-pattern fix, not a queue-depth one. Measured on DeepSeek-R1 MXFP4 (350 GiB, TP=4): cold load 154s → 69s. Set to `false` to restore demand faulting. Has no effect when `ATOM_DISABLE_MMAP=true`. |
| **ATOM_LOADER_PREFETCH_THREADS** | int | 4 | Concurrent sequential readers used by the prefetcher. The device saturates at ~2 streams, so raising this mostly adds contention with the loader; `0` is clamped to `1` (use `ATOM_LOADER_PREFETCH=false` to switch prefetching off). |
| **ATOM_LOADER_PREFETCH_BLOCK_MB** | int | 16 | Read block size for the prefetcher, in MiB. |
| **ATOM_LOADER_FADVISE** | bool | `false` | Issue `posix_fadvise(SEQUENTIAL\|WILLNEED)` per shard before reading it. Off by default and ignored while `ATOM_LOADER_PREFETCH` is on: `WILLNEED` is a hint the kernel drops for most of a 350 GiB checkpoint, and running both makes the kernel read ahead over random-ish ranges while the prefetcher streams the same files, so the two compete for the device. Only useful with prefetching disabled. |
| **ATOM_ONLINE_QUANT_STREAMING** | bool | `false` | Opt in to quantizing eligible online-quant modules as soon as their checkpoint weights are complete, then release source storage to reduce load-time peak memory. Only active with a valid online quantization config. See the [streaming online quantization guide](./online_quantization_streaming_guide.md). |
| **ATOM_ONLINE_QUANT_STREAMING_HOST_STAGING** | bool | `true` | Assemble streamed module weights in CPU storage before one H2D transfer. Keeps the checkpoint walk parallel; disabling it buffers loader calls and forces the checkpoint walk to one thread. |
| **ATOM_ONLINE_QUANT_STREAMING_THREADS** | int | `4` | Tail workers for H2D, per-module quantization, and source release. More workers increase overlap and in-flight memory; `0` runs finalization inline. |

## Plugin mode

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DISABLE_VLLM_PLUGIN** | bool | 0 (false) | If set to `1`, disable the vLLM plugin registration entirely. |

## Kernel / backend selection

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_USE_TRITON_GEMM** | bool | 0 (false) | If set to `1`, use AITER Triton FP4 weight preshuffled GEMM. Otherwise use AITER ASM FP4 weight preshuffled GEMM. |
| **ATOM_USE_FP4_NON_SHUFFLE_TRITON_GEMM** | bool | 0 (false) | If set to `1`, use AITER Triton FP4 GEMM with non-shuffled weights. Takes precedence over the FP4 preshuffled GEMM path selected by `ATOM_USE_TRITON_GEMM`. |
| **ATOM_USE_TRITON_MXFP4_BMM** | bool | 0 (false) | If set to `1`, use FP4 BMM in MLA attention module. |

### GLM-5.3

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_GLM5_KPOOL** | bool | 1 (true) | Enable the pooled sparse indexer. Setting `0` is an exact token-granular A/B only at or below `index_topk`; longer requests are refused. |
| **ATOM_GLM5_FORCE_DENSE_MLA** | bool | 0 (false) | Disable sparse MLA for short-context bring-up comparisons. |
| **ATOM_GLM5_DISABLE_FUSED_MHC** | bool | 0 (false) | Force the PyTorch mHC reference path instead of AITER's fused kernels. |

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
| **ATOM_MORI_FP4_DISPATCH** | bool | 0 (false) | If set to `1`, quantize activations to packed FP4 (E2M1, `per_1x32`) before the MoE all2all instead of sending bf16 — a quarter of the bytes on the dispatch wire — which selects `EpDispatchIntraNodeKernel_fp4`. MoRI picks its dispatch kernel from the dtype of the tensor handed to `dispatch()` but sizes its staging buffers from the config built at init, so this also switches `scale_dim` to `hidden_dim/32` and the scale type to e8m0. All three are resolved together by `mori_prepare_finalize.resolve_mori_dispatch()`; never set one without the others, as a mismatch strides the staging scale buffer wrong and faults on the first real batch instead of erroring cleanly. |
| **ATOM_MORI_COMBINE_QUANT** | str | `none` | Combine-side codec passed into the MoRI config. `none` returns bf16; `fp8_blockwise` selects `EpCombineIntraNodeKernel_*_fp8bwq_*`; MoRI also accepts `fp8_direct_cast`. |

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
| **ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD** | int | 1024 | Upper bound on MoE token count (`num_tokens` in the MoE forward) for using the dual-stream path: shared experts on a secondary CUDA stream while routed experts run on the default stream. If `num_tokens` exceeds this value, that forward uses single-stream MoE instead. Set to `0` to disable dual-stream setup entirely (no alt stream, no `maybe_dual_stream_forward` registration). |
| **ATOM_DUAL_STREAM_PIECEWISE** | bool | 0 | Opt-in: allow a PIECEWISE-captured graph piece to hold the MoE dual-stream fork/join (shared experts on `alt_stream` overlapping routed experts). Capture is not the obstacle — `set_forward_context` runs inside `graph_capture()`, so the main stream the fork waits on is the stream capture runs on — and vLLM and SGLang both keep this overlap on inside piecewise graphs (SGLang runs dual-stream *only* inside a graph). Measured on V4-Pro-DSpark under `AF_PIECEWISE`: the fork survives capture, hides 77.5% of shared-expert time, and leaves GSM8K and MTP acceptance unmoved. Off by default only because no throughput win has been demonstrated, and because each replayed piece then carries its own driver-allocated stream (368 vs 2 distinct streams on a tp8 rank trace). The dispatcher is shared, so this moves V2/V3.2/K3 as well. Eager (`NONE`) and whole-model `FULL` are unaffected. |

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

### Qwen3 style

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION** | bool | 0 (false) | If set to `1`, fuse QK norm, RoPE, and cache quantization into one kernel for Qwen3 dense and MoE models. |

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
reachable shape one set rather than two lists that drift. The switch below decides whether that warm also *records*.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DRAFT_CUDAGRAPH** | bool | 1 (true) | Capture each declared draft pass into a per-`capture_sizes` CUDAGraph as it is warmed, so a draft pass replays instead of relaunching every kernel. `0` keeps the warmup (and therefore the JIT saving) but drafts eagerly. Only passes that declare a graph are captured — the separate-draft Kimi-K3 path declares none, so this is inert there. EPLB no longer declines the padding: the target pads on every cudagraph decode step and its rows reach the same expert-load recorder, so declining on the draft protected nothing. A DP-sync dummy DOES replay, in lockstep with the ranks holding work — `is_dummy_run` is per-rank, so gating on it splits one DP group across two collectives. Measured on V4-Flash-DSpark tp1: GSM8K 0.9527 / acceptance 65.25% captured against 0.9497 / 65.21% eager, i.e. indistinguishable; on tp4 with the LM head inside the capture, draft kernel launches went 30 → 0 per pass and draft wall time 915.8 → 118.9 µs. Read per pass at warmup time, so set it before the server starts. Grep a trace for a trailing ` graph` in a `propose_*` label to confirm which passes replayed. |

### DSpark drafting

The Kimi-K3 DSpark draft writes the target's context rows into its own paged MLA
cache once per draft layer per drafting step; the switch below shortens that
path. The first write of each process logs which path it took, and logs again if
that ever changes, so a fusion left inert by an unrecognised layout says so.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_DSPARK_FUSED_CTX_KV** | bool | 1 (true) | Write the context rows with one Triton kernel (RMSNorm + RoPE + concat + paged store) instead of four launches plus a throwaway `empty_like` for the RoPE's query side. Falls back per call when the cache layout or the RoPE is not the plain one the kernel understands (seg / shuffled-KV layouts keep their own write kernels), and until the RoPE's cos/sin cache has reached the device. Measured on Kimi-K3 (MI355X, TP8, fp8 KV): one 4.65 µs kernel replaces a 14 µs three-kernel chain, saving ~39 µs per drafting step at B=1 and ~36 µs at B=64. Set to `0` to force the per-op chain; that chain is the fallback above rather than debug code, so it stays reachable either way (it runs the first write of every layer). |

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

### LMCache offload tier

Two knobs that govern the LMCache CPU/NVMe offload connector are read directly
via `os.environ` rather than through `atom.utils.envs`, because ATOM does not
own either default: one belongs to the LMCache library, the other to the
offload connector itself (defined in
`atom/kv_transfer/offload/_offload_common.py` and documented in full in
`atom/kv_transfer/offload/README.md`). They are listed here so they are
discoverable from the central env reference despite bypassing the registry.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **LMCACHE_EC_PIN_TIMEOUT_SEC** | float | LMCache's own (300) | LMCache's source-pin timeout. ATOM reads it only to derive the engine's save-abandon window (`pin + 30s`), so the two stay ordered — a lost store report is reclaimed only after LMCache would already have force-unpinned its source. Non-positive disables ATOM's reclamation. ATOM sets no default of its own; when unset it assumes LMCache's. |
| **OFFLOAD_MAX_PENDING_SAVES** | int | **2**, flat, for the engine-side/state-tier reader (`scheduler.py`); `max(2, 2 × OFFLOAD_COPY_WORKERS)` for the KV-leg reader (`_offload_common.py`) | Bound on total in-flight offload transfers (running + queued) held before a SLOT snapshot or executor submission. A KV save and a state store both pin bytes out of the same pool while they run, so the KV leg and the K3 state tier share this one number rather than each carrying its own. Two readers compute it, though: the KV leg's canonical `_offload_common.max_pending_saves` derives the shown default from `OFFLOAD_COPY_WORKERS` and **raises** on an unparseable value, while the scheduler's state-tier reader (`_offload_max_pending_saves`) has a simpler fallback — a flat default of **2** (no `OFFLOAD_COPY_WORKERS` scaling) that **warns and uses 2** on an unparseable value rather than raising. Set the env to an explicit integer to pin both. |

## Profiling & debugging

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| **ATOM_TORCH_PROFILER_DIR** | str | — | When set, enables PyTorch profiler and writes traces to this directory. Create subdirectories per rank (e.g., `rank_0`, `dp0_tp0`). |
| **ATOM_PROFILER_MORE** | bool | 0 (false) | When `ATOM_TORCH_PROFILER_DIR` is set and this is `1`, enables detailed profiling: `record_shapes`, `with_stack`, and `profile_memory`. Applies to both the run-phase profiler and the CUDA-graph capture profiler. |
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
| **ATOM_GC_THRESHOLD** | csv int | "" (= CPython default 700,10,10) | `t0,t1,t2` for `gc.set_threshold()`. Thresholds are per-interpreter, so each process reads it independently. A fallback for `ATOM_GC_FREEZE=0`: this spaces collections out, freezing removes what one costs. |

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

# QSA index scorer: opt-in aiter in-tree op (Qwen3.8-Flash-Next)

The Qwen QSA indexer scores compressed key groups with ATOM's
`qsa_paged_mqa_logits` (`atom/model_ops/qwen4_exp/ops/qsa.py`, unchanged: the
frozen golden scorer, sha256 `c2e46e50…` LF / `aaa78f43…` CRLF). With
`--qsa-index-scorer-profile` the same calls go to aiter's **in-tree** op
`aiter.ops.qsa_index_scorer` (ROCm/aiter#7377, FlyDSL gfx1250, cross-repo
contract API 1.x). This is a call-site integration of a new coarse aiter op; it
does not use `aiter.plugin` (the out-of-tree provider interface) and has nothing
to do with `atom/plugin/` (ATOM's vLLM/SGLang/rtp-llm frontend adapter).

## Switch

| CLI | Config | Env (operator override) | Default |
|---|---|---|---|
| `--qsa-index-scorer-profile NAME\|PATH.json` | `Config.qsa_index_scorer.profile` | `ATOM_QSA_INDEX_SCORER_PROFILE` (read once by `EngineArgs`, only when the flag is absent) | `""` (OFF) |
| `--qsa-index-scorer-capture PATH` | `Config.qsa_index_scorer.capture` | — | `""` |

* **OFF** (default): `Qwen4ExpAttention.qsa_scorer is None`; neither
  `atom.model_ops.qwen4_exp.qsa_index_scorer` nor `aiter.ops.qsa_index_scorer`
  is imported (also not by `ModelRunner`); `_select_tokens` calls the stock
  `qsa_select_paged_tokens`; `Config.compute_hash()` is unchanged
  (`QSAIndexScorerConfig.hash_factors()` is empty while OFF).
* **ON**: `compute_hash()` includes `("qsa_index_scorer", profile)`. The profile
  is passed as `profile=` to every aiter call; ATOM never sets aiter's
  `AITER_QSA_INDEX_SCORER_*` variables. Do not set `QSA_*` kernel env vars (any
  non-profiled value makes every call `kernel_env_mismatch` → stock).
* **Capture** works with or without a profile (without one every call runs the
  stock scorer with reason `atom_capture_only`).

## Lifecycle (per worker process)

```
ModelRunner.__init__
  _setup_device_and_distributed()
  check_aiter_fused_ops(config)            atom/aiter_fused_ops.py (ON only)
    -> capability.init_qsa_index_scorer    once per process: import aiter op, API_VERSION 1.x,
                                           scorer signature, load_profile(profile); one startup line
  _build_and_load_model()
    Qwen4ExpAttention.__init__ -> build_route(cfg, site)   binds the cached capability
  _maybe_warmup()                          (attention short-circuited: no scorer call)
EngineCore: get_num_blocks -> allocate_kv_cache
  Qwen4ExpMetadataBuilder.build_kv_cache_tensor -> Qwen4ExpAttention.bind_caches
    route.prepare(compressed_key_cache, scale)   aiter layout_eligibility (once per cache)
                                                 + aiter prewarm, once per (dtype, page) per process
capture_cudagraph
  per-bucket eager warm-up forward: first real scorer call of each class -> aiter decide (cached)
  record: launch(cached_decision, ...) captured; replays run no Python
```

Startup line (logger `atom`), e.g.
`QSA index scorer [dp0_tp0]: enabled profile=qsa_page4_gfx1250_v1 aiter_api=1.1
kernel_sha=58051973… fused_call=launch prewarm=cache-scoped uncovered=stock
kv_block=32 index_page=8`; on any capability failure:
`DISABLED (stock scorer) … reason=<reason>` (WARNING) and every QSA layer keeps
the stock scorer. The check never raises.

## Call path

```
Qwen4ExpAttention._select_tokens
  route.select_paged_tokens(...)  == qsa_select_paged_tokens_routed(..., scorer=route)
    (stock loop: max_seq_len pow2 max_columns bucket, row chunks under the
     128 MiB logits workspace, per-chunk row_starts, top-k, expand)
    route(q[rows], cache, table, t2r[rows], pos[rows], ctx, 4,
          max_columns=..., row_starts=..., allow_flydsl=is_prefill, compressed_k_scale=...)
      entry = cache[class key] or aiter.decide(...)                (once per class)
      fused        -> aiter.launch(decision, <12 args>)            (API 1.1; 1.0: public op)
      aiter-native -> aiter.launch(decision, ...)  (aiter Decision.path == "native")
      stock        -> ATOM qsa_paged_mqa_logits(<same 12 args>)
```

* **Paths.** `fused` = aiter's FlyDSL kernel. `aiter-native` = aiter's op
  serving an uncovered class on aiter's own native path — only with an aiter
  whose `Decision` carries `path` (detected at the capability check); ATOM then
  adds no fallback. `stock` = ATOM's scorer: with aiter ≤ API 1.1 (no native
  path in aiter) this is the **interim** behaviour for uncovered classes, and it
  is always the path of a site whose capability or cache layout failed.
* `qsa_select_paged_tokens_routed` (`qsa_index_scorer/select.py`) is the stock
  loop with the scorer injected; a CPU test AST-compares it to `qsa.py`.
* **Decision caching.** The class key holds everything aiter's `decide` reads
  (shapes/dtypes/strides/devices, q 16-byte alignment, cache identity and scale
  plane, ratio, divisor, `max_columns`, `row_starts` layout). `decide` never
  reads tensor values, so the cached answer is exact; under CUDA graphs it is
  made in the eager warm-up of each bucket, outside the recording.
* **Rank uniformity.** Decisions are pure functions of config, shapes, static
  layouts, device arch/CU and kernel env; every TP rank builds identical cache
  layouts, so ranks decide identically without a collective.
* **Failure boundary.** aiter's O(1) launch guard refusing a decided class
  (`QSAIndexScorerUnsupported`, raised before any launch) re-pins that class to
  the stock scorer; any other exception propagates (no retry after a launch).
* **Observability.** One log line per (site, reason); `route.counts[(path,
  reason)]`, `route.summary()`, and `process_summary()` (all routes of the
  process), logged once at exit as `QSA index scorer decisions: fused=…,
  aiter-native=…, stock=…`. Counts are Python calls: a CUDA-graph-captured call
  counts once (at capture), eager calls every time. Nothing is registered while
  OFF.

## Supported matrix

| Dimension | aiter op path | otherwise |
|---|---|---|
| model | Qwen3.8-Flash-Next (`Qwen4ExpAttention`, target layers and MTP draft layer) | — |
| KV block size / index page | 16 / 4, 32 / 8, 64 / 16 (page sizes the kernel reports) | `cache_shape_unsupported` → stock |
| index cache | bf16, fp8 (E4M3 + fp32 row scale) | fp4 → `cache_format_unsupported` → stock |
| query positions | int64 (ATOM's `logical_positions`) | — |
| rows | decode, draft (tps 1), MTP verify (`num_speculative_tokens + 1` rows/request: 4 at MTP 3 = aiter's decode regime, 3 at MTP 2 = aiter's `rows == 3×requests` bucket, on by default), prefill chunks | per profile rule; uncovered → stock (interim) |
| graph mode | eager and CUDA graph (decided at capture) | — |
| arch | gfx1250 (profile) | `arch_unsupported` → stock |
| frontends | native ATOM server / offline | plugin hosts (SGLang / RTP-LLM): switch not carried, stock |

## Workspace

aiter keeps no persistent device workspace on the default route (the verify
bucket launches native 3-row groups on the caller's tensors); `prewarm`
launches on a private zero problem and frees it. Outputs (`logits`, `visible`)
are allocated per call exactly like the stock scorer. Nothing is charged via
`_kv_budget_extra_reserve`. The pad-4 staging bucket (a fallback only when the
kernel lacks native 3-row groups) allocates small staging planes at `decide`,
i.e. in the per-bucket warm-up before recording.

## Golden adapter and capture

* `python -m atom.model_ops.qwen4_exp.qsa_index_scorer.golden_adapter` (GPU):
  replays the Phase-1 golden cases through the route (site / engine /
  engine_pad / fallbacks), comparator `qsa_golden.compare_logits(rtol=1e-4,
  atol=1e-4)` over `valid_page & c < visible`.
* Capture: `--qsa-index-scorer-capture /out/cap_{rank}.json` writes every call
  class with its decisions keyed `path:reason`; merge into aiter's plan file and
  a coverage matrix:
  `python -m atom.model_ops.qwen4_exp.qsa_index_scorer.capture merge '/out/cap_*.json' --plan plan.json --matrix matrix.json`.
  Plan rows cover index pages 4/8/16 (`page_size` carried when ≠ 4).

## Known limitations

* Pre-existing ATOM issue: the stock scorer's fp8 MFMA route (more than 256
  rows) aborts on gfx1250 (LLVM PHINode assertion); fused-covered fp8 prefill
  chunks avoid it, uncovered ones (stock path) still reach it.
* Until aiter ships `Decision.path`, uncovered classes run ATOM's stock scorer
  (interim; counted as `stock:<reason>`).
* `--enforce-eager` re-runs the route's Python on every call (cached decision +
  `launch`, ~4–5 µs host per call measured on CPU with stub aiter); graph replay
  runs none.

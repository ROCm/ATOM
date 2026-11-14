# QSA index scorer: aiter's neutral op `aiter.ops.qsa.qsa_paged_mqa_logits` (Qwen3.8-Flash-Next)

> **Provider route (this branch).** The scorer is enabled by `--aiter-provider-enable
> qsa-index-scorer-provider` (`Config.aiter_provider_enable`, env
> `ATOM_AITER_PROVIDER_ENABLE`): every worker calls `aiter.plugin.configure(source="atom
> <version>", enable=[...])` in `ModelRunner.__init__` before the model is built; when an enabled
> provider declares aiter's routed op `qsa_paged_mqa_logits` and this aiter routes it, the
> indexer scores through `aiter.ops.qsa.qsa_paged_mqa_logits` (provider on the classes it
> covers, aiter native otherwise; ATOM adds no fallback). After KV allocation and before
> CUDA-graph capture ATOM calls the provider's `warmup(payload, device=...)` once per bound
> index-cache layout. `--qsa-index-scorer-capture` asks `aiter.plugin.explain` once per class.
> `--qsa-index-scorer-profile`, `ATOM_QSA_INDEX_SCORER_PROFILE` and `aiter.ops.qsa.configure /
> prewarm / status` are gone (the profile is the provider's). The sections below still describe
> the previous in-tree revision and are to be rewritten.
The Qwen QSA indexer scores compressed key groups with ATOM's own
`qsa_paged_mqa_logits` (`atom/model_ops/qwen4_exp/ops/qsa.py`; the scorer is
the frozen golden reference of 8a07708, whole file sha256 `c2e46e50…` LF /
`aaa78f43…` CRLF through 10b99a08; main's #2847 split only the selection loop,
so the test now pins qsa.py outside the selection entries, sha256
`4f48345d…`). With
`--qsa-index-scorer-profile` the same calls go to aiter's in-tree **neutral op**
`aiter.ops.qsa.qsa_paged_mqa_logits` (op contract `qsa_paged_mqa_logits` v1;
its 12 leading parameters are ATOM's scorer's). aiter owns the dispatch: its
fused FlyDSL gfx1250 kernel where the configured profile covers the call
class, aiter's native paths (Triton / FlyDSL page-16 prefill / torch)
otherwise; the op never raises for a valid call. ATOM therefore keeps **no
route, no capability pinning and no fallback** of its own. This uses neither
`aiter.plugin` (the out-of-tree provider interface) nor `atom/plugin/`
(ATOM's vLLM/SGLang/rtp-llm frontend adapter).

## Switch

| CLI | Config | Env (operator override) | Default |
|---|---|---|---|
| `--qsa-index-scorer-profile NAME\|PATH.json` | `Config.qsa_index_scorer.profile` | `ATOM_QSA_INDEX_SCORER_PROFILE` (read once by `EngineArgs`, only when the flag is absent) | `""` (OFF) |
| `--qsa-index-scorer-capture PATH` | `Config.qsa_index_scorer.capture` | — | `""` (no observer) |

* **OFF** (default): `Qwen4ExpAttention.qsa_select` and `.qsa_select_groups`
  are None; neither
  `atom.model_ops.qwen4_exp.qsa_index_scorer` nor `aiter.ops.qsa`'s lifecycle
  API is touched (also not by `ModelRunner`); `_select_tokens` calls the stock
  `qsa_select_paged_tokens` (whole batch) or `qsa_select_paged_groups`
  (row-sharded, #2847); `Config.compute_hash()` is unchanged
  (`QSAIndexScorerConfig.hash_factors()` is empty while OFF).
* **ON**: `compute_hash()` includes `("qsa_index_scorer", profile)`. The
  profile reaches aiter only through `aiter.ops.qsa.configure(profile=...,
  source="atom:Config.qsa_index_scorer.profile")`; ATOM never sets aiter's
  env. aiter's own operator controls stay aiter's: `AITER_QSA_PROFILE`
  (override) and `AITER_QSA_FUSED=0` (kill switch: native only).
* **Fail-open**: an aiter without `aiter.ops.qsa.configure` (an image that
  predates the neutral op), an `aiter.ops.qsa` that does not import, or a
  `configure` that raises → one WARNING `QSA index scorer [rank]: DISABLED
  (ATOM scorer) … reason=…`, and every QSA layer keeps ATOM's own scorer.
  This is decided once, before the first scorer call; nothing is retried or
  re-routed after the op has been called.

## Lifecycle (per worker process)

```
ModelRunner.__init__
  _setup_device_and_distributed()
  configure_aiter_fused_ops(config)       atom/aiter_fused_ops.py (ON only)
    -> qsa_index_scorer.configure         aiter.ops.qsa.configure(profile, source); one startup
                                          line from aiter.ops.qsa.status()
  _build_and_load_model()
    Qwen4ExpAttention.__init__ -> index_scorer(config)    aiter.ops.qsa.qsa_paged_mqa_logits
                                                          (or None: ATOM's scorer)
  _maybe_warmup()                         (attention short-circuited: no scorer call)
EngineCore: get_num_blocks -> allocate_kv_cache
  bind_caches (every layer) ... KV-size check
  _prewarm_aiter_fused_ops                aiter.ops.qsa.prewarm(page_sizes, cache_dtypes,
                                          pos_dtypes=(int64,), compressed_k_cache, scale),
                                          once per distinct bound index-cache layout
capture_cudagraph
  per-bucket eager warm-up forward: first call of each class -> aiter decides (cached in aiter)
  record: the decided kernel is captured; replays run no Python
```

`ModelRunner` also prewarms on the disaggregated-decode bind path
(`RapidServeModelRunner._bind_kv_cache_to_modules`). Plugin hosts
(SGLang / rtp-llm) do not carry the switch; were it set there,
`index_scorer()` configures lazily at layer construction (still before the
first call) and variants build on first use (no prewarm hook).

Startup line (logger `atom`), e.g.
`QSA index scorer [dp0_tp0]: enabled profile=qsa_page4_gfx1250_v1
source=atom:Config.qsa_index_scorer.profile op=aiter.ops.qsa.qsa_paged_mqa_logits
status: <aiter.ops.qsa.status() as key=value>`; prewarm:
`QSA index scorer [dp0_tp0]: prewarm page=4 cache=bf16 pos=int64 -> N variant(s) in S s: …`.

## Call path

```
Qwen4ExpAttention._select_tokens
  rows is None:  self.qsa_select(...) == qsa_select_paged_tokens_with_scorer(..., scorer=op)
                   -> qsa_select_paged_groups_with_scorer(...), then expand
  rows given:    self.qsa_select_groups(local rows) == qsa_select_paged_groups_with_scorer(..., scorer=op)
                   -> gather, then expand (#2847's row-sharded indexer selection)
    (stock loop: max_seq_len pow2 max_columns bucket, row chunks under the
     128 MiB logits workspace, per-chunk row_starts, top-k)
    op = aiter.ops.qsa.qsa_paged_mqa_logits(q[rows], cache, table, t2r[rows], pos[rows], ctx, 4,
          max_columns=..., row_starts=..., allow_flydsl=is_prefill, compressed_k_scale=...)
      -> aiter: fused | native-triton | native-torch   (aiter's per-class decision cache)
```

* `qsa_select_paged_groups_with_scorer` / `qsa_select_paged_tokens_with_scorer`
  (`qsa_index_scorer/select.py`) are the stock `qsa_select_paged_groups` /
  `qsa_select_paged_tokens` with the scorer injected; CPU tests AST-compare
  both to `qsa.py` (kept as drift-tested copies so `qsa.py` is not modified).
* Composability with other scorers in the stock loop (e.g. ROCm/ATOM #3107's
  gfx1250 `qsa_index_scores` branch): the routed op replaces the stock
  `qsa_paged_mqa_logits` call only when a provider is enabled; with the switch
  OFF the stock loop, including any such branch, runs unchanged. When such a
  branch lands in `qsa_select_paged_groups`, the AST test fails until the copy
  is re-synced (the branch is then carried as-is, with only the
  `qsa_paged_mqa_logits` call replaced).
* The scorer is the op object itself: no ATOM wrapper, no per-call Python of
  ATOM's on the default ON path. Anything the op raises propagates.
* Traced regions: `Qwen4ExpForConditionalGeneration` is not
  `@support_torch_compile`-decorated; the choice of scorer is made at layer
  `__init__`, never per forward.

## Observability (opt-in)

`--qsa-index-scorer-capture /out/cap_{rank}.json` wraps the scorer in
`ClassObserver`: the first call of each call class runs
`aiter.ops.qsa.explain(*12 args)` (host-side; under CUDA graphs that first
call is the eager per-bucket warm-up, outside the recording), every call is
counted per `path:reason`, the class goes to the capture JSON (schema
`atom.qsa_index_scorer.capture/2`), and the process logs
`QSA index scorer decisions: fused=… native-torch=… native-triton=…` at exit.
Without a profile the observer wraps ATOM's own scorer (path `atom`). Counts
are Python calls: a captured call counts once (at capture). Off by default:
no wrapper, zero cost. Merge per-rank files into aiter's plan file and a
coverage matrix:
`python -m atom.model_ops.qwen4_exp.qsa_index_scorer.capture merge '/out/cap_*.json' --plan plan.json --matrix matrix.json`.

## Supported matrix

aiter decides coverage; ATOM gates nothing beyond "profile set and
`aiter.ops.qsa.configure` present".

| Dimension | fused (aiter profile) | otherwise |
|---|---|---|
| model | Qwen3.8-Flash-Next (`Qwen4ExpAttention`, target layers and MTP draft layer) | — |
| KV block size / index page | 16 / 4, 32 / 8, 64 / 16 | aiter native |
| index cache | bf16, fp8 (E4M3 + fp32 row scale) | aiter native (torch for fp8 / fp4 + scale) |
| query positions | int64 (ATOM's `logical_positions`) | — |
| rows | decode, draft, MTP verify, prefill chunks per profile rule | aiter native |
| graph mode | eager and CUDA graph (decided at the capture warm-up) | — |
| arch | gfx1250 (profile) | aiter native |
| frontends | native ATOM server / offline | plugin hosts: switch not carried, ATOM's scorer |

## Workspace

ATOM allocates nothing for the op: outputs (`logits`, `visible`) are
allocated per call exactly like the stock scorer, and `prewarm` runs on a
private problem after the KV-size check and frees it. Nothing is charged via
`_kv_budget_extra_reserve`. Any persistent workspace of aiter's would have to
be reported by aiter (none on the R2 kernel).

## Known limitations

* ATOM's stock scorer's fp8 MFMA route (more than 256 rows) aborts on gfx1250
  (LLVM PHINode assertion) and the stock FlyDSL page-16 prefill aborts on
  gfx1250 (`permlane32.swap`); with the switch ON, uncovered classes run
  aiter's native paths instead, whose gfx1250 behaviour for these classes is
  aiter's to state.
* `--enforce-eager`: aiter's per-class decision runs on every call (O(1)
  after the first); graph replay runs no Python.
* Golden correctness of the op is aiter's (`op_tests` of the op); ATOM's
  CPU tests cover enablement, ordering and the call site only.

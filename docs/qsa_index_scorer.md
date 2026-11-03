# QSA index scorer: opt-in aiter fused kernel (Qwen3.8-Flash-Next)

The Qwen QSA indexer scores compressed key groups with ATOM's
`qsa_paged_mqa_logits` (`atom/model_ops/qwen4_exp/ops/qsa.py`, unchanged).
With `--qsa-index-scorer-profile` the same calls can run aiter's FlyDSL gfx1250
kernel (`aiter.ops.qsa_index_scorer`, cross-repo contract API 1.x) wherever
the named profile covers the call; everything else stays on ATOM's scorer.

## Switch

| CLI | Config | Default |
|---|---|---|
| `--qsa-index-scorer-profile NAME\|PATH.json` | `Config.qsa_index_scorer.profile` | `""` (OFF) |
| `--qsa-index-scorer-capture PATH` | `Config.qsa_index_scorer.capture` | `""` |

* **OFF** (default): `Qwen4ExpAttention.qsa_scorer is None`; neither
  `atom.model_ops.qwen4_exp.qsa_index_scorer` nor `aiter.ops.qsa_index_scorer`
  is imported; `_select_tokens` calls the stock `qsa_select_paged_tokens`;
  `Config.compute_hash()` is unchanged (`QSAIndexScorerConfig.hash_factors()`
  is empty while OFF).
* **ON**: the profile is passed as `profile=` to every aiter call. ATOM never
  relies on aiter's `AITER_QSA_INDEX_SCORER_PROFILE`; do not set `QSA_*`
  kernel env vars (any non-profiled value makes every call
  `kernel_env_mismatch` → native).
* **Capture** works with or without a profile (without one every call runs
  native with reason `atom_capture_only`).

## Call path

```
Qwen4ExpAttention.__init__     build_route(cfg, site=<layer prefix>)       (ON only)
Qwen4ExpAttention.bind_caches  route.prepare(compressed_key_cache, scale)  -> aiter layout_eligibility, once per cache
Qwen4ExpAttention._select_tokens
  route.select_paged_tokens(...)  == qsa_select_paged_tokens_routed(..., scorer=route)
    (stock loop: max_seq_len pow2 max_columns bucket, row chunks under the
     128 MiB logits workspace, per-chunk row_starts, top-k, expand)
    route(q[rows], cache, table, t2r[rows], pos[rows], ctx, 4,
          max_columns=..., row_starts=..., allow_flydsl=is_prefill, compressed_k_scale=...)
      decision = cache[class key] or aiter.decide(...)      (once per class)
      ok      -> aiter.qsa_paged_mqa_logits(..., profile=P)  (fused)
      else    -> ATOM qsa_paged_mqa_logits(...)              (native, same args)
```

* `qsa_select_paged_tokens_routed` (`qsa_index_scorer/select.py`) is the stock
  loop with the scorer injected; a CPU test AST-compares it to `qsa.py`, so an
  upstream change to the loop fails CI until re-synced. `qsa.py` stays
  byte-identical (sha256 `aaa78f43…`, the frozen golden baseline).
* **Decision caching.** The class key holds everything aiter's `decide` reads:
  shapes/dtypes/strides/devices of every tensor, q 16-byte alignment, cache
  identity (base, shape, dtype, strides) and scale plane, `compress_ratio`,
  `score_divisor`, `max_columns`, `row_starts` layout. `decide` never reads
  tensor values, so the cached answer is exact. Under CUDA graphs Python runs
  only at capture: the decision taken at capture is the one replayed.
* **Rank uniformity.** Decisions are pure functions of config, shapes, static
  layouts, device arch/CU and kernel env. Every TP rank builds identical cache
  layouts (same `LayerMajorArena` offsets), so ranks decide identically; no
  collective, per step or otherwise. The capture `merge` flags any class whose
  decision differs across sites/ranks (`decision_uniform`).
* **Fallback reasons.** Every native call carries a reason: aiter's `Reason`
  values (contract §3) or ATOM's `atom_aiter_unavailable`,
  `atom_api_version_mismatch`, `atom_signature_drift`, `atom_capture_only`.
  One log line (logger `atom`) per (site, reason); counts in
  `route.counts[(path, reason)]` / `route.summary()`. A fused call that aiter
  still refuses (`QSAIndexScorerUnsupported`) runs native and re-pins its class;
  any *other* exception from the fused op propagates (never a silent wrong
  answer).

## Golden adapter and capture

* `python -m atom.model_ops.qwen4_exp.qsa_index_scorer.golden_adapter`
  (GPU): replays the 22 Phase-1 golden cases through the route (`site` mode,
  incl. fused vs native and CUDA-graph replay), the full selection path with
  engine metadata and chunking (`engine`, plus the 2880-row prefill =
  5×512+320), CUDA-graph pad-M decode (`engine_pad`), and every fallback
  reason reachable from the call site (`fallbacks`). Comparator/tolerance: the
  frozen `qsa_golden.compare_logits(rtol=1e-4, atol=1e-4)` over
  `valid_page & c < visible`, plus outside-mask bit-identical `-inf`, no NaN,
  `visible` exact, `row_starts` zeroed.
* Capture: `--qsa-index-scorer-capture /out/cap_{rank}.json` writes every call
  class; merge and get the aiter plan + coverage matrix:
  `python -m atom.model_ops.qwen4_exp.qsa_index_scorer.capture merge '/out/cap_*.json' --plan plan.json --matrix matrix.json`
  then on the aiter side `gen_profile emit --plan-file plan.json` (new rows
  `pending`) and `gen_profile check` on GPU. Never claim coverage beyond
  captured ∩ golden (`matrix[].status`).

## Known limitations

* The kernel is page-4 only (KV block 16, `--block-size 16`). ATOM block 64
  (compressed page 16) → `cache_shape_unsupported`, all native.
* Coverage is exact profile rows. With profile v1 that is decode 64 rows / 16
  requests and draft 16/16 at `columns` 65536 (and 32768) with table width
  16384 — i.e. `--max-model-len 262144`, block 16, bs 16, MTP 3 — the 512-row
  prefill chunk, and the Phase-1 small shapes. Other batch sizes, other
  `--max-model-len` (different table width / column bucket) and most prefill
  chunk sizes run native until captured, added to the profile and GPU-checked.
* Index cache formats: bf16 and fp8 (E4M3 + fp32 row scale). `fp4`
  (`--index-cache-dtype fp4`) → `cache_format_unsupported`.
* Eager mode pays aiter's `decide` again inside its public op on every fused
  call (pure Python, no sync); CUDA-graph replay pays nothing.
* Plugin hosts (SGLang / RTP-LLM) do not carry the switch: always stock.
* The fused kernel is built lazily on the first covered call; the CUDA-graph
  smoke must confirm the build happens in ATOM's warm-up, not inside capture.

## GPU run plan (root; a08-1 GPU3 under gpu-lock)

Image `rocm/fw-bringup-mlperf:gfx1250-atom-20261008`; aiter from
`demonsan/aiter:qsa-index-scorer/stage2`; ATOM from
`demonsan/ATOM:qsa-index-scorer/stage2`.

```bash
# inside the container, aiter + ATOM checkouts installed editable (pip install -e .)
export AITER_LOG_LEVEL=WARNING QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline
unset AITER_QSA_INDEX_SCORER_PROFILE; env | grep '^QSA_' && echo "unset QSA_* first"

# 1. CPU suites (also run in CI without a GPU)
python -m pytest -q tests/model_ops/test_qsa_index_scorer_route.py tests/test_qsa_index_scorer_config.py

# 2. golden replay through the ATOM call site (site / engine / engine_pad / fallbacks)
python -m atom.model_ops.qwen4_exp.qsa_index_scorer.golden_adapter --json /results/atom_golden_adapter.json
python -m pytest -q tests/model_ops/test_qsa_index_scorer_gpu.py    # same + kernel_env_mismatch subprocess

# 3. model smoke, seeded, eager and CUDA graph, OFF vs ON (checkpoint path: root)
M=/models/Qwen3.8-Flash-Next
COMMON="--model $M --block-size 16 --max-model-len 262144 --kv_cache_dtype fp8 -tp 2"; GEN="--temperature 0 --max-tokens 64"   # greedy: no --seed flag exists
rm -rf /root/.cache/atom/*
python -m atom.examples.simple_inference $COMMON $GEN --enforce-eager                          > off_eager.txt
python -m atom.examples.simple_inference $COMMON $GEN --enforce-eager \
  --qsa-index-scorer-profile qsa_page4_gfx1250_v1 --qsa-index-scorer-capture /results/cap_eager_{rank}.json > on_eager.txt
python -m atom.examples.simple_inference $COMMON $GEN                                          > off_graph.txt
python -m atom.examples.simple_inference $COMMON $GEN \
  --qsa-index-scorer-profile qsa_page4_gfx1250_v1 --qsa-index-scorer-capture /results/cap_graph_{rank}.json > on_graph.txt
grep "QSA index scorer" on_*.txt          # one line per (layer, reason): fused/native split
python -m atom.model_ops.qwen4_exp.qsa_index_scorer.capture merge '/results/cap_*.json' \
  --plan /results/plan.json --matrix /results/matrix.json

# 4. accuracy (same seed/settings OFF vs ON), per /ci-pr-guide lm_eval gsm8k against a server
python -m atom.entrypoints.openai_server $COMMON [--qsa-index-scorer-profile qsa_page4_gfx1250_v1]  # run OFF and ON, same lm_eval args
```

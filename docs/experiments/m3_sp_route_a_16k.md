# MiniMax-M3 SP4 route A: T = 65536 and the extended large band

Status: **prepared, not pushed.** This is the ATOM-side counterpart of the route-A aiter
profile (`minimax_m3_sp4_gfx950_v1`: 147 rows / 24 rules, `extended_large` rows at local
m 9216..16384 in 1024 steps, arena 788 MiB). It records what ATOM needs so a 65536-token
batch actually routes to the new band, and the GPU checks that can be run before the
MiniMax-M3 checkpoint is available.

## Band geometry

| SP4 global T | local m = T/4 | profile row used | plan |
|---|---|---|---|
| <= 32768 | <= 8192 | old r13 / r18 grid (unchanged) | large r13 / r18, 19-bit tag |
| 32772 .. 65535 | 8193 .. 16383 | next extended row in the 1024 grid starting at **9216** (real m 8193..9216 pads there) | route A r19, 18-bit tag |
| 65536 | 16384 | exact extended row 16384 | route A r19, 18-bit tag |

The first in-band forward needs **T >= 4 * 8193 = 32772**. A 65536 prompt is block-aligned
(512 x 128) and hits the exact local-m 16384 row; prompts below 32772 stay on the shipped
grid and do not exercise route A. If the scheduler rounds a chunk down to a 128-token
boundary, use prompts >= 32896 (first aligned chunk above the floor), or simply 65536.

## What changed in ATOM (this commit)

1. **Tag widths are per plan, not per source name.** `resources.large_plan_tag_bits(plan)`
   is the single helper (re-exported through `FusionResourcesWrapper.plan_tag_bits` and
   `hooks.plan_tag_bits`):
   - route A **r19 large = 18 bits** (`LARGE_ABI` generation 6; aiter
     `specs.large_tag_bits(src)` is used when importable, with the
     `{r11..r14: 19, r18: 19, r19: 18}` fallback for older builds);
   - r13 / r18 large = 19 bits, r14 large = 19 bits (same tag word);
   - every SMALL plan is `None` (full `call_id` in the tag) -- in particular the SMALL
     `r19` source is ABI generation 3 and must not inherit the 18-bit width.
   The gate is (large `vid` `sp_qkv.large.v1`) **and** ABI generation (4/5/6), never `src`
   alone.
2. **`hooks._build_token_call_ids` assigns the width per token count** and passes it to
   both `tag_unique_ids(keys, tag_bits=...)` and `check_call_ids(..., tag_bits=...)`.
   `FusionResourcesWrapper.tag_unique_ids` forwards the width to aiter (an aiter without
   the argument, or a mixed-width token count, yields no ids and the affected launches go
   **native** on every rank -- a safe downgrade, never a guess).
3. `site._call_id` / `site._tag_refused` ask `hooks.plan_tag_bits` (the same helper), so the
   id ATOM launches with and the width ATOM checks against cannot drift.
4. Tests: `tests/model_ops/test_gemm_fusion_route_a.py` (CPU, no aiter/GPU required) pins
   large r19 abi 6 -> 18, large r13/r18 abi 4 -> 19, SMALL r19 abi 3 -> None, the
   `tag_bits=18` forwarding, a forced low-18 collision refusal and the mixed-width native
   fallback.

This is a new feature commit: the "5 shipped files byte-identical to `efb025284`" pin
belongs to the PR #2643 rebase commits (`c645d60f` / `6df8f5794`) and does not apply here.

## Launch recipe (ATOM, single server, SP4)

The M3 SP4 experiment recipe (`docs/experiments/m3_tp_replicated_o_proj.md`) keeps its
existing values except the two token-budget knobs:

```bash
python -m atom.entrypoints.openai_server \
  --model  <MiniMax-M3-MXFP4 path> \
  --tensor-parallel-size 1 --sequence-parallel-size 4 \
  --gemm-fusion-profile minimax_m3_sp4_gfx950_v1 \
  --gemm-fusion-check-period 1 \
  --max-model-len 132096 \
  --max-num-batched-tokens 65536 \
  --long-prefill-token-threshold 0 \
  --max-num-seqs 128 --block-size 128 \
  --gpu-memory-utilization 0.8 \
  --kv_cache_dtype fp8 --no-enable-prefix-caching
```

- `--max-num-batched-tokens 65536` is the required change (from 32768 / the repo default
  16384). At 32768 the largest forward is local m 8192 and route A is **never** reached.
  `--max-model-len` must be >= 65536 for a 65536-token prompt; the M3 recipe already uses
  132096.
- `--long-prefill-token-threshold 0` (default) and dynamic chunking off (default) keep a
  prompt in one scheduling chunk; a threshold or the dynamic-chunking predictor can split
  it below the 32772 floor.
- `--attn-prefill-chunk-size` is the **MLA** budget and does not apply to M3's MHA path;
  changing it will not enable 65K prefills.
- `--gemm-fusion-check-period 1` is for correctness runs (synchronous device error check);
  the default 14 is the production setting.
- The profile may be passed as a name under `aiter/configs/gemm_fusion/` or as an
  explicit path (aiter's `_resolve` accepts a path / `.json`); ATOM passes the string
  through unchanged to `load_profile`, `FusionResources(profile=...)` and
  `prebuild_profile`.
- Pair with the route-A aiter build (ABI generation 7, matching MORI/`AITER_GEMM_FUSION_MORI_ABI_PIN`).
  The shipped flip tree has no r19 / `extended_large` rows, so the new profile would fall
  back to native (or fail prebuild) there.

## Tracked benchmark config entries (opt-in, existing runs unchanged)

- `.github/benchmark/oot_benchmark_models.json`: a new MiniMax-M3-MXFP4 variant
  (`display`: `MiniMax-M3-MXFP4 SP4 16K Route-A`, prefix
  `minimax-m3-mxfp4-sp4-16k-route-a`) with `--sequence-parallel-size 4`,
  `--gemm-fusion-profile ...`, `--max-model-len 65536`, `--max-num-batched-tokens 65536`.
  The workflow selects variants by **`display`** against a fixed allow-list
  (`atom-vllm-benchmark.yaml`), so the new display is inert until it is added there; the
  existing `MiniMax-M3-MXFP4 TP4` entry is unchanged.
- `.github/benchmark/models_atomesh.yaml`: a new `route_a_16k` suite (aggregated,
  SP4) under the MiniMax-M3-MXFP4 model with the same 65536 limits. The scheduled/manual
  runners iterate the existing suite names (`nightly` / `weekly` / `smoke` / `accuracy`),
  so no existing cell changes; schedule the new suite explicitly when the route-A image is
  deployed.

## GPU validation list (E1c / E6a style)

"Model-free" items need only the route-A aiter build and GPUs, not the M3 checkpoint.

| ID | Needs | Check | PASS |
|---|---|---|---|
| RA0 | CPU | `pytest tests/model_ops/test_gemm_fusion_route_a.py` | all green; widths 18/19/None as above, forced low-bit collision goes native |
| RA1 | CPU + route-A aiter | `python -c` widths + `aiter.gemm_fusion.callid.tag_unique_ids(keys, tag_bits=18)` over ~80 synthetic site keys at tokens=65536 | r19 large 18, r13/r18 19, SMALL r19 None; low-18 ids distinct and non-zero; `check_call_ids(..., tag_bits=18, per_tokens=True)` reports no collision |
| RA2 | GPU, no model | `torchrun --nproc-per-node=4 tests/distributed/check_ulysses_exchange.py` | PASS bitwise on every cell; the script already covers totals up to 131072 (a superset of the 65536 forward) |
| RA3 | GPU, no model | `ATOM_CHECK_MXFP4_TOKENS=16,8192,32768,65536 torchrun --nproc-per-node=4 tests/distributed/check_ulysses_mxfp4.py` | PASS at 65536: BF16-gather fallback byte-exact (or the prequant branch byte-exact when eligible), MoE/reduce-scatter/graph replay unchanged |
| RA4 | GPU, no model | route-A profile prebuild: `python -c` `load_profile` + `prebuild_profile(name, arch=..., cu=..., world=4)` (see the aiter route-A branch) | prebuild rc=0, no OOM; profile 147 rows / 24 extended r19 rows, max m 16384, first extended row m 9216; window arena 788 MiB |
| RA5 | GPU + M3 checkpoint | e2e greedy, T in {32896, 40000, 65536}, fusion ON vs OFF | token-identical outputs; decision record shows the large r19 plan (`bucket_m` 10240/16384, large vid) for the band; no tag refusal / hang |
| RA6 | GPU + M3 checkpoint | same runs as RA5 | startup prebuild/prepare time, KV block count and peak HBM vs the 32768 run; bind succeeds with the 788 MiB arena |
| RA7 | GPU + M3 checkpoint | non-fusion SP reference at T=65536 | correct output (expected PyNccl fallback for the head exchange / quick-reduce above their size bounds); performance recorded, not a gate |
| RA8 | GPU + M3 checkpoint | optional captured 65536 prefill (`ATOM_TOKEN_GRAPH_SIZES` includes 65536) | only if capture is wanted; band routing itself runs eager |

RA5..RA8 are **NOT RUN** until the M3-MXFP4 checkpoint is available. RA0/RA1 are runnable
now and RA2/RA3/RA4 need only the route-A aiter build on a gfx950 node.

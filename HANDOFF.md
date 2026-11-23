# MiniMax-M3.1 TP4 perf reproduction: ATOM + aiter branches

Two branches:

- **ATOM**: `ganyi1996ppo/ATOM` branch `p1-c17-reproducible`
  - commit 1 (`04ab0f63`): full tree snapshot of the internal working tree
    `/shared_nfs/ygan/work/p1-c17` @ `2d96c9954`, which is where every
    measured number in this project was actually produced. This replaces
    ATOM's origin/main tree wholesale (p1-c17 diverges from main by ~79
    commits across many experimental tracks).
  - commit 2 (`99cbd9b4`): the c24opt scheduler serial-fix
    (`ATOM_P1_SCHED_SERIAL_MAXN` etc.), ported from `p1/c24opt` @ `af57ffe7a`.
    Needed specifically for the measured c2 win; c1 does not need it.
  - commit 3 (`c2139153`): the `p1_sls_shared` PYTHONPATH-overlay package
    (SP-prefill shared-expert kernels) that C17's recipe loads from outside
    the git tree. Added at repo root here.
  - commit 4 (`17173e14`): the previous HANDOFF.md (import/unit checks only).
  - commit 5 (`6d95d877`): revert `.github/workflows/` to upstream main.
  - commit 6 (this one): this HANDOFF.md update, with the actual wired e2e
    verification this branch was missing (see below).

- **aiter**: `ganyi1996ppo/aiter` branch `p1-integration` (already existed,
  re-verified this pass): bundles PRs #8465, #8472, #8473 on a clean
  `origin/main` base, zero conflicts. Confirmed this pass: the production
  container's baked-in aiter commit (`b9e08daca`) **is a direct ancestor**
  of this branch's HEAD (114 commits behind, zero divergence) -- this
  branch is a strict superset of what the container ships today, not a
  fork that reverts or replaces anything production currently relies on.

## This campaign's other tracks: what actually landed (honest summary)

Two other tracks ran concurrently this campaign and were read (not
duplicated) for this final-packaging pass:

- **Mono G2 (>=10% gain target): NOT achieved.** Full ticket chain
  (016/020/024/025/026/030/031/032, kb `decode-kernels.md`): the one
  real, GPU-measured, bit-identical GO win found is a down-projection
  scale-byte-load batching fix (~2.6-3.4%/layer, ticket 016) --
  "not sufficient alone to flip G2 mono to a win" per its own writeup.
  Every other avenue tried was NO-GO or zero-benefit: the design panel's
  own top pick (wave-owned up/gate reduction, ticket 032) measured a
  reproducible **~2.1x slowdown** (VGPR spilling to scratch); the
  attention-front split-waste hypothesis (024/025) was implemented
  correctly and verified safe but measured **zero benefit** (hypothesis
  falsified); the full K4 stage audit (ticket 026, all 6 pipeline stages)
  concludes explicitly: "the 10%-ahead-of-production target remains
  unsupported by anything found in this full sweep." **Nothing from this
  track is merged into this branch** -- the one real win (scalefix) is on
  a code path (G2 mono) that the C17 recipe doesn't even run in
  production (`ATOM_P1_MONO_MAX_ROWS=8` caps mono to G1 rows), so merging
  it would change nothing measurable; the rest is NO-GO or inconclusive.
- **amd_kernels elimination: NOT completed, NOT claimed.** The
  `aiter-moe-gap` one-launch fused-MoE prototype meant to replace
  amd_kernels' GEMM1->GEMM2 path is still not correctness-verified (ticket
  033: the originally assigned root-cause hypothesis was directly
  REFUTED by a live GPU dump; the corrected failure signature points at
  an unresolved producer/consumer visibility race, not yet fixed). Even
  if it were fixed, a prior pass found it imports amd_kernels' own
  private tile/MFMA/buffer-pointer primitives directly -- a second,
  separate blocker before it could satisfy the project's
  no-amd_kernels-at-runtime rule. **Direct grep of this branch's own
  `atom/` tree, this pass: 19 files still `import amd_kernels` /
  `from amd_kernels import ...`**: `model_ops/{ak_gemm,layernorm,linear}.py`,
  `model_ops/minimax_m3/{ak_kernels,p1_front,prefill_fast,prefill_kernels}.py`,
  `model_ops/{p1_lmhead_ring,p1_markov_block}.py`,
  `models/minimax_m3/{ak_moe,ak_moe_g1prio,ak_moe_tiles,dspark,mbf_back,
  model,sp_moe_compact,sp_prefill,verify_head}.py`, `utils/envs.py`.
  **This branch still depends on amd_kernels at runtime. Do not read the
  absence of a changelog entry here as evidence otherwise.**

## Required runtime env vars (verbatim, from env_armC17.sh -> env_armB.sh)

```
# base (env_armB.sh)
export ATOM_MONO_ENABLE=1 ATOM_P1_MONO_MAX_ROWS=8
export ATOM_MONO_STATS_EVERY=200 ATOM_ENABLE_METRICS_DEVICE_TIMER=1

# C17 additions
export ATOM_P1_SP_SHARED_LOCAL=1 ATOM_P1_SP_SHARED_LOCAL_MIN_TOKENS=2048 ATOM_P1_SP_SHARED_LOCAL_SIDE_ROWS=1
export ATOM_M3_SP_PREFILL=1 ATOM_P1_SP_MOE_SLOTS=1 ATOM_M3_AK_MOE=1 ATOM_P1_SP_MOE_COMPACT=1

# c2-specific (on top of the above, not needed for c1)
export ATOM_P1_SCHED_SERIAL_MAXN=2 ATOM_P1_SCHED_SERIAL_AGE_TOKENS=256 ATOM_P1_SCHED_SERIAL_GAP_TOKENS=64
```

The full, real recipe also carries ~30 more `ATOM_*`/`AITER_*` flags
accumulated through the C1->C17 arm chain (dense/draft GEMM fusions, MoE
tile/ID-row routing, split4 indexer, drafter Markov persistence, prefill
SP/TBO, etc.) -- see `/shared_nfs/ygan/p1/grid2/c1/recipe/env_armC17.sh`
(and the arms it sources back to `int1_recipe.sh`) on the cluster for the
byte-exact, currently-in-force set; this HANDOFF only calls out the ones
that differ between c1 and c2 and the ones a reader is most likely to get
wrong.

PYTHONPATH must include, in this order: (1) the aiter branch's root (so
`import aiter` resolves there, not to a pip/baked copy), (2) this repo's
root (for `atom` and `p1_sls_shared`), (3) whatever non-git overlay
packages the recipe's `PYPATH_PRE` chain adds for other flags (amd_kernels
snapshot, draft-fusion/campaign/k2-compose/sp-local-shared overlays -- see
the cluster recipe; these are infrastructure this branch's flags still
call into, not part of either branch under test).

**IMPORTANT -- the submodule gotcha that bit this pass**: a shallow
`git clone` of the aiter branch does **not** fetch its
`3rdparty/composable_kernel` submodule; several JIT kernel builds (e.g.
`pa_ps.py`) hard-fail with `FileNotFoundError` on
`3rdparty/composable_kernel/include` without it. Always
`git submodule update --init --recursive` right after cloning aiter.

## Verification actually performed

### 2026-11-23 (earlier pass): import/unit level only
- `import aiter` succeeded with a real JIT build on real hardware (not
  just py_compile). `test_fused_qknorm_nvfp4.py` / `test_flydsl_prefill_
  launch_plan.py` -- all cases passed (these are the repro commands from
  PRs #8472/#8473/#8465's own descriptions).
- `import atom` succeeded on real hardware; `pytest tests/test_scheduler.
  py` -- 225 passed.
- Explicitly flagged as NOT done: "a full end-to-end server forward
  pass... those numbers stand as originally measured... not re-validated
  from this exported branch."

### 2026-11-23 (this pass): the missing piece -- a real, wired e2e

This closes exactly the gap the previous pass flagged. Procedure:
1. **Fresh clones** of both fork branches into a new directory
   (`/shared_nfs/ygan/p1/grid2/final-e2e-repro/{ATOM,aiter}`), not the
   cluster's original working trees.
2. A **new, isolated container** (`p1final35`, node 035 MI355X -- separate
   from the shared `p1` container other tracks depend on).
3. **Verified the wiring directly** (not assumed): with no PYTHONPATH
   override, `import aiter` (run from a neutral cwd) resolves to
   `/app/aiter-test/aiter/__init__.py` -- the image's baked-in copy. With
   `PYTHONPATH=<aiter checkout>:<rest>`, it resolves to our checkout
   instead, and performs a real JIT kernel build (`module_aiter_core`,
   ~11s) on real MI355X hardware.
4. Launched the **real C17 recipe, byte-for-byte** (same ~35 env vars,
   same MEM/k/AL/cudagraph sizes), via `gpurun.sh GPUS=4 GPURUN_PRIO=6`,
   with only `REPO` and `PYPATH_PRE` swapped to point at the fresh
   checkouts -- not a from-scratch reimplementation of the recipe.
5. Confirmed from **inside the actually-running server process's own
   log**: `aiter.__file__=/shared_nfs/ygan/p1/grid2/final-e2e-repro/
   aiter/aiter/__init__.py` and `atom.__file__=/shared_nfs/ygan/p1/
   grid2/final-e2e-repro/ATOM/atom/__init__.py` -- both branches' real
   code, not the baked-in / cluster-tree copies.
6. Ran one real AgentX-style point: **c1 (concurrency 1), 900s**
   (aiperf's `inferencex-agentx-mvp` scenario floor; a 3600s official
   point was out of scope for this verification pass), against the real
   `semianalysis_cc_traces_weka_062126` dataset, 393 traces, seed 42.

### Result (actual; n=1, 900s, c1 conc=1, node 035 GPUs 0-3, seed 42)

| | RS tok/s/GPU | ix_p90 | ix_p50 | TTFT p50/p90 ms | hit/theo | AL | ISL mean | n/err |
|---|---|---|---|---|---|---|---|---|
| **this pass (wired, fresh checkouts, 900s)** | 6817 | 579.8 | 696.4 | 90/450 | .949/.958 | 4.04 | 167K | 148/0 |
| known baseline (ATOM C17, quiet, official 3600s) | 8029 | 655.9 | 686.3 | 144/205 | .981/.984 | -- | 292K | 393/0 |
| vLLM published c1 | 7744 | 621.1 | -- | --/431 | -- | -- | -- | -- |

**Honest read of this result -- NOT a clean match, and here is exactly
why, measured not guessed**: RS is 15.1% below the 3600s baseline (12.0%
below vLLM-published), ix_p90 11.6% below, and TTFT p90 is worse (450 ms
vs 205 ms baseline, 431 ms published). The single largest confound,
visible directly in the numbers: **ISL mean 167K vs 292K, n=148 vs
393** -- c1 (concurrency 1) is a single-session, think-time-limited
workload where "which trace(s) happen to be in flight" entirely
determines the sample (already documented as a known property of c1 in
this project's own kb before this pass: "n=1; one session so trace mix
dominates"; "no same-condition vLLM c1 control" caveats on every prior
c1 number). A 900s window catches roughly 148/393 = 38% as many
completed requests as the 3600s window, inherently biased toward
shorter, earlier-completing traces (hence the much smaller ISL mean) --
this is a materially different, not just noisier, sample, not an
apples-to-apples short rerun of the same distribution.
**What is NOT confounded, and is the actual load-bearing evidence this
ticket set out to get**: the run completed its full 900s window with
**zero errors, zero stalls, zero allocator/OOM events** (checked every
60s for the whole span), a synthetic acceptance length that landed
exactly on the golden value for k=7 (AL=4.04, i.e. the DSpark path
behaved as expected), and a prefix-cache hit rate (0.949 actual / 0.958
theoretical) in the normal range for this recipe -- i.e. **both branches'
real code, wired together for the first time, ran correctly end-to-end
under the real production recipe**, which is what the known prior gap
was about. The specific throughput/latency numbers are evidence of
correct, healthy operation, not a load-bearing performance claim at this
duration/concurrency -- that would need a same-condition baseline rerun
(original tree, same 900s/c1/seed) to separate "fresh-checkout effect"
from "c1-at-900s sampling effect," which this ticket did not do (out of
scope: the goal was reproduction, not a new A/B).

**What this verification does and does not establish**: it establishes
that both branches' real code runs correctly together, end-to-end, under
the real recipe, on real hardware, serves real traffic with zero errors
for the full window. It does **not** establish a performance number
comparable to the 3600s official baseline at this duration/concurrency,
for the sampling reason above -- that was foreseeable going in (c1's own
kb history already flags this), and this HANDOFF says so plainly rather
than reporting the 900s number as if it settled anything about relative
performance. For a load-bearing performance claim, rerun the full 3600s
point from these same checkouts (expensive; not done this pass), or at
minimum a same-duration same-condition control from the original
`/shared_nfs/ygan/work/p1-c17` tree to isolate the sampling effect.

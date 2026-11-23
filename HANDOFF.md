# MiniMax-M3.1 TP4 perf reproduction: ATOM + aiter branches

Two branches, no cluster access needed:

- **ATOM**: `ganyi1996ppo/ATOM` branch `p1-c17-reproducible`
  - commit 1 (`04ab0f63`): full tree snapshot of the internal working tree
    `/shared_nfs/ygan/work/p1-c17` @ `2d96c9954`, which is where every
    measured number in this project was actually produced. This replaces
    ATOMs origin/main tree wholesale (p1-c17 diverges from main by ~79
    commits across many experimental tracks -- a curated cherry-pick was
    attempted and abandoned: several individually-opened PRs turned out to
    be redundant/superseded by what p1-c17 already independently built).
  - commit 2 (`99cbd9b4`): the c24opt scheduler serial-fix
    (`ATOM_P1_SCHED_SERIAL_MAXN` etc.), ported from `p1/c24opt` @ `af57ffe7a`.
    Needed specifically for the measured c2 win; c1 does not need it.
  - commit 3 (`c2139153`): the `p1_sls_shared` PYTHONPATH-overlay package
    (SP-prefill shared-expert kernels) that C17s recipe loads from outside
    the git tree. Added at repo root here.

- **aiter**: `ganyi1996ppo/aiter` branch `p1-integration` (already existed,
  re-verified this pass): bundles PRs #8465, #8472, #8473 on a clean
  `origin/main` base, zero conflicts.

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

PYTHONPATH must include both this repos root (for `atom` and `p1_sls_shared`)
and the aiter branchs root (for `aiter`).

## Verification actually performed (2026-11-23, real GPU, node 034/035, MI355X)

- **aiter**: `import aiter` succeeded with a real JIT build on real hardware
  (not just py_compile). `test_fused_qknorm_nvfp4.py -t 1 7 8 16 64 300
  --perf_tokens 8` -- 1260+ byte-for-byte (atol=0 rtol=0) checks, all
  passed. `test_flydsl_prefill_launch_plan.py --case plan lazy` -- all 25
  checks PASS. These are the exact repro commands from PRs #8472/#8473/#8465
  own descriptions.
- **ATOM**: `import atom` succeeded on real hardware. `atom.models.
  minimax_m3.mono.{runner, kernels.stages.{moe_wide,attention,router,
  select}}` all import cleanly (the mono decode path specifically).
  `pytest tests/test_scheduler.py` -- 225 passed.
- **NOT performed this pass**: a full end-to-end server forward pass, or
  any AgentX/GSM8K re-run of the original c1/c2 performance numbers
  themselves -- those numbers stand as originally measured on the cluster
  (see ticket history), not re-validated from this exported branch. A
  colleague checking this out should expect to need a short smoke run of
  their own before trusting it for anything beyond "the code is correct
  and imports/runs on real hardware."

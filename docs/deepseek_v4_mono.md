# DeepSeek-V4 Mono ownership

The V4 backend follows the V4.1 mono separation between deployment admission,
execution, device plans and owned state. It retains V4's own arithmetic and
persistent-mailbox protocol; it does not substitute V4.1 kernels.

```text
deepseek_v4/model.py                 native model and post-load entry
deepseek_v4/mono/
  dispatch.py                       installation and deployment refusals
  profile.py                        ordinary / MTP3 target shape policy
  session.py                        serving, graph and publication lifetimes
  contract.py                       exact request-frontier continuation
  cache.py                          real KV/state views and metadata buckets
  runner.py                         whole-stack prepare / preflight / forward
  weights.py                        production binding and lossless packing
  state.py                          compiled variants, scratch and peer arenas
  layer.py                          single-layer raw-pointer ABI wrapper
  sources.py                        explicit device dependency digest
  kernels/plan.py                   device regions, tiling and task counts
  kernels/*.py                      FlyDSL device computation
```

Reusable compilation, TP agreement, IPC and device helpers live in `atom/mono/`.
The serving session owns its runner and cache binding; the runner owns distinct
per-width/per-lane variants. Large production expert banks are bound in place.
Small packed weights are prepared before the engine sizes its KV budget.

## Execution and lifetimes

S1 executes one whole-layer launch. S2/S4/S8 execute attention then MoE using the
same scratch, peer buffers and epoch. Each variant advances its epoch once after
the layer stack, never between phases. Repeated micro-buckets use distinct lanes
to prevent a faster TP rank overwriting a peer mailbox still being consumed.

The cache adapter binds engine-owned paging, RoPE and rolling state; it neither
allocates a second production cache nor clears live request state. Metadata is
published outside graph capture/replay into stable per-bucket buffers. A stable
continuation must preserve request membership, slot generations and the block
frontier. Weight replacement, slot reuse, cache rebinding and graph recapture
invalidate the relevant publication or graph generation before a new forward.

The native path remains available with `--mono-decode off`. `auto` selects only
qualified widths and requests; `force` fails on unsupported live decode instead
of silently benchmarking native fallback. The first qualified profiles are
pure TP8 on gfx950, FP8 KV / FP4 index, ordinary greedy decode at C1/C4/C8 with
index-cache reuse, or the fixed MTP3 target verify window (q=4). Deployment
refusals and exact width policy are authoritative in `dispatch.py`/`profile.py`.
DSpark mono adaptation is not enabled by this backend; existing shared V4 import
and custom-op tests are compatibility coverage, not a new DSpark execution path.

## Compilation and validation

FlyDSL's automatic source walk does not cover every helper reached in a traced
branch or every build-time layout. `sources.py` uses the shared `source_digest`
to key device helpers, formats, V4 arithmetic configuration and kernel plans.
Both the layer and epoch-advance launchers capture that digest. Serving-only
changes do not invalidate device binaries. The compiled-key tests prove reuse
for identical dependencies and a new binary when the digest changes.

Structural moves retain arithmetic, quantization boundaries, task placement,
launch order and original numerical tolerances. Validation must include existing
publication/lifecycle tests, fresh gfx950 compilation, real-model generation,
and matched before/after workloads; skipped tests are not passes. Use the
repository `run-atom-workload` skill and `scripts/` ready/drain/stop gates. For
V4-Pro, keep `AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1` in both runs, and
compare identical model, TP, graph sizes, sequence cap, prompt lengths and client
seeds. `UPSTREAM.json` records the Apache-2.0 device sources and integrated hashes.

For a quick structure-only iteration, run the publication, build-key, shared
mono plan/runtime and V4.1 dispatch tests first. The publication tests include
the existing MTP3 q=4 admission and continuation contract. Build-key tests
compile for gfx950 without launching device work and must run with FlyDSL
installed; a skipped compile test is not compilation evidence. This quick gate
does not replace matched accuracy evaluation, real MTP3 serving or a full
performance qualification.

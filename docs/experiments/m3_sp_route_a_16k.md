# MiniMax-M3 SP4 T=65536: phased migration to the `sp_qkv_fused` route

Status: **prepared, not pushed.** This is the ATOM-side counterpart of the aiter
`sp_qkv_fused` v1 contract (aiter PR #6689 / branch `blyu/gemm-fusion-sp-qkv`,
contract sheet `aiter/ops/gemm_fusion/sp_qkv/CONTRACT.md`): one neutral routed op
for the MiniMax-M3 SP4 QKV step (QKV+indexer GEMM, Ulysses head exchange, head
Gemma-RMSNorm, partial-NeoX RoPE, SHUFFLE KV/index cache writes and per-token FP8
scales). The op's native runtime owns profile/variant/plan selection; ATOM owns
an opt-in flag, a static coverage matrix and the tensor mapping.

**This PR adds the provider-ready path and gates it off. The existing ATOM path
is unchanged and remains the active path for the 16K band.** The migration is
phased because aiter's coarse native implementation currently covers the
2-stage *prologue* cells only:

| Path | Status | Band |
|---|---|---|
| Existing ATOM hooks/framework path (`--gemm-fusion-profile`, PR #2643 + this branch's r19 18-bit tag classification) | **active in this PR** | large plans local m 386..8192 and r19 rows m 9216..16384 (T up to 65536) |
| Neutral `sp_qkv_fused` provider path (`--aiter-provider-enable` + `--enable-sp-qkv-fused`) | added, default OFF, coverage-gated | 2-stage prologue cells: local m 1..256 (all modes, pad hits included) |
| Large/16K band on the neutral op | **Phase-2 follow-up** | needs aiter's native post-exchange prologue op (head norm + RoPE + KV/index-cache insert); `sp_qkv_native` raises `ContractError` for uncovered calls today |

`aiter.sp_qkv_covers(...)` is the cheap static predicate ATOM consults at bind
time (no execution). Since it currently accepts only prologue cells, the mode-B
call site binds only in the m <= 256 band; the 16K recipe keeps
`--max-num-batched-tokens 65536` and continues on the existing path. When the
Phase-2 native prologue lands, the same flag flips the band over with no ATOM
call-site change.

## Coverage and band geometry

| SP4 global T | local m = T/4 | detail |
|---|---|---|
| <= 1024 | <= 256 | both paths: neutral op when the provider path is enabled and `sp_qkv_covers` accepts the cell; otherwise the existing path |
| 1025 .. 32768 | 257 .. 8192 | existing ATOM path (aiter one-stage large band is not in the coarse op's native coverage yet) |
| 32772 .. 65535 | 8193 .. 16383 | existing ATOM path (pads into the r19 grid; first row m=9216) |
| 65536 | 16384 | existing ATOM path; exact r19 row m=16384 |

The first in-band forward needs `T >= 4 * 8193 = 32772`; a 65536-token prompt is
block-aligned and hits the exact local-m 16384 row. The r19 18-bit tag
classification added on this branch is what keeps the existing path safe across
that band (the PR #2643 hooks were certified for the 19-bit generation only).

## What ATOM does (this PR)

1. **Enablement rail** (`atom/aiter_provider.py`, default OFF): one
   `aiter.plugin.configure(source="atom <version>", enable=[...])` call per
   worker process, after device/distributed setup and before model build /
   warm-up / capture. It checks `aiter.plugin.API_VERSION`, logs `status()` once,
   warns when a routed op ATOM expects is missing, and is fail-open (missing
   interface / provider error -> native, never raises). `AITER_PLUGINS=0` is
   aiter's own kill switch; ATOM never sets it.
2. **Call site** (`atom/model_ops/sp_qkv_fused.py`, default OFF): `bind()` checks
   the whole matrix once -- flag, op routed in this aiter, `sp_qkv_covers(...)`
   truthy for **every local-M cell this deployment can present**, gfx950, FP8
   page-128 KV, 16x16-preshuffled FP8 QKV weight + scale, SP world 4,
   dense/indexer/reuse mode, the #2643 FP8-output/o_proj gate, and the
   attention-impl prologue seam. Installed on the M3 attention modules after
   weight loading and before warm-up/compile/capture, the same model-adapter seam
   as #2643; the installed forward is one splitting custom op plus `o_proj`, with
   no per-forward Python branch and no packed-qkv leak into the attention tail.
   The tail reuses `impl.forward_after_prologue` + the FP8 head gather, so fused
   and native paths share it. There is **no try/except fallback** around the
   routed call: aiter's coverage gate decides before the call; an uncovered call
   is a contract error, not an ATOM retry.
   - The coverage query is structural (aiter reads shape/dtype/numel only, never
     values/devices). It runs on the real GPU worker, which supplies the live
     ROCm context and the real SP group; the predicate is asked for the
     deployment's small-plan cells `{1,2,4,8,12,16,32,64,128,256}` filtered to
     its maximum local rows, plus that maximum itself, and the per-(mode, cell
     set) verdict is cached across layers. A recipe whose maximum local M is in
     the Phase-2 large band therefore refuses the binding even though the small
     cells are covered.
   - cos/sin are bound as the **two halves of M3's concatenated cache**, which
     the kernels accept zero-copy when the driver constraints hold: equal
     `stride(0)` on both halves, `stride(0) % 8 == 0`, `stride(1) == 1`, and
     16-byte-aligned base pointers. The wrapper validates those; if a rotary
     module ever produces another layout it falls back to contiguous copies.
   - Outputs are one **bucket-sized, address-stable set shared by all sites**
     (`q/k/v/index_q`, keyed by device/mode/geometry/max tokens). The runtime
     always binds its own bucket-sized staging and copies the real
     `4*real local M` rows into the caller's buffer on the call stream, so caller
     output pointers are **not** part of the binding identity; per-call row views
     are smaller than the buffer while the base pointers stay fixed for the
     process lifetime.
3. **Config/CLI/env**: `Config.aiter_provider_enable` (tuple, normalized),
   `Config.enable_sp_qkv_fused` (bool), `--aiter-provider-enable` (comma list),
   `--enable-sp-qkv-fused`, `ATOM_AITER_PROVIDER_ENABLE` (deployment override),
   and both graph-changing knobs key `Config.compute_hash()` only when ON, so
   every existing compile artifact and default path is unchanged.
4. **r19 tag widths stay**: the custom 18-bit classification for route A's
   large r19 source remains in the legacy hooks path (the existing
   `resources`/`hooks`/`site` helpers), because that path is the one carrying
   16K in this PR.

## Launch recipes

**16K band (this PR, active path)** -- the M3 SP4 recipe plus the r19 profile:

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

**Prologue band on the neutral op** (default OFF; requires the provider image
and the provider-interface aiter build):

```bash
AITER_GEMM_FUSION_PROFILE=<aiter profile path or name> \
python -m atom.entrypoints.openai_server \
  --model <MiniMax-M3-MXFP4 path> \
  --tensor-parallel-size 1 --sequence-parallel-size 4 \
  --aiter-provider-enable <provider-name> \
  --enable-sp-qkv-fused \
  --max-num-batched-tokens 1024 \
  ...   # same serving knobs as above
```

- `--max-num-batched-tokens 65536` is required for the 16K band (32768 keeps
  local m <= 8192); the neutral op path in this PR is bounded by
  `sp_qkv_covers` (prologue m <= 256 today) and will bind None on a 65536
  recipe with a warning -- the existing path serves the run.
- `--long-prefill-token-threshold 0` (default) and dynamic chunking off
  (default) keep the prompt in one scheduling chunk; otherwise it can split
  below the 32772 floor.
- `--attn-prefill-chunk-size` is the **MLA** budget and does not apply to M3's
  MHA path.
- `--aiter-provider-enable <name>` must name the provider that implements the
  op; without the provider interface both paths degrade to native with a
  warning.
- `AITER_GEMM_FUSION_PROFILE` is the aiter-side profile environment variable.
  The aiter PR ships the default profile (`minimax_m3_sp4_gfx950_v1.json`,
  r19-free) and the r19 band only as the alternate opt-in
  `minimax_m3_sp4_gfx950_v1_r19.json`; profile selection is never an ATOM op
  argument. Pair the provider path with the provider-interface aiter build and
  matching MORI/`AITER_GEMM_FUSION_MORI_ABI_PIN`.

## aiter runtime contract notes (answers from the runtime owner)

For the head `blyu/gemm-fusion-sp-qkv-refactor` @ `af83fa7b9` (runtime.py
sha256 `F0B94C23…CE98`):

- `sp_qkv_covers(payload)` is shape/dtype/numel-only and non-raising; it needs a
  live GPU/ROCm context and a real SP group, no tensor values or device
  residency. It delegates to the native runtime's planner, so it is the same
  predicate the native path uses; `sp_qkv_native` raises `ContractError` for
  uncovered cells (no silent native fallback).
- Binding identity = `(device, rank, world, group id)` + `(mode, bucket M)` +
  bound pointers {w_scale, k/v caches + k/v scales, q/k norm weights, cos/sin
  caches, index norms/cache, runtime bucket staging}. Not bound: caller
  outputs/views, x, w, x_scale, positions, slot_mapping, stream.
- `_MAX_BINDINGS=64` is the variant plan's internal cap, not the runtime's; the
  runtime prepares once per `(mode, bucket)`, and prologue coverage has at most
  10 buckets per mode. Lazy staging memory is roughly `18,432 B × bucket_m` per
  dense/reuse bucket and `17,408 B × bucket_m` per indexer bucket, so all ten
  buckets × three modes stay under ~140 MiB (typically a few MiB).
- Outputs are written in place through a staging→caller copy on the call stream;
  captures/replays are fine, but a captured graph bakes the per-call row count,
  so capture per `(bucket, real M)` with an eager warm-up per bucket first.
  The optional `native_stage` continuation receives the runtime staging rows
  (real rows first), not the caller buffers.

## Tracked benchmark config entries (opt-in, existing runs unchanged)

- `.github/benchmark/oot_benchmark_models.json`: a new MiniMax-M3-MXFP4 variant
  (`display`: `MiniMax-M3-MXFP4 SP4 16K Route-A`, prefix
  `minimax-m3-mxfp4-sp4-16k-route-a`) with SP4, the existing gemm-fusion path,
  65536 limits and FP8 KV. The workflow selects variants by a fixed `display`
  allow-list (`atom-vllm-benchmark.yaml`), so the new display is inert until it
  is added there; the existing `MiniMax-M3-MXFP4 TP4` entry is unchanged.
- `.github/benchmark/models_atomesh.yaml`: a new `route_a_16k` suite
  (aggregated, SP4) under MiniMax-M3-MXFP4 with the same 65536 limits and the
  existing path. Existing suite names (`nightly` / `weekly` / `smoke` /
  `accuracy`) are untouched, so no scheduled cell changes.
- When Phase-2 lands, both entries flip to `--aiter-provider-enable ... +
  --enable-sp-qkv-fused` (and drop `--gemm-fusion-profile`); the 65536 recipe is
  already in place.

## GPU validation list (E1c / E6a style)

| ID | Needs | Check | PASS |
|---|---|---|---|
| RA0 | CPU | `pytest tests/test_aiter_provider.py tests/model_ops/test_sp_qkv_fused.py tests/model_ops/test_gemm_fusion_route_a.py` | enable mapping/fail-open/hash pass; bind matrix refuses every unsupported/de- covered cell with the existing path; supported fake op receives the 15 positional tensors + kwargs; flag-off forward untouched; r19 tag widths unchanged |
| RA1 | CPU + aiter route-A tree | provider tests against the real aiter modules | `aiter.plugin` checks match `API_VERSION`; routed-op/coverage warnings when absent; ATOM never calls configure() on an empty enable list |
| RA2 | GPU, no model | `torchrun --nproc-per-node=4 tests/distributed/check_ulysses_exchange.py` | PASS bitwise; script covers totals up to 131072 (superset of a 65536 forward) |
| RA3 | GPU, no model | `ATOM_CHECK_MXFP4_TOKENS=16,8192,32768,65536 torchrun --nproc-per-node=4 tests/distributed/check_ulysses_mxfp4.py` | PASS at 65536: BF16-gather fallback byte-exact (or prequant branch byte-exact when eligible), MoE/reduce-scatter/graph replay unchanged |
| RA4 | GPU, no model | route-A profile prebuild + `sp_qkv_fused` import smoke on gfx950 | op importable, `sp_qkv_covers` present, profile 147 rows / 24 extended r19 rows, arena 788 MiB, prebuild rc=0, no OOM |
| RA5 | GPU + M3 checkpoint | 16K recipe e2e greedy T in {32896, 40000, 65536}, existing path on/off | token-identical outputs and no tag refusal; proves the band is carried by the existing path in this PR |
| RA6 | GPU + M3 checkpoint | prologue-band run (m <= 256) with provider on, eager | `aiter.plugin.decision_counts()` shows `("sp_qkv_fused", "provider", "selected", <provider>)`; ATOM output matches the native composition |
| RA7 | GPU + M3 checkpoint | capture on/off with provider on; kill switch `AITER_PLUGINS=0` and flag off | capture succeeds and replays with the provider kernel; the free paths record no Decision (`last_decision()` is None) and accuracy is unchanged |
| RA8 | GPU + M3 checkpoint | multi-rank TP/DP/disagg where used | all ranks agree on route/coverage; no rank splits the step |

RA5..RA8 are **NOT RUN** until the M3-MXFP4 checkpoint and a provider image are
available. RA0/RA1 run on CPU now; RA2/RA3/RA4 need only the aiter build on a
gfx950 node.

### Unverified scope

- The provider interface (`aiter.plugin`) is a stub in CPU tests; the real
  interface/first-live-call path is unverified until that stack lands in a
  worker image.
- The runtime/op are CPU-verified only on the aiter side. Real CCO/arena/ERR-page
  bring-up, the new staging/copy path, compiled-kernel runs, CUDAGraph
  capture/replay and byte-exactness against the #6689 golden cells remain
  GPU-unverified; ATOM's GPU evidence is likewise outstanding (RA5..RA8).
- Phase-2 (large/16K band on the neutral op) is not implemented on either side;
  the 16K recipe in this PR runs the existing ATOM path end-to-end.

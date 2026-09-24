# DeepSeek-V4 FP8 Attention Tuning Notes

This document records the production routing and measured tuning results for
DeepSeek-V4 native two-buffer FP8 attention on gfx950.

Last updated: 2026-09-24.

## Final production policy

Native-FP8 decode no longer has a Triton implementation in this change. The
production policy is:

| Workload | Backend | Status |
|---|---|---|
| H128/q7 CSA B1-B32 | FlyDSL | Enabled by default on gfx950 |
| H128/q7 HCA B6 | FlyDSL | Enabled by default on gfx950 |
| Other native-FP8 decode shapes | AITER/ASM | Fallback |
| gfx1250 H128 decode eligible for prefill ASM reuse | AITER prefill ASM | Existing fallback preserved |
| H128 native-FP8 prefill, no sentinel, Q <= 127 and K < 4096 | FlyDSL | Opt-in with `ATOM_V4_FLYDSL_FP8_PREFILL=1` |
| Other native-FP8 prefill shapes | AITER OPUS | Fallback |

The removed Triton decode results remain useful only as a historical tuning
baseline. There is no Triton native-FP8 decode fallback after this change.

## Backend controls

| Variable | Default | Effect |
|---|---:|---|
| `ATOM_USE_TRITON_ATTN` | `1` | Compatibility master switch for custom attention kernels. For native-FP8 decode, `0` forces AITER/ASM. Legacy Triton attention paths outside this native-FP8 work still use the same variable. |
| `ATOM_V4_FLYDSL_FP8_DECODE` | `1` | Enables qualified gfx950 H128/q7 FlyDSL decode. `0` forces AITER/ASM. |
| `ATOM_V4_FLYDSL_FP8_PREFILL` | `0` | Enables qualified gfx950 H128 FlyDSL prefill. |
| `ATOM_FORCE_V4_PREFILL_OPUS` | `0` | Forces native-FP8 prefill to AITER OPUS. |

The obsolete native-FP8-only controls `ATOM_V4_TRITON_HYBRID_DECODE` and
`ATOM_V4_TRITON_NATIVE_BF16_V` were removed.

## Decode schedules

The FlyDSL dispatcher uses only capture-time shapes and GPU-resident
`kv_indptr`. It does not read GPU data on the host and does not derive the live
K length from the capacity of `kv_indices`, so CUDA Graph replay remains safe.

```text
CSA B1:
  K16, cap15, long5, mid3<=40, short2<=24,
  head_group1, weu1, machine_sink, fixed_reduce15

CSA B2-B3:
  K16, cap8, long9, mid5<=40, short3<=24,
  head_group1, weu1, machine_sink, fixed_reduce8

CSA B4:
  K16, cap6, long12, mid7<=40, short4<=24,
  head_group1, weu1, machine_sink, fixed_reduce6

CSA B5-B6:
  K32, cap6, long6, short3<=20, head_group8, weu0

CSA B7-B8:
  K32, cap6, long6, short4<=12, head_group8, weu1

CSA B9-B12:
  K32, cap3, long8, short6<=12, head_group8, weu1

CSA B13-B16:
  K32, cap2, long9, short6<=12, head_group8, weu1

CSA B17-B18:
  K32, cap2, long10, short6<=12, head_group8, weu1

CSA B19-B32:
  K32, cap2, long9, short6<=12, head_group8, weu1

HCA B6:
  K32, cap13, long14, mid11<=64, short10<=50,
  head_group8, weu1
```

For the fixed reducers used at CSA B1-B4, stage 1 writes zero partials for
inactive splits. Reducing the fixed cap is therefore correct for every live
length inside the qualified schedule.

## Decode performance summary

All percentages use the throughput-style ratio `reference_us / candidate_us -
1`. Positive values mean FlyDSL is faster. Every promoted matrix used CUDA
Graph replay and finite-output/cosine checks.

| Scope | FlyDSL result | Historical Triton baseline | Decision |
|---|---:|---:|---|
| CSA B1, K384/640/1152, two seeds | 6/6 wins | 6 points | FlyDSL |
| CSA B2-B3, K384/640/1152, two seeds | 12/12 wins | 12 points | FlyDSL |
| CSA B4, K640/K1152 | Faster | Retired Triton auto | FlyDSL |
| CSA B4, K384 | 32.32 us | 32.00 us | FlyDSL accepted at about 0.99% latency regression |
| CSA B5-B32, K384/640/1152, two seeds | 168/168 wins; +0.56% to +43.85%, +18.91% average | Retired Triton auto | FlyDSL |
| HCA B6, eight heterogeneous vectors, two seeds | 16/16 wins; +1.42% to +19.99%, +8.01% average | Retired tuned Triton | FlyDSL |

The B4/K384 result is the only accepted point that does not beat the retired
Triton baseline. The approximately 0.3 us difference is intentionally traded
for one implementation and one production policy across CSA B1-B32.

Correctness from the promoted matrices:

- CSA B5-B32: minimum cosine `0.99999595`, maximum relative RMSE
  `0.00282215`, maximum absolute error `0.0078125`, no NaN or Inf.
- HCA B6: minimum cosine `0.9999961`, no NaN or Inf.
- Fixed-grid replay with live K384/K640/K1152 changes: minimum cosine
  `0.99999630`, no NaN or Inf.

HCA remains deliberately narrow. Uniform HCA points at other batch sizes can
look faster with fixed splits, but heterogeneous vectors regressed by about
6%-21%. Those shapes continue to use AITER.

## Prefill policy and results

The retained FlyDSL prefill route is narrower than decode. Production uses
FlyDSL only for H128, no-sentinel inputs with `max_seqlen_q <= 127` and
`max_seqlen_k < 4096`. The boundary is host-visible and CUDA Graph safe.

| Shape | AITER OPUS | FlyDSL | Result | Production |
|---|---:|---:|---:|---|
| T2048/P1151/E127 mixed | 486.56 us | 435.58 us | +11.70% | FlyDSL |
| T4096/P1151/E127 mixed | 901.45 us | 775.21 us | +16.28% | FlyDSL |
| T2048/P2047/E127 mixed | 673.88 us | 638.32 us | +5.57% | FlyDSL |
| T2048/P3071/E127 mixed | 914.13 us | 895.01 us | +2.14% | FlyDSL |
| T2048/P4095/E127 mixed | 1119.61 us | 1139.45 us | 1.77% latency regression | OPUS |
| T2048/P7000/E127 mixed | 1688.17 us | 1807.25 us | 7.05% latency regression | OPUS |
| T2048/P8192/E127 mixed, seed 31 | 1854.93 us | 1871.81 us | 0.91% latency regression | OPUS |
| T2048/P8192/E127 mixed, seed 47 | 1854.89 us | 1873.81 us | 1.02% latency regression | OPUS |
| T2048/P1151/E2048 causal | 1462.77 us | 1533.97 us | 4.87% latency regression | OPUS |
| T3072/P1151/E3072 causal | 2750.78 us | 2968.46 us | 7.91% latency regression | OPUS |

The earlier Triton prefill experiment was removed: it measured
577.06-577.74 us versus OPUS 464.48-464.86 us at P1151/E127, and
2486.29-2489.97 us versus OPUS 1849.35-1849.83 us at P8192/E127.

## Rejected decode experiments

| Experiment | Result |
|---|---|
| Branchless tail mask | About 0.62% slower |
| 32-lane-per-head reducer | No stable benefit |
| q7-fused K16 stage 1 | Output became non-finite; removed |
| Balanced q7 K16 stage 1 | Output became non-finite; removed |
| Partial-L parallel reduction | Slower |
| q7 K32 for all low batches | K384 was about 2.5% faster, but K640/K1152 regressed about 24% |
| Fixed splits for all HCA inputs | Fast on uniform inputs, unsafe for heterogeneous performance |
| Separate GPU planning kernel | Added about 5 us launch overhead |
| Persistent worker loop | Regressed K192/K384/K1152 by about 10.6%/13.6%/20.4% |

## Source layout

- `atom/model_ops/v4_kernels/paged_decode.py`: production dispatch and AITER
  fallbacks.
- `atom/model_ops/v4_kernels/paged_decode_fp8_flydsl.py`: native-FP8 FlyDSL
  decode stage 1, reducer, and schedule selection.
- `atom/model_ops/v4_kernels/paged_prefill.py`: production prefill dispatch.
- `atom/model_ops/v4_kernels/paged_prefill_fp8_flydsl.py`: FlyDSL prefill and
  decode stage-1 building blocks.
- `scripts/performance/bench_v4_fp8_flydsl_prefill.py`: retained prefill
  benchmark.

The removed native-FP8 Triton decode source, its direct tests, and its four
dedicated benchmark/tuning scripts are intentionally not part of the final
tree.

## Validation

Static and dispatch checks:

```bash
python3 -m py_compile \
  atom/model_ops/v4_kernels/paged_decode.py \
  atom/model_ops/v4_kernels/paged_decode_fp8_flydsl.py \
  atom/model_ops/v4_kernels/paged_prefill.py \
  atom/model_ops/v4_kernels/paged_prefill_fp8_flydsl.py \
  scripts/performance/bench_v4_fp8_flydsl_prefill.py

git diff --check
pytest -q \
  tests/test_paged_attention_dispatch.py \
  tests/test_v4_prefill_asm_decode_dispatch.py
```

Before release, repeat the promoted operator matrix on gfx950 and run the
matched 900-second AgentX gate. The older 500-second end-to-end result used a
Triton/FlyDSL hybrid and must not be presented as proof for this FlyDSL-only
dispatcher.

## Qualification rules

1. Require finite output and report cosine, maximum absolute error, and
   relative RMSE where available.
2. Use same-process interleaving or ABBA order; use two seeds when the margin
   is below 2%.
3. Cover heterogeneous HCA vectors, not only uniform lengths.
4. Keep dispatch decisions host-shape-only or GPU-resident; no GPU-to-host
   synchronization is allowed in capture.
5. Preserve the gfx1250 prefill-ASM decode fallback.
6. Do not widen an envelope based on one operator point or one short
   end-to-end run.

## Artifact index

The latest low-batch decode tuning artifacts are under:

```text
runs/v4-fp8-flydsl-q7-fused-20260924/
```

The broader FlyDSL decode and prefill qualification artifacts are under:

```text
runs/v4-fp8-flydsl-e127-20260923/
runs/v4-fp8-flydsl-cluster-20260922/
```

# DeepSeek-V4 FP8 Attention Tuning Notes

This document records the current Triton and FlyDSL tuning state for
DeepSeek-V4 native two-buffer FP8 attention on gfx950. It separates production
routing from benchmark-only candidates and records the workload definitions
needed to reproduce each result.

Last updated: 2026-09-23.

## Scope

The work covered four related paths:

1. Native two-buffer FP8 paged decode implemented in Triton.
2. Native two-buffer FP8 sparse prefill implemented in Triton.
3. An H=128 sparse-prefill kernel implemented in FlyDSL.
4. Benchmark-only H=128 FlyDSL split1 and compact split-K decode paths.

The current result is not an unconditional replacement for every AITER
attention kernel:

- Decode uses a production hybrid policy on gfx950.
- Triton FP8 prefill is opt-in and still trails AITER OPUS at long context.
- FlyDSL prefill is production-routed on gfx950 only for H=128, no-sentinel
  native FP8 inputs with `max_seqlen_q <= 127` and `max_seqlen_k < 4096`.
  Longer extend tails and total K lengths at or above 4096 stay on AITER OPUS.
  The post-ABI-fix production dispatch beats OPUS by 11.70% at
  T2048/P1151/E127; the P4095/E127 and P8192/E127 fallback outputs are
  bit-identical to direct OPUS.
- FlyDSL decode is benchmark-only. Split1 cannot replace the complete Triton
  CSA matrix. The host-planned compact split-K prototype beats both the current
  packed Triton candidate and AITER on all eight HCA B6 heterogeneous vectors,
  but its GPU-resident graph-safe successor wins only 7/8. Production therefore
  retains the 8/8-winning Triton HCA B6 path.

## Status summary

| Path | Workload | Candidate | AITER/OPUS | Result | Routing state |
|---|---|---:|---:|---:|---|
| Triton decode | CSA B10-B16, two seeds, K=384/640/1152 | 42/42 wins | 42 reference points | All positive | Production hybrid |
| Triton decode | CSA B12, K=1152 | 86.241 us | 88.760 us | +2.92% | Production hybrid |
| Triton decode | HCA B6 heterogeneous vectors | All 8 vectors win | AITER decode | About +4% to +43% | Native-V opt-in |
| Triton prefill | T=2048, P=1151, E=127, mixed | 577.06-577.74 us | 464.48-464.86 us | Candidate latency about 24.2% higher | Opt-in |
| Triton prefill | T=2048, P=8192, E=127, mixed | 2486.29-2489.97 us | 1849.35-1849.83 us | Candidate latency about 34.4% higher | Opt-in |
| FlyDSL prefill | T=2048, P=1151, E=127, mixed | 402.92 us | OPUS 464.48 us | +15.28% throughput-style speedup | Benchmark-only |
| FlyDSL prefill | T=2048, P=8192, E=127, mixed, two seeds | 1871.81-1873.81 us | OPUS 1854.89-1854.93 us | Latency 0.91%-1.02% higher | Benchmark-only |
| Production prefill dispatch | T=2048, P=1151, E=127, mixed | FlyDSL 435.58 us | OPUS 486.56 us | +11.70% | Production for Q <= 127 and K < 4096 |
| FlyDSL prefill | T=2048, P=3071, E=127, mixed | 895.01 us | OPUS 914.13 us | +2.14% | Inside production envelope |
| FlyDSL prefill | T=2048, P=4095, E=127, mixed | 1139.45 us | OPUS 1119.61 us | Candidate latency 1.77% higher | Rejected; production OPUS fallback |
| FlyDSL prefill | T=2048, P=7000, E=127, mixed | 1807.25 us | OPUS 1688.17 us | Candidate latency 7.05% higher | Rejected; production OPUS fallback |
| FlyDSL prefill | T=2048, P=1151, E=2048, causal | 1533.97 us | OPUS 1462.77 us | Candidate latency 4.87% higher | Rejected; production OPUS fallback |
| FlyDSL prefill | T=3072, P=1151, E=3072, causal | 2968.46 us | OPUS 2750.78 us | Candidate latency 7.91% higher | Rejected; production OPUS fallback |
| Production prefill dispatch | T=2048, P=4095, E=127, mixed | OPUS fallback 1094.93 us | Direct OPUS 1091.45 us | Bit-identical; 0.32% timing noise | Production OPUS fallback |
| Production prefill dispatch | T=2048, P=8192, E=127, mixed | OPUS fallback 1857.45 us | Direct OPUS 1854.31 us | Bit-identical; 0.17% timing noise | Production OPUS fallback |
| FlyDSL prefill | T=8, P=32, E=32, uniform | 34.06 us | 42.24 us | +24.02% | Benchmark-only |
| FlyDSL prefill | T=2048, P=1151, E=2048, mixed | 830.18 us | 831.84 us | +0.20% | Benchmark-only |
| FlyDSL prefill | T=64, P=1152, E=2048, uniform | 268.12 us | 265.84 us | -0.85% | Benchmark-only |
| FlyDSL decode split1 | CSA B10-B16, two seeds, K=384/640/1152 | 40/42 wins vs AITER; 15/42 wins vs Triton auto | 42 reference points | Median 2.88% behind Triton auto | Benchmark-only |
| FlyDSL decode split1 | HCA B6 heterogeneous vectors | 0/8 wins | AITER and tuned Triton | 58.36%-73.21% behind AITER | Rejected for HCA |
| FlyDSL decode split-K | HCA B6, eight heterogeneous vectors | 49.92-94.88 us | Packed Triton 56.54-112.50 us; AITER 61.18-127.64 us | 8/8 wins; +5.36% to +52.78% vs Triton | Benchmark-only |
| FlyDSL decode graph-safe split-K | HCA B6, eight heterogeneous vectors | 7/8 wins | AITER decode | Worst regression 1.59% at cap10 and 3.01% at cap12 | Not promoted |

In the benchmark JSON, `delta_pct` is computed from the throughput-style ratio
`reference_us / candidate_us - 1`. A negative value therefore does not equal
the candidate's raw latency overhead. Both values should be reported when the
difference is material.

The new E127 FlyDSL rows are directly comparable with the Triton production
workload. The older E2048 FlyDSL rows remain separate workloads and must not be
compared directly with E127 Triton numbers.

## Backend controls

| Variable | Default | Effect |
|---|---:|---|
| `ATOM_USE_TRITON_ATTN` | `1` | Master switch for the native Triton attention paths. `0` permits AITER fallback. |
| `ATOM_V4_TRITON_HYBRID_DECODE` | `1` | On gfx950, route only measured decode winners to Triton. `0` forces all supported native-FP8 decode shapes to Triton. |
| `ATOM_V4_TRITON_NATIVE_BF16_V` | `0` | Enables the experimental gfx950 native FP8-to-BF16 V path for eligible HCA decode shapes. It also enables the tuned B6 HCA specialization. |
| `ATOM_V4_TRITON_FP8_PREFILL` | `0` | Enables the gfx950 native-FP8 Triton prefill candidate. Unset keeps AITER OPUS. |
| `ATOM_V4_FLYDSL_FP8_DECODE` | `0` | Enables the experimental gfx950 FlyDSL decode path. It remains off in production and in the matched E2E run. |
| `ATOM_V4_FLYDSL_FP8_PREFILL` | `0` | Enables the gfx950 H=128 FlyDSL prefill route for eligible no-sentinel inputs with `max_seqlen_q <= 127` and `max_seqlen_k < 4096`; all other inputs use OPUS. |
| `ATOM_FORCE_V4_PREFILL_OPUS` | `0` | Forces the native-FP8 prefill path back to OPUS. |

Relevant dispatch implementations:

- `atom/model_ops/v4_kernels/paged_decode.py`
- `atom/model_ops/v4_kernels/paged_decode_fp8_triton.py`
- `atom/model_ops/v4_kernels/paged_decode_fp8_flydsl.py`
- `atom/model_ops/v4_kernels/paged_prefill.py`
- `atom/model_ops/v4_kernels/paged_prefill_fp8_triton.py`
- `atom/model_ops/v4_kernels/paged_prefill_fp8_flydsl.py`

## Triton FP8 decode

### Retained implementation work

The decode path directly consumes the V4 two-buffer layout:

- `[P, 512]` FP8 NoPE data with embedded E8M0 scales.
- `[P, 64]` BF16 RoPE data.
- Pre-packed FP8 query data without a second quantization pass.

The retained kernel and scheduling optimizations are:

| Optimization | Purpose | Current use |
|---|---|---|
| Native `v_cvt_scalef32_pk_bf16_fp8` conversion | Convert packed FP8 V to BF16 using gfx950 instructions instead of generic software expansion | Automatic for validated CSA shapes; opt-in for HCA |
| Q4/Q7 query fusion | Reuse one KV traversal across adjacent verification queries | Low-batch and DSpark paths |
| Two q4 stripes for q7 | Avoid the register footprint of one monolithic q7 program | DSpark at larger active batch |
| Per-shape `BLOCK_K`, split-K and stage selection | Keep enough CTAs for short batches without over-splitting larger batches | Auto dispatcher |
| Packed active-task map | Compact runtime-active query/head/split tasks at the front of a fixed graph-safe grid | HCA B6 specialization |
| Adaptive segment sizes | Select split size from the GPU-resident request KV length | HCA B6 specialization |
| Early inactive-CTA return | Avoid wide Q and scale loads for inactive packed tasks | Packed decode path |
| Dedicated split-2 and split-3 reducers | Remove dynamic vector reduction overhead from common split counts | Split-K decode |
| Low-register sequential reducer | Reduce scalar softmax metadata first, then stream one D vector per split | Retained as an experimental option |
| FP16 partial accumulators | Reduce partial-buffer traffic while maintaining measured accuracy | Tuned native-FP8 paths |
| `schedule_hint="attention"` | Improve gfx950 MFMA and wait scheduling | Validated CSA native-V hotspot |

### CSA routing

For gfx950 H=128 q7 CSA, the tuned native-V schedule uses a regular qh64
kernel. The relevant high-batch policy is:

```text
B10-B12: BH64 / BK64 / split3 / stage2 / MI16 / native V / attention hint
B13+:    BH64 / BK64 / split1 / stage2 / MI16 / native V / attention hint
```

Robustness coverage used:

```text
batches: B10-B16
KV rows: 384, 640, 1152
seeds:   20260917, 20260918
total:   42 points
```

All 42 points beat AITER. For B12/K1152, the production auto path measured
86.241 us versus 88.760 us for AITER, a 2.92% speedup.

### FlyDSL decode split1 experiment

The first true FlyDSL decode candidate reuses the H=128 FlyDSL attention data
flow with the decode KV list as its only segment. A `prefix_only` compile-time
specialization removes the unused extend segment. This remains isolated from
production dispatch and has no split-K reducer or q7 query sharing.

CUDA-Graph replay coverage used the same 42-point CSA matrix as the Triton
robustness run, with 50 samples per implementation per point:

| Comparison | Wins | Median speedup | Range |
|---|---:|---:|---:|
| FlyDSL split1 vs AITER | 40/42 | +83.60% | -14.31% to +318.47% |
| FlyDSL split1 vs Triton auto | 15/42 | -2.88% | -19.61% to +11.98% |

The raw 15/42 count includes five sub-1% differences. The repeatable useful
wins are concentrated in B10-B12/K384 (+6.1% to +12.0% versus Triton auto) and
B11-B12/K640 (+3.3% to +4.5%). K1152 loses to Triton auto in all 14 points by
6.44% to 19.61%. B12/K1152 is also the only AITER regression, repeated across
both seeds at 15.19% and 16.70% higher latency.

Correctness across all 42 points: minimum cosine `0.99999624`, maximum relative
RMSE `0.00275328`, maximum absolute error `0.00390625`, and zero non-finite
outputs. The `prefix_only` specialization improved FlyDSL latency by a median
1.52% over the initial empty-extend adapter.

This establishes that FlyDSL can replace selected CSA entries, but not the
complete Triton CSA policy. The next structural requirement is FlyDSL split-K
plus reduction for the long-K B10-B12 cases; B13+ also needs a smaller/lower-LDS
single-query kernel to beat the existing qh64 Triton path.

### FlyDSL HCA compact split-K experiment

The HCA follow-up adds a benchmark-only compact active-task schedule around the
same H128 FlyDSL attention data flow:

- K is processed in K32 tiles with at most 16 active splits per query.
- Requests with at most 64 K32 tiles use nine tiles per split; longer requests
  use fourteen tiles per split.
- The host creates compact `(query, start, length, partial-row)` arrays so stage
  1 launches only active work.
- Stage 1 writes FP32 `m/l` metadata and FP16 accumulator partials.
- The final reducer is also FlyDSL: one wave per head, four heads per CTA, eight
  D elements accumulated per lane, with the normalization scale broadcast by
  `ds_bpermute` inside each wave.

The final CUDA-Graph replay run used the same eight heterogeneous B6 vectors,
ten warmups and 50 measured iterations:

| Vector | Packed Triton | FlyDSL stage1 + Triton reducer | Pure FlyDSL split-K | Pure vs Triton | Pure vs hybrid |
|---:|---:|---:|---:|---:|---:|
| 1 | 58.08 us | 51.08 us | 49.92 us | +16.35% | +2.32% |
| 2 | 102.76 us | 68.42 us | 67.26 us | +52.78% | +1.73% |
| 3 | 66.18 us | 63.32 us | 61.92 us | +6.88% | +2.26% |
| 4 | 112.50 us | 96.40 us | 94.88 us | +18.57% | +1.60% |
| 5 | 103.90 us | 84.76 us | 82.64 us | +25.73% | +2.57% |
| 6 | 56.54 us | 50.84 us | 50.34 us | +12.32% | +0.99% |
| 7 | 66.04 us | 63.92 us | 62.68 us | +5.36% | +1.98% |
| 8 | 66.18 us | 64.00 us | 62.08 us | +6.61% | +3.09% |

The pure FlyDSL path wins 8/8 versus the current packed Triton candidate and
8/8 versus AITER. Its speedup range is 5.36%-52.78% versus Triton and
21.53%-89.77% versus AITER. Correctness across the eight vectors has minimum
cosine `0.99999619`, maximum relative RMSE `0.00273735`, maximum absolute error
`0.00390625`, and zero non-finite values.

This result removes the Triton reducer from the best HCA prototype, but it is
not production-ready. The compact task map is still prepared on the host, the
launch grid is sized to the active task count rather than a graph-safe maximum,
and sentinel/empty/ragged coverage plus a matched 900-second AgentX run remain
outstanding.

The production-integration follow-up moved planning onto the GPU and used a
fixed graph-safe maximum grid. With planner caps 10 and 12 it was correct but
won only seven of the eight heterogeneous vectors. The losing vector was 1.59%
slower than AITER at cap10 and 3.01% slower at cap12. Since this no longer meets
the 8/8 promotion gate, `ATOM_V4_FLYDSL_FP8_DECODE=0` is used for E2E and the
existing tuned Triton HCA B6 specialization remains active.

### HCA routing boundary

HCA requires a conservative hybrid policy because uniform and heterogeneous
requests prefer different work decompositions.

- Fixed-split B3, B9 and B12 configurations win uniform benchmarks.
- The same configurations regress heterogeneous vectors by approximately
  6% to 21% and are not production-safe.
- The adaptive B6 packed path wins all eight tested heterogeneous vectors.
- The compact FlyDSL split-K prototype is faster on all eight vectors, but its
  host planner and active-sized grid are not valid production routing yet.
- More aggressive native-V routing produced favorable 360-second results but
  later regressed ITL in a matched 900-second C96 run.

Therefore, with the default gfx950 hybrid policy, H=128 q7 HCA normally falls
back to AITER. The B6 Triton specialization additionally requires
`ATOM_V4_TRITON_NATIVE_BF16_V=1`.

For the matched C96 E2E comparison in this document, native BF16 V is enabled,
so the validated Triton B6 specialization is used while FlyDSL decode remains
disabled.

The FlyDSL split1 adapter was also tested on the same eight B6 heterogeneous
vectors with CUDA Graph replay. It was correct (minimum cosine `0.99999619`,
maximum relative RMSE `0.00273328`, maximum absolute error `0.00390625`, zero
non-finite outputs), but lost all eight vectors: 58.36%-73.21% behind AITER and
61.08%-78.95% behind the tuned packed/adaptive Triton path. One CTA per query
serializes each request's full variable-length KV range, so this result confirms
that HCA replacement requires split scheduling plus packed active-task
compaction. The new split-K prototype supplies both and supersedes split1 for
HCA tuning, but it must not be routed until planning and graph-safety are fixed.

### Decode experiments not promoted

| Experiment | Result |
|---|---|
| Fixed split for all HCA inputs | Fast on uniform inputs; regressed heterogeneous inputs |
| Uniform-three-split detection | Improved uniform long-K, but remained slightly behind the current production Triton path on most heterogeneous vectors |
| Persistent worker loop | Regressed K192/K384/K1152 by about 10.6%/13.6%/20.4% versus AITER |
| Request-local scattered mapping | Regressed K192 by 2.6% and K1152 by 33.9% |
| Separate GPU pre-plan kernel | Added about 5 us of launch overhead |
| QK scale lifetime extension | Increased register pressure and regressed latency |
| Direct FP8 V dot path | Materially slower than native FP8-to-BF16 conversion |
| Broad native-V HCA promotion | Rejected after the matched 900-second ITL regression |

## Triton FP8 prefill

### Retained dispatch configuration

The current large-head, large-token candidate uses:

```text
BLOCK_H                  = 32
BLOCK_K                  = 32
num_warps                = 2
prefix num_stages        = 3
extend num_stages        = 1 when T >= 1024
waves_per_eu             = 0
matrix_instr_nonkdim     = 16
schedule_hint            = attention
no_sentinel_hot_loop     = true when the caller proves no sentinel
tail_block_k             = 32
full_bf16_v              = true
grid_group_tokens        = 8 when T is divisible by 8
compiled_launch          = true
```

The implementation directly consumes prefix and extend FP8/RoPE buffers and
updates one online-softmax state across both CSR segments.

The two most useful scheduling changes were:

1. Increase the long prefix loop from stage 2 to stage 3.
2. Keep the short extend loop at stage 1 for T >= 1024.

The G8 grid groups eight tokens at a time and dispatches their head CTAs close
together. This improves cache reuse without fully serializing all four BH32
head CTAs belonging to one token.

### Performance progression

These results use `tokens=2048`, `heads=128`, `extend_len=127`, mixed prefix
lengths, no injected sentinel, and ABBA40 timing.

| Shape | Initial Triton | S3/G8 Triton | Final S3-prefix/S1-extend Triton | OPUS | Total Triton improvement |
|---|---:|---:|---:|---:|---:|
| P1151 | 638.76 us | 617.90 us | 577.06-577.74 us | 464.48-464.86 us | About 9.6% |
| P8192 | 2907.54 us | 2508.59 us | 2486.29-2489.97 us | 1849.35-1849.83 us | About 14.4% |

Correctness for the final runs:

| Shape | Cosine | Max absolute error | Non-finite values |
|---|---:|---:|---:|
| P1151 | 1.0 | 0.0009765625 | 0 |
| P8192 | 1.0 | 0.00048828125 | 0 |

### Counter diagnosis

| Counter | P1151 Triton/OPUS | P8192 Triton/OPUS |
|---|---:|---:|
| Total instructions | 1.46x | 1.59x |
| VMEM read instructions | 6.67x | 7.09x |
| L2 requests | 2.04x | 2.75x |
| Occupancy | 10.66% / 15.78% | 10.55% / 17.20% |

The remaining gap is structural. Triton uses four BH32 CTAs for one H=128
token, so each CTA loads and dequantizes the same KV rows. OPUS uses one
eight-wave workgroup, stages KV in LDS once, and shares it across all heads.
Cache hints can improve hit rate but cannot remove the duplicated instructions.

### Triton prefill experiments not retained

The following paths were measured and removed from the active candidate:

- BH64 and BH128 ordinary Triton kernels.
- BH32 with four or eight warps.
- Stage 4 and stage 3 plus async copy.
- `memory-bound-attention` and combined scheduling hints.
- `waves_per_eu=1`, `.ca` cache hints and explicit keep-KV variants.
- Prefix BK32 plus extend BK64.
- Adaptive K16/K32 tail selection.
- Compact active-row worklists.
- Precomputed packed-row pointers.
- Reusing QK-loaded KV directly for PV; this pushed VGPR use close to 512.
- BF16 loop-carried accumulator and large-head spill-reduction experiments.

## FlyDSL FP8 prefill

### Current best configuration

The current FlyDSL kernel is limited to H=128. Its narrowly qualified
short-extend/short-context envelope is production-routed behind an opt-in flag;
all other configurations in this section remain benchmark-only. The best
retained configuration is:

```text
BH128 / BK32 / W8 / S2 / WEU1
pipeline_two
wave_padded_k
alpha_bpermute
lds_padding=4
fixed_softmax_ref
transpose_v
reuse_q_across_n
no_sentinel
full_tile_fastpath
cache_all_q
permute_k_scales
pairwise_pv
dynamic_full_prefix for mixed prefix lengths
assume_full_tiles only for exact-K32 uniform inputs
```

### Retained optimizations

| Optimization | Effect |
|---|---|
| One H128 workgroup per token | Removes the four-CTA KV duplication of the BH32 Triton path |
| `pipeline_two` | Double-buffers raw K/RoPE and BF16 P/V so current QK overlaps previous-tile PV |
| Per-wave 32-byte K padding | Reduces harmful LDS bank aliasing between wave-owned K regions |
| `lds_padding=4` | Improves the P/V LDS stride |
| Fixed-reference softmax | Simplifies the softmax update and rescale path |
| Transposed V reads | Uses `ds_read_b64_tr_b16` for the PV operand |
| Reuse Q across N halves | Avoids repeated Q preparation for the two K16 halves |
| Cache all Q | Keeps the complete query operand available through the KV loop |
| No-sentinel path | Removes per-entry sentinel tests when the input contract permits it |
| Full-tile fast path | Removes tail work from complete K32 tiles |
| Exact-full specialization | Removes all per-tile bounds checks for uniform K32-multiple inputs |
| Permuted K scales | Replaces four groups of shift/and/or packing with two gfx950 `v_perm_b32` instructions |
| Pairwise PV | Pipelines two adjacent D16 fragments as one D32 group using four transpose reads and two MFMAs |
| `alpha_bpermute` | Broadcasts final per-head normalization without an LDS round trip |
| `dynamic_full_prefix` | Selects the exact-K32 prefix loop per query while retaining a correct tail path for mixed lengths |
| `waves_per_eu=1` | Best measured compiler occupancy hint for the final long-context kernel |

`pairwise_pv` requires `transpose_v`.

### Final performance

| Shape | OPUS | FlyDSL | Speedup | Cosine | Max absolute error |
|---|---:|---:|---:|---:|---:|
| T8/P32/E32 uniform | 42.24 us | 34.06 us | +24.02% | 0.99999553 | 0.000244141 |
| T2048/P1151/E2048 mixed, seed 31 | 831.84 us | 830.18 us | +0.20% | 0.99999583 | 0.00195312 |
| T64/P1152/E2048 uniform | 265.84 us | 268.12 us | -0.85% | 0.99999571 | 0.0000610352 |

### Production-shaped E127 follow-up

The same retained FlyDSL configuration was rerun against the exact E127 mixed
workloads used by Triton production dispatch. ABBA80 results were:

| Shape | OPUS p50 | FlyDSL p50 | OPUS/FlyDSL speedup | FlyDSL p10-p90 | FlyDSL latency vs Triton |
|---|---:|---:|---:|---:|---:|
| T2048/P1151/E127, seed 31 | 464.48 us | 402.92 us | +15.28% | See artifact samples | About 30.2% lower |
| T2048/P8192/E127, seed 31 | 1854.93 us | 1871.81 us | -0.90% | See artifact samples | About 24.7% lower |
| T2048/P8192/E127, seed 47 | 1854.89 us | 1873.81 us | -1.01% | See artifact samples | About 24.7% lower |

Both runs had cosine `0.99999589`, maximum absolute error `0.001953125`, and
zero non-finite outputs. The result materially changes the earlier conclusion:
FlyDSL is already a better implementation than Triton for these two prefill
workloads, but P8192 still does not justify replacing OPUS.

### Production dispatch qualification

The production wrapper now receives both `attn_md.max_seqlen_q` and
`attn_md.max_seqlen_k` from the DSV4 CPU metadata path. This keeps the routing
decision host-side and avoids a GPU-to-CPU synchronization. On gfx950, H=128,
no-sentinel native FP8 inputs route as:

```text
max_seqlen_q <= 127 and max_seqlen_k < 4096: FlyDSL
otherwise: AITER OPUS
```

The final boundary comes from the post-pointer-ABI qualification matrix. The
benchmark script calls the FlyDSL candidate `triton` in its JSON schema; the
numbers below are FlyDSL timings. `P` is the maximum prefix length, while the
dispatch's `max_seqlen_k` is the full prefix-plus-extend context, so
P4095/E127 is outside the K < 4096 envelope.

| Shape | Direct OPUS | FlyDSL candidate | OPUS/FlyDSL speedup | Production decision |
|---|---:|---:|---:|---|
| T2048/P1151/E127 mixed | 486.56 us | 435.58 us | +11.70% | FlyDSL |
| T4096/P1151/E127 mixed | 901.45 us | 775.21 us | +16.28% | FlyDSL |
| T2048/P2047/E127 mixed | 673.88 us | 638.32 us | +5.57% | FlyDSL |
| T2048/P3071/E127 mixed | 914.13 us | 895.01 us | +2.14% | FlyDSL |
| T2048/P4095/E127 mixed | 1119.61 us | 1139.45 us | -1.74% | OPUS fallback |
| T2048/P7000/E127 mixed | 1688.17 us | 1807.25 us | -6.59% | OPUS fallback |
| T2048/P1151/E2048 causal | 1462.77 us | 1533.97 us | -4.64% | OPUS fallback |
| T3072/P1151/E3072 causal | 2750.78 us | 2968.46 us | -7.33% | OPUS fallback |

All candidate rows above had cosine similarity from `0.99999583` to
`0.99999696`, maximum absolute error at most `0.001953125`, and zero non-finite
outputs. The negative percentages are the benchmark's throughput-style
`OPUS/FlyDSL - 1` values; the corresponding raw FlyDSL latency overheads are
1.77%, 7.05%, 4.87%, and 7.91%.

Fresh production-dispatch fallback checks were:

| Shape | Direct OPUS | Production dispatch | Result | Correctness |
|---|---:|---:|---:|---:|
| T2048/P4095/E127 mixed | 1091.45 us | OPUS 1094.93 us | Equivalent within run noise | max abs 0 |
| T2048/P8192/E127 mixed | 1854.31 us | OPUS 1857.45 us | Equivalent within run noise | max abs 0 |

Both fallback outputs are bit-identical to direct OPUS. Their 0.32% and 0.17%
timing deltas are treated as measurement noise, not regressions.

### Large-cache FlyDSL pointer ABI fix

The first end-to-end attempt exposed an integration failure that operator-sized
tests did not trigger. The FlyDSL launcher accepted the flattened unified KV
cache as a dynamic `fx.Tensor`, so its generated C ABI encoded the flattened
shape as a signed 32-bit integer. The real DSV4 cache exceeded `2^31 - 1`
elements and Python failed before kernel launch with:

```text
struct.error: 'i' format requires -2147483648 <= number <= 2147483647
```

The sparse-prefill launcher now accepts raw `fx.Pointer` arguments and passes
all tensors with `ptr_arg()`. Typed views are rebuilt inside the kernel with
`ptr_buf_tensor()`, including the byte-wise logical-divide view. This removes
large tensor shapes from the dynamic ABI without changing address arithmetic or
the kernel's data contract. The failure was not a GPU kernel hang: one DP model
runner exited while the HTTP health endpoint remained live, which is why the
client appeared stuck near the end of the trace.

A fresh `FLYDSL_DUMP_IR=1` build of the final P8192 kernel reported 128,256
bytes of group-segment LDS, zero private/scratch bytes, 238 VGPRs, and no VGPR
or SGPR spills. Code-object metadata reports 43 SGPRs and the final ISA reaches
SGPR 95 (`next_free_sgpr=96`). The workgroup has 512 threads. The large LDS
footprint still permits only constrained residency and remains a
production-integration risk even where latency wins.

The real `waves_per_eu` screen measured no hint plus values 1-4. WPE1 was best;
WPE3 and WPE4 requested occupancy the compiler could not satisfy and both
settled at occupancy 2. A two-launch compact partition of K32-aligned versus
tail-prefix queries was also tested as a structural alternative. It remained
correct but measured 1930.45 us versus 1868.95 us for OPUS (-3.19%
throughput-style), so the extra launch and query-map indirection were rejected.

Pairwise PV compared with the preceding `permute_k_scales` build showed:

| Counter/resource | Change |
|---|---:|
| MFMA co-execution cycles | +26.26% |
| `SQ_WAVE_CYCLES` | -0.38% |
| Static `s_waitcnt` count | 200 to 136 |
| VGPR | 128, unchanged |
| Scratch | 0, unchanged |
| LDS bank conflict | Approximately unchanged |
| LDS wait stall | +42.66% |

The remaining difference from OPUS is dominated by layout/address work and
LDS behavior rather than missing MFMA operations:

```text
SQ_INSTS_VALU_INT32: FlyDSL about 1.247M, OPUS about 0.295M
LDS bank conflicts: FlyDSL about 11.57M, OPUS about 4.30M
```

### FlyDSL experiments not retained

| Experiment | Reason for rejection |
|---|---|
| Narrow K-scale `ds_read_u8` | VGPR fell from 128 to 124, but bank conflicts increased 17.7%, LDS wait increased 31.9%, and P1152 regressed to about -5% |
| Early PV accumulator rescale | P1152 regressed to approximately -2.24% |
| `cluster_two` | Did not beat the final `pipeline_two` schedule |
| Register P and P-lane layout | Increased register/layout overhead |
| Pairwise V decode/scale reuse | No stable gain |
| Broadcast indices | No stable gain |
| Q staging | Increased resource pressure or failed to improve latency |
| `setprio` | Reduced some resource counts but was performance-neutral |
| Post-misched and machine-sink controls | No stable gain |
| OPUS packed-K physical layout alone | Did not reproduce the full OPUS schedule |
| Explicit transpose unpack | Generated repeated identity-like `v_bfi` instructions |
| Two-launch aligned/tail prefix partition | Correct, but P8192 measured 1930.45 us versus OPUS 1868.95 us; launch and task-map overhead outweighed specialization |

## DSV4 C96 end-to-end comparison

The matched AgentX comparison used PR #2256 commit
`ab3bd3cadcdf933b689098810d3c6d966fea483d`, DeepSeek-V4-Pro-DSpark, TP8,
DPA8/EP8, concurrency 96, FP8 KV, FP4 index cache, DSpark K6, prefix caching,
and the same public 393-trace seed-42 workload. Each side received a separate
500-second aiperf heat pass before the 500-second measured run. Both measured
runs also completed their built-in 1,061-request trace warm-up with zero
errors.

The AITER control disabled all native Triton/FlyDSL attention switches. The
optimized server used:

```text
ATOM_USE_TRITON_ATTN=1
ATOM_V4_TRITON_HYBRID_DECODE=1
ATOM_V4_TRITON_NATIVE_BF16_V=1
ATOM_V4_FLYDSL_FP8_DECODE=0
ATOM_V4_FLYDSL_FP8_PREFILL=1
ATOM_V4_TRITON_FP8_PREFILL=0
```

| Metric | AITER | Optimized hybrid | Change |
|---|---:|---:|---:|
| Output throughput | 1237.78 tok/s | 1278.71 tok/s | +3.31% |
| Output throughput/GPU | 154.72 tok/s | 159.84 tok/s | +3.31% |
| Input throughput | 173996.74 tok/s | 176776.30 tok/s | +1.60% |
| Total throughput | 175234.52 tok/s | 178055.01 tok/s | +1.61% |
| Successful requests | 930 | 954 | +24 |
| Request errors | 0 | 0 | no change |
| Deadline-drain cancellations | 33 | 30 | -3 |
| Effective concurrency, average | 44.276 | 45.181 | +2.04% |
| TTFT p50 | 8583.51 ms | 8346.93 ms | -2.76% |
| TTFT p90 | 15190.97 ms | 13960.08 ms | -8.10% |
| ITL p50 | 18.327 ms | 18.670 ms | +1.87% latency regression |
| ITL p90 | 28.837 ms | 29.663 ms | +2.86% latency regression |
| E2E latency p50 | 17311.69 ms | 16946.46 ms | -2.11% |
| E2E latency p90 | 48374.51 ms | 47040.83 ms | -2.76% |
| Output throughput/user, average | 57.603 tok/s/user | 56.512 tok/s/user | -1.89% |
| Output throughput/user, p50 | 54.563 tok/s/user | 53.562 tok/s/user | -1.84% |
| E2E throughput/user, average | 24.118 tok/s/user | 23.984 tok/s/user | -0.56% |
| E2E throughput/user, p50 | 23.546 tok/s/user | 22.873 tok/s/user | -2.86% |
| Theoretical prefix-cache hit | 95.860% | 95.627% | -0.233 percentage points |

The hybrid therefore improves aggregate output throughput by 3.31%, input and
total throughput by about 1.6%, and TTFT/E2E latency. It does not fully dominate
AITER: ITL and per-user throughput regress slightly. These 500-second runs used
`--unsafe-override`, so aiperf marks them non-submittable; they are valid as a
same-host matched A/B but do not replace the 900-second promotion gate.

### C96 ITL follow-up: native-V isolation and PDI 20

Two additional C96 runs isolated the two leading explanations for the ITL
regression. Both reused the same server, workload, FP8 KV configuration,
separate 500-second heat pass, and 500-second measured pass as the AITER and
PDI-10 hybrid runs above:

1. `Native-V off`: keep FlyDSL prefill and the hybrid dispatcher, set
   `ATOM_V4_TRITON_NATIVE_BF16_V=0`, and retain
   `ATOM_PREFILL_DECODE_INTERVAL=10`.
2. `Hybrid PDI 20`: retain the complete hybrid attention routing, including
   native BF16-V decode, and change only
   `ATOM_PREFILL_DECODE_INTERVAL=10` to `20`.

| Metric | AITER | Hybrid PDI 10 | Native-V off | Hybrid PDI 20 |
|---|---:|---:|---:|---:|
| Output throughput | 1237.78 tok/s | 1278.71 tok/s | 1264.72 tok/s | 1274.94 tok/s |
| Input throughput | 173996.74 tok/s | 176776.30 tok/s | 177038.43 tok/s | 179686.71 tok/s |
| Total throughput | 175234.52 tok/s | 178055.01 tok/s | 178303.15 tok/s | 180961.65 tok/s |
| ITL p50 | 18.327 ms | 18.670 ms | 18.376 ms | 17.792 ms |
| ITL p90 | 28.837 ms | 29.663 ms | 28.706 ms | 27.461 ms |
| TTFT p50 | 8583.51 ms | 8346.93 ms | 9072.24 ms | 8719.85 ms |
| TTFT p90 | 15190.97 ms | 13960.08 ms | 15054.65 ms | 14870.11 ms |
| Output throughput/user, average | 57.603 tok/s/user | 56.512 tok/s/user | 57.804 tok/s/user | 60.249 tok/s/user |
| Effective decode concurrency, average | 27.226 | 28.343 | 27.719 | 26.766 |
| Successful requests | 930 | 954 | 942 | 948 |
| Deadline-drain cancellations | 33 | 30 | 29 | 35 |
| Request errors | 0 | 0 | 0 | 0 |
| Theoretical prefix-cache hit | 95.860% | 95.627% | 95.742% | 95.916% |
| Admitted server prefix-cache hit | 87.020% | 86.970% | 86.910% | 87.050% |

Relative to AITER, native-V off retains +2.18% output throughput and is nearly
ITL-neutral: p50 is 0.27% slower and p90 is 0.45% faster. Relative to the full
PDI-10 hybrid, it gives back 1.09% output throughput while improving ITL p50 by
1.57% and p90 by 3.23%. It therefore confirms that the native-V route
contributes to the observed scheduling shift, but disabling it is not the best
end-to-end tradeoff.

PDI 20 is the best balanced configuration in this four-way run. Relative to
AITER, it improves output throughput by 3.00%, input and total throughput by
3.27%, ITL p50 by 2.92%, ITL p90 by 4.77%, TTFT p90 by 2.11%, and average
output throughput/user by 4.59%. Relative to PDI-10 hybrid, it gives back only
0.30% output throughput while improving ITL p50 by 4.70%, ITL p90 by 7.42%,
and average output throughput/user by 6.61%. The cost versus PDI-10 hybrid is
TTFT: p50 is 4.47% slower and p90 is 6.52% slower.

The server histogram explains the ITL recovery:

| Decode scheduler metric | AITER | Hybrid PDI 10 | Native-V off | Hybrid PDI 20 |
|---|---:|---:|---:|---:|
| Decode forwards | 51,340 | 50,184 | 51,466 | 53,951 |
| Real decode rows | 216,975 | 223,543 | 218,395 | 224,365 |
| Average real batch | 4.226 | 4.454 | 4.243 | 4.159 |

PDI 20 executes 7.51% more decode forwards than PDI 10 while processing almost
the same number of real decode rows (+0.37%). The average real batch is 6.64%
smaller. This is the expected signature of more frequent decode ticks: it
removes the ITL regression without discarding the native-V specialization.

Stable trace/turn pairing supports the aggregate result:

| Candidate versus control | Matched requests | Mean paired ITL change | Median paired ITL delta | Requests with lower ITL | Mean paired TTFT change | Mean paired E2E change |
|---|---:|---:|---:|---:|---:|---:|
| Native-V off versus AITER | 897 | -0.03% | -0.235 ms | 52.95% | +2.35% | +0.40% |
| Native-V off versus PDI-10 hybrid | 907 | -2.10% | -0.235 ms | 53.25% | +7.40% | +2.09% |
| PDI 20 versus AITER | 889 | -4.90% | -0.964 ms | 63.89% | -0.21% | -3.17% |
| PDI 20 versus PDI-10 hybrid | 906 | -7.18% | -0.913 ms | 64.79% | +4.78% | -1.75% |

All paired requests have identical OSL. ISL differs by at most five tokens in
these comparisons, so pairing is descriptive rather than an independent
replication. The current decision is to keep native BF16-V enabled and carry
PDI 20 forward as the next end-to-end candidate. Do not change the production
default from PDI 10 until PDI 20 repeats or passes the matched 900-second gate.

## Reproduction

Run benchmarks from the repository root with an isolated Triton cache for each
code variant. The examples below assume GPU 0 is idle.

### Triton production-dispatch prefill

```bash
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
TRITON_HIP_USE_ASYNC_COPY=0 \
TRITON_CACHE_DIR=/tmp/v4-fp8-prefill-dispatch-p1151 \
ATOM_USE_TRITON_ATTN=1 \
ATOM_V4_TRITON_FP8_PREFILL=1 \
python scripts/performance/bench_v4_fp8_triton_prefill.py \
  --backend dispatch \
  --tokens 2048 \
  --heads 128 \
  --prefix-len 1151 \
  --extend-len 127 \
  --scenario mixed \
  --warmup 10 \
  --iterations 40 \
  --config 32,32,2,3,0 \
  --output /tmp/v4-fp8-prefill-dispatch-p1151.json
```

Change `--prefix-len` to `8192` and use a new cache/output path for the P8192
case.

### FlyDSL short exact-full case

```bash
HIP_VISIBLE_DEVICES=0 \
PYTHONPATH=. \
TRITON_HIP_USE_ASYNC_COPY=0 \
TRITON_CACHE_DIR=/tmp/v4-fp8-flydsl-p32 \
python scripts/performance/bench_v4_fp8_triton_prefill.py \
  --backend flydsl \
  --tokens 8 \
  --heads 128 \
  --prefix-len 32 \
  --extend-len 32 \
  --scenario uniform \
  --seed 31 \
  --warmup 10 \
  --iterations 20 \
  --config 128,32,8,2,0 \
  --flydsl-pipeline-two \
  --flydsl-wave-padded-k \
  --flydsl-lds-padding 4 \
  --flydsl-fixed-softmax-ref \
  --flydsl-transpose-v \
  --flydsl-reuse-q-across-n \
  --flydsl-no-sentinel \
  --flydsl-full-tile-fastpath \
  --flydsl-cache-all-q \
  --flydsl-permute-k-scales \
  --flydsl-pairwise-pv \
  --assume-full-tiles \
  --output /tmp/v4-fp8-flydsl-p32.json
```

### FlyDSL mixed case

Use the same FlyDSL flags, omit `--assume-full-tiles`, and set:

```text
--tokens 2048
--prefix-len 1151
--extend-len 2048
--scenario mixed
--seed 31
```

For the production-shaped comparison, change `--extend-len` to `127`. The
ABBA80 artifacts retain every raw timing sample plus standard deviation and
coefficient of variation.

### FlyDSL long uniform case

Use the same FlyDSL flags, retain `--assume-full-tiles`, and set:

```text
--tokens 64
--prefix-len 1152
--extend-len 2048
--scenario uniform
--iterations 50
```

### Decode robustness

CSA B10-B16 robustness:

```bash
HIP_VISIBLE_DEVICES=0 PYTHONPATH=. \
python scripts/performance/bench_v4_fp8_triton_csa_robustness.py \
  --batches 10,11,12,13,14,15,16 \
  --kv-lens 384,640,1152 \
  --seeds 20260917,20260918 \
  --iterations 100 \
  --json /tmp/v4-fp8-csa-robustness.json
```

Add `--include-flydsl` to include the benchmark-only FlyDSL split1, split-K plus
Triton reducer, and pure FlyDSL split-K candidates. They are timed through CUDA
Graph replay like the other backends.

HCA tuning must include heterogeneous context vectors. Uniform-only winners
must not be promoted into the hybrid dispatcher.

### Static validation

```bash
python3 -m py_compile \
  atom/model_ops/v4_kernels/paged_decode.py \
  atom/model_ops/v4_kernels/paged_decode_fp8_flydsl.py \
  atom/model_ops/v4_kernels/paged_decode_fp8_triton.py \
  atom/model_ops/v4_kernels/paged_prefill.py \
  atom/model_ops/v4_kernels/paged_prefill_fp8_triton.py \
  atom/model_ops/v4_kernels/paged_prefill_fp8_flydsl.py \
  scripts/performance/bench_v4_fp8_triton_prefill.py \
  scripts/performance/bench_v4_fp8_triton_csa_robustness.py \
  scripts/performance/tune_v4_fp8_triton_heterogeneous.py

git diff --check

pytest -q tests/test_paged_attention_dispatch.py
```

## Qualification rules

A candidate should not replace AITER/OPUS unless all relevant gates pass:

1. Correctness:
   - No NaN or Inf.
   - Report cosine similarity, max absolute error and relative RMSE where
     available.
   - Include tail, sentinel and ragged-length inputs unless the specialization
     explicitly rejects them.
2. Performance:
   - Use same-process interleaving or ABBA order.
   - Use at least two seeds for close results.
   - Report every round when the margin is below 2%.
   - Compare identical token, prefix, extend, head and cache configurations.
3. Coverage:
   - Test both uniform and heterogeneous HCA inputs.
   - Test short, mixed and long prefill shapes.
   - Do not infer P8192 behavior from P1151.
4. Resources:
   - No scratch spill for the promoted kernel.
   - Inspect VGPR, SGPR, LDS size, occupancy and generated ISA.
5. Integration:
   - Preserve gfx1250 ASM fallback behavior.
   - Keep all dispatch decisions CUDA-Graph safe.
   - Run the dispatch test suite and static checks.
6. End-to-end:
   - After the operator matrix passes, run at least a matched 900-second AgentX
     workload before changing the default routing.

## Next tuning priorities

1. Replace the HCA host-built compact task map with a GPU-resident planner and
   a graph-safe maximum grid, while preserving the 8/8 pure-FlyDSL result.
2. Extend the compact FlyDSL split-K path to CSA B10-B12/K1152 and reduce the
   single-query workgroup/LDS footprint so B13+ can beat qh64 Triton split1.
3. Reduce FlyDSL prefill integer layout/address instructions. The current
   `SQ_INSTS_VALU_INT32` count is approximately 4.2x OPUS.
4. Reduce FlyDSL LDS bank conflicts without reintroducing the failed narrow
   K-scale read path.
5. Extend FlyDSL qualification to sentinel, empty-query, ragged-tail, uniform,
   and additional shapes before widening the current Q <= 127/K < 4096
   production envelope.
6. Revisit P8192 only with a single-launch structural reduction in integer
   address work or LDS conflicts; launch-parameter sweeps and two-launch prefix
   partitioning have been exhausted.
7. Keep production HCA routing unchanged until the GPU planner and graph-safe
   grid pass the full matrix and a matched 900-second AgentX run.
8. Only after attention promotion, continue replacing the remaining AITER
   QK norm/RoPE/quant, FP4 indexer and compressor kernels.

## Artifact index

Triton prefill:

```text
runs/v4-fp8-stage2-20260919/takeover-continuation-20260919/
  dispatch-stage3ext1-g8-p1151-final-r1-abba40.json
  dispatch-stage3ext1-g8-p1151-final-r2-abba40.json
  dispatch-stage3ext1-g8-p8192-final-r1-abba40.json
  dispatch-stage3ext1-g8-p8192-final-r2-abba40.json
```

FlyDSL prefill:

```text
runs/v4-fp8-flydsl-cluster-20260922/
  prefill-flydsl-wavepadk32-pipeline-v37-permkscale2-pairpv4-p32-e32-t8-abba20.json
  prefill-flydsl-wavepadk32-pipeline-v37-permkscale2-pairpv4-p1151-t2048-mixed-seed31-abba20.json
  prefill-flydsl-wavepadk32-pipeline-v37-permkscale2-pairpv4-p1152-e2048-t64-uniform-abba50.json
```

FlyDSL E127 prefill and decode follow-up:

```text
runs/v4-fp8-flydsl-e127-20260923/
  p1151-e127-dynfull-alpha-seed31-abba80.json
  p8192-e127-dynfull-alpha-weu-real-screen-abba20.json
  p8192-e127-dynfull-alpha-weu1-seed31-abba80.json
  p8192-e127-dynfull-alpha-weu1-seed47-abba80.json
  p8192-e127-partition-prefix-abba20.json
  decode-csa-b10-b16-k384-640-1152-flydsl-split1-42x50.json
  decode-csa-b10-b16-k384-640-1152-flydsl-prefixonly-42x50.json
  decode-hca-b6-8vectors-flydsl-split1-50.json
  decode-hca-b6-flydsl-splitk-reducer-screen-3x20.json
  decode-hca-b6-flydsl-flyreduce-wavegroup-screen-3x20.json
  decode-hca-b6-flydsl-pure-splitk-hg4-cap16-s9-l14-8vectors-50.json
  ir-p8192-final/.../21_final_isa.s
```

FlyDSL production qualification and pointer-ABI coverage:

```text
runs/v4-fp8-flydsl-production-20260923/
  pointer-abi-production-dispatch-p1151-e127-abba10.json
  pointer-abi-t2048-p2047-e127-mixed-abba5.json
  pointer-abi-t2048-p3071-e127-mixed-abba5.json
  pointer-abi-t2048-p4095-e127-mixed-abba5.json
  pointer-abi-t2048-p7000-e127-mixed-abba5.json
  pointer-abi-t4096-p1151-e127-mixed-abba5.json
  pointer-abi-t2048-p1151-e2048-causal-abba5.json
  ragged-t3072-p1151-e3072-causal-abba3.json
  pointer-abi-production-dispatch-p4095-e127-abba5.json
```

Matched C96 end-to-end A/B:

```text
runs/pr2256-fp8-aiter-c96-w500-m500-20260923-r1/
runs/pr2256-fp8-hybrid-c96-w500-m500-20260923-r2/
runs/pr2256-fp8-hybrid-nativenvoff-c96-w500-m500-20260923-r1/
runs/pr2256-fp8-hybrid-pdi20-c96-w500-m500-20260923-r1/
```

Decode robustness and HCA exploration:

```text
runs/pr2256-alltriton-csa-b12-neighborhood-20260918/
runs/pr2256-hca-plan-implementation-20260918/
runs/pr2256-native-group-lowbatch-20260918/
```

# Qwen3.5-35B-A3B MXFP4 on 4 x MI455X (gfx1250): MLPerf Endpoints agentic, the 2026-11-01 to 11-24 points

How to reproduce the numbers below: the exact code, the settings, the run, and how each number is computed.
They are 10-minute measurement points, not official results (see [What these numbers are](#what-these-numbers-are)).

## 2026-11-24: low-concurrency decode bundle at c64, XCD rotation at c1024

`agentic_20261124.env` on top of the 11-01 to 11-23 env files. Code: `alizaidy/gfx1250-qwen-support` (ATOM 21f211740, aiter
375152d4e) plus, all opt-in: the aiter MXFP8 GEMM early residual / XCD-grouped tiles (`AITER_GEMM_RES_FAST`), the router +
shared-expert kernel scheduling (`AITER_ROUTER_SHARED_V2_LATE_WAIT`, `_XCD_ORDER`), MoE stage 1 + 2 in one launch
(`AITER_MOE_FUSED12`), the Qwen3.6 grouped-MoE CSV rows for 9-32 and 257-512 rows, and the ATOM low-concurrency step cuts
(embedding + layer-0 norm fold, packed prologue upload, presence anchors, final norm into the output, stage cache,
persistent verdict). Every configuration warmed first; the GPU driver loaded as on previous days (`mtype_local=0`).

c64: the 11-23 settings plus the bundle in `agentic_20261124.env` (the 11-21 low-concurrency flags, which were optional
until now, the ATOM step cuts and the aiter changes). Interleaved 10-minute runs:

| c64 | run 1 | run 2 | mean | accuracy |
|---|---:|---:|---:|---|
| 11-23 settings | 1154.5 | 1163.4 | 1159.0 | 0.5602 / 0.5657 |
| + 11-24 bundle | 1233.5 | 1221.3 | **1227.4 (+5.9%)** | 0.5615 / 0.5603 |

MTP acceptance 73.8% vs 73.9%; no failed requests. Official-length run of the bundle at c64 (`run_official_host.sh
<container> 64`): **e2e 1176.1, 12,675 tok/s, accuracy 0.5609** (coding 0.5308, workflow 0.7002), 120,914 / 120,914
requests, 0 failed, 59.5 minutes, TTFT p99 102 ms.

c1024: `AITER_UA3D_XCD_ROTATE=1` (bitwise) 173.4 vs 172.3 (+0.6%), and +0.6% again in a second pair.

Notes:
- The parts were also run alone at c64: the aiter GEMM and router changes and the MoE config each stayed within run-to-run
  noise (step traces show 6-60 us saved per step at batch 1-4). Only the full bundle was measured above noise.
- All parts are bitwise except `ATOM_MTP_DRAFT_LMHEAD_STATS` (draft-head sampling statistics differ in the last bits;
  sampled tokens identical).
- The bundle is set at the c64 point only; c32 and c128 were not measured.

## 2026-11-23: one-shot split reduce sized by the launch, 3D decode XCD rotation, 896 token-split programs at c256-c512

`agentic_20261123.env` on top of the 11-01 to 11-21 env files. Code: `alizaidy/gfx1250-qwen-support` (ATOM 104232865,
aiter 601e663db, which include the upstream GDN kernel changes) plus the aiter one-shot split reduce shape rule
(`AITER_GFX12_SPLIT_REDUCE_ONESHOT_AUTO`) with its ATOM fold hook, and the aiter 3D decode XCD rotation
(`AITER_UA3D_XCD_ROTATE`). Same image, topology, client and metric as before; every configuration warmed up first.
The baseline column is the 2026-11-21 recipe on this code, so the gains below are these changes only.

Per-point settings are the 11-21 ones plus `AITER_GFX12_SPLIT_REDUCE_ONESHOT_AUTO=1` everywhere and, at c256-c512 only,
`AITER_UA3D_XCD_ROTATE=1 AITER_GFX12_TOKEN_SPLIT_PROGRAMS=896` (after the point's fp4pt split extras).

| point | baseline (11-21 recipe) | + AUTO | + AUTO + rotation | + AUTO + rotation + 896 programs (recipe) | accuracy (recipe) |
|---|---:|---:|---:|---:|---|
| c64 | 1144.6 (0.5614) | **1163.2** (+1.6%) | | | 0.5619 |
| c256 | | 585.4 (0.5585) | | **598.4** (+2.2%) | 0.5623 |
| c384 | | 439.0 (0.5629) | 440.0 (0.5597) | **450.4** (+2.6%) | 0.5616 |
| c512 | | 344.3 (0.5631) | | **350.5** (+1.8%) | 0.5623 |

Official-length run of the recipe at c384 (`run_official_host.sh <container> 384`): **e2e 438.2, 59,038 tok/s, accuracy
0.5617** (coding 0.5309, workflow 0.7041), 356,549 / 356,549 requests, 0 failed, 37.9 minutes.

Notes:
- AUTO: the one-shot reduce used one fixed (CHUNK, SPEC, warps) = 16,8,2. A pass now covers ~4x the expected live
  segments of a sequence (token-split programs / (KV heads x sequences)) while every program's LDS stays resident, with
  speculative boxes only when the live segments fill them. Kernels unchanged; bitwise (only the TDM pass size,
  speculation and warps change). Reduce us a call, 16,8,2 -> AUTO: 1x4 6.3 -> 3.1, 8x4 6.6 -> 3.9, 32x4 5.4 -> 4.8,
  120x4 13.6 -> 12.8. c1024 was not served-measured (kernel ~-10 us a step).
- XCD rotation: workgroups go to the 8 XCDs by dispatch id mod 8 and the 3D grid is (sequence, head, segment) with the
  sequence fastest, so with a multiple of 8 sequences all segments of a sequence ran on one XCD (128 resident programs);
  long contexts overflowed it into a second round (+40-50% call time; 32x4 lognormal contexts 167.7 -> 118.6 us with the
  rotation). Bitwise. On its own it only removes those cliffs (c384 440.0 vs 439.0); it is what makes larger splits pay.
- 896 programs (was 512) is numerical (summation order only): -55 us a decode step at 24 sequences, -125 at 32, -170 at 40.
  Without the rotation 768 / 896 programs overflow an XCD often.
- The c256 baseline's 0.5585 has no numerical change against the 11-21 recipe (10-minute accuracy moves +-0.004).

## 2026-11-21: the 11-20 settings at every point, LPT prefill order, c256-c512 decode configs, optional c64 decode flags

`agentic_20261121.env` on top of `agentic_20261101.env`, `agentic_20261107.env`, `agentic_20261119.env` and
`agentic_20261120.env`. Code: `alizaidy/gfx1250-qwen-support` (ATOM 31b6c86fa, aiter ff25bd74c) plus the aiter 2D prefill
longest-first order (`AITER_UA_2D_LPT`), the aiter qkv / router+shared decode configs for 65-256 rows and, for the optional
c64 lines, the ATOM + aiter low-concurrency decode change. Same image, topology, client and metric as the 2026-11-19
section; every configuration was warmed up (a 2-minute point) before its 10-minute run.

Per-point settings are the 11-19 ones (KV, interval, extras, dense CAPS at c256 / c384, the c64 fp4pt split settings at
c64-c384), plus `agentic_20261120.env` and `AITER_UA_2D_LPT=1`.

**Fixed max is off by default from this recipe on** (`AITER_UA_2D_FIXED_MAX=0` in `agentic_20261121.env`): an accuracy concern
with the fixed-reference-max 2D prefill was reported outside these runs. It stays available as an option where a point's
accuracy check passes. Every run in the table below except the "no fixed max" column had it on (`agentic_20261120.env`);
LPT does not depend on it (only the program-to-block mapping changes, bitwise with fixed max on or off).

| point | 11-19 settings | + 11-20 env | + LPT | + LPT, no fixed max (recipe default) | other | accuracy (final) |
|---|---:|---:|---:|---:|---|---|
| c64 | 1080.7 | 1105.0 / 1105.2 / 1092.6 (+1.7% mean) | | | + optional c64 flags with the fp8 verify head: 1115.0 / 1110.3 (+1.3%) | 0.5639 / 0.5589 |
| c256 | 522.7 | | **549.2** (+5.1%) | | | 0.5612 |
| c384 | 389.7 | 402.5 (+3.3%) | **408.2 / 408.8** (+4.7%) | **405.2** (with the c256-c512 decode configs; +4.0% vs 389.7; accuracy 0.5628) | + c256-c512 decode configs: 411.1 (+0.6%, accuracy 0.5619) | 0.5622 / 0.5613 |
| c512 | 302.5 | | **320.7** (+6.0%) | | | 0.564 |
| c1024 | (11-20 base 151.4) | 157.8 / 157.8 | **159.8** (+1.3%) | **158.0** (+0.8% vs 156.8, the 11-20 settings without fixed max; accuracy 0.5677) | + fp8 verify head: 161.4 (+1.0%) | 0.5683 |

Official-length run of the recipe default at c1024 (no fixed max, LPT; `CAP_MIN=600`, 19,616 trajectories,
`run_official_host.sh <container> 1024`): **e2e 137.0, 86,606 tok/s, accuracy 0.5615** (coding 0.5312, workflow 0.7018),
609,122 / 609,122 requests, 0 failed, 44.1 minutes, TTFT p99 1.42 s. The 2026-11-20 official-length run with fixed max:
e2e 136.6, 86,613 tok/s, accuracy 0.5627.

Notes:
- Fixed max (optional now) is worth +1.1% at c1024 (158.0 -> 159.8) and +1.5% at c384 (405.2 -> 411.1) in these runs; the
  c64, c256 and c512 numbers above were measured with it on.
- LPT: the gfx1250 2D kernel took the batch's q blocks in the caller's order, so a long-context prefill early in the batch
  ran in the last wave. Longest-first is bitwise and never slower (kernel -2.2..-2.9% on c1024-like batches).
- At c384 the point's `pt` extra (`AITER_GFX12_TOKEN_SPLIT_MIN_TILES_FP4PT=16`) comes after the extra flags and wins, so
  the c64 split settings there are `PROGRAMS=512` only; dropping that (aiter default 2048) scored 391.1 vs 402.5.
- MTP depth 2 at c1024 (needs `ATOM_GDN_FUSED_DECODE_PAD=0`): 148.4 vs 157.8 (-6.0%; acceptance ~81% vs ~74%). MTP-3 stays.
- `--long-prefill-token-threshold` 4096 / 6144 at c1024: 157.3 / 160.4 vs 159.8, TTFT p99 up: off.
- The fp8 verify LM head (`ATOM_VERIFY_LM_HEAD_FP8`) is numerical: -0.002..-0.0035 accuracy in every 10-minute pair, and
  10-minute accuracy moves +-0.004 run to run. It stays out of the defaults until an official-length run checks the margin.
- The optional c64 flags alone (no fp8 head) measured 1098.5: the host trims count only once the GPU part of the step is
  shorter (c64 runs close to the host / GPU balance point).

## 2026-11-20: c1024 (prefill idle time, decode GEMMs at 513-640 rows, fixed-max prefill attention)

`agentic_20261120.env` on top of `agentic_20261101.env`, `agentic_20261107.env` and `agentic_20261119.env`. Code:
`alizaidy/gfx1250-qwen-support` (ATOM 1fb8453f6, aiter 524bf4cd6) plus three changes: the aiter decode GEMM configs for
513-640 rows, the aiter fixed-reference-max 2D prefill (`AITER_UA_2D_FIXED_MAX`) and ATOM `ATOM_PREFILL_EARLY_PREP`.
Same image, topology, client and metric as the 2026-11-19 section.

c1024 point: `KV=fp8_k4v4pt`, decode interval 4, `PREFILL_TOKENS=16384`, max_num_seqs 320, the 11-18 / 11-19 flags with
`ATOM_FOLD_QKNORM_QKV_FP4PT=1 ATOM_FOLD_QKNORM_QKV_MAX_M=48` but without the c64-tuned fp4pt split settings (they slow decode
1.5-4% at 104-127 rows), and

```
CAPS=[1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,72,80,88,96,104,112,120,128,136,144,152,160,168,176,184,192,224,256,320]
```

(neutral at c1024: 152.1 vs 151.5 without; listed because every run below used it).

| run | adds | e2e | vs base | tok/s | accuracy |
|---|---|---:|---:|---:|---|
| base | the point above with the 11-19 flags | 151.4 | | 87,316 | 0.5685 |
| bitwise | + split-reduce config, `ATOM_EARLY_PREFILL_SCHEDULE=1`, `ATOM_PREFILL_EARLY_PREP=1` | 153.1 | +1.1% | 87,867 | 0.5695 |
| + no drain + GEMMs | + `ATOM_STATE_OPS_DRAIN=0`, decode GEMM configs, `ATOM_FUSED_ROUTER_SHARED_EXPERT_MAX_TOKENS=640` | 156.8 | +3.6% | 89,163 | 0.5688 |
| **final** | + `AITER_UA_2D_FIXED_MAX=1` (`agentic_20261120.env`) | **157.8** | **+4.2%** | 89,655 | 0.566 |

Official-length run of the final settings (`CAP_MIN=600`, 19,616 trajectories, `run_official_host.sh <container> 1024`):
**e2e 136.6, 86,613 tok/s, accuracy 0.5627** (coding 0.5334, workflow 0.6986), OSL per-turn mean 378.9, 608,677 / 608,677
requests, 0 errored turns, 44.4 minutes, TTFT p99 / p99.9 1.42 / 1.87 s. The previous official-length c1024 run (11-16 code)
was e2e 115.7, 77,712 tok/s, accuracy 0.5634, 49.3 minutes.

Notes:
- c1024 is GPU-work bound in a closed loop: e2e follows completed requests/s (a 1% cut in GPU time per request is ~+1.7%
  e2e). Scheduling knobs only move time between TTFT and decode: budget / interval points (16k / 4, 24k / 9, 128k / 14,
  128k / 19; effective interval 14 / 19 / 24 / 29) all scored 150.7-152.6.
- The effective decode interval at c1024 is 14, not 4: the prefill coalescer probe (`_probe_local_prefill_work`) counts each
  waiting request's full length (~72k tokens) against the budget, never fills, and fires on the stall bound (4 + 10 ticks).
- Around every prefill the GPU sat idle ~5.4 ms (~2.1-2.4% of rank time): the scheduler's prefill step, the state-ops drain
  (`synchronize()` whenever a GDN checkpoint is restored) and pageable checkpoint copies. Early schedule + early prep + no
  drain remove most of it.
- Fixed max is the one numerical change (lower rms error than the running max in the kernel tests); without it every
  change above is bitwise.
- Under c1024 load the GPUs are power-capped (PPT throttling, sclk ~1.6-1.7 GHz vs ~2.28 idle); kernel speedups measured
  at idle clocks overstate serving gains.
- Not used: a windowed MTP draft attention (+1.9% e2e) changes the published drafter's attention, which the MLPerf
  Endpoints rules (2.9.4, 2.9.6.5) disallow.

## 2026-11-19: decode folds, early spec-decode metadata, fp4pt V from c64, dense graph sizes at c256 / c384

`agentic_20261119.env` on top of `agentic_20261101.env` and `agentic_20261107.env`. Code: `alizaidy/gfx1250-qwen-support`
with ROCm/ATOM#3537 and #3606 and ROCm/aiter#8248, #8249, #8250, #8330 (post-norm fold) and #8331 (unified-attention
runtime page count). Image: `rocm/fw-bringup-mlperf:gfx1250-atom-qwen-20261117`, DP4 with every
expert on every GPU, MTP-3. Client: the submission settings (`STOP_FIRST=0 ACC_INLINE=1`, `CAP_MIN=10`), 10-minute
points on warm caches; "e2e" is the report's E2E average interactivity.

Per concurrency (decode interval and extras as in the 11-17 image's `points.tsv`):

| concurrency | KV | decode interval | extras | fp4pt-only flags |
|---|---|---:|---|---|
| 32 | `fp8_k4v8` | 24 | `ATOM_PREFILL_DELAYER_STALL_TICKS=1 ATOM_GDN_PREFILL_IN_GRAPH=1` | |
| 64 | `fp8_k4v4pt` | 8 | same | `ATOM_FOLD_QKNORM_QKV_FP4PT=1 ATOM_FOLD_QKNORM_QKV_MAX_M=48 AITER_GFX12_TOKEN_SPLIT_MIN_TILES_FP4PT=8 AITER_GFX12_TOKEN_SPLIT_PROGRAMS=512` |
| 128 | `fp8_k4v4pt` | 8 | same | same |
| 256 | `fp8_k4v4pt` | 8 | `ATOM_GDN_PREFILL_IN_GRAPH=1` | same, plus `CAPS` below |
| 384 | `fp8_k4v4pt` | 8 | same | same, plus `CAPS` below |

CUDA-graph capture sizes at c256 / c384 (`serve_ep.sh` reads `CAPS`; the list must end at the point's max_num_seqs,
80 at c256 and 120 at c384, which `run_ep.sh` sets from the concurrency). The default list jumps 16, 24, 32, 40, 48, so
the c256 / c384 batches (bs 13-26 / 26-42 a rank) pad up to 7 sequences a step; this list pads at most 1 below 32 and 3
below 64:

```
c256: CAPS=[1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,80]
c384: CAPS=[1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,30,32,36,40,44,48,52,56,60,64,80,96,112,120]
```

`BLOCK_SIZE=128` everywhere. c32 stays on k4v8: fp4pt was 1.4% slower there (MTP acceptance 73.2-73.6% vs 74.0%,
prefill ~5% slower, and no bandwidth win at bs 1-2).

| concurrency | stock (with a559cd926) | + #3537 and aiter #8248-#8250 | + this date's flags (final) | final vs stock | final accuracy |
|---|---:|---:|---:|---:|---|
| 32 | 1078.2 | 1175.7 | 1222.1 (mean of 1244.5 / 1199.7) | +13.3% | 0.5614 / 0.5558 / 0.5550 / 0.5640 |
| 64 | 957.6 | 1034.0 | 1095.4 (mean of 1098.0 / 1091.1 / 1097.0) | +14.4% | 0.5676 / 0.5593 / 0.5612 |
| 128 | 711.4 | 776.1 | 798.5 | +12.2% | 0.5627 / 0.5646 |

c256 and c384 (stock here = the image's set plus `AITER_UA_FP4PT_NO_SPECIALIZE=1`):

| concurrency | stock e2e | stock tok/s | + this date's flags | + dense `CAPS` (final) | final tok/s | final vs stock | final accuracy |
|---|---:|---:|---:|---:|---:|---:|---|
| 256 | 479.4 | 39,935 | 501.3 | 522.3 | 41,442 | +9.0% e2e, +3.8% tok/s | 0.5608 |
| 384 | 361.0 | 53,056 | not run | 386.9 | 54,860 | +7.2% e2e, +3.4% tok/s | 0.5619 |

Notes:
- At c64 each step stacks: the 11-18 PRs +8.0%, the post-norm fold + early metadata +3.4%, the fp4pt split +2.7%.
  The early metadata (`ATOM_HOSTNEXT_EARLY_META`) removed the host cap: the GPU now waits only 33-35 us a step for
  the next verify, and draft-pass savings reach the step at 75-100% (about 0 before).
- 10-minute c32 accuracy sits around the 55.86% gate on every configuration, stock included (stock: 0.5587 / 0.5576
  / 0.5542; run-to-run about +-0.003). The post-norm fold is the only non-bitwise change; its c32 mean (0.5591) is
  the same as without it (0.5594). Full-length runs should confirm c32 accuracy.
- c32 e2e varies about +-2% between runs of the same config (rank idle share 26-32% with the closed-loop client).
- Without `AITER_UA_FP4PT_NO_SPECIALIZE=1`, any change in the KV pool size (a new server, other flags, another rank)
  recompiles the unified-attention prefill kernels on first use: ~7 s stalls, TTFT p99.9 7-9 s, 15-20% lower e2e.
  On a brand-new Triton cache the first server still compiles each kernel configuration once: warm up first.
- c256 / c384: a rank runs bs 13-26 / 26-42 (52-168 verify rows), past the 48-row limit of the router and qkv folds;
  the flags still cut bs 17-40 steps by 290-470 us through the parts with no row limit. Prefill is ~24% of rank time
  at c256, and the ~18-step cadence (interval 8 + the delayer's 10-step hold) is at its optimum there.
  `ATOM_EARLY_PREFILL_SCHEDULE=1` (+0.7%) and `ATOM_API_STREAM_FAST=1` (+0.8%) were within noise at c256 and are off.
- The post-norm fold applies up to 16 verify rows (bs 4); at c128 most passes have 20-48 rows, so the 11-19 flags
  add only +0.3% to +2.9% there.

## 2026-11-07: Qwen3.6, GDN mid-step checkpoints, 1x32 MXFP8 projections, ~2x decode intervals

`agentic_20261107.env` on top of `agentic_20261101.env`, run with `run_point_20261107.sh <concurrency> [run dir]`
(`AITER_DIR` = the aiter tree). Over 11-05: the Qwen3.6-35B-A3B MXFP4 checkpoint, the GDN verify split / unroll and
`ATOM_GDN_STATE_MIDSTEP=1` (about half the TTFT, spent on decode intervals about twice the 11-05/06 ones), and 1x32
MXFP8 projections (`ATOM_MXFP8_1X32=1`) with the activation quant fused into every producer, including the GDN verify
(matrix form) and the MoE gather + tail RMSNorm. Client: the submission settings (`STOP_FIRST=0 ACC_INLINE=1`, 6743
trajectories, `CAP_MIN=10`), every point a complete run read from `result_summary.json` (`METRICS_ISL=0`).

| concurrency | decode interval | output tok/s | tps_per_user (1000 / p90 TPOT) | TTFT p50 / p90 (ms) | inline accuracy | MTP accept |
|---|---|---:|---:|---|---:|---:|
| 1536 | 10, 20,480-token prefill budget | 81,856 | 136.5 | 2651 / 3979 | 57.76% | 63.5% |
| 1024 | 24 | 63,976 | 231.0 | 2693 / 3659 | 57.46% | 63.8% |
| 768 | 32 | 58,349 | 262.5 | 1887 / 2680 | 57.21% | 64.3% |
| 512 | 48 | 50,144 | 314.2 | 1032 / 1670 | 56.59% | 65.4% |
| 384 | 80 | 40,406 | 391.5 | 905 / 1595 | 56.74% | 65.9% |
| 256 | 128 | 31,799 | 477.6 | 568 / 1198 | 56.38% | 65.2% |
| 128 | 96 | 20,441 | 632.1 | 221 / 412 | 56.48% | 67.2% |
| 64 | 96 | 10,995 | 785.2 | 160 / 312 | 55.77% | 65.8% |

At matched intervals the 1x32 MXFP8 projections add 2-5% tok/s and 2-5% tps_per_user over bf16 (c128 bf16: 19,524 @
614.0, accuracy 56.59%). c64 is 0.09 points under the 55.86% accuracy gate on 17.5k outputs (c128 and up: 32k-132k).

## 2026-11-05: DP4 with replicated experts, API offload, MoE fusions

`agentic_20261105.env` on top of `agentic_20261101.env`: DP4 attention with every routed expert on every GPU and
independent ranks, the A8K4V8 KV cache with 128-token blocks, MTP-3, the MoE / attention fusion flags of this PR, and
request preparation in 64 tokenize workers (`ATOM_API_OFFLOAD_PREPROCESS=1`, #3018). Client: the submission settings
(`STOP_FIRST=0 ACC_INLINE=1`, 6743 trajectories, `CAP_MIN=10`, a 10-minute issue cap).

| concurrency | decode interval | output tok/s | tps_per_user (1000 / p90 TPOT) | requests/s |
|---|---|---:|---:|---:|
| 1536 | 10, 20,480-token prefill budget | 65,987 | 137.2 | 190.7 |
| 1024 | 8 | 63,723 | 137.0 | 180.1 |
| 768 | 16 | 56,503 | 181.2 | 157.3 |
| 512 | 24 | 46,440 | 239.8 | 130.5 |
| 256 | 32 | 29,326 | 380.2 | 82.0 |
| 128 | 48 | 16,993 | 578.3 | 47.0 |

Scored with `quick_score.sh <run dir>` (run inside the container next to a `bench_point.sh` run started with
`STOP_S=0`): when the client's performance window closes and its tracked turns complete, `replay_report.py` scores
`events.jsonl` with the client's own code (the same window, OSL and TPOT rules as `result_summary.json`), writes
`<run dir>.quick_summary.json` and stops the run, about 15 minutes a point instead of 35-40. A run the client ends by
itself (c128: no user finishes before the issue cap) is read from its `result_summary.json` (c128: complete, inline
accuracy 55.48%, full-run OSL 361.3). Not official results: Qwen3.5 MXFP4 is not an approved checkpoint, and the
Qwen3.5 chat template drops earlier turns' reasoning.

```bash
set -a; . ./agentic_20261101.env; . ./agentic_20261105.env; set +a
export ATOM_PREFILL_DECODE_INTERVAL=8 PREFILL_TOKENS=16384 NUM_TRAJ=6743 STOP_FIRST=0 ACC_INLINE=1 CAP_MIN=10
STOP_S=0 ./bench_point.sh 1024 /work/qwen35_ep/c1024_1105 &
./quick_score.sh /work/qwen35_ep/c1024_1105
```

## Results

DP4 attention + EP4 MoE, MTP with 3 draft tokens, fp8 KV cache. Minutes 2-8 of 10-minute points.

| concurrency | output tok/s | TPOT p50 / p90 (ms) | tps_per_user (1000 / p90 TPOT) | requests/s | requests in the server | running / waiting per engine |
|---|---:|---|---:|---:|---:|---|
| 384 | 31,973 | 5.71 / 7.12 | 140.4 | 81.5 | 203 | 47.1 / 3.0 |
| 512 | 37,852 | 6.78 / 8.65 | 115.6 | 99.5 | 289 | 65.2 / 5.5 |
| 768 | 47,713 | 8.32 / 10.61 | 94.3 | 125.2 | 466 | 101.6 / 11.9 |
| 1024 | 52,244 | 8.66 / 10.90 | 91.7 | 136.9 | 667 | 113.9 / 51.5 |

100 tok/s/user is crossed between c512 and c768, at about 45k tok/s by interpolation. Above c768 most of the
extra concurrency waits for prefill (TTFT), which TPOT does not count.

## A8K4V8 KV cache

`KV=fp8_k4v8` stores K in NVFP4 (e2m1 + one e4m3 scale per 16 values) and V in fp8, and hands Q to aiter's
unified attention in fp8: 400 bytes a token and KV head against 512 for fp8, so the KV pool holds about 30%
more tokens. It needs ATOM `dd4573b4b` and aiter `7b1d271e4` (both on `alizaidy/gfx1250-qwen-support`).

```bash
set -a; . ./agentic_20261101.env; set +a
export KV=fp8_k4v8          # after the env file, or before it: the file keeps a KV from the environment
STOP_S=615 ./bench_point.sh 256 /work/qwen35_ep/c256_k4v8
```

`serve_ep.sh` then sets `ATOM_USE_UNIFIED_ATTN=1` (A8K4V8 is read only by unified attention; the server
refuses to start without it) and 128-token pages (`BLOCK_SIZE` in the environment wins; 64, 128 or a multiple
of 128). Nothing is set on the aiter side: unified attention picks A8K4V8 from the cache dtypes, with its own
`DT_fp8_nvfp4_fp8` configs, and Qwen3.5's fused Q/K norm + MRoPE + KV write has an A8K4V8 mode. A plain
server: `--kv_cache_dtype fp8_k4v8 --block-size 128` with `ATOM_USE_UNIFIED_ATTN=1`.

10-minute points, minutes 2-8, DP4 attention + EP4 MoE, MTP-3, measured 2026-11-03 on the 11-02 fusion image
with these changes applied (fp8: the image as shipped, 64-token pages):

| point | KV cache | pages | output tok/s | TPOT p50 / p90 (ms) | tps_per_user (1000 / p90) | requests/s |
|---|---|---:|---:|---|---:|---:|
| c128 | fp8 | 64 | 14,504 | 3.18 / 4.01 | 249.4 | 39.0 |
| c128 | fp8_k4v8 | 64 | 14,787 | 3.09 / 3.89 | 256.9 | 38.8 |
| c128 | fp8_k4v8 | 128 | **15,003** | **3.04 / 3.86** | **259.2** | 40.0 |
| c256 | fp8 | 64 | 25,752 | 4.58 / 5.74 | 174.3 | 67.6 |
| c256 | fp8_k4v8 | 64 | 26,007 | 4.44 / 5.56 | 179.9 | 67.7 |
| c256 | fp8_k4v8 | 128 | **26,675** | **4.34 / 5.42** | **184.5** | 68.6 |

GSM8K (3-shot, completions, DP4 + EP4, fp8_k4v8): 0.849 flexible / 0.838 strict at 256 concurrent requests
(fp8 on this recipe: 0.8635 / 0.8499). The TP points (c <= 16) were not measured with `fp8_k4v8`.

## Code

| component | where | what it is |
|---|---|---|
| image | `rocm/fw-bringup-mlperf:gfx1250-atom-20261031-qwen35` | ROCm, PyTorch, FlyDSL 0.4.0 (`/opt/flydsl-0.4.0`), the MLCommons client (`/app/mlcommons-endpoints`, d3c57f2) |
| ATOM | `github.com/vgokhale/ATOM`, branch `vgokhale/qwen35-mlperf-agentic-20261101` | this recipe and the server code that ran |
| aiter | `github.com/ROCm/aiter`, branch `vgokhale/qwen35-mlperf-agentic-20261101` (1a168ae00) | the kernels that ran |
| mori | `github.com/ROCm/mori` c1f2f646 + 460a03c2 (ROCm/mori#881) | EP dispatch / combine, with the EP4 hidden-2048 dispatch schedule |
| model | `/data/Qwen3.5-35B-A3B-MXFP4` | Qwen3.5-35B-A3B with OCP MXFP4 weights (Quark), MTP head included |

The two `vgokhale/qwen35-mlperf-agentic-20261101` branches pin what ran. The same changes are on their way into
`alizaidy/gfx1250-qwen-support`: aiter 2558555bf, 4d14b5df0, 06ab18f6c (pushed); ATOM PRs #2884 (state image
budget), #2888 (tokenize worker processes), #2889 (fp8 fused QK-norm + MRoPE + KV write), #2890 (this recipe's
client knobs). That branch also has newer kernels, so its numbers can differ.

## Setup

Host: 4 x MI455X with the amdgpu driver loaded. A container of the image, with the model under `/data` and a
work directory under `/work`:

```bash
docker run -d --name qwen35 --network host --ipc host --privileged --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /data:/data -v <host work dir>:/work \
  rocm/fw-bringup-mlperf:gfx1250-atom-20261031-qwen35 sleep infinity
```

Inside it (`docker exec -it qwen35 bash`):

```bash
cd /work
git clone -b vgokhale/qwen35-mlperf-agentic-20261101 https://github.com/vgokhale/ATOM.git
git clone -b vgokhale/qwen35-mlperf-agentic-20261101 https://github.com/ROCm/aiter.git
(cd aiter && git submodule update --init 3rdparty/composable_kernel)   # aiter JIT-builds its C++ ops

git clone https://github.com/ROCm/mori.git
cd mori && git checkout c1f2f646 && git cherry-pick 460a03c2
git submodule update --init 3rdparty/spdlog 3rdparty/msgpack-c
ROCM_PATH=/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel GPU_TARGETS=gfx1250 \
  python3 setup.py build_ext --inplace
cd /work

R=/work/qwen35_ep bash ATOM/recipes/mlperf_endpoints_qwen35/get_dataset.sh
```

The first server start JIT-compiles aiter's ops into the aiter checkout. Later starts reuse them.

## Run a point

```bash
cd /work/ATOM/recipes/mlperf_endpoints_qwen35
set -a; . ./agentic_20261101.env; set +a
export ATOM_DIR=/work/ATOM PYTHONPATH=/work/mori/python:/opt/flydsl-0.4.0:/work/aiter
export R=/work/qwen35_ep AITER_LOG_LEVEL=WARNING
STOP_S=615 ./bench_point.sh 1024 /work/qwen35_ep/c1024_repro
```

`bench_point.sh` starts `run_ep.sh` (server, smoke and prefix checks, then the MLCommons client), stops the
point 615 s after the benchmark starts, and prints minutes 2-8:

```
c1024_repro      min 2-8: gen   51,794 tok/s  prompt   176,217  running/eng 113.9  waiting/eng  51.5  ...
window min 2-8: 49036 responses, 52,244 output tok/s (client-counted)
  TPOT ms: p50 8.66  p90 10.90  p99 13.50  ->  tps_per_user (1000/p90) 91.7  (1000/p50 115.5)
```

A point takes about 13 minutes (server start ~1.5 min, smoke checks, 10 minutes of benchmark). Run one point at
a time: a server uses all four GPUs.

## What the settings do

`agentic_20261101.env`:

| setting | why |
|---|---|
| `ATOM_TOKENIZE_PROCESSES=16`, `ATOM_TOKENIZE_CACHE_MB=0` | The API server tokenizes every prompt (~100k tokens on average). In one process that caps the request rate near 100 req/s whatever the concurrency: with the segment tokenizer cache its threads queue on the cache lock (3-8 s before the engine sees a request), without it the encodes do not run in parallel. 16 worker processes: 120-164 req/s at c1024. |
| `AITER_GFX12_TOKEN_SPLIT=1`, tiles 16-128 | Decode attention split into equal-size KV segments, so the longest contexts do not leave a tail. |
| `ATOM_DP_LM_HEAD_MODE=default` | The LM head runs on each DP rank's own rows: no per-step hidden and logits all-gathers (~0.7 ms a decode step). |
| `MEGA_DISPATCH_WIRE=fp4`, `ATOM_MEGA_STAGE1_FUSED=0` | MoE activations go over the EP dispatch in MXFP4. The compact stage-1 path stays off. |
| `ATOM_STATE_CHECKPOINT_GPU_SIZE=128` | GDN state checkpoint images get a 128 GiB budget an engine (3,936 images) and keep each conversation's chain root and tip; without a budget they fill ~70% of the KV pool by minute 7. At 32 GiB (984 images) the pool turns over in seconds from c1024 up: an image is released before the conversation's next turn reads it, and the turn re-prefills its whole history (`[Image Diag]` short resumes: 11% of all prefill tokens at c1536, 73-75% at c2048; `[Cache Pools]` binding: state). 128 GiB, client-scored 10-minute points: c2048 26.0-30.3k -> 57.4k output tok/s, c1536 57.4k -> 58.7k. |
| `ATOM_PREFILL_DECODE_INTERVAL=16`, `ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT=28000` | 16 decode steps after any rank prefills; DP prefill balance. |
| `CLIENT_MAX_IDLE=4` | The client keeps idle connections 4 s (its own default) instead of the example's 0.5 s. |
| `CAP_MIN=600 STOP_FIRST=0 TURN_TIMEOUT=14400` | The example's issue cap and turn timeout; `bench_point.sh` stops the point after `STOP_S`. |
| `ATOM_PRE_ENGINE_TIMING=1`, `ATOM_STATE_CHECKPOINT_DIAG=1` | Logging only: a pre-engine timing summary and an image-budget line every 10 s in `server.log`. |

## What these numbers are

- **Window**: minutes 2-8 after the benchmark starts. Minutes 0-1 hold the cold start of every conversation.
- **output tok/s**: output tokens of the responses issued in the window, counted with the model's tokenizer over
  each response's text, reasoning and tool-call text (`tpot_p90.py`), divided by 360 s. The server's own count
  (`window_stats.py`, `gen`) agrees within ~1%.
- **tps_per_user**: 1000 / p90 TPOT, as in the rules. TPOT of a response = (completion - first chunk) /
  (its tokens - 1), from the client's event log. The rules count the tokens after the first output-bearing chunk;
  with MTP that chunk often holds 2-3 tokens, so the official p90 TPOT comes out slightly higher (a few % lower
  tps_per_user).
- **Not official results**: a compliant point runs every trajectory (`NUM_TRAJ`, a multiple of 613), scores
  inline accuracy (`ACC_INLINE=1`) and reports from the client's final summary over the steady-state window. These
  points stop after 10 minutes and ran without inline accuracy. The model is Qwen3.5; the benchmark's model is
  Qwen3.6, whose chat template keeps earlier reasoning (`preserve_thinking`), which Qwen3.5's ignores.
- **Run to run**: two c1024 runs of this configuration gave 136.9 and 135.8 req/s and 51.8k and 51.3k
  server-side tok/s. Reproduced on 2026-11-02 from these branches with the commands above (fresh mori build):
  c1024 52,053 output tok/s, TPOT p50 / p90 8.62 / 10.86 ms, tps_per_user 92.1.

## Known traps

- **`ATOM_MEGA_STAGE1_FUSED=1` hung all four GPUs** at c768 (a cross-rank wait that never completed). On a
  host with amdgpu `gpu_recovery=0` only a reboot recovers. Keep it 0.
- **A stopped client leaves its event log** (`/dev/shm/benchmark_cli_benchmark_*`, 10-17 GB for a 10-minute
  point at c1024) in tmpfs, i.e. RAM. `bench_point.sh` removes the one its point made.
- **Concurrency** must be 1-2048 and a power of two or three times one (`run_ep.sh` refuses others).
- **Stale compile cache**: after a code change, `rm -rf /root/.cache/atom/*` before the next server start.

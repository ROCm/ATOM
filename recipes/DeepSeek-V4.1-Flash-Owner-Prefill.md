# DeepSeek-V4.1-Flash row-owned prefill

This opt-in TP4 path keeps the early prefill residual, mHC gates and full-head
attention on row owners. Only the 512-wide KV latent and selected index IDs
are gathered for attention; queries and output projections stay local. Routed
and shared FFN partials are reduce-scattered. With the accepted `settled`
tail mode, state stays on owners through layer 20's complete all-row
compressor/cache update; only the replay frontier is restored for the native
decoder. Loaded FP8 parameter and scale bytes are replicated once without
requantizing them.

Use this path for the measured high-concurrency **8192/1024, C256** text
workload on four MI355X GPUs. It requires TP4 without EP, DPA, context
parallelism, side streams, TBO or speculation; BF16 KV, FP4 index, and decoder
bounded replay. Encoder layers 2–19 use equally padded owners when a forward
contains at least 4096 rows. Smaller forwards and multimodal requests use the
native early path. Normal level-3 compilation and FULL decode graphs remain
enabled.

```bash
ROCR_VISIBLE_DEVICES=0,1,2,3 ATOM_DSV41_ROW_SHARDED_MHC=1 \
  ATOM_DSV41_PREFILL_TAIL_MODE=settled \
  ATOM_DSV41_PREFILL_TAIL_TRANSPORT=packet \
  python -m atom.entrypoints.openai_server \
  --model /models/DeepSeek-V4.1-Flash \
  --kv_cache_dtype bf16 --index_cache_dtype fp4 -tp 4
```

The extra full-head attention weights cost about 1.97 GiB per rank. The
default path is unchanged unless `ATOM_DSV41_ROW_SHARDED_MHC=1` is set.

The tail seam is selected with `ATOM_DSV41_PREFILL_TAIL_MODE=settled`
(default `off`). On eligible
actual prefills it retains the row-owned state through layer 20's all-row
mHC/norm and compressor work, then restores only the bounded replay frontier
for the native decoder. `settled` pays the pending post before leaving the
early stage. The diagnostic `delayed` mode folds it into layer 20's mHC;
its full GSM8K result failed the acceptance threshold, so use `settled` for
the accepted configuration. Mixed BF16/FP32 seam fields
are transported losslessly in one packet by default;
`ATOM_DSV41_PREFILL_TAIL_TRANSPORT=fields` is the independent per-field
control. Empty-tail chunks still complete their full compressor/state work.
Deployment refusals and native fallbacks apply before state writes.

## Accepted C256 settled+packet result

Four fresh normal level-3/FULL servers ran native / settled+packet /
settled+packet / native. Each server completed 2560 warm and 2560 formal
requests, at C256 with random 8192/1024 lengths, range ratio 0.8, fixed seed,
chat template, normal prefix caching and no speculation. All arms used the
same source, image, devices and node-local raw-output storage. Every request,
length plan, prefix-cache delta, four-rank ownership and FULL route passed
independent audit. Formal timing includes initialization, JIT, waits and drain.

| Formal arm | Total tok/s/GPU | Duration s |
| --- | ---: | ---: |
| Native A1 | 12729.010267 | 417.493103 |
| Settled+packet B1 | 13759.762488 | 386.218440 |
| Settled+packet B2 | 13740.581531 | 386.757576 |
| Native A2 | 12153.487817 | 437.263284 |

Forward gain is **8.0977%**, reverse gain **13.0588%**, and pooled gain
**10.5799%**: 12434.593237 to 13750.165320 tok/s/GPU. Native drift is
-4.5213%; candidate drift is -0.1394%. Formal phases still contain observed
JIT/initialization activity; no time was subtracted. The interrupted shared
storage run contributes no formal measurements.

Full GSM8K, 1319 unique questions, 3-shot flexible-extract, scored
**1216/1319 = 0.9219105382865808 >= 0.91** in this same mode and source.
Actual scored owner/FULL routing and empty-tail/continuation behavior were
verified. The candidate includes the historical row-owner foundation plus
the new layer-20 seam: this serving comparison does not isolate the tail's
increment or establish C16/C64, vision, TP2, or cumulative DSpark5 gains.

Complete-stage profile diagnostics give full-prefill p50 145.775730 ms,
versus native 182.807990 ms (-20.2575%) and historical owner/off
149.247050 ms (-2.3259%). Profiles include long stalls and do not replace
serving throughput. Decode arithmetic is unchanged. The 25-35% prefill,
15-25% decode package budget and whole-model 2x target remain unfinished.

Changed Python files pass Ruff, Black and syntax checks; 54 ownership,
frontier and bounded-replay CPU tests passed. Four-rank lossless transport,
529-request FULL/C256/drain smoke and compiled-kernel resource evidence are
preserved. Workspace evidence is under `validation/prefill_tail_20261024/`:
`c256_serving_settled_v2/{RESULTS.md,acceptance_receipt.json}`, full same-mode
quality in `c256_accuracy_settled_v1/`, complete profiles in
`full_stage_settled_v1/`, and the formatting-only AST identity bridge.

## Historical owner/off checkpoint

The measurements below describe the earlier tail-mode `off` checkpoint.
Later clean owner-prepared C256/C16/C64 regressions remain preserved; they
are separate candidates and comparisons.

The frozen experiment used official `rocm/atom-dev:latest`, udocker P1,
ATOM base `443832207`, AITER `eb379ca87`, and normal prefix caching. Each arm
completed 2560 requests with identical actual input/output lengths. Profiler
was disabled and there were no auxiliary GPU workloads during timing.

| Arm | Native Total Tput/GPU | Row-owned Total Tput/GPU | Gain |
|---|---:|---:|---:|
| First pair | 12326.59 | 13319.10 | 8.05% |
| Second pair | 12752.67 | 13621.23 | 6.81% |
| Pooled work / total wall time | 12536.01 | 13468.47 | 7.44% |

These measurements include cache history: cached input increased across the
four arms from 1.65% to 6.08%, and the native baseline drifted by 3.46%.
They establish a promising whole-model result, with reversed-order and final
production timing still required for a precise structural gain. The previous
C16 run regressed by 12.60%, including long output stalls; use the high-C
opt-in deployment and preserve that regression when evaluating wider rollout.

Full GSM8K test, 1319 unique questions, 3-shot, flexible-extract, scored
`0.9211523881728583 >= 0.91`. Post-score four-rank traces confirmed all 18
owner-attention/FFN boundaries. This accepts the stated text task and
deployment; it does not establish vision quality or other concurrency gains.

Raw experiments live in the optimization workspace under
`validation/wide_decode_20261022/allocations/234185/`:
`owner_attention_abab_c256_v2/RESULTS.json` and
`owner_attention_accuracy_v1/{manifest.json,route_audit.json,lm_eval/}`.

The cleaned production path passed five normal level-3/FULL graph requests
with 4097, 6611, 8191, 8192 and 16381 scheduled rows, and all four rank traces
confirmed the 18 owner-attention/FFN layers. Existing candidate tests passed
194 cases; five added ownership/state tests passed separately. Uneven
checkpoint loading exposed a first-collective startup timeout; binding now
waits on the existing TP CPU group before initializing the GPU collective.
No model arithmetic or timeout value changed for that fix.

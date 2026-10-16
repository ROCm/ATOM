# Qwen3.8-Flash-Next Mono accuracy investigation

Accuracy acceptance is **incomplete**. This document records intermediate
evidence from Job 217242, 8 × MI355X (gfx950), checkpoint
`Qwen/Qwen3.8-Flash-Next-FP8`. TP8 uses expert parallelism, BF16 KV caches,
maximum two requests, 8192-token context, and FULL graphs captured at widths
1 and 2. Width 2 is the bootstrap's partial Mono finish path with native
attention; it is not a full Mono attention implementation.

## Proven numerical fixes

Native Triton causal convolution rounds each BF16 operand product to BF16
before adding it to the FP32 accumulator. Mono now preserves that boundary
instead of using FP32 FMA across the taps.

On gfx950, native GDN decode uses the Triton implementation, which consumes
BF16 sigmoid beta. Qwen's composed FlyDSL GDR now rounds beta to BF16 before
the FP32 recurrence. The composable GDR emitter now resides in ATOM at
`atom/models/qwen4_exp_mono/kernels/gdr_decode.py`; its optional BF16 beta
rounding preserves the previously tested semantics. No custom AITER branch
or tuning CSV is required. Historical numerical evidence used AITER revision
`2eb02e58dbc47243393773d79828b059ab0960b0`. The generated kernel name
includes `_bf16beta` for the new semantics.

Real checkpoint layer 0/1/2 convolution state and GDN outputs matched exactly
after these fixes; recurrent state relative L2 was approximately 1e-8 to 1e-7.
An independent FP32 recurrence oracle across eight ranks found native state
relative L2 of 5e-8 to 1.6e-7 with BF16 beta, compared with 3e-4 to 1.3e-3
with FP32 beta. Other layers still have small projection/norm differences.
TP1 GPU validation of the same changes remains pending.

The independent CPU oracle is included as
`tools/qwen38_mono_repro/gdn_reference.py`. It infers head counts from the
packet state shape, computes convolution and recurrence without importing
ATOM/FlyDSL kernels, and reports both BF16 and FP32 beta results. Rechecking
the eight checkpoint packets exactly reproduced the earlier analysis.

`tools/qwen38_mono_repro/moe_reference.py` retains the independent CPU MoE
oracle for TP8 width-1 checkpoint packets. It checks stable global top-k,
local expert mapping/ownership, and route weights, then unshuffles FP8
weights and computes explicit matrix products and BF16 contribution sums.
For the eight saved layer-0 packets, oracle-vs-Mono relative L2 is 0.000501,
oracle-vs-native is 0.001820, and active route-weight relative L2 is at most
1.16e-7. This localized evidence does not prove whole-model equivalence.

## Free generation evidence

The protocol is zero-shot greedy chat on GSM8K test, seed 1234, top-p 1,
4096 generated tokens, with an explicit request to finish using `####`
and a numerical answer. Decimal normalization compares answers such as
`57.00` and `57` equally. Length truncations are retained as failures and
reported separately. This is not the standard few-shot GSM8K protocol.

| Path | Samples | Correct | Length truncations |
| --- | ---: | ---: | ---: |
| Native, concurrency 1 | 300 | 295 | 3 |
| Native, concurrency 2 | 300 | 294 | 4 |
| Mono, concurrency 1 | 300 | 292 | 3 |
| Partial Mono, concurrency 2 | 300 | 295 | 2 |

Both 300-sample native runs completed. Mono subsequently completed all 1319
samples in each run (results below); full native comparison and broader
context/state-reuse checks remain pending. Native-correct/Mono-wrong C1 sample
indices are 13, 161, and 187 (zero-based); 13 and 161 truncated. Sample 187
produced 106.12 instead of 106. Multiple shared errors also occur in native.
At concurrency 2, native-correct/Mono-wrong index is 100;
native-wrong/Mono-correct indices are 45 and 184.
Free generation and request schedules diverge, so these observations do not
isolate a kernel error.

Cross-startup first-decode hidden/logit relative L2 was about 0.20/0.19 after
the GDN fixes. Independent native startups themselves differed by about
0.16/0.16. A fixed 128-token trajectory matched all argmax decisions and final
text, but does not establish general whole-model equivalence.

## Reproduction tools

`tools/qwen38_mono_repro/gsm8k.py` evaluates and resumes sample JSONL logs.
`gsm8k_compare.py` rescores against the dataset, checks sample identity and
duplicate indices, and writes per-sample paired disagreements. For example:

```bash
python3 tools/qwen38_mono_repro/gsm8k_compare.py \
  --dataset "$accuracy_root/gsm8k_test.jsonl" \
  --reference "$accuracy_root/gsm_native_c1.jsonl" \
  --target "$accuracy_root/gsm_mono_c1.jsonl" \
  --output "$accuracy_root/gsm_paired_c1.json"
```

Use `gsm8k.py --indices 13,161,187 --tokens 7600` with a separate output log
to retry selected original dataset indices. Selection preserves original
prompts and identifiers; new rows record token budget and concurrency. The
budget must fit the server context after prompt tokens. Retry results
supplement the 4096-token protocol and must not overwrite it.

The bootstrap's same-state shadow diagnostic requires `GEN_EAGER=1`,
`GEN_SHADOW_DIR`, and optionally `GEN_SHADOW_STEPS` (default 32). It snapshots
the request's GDN, QSA, and PLE cache slots/pages, executes the native model,
then restores state before the actual decode. By default it also executes
a second restored native reference to measure native repeatability; set
`GEN_SHADOW_NATIVE_REPEAT=0` to disable this extra reference.
Rank 0 writes hidden/logits, errors, per-row argmax matches, top-two reference
margins, and native repeatability metrics. Shadow execution is diagnostic
only and invalidates performance measurements. GPU integration completed
96-step native and Mono runs covering widths 1/2 and a roughly 6K-token
record lookup prompt. Native actual vs reference and native repeated references
matched argmax on 96/96 steps. Mono matched on 95/96 steps; native repeated
references in that run matched on 96/96. At Mono step 60, width 2, one row's
reference top-two logits were 21.25 and 21.125 (token IDs 24 and 16). Mono
produced a 21.25 tie and selected token 16. This divergence remains unresolved.
Native repeated logit relative L2 averaged 0.051 in the native run and 0.057
in the Mono run; Mono-vs-native averaged 0.056. State-coverage auditing and
layer localization remain necessary before accepting Mono.
An additional 64-step layer trace matched all argmax decisions. TP8 Mono
publishes rank-local pending MoE output and reduces it at the next layer;
the trace therefore captures pending output after the existing peer pull
to compare it with native's already reduced result. Layer-0 MoE relative L2
averaged approximately 0.0019 (native repeat approximately 0.0005), while
layer-0 width-2 streams and raw matched exactly. Later layers amplify both
Mono differences and native repeat variability. This has not established
the cause of the earlier argmax divergence.

The full 1319-sample Mono C1/C2 evaluation completed by resuming the
300-sample logs. Mono C1: 1283 correct (97.2707%), including 13 length
truncations and 23 other failures. Partial Mono C2: 1286 correct (97.4981%),
including 12 length truncations and 21 other failures. Full native C1/C2
evaluation is running from its existing 300-sample logs.
All scores are recomputed from raw answer text. Six C1 JSONL rows originally
retained stale `correct=false` fields for equivalent decimal forms from the
earlier scorer. Resume now atomically persists rescored rows before appending
new samples; the completed C1 log has been normalized without new requests.
The completed scorer's summary and `gsm8k_compare.py` agree on 1283 correct.
Full native comparison, TP1 GPU checks, and performance
validation remain pending.

Set `GEN_SHADOW_LAYERS=1` to save per-layer streams, reduced pending output,
and HC raw metrics for Mono and repeated native references. This incurs
additional clones and must be disabled for performance measurements.
`GEN_SHADOW_MOE_SAME_INPUT=1` additionally compares selected layers (0, 1, 47)
against two native MoE executions using the exact Mono `mixed` input, after
the existing peer pull. This diagnostic is CPU checked and awaits GPU
integration; it requires layer tracing.
Eight CPU checks cover successful/exceptional restoration,
shared/separate state slots, native/wrapped models, and explicit QSA write
pages absent from the block table.

Artifacts currently reside under
`/home/yifehuan/data/qwen38_accuracy_20261016`. Independent numerical analyzers
are in the workspace's `data/qwen38_analyze_gdn.py` and
`data/qwen38_analyze_moe.py`. Host/runtime/plan/shard validation passed 153
tests after the GDN fixes. These checks do not establish whole-model accuracy,
graph/resource acceptance of TP1, or performance acceptance of the new code.

The first full native attempt stopped at 382 samples after logging reported
`Disk quota exceeded`; its frontend health endpoint still returned 200 while
EngineCore was a zombie. The run resumed from saved rows after storage
recovery. The eight raw MoE diagnostic packets were relocated to container
local `/var/tmp/qwen38_accuracy_20261016/moe_diagnostic` in
`yifehuan_qwen_tp8_native_217225` (Job 217242), with per-file SHA-256
verification before deleting shared duplicates. The original shared directory
retains `relocation_manifest.json`. These local packets must be preserved
before that container is removed.

## Branch consolidation (accuracy paused)

The branch was rebased onto ATOM `9edc7859` and condensed into four commits.
The GDR composition implementation is carried locally in ATOM; its factory
setup, recurrence emitter and composition dispatch are AST-identical to the
previously tested AITER implementation. The unrelated tuning CSV change was
removed. With clean upstream AITER `ea3d22a3`, 163 Qwen/Mono host, runtime,
plan and shard tests passed (three existing multiprocessing fork warnings).
Python syntax, shell syntax and diff whitespace checks passed. This verifies
the repository migration and rebase contracts; whole-model GPU acceptance
remains paused and is not inferred from these checks.

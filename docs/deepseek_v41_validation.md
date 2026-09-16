# DeepSeek-V4.1 text baseline validation

P04 was accepted by the user on 2026-09-13 as the baseline for P05. The
independent TP4 NLL and repeatability checks pass; the task-quality and
non-near-tie differences below remain accepted limitations, not passes of the
original thresholds. P05 paged runtime and lifecycle integration is complete; see
`docs/deepseek_v41_runtime.md` for its separate acceptance results.

The numerical and task results below were taken on the eager expert loop, which
has since been replaced by V4's `FusedMoE`; see
[the P09 report](deepseek_v41_performance.md). The current expert path does not
inherit these task scores.

V4 GPU inverse RoPE is integrated in `ce26bb149` at the user's explicit direction;
`f9f8d419f` preserves empty-batch behavior without changing nonempty arithmetic.
The existing V4 BF16 attention dispatch and kernels remain unchanged. Final
normalization uses V4/AITER RMSNorm. Commit `ff95c8e4c` also reuses that kernel
in index-key normalization after independent numerical and complete task
regression. Those P04 decisions are historical: text/draft normalization now
uses the repository `layernorm.RMSNorm` directly. Current operator reuse and
its separate validation scope are recorded in the [performance report](deepseek_v41_performance.md).

## Reproduction environment

The recorded runs use MI355X, TP4, PyTorch on ROCm and AITER revision
`2039d2b96cd547ebc52f8d55f5f29ec1b8290796`. The optional evaluation dependencies
are listed in `tests/models/deepseek_v41/README.md`. Set `PYTHONPATH` to the ATOM
checkout and an isolated checkout of this AITER revision. Use separate
`AITER_META_DIR` and `AITER_JIT_DIR` directories to prevent shared build updates
from changing a live evaluation.

The original shared AITER checkout advanced during the index-key task run.
The retained broad source audit reports that drift; review found only unrelated
FlyDSL/fused-MoE changes, outside the native V4.1 expert path. The norm outputs
were bitwise equal in 12 pinned/shared cases. The subsequent pinned query-norm
check reproduced the frozen baseline NLL for every numerical record. Original
cache identities are retained; no mismatched response cache was reopened.

Detailed evidence is stored under
`/app/logs_claude/atom_dsv41_flash_impl_20260912/kernel_alignment` in `ljin_dev`.
The corresponding host export includes source patches, reports and SHA256
manifests, but excludes checkpoint weights, tensor dumps and response databases.

## Current independent numerical result

The reference is pinned to `deepseek-ai/DeepSeek-V4.1-Flash` revision
`dba1be0a40aa45a94ad051997016db3960a90277`. The loader checks the source
hashes in `tests/models/deepseek_v41/fixtures/reference_manifest.json`.
Independent PyTorch quantization, GEMM, attention and Sinkhorn definitions
preserve the published model math. Neither graph receives residuals, attention
outputs or logits from the other. Checkpoint I/O and Engram table gathers have
separate P02/P03 contract checks.

The current index-key integration TP4 run covers 37 cases, 148 prefill/decode records,
7,258 labels and 7,295 output positions. All logits are finite. Every prefill
is bitwise identical across four repetitions, including the 2,049-token case.

| Metric | Current result |
|---|---:|
| Target mean NLL | 0.6042607703 |
| Mathematical reference mean NLL | 0.5959472320 |
| NLL increase | +0.0083135383 |
| Maximum allowed NLL increase | +0.01 |
| Top-1 agreement | 96.477039% |
| Top-1 mismatches | 257 |
| Mismatches with reference margin greater than 1 / 2 nats | 46 / 6 |

The earlier compressor/index-key pair reproduced every field of all 148
numerical records from the accepted inverse baseline but later regressed on
a Chinese task item. Separately paired index-key substitution has zero unequal
logits on this numerical corpus. The completed independent run of the
narrowed index-key integration again reproduces every field of all 148 records
from the accepted inverse baseline.
The complete V4.1 test directory passes 220 tests with zero skips (12.64 s).
Without lm-eval/SciPy, the optional-dependency control passes two pure cache
identity tests and skips the 13 harness/statistics cases as intended.
The repetitive long case contributes little NLL error; the diagnostic breakdown
in `kernel_alignment/index_norm_numerical_error_distribution.json` localizes
most mismatches to the short fixtures and GSM8K texts. This does not change
the original aggregate gate.

Artifacts: `kernel_alignment/index_norm_integrated_numerical_tp4.json`,
`inverse_integrated_numerical_tp4.json`, `paired_norm_pair_after_inverse_tp4.json`,
and `index_norm_integrated_tests.log`. A numerical report's `passed` field
covers its NLL threshold, not overall P04 acceptance.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  AITER_REUSE_IDENTICAL_COMM_GROUPS=1 OMP_NUM_THREADS=4 \
  torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.validate_checkpoint \
  --model /mnt/DeepSeek-V4.1-Flash \
  --gsm8k-samples 32 --long-length 2049 \
  --output /tmp/p04-numerical.json
```

This validator loads two models. Measured TP4 residency is 143.45 GiB per rank
before scratch and allocator reservations. Run it with sufficient exclusive
GPU capacity; an earlier trial with a third model exhausted memory and is
excluded. Each case records its token-ID hash; the last three tokens run as
individual decode calls. The validator rejects non-finite outputs, incomplete
runs and an aggregate NLL increase above `--max-nll-increase` (default 0.01).

## Current task results

All comparisons use identical documents, prompts, targets, seeds and few-shot
settings. Code/ARC/HellaSwag are zero-shot; GSM8K is five-shot, greedy, with at
most 256 generated tokens. The earlier zero-shot GSM8K attempt is marked
NONCOMPARABLE and excluded.

| Metric | Accepted inverse baseline | Mathematical reference | Published-kernel transport |
|---|---:|---:|---:|
| ARC accuracy | 51/64 | 50/64 | 51/64 |
| ARC normalized accuracy | 54/64 | 53/64 | 57/64 |
| HellaSwag accuracy | 33/64 | 33/64 | 33/64 |
| HellaSwag normalized accuracy | 49/64 | 49/64 | 51/64 |
| Code line description | 35/60 | 36/60 | 33/60 |
| GSM8K strict / flexible exact match | 12/16 | 12/16 | 13/16 |
| Chinese LogiQA accuracy | 475/651 | 481/651 | 475/651 |
| Chinese LogiQA normalized accuracy | 425/651 | 432/651 | 428/651 |

The earlier compressor/index pair preserved the four smaller task groups but
lost one Chinese normalized answer (424/651), with raw accuracy unchanged.
Same-model replay of six affected option scores under four norm configurations,
twice each, reproduces the exact baseline with index-only and the exact changed
scores with compressor-only. The compressor has been restored to eager
normalization. The narrowed index-only implementation completed the full task regression:
all reported per-document metrics match the accepted inverse baseline, with
zero gains or losses. This does not imply identical likelihoods: 43 Chinese
option scores differ, with maximum absolute difference 0.446713448 nats. See
`index_norm_task_summary.json`, `index_norm_zh_response_audit.json` and
`cache_norm_zh_root_cause.md` in the evidence directory.

The inverse-specific arithmetic/score tradeoff was explicitly accepted. It does
not waive the remaining original criteria: code is 1.6667 percentage points
below the mathematical reference and Chinese normalized accuracy is 1.0753
points below it. Non-near-tie token agreement also remains open. Statistical
nonsignificance or passing mean NLL does not establish these criteria.

The published-kernel transport is supplementary, not a replacement oracle.
It uses the pinned published TileLang GEMM, sparse attention and Sinkhorn with
explicit HIP transports, including exact E8M0 conversion and FP4 unpack.
The before-inverse candidate's code score was 33/60 and Chinese score was
472/651 (normalized 420/651); those are historical results, not current scores.

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 \
  torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.lm_eval_checkpoint \
  --model /mnt/DeepSeek-V4.1-Flash --implementation target \
  --tasks arc_easy hellaswag --limit 64 --fewshot 0 \
  --response-cache /tmp/p04-target-responses --output /tmp/p04-target.json

# Run reference separately with its own cache and output, then compare:
python -m tests.models.deepseek_v41.compare_lm_eval \
  --target /tmp/p04-target.json --reference /tmp/p04-reference.json \
  --output /tmp/p04-paired.json
```

Use `bigbench_code_line_description_multiple_choice --limit 64 --fewshot 0`
for all 60 code items, `agieval_logiqa_zh --limit 651 --fewshot 0` for the full
Chinese task, and `gsm8k --limit 16 --fewshot 5`. The adapter rejects non-finite
likelihoods and generation logits, and never silently truncates oversized
requests. The paired reporter checks document/prompt/target hashes, finite
responses, per-document gains/losses, bootstrap intervals and McNemar tests.

Response-cache identity includes source, tokenizer/configuration, software and
parallelism. All ranks resume rank zero's committed snapshot. Select a new
cache directory after source or execution changes. Initial pilot artifacts
containing non-finite choice scores are invalid for acceptance and excluded.

## Arithmetic and implementation boundaries

Weights retain native group32 FP8/packed FP4 storage and the original A8
quantization. The correctness GEMM uses BF16 MFMA per group and FP64 scaled
accumulation until output conversion. No full dequantized weight matrix is
used. Independent cancellation and E8M0 boundary tests cover both formats,
output dtypes and split/unsplit reductions. E8M0 code 0 is 2**-127 and code
255 is NaN; the decoder explicitly handles these rather than bit-shifting to
FP32 zero/infinity. All 64 analytic cases pass. The decoder correction leaves
28,571 real-checkpoint GEMMs (1,772,408,832 output elements) bitwise unchanged.

At the time of these P04 measurements, intermediate RMSNorm retained the
published FP32 reduction order and index-key normalization used a separate
fused wrapper. That wrapper has since been removed; target, draft, compressor
and index-key normalization directly use `atom.model_ops.layernorm.RMSNorm`.
The historical task scores below do not validate this later substitution.

An independent FP64 calibration found that current GEMM accumulation is often
locally closer to exact arithmetic than the FP32-BLAS reference. Real inverse
RoPE tracing likewise pins a difference to a BF16 midpoint: V4 FMA is 3.73e-9
below a midpoint where the eager FP32 intermediate lands exactly on it. One
changed rotation value changes seven projected BF16 values and one A8 byte,
then propagates through the model. Shape, positions and NoPE preservation are
correct; disabling FMA does not restore general equality. These diagnoses
explain contributors, not every remaining task or token mismatch.

## High-margin token diagnosis

A follow-up traces two unchanged GSM8K corpus inputs through all 40 layers.
All four ranks exactly reproduce the saved target/reference NLL sums.
Normal execution matches 323/334 prefill top-1 positions. Supplying identical
reference inputs at each attention/FFN boundary in a diagnostic replay gives
334/334 matches and removes expert-set differences. This localizes accumulated
input-error propagation; the replay is not an independent quality result.

For `gsm8k/test/27`, position 34 has identical queries and causal index rows on
all ranks, but one earlier KV value differs. The unchanged V4 attention gives
bitwise-identical output at that position when given the same reference Q/K.
The discrepancy therefore precedes inverse RoPE. WKV projection first differs
in four BF16 values, with identical input activation bytes/scales. Independent
FP64 GEMM rounds to the ATOM result for all 69,120 outputs, and exact rational
sums confirm that all four differing ATOM values are nearer the exact result.
One reference FP32 dot lands on a BF16 midpoint where the exact dot is below
it by `2**-30`; the subsequent KV normalization/A8 boundary changes the cache
value. No shared V4 kernel modification follows from this finding.

This explains one high-margin case without changing the frozen reference or
waiving the remaining task/token criteria. See `high_margin_propagation.md`,
`first_attention_high_margin.md` and the model-free
`probe_wkv_high_margin_exact.py` in the evidence directory.

## Collective reliability

The production offline entry point and evaluators select RCCL through
`set_custom_all_reduce(False)` before group creation. No attention or model
precision change is used to hide collective failures.

The unchanged model-free reproducer fails the default custom path even on
AITER `2039d2b96cd547ebc52f8d55f5f29ec1b8290796`: 12/8/11/7 errors per rank
in 2,400 calls. Legacy custom and RCCL pass that control. Subsequent private
builds identify nonuniform block-barrier participation in the two-stage
reduce-scatter tail. Changing only the loop bound to be block-uniform and
masking invalid lanes gives zero errors on every rank in 24,000 calls; the
private original gives 96/146/151/61 errors. Both use exact integer sums.

The patch, reproducible private build setup, loaded-library hashes and logs
are in `kernel_alignment/allreduce_uniform_loop_results.md/json`. Installed
AITER is not modified. A subsequent paired checkpoint check
covers five inputs, including 2,049 tokens, with four repeats per prefill/decode
sequence in each mode. All 160 records are finite and each mode is bitwise
repeatable. The corrected custom path makes 9,680 dispatches on the recorded
rank, but changes 36 of 2,750 top-1 positions versus RCCL, with mean NLL delta
+0.0000438787 on this bounded corpus. Full task quality and performance remain
unvalidated, so this does not enable custom collectives in the model.

The repository collective regression exercises the production initialization:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.validate_collectives
```

Historical real-checkpoint RCCL controls include 48 finite, repeatable outputs
on three previously failing Chinese inputs with unused cache rows poisoned.
TP8 also passes the model-free production collective policy regression.

## Further kernel reuse and performance evidence

Each substitution is measured against the frozen accepted baseline before task
regression. Engram staging views are snapshotted in paired diagnostics, and
baseline NLL is checked by token hash/position; the older diagnostic that
retained overwritten views is invalidated.

| Candidate | Historical P04 decision / evidence |
|---|---|
| V4 BF16 attention | Reused unchanged |
| V4 inverse RoPE | Integrated; accepted FMA tradeoff; 4.14-7.06x warm operator speedup |
| Final RMSNorm | Reused; prior complete task regression preserves every metric |
| Index-key RMSNorm | Integrated in `ff95c8e4c`; all task document metrics preserved; 8.22-12.16x warm operator speedup |
| Compressor RMSNorm | Full Chinese loses one normalized answer; causal replay isolates compressor; restored eager |
| Query RMSNorm | Not adopted; full Chinese falls to 471/651 raw and 422/651 normalized |
| KV RMSNorm | NLL passes but code loses one answer (35/60 to 34/60); not adopted |
| Attention-input / FFN-input RMSNorm | NLL increases +0.0122933 / +0.0107104; not adopted |
| V4 small-token grouped wo_a | Task regressions and no warm operator improvement; not adopted |
| Fused SwiGLU | NLL increase +0.0111406 exceeds +0.01; shared TP4 width 576 unsupported; not adopted |
| sqrtsoftplus router | Negative-tail logits -20/-30 lose scores in current AITER; not adopted |
| mHC post / native FP8 dot | Not adopted; numerical and FP64-calibration evidence retained |

Query-norm diagnostics use the accepted index-key integration and pinned AITER.
NLL is 0.6043518102, or +0.0084045782 versus the mathematical reference.
ARC and HellaSwag preserve all document metrics. Code remains 35/60 with one
gain and one loss. GSM8K five-shot improves from 12/16 to 15/16, with three
gains and no losses. The completed Chinese regression covers all 651 documents
and 2,604 option responses. Raw accuracy falls from 475/651 to 471/651 (six
gains, ten losses); normalized accuracy falls from 425/651 to 422/651 (six gains,
nine losses). The substitution is not adopted. Both shards used identical
source/cache identities, and the final source audit found no changes during
execution. Query normalization remains eager.

Reconstructing token decisions from the saved paired numerical reports finds
14 resolved mathematical-reference disagreements, 12 new disagreements and
two changed predictions that still disagree. Total disagreement becomes
255 rather than 257. This diagnostic does not close the non-near-tie criterion;
mathematical margins for the newly introduced disagreements were not saved.

Inverse and norm speedups are isolated GPU-graph operator measurements with
warm weights/storage, after other model jobs exited. They are not end-to-end
serving speedups. The FP64-accumulation correctness GEMM's measured cost ratio
against the earlier candidate is 0.937-1.102 over 15 real-weight cases; small
apparent improvements are not claimed as speedups. Eager intermediate RMSNorm
still has material launch cost. P05 now provides a serving lifecycle and verifies pool byte accounting.
End-to-end throughput and latency optimization remain separate work.

## Long-context coverage and accepted P04 limitations

The full-model 2,049-token case crosses top-512 selection for ratio-1 and
ratio-2 owners. Separate indexer tests cover 32,771 keys with 32 heads of
dimension 128, top-512 selection, 2,048-key block pruning, Full/Reindex,
partial newest blocks and both position tie policies. This is not a full-model
32K or 1M quality claim. Attention's main head dimension remains 512 and its
rotary dimension 64.

SWA ring wrap, odd compression boundaries, owner aliases and read/write order
have state tests. V4 BF16 attention retains its 0.003 relative-L2 operator
budget; pinned TileLang HIP probes measured at most 0.002461 for decode and
0.001203 for prefill. Controlled graph-composition tests isolate wiring by
substituting a checked reference output; independent checkpoint/task runs
make no such substitution.

The original task/token differences above were explicitly accepted for the
P04 baseline; they were not resolved by relaxing the historical report fields.
Any subsequent kernel substitution still requires its own complete regression.
Historical cross-run likelihood outliers remain unexplained: request replays
and rank-cache audits narrow them, but fixed-corpus repeatability does not prove
arbitrary request-sequence repeatability. P05 validates paging and request
lifecycle against this accepted arithmetic without substituting new kernels.

# DeepSeek-V4.1 text baseline validation

P04 remains open. The independent TP4 NLL and fixed-corpus repeatability checks
pass, but task quality and non-near-tie token agreement do not yet satisfy every
original criterion. P05 runtime registration has not started.

V4 GPU inverse RoPE is integrated in `ce26bb149` at the user's explicit direction;
`f9f8d419f` preserves empty-batch behavior without changing nonempty arithmetic.
The existing V4 BF16 attention dispatch and kernels remain unchanged. Final
normalization uses V4/AITER RMSNorm. Commit `ff95c8e4c` also reuses that kernel
in index-key normalization after independent numerical and complete task
regression. Compressor and KV normalization retain their eager definitions
because their substitutions regressed on task quality.

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

Intermediate RMSNorm before quantized projections retains the published FP32
reduction and affine order. A small `FusedRMSNorm` leaf owns the validated
index-key reuse, including CPU/empty behavior and strided input
adaptation; it has no TP initialization or communication-policy dependency.
Model composition chooses the role, while `normalization.py` owns the math.

An independent FP64 calibration found that current GEMM accumulation is often
locally closer to exact arithmetic than the FP32-BLAS reference. Real inverse
RoPE tracing likewise pins a difference to a BF16 midpoint: V4 FMA is 3.73e-9
below a midpoint where the eager FP32 intermediate lands exactly on it. One
changed rotation value changes seven projected BF16 values and one A8 byte,
then propagates through the model. Shape, positions and NoPE preservation are
correct; disabling FMA does not restore general equality. These diagnoses
explain contributors, not every remaining task or token mismatch.

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

| Candidate | Current decision / evidence |
|---|---|
| V4 BF16 attention | Reused unchanged |
| V4 inverse RoPE | Integrated; accepted FMA tradeoff; 4.14-7.06x warm operator speedup |
| Final RMSNorm | Reused; prior complete task regression preserves every metric |
| Index-key RMSNorm | Integrated in `ff95c8e4c`; all task document metrics preserved; 8.22-12.16x warm operator speedup |
| Compressor RMSNorm | Full Chinese loses one normalized answer; causal replay isolates compressor; restored eager |
| Query RMSNorm | Not integrated; NLL passes, 28 top-1 changes; smaller tasks complete, full Chinese pending |
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
gains and no losses. The complete 651-document Chinese regression is still
required before adoption; aggregate gains elsewhere do not replace it.

Inverse and norm speedups are isolated GPU-graph operator measurements with
warm weights/storage, after other model jobs exited. They are not end-to-end
serving speedups. The FP64-accumulation correctness GEMM's measured cost ratio
against the earlier candidate is 0.937-1.102 over 15 real-weight cases; small
apparent improvements are not claimed as speedups. Eager intermediate RMSNorm
still has material launch cost. Runtime latency, throughput and memory remain
unmeasured until the P05 serving lifecycle exists.

## Long-context coverage and remaining P04 work

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

P04 still requires resolution of the original task/token quality gaps and any
new kernel's complete regression. Historical cross-run likelihood outliers
also remain unexplained: request replays and rank-cache audits narrow them,
but fixed-corpus repeatability is not a proof of arbitrary request-sequence
repeatability. P05 remains gated and will own paged request state, scheduler
lifecycle and serving integration after P04 acceptance is resolved.

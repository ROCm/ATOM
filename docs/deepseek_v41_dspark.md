# DeepSeek-V4.1 DSpark

Native DSpark uses the checkpoint's three draft stages to propose five tokens.
Target verification processes the anchor plus up to five proposals and commits
only the accepted input prefix. It reuses ATOM's DSparkProposer, VerifyScheduler,
request scheduler and unchanged V4 BF16 attention kernels.

## Supported configuration

The P10 acceptance configuration is TP4 with whole-expert EP, BF16 KV and index
caches, and text requests. Target execution can be eager or use PIECEWISE
graphs; the draft currently runs eagerly. Packed speculative caches, multimodal
speculation, synthetic acceptance and relaxed MTP acceptance are not admitted. Speculative output token logprobs are
also rejected because the current output protocol cannot return them correctly.
Non-speculative vision and packed cache support are independent. Both target
and draft MoE reuse V4 FusedMoE.

Fixed-length verification is the default when DSpark is explicitly selected:

```python
from atom.config import Config, SpeculativeConfig

model = "/mnt/DeepSeek-V4.1-Flash"
config = Config(
    model=model,
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    kv_cache_dtype="bf16",
    index_cache_dtype="bf16",
    max_num_seqs=4,
    max_num_batched_tokens=512,
    max_model_len=4096,
    enforce_eager=True,
    speculative_config=SpeculativeConfig(
        method="dspark", model=model, num_speculative_tokens=5,
    ),
)
```

For target graphs, set `enforce_eager=False` and
`compilation_config=CompilationConfig(level=0,
cudagraph_mode=CUDAGraphMode.PIECEWISE)`. Whole-draft capture is explicitly
disabled: its request-window reads still need live request metadata.
It must not silently capture warmup slots or reuse another request's window.

## State and sampling contracts

- Target layer **inputs** 37, 38 and 39 contribute the mean of the four residual
  streams. The target embedding and output head are shared with the drafter.
- Draft blocks use 128 experts/top-3, Markov rank 256 and noise token 128799;
  target blocks retain 384 experts/top-6. The draft block is bidirectional over
  its own five positions and attends to the valid target context window.
- There are 43 request-owned windows: 40 target and three draft context windows.
  Logical visibility remains 128; physical rings have 133 rows so every possible
  accepted prefix retains its preceding window after six-row verification.
- Compressor tails and compressed Engram histories are staged per input prefix.
  Committing zero draft tokens still commits the anchor input. Checkpointing and
  drafting require the tentative state to have been resolved.
- Preemption replays only finalized host token IDs, excluding deferred output and
  draft placeholders. A replayed prefill discards stale deferred results for the
  same request ID. Output timing likewise counts finalized tokens only.
- Greedy verification matches target argmax. Stochastic verification draws from
  each request's target distribution with independent noise per row, accepts
  matching proposals, and stops at the first mismatch. Draws are broadcast before
  rejection so all TP ranks commit the same prefix. This preserves the target
  sampling distribution; it is not probability-ratio rejection sampling and
  does not promise the latter's acceptance rate.

Losslessness describes the sampling and accepted-state contracts for the target
logits. It does not promise bitwise equality of free-running generations across
batch shapes. BF16 reduction and quantization boundaries can change close logits;
quality acceptance therefore also compares paired teacher-forced logits and
standard task metrics. The numerical budget allows mean NLL to increase by at
most 0.01 nats/token and task accuracy to decrease by at most one percentage
point. Paired confidence intervals and systematic shifts are reviewed separately;
a point estimate within tolerance does not prove statistical noninferiority.
Integer state, accepted-prefix and output ownership contracts remain exact.
The current FusedMoE and AITER mHC paths require fresh quality and performance
validation; earlier eager-expert measurements do not establish their behavior.

Model math lives in `models/deepseek_v41/dspark.py` and
`model_ops/deepseek_v41/dspark.py`. Tentative state belongs to the V4.1 cache
backend. Generic speculation code owns proposal execution, confidence scheduling
and sampling, without Engram hash or CSA2 compression formulas.

## Offline calibration

Dynamic verification requires `DSparkConfig(confidence_schedule=True, ragged=True,
calibration_profile="/path/to/profile.json")`. No V4 SPS curve or synthetic cost
stub is admitted for V4.1 dynamic scheduling. The explicit profile takes
precedence over automatic warmup calibration.

The frozen token IDs and fit/holdout partition are provided in
`tests/models/deepseek_v41/fixtures/dspark_calibration.json`; token IDs are the
authoritative prompt representation. With the pinned environment from the
runtime guide, collect and fit confidence:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.validate_dspark_runtime \
  --production --graph --collect-confidence --cases all --output-tokens 64 \
  --prompts tests/models/deepseek_v41/fixtures/dspark_calibration.json \
  --output /tmp/dspark-confidence.json
python -m tests.models.deepseek_v41.calibrate_dspark_confidence \
  --input /tmp/dspark-confidence.json --output /tmp/dspark-sts.json
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.calibrate_dspark_sps \
  --workload tests/models/deepseek_v41/fixtures/dspark_calibration.json \
  --output /tmp/dspark-sps.json
```

Run SPS collection with one model instance on GPUs 0–3. It controls verification
lengths for measurement and explicitly bypasses the production profile requirement;
all target/state operations still use the normal runtime. Then combine artifacts:

```bash
python -m tests.models.deepseek_v41.build_dspark_profile \
  --model /mnt/DeepSeek-V4.1-Flash \
  --sps /path/to/sps/result.json \
  --confidence /path/to/sts_candidate.json \
  --output /path/to/profile.json
```

The calibration results below were collected before the FusedMoE/AITER mHC
replacement; regenerate the profile for the current runtime before using dynamic
verification.

Profiles validate the config/index-file hashes, model type, TP size, cache types,
expert backend, graph setting, GPU name and Torch/HIP versions. They also validate
proposal width, batch coverage, positive finite temperatures and a non-increasing
SPS table. These hashes identify checkpoint metadata, not every weight byte.
Recalibrate after model/runtime changes or for a materially different workload.
The measured profile covers at most four concurrent requests and 24 verification
rows; it is an optional deployment artifact, not a universal model default.

The fitting workload used 48 frozen GSM8K **train** prompts (32 fit, 16 held out),
64 generated tokens, greedy sampling and batch four. The 966 aligned blocks were
split 634/332; unverified proposals are excluded. The sequential temperature
objective is cumulative-survival ECE with 15 bins. Held-out mean ECE improved
0.07544 to 0.06853 and Brier 0.13018 to 0.12772, while binary NLL increased
0.40537 to 0.40701. This is confidence calibration, not GSM8K accuracy.

SPS measures max-rank target-forward CUDA-event time for request counts 1/2/4
and query widths 1–6, with warmup and three repetitions. Shapes at request counts
one/two used repeated 128-token text; batch four used four frozen GSM8K prompts.
Medians are aggregated by total query rows, converted to a conservative monotone
latency envelope, then interpolated. This one-dimensional scheduler cost model
does not capture all context/routing variation; final performance is measured
separately with actual accepted outputs and draft cost included.

## Validation and reproduction

Use the pinned environment from [the runtime guide](deepseek_v41_runtime.md).
A real lifecycle check goes through production configuration admission:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.validate_dspark_lifecycle \
  --production --graph --sampling-probe \
  --calibration-profile /path/to/profile.json \
  --output /tmp/dspark-lifecycle.json
```

`lm_eval_dspark_runtime` evaluates the actual scheduler path. Multiple-choice
likelihoods score untouched full-vocabulary target logits while forcing known
continuations through the sampler/proposer, exercising verification and state
commit. GSM8K uses native proposals and sampling. Pair identical task/document
limits with `--baseline` and `--calibration-profile`, and compare artifacts using
`compare_dspark_quality`. Raw logits stay beside each result under `logits/`.

To reproduce the established five-shot GSM8K configuration, use the same
32-document limit and 256-token cap for each mode:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.lm_eval_dspark_runtime \
  --production --baseline --tasks gsm8k --fewshot 5 --limit 32 \
  --max-output-tokens 256 --output /tmp/gsm8k-baseline/result.json
```

Create the output directory before launching. Omit `--baseline` for fixed-length
DSpark and use a different output directory. Add `--calibration-profile` for the
optional dynamic mode. The expanded raw zero-shot check instead uses
`--fewshot 0 --limit 100`. Routine generation quality checks are capped at
100 requests, using the same first100 document IDs for every mode. Existing
32-document supplemental checks need not be expanded. `--chat` enables the published chat encoder. Keep those
prompt configurations separate when reporting results. Compare a completed pair
with `python -m tests.models.deepseek_v41.compare_dspark_quality --baseline
/path/to/baseline/result.json --candidate /path/to/candidate/result.json --output
/path/to/comparison.json`.

For isolated performance, run `validate_dspark_runtime --production --graph
--profile --repeats 3 --output-tokens 64 --cases all
--prompts tests/models/deepseek_v41/fixtures/dspark_performance.json` separately
in baseline (`--baseline`), fixed DSpark, and calibrated DSpark
(`--calibration-profile /path/to/profile.json`) modes. It reports actual finalized output
throughput, TTFT/TPOT, acceptance histograms, and target/draft CUDA-event costs.
Repeated English is a deliberately favorable acceptance case and should not be
presented as a production-wide speedup. Initialization, capture and warmup are
excluded from the measured cases. The harness synchronizes after each scheduler
step and deliberately ignores EOS to measure exactly 64 finalized outputs per
request. TTFT and TPOT are per-request means within each run, reduced by the
maximum across TP ranks; the comparison reports medians/ranges across three
runs. These are bounded scheduler measurements, not HTTP latency percentiles or
production goodput. Acceptance counts precede output-cap/stop truncation.

### State and shared-runtime regression

The final combined regression suite passed 1,031 tests with zero skips. It covers
V4.1 model/cache contracts, existing DSpark scheduling, shared request scheduling,
preemption, deferred outputs, multimodal runtime, Engram, checkpoint/transfer and
stochastic sampling. Black and whitespace checks pass; the nine Ruff findings
are unchanged from the parent commit.

Production admission with the calibrated profile passed nine real-checkpoint
lifecycle scenarios: mixed stochastic requests, prefill/decode preemption and
reorder, exact prefix fork, abort with a surviving request, abort after verify,
slot reuse/output cap, and stop/EOS inside an accepted block. The target performed
5,160 graph replays using captured token buckets 6/12/24. That high-acceptance
workload used verification shapes `(6,)` and `(6,6)`; it does not independently
prove dynamic contraction. A separate controlled schedule exercised ragged
widths across ten lifecycle scenarios and 12,600 graph replays. Native checkpoint
draft comparison matched all 20 proposal IDs against the reference draft math.

### Paired numerical and task checks

The first raw-completion matrix used 32 documents per task and 256 output tokens
for GSM8K. Across 514 teacher-forced continuations / 5,002 token positions, mean
NLL was 1.960597 without speculation and 1.962574 with calibrated DSpark: an increase
of 0.001977 nats/token, within the 0.01 numerical budget. Mean target-to-candidate
KL was 0.006451; top-1 agreement was 97.381%. There were 66 changed top-1 decisions
with baseline logit margin above 0.1. Thus these results do not establish bitwise
logit equivalence or unchanged non-near-tie decisions.

ARC Easy, HellaSwag and code-line description scores were unchanged. Chinese
LogiQA raw accuracy was 28/32 versus 27/32; normalized accuracy was 25/32 in both
runs (two gains and two losses).

The initial GSM8K results (raw 11/32 versus 9/32, and the first chat run) are
**invalid generation-quality evidence**. The adapter instantiated ModelRunner
without LLMEngine's tokenizer initialization, leaving `Config.eos_token_id=-1`.
It also omitted generation stop tokens. The model continued after EOS into
unrelated text, corrupting last-number extraction. This was an evaluation-adapter
bug, not a production model change. Those artifacts are preserved with
`generation_invalid.json` markers. Teacher-forced scores remain valid because
known continuations deliberately ignore EOS. The corrected adapter mirrors
LLMEngine's EOS/stop initialization and records output IDs and termination reasons.

The expanded Chinese LogiQA comparison used 128 documents: raw accuracy was
109/128 without speculation and 112/128 with DSpark (four gains, one loss);
normalized accuracy was 100/128 and 104/128 (four gains, zero losses). Across
512 continuations / 8,229 positions, mean NLL increased 0.000117 nats/token,
mean KL was 0.001402 and top-1 agreement was 99.441%. There were 24 top-1 changes
with baseline margin above 0.1. These figures include the earlier 32-document
Chinese subset; they must not be pooled as independent observations.

Corrected chat GSM8K used the published V4.1 encoder, zero-shot greedy generation,
32 test documents and a 256-token cap. Flexible-extraction accuracy was 27/32
without speculation and 30/32 with DSpark (three gains, no losses; exact McNemar
p=0.25). This bounded sample does not establish a model-wide improvement. Strict
`####` extraction scored zero in both runs because responses did not use that
answer delimiter.

The corrected **zero-shot raw** counterpart regressed from 23/32 to 17/32
(two gains, eight losses; paired 95% bootstrap interval for the change
[-0.375, 0.0], exact McNemar p=0.109375). Six losses hit the 256-token cap and
two ended at EOS. Both modes sometimes continued after giving a correct answer;
standard last-number extraction penalized DSpark more often. This is valid
negative generation evidence, not the earlier invalid-EOS result, and has not
been waived. P04/P09 used five-shot GSM8K. The matched five-shot comparison
expanded to 32 test documents with the same greedy/256-token configuration:
**both modes scored 29/32 under strict and flexible extraction, with identical
per-document correctness**. This does not cancel the separate zero-shot raw
regression. A subsequent fixed-length DSpark run scored 18/32 against the same
23/32 baseline (one gain, six losses), so the issue is not confined to confidence
scheduling. The expanded fixed-length/raw comparison completed on 128 questions
with unchanged prompt format, cap and scoring: **90/128 without speculation
versus 80/128 with fixed DSpark**, nine gains and 19 losses. The paired 95%
bootstrap interval for the change is [-0.15625, 0.0], exact McNemar p=0.08716.
This does not establish equivalence. Fifteen losses hit the output cap and four
ended at EOS. Both modes capped 30 of 128 responses overall, so this is not
simply an increase in cap frequency: capped-response accuracy was 9/30 without
speculation and 2/30 with fixed DSpark. On the additional 96 questions alone,
scores were 67/96 and 63/96; the decline is not confined to the original
32-question subset.
The separate default fixed-length five-shot check scored 30/32 versus the
non-speculative baseline's 29/32 under both strict and flexible extraction:
one gain and no losses. The calibrated five-shot result remains 29/32.

Repeating the fixed raw run in a new process as part of the 128-document matrix
changed three of the original 32 token sequences and one correctness result
(18/32 to 17/32). The runtime sources were identical; concurrent GPU workloads
differed. This demonstrates run-to-run variation under those conditions, without
identifying atomics or any one kernel as its cause. In contrast, the repeated
non-speculative baseline reproduced all 32 original output token sequences and
its 23/32 score exactly. Repeat variation alone does not resolve the observed
quality decline.

An initial Engram-only row-serialization experiment did not exercise the
intended intervention: PIECEWISE replay bypassed the Python hook installed on
Engram. Its 18/32 output reproduced the fixed run exactly and cannot exclude
Engram as a contributor. The artifact is marked invalid for that intervention.
A diagnostic graph-boundary fix now preserves input buckets/padding and records
actual intervention calls; a real GPU graph test covers verify versus prefill and
draft. This remains diagnostic code, not a product arithmetic change.

Calling the original attention kernel one row at a time scored 16/32 and did
not recover the baseline. These experiments do not demonstrate an attention
implementation defect or justify changing the V4 kernel. Further investigation
uses identical prefixes/state and traces the earliest differing stage, retaining
the original operators. P10 quality acceptance remains open.

### Independent history/output and persistent-state diagnostics

`lm_eval_dspark_runtime --production --audit-state --tasks gsm8k --limit 32
--fewshot 0 --max-output-tokens 256` runs the original scheduler and scoring with
extra assertions. Add `--baseline` for the non-speculative control. The audit
builds an independent raw-token ledger, derives compressed lookback by slicing
it, gathers expected Engram embeddings without the prefetch/staging cache, and
checks target-argmax acceptance and every finalized output token. It preserves
EOS, stop strings, output caps and PIECEWISE target execution. These extra CPU/GPU
synchronizations are diagnostic overhead, not performance measurements.

The fixed-mode audit passed all four ranks and reproduced all 32 original fixed
outputs exactly (18/32, versus the non-speculative 23/32). Per rank it checked
1,209 incoming/prepared/committed histories, 17,564 embedding rows, 6,870 verify
inputs and 4,101 finalized generated tokens. Draft acceptance counts 0–5 and
live request counts 1–4 were exercised. This finds no history, embedding staging,
acceptance-length or emitted-token corruption in that workload; it does not
establish floating-point cache equivalence or resolve the quality gap.

The non-speculative control also passed all four ranks and reproduced all32
baseline sequences and termination reasons exactly (23/32). Per rank it checked
4,063 histories at each boundary,11,822 embedding rows and3,999 finalized generated
tokens. Neither audit changed the compared generation results.

For a bounded first-divergence fixture, `validate_dspark_runtime --trace
--trace-batched --trace-persistent` retains independently replayed accepted state
across verification blocks. It supports one case/repetition with no later
prefill. Cursor/tails roll back to the selected accepted prefix; visible window
and global rows are compared separately, excluding rejected/expired storage and
draft-layer windows. Layer hooks require eager execution, and the emitted prefix
must match the actual graph-mode quality artifact before interpreting a trace.

For GSM8K document4, both executions first emit tokens223/397. Replaying only the
second block from the speculative state still selects22185, while retaining
one-token replay state from the first block recovers baseline token271. The
first accepted anchor has matching small-state cursors/tails, but its layer8
window at position111 and compressed main/index row55 differ. The next block's
new layer8 window at position112 still agrees. Thus a current-block-only shadow
can miss numerical differences carried by previously accepted cache rows. The
first block's position112 contains a rejected draft token856 and cannot stand
in for the later real token397 at that position. No V4 attention/inverse RoPE
kernel is changed; this case does not establish an attention implementation
bug or explain all paired quality losses.

## Isolated performance

TP4 on GPUs 0–3, BF16 caches, eager experts, target PIECEWISE graphs, eager draft. Each point has per-case warmup and three repetitions of 64 finalized outputs per request. All three complete runs use the same frozen workload SHA256.

| Workload | Non-spec tok/s | Fixed DSpark tok/s | Calibrated DSpark tok/s | Fixed / base | Calibrated / base |
|---|---:|---:|---:|---:|---:|
| English9 | 4.767 | 5.562 | 6.974 | 1.167x | 1.463x |
| Repeated English129 | 4.616 | 12.721 | 12.615 | 2.756x | 2.732x |
| Chinese9 + repeated English257 | 7.038 | 11.750 | 10.492 | 1.670x | 1.491x |
| GSM8K train holdout batch4 | 11.311 | 17.115 | 15.464 | 1.513x | 1.367x |

Fixed DSpark remains the default after explicitly selecting speculation. The optional calibrated profile improves the short-English case by 25.4% relative to fixed verification, but is 10.7% slower on mixed requests and 9.6% slower on the four-prompt math batch. It is not installed as a universal default. All speculative throughput ranges remain above their matching baseline ranges, including the mixed baseline outlier.

| Workload | TTFT ms: base / fixed / calibrated | TPOT ms: base / fixed / calibrated | Acceptance: fixed / calibrated |
|---|---|---|---|
| English9 | 603.70 / 731.82 / 745.88 | 203.54 / 171.05 / 133.84 | 23.64% / 57.81% |
| Repeated English129 | 1112.98 / 1282.35 / 1290.54 | 202.39 / 59.43 / 60.05 | 100.00% / 100.00% |
| Chinese9 + repeated English257 | 1758.51 / 1930.74 / 1939.41 | 260.79 / 114.09 / 120.73 | 60.57% / 70.07% |
| GSM8K train holdout batch4 | 2134.52 / 2456.91 / 2433.93 | 325.38 / 180.18 / 196.24 | 53.25% / 57.68% |

TTFT increases in every measured speculative case; decode TPOT improves. Acceptance counts precede cap/stop truncation. Repeated English has 100% acceptance and is a deliberately favorable case, not a representative production speedup. TTFT/TPOT are request means within a run, max-reduced across ranks, then summarized across runs; they are not latency percentiles.

| Workload | Target verify median ms: base / fixed / calibrated | Draft median ms: fixed / calibrated |
|---|---|---|
| English9 | 202.09 / 314.76 / 256.00 | 20.83 / 21.62 |
| Repeated English129 | 201.14 / 316.88 / 318.62 | 20.44 / 21.73 |
| Chinese9 + repeated English257 | 257.75 / 435.22 / 306.08 | 21.66 / 20.67 |
| GSM8K train holdout batch4 | 324.31 / 599.03 / 466.91 | 26.03 / 27.45 |

These are medians of per-forward CUDA-event costs after taking the maximum over TP ranks; dynamic verification mixes request counts and query lengths. The full report retains counts, ranges, acceptance histograms and verification-shape frequencies.

Calibrated production execution used widths 1/2/3/6 for short English; mixed requests included `(1,6)`, `(2,6)` and `(4,6)`. The four-prompt workload exercised many unequal shapes and single-row verification. Repeated English stayed at width6. This provides actual dynamic contraction coverage in addition to the earlier controlled lifecycle probe.

One mixed baseline repetition fell to 5.893 tok/s versus 7.108 and 7.038, with identical outputs and slower target execution. It is retained in the range. The speculative ranges remain well separated, so this outlier does not reverse the throughput conclusion. Clocks were not locked.

Peak PyTorch allocation was 72.605–72.664 GiB/rank without speculation and 74.933–74.992 GiB/rank with speculation; these are not total device-memory readings. BF16 STATE per request grows from 5,256,192 to 5,869,568 bytes for the three additional draft windows and five-row ring slack.

The harness synchronizes each scheduler step and ignores EOS for this fixed-output benchmark. No HTTP goodput, natural-completion throughput, long-context scaling, packed speculative cache, AITER speculative experts or draft graph performance is claimed.


### Current100-request acceptance scope

The full1319-document raw runs were stopped at the user's request to keep routine
quality evaluation to100 requests. Their already generated first100 document IDs
were frozen and rescored through the unchanged standard lm_eval task, with every
generation argument checked. The source full runs remain incomplete. The bounded
raw result is71/100 without speculation and62/100 with fixed DSpark, five gains
and14 losses (paired95% interval[-17,-1] percentage points; exact McNemar p=.063568).
This exceeds the1-percentage-point practical budget. P10 is not yet accepted.
Earlier32/128 cohorts overlap these100 and are retained as historical evidence.
Calibrated raw100 scored64/100 against baseline71/100, also outside the quality
budget. Fixed-mode Chinese100 improved from84 to88 raw and80 to84 normalized;
chat32 stayed27/32. Fixed production lifecycle passed nine scenarios and5040
target graph replays. The corrected Engram-only diagnostic actually executed
but scored17/32 against23/32 baseline; it did not fix the raw decline.

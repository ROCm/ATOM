# DeepSeek-V4.1 DSpark

Native DSpark uses the checkpoint's three draft stages to propose five tokens.
Target verification processes the anchor plus up to five proposals and commits
only the accepted input prefix. It reuses ATOM's DSparkProposer, VerifyScheduler,
request scheduler and unchanged V4 BF16 attention kernels.

**Quality acceptance is open.** Fixed DSpark scores below the non-speculative
baseline on raw-completion GSM8K by more than the practical budget; see
[Open quality regression](#open-quality-regression) before enabling it.

## Supported configuration

The validated configuration is TP4 with whole-expert EP, BF16 KV, the FP8 index
plane and text requests. Target execution can be eager, one whole-forward graph
per decode step (`FULL`), or PIECEWISE; the draft has its own graph.

Not admitted: packed speculative caches, multimodal speculation, synthetic
acceptance, relaxed MTP acceptance, and speculative output token logprobs — the
current output protocol cannot return the last of these correctly. Both target
and draft MoE reuse V4 FusedMoE. Non-speculative vision and packed cache support
are independent of all of this.

Fixed-length verification is the default when DSpark is explicitly selected:

```python
from atom.config import Config, SpeculativeConfig

model = "/mnt/DeepSeek-V4.1-Flash"
config = Config(
    model=model,
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    kv_cache_dtype="bf16",
    index_cache_dtype="fp8",
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
`compilation_config=CompilationConfig(level=0, cudagraph_mode=CUDAGraphMode.FULL)`.
A decode step is then two replays: the draft's, keyed by request count, and the
target's, keyed by `(batch size, query bucket)`. `PIECEWISE` remains available
and records the dense pieces only. Capture must not bind warmup slots or another
request's window; `validate_runtime --graph` is what asserts it does not.

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

Model math lives in `models/deepseek_v41/dspark.py` and
`model_ops/deepseek_v41/dspark.py`. Tentative state belongs to the V4.1 cache
backend. Generic speculation code owns proposal execution, confidence scheduling
and sampling, without Engram hash or CSA2 compression formulas.

## Dynamic verification and its calibration profile

Dynamic verification requires `DSparkConfig(confidence_schedule=True,
ragged=True, calibration_profile="/path/to/profile.json")`. No V4 SPS curve or
synthetic cost stub is admitted for V4.1, and an explicit profile takes
precedence over automatic warmup calibration.

A profile carries two fitted objects. **Confidence temperatures** rescale the
drafter's per-position survival probabilities; they are fitted against
cumulative-survival ECE on held-out blocks. **An SPS table** maps total query
rows to target-forward throughput, aggregated from max-rank CUDA-event medians
and converted to a conservative monotone latency envelope before interpolation.
This one-dimensional cost model does not capture context or routing variation.

Loading validates the config and index-file hashes, model type, TP size, cache
types, expert backend, graph setting, GPU name and Torch/HIP versions, plus
proposal width, batch coverage, positive finite temperatures and a non-increasing
SPS table. Those hashes identify checkpoint metadata, not every weight byte.
Recalibrate after a model or runtime change, or for a materially different
workload; a profile is an optional deployment artifact, not a model default.

The offline fitting harness is not shipped with the repository. The profile that
was fitted for this configuration covered at most four concurrent requests and
24 verification rows, so it does not generalize to a wider deployment on its own.

## Validation

Use the pinned environment from [the runtime guide](deepseek_v41_runtime.md).
A real lifecycle check goes through production configuration admission:

```bash
torchrun --standalone --nproc_per_node=4 \
  -m tests.models.deepseek_v41.validate_dspark_lifecycle \
  --production --graph --sampling-probe \
  --output /tmp/dspark-lifecycle.json
```

This is a manual GPU gate, not a pytest module; `tests/models/deepseek_v41/` and
`tests/attentions/deepseek_v41/` hold the automated coverage.

Production admission with a calibrated profile passed nine real-checkpoint
lifecycle scenarios: mixed stochastic requests, prefill/decode preemption and
reorder, exact prefix fork, abort with a surviving request, abort after verify,
slot reuse/output cap, and stop/EOS inside an accepted block. The target
performed 5,160 graph replays using captured token buckets 6/12/24. That
high-acceptance workload used verification shapes `(6,)` and `(6,6)`, so it does
not independently prove dynamic contraction; a separate controlled schedule
exercised ragged widths across ten scenarios and 12,600 graph replays. Native
checkpoint draft comparison matched all 20 proposal IDs against the reference
draft math.

## Paired numerical checks

Teacher-forced continuations, 32 documents per task, 256 output tokens for
GSM8K. Across 514 continuations / 5,002 token positions, mean NLL was 1.960597
without speculation and 1.962574 with calibrated DSpark — an increase of 0.001977
nats/token, inside the 0.01 budget. Mean target-to-candidate KL was 0.006451 and
top-1 agreement 97.381%, with 66 changed top-1 decisions at a baseline logit
margin above 0.1. These results do not establish bitwise logit equivalence or
unchanged non-near-tie decisions.

The expanded Chinese LogiQA comparison used 128 documents. Across 512
continuations / 8,229 positions, mean NLL increased 0.000117 nats/token, mean KL
was 0.001402 and top-1 agreement 99.441%, with 24 top-1 changes above the same
margin. It includes the earlier 32-document subset and must not be pooled with
it as independent observations.

## Open quality regression

Generation quality, unlike the teacher-forced numbers above, is outside budget.

| Check | Without speculation | With fixed DSpark |
|---|---:|---:|
| GSM8K raw, zero-shot, 100 documents | 71/100 | 62/100 |
| GSM8K raw, zero-shot, 128 questions | 90/128 | 80/128 |
| GSM8K, five-shot, 32 documents | 29/32 | 29/32 |
| Chinese LogiQA raw, 100 documents | 84/100 | 88/100 |
| Chinese LogiQA normalized, 100 documents | 80/100 | 84/100 |

The 100-document raw result is five gains and 14 losses (paired 95% interval
[-17, -1] percentage points, exact McNemar p = 0.0636) — **beyond the
one-percentage-point budget**. Calibrated DSpark scores 64/100 against the same
71/100 baseline, so the decline is not confined to confidence scheduling. The
128-question result is nine gains and 19 losses (paired 95% bootstrap
[-0.156, 0.0], exact McNemar p = 0.0872); it overlaps the 100-document cohort
and is not an independent observation.

The five-shot comparison is identical per document, and chat-encoder zero-shot
scored 27/32 without speculation against 30/32 with it. Neither cancels the raw
regression: they are different prompt configurations, and the raw loss is where
the budget is spent.

What the losses look like: 15 of the 128-question losses hit the 256-token
output cap and four ended at EOS. Both modes capped 30 of 128 responses, so cap
frequency is not the mechanism — capped-response accuracy was 9/30 without
speculation and 2/30 with DSpark. Both modes sometimes continue after giving a
correct answer, and standard last-number extraction penalizes DSpark more often.

What has been ruled out:

- **Not an evaluation-adapter artifact.** An earlier adapter left
  `Config.eos_token_id = -1` and omitted stop tokens, letting the model continue
  past EOS into unrelated text. Those runs are invalid and are not quoted here.
  The corrected adapter mirrors LLMEngine's EOS/stop initialization and records
  output IDs and termination reasons.
- **Not history, embedding staging, acceptance length or emitted tokens.** An
  independent raw-token ledger audit — deriving compressed lookback by slicing
  it, gathering expected Engram embeddings without the prefetch cache, and
  checking target-argmax acceptance and every finalized output token — passed on
  all four ranks and reproduced the compared outputs exactly. It does not
  establish floating-point cache equivalence.
- **Not the V4 attention kernel.** Calling the original kernel one row at a time
  scored 16/32 and did not recover the baseline. No attention or inverse-RoPE
  kernel is implicated, and none has been changed.
- **Not solely run-to-run variation.** Repeating the fixed raw run in a new
  process changed three of 32 token sequences and one correctness result, while
  the non-speculative baseline reproduced all 32 sequences and its score exactly.
  Variation is real but does not account for the gap.

What remains suspected: numerical difference carried by *previously accepted*
cache rows. In a first-divergence trace on one GSM8K document, replaying only
the current block from speculative state still diverged, while retaining
one-token replay state from the preceding block recovered the baseline token.
The first accepted anchor had matching cursors and tails, but its layer-8 window
at position 111 and compressed main/index row 55 differed, and the next block's
new layer-8 window at position 112 agreed again. A current-block-only shadow
therefore cannot see the difference. Further investigation should use identical
prefixes and state, trace the earliest differing stage, and retain the original
operators.

## Performance

Fixed DSpark remains the default once speculation is explicitly selected;
the optional calibrated profile is not installed as a universal default, because
it wins on short single requests and loses on mixed and batched ones.

Directionally, on the configuration above: TTFT rises in every speculative case
(the draft pass is added before the first token) while decode TPOT falls, and
acceptance varies from roughly a quarter on short English to 100% on repeated
text — the latter is a deliberately favorable case and not a production speedup.
Peak PyTorch allocation grows about 2.3 GiB per rank, and BF16 STATE per request
grows from 5,256,192 to 5,869,568 bytes for the three extra draft windows and
five-row ring slack.

The measured tables that accompanied this section were taken on eager experts,
before the FusedMoE and AITER mHC paths replaced them, so they no longer
describe this code. Re-measure on the current path with the standard serving
benchmark before quoting any speedup.

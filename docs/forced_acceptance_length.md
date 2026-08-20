# Forced acceptance length

Speculative throughput is dominated by one number: how many tokens each target
forward emits. Two engines serving the same model at the same batch size report
very different throughput when their draft heads agree with the target at
different rates, so a raw speculative benchmark measures draft quality as much
as it measures the serving system.

`--spec-decode-acceptance-length` takes that variable out. The rejection sampler
stops comparing draft tokens against the target and instead accepts them with a
fixed per-position probability, chosen so the run converges on the mean
acceptance length you asked for. What stays under measurement is the system:
attention, scheduling, graph capture, and the drafter's own cost.

Two situations call for it:

- **Bring-up.** Benchmark the speculative path while a draft head is still
  training and its real acceptance is not yet representative.
- **Cross-engine comparison.** Replay a published acceptance-length figure, such
  as an [InferenceX golden AL](https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/README.md),
  so an ATOM number and a vLLM or SGLang number describe the same workload.

> **This is a benchmarking knob.** Accepted tokens never get checked against the
> target, so the generated text is not the model's output. Never run an accuracy
> evaluation with it enabled.

For the speculative framework that hosts it, see
[`serving_benchmarking_guide.md` § Speculative decoding](serving_benchmarking_guide.md#speculative-decoding-mtp).

## Quick start

```bash
python -m atom.entrypoints.openai_server \
  --model moonshotai/Kimi-K3 \
  --draft-model Inferact/Kimi-K3-DSpark \
  --tensor-parallel-size 8 \
  --kv_cache_dtype fp8 \
  --method dspark \
  --num-speculative-tokens 7 \
  --spec-decode-acceptance-length 3.78 \
  --trust-remote-code \
  --server-port 7777
```

The resolved schedule is logged once at startup, so a run that quietly failed to
pick the flag up is easy to spot:

```text
Forced speculative acceptance ON: mean acceptance length 3.7800 over 7 draft
positions (per-position rates [1.0, 1.0, 0.78, 0.0, 0.0, 0.0, 0.0]). Throughput
numbers from this run are synthetic; output text and accuracy are meaningless.
```

## The two spellings

| Flag | Range | Meaning |
|---|---|---|
| `--spec-decode-acceptance-length` | `[1, num_speculative_tokens + 1]` | Mean acceptance length (AL), counting the target's own guaranteed token |
| `--spec-decode-acceptance-rate` | `[0, 1]` | The same target as a mean acceptance rate, accepted draft tokens over drafted slots |

They describe the same curve, so setting both is rejected at startup. With
`n = num_speculative_tokens`:

```text
length = 1 + n * rate
rate   = (length - 1) / n
```

Prefer the length form. Published figures are quoted as lengths, so the rate
form only inserts a conversion step where a rounding slip can move the target
without anything complaining.

## What the number means

Acceptance length counts the token the target produces on its own. It is
therefore at least `1` even when every draft token is rejected, and at most
`n + 1` when all `n` are accepted. A length of `3.78` over 7 draft positions
means each target forward emits 3.78 tokens on average: its own, plus 2.78
accepted draft tokens.

That convention is deliberate — it matches vLLM's `synthetic_acceptance_length`
and SGLang's `SGLANG_SIMULATE_ACC_LEN`, so a published AL can be passed through
unchanged, with no off-by-one and no division.

## The schedule it resolves to

Any per-position schedule summing to `length - 1` hits the requested mean, but
they are not interchangeable: the spread around that mean drives ITL tails and
how a batch drains, so two engines quoting the same AL can still produce
different latency distributions.

ATOM resolves the length to the minimum-variance schedule — accept
`floor(length - 1)` positions outright, and the next one with the leftover
fraction. At length `3.78` over 7 positions:

| Draft position | 1 | 2 | 3 | 4–7 |
|---|---|---|---|---|
| P(accepted) | 1.00 | 1.00 | 0.78 | 0 |

Every step accepts either 2 or 3 draft tokens, 22% / 78%, and never anything
else. This is what vLLM resolves `synthetic_acceptance_length` to and what
SGLang's `match-expected` draws, so the accepted-length *distribution* matches
across engines and not merely its mean.

Accepted positions emit the draft's own token IDs, so the drafter still runs and
is still paid for on every step. Only the accept/reject decision is synthetic.

> `--spec-decode-acceptance-rate` previously resolved to a geometric decay,
> which hits the same mean with about 15× the variance (2.63 against 0.17 at
> length 3.78). It now shares the schedule above, so throughput numbers taken
> before that change are not comparable to ones taken after.

## Confirming the run hit the target

`GET /debug/mtp_stats` reports what actually happened:

```json
{
  "enabled": true,
  "total_draft_tokens": 1400000,
  "total_accepted_tokens": 553980,
  "acceptance_rate": 0.3957,
  "average_tokens_per_forward": 3.7699,
  "distribution": {"2": 46020, "3": 153980},
  "distribution_percent": {"2": 0.2301, "3": 0.7699}
}
```

`average_tokens_per_forward` is the realized acceptance length; compare it
against what you asked for. The figures above are from a Kimi-K3 +
Kimi-K3-DSpark TP8 run at `--spec-decode-acceptance-length 3.78`, which came
back at 3.7699 with the accepted-count split falling where the schedule predicts.

`acceptance_rate` counts accepted draft tokens over drafted *slots*, so at
length 3.78 over 7 positions it reads `2.78 / 7 ≈ 0.396`. That is the rate form
of the same target, not a shortfall.

The same values are on `/metrics` as `atom:mtp_average_tokens_per_forward`,
`atom:mtp_accepted_tokens`, and `atom:mtp_decode_steps{accepted_tokens="..."}`.

## Replaying an InferenceX golden AL

Under the AgentX fairness guidelines each (model, thinking mode, draft length)
combination has one committed golden AL, and every submission for that
combination replays it instead of quoting its own measurement. The curves live
in [`golden_al_distribution/`](https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/README.md)
as YAML keyed by `num_speculative_tokens`. For Kimi-K3 DSpark with thinking on:

| `--num-speculative-tokens` | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| `--spec-decode-acceptance-length` | 1.84 | 2.45 | 2.87 | 3.22 | 3.44 | 3.54 | 3.78 | 3.88 |

Pick the column for the draft length you are running and pass the value
straight through. A submission may choose any supported draft length, but not a
different acceptance target for it.

Those curves are measured on the coding category of the SPEED-Bench Qualitative
split, with real draft decoding and the model's production sampling settings.
ATOM replays them; it does not collect them.

## Restrictions

**No accuracy evaluation.** Accepted tokens come from the draft without ever
being compared against the target, so anything measuring quality — `lm_eval`,
gsm8k, a golden-output diff — is measuring noise.

**Not with the DSpark confidence scheduler.** The confidence scheduler
(`--dspark-config '{"confidence_schedule": true}'`, and the `ragged` path that
requires it) sizes each request's verify length `ell_r` at runtime. When `ell_r`
comes back shorter than the schedule needs, the run lands under the length it
was asked to reproduce — at length 3.78 over 7 positions, exact while
`ell_r >= 3`, but 3.39 at `ell_r = 2` and 2.89 at `ell_r = 1`. Because `ell_r`
is a runtime decision there is no upfront check that would catch the shortfall,
and a benchmark reporting a number it never hit is worse than one that refuses
to start, so ATOM rejects the combination at startup. Drop
`confidence_schedule` for forced-acceptance runs; see
[`recipes/DSpark.md`](../recipes/DSpark.md).

**Needs a speculative method.** The flag requires `--method` and a non-zero
`--num-speculative-tokens`. Without draft positions there is no schedule to
resolve, and startup fails rather than silently ignoring the flag.

## How it works

`SpeculativeConfig._resolve_synthetic_acceptance` (`atom/config.py`) validates
whichever knob was set, converts a rate into a length, and calls
`acceptance_length_to_rates` (`atom/model_ops/rejection_sampler.py`) to produce
the per-position unconditional rates. Everything downstream reads only the
resolved `synthetic_acceptance_rates`, so neither spelling survives past config.

`rejection_sample` then dispatches `rejection_synthetic_sample_kernel` in place
of the greedy kernel. The kernel walks positions in order and stops at the first
rejection, so it needs `P(accept i | accepted through i-1)` rather than the
unconditional rates the config carries; the conversion happens once and is
cached per device. On rejection it emits the target's argmax as the correction
token, keeping the output layout and `num_bonus_tokens` semantics identical to
the real path.

The accept/reject draw has to be bit-identical on every TP rank. `sampled_tokens`
is broadcast from rank 0 while `num_bonus_tokens` stays local, so a per-rank
`torch.rand` would desync the two and leave the anchor gather reading a rejected
column — the draft then embeds an invalid token ID and the run dies on an HSA
out-of-bounds fault. A dedicated device generator, re-seeded each step from a
counter that advances in SPMD lockstep, draws the same Philox stream everywhere.

Those uniforms are drawn on-device deliberately. Drawing on the host costs a
pageable H2D copy, which PyTorch issues as a memcpy plus a stream sync, landing
exactly between verify and propose: the caller would block until the whole
target forward drained, and the draft model's kernels could not be dispatched
behind it.

# Dynamic Chunked Pipeline Parallelism (Dynamic CPP)

When a long prompt is prefilled under Pipeline Parallelism (PP), the prompt is
split into chunks and the chunks are pushed through the stages as microbatches.
The chunks are not interchangeable: chunk `i` must attend over everything
already cached, so a fixed chunk size produces microbatches whose cost grows
along the prompt. A pipeline whose microbatches have unequal cost stalls on the
expensive ones.

**Dynamic CPP** picks a size for every chunk so that all chunks of one prompt
take roughly the same time. Early chunks stay large, later chunks shrink as the
cached prefix grows.

```
  fixed 32768, ISL 128K:      [32768][32768][32768][32768]
  cost per chunk:              572ms  1287ms 2002ms 2717ms   <- every stage waits
                                                                on the last chunk

  dynamic, same prompt:       [32768][16192][12416][10368][9088][8192]...[3840]
  cost per chunk:              572ms each (440ms remainder)  <- nothing to wait on
```

(Measured coefficients for Kimi-K2.5-MXFP4 on MI355X; see
[Measured results](#measured-results).)

- [When to use it](#when-to-use-it)
- [Enabling it](#enabling-it)
- [How the chunk size is chosen](#how-the-chunk-size-is-chosen)
- [Calibrating the latency model](#calibrating-the-latency-model)
- [Measured results](#measured-results)
- [Limits and gotchas](#limits-and-gotchas)
- [Reproducing](#reproducing)

> **Status.** Off by default, and meant to stay that way. Equal-latency chunking
> works and is worth having for **single-stream** long prefill once the latency
> model is calibrated on real requests. It costs 10-15% throughput as soon as
> two or more requests prefill concurrently, and that cost is a property of
> splitting a prompt into more chunks rather than a bug in the concurrency
> guard — see
> [At concurrency it loses throughput](#at-concurrency-it-loses-throughput).

## When to use it

Dynamic CPP only removes pipeline bubbles, so it only pays off when the pipeline
has bubbles to remove:

- **Good fit**: `pipeline_parallel_size > 1`, long prompts (tens of thousands of
  tokens and up), and **one request prefilling at a time** — a latency-bound
  long-context deployment, or a disaggregated prefill worker whose arrivals are
  spaced further apart than a prefill takes.
- **Harmful**: two or more concurrent prefills. Each request already feeds the
  stages its own chunks, so the budget refills from them no matter what this one
  is given: a smaller chunk removes no bubble and only re-pays the per-chunk
  prefix cost. The scheduler detects this and keeps chunks fixed, but the window
  in which the condition holds is narrow, and enabling the feature on a queue
  that is never empty is a 10-15% throughput loss.
- **No effect**: `pipeline_parallel_size == 1`. There is no pipeline to balance.

The gain is largest for architectures that pay a per-chunk cost proportional to
the cached prefix. MLA models rebuild the prefix from its compressed latent
representation on every chunk, so their per-chunk cost has a floor that grows
with the prefix, on top of the usual attention area term.

## Enabling it

```bash
python3 -m atom.entrypoints.openai_server \
  --model /path/to/model \
  --pipeline-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --enable-dynamic-chunking \
  --dynamic-chunking-smooth-factor 1.0
```

| Flag | Meaning |
| --- | --- |
| `--enable-dynamic-chunking` | Turn the feature on. Requires `pipeline_parallel_size > 1`. |
| `--dynamic-chunking-min-chunk-size` | Floor on the solved chunk size. Default `4096`. |
| `--dynamic-chunking-smooth-factor` | Blend between the base chunk (`0.0`) and the solver's answer (`1.0`). |
| `--max-num-batched-tokens` | The chunk the solver equalizes against, and the ceiling it is clamped to. |

Nothing has to be calibrated by hand. The latency model is measured by the server
itself, partly at startup and partly from the first real prefills, and chunking
stays fixed until it is ready — see
[Calibrating the latency model](#calibrating-the-latency-model).

## How the chunk size is chosen

One chunk of `x` new tokens on top of a cached prefix of `L` tokens is modeled
as

```
  t(L, x) = c + gamma*L + a*(2*L*x + x^2) + b*x
            ^^^^^^^^^^^^   ^^^^^^^^^^^^^^   ^^^
            per-chunk       attention        work
            floor           area             linear
                                             in x
```

- `a` scales the attention area. `2*L*x` is the new tokens attending over the
  prefix, `x^2` the block-causal part within the chunk.
- `b` scales everything linear in the chunk: MLP, MoE, projections.
- `c` is the fixed per-chunk cost.
- `gamma` scales the part that depends on the prefix but **not** on `x` — for
  MLA, decompressing the cached prefix. This term is why a chunk cannot be made
  arbitrarily cheap by making it small.

The scheduler asks for the `x` that makes a chunk after prefix `L` cost the same
as a base-size chunk after no prefix:

```
  t(L, x) = t(0, base_chunk)
```

which is a quadratic in `x`. `gamma*L` is a floor that is spent before any
`x`-dependent work happens, so it is subtracted from the budget first; if nothing
is left, there is no positive solution and the chunk stays at the base size. The
answer is then clamped to
`[--dynamic-chunking-min-chunk-size, --max-num-batched-tokens]` and aligned to
the KV block size.

The chunk the equality is taken against is `--max-num-batched-tokens`, the same
knob SGLang (`--chunked-prefill-size`) and vLLM-Ascend use for this. Because the
solver only ever returns something at or below it, running dynamic chunking at
the budget that is best for fixed chunking can only grow the chunk count — and
every added chunk re-pays `gamma*L`. Set the budget to 2-3x the best fixed chunk
size, as both references recommend, so the count stays roughly neutral while the
chunks even out.

The relevant code is `ChunkSizePredictor` in
`atom/model_engine/dynamic_chunking.py` and `Scheduler._dynamic_chunk_limit` in
`atom/model_engine/scheduler.py`.

## Calibrating the latency model

The four coefficients are **configuration-specific**. They change with the
model, the PP layer split, `kv_cache_dtype`, `attn_prefill_chunk_size`,
`max_model_len` and the attention backend, so they are measured on the running
server. No single measurement can identify all four, so it happens in two parts.

**Startup: `b` and `c`, from a dummy chunk sweep.** ATOM's dummy forwards return
early from attention — see the `is_dummy_run` guards in `attention_mla.py`,
`deepseek_v4.py` and `attention_mha.py` — so a dummy chunk sweep measures exactly
the MLP, MoE and projection cost and nothing else. That is a clean fit for the
two terms that carry no attention, and worthless for the other two. On
Kimi-K2.5-MXFP4 it is also the whole explanation for why early Dynamic CPP
measurements came out flat to slightly negative, back when the same sweep was
used to fit all four:

| Coefficient | Dummy sweep | Real prefills | Ratio |
| --- | --- | --- | --- |
| `a` (attention area) | 1.3e-10 | 3.25e-07 | ~2400x low |
| `b` (linear in chunk) | 6.45e-03 | 6.03e-03 | ~1x |
| `c` (per chunk) | 12.2 | 26.1 | ~2x low |
| `gamma` (prefix rebuild) | 2.3e-05 | 5.39e-04 | ~24x low |

Both attention terms collapse and the chunk-linear term is right, which is what
"attention did not run" looks like. Fed that model the solver has nothing to
solve: at a 98304-token prefix it asked for ~34000 tokens, above the 32768
ceiling, so the chunk was clamped back to the base size and the chunk **count**
never changed. It was never an algorithmic problem with equal-latency chunking.

**Serving: `a`, `gamma` and `c`, from a sweep over real prefills.** With `b`
known, the residual of a measured prefill is linear in the three remaining
unknowns:

```
  measured - b*x = a * (2*L*x + x^2) + gamma * L + c
```

`c` is refitted here rather than taken from startup because a dummy forward
reaches `b` without ever setting attention up, so its constant is not the one a
real chunk pays. Pinned, that error lands on `gamma` — the term the whole feature
turns on.

Telling the three apart needs chunk sizes that vary **independently of the cached
prefix**, and serving traffic never supplies that on its own:

- At one fixed chunk size, `2*L*x + x^2` is itself affine in `L`, so the three
  terms are not merely ill-conditioned but unidentifiable. Fitting anyway
  recovered a `gamma` around 60% high on Kimi-K2.5-MXFP4, which cost the feature
  its entire margin over a tuned fixed chunk.
- Samples taken off a solved ladder are worse, not better: there the chunk is a
  decreasing function of the prefix by construction. Freeing `c` against ladder
  samples sent it to 95ms against a true 20ms.

So until a model exists, `Scheduler._calibration_sweep_chunk` alternates requests
between `--max-num-batched-tokens` and a quarter of it
(`CALIBRATION_SWEEP_RATIO`), each base carried across a whole prompt. Two bases
that far apart put the solved ladder within about 5% of the one the true
coefficients give and keep it there under sampling noise; at 2x apart the same
sweep leaves a tail of fits five times worse. Four requests are enough, which
usually means the model is installed before a benchmark's warmup ends.

`ChunkLatencyCalibrator` in `atom/model_engine/dynamic_chunking.py` owns the fit;
the mechanics are:

- **Sampling** is on whenever the feature is. Each worker times its own forwards
  between the PP receive and the PP send, so a sample measures the chunk rather
  than how well the stage overlapped with its neighbours. The events are read
  only once `query()` reports them complete, one or two forwards later, so
  nothing on the serving path ever synchronizes.
- **Only single-sequence prefills** are sampled. A batch carrying more than one
  sequence spreads its time over shapes the model cannot separate. This costs
  nothing, because the solver only acts on a request that is the pipeline's sole
  prefill anyway.
- **The process's first request is discarded** (`DISCARD_FIRST_CHUNK_REQUESTS`).
  Its forwards pay kernel autotuning and lazy allocator growth, running up to 60x
  slower than the same shape does later, and by a different factor per chunk size
  — so keeping them hands the sweep an offset between its two bases that the
  fitted constant absorbs and `gamma` pays for. Counting in requests rather than
  forwards is the point: a fixed number of forwards covers most of a large-chunk
  request and a fraction of a small-chunk one.
- **Timings are grouped by `(chunk, prefix)` shape and reduced by median**, so a
  straggler forward moves the fit far less than it moves one sample.
- **The fit runs once**, when the samples reach `MIN_CALIBRATION_SHAPES` shapes
  spanning `MIN_CALIBRATION_PREFIXES` prefixes and `MIN_CALIBRATION_CHUNK_SIZES`
  chunk sizes. The sweep that conditioned it ends with it, so sampling stops
  there too — every later sample would come off the ladder, where the terms
  cannot be separated. A fit that raises is retried once there is a shape the
  last attempt did not see.
- **The PP head's EngineCore collects the result** every
  `DYNAMIC_CHUNKING_POLL_STEPS` steps and installs it in the scheduler. A fit is
  rejected if its residual exceeds `MAX_CALIBRATION_RESIDUAL_FRACTION` of the
  mean sampled time, if it measured no attention growth, or if it predicts no
  useful shrink one full chunk into a prompt. A rejected model still ends the
  sweep, and chunking stays fixed at the configured budget.

The model is fitted on the PP head, because that is the stage whose EngineCore
schedules. Measured across a PP4 x TP1 prefill worker the per-stage fits sit
within 4% of each other (equal-latency budget 550ms on stage 0 against 572ms on
the slowest stage), so this costs a few percent of accuracy and no collectives.

On Kimi-K2.5-MXFP4 (prefill PP4 x TP1, `fp8` KV, 128K prompts) the coefficients
this converges to are

```
  a = 3.25e-07   b = 6.03e-03   c = 26.1   gamma = 5.39e-04
```

and an offline fit of the same model on a standalone (non-disaggregated) PP4
server landed on `a = 3.21e-07, b = 6.25e-03, gamma = 6.18e-04` — close enough to
treat these as a property of the model and hardware rather than an artifact of
the fit. Feeding the sweep a deliberately biased `b` (7% high) moves the solved
chunk sizes by under 1%, because an error that is constant in the prefix lands in
the area term rather than in `gamma`. `b` is the only term the fit still inherits,
which is why that is the only bias it has to tolerate.

Startup profiling reads a different `c` on every stage (35.5ms on stage 0 against
13.7ms on stage 3 for this configuration), and the head is the stage with the
largest one. Refitting `c` is what makes that stop mattering: with it pinned, the
per-stage fits disagree on `gamma` by a factor of 1.7, and with it free they land
within 15% of each other.

## Measured results

Kimi-K2.5-MXFP4 on 8x MI355X, disaggregated: prefill PP4 x TP1, mooncake KV
transfer, decode TP4. ISL 131072, OSL 128, `random-range-ratio=1.0`,
`concurrency=1`, 10 measured requests after warmup.

| Configuration | Chunks per request | Throughput (tok/s) | Mean TTFT | Median TTFT |
| --- | --- | --- | --- | --- |
| fixed 32768 (default) | 4 | 7802 | 14896 ms | 14874 ms |
| fixed 16384 | 8 | 9809 | 11459 ms | 11432 ms |
| fixed 8192 | 16 | 10901 | 10121 ms | 10082 ms |
| fixed 4096 | 32 | 10742 | 10297 ms | 10261 ms |
| fixed 2048 | 64 | 9237 | 12286 ms | 12270 ms |
| **dynamic, calibrated** | **13** | **11225** | **9774 ms** | **9787 ms** |

The solved chunk sequence matches the model's prediction exactly:
`[32768, 16192, 12416, 10368, 9088, 8192, 7488, 6912, 6464, 6080, 5760, 5504,
3840]`.

Two numbers matter here, and quoting only the first one is misleading:

- **vs the default fixed 32768: −34.4% mean TTFT, +43.9% throughput.** This is
  the number that shows up when a deployment leaves `max-num-batched-tokens` at
  a value tuned for throughput rather than for pipeline balance.
- **vs the best fixed chunk (8192): −3.4% mean TTFT, +3.0% throughput.** Most
  of the headline gain is available from simply using a smaller fixed chunk,
  because the dominant effect at low concurrency is *how many microbatches
  exist*, not how evenly they are sized.

So the value of Dynamic CPP is not a large peak win over a well-tuned fixed
chunk. It is that it lands on a good operating point automatically, from a
measured model, for whatever prompt length shows up — where the fixed-chunk
optimum is a U-shaped curve that has to be re-swept per ISL and per topology,
and is 47% worse than optimal at the default and 21% worse two steps away
(2048).

Its structural advantage over a uniformly small chunk is visible in the chunk
counts: it reaches an even 572ms per chunk in 13 chunks, where fixed 8192 needs
16 and fixed 4096 needs 32. Since every chunk re-pays `gamma*L`, fewer chunks
at the same balance means less total prefix-rebuild work.

The +3% against a tuned baseline is also what the published numbers for the same
technique amount to: SGLang and vLLM-Ascend both report low single digits for
single-request long prefill under PP. Their baselines are a fixed chunk several
times smaller than the dynamic starting chunk, which is why a 3% figure and a
44% figure can describe the same feature — the second one is measured against an
untuned batch budget.

The end-to-end gain survives disaggregation. The same comparison on a
standalone PP4 server (no KV transfer, no router, 40 requests) gave −35.4% mean
TTFT and +54.8% throughput against fixed 32768, so mooncake transfer and router
overhead do not erode the prefill improvement.

### At concurrency it loses throughput

Same topology and prompt length, `concurrency` 4 and 16, 3 prompts per
concurrent slot, fixed 32768 vs the same calibrated dynamic model:

| Concurrency | Throughput (tok/s) | Mean TTFT | Verdict |
| --- | --- | --- | --- |
| 4 | 14618 → 13095 (**−10.4%**) | 30108 → 35451 ms (**+17.7%**) | regression |
| 16 | 15785 → 13377 (**−15.3%**) | 110727 → 131369 ms (**+18.6%**) | regression |

The first hypothesis was that the concurrency guard was failing to close. It is
not. Three thresholds were measured against the same workload — the original
`>= pipeline_parallel_size`, "any second prefill closes it", and the trailing
window this code now uses — and all three land within a point of each other at
both concurrencies. A fourth variant that keeps the whole feature configured but
never calls the solver reproduces the fixed baseline exactly, which is the
useful control: the plumbing, the predictor and the supply bookkeeping cost
nothing. What costs 10-15% is the chunk sizes themselves.

That is the point worth carrying away: **shrinking chunks is not free, and there
is nothing to buy with it once a second request is prefilling.** A smaller chunk
does not reduce work, it redistributes it — the same tokens are attended to
either way — while each added chunk pays one more `gamma*L` prefix rebuild and
one more per-forward overhead. On this model at a 128K prompt that surcharge is
around 465 ms per request per stage. A sole prefill more than recovers it from
the shorter pipeline drain (−35% TTFT). Two concurrent prefills have no drain
left to shorten, so the surcharge is the whole effect.

Two things about this are still open. First, the loss does not scale with how
often the solver actually fires: cutting solved decisions by roughly 90% at
`concurrency=16` left the regression essentially unchanged, which looks like a
threshold effect rather than a per-chunk cost — MoE kernel bucketing by token
count is the leading suspect and has not been confirmed. Second, every run
compared fixed and dynamic at the *same* `--max-num-batched-tokens`, so the
"raise the budget to 2-3x and keep the chunk count neutral" argument above is
reasoned from the cost model and from the reference implementations, not measured
here.

Practically: **enable Dynamic CPP only where prefill concurrency is reliably 1.**
The trailing-window guard narrows the exposure and encodes the right intent, but
it is not a validated no-op under load.

### Model accuracy

Predicting TTFT as `sum(chunk costs) + (P-1) * max(chunk cost)` from the fitted
coefficients reproduces the measurements to within a few percent at large
chunks and drifts optimistic as chunks get smaller:

| Configuration | Predicted TTFT | Measured TTFT | Error |
| --- | --- | --- | --- |
| fixed 32768 | 14729 ms | 14896 ms | −1.1% |
| fixed 16384 | 11306 ms | 11459 ms | −1.3% |
| fixed 8192 | 9766 ms | 10121 ms | −3.5% |
| fixed 4096 | 9684 ms | 10297 ms | −5.9% |
| fixed 2048 | 11105 ms | 12286 ms | −9.6% |
| dynamic | 8994 ms | 9774 ms | −8.0% |

The drift is one-sided and grows with chunk count, which points at per-chunk
overhead that the fit cannot see: the timer spans the forward only, so scheduler
work, P2P send/receive latency and KV connector metadata are excluded from `c`
but paid in reality. The practical consequence is that the solver is slightly
biased toward too many chunks, and the true optimum sits at marginally larger
chunks than it picks. This is also why dynamic beats the best fixed chunk by
3.4% in practice where the model predicts 7.1%.

The solved chunks do come out equal-cost under the model — 572, 571, 572, 570,
... 570, with a 440ms remainder — so the solver is doing what it is asked to do.
The residual gap is in the model, not the solver.

## Limits and gotchas

**The concurrency guard bounds exposure; it does not make the feature safe under
load.** `Scheduler._dynamic_chunk_limit` keeps the chunk fixed unless the request
being sized is the only one with prefill work left, measured as a peak over a
trailing window rather than instantaneously — a chunk sequence committed during a
momentary lull still executes alongside whatever arrives behind it. The intent is
right, but the measurements above show the regression is not what a threshold
fixes. Leave the feature off where prefill concurrency is above 1.

**A model fitted without a separate `gamma` produces wrong chunk sizes.** Least
squares folds the prefix growth into `a` and inflates it several-fold. The
solver charges `gamma*L` against the equal-latency budget, so a fit that hides
it in the area term both over-shrinks and mis-shapes the sequence.

**The first requests of a run are the calibration sweep.** They get fixed chunks,
alternating between the budget and a quarter of it, so their TTFT is whatever
fixed chunking at those sizes gives — not the dynamic result. Four requests
usually suffice, which fits inside a benchmark's warmup, but a short measured run
with no warmup will average the sweep into its result.

**Check that the feature actually engaged.** The prefill log states where the
model is in its lifecycle — `dynamic chunking chunk overhead b=... c=...` at
startup, `dynamic chunking still calibrating: ... N chunk sizes` while the sweep
runs, `dynamic chunking calibrated from N real prefill shapes over M chunk sizes`
once it fits, `Dynamic chunking latency model installed` when the scheduler adopts
it, and either `disabling dynamic chunking because ...` or `Ignoring dynamic
chunking calibration ...` when it declined to.

**`--dynamic-chunking-min-chunk-size` must be below `--max-num-batched-tokens`.**
A floor at or above the budget leaves the solver nothing to return, so chunking
stays fixed. The default `4096` assumes a budget in the tens of thousands of
tokens.

## Reproducing

The harness lives outside the repo, under `scripts/dynamic_ck/`:

| Script | Purpose |
| --- | --- |
| `run_kimi_pd_config.sh` | One disaggregated PD configuration (prefill PP4 x TP1 + mooncake + decode TP4), fixed or dynamic. |
| `run_kimi_pd_fixed_sweep.sh` | Add more fixed chunk sizes to an existing run root, to locate the best fixed baseline. |
| `run_kimi_pd_conc_sweep.sh` | Same A/B at higher concurrency, and across guard thresholds including a never-solve control. |
| `summarize_pd_ab.py` | Compare runs and recover each run's actual chunk sequence from the scheduler log. |

`fit_real_chunk_samples.py` and `run_kimi_pd_calibrated_ab.sh` fitted the model
offline from per-forward log lines, which the server now does itself. They are
kept as the reference the runtime fit was validated against.

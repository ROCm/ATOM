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
  --dynamic-chunking-smooth-factor 1.0 \
  --dynamic-chunking-calibration 3.25e-07,6.03e-03,26.1,5.39e-04
```

| Flag | Meaning |
| --- | --- |
| `--enable-dynamic-chunking` | Turn the feature on. Requires `pipeline_parallel_size > 1`. |
| `--dynamic-chunking-calibration` | `a,b,c,gamma` fitted from real requests, replacing startup profiling. Effectively required: see [Calibrating the latency model](#calibrating-the-latency-model). |
| `--dynamic-chunking-base-size` | The chunk the solver equalizes against. `0` (default) uses `--max-num-batched-tokens`. |
| `--dynamic-chunking-min-chunk-size` | Floor on the solved chunk size. Default `4096`. |
| `--dynamic-chunking-smooth-factor` | Blend between base size (`0.0`) and the solver's answer (`1.0`). |
| `--dynamic-chunking-calibration-logging` | Emit the per-forward samples used to fit the coefficients. Calibration runs only. |
| `--max-num-batched-tokens` | Ceiling: a dynamic chunk is never larger, whatever the base size is. |

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

The chunk the equality is taken against is `--dynamic-chunking-base-size`, which
by default is `--max-num-batched-tokens`. That default couples two unrelated
things: the batch budget is a memory and scheduling limit, while the base size
decides how many chunks a prompt is split into. Because the solver only ever
returns something at or below the base, inheriting the budget means the chunk
count can only grow relative to fixed chunking of the same budget — and every
added chunk re-pays `gamma*L`. Setting the base to 2-3x the best fixed chunk
size keeps the count roughly neutral while still evening the chunks out.

The relevant code is `ChunkSizePredictor` in
`atom/model_engine/dynamic_chunking.py` and `Scheduler._dynamic_chunk_limit` in
`atom/model_engine/scheduler.py`.

## Calibrating the latency model

The four coefficients are **configuration-specific**. They change with the
model, the PP layer split, `kv_cache_dtype`, `attn_prefill_chunk_size`,
`max_model_len` and the attention backend. They must be measured, and measuring
them on the wrong workload is the single most common way to make this feature
useless.

By default ATOM profiles at startup using dummy batches. That is cheap but
biased: dummy runs use `batch_size=1` with no real prefix, never trigger MoE
routing at scale, and sweep a grid that skips the realistic
(large prefix, moderate chunk) corner. On Kimi-K2.5-MXFP4 the two are not close:

| Coefficient | Dummy profile | Real requests | Ratio |
| --- | --- | --- | --- |
| `a` (attention area) | 8.2e-11 | 3.25e-07 | ~4000x low |
| `b` (linear in chunk) | 6.44e-03 | 6.03e-03 | ~1x |
| `gamma` (prefix rebuild) | 2.27e-05 | 5.39e-04 | ~24x low |

The pattern is the tell: the dummy profile recovers the chunk-linear term
correctly and collapses **both** prefix-dependent terms, because dummy batches
have no prefix to pay for. The resulting model claims a chunk costs almost the
same whether it follows 0 or 98304 cached tokens.

Fed that model, the solver has nothing to solve. At a 98304-token prefix it
asks for ~34000 tokens, above the 32768 ceiling, so the chunk is clamped back to
the base size — which is exactly the observed behaviour: 32768 became 32704, a
0.2% change that leaves the chunk **count** identical, so no bubble is removed.
The startup guard then (correctly, given its inputs) reports that shrinking is
pointless and falls back to fixed chunks.

This is the whole reason early Dynamic CPP measurements on Kimi came out flat to
slightly negative. It was never an algorithmic problem with equal-latency
chunking; it was a calibration problem.

To calibrate on real traffic instead:

1. Run the serving configuration you actually care about with fixed chunks and
   `--dynamic-chunking-calibration-logging`. Each PP stage then logs one
   `DYNAMIC_CHUNKING_SAMPLE` line per prefill forward with the chunk size, the
   cached prefix, the attention area and the measured GPU time. The timer is
   placed after the PP receive and before the PP send, so it captures the
   forward only, not P2P waiting.
2. Do this at **two different fixed chunk sizes** (e.g. 32768 and 8192). A
   single chunk size yields only `ISL/chunk` distinct shapes — 4 shapes for a
   128K prompt at 32768, which cannot support a 4-coefficient fit.
3. Fit with `scripts/dynamic_ck/fit_real_chunk_samples.py`. It groups repeated
   shapes and fits the median, which keeps rare MoE/OS jitter out of the
   coefficients.
4. Feed the bottleneck stage's coefficients back via
   `--dynamic-chunking-calibration` and drop the logging flag: it synchronizes
   on every prefill forward, which perturbs pipeline overlap enough to change
   the result it is being used to measure.

On Kimi-K2.5-MXFP4 (prefill PP4 x TP1, `fp8` KV, 128K prompts) this produced,
for the slowest stage:

```
  a = 3.25e-07   b = 6.03e-03   c = 26.1   gamma = 5.39e-04
  20 shapes, RMS error 16ms on a 695ms mean (2.3%), condition number 25
```

All four stages fit to within a few percent of each other, and an independent
calibration of a standalone (non-disaggregated) PP4 server landed on
`a = 3.21e-07, b = 6.25e-03, gamma = 6.18e-04` — close enough to treat these as
a property of the model and hardware rather than an artifact of the fit.

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
count is the leading suspect and has not been confirmed. Second, none of these
runs used `--dynamic-chunking-base-size`, so the "keep the chunk count neutral"
argument above is reasoned from the cost model and from the reference
implementations, not measured here.

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

**Coefficients do not transfer between configurations.** The prefill server in
a disaggregated deployment runs with a different `attn_prefill_chunk_size`,
`max_model_len` and `max_num_seqs` than a standalone server, and those change
the fit. Re-calibrate per serving configuration.

**A model fitted without a separate `gamma` produces wrong chunk sizes.** Least
squares folds the prefix growth into `a` and inflates it several-fold. The
solver charges `gamma*L` against the equal-latency budget, so a fit that hides
it in the area term both over-shrinks and mis-shapes the sequence.

**Startup profiling should be treated as a fallback, not a calibration.** It is
good enough to detect "this configuration has no prefix growth worth acting on"
and to keep the feature from doing harm. It is not accurate enough to size
chunks well. If you want the measured gains above, calibrate on real requests.

**Check that the feature actually engaged.** The prefill log states which model
is in use — `dynamic chunking using supplied calibration` for
`--dynamic-chunking-calibration`, `dynamic chunking selected PP stage N fit` for
profiled ones, or `disabling dynamic chunking because ...` when it fell back. A
run that silently fell back looks like "the feature does nothing".

**`--dynamic-chunking-min-chunk-size` must be below the base size.** A floor at
or above the base leaves the solver nothing to return, so chunking stays fixed.
The default `4096` assumes a base in the tens of thousands of tokens.

## Reproducing

The harness lives outside the repo, under `scripts/dynamic_ck/`:

| Script | Purpose |
| --- | --- |
| `run_kimi_pd_config.sh` | One disaggregated PD configuration (prefill PP4 x TP1 + mooncake + decode TP4), fixed or dynamic. |
| `run_kimi_pd_calibrated_ab.sh` | Two fixed-chunk runs, fit the model from their real samples, then run dynamic with the result. |
| `run_kimi_pd_fixed_sweep.sh` | Add more fixed chunk sizes to an existing run root, to locate the best fixed baseline. |
| `run_kimi_pd_conc_sweep.sh` | Same A/B at higher concurrency, and across guard thresholds including a never-solve control. |
| `fit_real_chunk_samples.py` | Fit `a,b,c,gamma` per PP stage from `DYNAMIC_CHUNKING_SAMPLE` logs. |
| `summarize_pd_ab.py` | Compare runs and recover each run's actual chunk sequence from the scheduler log. |

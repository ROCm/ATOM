# DeepSeek-V4.1-Flash — LMCache on/off throughput: the design, written first

Committed before any number exists, because every choice below is a degree of
freedom that could otherwise be made after seeing the data. What it is for is
to make "we measured a X% gain" checkable rather than negotiable.

## What is being compared, and what that is not

| | ON | OFF |
|---|---|---|
| `--kv-transfer-config` | `AtomLMCacheOffloadConnector`, inproc | absent |
| prefix caching | `--enable-prefix-caching` | `--no-enable-prefix-caching` |

The second row is **not a choice**. DeepSeek-V4.1 on the vLLM plugin refuses
prefix caching without the connector: a block-table hit restores the
compressed pages but not the per-request CSA2 state read alongside them, so
the runtime raises at `load_model`. The OFF arm therefore has no reuse of any
kind, and this pair measures **reuse against none**, not tier against HBM.
Any number from it that gets quoted as "what the offload tier buys" is being
quoted wrongly.

Held identical, and read back out of each arm's log rather than asserted:
`CUDAGRAPH=1` (both arms must log `cudagraph_mode=FULL_AND_PIECEWISE`), TP4 on
cards 0-3, `gpu_memory_utilization=0.85`, `max_model_len=32768`,
`max_num_seqs=64`, `max_num_batched_tokens=8192`, tier 64 GiB,
`state_interval=4096`, same client, same prompts, same order, same seed.

## Why the old harness could not answer this

`l4.sh` asked six distinct prompts and sent twenty-four requests, so four
copies of each prefix were in flight together. How many hit depended on
whether they arrived together or staggered, and graph-mode throughput came
out bimodal across repeats of one configuration: 2.63 to 7.74 req/s, a 2.5x
spread. Averaging over that measures the scheduler's luck, not the arm.

`perf.py` removes the four reasons:

1. **Reuse by design, not by race.** 12 documents, primed one at a time
   before the window; every windowed request carries a unique tail, so no two
   requests ever contend for the same store.
2. **Fixed decode work.** `ignore_eos` with `max_tokens=64`, so completion
   length is identical in every request and both arms. `l4.sh` let it vary 8%.
3. **A window long enough to not be jitter.** 240 requests at 24 lanes.
4. **Deltas, not lifetimes.** Tier supply and HBM hit rate are differenced
   across the window; the lifetime values carry the priming.

## The design

Three pairs, **alternating boots**: ON, OFF, ON, OFF, ON, OFF. Not all ON then
all OFF -- the arms need separate servers, so the only defence against drift
over the hours is time adjacency. Three measured rounds per boot, the first of
which is discarded as the warm round. That is 18 rounds run and **12 kept, 6
per arm**.

> Corrected before the run, not after: this line first said "18 kept rounds,
> 9 per arm", which is the total run rather than the total kept and is not
> divisible by two arms either. The design is unchanged -- three pairs, three
> rounds, discard the first -- only the arithmetic describing it was wrong.
> The warm round is worth its cost: in a trial pair the ON arm went 6.52 then
> 13.47 req/s between rounds 1 and 2 of one boot, so including round 1 would
> have halved the arm it belongs to.

## Decided now, so it cannot be decided later

**Validity.** A round counts only if its arm's log shows
`cudagraph_mode=FULL_AND_PIECEWISE`, the arm's identity line matches what was
asked for, `errors=0`, and `done == REQS`. The ON arm additionally needs
window tier supply > 50%; below that the tier did not participate and the
round measures something else. A failed round is reported, not dropped, and
its pair is void.

**The statistic.** Per pair, the ratio of medians
`median(ON rounds) / median(OFF rounds)`. The headline is the **median of the
three pair ratios**, with all three printed. Medians rather than means
because the old harness showed this quantity can be bimodal and a mean of a
bimodal sample names a value that never occurred.

**The noise floor is measured here, not inherited.** The within-arm spread
across the 9 rounds of each arm is reported alongside. A difference smaller
than the larger arm's own range is reported as "not measurable at this
sample size", not as a small effect. No noise floor from the eager runs
carries over: those had a 1.3% spread and graph mode showed 2.5x, so the
quantity is not the same quantity.

**Stopping.** Three pairs, decided now. If the three pair ratios do not agree
in sign, the answer is "not measurable at three pairs" and the remedy is more
pairs, not a fourth pair appended to a set of three that disagreed.

## What will be reported either way

Throughput per round; p50/p90/p99 latency; per-window tier supply and HBM hit
rate for both arms; the prompt and completion token totals, which must match
across arms or the arms were not given the same work. And the sentence this
document exists to force: that OFF means no reuse at all, because this model
cannot have prefix caching without the connector.

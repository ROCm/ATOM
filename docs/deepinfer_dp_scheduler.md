# DeepInfer DP phase scheduling

Native ATOM can select DeepInfer's NORMAL-engine prefill/decode policy with
`--dp-scheduler deepinfer`. The default remains `--dp-scheduler atom`.
This is for colocated DP attention; it supports the DP Attention + TP MoE
configuration (`-tp 8 --enable-dp-attention`, EP off) as well as EP. Pipeline
parallelism, RapidServe, P/D disaggregation and plugin engines are rejected.

```bash
# Append to the same native server command/configuration used for the baseline:
--dp-scheduler deepinfer \
--deepinfer-prefill-interval 70 \
--deepinfer-prefill-any-interval 100
```

The Python port is based on [DeepInfer scheduler.hpp at 5537415](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/csrc/engine/scheduler.hpp#L295-L339)
and its `schedule()` counter update. Defaults 70/100 are from that revision's
`deepinfer/environ.py`. The original license and port attribution are in
`licenses/DeepInfer-LICENSE` and `licenses/DeepInfer-NOTICE`.

## Policy

All ranks maintain the same `decode_step`, initially zero:

1. At multiples of 70, attempt an **ALL** vote: every rank must want prefill.
2. At nonzero multiples of 100, attempt **ANY**: one willing rank is enough.
   ANY wins when the two intervals coincide. Zero is ALL, allowing consecutive
   prefills while the whole group has work.
3. A rank wants prefill when it has pending work and KV usage is at most 0.9.
   This is not an allocation-fit probe. Slots or KV allocation may still fail.
4. After an affirmative group vote, willing ranks attempt native prefill
   allocation. A second ANY vote checks whether any rank obtained a prefill
   output. If none did, the group decodes instead, avoiding an empty-prefill loop.
5. A global prefill resets the counter to zero. Otherwise it increments once,
   including on ranks running an empty/dummy decode. Counts are scheduler
   steps, not tokens or milliseconds; DSpark acceptance does not advance them.

There is deliberately no unconditional first-prefill or no-decode shortcut.
If only one of eight ranks initially has work, it waits for an ANY point while
the ranks advance through dummy decode steps. This behavior is inherited from
the reference; it can increase low-load TTFT.

## ATOM integration

`DeepInferDPPolicy.begin_step()` precedes native admission;
`finish_step()` follows it, before any local prefill/decode return. Idle ranks
join both required votes. Collective failures propagate rather than letting
one rank choose a different phase.

DeepInfer holds partial prefills in its waiting queue; ATOM holds them in
`running`. The adapter includes those partials in the pending-work signal and
gates both partial resumes and fresh admission. Allocation failure, a full
sequence limit, or KV pressure does not reset the clock. A rejection-only
prefill output counts as a prompt step, as upstream's `ignored_seqs` does.

On a global prefill, a rank without its own batch pauses decode and submits an
empty scheduled batch, **even without prefill token graphs**. It creates no
synthetic request. `ForwardMode.decide()` takes the phase from its peers' real
prefills and supplies the padding rows needed by ATOM's collectives and graphs.
DeepInfer represents this as an empty scheduler output, but its
[worker also materializes a placeholder and executes the model](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/worker.py#L111-L117);
the [placeholder has one query token](https://github.com/deepseek-ai/DeepInfer/blob/553741557a57df44a46fb3ab72b6aafeaf945337/deepinfer/worker/model_runner.py#L355-L371).
An empty scheduler output must not be interpreted as skipping GPU execution.
ATOM materializes padding after the DP metadata exchange; graph and TBO
eligibility still follow ATOM's existing cross-rank metadata.
Consequently, enabling TBO does not guarantee it activates on dummy-heavy steps.

This mode replaces PrefillDelayer; it does not layer on top of it:

- `ATOM_ENABLE_PREFILL_DELAYER`, `ATOM_PREFILL_DECODE_INTERVAL` and the other
  `ATOM_PREFILL_DELAYER_*` settings do not control this mode.
- `--prefill-admit-policy` selects ATOM's native admission rule only. Its
  `interval` and `adaptive` settings do not alter DeepInfer's ALL/ANY votes.
- DP chunk-cost balancing is bypassed because it depends on the delayer's
  extra exchange and would introduce a different scheduling rule.
- Request routing, KV allocation, chunked prefill, native sequence/token
  limits, DSpark verification, MoE communication and kernels remain ATOM's.
  This is a port of **phase selection**, not the entire DeepInfer engine.

The startup log identifies the selected policy and intervals. Debug logs from
`atom.model_engine.deepinfer_scheduler` report the pre-step counter, vote type,
local willingness, local admission and final group phase.

## Validation and benchmark comparison

```bash
python -m pytest tests/test_deepinfer_scheduler.py tests/test_prefill_delayer.py \
  tests/test_scheduler.py tests/test_scheduler_partial_prefill_tail.py \
  tests/test_scheduler_spec_preempt.py tests/test_scheduler_mtp_max_tokens.py \
  tests/test_dspark_scheduler.py
```

The new tests cover ALL at zero, consecutive prefill, ANY precedence, uneven
arrival, watermark equality, full slots, allocation failure, partial prefills,
rejection-only output, idle/dummy participation, eight-rank voting masks and
two-process Gloo votes.

For the next GPU experiment, compare both policies at the **same ATOM commit**,
model and launch configuration. For DeepSeek-V4-Pro DSpark use the existing
DP8/EP-off, prefill-TBO, 8192/1024 workload at concurrency 256/512/1024. Keep
70/100 for the first semantic baseline; tune intervals only as separate cases.
Record actual prefill/decode/dummy phases, graph/TBO activation and latency
distributions in addition to throughput. Earlier interval-only measurements
are not measurements of this policy, nor proof of a performance gain.

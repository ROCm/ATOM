# Decode TTFT Stage Breakdown

Why `atom:time_to_first_token_seconds` is larger than
`queue_time + gpu_forward`, and the instrumentation that accounts for the
difference.

## Contents

- [The problem](#the-problem)
- [The identity](#the-identity)
- [The stages](#the-stages)
- [Clocks across process and thread boundaries](#clocks-across-process-and-thread-boundaries)
- [Reading the numbers](#reading-the-numbers)
- [A measured example](#a-measured-example)
- [Optional trace spans](#optional-trace-spans)
- [Limits](#limits)

---

## The problem

TTFT is a client-side quantity: it starts when the POST reaches the API
middleware. `queue_time` and `gpu_forward` are engine-side; the earliest either
can start is when the engine's input thread receives the request. Everything in
between — tokenizing the prompt, the hop into the engine, the hop back out, and
the SSE encode — was not measured by anything, so the residual could only be
guessed at.

The residual is not small. On the agentic PD workload used to validate this
change, `queue_time + forward_to_output` accounted for 944 ms of a 1155 ms mean
TTFT, leaving **210 ms (18%) unattributed**, almost all of it prompt
tokenization in the decode node's own API process.

## The identity

The six slices are cut so that each one ends exactly where the next begins:

```
api_preprocess + api_enqueue + queue_time + forward_to_output
    + output_to_callback + callback_to_sse  ==  time_to_first_token
```

Two properties make this useful, and both are consequences of that
end-meets-start construction:

- **No gaps.** A stage boundary is a single timestamp read, used as the end of
  one slice and the start of the next. Nothing between the middleware and the
  first SSE frame falls outside a slice.
- **No overlap.** Within one request these stages are strictly serial, so the
  slices partition an interval rather than sampling concurrent work.

Because they partition the interval, the slice **means** sum to the TTFT mean.
Their percentiles do not — see [Reading the numbers](#reading-the-numbers).

`pd_kv_transfer` is a *subset* of `queue_time`, not a seventh term. Adding it
double-counts.

## The stages

| Stage | Boundary | Observed in | Code |
| --- | --- | --- | --- |
| `api_preprocess` | middleware entry → `preprocess()` returns | API process | `request_timing.py` `RequestTimingMiddleware`; `api_server.py` `setup_streaming_request` |
| `api_enqueue` | `preprocess()` returns → `add_request()` returns | API process | `api_server.py` `setup_streaming_request` |
| `queue_time` | engine input thread receives → first real forward dispatched | scheduler | `scheduler_metrics.py` `record_forward` |
| `forward_to_output` | first forward dispatched → scheduler emits first generated token | scheduler | `atom/metrics/scheduler.py` `record_first_scheduler_output`, called from `scheduler.py` |
| `output_to_callback` | scheduler emit → API `stream_callback` | crosses engine → API | `ttft_breakdown.py` `mark_first_callback`, called from `api_server.py` `_send_stream_chunk_direct` |
| `callback_to_sse` | `stream_callback` → first SSE payload with content | API process | `api_server.py` `_client_stream` |

`api_preprocess` is the one most likely to be misread. Tokenization runs in a
`ThreadPoolExecutor`, so it overlaps *other* requests freely — while request A
is on the GPU, request B can be tokenizing. But for any single request it is
strictly serial: `preprocess()` must return before `add_request()` is called, so
its cost lands squarely on that request's own TTFT and cannot be hidden by
concurrency.

Under 1P1D this cost is easy to overlook, because prefill already happened on
the prefill node — but the decode node runs its own `io_processor.preprocess()`
on the forwarded request, and the router forwards `messages`, not tokens.

**This is now avoided when the pair carries token ids.** atomesh asks the
prefill node for `return_token_ids` and hands the result to decode in
`kv_transfer_params["prompt_token_ids"]`; decode then skips both the template
render and the tokenize, and `api_preprocess` collapses to the cost of building
a `Sequence`. See
[Reusing the prefill's token ids](../recipes/pd_disaggregation_guide.md#reusing-the-prefills-token-ids).
The measurements below predate that change and are what this stage looks like
without it — which is also what you still get from a prefill node too old to
answer `return_token_ids`, since decode falls back to tokenizing.

### `forward_to_output` is not `gpu_forward`

`forward_to_output` is a superset. It spans the whole wall-clock interval from
dispatch to emit, which includes `prepare_model`, `run_model`, MTP drafting,
sampling and `postprocess`, plus any scheduled step in which the request did not
yet produce a token. `atom:gpu_forward_seconds` measures only the device-event
duration inside `run_model()`; its own help text says it "excludes input
preparation, sampling and drafting".

Substituting one for the other understates TTFT. Since `forward_to_output` is
the larger of the two, a gap computed with it is a *lower bound* on the gap.

## Clocks across process and thread boundaries

`time.perf_counter()` is monotonic per process. It is safe across threads within
one process, and meaningless across processes. Two boundaries therefore need
care:

**Engine → API (`output_to_callback`).** The scheduler additionally stamps
`time.time()` when it first emits, and ships it to the API process on
`RequestOutput.scheduler_output_at`. The API side subtracts it from its own
`time.time()`. Wall clock is the only shared domain here.

**Output thread → event loop (`callback_to_sse`).** Both ends are in the API
process, so `perf_counter` is correct — but the two ends are different *threads*,
and `stream_callback` runs on the engine output thread, which has no access to
the request's `ContextVar`. The stamp is therefore parked in a module-level
`dict[request_id, float]` in `ttft_breakdown.py`, written by the output thread
and read by the request's event loop task.

That map does double duty: it is also the dedupe record. The engine stamps the
same `scheduler_output_at` on *every* `RequestOutput` of a request, so the API
side cannot otherwise distinguish the first chunk from later ones. Entries are
added by the first callback and dropped in `cleanup_request()`, alongside the
existing `_stream_loops` and `_request_start_times` teardown — every path that
can create an entry reaches that function through a `finally`.

## Reading the numbers

Use `_sum / _count`, not percentiles, when checking how the stages compose:

```bash
python3 tools/analyze_ttft_breakdown.py http://127.0.0.1:8020/metrics
```

The tool aggregates buckets across ranks and prints per-stage count, mean and
p50 estimate.

Percentiles are not additive, and in practice they miss in both directions. On
the validation run, summing per-stage p50 gave 611 ms against a measured TTFT
p50 of 787 ms (22% low), while summing per-stage p90 gave about 2418 ms against
a measured p90 of 2254 ms (7% high). The reason is that the long tails do not
coincide: a request slow to tokenize is usually not the one slow to load KV.
Summing p50 assumes every stage is simultaneously median; summing p90 assumes
every stage is simultaneously in its tail. Only the mean is linear.

## A measured example

GLM-5.2-MXFP4, 1P1D CPP4+DCP4+MTP3, AIPerf agentic at concurrency 32, 406
streaming decode requests, means from `_sum / _count`:

| Stage | Mean | Share of TTFT |
| --- | --- | --- |
| `api_preprocess` | 203.41 ms | 17.6% |
| `api_enqueue` | 0.47 ms | 0.04% |
| `queue_time` | 913.20 ms | 79.1% |
| ↳ of which `pd_kv_transfer` | 901.54 ms | 78.1% |
| `forward_to_output` | 31.24 ms | 2.7% |
| `output_to_callback` | 3.64 ms | 0.3% |
| **sum of stages** | **1151.96 ms** | |
| `time_to_first_token` | 1154.90 ms | 100% |

The 2.94 ms residual (0.25%) is `callback_to_sse`, which had no samples in that
run — the cross-thread stamp described above was added afterwards, and the
residual is what motivated it. Everything else closes.

Two distribution details are worth keeping:

- `forward_to_output` is tightly concentrated: no sample below 20 ms, 386 of 406
  (95%) between 20 and 50 ms, the remaining 20 between 50 and 100 ms. Since
  20–50 ms is one decode step at this configuration, nearly every request
  emitted its first token in the first step it was scheduled into, rather than
  being deferred a step.
- `api_preprocess` is the opposite — a p50 near 122 ms but 12% of requests above
  500 ms. Tokenizing an agentic prompt (68 k tokens per request here) is not a
  fixed cost, and thread-pool contention widens it further.

This run was PD-transfer bound, which is why `queue_time` dominates. The shares
move a lot with the workload: when KV transfer is fast, `api_preprocess` becomes
the largest term — which is what motivated carrying the prefill's token ids
across, and why a PD run whose `api_preprocess` still looks like the table above
is worth checking against
[Reusing the prefill's token ids](../recipes/pd_disaggregation_guide.md#reusing-the-prefills-token-ids)
before optimizing anything else.

## Optional trace spans

`ATOM_TTFT_TRACE=1` wraps the same boundaries in `torch.profiler.record_function`
spans, so a captured trace shows `ttft[api_preprocess]`, `ttft[api_enqueue]`,
`ttft[prepare_model]`, `ttft[gpu_forward]` and `ttft[postprocess]` in Perfetto.
This is what splits the `forward_to_output` interval further; the histograms
alone cannot.

Default is off. When off, each span still constructs a context manager per
forward — three of them in `ModelRunner`, on the order of a microsecond against
a 20–50 ms step.

`ATOM_API_PROFILER_DIR=<dir>` additionally makes `/start_profile` and
`/stop_profile` capture a CPU-only profile of the API process itself, written to
`<dir>/api_process_trace.json`. This is how `api_preprocess` was attributed to
tokenization rather than to waiting.

To split `forward_to_output` against real device time, also set
`ATOM_ENABLE_METRICS_DEVICE_TIMER=1` so `atom:gpu_forward_seconds`,
`atom:gpu_sample_seconds` and `atom:gpu_propose_seconds` produce samples.

On `/v1/chat/completions`, `api_preprocess` is further subdivided into
`api_body_parse`, `api_chat_template`, `api_tokenize` and `api_preprocess_wait`
(observing 0 when a stage is skipped so counts stay aligned). Streaming also
records `api_detokenize_chunk_seconds` on the ITL path. The same subdivides are
**not** yet wired on `/v1/completions`, `/v1/messages`, or atomesh standalone.

## Limits

- **Streaming only (original TTFT stages).** Non-streaming requests do not pass
  through `setup_streaming_request`, so `api_preprocess` / `api_enqueue` /
  `output_to_callback` / `callback_to_sse` are not sampled for them. TTFT itself
  is still recorded with `streaming="false"`. Chat non-stream still observes the
  preprocess subdivides (`body_parse` / `chat_template` / `tokenize` /
  `preprocess_wait`).
- **Fan-out (`n > 1`) is not covered for callback stamps.** Those requests
  dispatch through `_send_stream_chunk_tagged`, which does not stamp the
  callback. Their `output_to_callback` and `callback_to_sse` are simply not
  sampled; tokenize / wait subdivides are still observed.
- **`gpu_forward` / `gpu_sample` / `gpu_propose` are opt-in** and off by
  default, so the containment relationship with `forward_to_output` usually
  cannot be checked numerically on a default deployment.
- **One boundary is inferred, not measured.** `queue_time` starts when the
  engine input thread receives the request, which is slightly after
  `add_request()` returned in the API process. The ZMQ transit itself is charged
  to `queue_time` rather than to `api_enqueue`.

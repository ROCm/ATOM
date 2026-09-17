# TTFT diagnostics

The API exports a small set of preparation timings, scheduler first-output
latency, and an optional scheduler-to-SSE delivery histogram. These help locate
latency without keeping per-chunk detokenization instrumentation enabled.

## Retained measurements

| Metric (`atom:` prefix omitted) | Start → end | Population |
| --- | --- | --- |
| `ttft_api_preprocess_seconds` | API timing middleware entry → preprocess return | Single-sequence streaming requests |
| `api_body_parse_seconds` | Middleware entry → chat handler entry | Chat requests reaching the handler, including some later failures |
| `api_chat_template_seconds` | Actual chat template call | Calls that execute; excludes reused token ids and multimodal preparation |
| `api_tokenize_seconds` | Actual tokenizer.encode call | Calls in shared API preprocessing, including chat and completions |
| `request_queue_time_seconds` | Engine receipt → first real forward dispatch | Dispatched sequences |
| `pd_kv_transfer_seconds` | Start of decode KV load wait → successful completion | Successful PD KV waits; part of queue time |
| `ttft_forward_to_output_seconds` | First real forward dispatch → scheduler first generation emit | Sequences that produce a token |
| `ttft_output_delivery_seconds` | Scheduler first generation emit → first generated SSE payload in the API generator | Single-sequence streaming requests; opt-in |

`api_body_parse_seconds` includes body reception, JSON/Pydantic work,
FastAPI dependencies and intervening waits. It is not exclusively parser CPU time.

Template/tokenize histograms describe executed operations. When PD reuses
prefill token ids, decode skips these observations entirely. Fallback requests
that do execute the work still produce samples, irrespective of the node role.
No calls during a reporting interval mean unavailable data, not measured zero.
See [token ID reuse](../recipes/pd_disaggregation_guide.md#reusing-the-prefills-token-ids).

## Optional output delivery

Set `ATOM_ENABLE_METRICS_OUTPUT_DELIVERY=1` before starting both API and engine
processes. The scheduler attaches its wall-clock timestamp to the first
nonempty output only. The API callback captures the request's timing object and
copies that stamp; it does not read callback clocks or maintain a global stamp
map. `_client_stream()` observes the elapsed time at the first SSE payload with
generated content. Empty/role-only frames and error frames are not endpoints.

The interval covers IPC, the output-thread-to-event-loop handoff, detokenization,
reasoning/tool-call processing and frame construction. It ends before HTTP
transport and client receipt. Fanout (`n > 1`) is not instrumented for this
measurement. A missing engine stamp produces no observation.

The diagnostic is disabled by default: its histogram is not registered and no
scheduler wall timestamp is recorded. Enable it consistently on both processes.
Across hosts, wall-clock synchronization is required; negative/nonfinite elapsed
times are discarded, but positive clock skew cannot be corrected by this metric.

## Reading the measurements

```bash
python tools/analyze_ttft_breakdown.py http://127.0.0.1:8020/metrics
```

The tool displays count, mean and an estimated P50, using only streaming samples
for overall TTFT. Missing or unsampled stages display `n/a`.

Do not sum stage means or percentiles. API metrics count HTTP requests or
executed operations, while scheduler metrics count sequences. Failed requests,
fanout and non-streaming traffic change those populations. Also, API submission
and ZMQ transit are not fully covered: queue timing begins after engine receive.
The tool therefore does not report an additive total or an inferred residual.

GPU forward measures device-event duration for participating worker batches.
Forward-to-output measures wall time over the sequence's path to its first token,
including host preparation and output handling. They cannot be interchanged.

## Optional GPU and trace detail

`ATOM_ENABLE_METRICS_DEVICE_TIMER=1` enables target forward and accumulated
request-prefill GPU timing. Sampling and MTP propose require the additional
`ATOM_ENABLE_METRICS_DEVICE_STAGES=1` switch. Both switches default to 0; agentic
CI enables only the basic device timer automatically. Enabled stages share the
bounded FIFO event pool and never synchronize the GPU for telemetry.

`ATOM_TTFT_TRACE=1` emits `record_function` spans for `ttft[api_preprocess]`,
`ttft[prepare_model]`, `ttft[gpu_forward]` and `ttft[postprocess]`.
Set it before starting the API and workers: the flag is read once when the
trace module loads. Disabled call sites execute the stages directly, without
creating or entering trace context managers or rereading the environment.
`ATOM_API_PROFILER_DIR=<dir>` enables API CPU profiling through `/start_profile`
and `/stop_profile`, writing `<dir>/api_process_trace.json`. Use profiling for
occasional executor-wait and detokenization attribution. Record process CPU use
externally, for example with `pidstat`, alongside performance benchmarks.

## Migration from the detailed metrics

The following families are removed: `atom:api_detokenize_chunk_seconds`,
`atom:api_preprocess_wait_seconds`, `atom:ttft_api_enqueue_seconds`,
`atom:ttft_output_to_callback_seconds`, `atom:ttft_callback_to_sse_seconds`,
`atom:process_cpu_seconds_total`, `atom:process_threads`, and `atom:process_cpus`.

The two callback families are replaced by the directly measured, optional
`atom:ttft_output_delivery_seconds`; it has a new name and boundary. Template
and tokenize no longer add synthetic zero observations. Update dashboards and
alerts to these semantics; restarting services alone does not remove historical
series already stored by Prometheus. Clean externally managed multiprocess
storage between service lifetimes using the launcher's normal cleanup process.

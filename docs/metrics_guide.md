# ATOM Metrics Reference

Every metric ATOM exports on `/metrics`, what each one actually measures, and
how to add a new one.

For the offline HTML dashboard that consumes these metrics in agentic PD CI,
see `.github/scripts/atomesh/observability/README.md`. This document covers the
engine side: the metric definitions and the registration interfaces.

## Contents

- [How a metric reaches `/metrics`](#how-a-metric-reaches-metrics)
- [Metric reference](#metric-reference)
- [Adding a new metric](#adding-a-new-metric)

---

## How a metric reaches `/metrics`

`GET /metrics` renders one `prometheus_client.CollectorRegistry`. Nothing in
that path issues an engine RPC, synchronizes a device, or consumes an
observation — a scrape is a read of already-materialized state.

The API captures the live stream-silence gauge on its event loop, then awaits
on-demand rendering in a background thread. Concurrent scrapes share the one
in-flight render; cancellation of an HTTP request does not cancel that work.
Shutdown drains it. Completed responses are not cached, so a later scrape
observes current API instruments without waiting for a periodic text refresh.
The rendering thread still shares CPU and the GIL with the API loop; collectors
must remain bounded. Engine snapshots retain their existing refresh interval.

### Native instruments and node-local export

API, scheduler and GPU timing owners use `prometheus_client` instruments.
Server entrypoints initialize `PROMETHEUS_MULTIPROC_DIR` before importing the
client or spawning workers. Children inherit this node-local directory;
`MultiProcessCollector` reads their mmap files at scrape time. Histogram buckets,
counts and sums are maintained by the client, without histogram snapshots,
worker telemetry replies or API-side bucket conversion.

Each service launch gets a fresh temporary directory, removed when its parent
exits normally after its children. If the launcher supplies a directory, it
must create/clean it before startup and remove it after all processes exit.
Do not share one directory between service instances or across nodes. A hard
kill can leave the automatic directory behind, but the next launch uses a new
one. The client's Histogram/Counter files retain cumulative values after a
worker exits; `mark_process_dead()` only removes live-mode Gauge files.
Multiprocess exposition omits `_created` and exemplars, and uses the client's
canonical numeric `le` labels (for example `1.0`). API TTFT/ITL use the
same native storage; instruments are not also registered in the scrape registry.

On a non-coordinator DP node, the launcher exposes native metrics on its own
`--host` / `--server-port` while its engines run. Scrape every node, including
the coordinator, as a separate target. Preserve `instance`, `dp_rank`,
`pp_rank`, `tp_rank` and `engine_role` as applicable; aggregate rates with PromQL
before computing histogram quantiles. A coordinator exports only its **local**
native instruments, so scraping it alone misses remote histograms. The CI
collector accepts repeated `--prefill` / `--decode` target addresses.

This replaces custom histogram transport with the standard client while
keeping one endpoint per node rather than one per internal worker. Existing
engine, cache, queue and offload state metrics still use their pre-existing
snapshots and exporter. Their refresh and idle GPU-event polling use
`ATOM_METRICS_UPDATE_INTERVAL_S` (default 1 second). Invalid values log a warning
and fall back to 1 second; valid values must be finite and positive. GPU polling
only observes completed events; it returns no metrics payload and never
synchronizes a GPU.
Native histogram observations do not wait for a snapshot refresh.

The API endpoint retains on-demand thread rendering. Multiprocess storage does
not itself move collection/encoding into another process. Prometheus scraping
has its own interval; the CI collector accepts `--scrape-interval-seconds`
(default `1`, minimum `0.001`, millisecond precision).

---

## Metric reference

Counters are listed by their family name; the exposed series carries the
`_total` suffix (`atom:requests_finished` → `atom:requests_finished_total`).
Histograms expose `_bucket`, `_count` and `_sum`.

### API request and stream latency

Observed in the API process, plus one live-read gauge.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:time_to_first_token_seconds` | Histogram, s | Local API request arrival to first output. `streaming="true"` observes the first generated SSE payload; `streaming="false"` observes the first internal token delivery. One sample per request. Label: `streaming`. |
| `atom:inter_token_latency_seconds` | Histogram, s | Frontend-observed output interval divided by new token count, weighted by that count. Excludes the first output batch. |
| `atom:stream_longest_silence_seconds` | Gauge, s | Seconds the most starved in-flight SSE stream has gone without a chunk; 0 when none is waiting. Read live at scrape time from the event loop serving the stream, not from the snapshot. |

### Scheduler

Observed in each scheduler; exported per DP rank. All carry
`dp_rank` and `engine_role`.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:request_queue_time_seconds` | Histogram, s | Engine receipt to first real forward dispatch, including KV loading waits. Once per executed request. |
| `atom:pd_kv_transfer_seconds` | Histogram, s | Decode-side PD KV load wait until all workers report completion; includes dispatch, handshake and notification. Successful loads only. |
| `atom:decode_batch_size` | Histogram, requests | Real decode request rows per forward (`ScheduledBatch.total_seqs_num_decode`). Excludes dummy work and graph padding. |
| `atom:prefill_request_tokens` | Histogram, tokens | `num_prompt_tokens - batch.num_cached_tokens` at first local prefill dispatch, floored at 0. Once per request. |
| `atom:prefill_batch_tokens` | Histogram, tokens | `total_tokens_num_prefill` per real forward. One sample per chunk; excludes cached prefix, decode tokens and padding. |
| `atom:prefill_context_tokens` | Histogram, tokens | Sum of the prefill rows' logical context lengths through the current chunk, including cached prefixes. One sample per real forward. |
| `atom:prefill_request_context_tokens` | Gauge, tokens | Full input length (`num_prompt_tokens`), including cached prefixes, recorded at first real local prefill dispatch. Once per request sequence across chunks and preemption. Additional labels: `request_id`, `sequence_id`, `started_at` (Unix seconds). |
| `atom:decode_context_tokens` | Histogram, tokens | Sum of the decode rows' logical sequence lengths per real forward. |
| `atom:decode_request_context_tokens` | Gauge, tokens | Exact logical context length at the first real decode dispatch. One sample per request sequence, preserved across preemption. Additional labels: `request_id`, `sequence_id`, `started_at` (Unix seconds). |
| `atom:scheduler_requests` | Gauge, requests | Requests by scheduler state. Label `state`: `running`, `waiting` (excludes KV waits), `waiting_kv` (external KV load or shared-cache prefill wait). |
| `atom:scheduler_kv_cache_blocks` | Gauge, blocks | KV block pool by state. Label `state`: `used`, `evictable`, `vacant`, `total`, where `used + evictable + vacant = total`. |

Batch context histograms use fixed buckets through 8,589,934,592 tokens
(1024 rows of 8,388,608 tokens). Uncached prompt token buckets end at 8,388,608.
This keeps long-context batch totals in finite buckets when computing
percentiles. Prefill and decode request contexts are exact gauges.
KV block partition counts are maintained as blocks change state; snapshot
collection does not scan the free block pool.

Both request context gauges replace the former histograms of the same names;
their `_bucket`, `_count`, and `_sum` series are no longer emitted. Reports show
exact request points, a table, and CSV rows instead of P50/P99 curves. The
`started_at` label records the first dispatch time for that phase, not scrape time.
Repeated scrapes produce one report row; requests from previous runs are excluded.
A request has no sample for a phase it never dispatches: for example, a request
that finishes during prefill has no decode sample. Prefill records the full input
length once, so a 100K-token prompt split into 4K chunks still has one 100K value,
regardless of prefix cache hits. The per-batch context histogram continues to
measure context through each chunk end.
For `n > 1`, each generated sequence has its own `sequence_id` and sample.

These are intentionally per-request metrics: series count grows with requests.
Samples remain after request completion so even short requests can be scraped.
Native multiprocess files retain them until the service's metrics directory is
cleaned; there is no automatic per-request expiry. Use a fresh service/metrics
directory between long benchmark runs when this accumulated cardinality matters.

### GPU forward timing

Device-event histograms, one entry per worker. Labels: `dp_rank`,
`pp_rank`, `tp_rank`, `engine_role`.

**Opt-in.** Set `ATOM_ENABLE_METRICS_DEVICE_TIMER=1` before starting the service;
default `0` emits no samples. See `docs/environment_variables.md`.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:gpu_forward_seconds` | Histogram, s | Per-step target forward device-event duration on each worker, including stream communication and waits. One observation per completed forward, pooling prefill, decode and mixed steps. Excludes `prepare_model`, sampling and MTP drafting. |
| `atom:prefill_request_gpu_forward_seconds` | Histogram, s | Per-worker sum of the batch device durations a request participated in across its initial local prefill chunks. One sample once every chunk has been measured. |

Each worker exports two distributions: per-step forward time and per-request
prefill time. Both retain their histogram buckets, count and sum for means and
percentiles. The step histogram has no `phase` label; P/D services remain
distinguishable by `engine_role` or the scrape's `role` label. Queries that
previously filtered GPU steps by `phase` must use the service role instead.
In a standalone service, all forward modes share the same step distribution.
These histograms aggregate observations; the report does not retain or display
individual step timestamps or request records.

### Prefix cache and KV reuse

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:prefix_cache_requests` | Counter, requests | Prefill requests observed by prefix-cache accounting. |
| `atom:prefix_cache_cached_tokens` | Counter, tokens | Prompt tokens served from the admitted GPU/HBM prefix. |
| `atom:prefix_cache_offload_tokens` | Counter, tokens | Prompt tokens reused from LMCache **beyond** the admitted GPU prefix. Shares prefix-cache input accounting; not transfer volume. |
| `atom:prefix_cache_compressed_tokens` | Counter, tokens | Tokens matched by the compressed-prefix index. |
| `atom:prefix_cache_full_tokens` | Counter, tokens | Full input tokens considered by prefix-cache accounting. |
| `atom:prefix_cache_wanted_tokens` | Counter, tokens | Reusable tokens wanted after checkpoint gates. |
| `atom:prefix_cache_checkpoints_kept` / `_dropped` / `_evicted` / `_orphaned` | Counter | Prefix-cache checkpoint outcomes. |
| `atom:prefix_cache_hit_ratio` | Gauge, ratio | Admitted prefix-cache token hit ratio. |
| `atom:prefix_cache_compressed_hit_ratio` | Gauge, ratio | Compressed-prefix token hit ratio before state gates. |
| `atom:prefix_cache_lost_to_checkpoint_ratio` | Gauge, ratio | Reusable-token ratio lost because a checkpoint was unavailable. |
| `atom:prefix_cache_lost_unrecoverable_ratio` | Gauge, ratio | Reusable-token ratio not recoverable by checkpointing. |

### Engine aggregate state

Summed across DP ranks by `LLMEngine.get_metrics_statistics()`.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:requests_running` / `atom:requests_waiting` | Gauge, requests | Requests running / waiting across DP ranks. |
| `atom:requests_parked_kv_load` | Gauge, requests | Requests parked for an external KV load. |
| `atom:requests_partial_prefill` | Gauge, requests | Requests currently in chunked prefill. |
| `atom:kv_cache_blocks_used` / `_free` / `_total` / `_indexed` | Gauge, blocks | Aggregate KV block pool. `indexed` = blocks reachable by prefix hash; it spans both in-use and free blocks and is **not** an occupancy figure. |
| `atom:kv_cache_usage_ratio` | Gauge, ratio | Fraction of KV blocks currently allocated. |
| `atom:requests_finished` | Counter, requests | Requests completed by the scheduler. |
| `atom:prompt_tokens` / `atom:generation_tokens` | Counter, tokens | Tokens in completed requests. |
| `atom:preemptions` | Counter | Scheduler preemptions. |

### Speculative decoding (MTP)

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:mtp_draft_tokens` / `atom:mtp_accepted_tokens` | Counter, tokens | Draft tokens considered / bonus tokens accepted. |
| `atom:mtp_acceptance_rate` | Gauge, ratio | Fraction of draft tokens accepted. |
| `atom:mtp_average_tokens_per_forward` | Gauge, tokens | Average emitted tokens per speculative decode forward. |
| `atom:mtp_decode_steps` | Counter, steps | Decode steps by accepted bonus-token count. Label: `accepted_tokens`. |

### DP router

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:dp_affinity_new` | Counter | New sticky DP sessions assigned to a load-aware cache owner. |
| `atom:dp_affinity_owner_hit` | Counter | Requests routed to an existing session cache owner. |
| `atom:dp_affinity_spill` | Counter | Existing sessions moved off their cache owner; strict affinity keeps this at 0. |
| `atom:dp_affinity_parent_ignored` | Counter | New child sessions placed independently instead of inheriting a parent owner. |
| `atom:dp_route_explicit` / `atom:dp_route_load_balanced` | Counter | Requests routed by explicit rank / by the load balancer. |
| `atom:dp_requests_routed` | Counter, requests | Cumulative requests per rank. Label: `rank`. |
| `atom:dp_inflight_requests` | Gauge, requests | In-flight requests charged to each rank. Label: `rank`. |
| `atom:dp_queued_prefill_tokens` | Gauge, tokens | Estimated uncached prefill-token debt per rank; sticky follow-up turns charge only positive prompt growth. Label: `rank`. |
| `atom:dp_sessions` | Gauge, sessions | Sticky sessions owned by each rank. Label: `rank`. |

### LMCache offload

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:lmcache_load_requests` / `atom:lmcache_save_requests` | Counter | Completed LMCache load / save operations. |
| `atom:lmcache_loaded_tokens` / `atom:lmcache_saved_tokens` | Counter, tokens | Tokens loaded from / saved to LMCache. This is transfer volume, **not** admitted reuse — for reuse accounting use `atom:prefix_cache_offload_tokens`. |
| `atom:lmcache_load_failures` | Counter | Failed LMCache loads. |
| `atom:lmcache_loads_pending` / `atom:lmcache_saves_pending` | Gauge | Operations currently in flight. |

### Process and exporter health

`gc_*` describe the API process's own collector, not the engine's — each
interpreter keeps its own counters.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:gc_collections` / `atom:gc_collected` / `atom:gc_uncollectable` | Counter | Per-generation collections run, objects reclaimed, objects found unreclaimable. Label: `generation`. `gc_collected` flat after startup means raising `ATOM_GC_THRESHOLD` costs nothing; growth means it would defer real work. |
| `atom:gc_threshold` | Gauge | Collection threshold in effect. Label: `generation`. |
| `atom:metrics_snapshot_available` | Gauge, 0/1 | Whether a runtime snapshot has been collected successfully. |
| `atom:metrics_refresh_errors` | Counter | Failed refreshes. Refresh failure, not scrape failure. |
| `atom:metrics_last_refresh_timestamp_seconds` | Gauge, unix s | Timestamp of the last successful refresh. Stale value with a live scrape means the engine stopped answering. |

> `gc.get_freeze_count()` and the tracked-set size are deliberately **not**
> exported: the former walks the permanent generation (11.9 ms at 430k frozen
> objects, against 0.5 µs for `gc.get_stats()`) for a number that changes twice
> in a process's life. Even in a rendering thread, that work competes with SSE
> delivery for CPU and the GIL. Both live in `/debug/gc_census`, which is asked
> for rather than scraped.

---

## Adding a new metric

Define a standard instrument in the component that observes the event. Bind
rank/role labels once at initialization and call `observe()`, `inc()` or `set()`
where the value becomes known. Do not add a metrics RPC or a snapshot converter.

```python
from prometheus_client import Histogram

class ComponentMetrics:
    def __init__(self, dp_rank, registry=None):
        self.dispatch_time = Histogram(
            "atom:dispatch_seconds",
            "Time waiting for dispatch.",
            ["dp_rank"],
            buckets=(0.001, 0.01, 0.1, 1.0),
            registry=registry,
        ).labels(dp_rank=str(dp_rank))
```

`registry=None` uses native multiprocess storage in server processes without
registering a second collector. Unit tests can pass an isolated
`CollectorRegistry`. Embedders must set `PROMETHEUS_MULTIPROC_DIR` **before**
importing `prometheus_client` and before spawning any contributing processes.

Create zero-valued label children at startup when a `rate()` baseline is
needed; creating a child does not observe a sample. Keep labels bounded and
never include request IDs. For new Gauges, choose `multiprocess_mode` explicitly
according to ownership/aggregation semantics; live modes also require the
process supervisor to call `mark_process_dead(pid)` after that process exits.
Validate device-derived values at the measurement boundary with diagnostic
context, rather than silently discarding them in a generic histogram.

The `atom.metrics` package contains instruments, shared histogram helpers,
Prometheus storage setup, and export infrastructure. Schedulers and model
runners own their instrument instances and trigger observations. Entrypoints
initialize storage before importing the Prometheus client and compose the API
exporter. The API supplies the live stream-silence callback; the exporter reads
it on the event loop before rendering in a worker thread.

| Module | Responsibility |
| --- | --- |
| `metrics/prometheus.py` | Node-local directory lifetime and worker-node HTTP endpoint. |
| `entrypoints/openai/metrics_setup.py` | Native collector and API instrument composition. |
| `metrics/exporter.py` | State and GC metrics, snapshot cache and async rendering. |
| `metrics/request.py` | TTFT and token-weighted ITL instrument definitions and recording interfaces. |
| `entrypoints/openai/request_timing.py` | Request lifecycle timing and generated-output detection. |
| `entrypoints/openai/streaming_dispatch.py` | Streaming delivery, observation timing and live stream silence. |
| `metrics/scheduler.py` | Scheduler observations using standard Histograms. |
| `metrics/gpu.py` | Bounded device-event lifecycle and standard Histograms. |
| `metrics/histogram.py` | Shared latency bounds and the weighted ITL extension. |
| `.github/scripts/atomesh/observability/` | Prometheus collection and HTML reporting. |

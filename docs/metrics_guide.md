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

```
                       API process                      │  Engine process(es)
                                                        │
  RequestTimingMiddleware ─┐                            │
  StreamBatchDispatcher  ──┤ direct instrument          │
                           ▼                            │
                    exporter.registry ◄─────────┐       │
                           ▲                    │       │
                           │                    │       │
  GET /metrics ─► exporter.render()             │       │
                           │                    │       │
                           ▼            _SnapshotCollector       Scheduler ─┐
                    pinned snapshot ────────────┘       │        GPU events─┤
                           ▲                            │        Engine     ┤
                           │                            │                   │
             exporter.update(snapshot) ◄────────────────┴─── METRICS push ──┘
                    (1 Hz refresh loop)                      (engine clock)
```

There are exactly **two ways** a number gets exported, and which one you need is
decided by *which process observes the event*.

### Path A — direct instrument (observed in the API process)

The owning module defines a normal `prometheus_client` instrument against
`exporter.registry` and observes it inline at the event. Cumulative state lives
in the instrument itself; a scrape neither resets nor advances it.

Used by `RequestMetrics` (TTFT) and `StreamMetrics` (ITL).

### Path B — snapshot collector (observed in an engine/worker process)

The engine cannot share a Prometheus registry with the API process, so the
producer accumulates into a plain `CumulativeHistogram`/counter, ships it inside
the existing `METRICS` snapshot, and a thin pure function converts that snapshot
into `MetricFamily` objects at render time.

Used by `collect_scheduler_metrics`, `collect_gpu_metrics`,
`collect_engine_metrics`, and the legacy `_AtomMetricsCollector`.

The snapshot is refreshed by `_refresh_metrics_once()` in `api_server.py` once
per second (`_METRICS_REFRESH_INTERVAL_SECONDS = 1.0`), independently of scrapes.
`render()` pins one snapshot revision in a `ContextVar` for the whole response,
so every collector in a single scrape sees the same revision even if the refresh
loop fires concurrently.

**Histogram events are never lost to the snapshot interval** — the producer
observes at the event; the snapshot only transports accumulated buckets.
Gauges are the exception: they are sampled state, so a queue spike that opens and
closes between two snapshots is invisible.

---

## Metric reference

Counters are listed by their family name; the exposed series carries the
`_total` suffix (`atom:requests_finished` → `atom:requests_finished_total`).
Histograms expose `_bucket`, `_count` and `_sum`.

### API request and stream latency

Observed in the API process (Path A), plus one live-read gauge.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:time_to_first_token_seconds` | Histogram, s | Local API request arrival to first output. `streaming="true"` observes the first generated SSE payload; `streaming="false"` observes the first internal token delivery. One sample per request. Label: `streaming`. |
| `atom:inter_token_latency_seconds` | Histogram, s | Frontend-observed output interval divided by new token count, weighted by that count. Excludes the first output batch. |
| `atom:stream_longest_silence_seconds` | Gauge, s | Seconds the most starved in-flight SSE stream has gone without a chunk; 0 when none is waiting. Read live at scrape time from the event loop serving the stream, not from the snapshot. |

### Scheduler

Observed in each scheduler; transported per DP rank (Path B). All carry
`dp_rank` and `engine_role`.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:request_queue_time_seconds` | Histogram, s | Engine receipt to first real forward dispatch, including KV loading waits. Once per executed request. |
| `atom:pd_kv_transfer_seconds` | Histogram, s | Decode-side PD KV load wait until all workers report completion; includes dispatch, handshake and notification. Successful loads only. |
| `atom:decode_batch_size` | Histogram, requests | Real decode request rows per forward (`ScheduledBatch.total_seqs_num_decode`). Excludes dummy work and graph padding. |
| `atom:prefill_request_tokens` | Histogram, tokens | `num_prompt_tokens - batch.num_cached_tokens` at first local prefill dispatch, floored at 0. Once per request. |
| `atom:prefill_batch_tokens` | Histogram, tokens | `total_tokens_num_prefill` per real forward. One sample per chunk; excludes cached prefix, decode tokens and padding. |
| `atom:prefill_context_tokens` | Histogram, tokens | Sum of the prefill rows' logical context lengths through the current chunk, including cached prefixes. One sample per real forward. |
| `atom:prefill_request_context_tokens` | Histogram, tokens | Same quantity per prefill request row. Request-forward weighted; no padding, no TP multiplication. |
| `atom:decode_context_tokens` | Histogram, tokens | Sum of the decode rows' logical sequence lengths per real forward. |
| `atom:decode_request_context_tokens` | Histogram, tokens | Logical context length per decode request row on each forward. Request-forward weighted. |
| `atom:scheduler_requests` | Gauge, requests | Requests by scheduler state. Label `state`: `running`, `waiting` (excludes KV waits), `waiting_kv` (external KV load or shared-cache prefill wait). |
| `atom:scheduler_kv_cache_blocks` | Gauge, blocks | KV block pool by state. Label `state`: `used`, `evictable`, `vacant`, `total`, where `used + evictable + vacant = total`. |

Batch context histograms use fixed buckets through 8,589,934,592 tokens
(1024 rows of 8,388,608 tokens). Per-request context buckets end at 8,388,608.
This keeps long-context batch totals in finite buckets when computing
percentiles. KV block partition counts are maintained as blocks change state;
snapshot collection does not scan the free block pool.

### GPU forward timing

Device-event histograms, one entry per worker (Path B). Labels: `dp_rank`,
`pp_rank`, `tp_rank`, `engine_role`.

**Opt-in.** Set `ATOM_ENABLE_METRICS_DEVICE_TIMER=1` before starting the service;
default `0` emits no samples. See `docs/environment_variables.md`.

| Metric | Type / unit | Definition |
| --- | --- | --- |
| `atom:gpu_forward_seconds` | Histogram, s | Per-worker target forward device-event duration, including stream communication and waits. Excludes `prepare_model`, sampling and MTP drafting. Extra label `phase`: `prefill`, `decode`, `mixed`. |
| `atom:prefill_request_gpu_forward_seconds` | Histogram, s | Per-worker sum of the batch device durations a request participated in across its initial local prefill chunks. One sample once every chunk has been measured. |

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
> in a process's life, and rendering runs inline on the loop that delivers every
> stream. Both live in `/debug/gc_census`, which is asked for rather than scraped.

---

## Adding a new metric

**Define the metric where it is observed, not in a central file.** Pick the path
by which process sees the event.

### Path A — the API process observes it

Define the instrument in a class owned by the module that observes it, taking
the registry as a constructor argument:

```python
# atom/entrypoints/openai/streaming_dispatch.py

class StreamMetrics:
    """Delivery metrics owned by the API stream dispatcher."""

    def __init__(self, registry):
        from prometheus_client import Histogram

        self._chunk_bytes = Histogram(
            "atom:stream_chunk_bytes",
            "Bytes per delivered SSE chunk.",
            buckets=(64, 256, 1024, 4096, 16384),
            registry=registry,
        )

    def observe_chunk_bytes(self, size: int) -> None:
        self._chunk_bytes.observe(size)
```

Then construct it in `metrics_setup.py` and pass the handle to the component.

If you need labels that must exist before the first request — so a
`rate()` baseline can be established — register the zero-valued children up
front, as `RequestMetrics` does:

```python
for streaming in ("true", "false"):
    self._time_to_first_token.labels(streaming=streaming)
```

Registering a label child does not record a sample.

### Path B — an engine or worker process observes it

Three pieces:

**1. Accumulate at the event**, in the owning component, using
`atom/utils/histogram.py`:

```python
# atom/model_engine/<component>_metrics.py
from atom.utils.histogram import LATENCY_BUCKETS, CumulativeHistogram, prometheus_buckets


class ComponentMetrics:
    def __init__(self):
        self.dispatch_time = CumulativeHistogram(LATENCY_BUCKETS)

    def snapshot(self) -> dict:
        return {"dispatch_time": self.dispatch_time.snapshot()}
```

**2. Put it in the snapshot.** The producer merges its `snapshot()` into the
existing `METRICS` push (`EngineUtilityHandler.collect_metrics` /
`push_metrics`), and `LLMEngine.get_metrics_statistics()` shapes the per-rank
list the API side reads. No new IPC channel.

**3. Convert snapshot → families** with a pure function:

```python
from prometheus_client.core import GaugeMetricFamily


def collect_component_metrics(snapshot):
    metric = GaugeMetricFamily(
        "atom:component_queue_depth",
        "Number of items waiting in the component queue.",
    )
    # `None` is the registration-time metadata query: declare every family that
    # can ever appear. On a real scrape a missing field stays sample-less, so an
    # unknown state is never dressed up as a zero.
    if snapshot is not None and "component_queue_depth" in snapshot:
        metric.add_metric([], snapshot["component_queue_depth"])
    yield metric
```

Register it once, in `atom/entrypoints/openai/metrics_setup.py`:

```python
for collect in (
    collect_scheduler_metrics,
    collect_gpu_metrics,
    collect_engine_metrics,
    collect_component_metrics,   # <-- new component
):
    exporter.register_snapshot_collector(collect)
```

`AtomMetricsExporter` itself does not change.

### Rules the collector contract imposes

- `collect(None)` is called at **registration** time. It must declare every
  family name that can ever appear, including optional ones, and must not do
  RPCs, I/O or GPU synchronization. The registry uses these names to reject
  collisions — including generated series such as `_total`, `_bucket`, `_count`
  and `_sum`.
- `collect(snapshot)` may only **read**. It must not mutate the shared snapshot,
  re-observe a value, or reset a counter.
- Keep the existing zero-vs-missing convention: a field absent from an older
  snapshot must not render as `0` for a metric where 0 is a meaningful value.
- Adding a metric to an **existing** component means editing that component's
  metrics class and its `collect_*` function. Only a genuinely new component
  needs a line in `metrics_setup.py`.

### Where things live

| Module | Defines |
| --- | --- |
| `entrypoints/openai/metrics_setup.py` | Composition root: registers collectors, builds the API instrument classes. |
| `entrypoints/openai/metrics.py` | `AtomMetricsExporter` (snapshot cache, registry, render), plus the legacy `_AtomMetricsCollector` and `_gc_metrics` for engine, DP, offload, GC, stream-silence and refresh-health metrics. |
| `entrypoints/openai/request_timing.py` | `RequestMetrics` (TTFT) and the middleware that marks eligible responses. |
| `entrypoints/openai/streaming_dispatch.py` | `StreamMetrics` (ITL) and `longest_silence_seconds()`. |
| `model_engine/scheduler_metrics.py` | `SchedulerMetrics` sampling and `collect_scheduler_metrics`. |
| `model_engine/gpu_metrics.py` | `GPUForwardMetrics` device-event sampling and `collect_gpu_metrics`. |
| `model_engine/engine_stats.py` | `collect_engine_metrics` (supplemental prefix reuse). |
| `model_engine/engine_utility.py`, `llm_engine.py` | Snapshot production and per-rank aggregation. |
| `utils/histogram.py` | `LATENCY_BUCKETS`, `CumulativeHistogram`, `WeightedHistogram`, `prometheus_buckets`. |
| `.github/scripts/atomesh/observability/` | CI collection, HTML report and export. |

Pre-existing engine, DP, offload and GC metrics still live centrally in
`metrics.py`. That is deliberate scope containment, not the pattern to copy —
**new** metrics belong in their owning component.

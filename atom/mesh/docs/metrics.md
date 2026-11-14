# ATOMesh Routing Metrics

What the routing decision metrics measure, how to read them when analyzing
routing behavior and benchmark results, and how to collect them. The other
`mesh_*` families are summarized in the
[README](../README.md#prometheus-metrics).

## Contents

- [Collecting](#collecting)
- [Metric reference](#metric-reference)
- [Branches](#branches)
- [Reading the values](#reading-the-values)
- [Example queries](#example-queries)
- [Sample output](#sample-output)

## Collecting

Two endpoints render the same registry:

- the Prometheus exporter, `0.0.0.0:29000` by default (`--prometheus-host`,
  `--prometheus-port`);
- `GET /metrics` on the router port, which returns 503 when the exporter is not
  running.

```bash
curl -s http://<router-host>:<router-port>/metrics \
  | grep -E '^mesh_(route|session|selected_dp_rank)'
```

Values accumulate from router start. Use `rate()` or `increase()` in
Prometheus, or subtract a scrape taken before a benchmark from one taken after
it. A series appears once it has a sample, so a policy or branch that never ran
has no line. `--route-log-sample-rate` logs individual decisions with the same
fields (see [Logging](../README.md#logging)).

Agentic PD CI runs every concurrency level against one router and saves three
files per level under `metrics/` in the run artifact:

- `router-metrics-before.prom`: the router port's `/metrics` just before aiperf
  starts;
- `router-metrics.prom`: the same just after aiperf exits, which still includes
  the earlier levels;
- `router-metrics-delta.prom`: the level's own increase of every counter and
  histogram series, buckets, sums and counts included, computed by
  [`router_metrics_delta.py`](../../../.github/scripts/atomesh/observability/router_metrics_delta.py).

Both snapshots start with a `# scraped_at` line, and the window covers the whole
aiperf run, warmup included. A counter that went down means the router
restarted during the level; the delta then counts from the restart, and its
header says `counter reset: yes`.

## Metric reference

Counters carry the `_total` suffix; histograms expose `_bucket`, `_sum` and
`_count`. `pool` is the worker pool the selection was made in: `regular`
without PD disaggregation, otherwise `prefill` or `decode`. `policy` is one of
`random`, `round_robin`, `power_of_two`, `cache_aware`, `prefix_hash` and
`dp_sticky`.
Decisions are counted per attempt: a PD request adds one per pool, and a retry
adds another. Dispatches count the requests actually sent, so a decode worker
skipped after a failed prefill adds none.

| Metric | Type / unit | Labels | Recorded when |
| --- | --- | --- | --- |
| `mesh_route_branch_total` | Counter, decisions | `pool`, `policy`, `branch` (see [Branches](#branches)) | Every selection, including one in a pool without candidates. |
| `mesh_route_prefix_match_ratio` | Histogram, ratio 0 to 1 | `pool`, `policy` | `cache_aware` branches `affinity` and `relocate`. |
| `mesh_route_selected_worker_active_requests` | Histogram, active requests | `pool`, `policy`, `branch` | Every selection that picks a worker, before the router adds the request (see [below](#selected-workers-active-requests)). |
| `mesh_session_worker_change_total` | Counter, requests | `pool`, `reason`: `rebind`, `expired`, `rank_mapping` | A `dp_sticky` session request is sent to a different worker (URL and DP rank) than the session's previous request (see [below](#worker-changes)). |
| `mesh_selected_dp_rank_total` | Counter, dispatches | `pool`, `worker` (base URL), `dp_rank` (rank or `none`) | HTTP routers send a request, after DP rank mapping (see [below](#dp-ranks)). |
| `mesh_route_cache_tokens_total` | Counter, tokens | `pool` (`regular` or `prefill`), `kind`: `prompt`, `cached` | HTTP routers; the response reported cached tokens. Any policy (see [below](#router-estimate-and-engine-reported-cached-tokens)). |
| `mesh_route_cache_estimate_tokens_total` | Counter, tokens | `pool`, `kind`: `prompt`, `estimated`, `actual` | The same, when the policy also estimated a hit on the worker the prompt ran on; today only `cache_aware`. |
| `mesh_route_cache_estimate_error_ratio` | Histogram, ratio -1 to 1 | `pool` | Same requests: (estimated - actual) / prompt tokens. |
| `mesh_prefix_hash_policy_branch_total` | Counter, decisions | `branch` | `prefix_hash`, the same branches without `pool`. Deprecated by [#3141] for `mesh_route_branch_total{policy="prefix_hash"}`. |

Histogram buckets:

- prefix match ratio: 0, 0.1, 0.2, ..., 0.9, 0.95, 1;
- selected worker's active requests: 0, 1, 2, 4, ..., 4096;
- estimate error ratio: -1, -0.5, -0.25, -0.1, -0.05, -0.01, 0, 0.01, 0.05,
  0.1, 0.25, 0.5, 1.

## Branches

`no_healthy_workers` exists for every policy: no candidate is healthy, so no
worker is selected. A policy returns it when none of its candidates is
healthy. The router counts it once, under the pool's policy, when the health
filter leaves no candidate or the pool has no worker, since the policy then
never runs. Every branch that selects a worker also records the
[selected worker's active requests](#selected-workers-active-requests);
"Samples" lists what else the decision adds beyond `mesh_route_branch_total`.
Defaults are the CLI defaults.

| Policy | Branch | Selected worker | Samples |
| --- | --- | --- | --- |
| `random` | `random` | A random healthy worker. | |
| `round_robin` | `round_robin` | The next healthy worker. | |
| `power_of_two` | `single_worker` | The only healthy worker. | |
| | `token_load` | The lower of two random workers by the token load the load monitor last reported. | |
| | `request_load` | The same by active requests, when either worker has no token load. | |
| `cache_aware` | `affinity` | Match ratio above `--cache-threshold` (0.3): the worker holding the longest match. | ratio |
| | `relocate` | Match ratio at or below the threshold: the least-loaded worker, ties broken at random. | ratio |
| | `imbalanced` | The max minus min load exceeds `--balance-abs-threshold` (64) and the max exceeds `--balance-rel-threshold` (1.5) times the min: the least-loaded worker, without a tree lookup. | |
| | `stale_tenant` | Affinity, but that worker is gone or unhealthy: it leaves the tree and the first healthy worker is used. | |
| | `no_tree` | No tree for the model and pool: a random healthy worker. | |
| `prefix_hash` | `ring_hit` | The hash ring owner of the first `--prefix-token-count` (256) tokens, when its load is at most `--prefix-hash-load-factor` (1.25) times the average load including this request. | |
| | `load_balance_walk` | The owner is over that bound: the least-loaded worker within it, or the owner if none is. | |
| | `fallback_least_load` | No ring, or the owner is not healthy: the least-loaded worker. | |
| | `no_tokens` | The request has no tokens: no worker. | |
| `dp_sticky` | `hit`, `new`, `rebind`, `expired` | See [session affinity](#session-affinity). | |
| | `no_session` | No `x-session-id` header: the least-loaded worker. | |

## Reading the values

### Prefix match ratio

`cache_aware` keeps a character tree of the request texts it routed, per model
and pool, and records matched characters over input characters for the best
match. It counts characters of the prompt text, not tokens or KV blocks. The
router prunes the tree on its own schedule (`--eviction-interval`,
`--max-tree-size`) and does not see engine eviction, so the ratio estimates
reuse rather than measuring it. `imbalanced` and `no_tree` skip the lookup, and
the match of a `stale_tenant` is on a worker that is gone, so none of them
records a ratio and the histogram count can be lower than the `cache_aware`
decision count. Mass above the threshold corresponds to `affinity`; mass near 0
comes from first turns and unrelated prompts.

### Selected worker's active requests

`mesh_route_selected_worker_active_requests` samples the requests this router
has in flight on the selected worker (`Worker::load()`). It is read when the
policy returns, before the router adds the request, so a worker with nothing
else in flight reads 0. Other routers and the engine's own queue are not
included. Every selection that picks a worker records one sample under its
branch, so the `_count` of a branch equals its `mesh_route_branch_total`;
`no_healthy_workers` and `no_tokens` select no worker and record none. Compare
values within a branch: high values on `relocate` mean cache misses land on busy
workers, and on `affinity` that cache hits queue behind other requests.

### Session affinity

`dp_sticky` pins a session, identified by a non-empty `x-session-id` header,
to a worker (URL and DP rank) in router memory:

- `hit`: the pinned worker is healthy and its circuit breaker admits requests;
- `new`: first request of the session; it goes to the worker with the fewest
  live sessions, ties broken by load;
- `rebind`: the pinned worker is unhealthy, removed or circuit-open, so the
  session is reassigned like a new one;
- `expired`: the session was idle for more than 120 minutes and is reassigned.

With N turns per session and stable workers, the hit rate approaches
(N - 1) / N. `rebind` points at worker churn; a router restart makes every
session `new` again. Each pool that runs `dp_sticky` records its own branches.

### Worker changes

`mesh_session_worker_change_total` counts the session requests that run on a
different worker than the previous request of the same session. The worker is
the one the request is sent to, after DP rank mapping, so ranks behind one URL
are different workers. It is counted when the request is sent: a reserved
worker that receives nothing, such as the decode worker after a failed prefill
or an unexecuted external-processing reservation, neither counts nor becomes
the session's previous worker. `rebind` and `expired` name the affinity result
that moved the session; an expired session reassigned to its old worker counts
nothing. `rank_mapping` means the pin held, but
`--atom-pd-rank-mapping-policy idx2idx` mapped the prefill request to a
different DP rank than the previous one. A session that moves leaves its KV
cache on the old worker. The HTTP routers and the external-processing PD
executor record it; gRPC routers do not.

### DP ranks

`mesh_selected_dp_rank_total` counts the requests the router sends, after it
maps the selection to a DP rank. `worker` is the base URL, which ranks behind
one endpoint share; `dp_rank` is `none` for workers without ranks. Uneven
counts across ranks of one endpoint show rank skew.

### Router estimate and engine-reported cached tokens

`kind="cached"` is the engine's own count, read from the response usage
(`usage.prompt_tokens_details.cached_tokens`,
`usage.input_tokens_details.cached_tokens`, `usage.cache_read_input_tokens`, or
`meta_info.cached_tokens` from `/generate`).
A streaming response carries usage only if the client asks for it, for example
`stream_options.include_usage`; requests without a cached count are not
recorded. Each request is recorded once, after its last attempt, on the pool
that ran the prompt: `regular`, or `prefill` in PD. A PD response comes from
decode, but its cached count is prefill's: in a Kimi-K3 1P1D run, every one of
226 responses matched the tokens prefill had cached, while decode's own prefix
cache hit rate stayed at 0 ([#3142](https://github.com/ROCm/ATOM/pull/3142)).

`kind="estimated"` is the router's guess at selection time:
round(estimated hit ratio x prompt tokens). `cache_aware` estimates its prefix
match ratio only when it picked the worker holding the best match, which covers
every `affinity` decision and any `relocate` that happened to pick it. With
`--kv-events`, its `kv_affinity` decisions estimate the selection from the KV
event index, short of the block of the prompt's last token, which the engine
always computes. The
router drops the estimate when it sends the prompt to a different worker than
the policy chose, such as another DP rank under
`--atom-pd-rank-mapping-policy idx2idx`, because ranks keep separate KV caches.
`kind="actual"` repeats the engine count for exactly those requests, so
`estimated` and `actual` compare like for like.

The two differ because the ratio counts characters while the engine counts
tokens in blocks, and the tree does not see engine eviction, chat templates or
KV offload. A positive error means the router overestimated; a negative one
means the engine reused more than predicted.

## Example queries

Branch share per policy and pool:

```promql
sum by (pool, policy, branch) (rate(mesh_route_branch_total[5m]))
  / ignoring (branch) group_left
    sum by (pool, policy) (rate(mesh_route_branch_total[5m]))
```

Session affinity hit rate, over the `dp_sticky` decisions of requests with a
session ID:

```promql
sum by (pool) (rate(mesh_route_branch_total{policy="dp_sticky",branch="hit"}[5m]))
  / sum by (pool) (rate(mesh_route_branch_total{policy="dp_sticky",branch=~"hit|new|rebind|expired"}[5m]))
```

Session worker changes per session request:

```promql
sum by (pool, reason) (rate(mesh_session_worker_change_total[5m]))
  / ignoring (reason) group_left
    sum by (pool) (rate(mesh_route_branch_total{policy="dp_sticky",branch=~"hit|new|rebind|expired"}[5m]))
```

Engine-reported cache hit rate, and the router's estimate bias:

```promql
sum by (pool) (rate(mesh_route_cache_tokens_total{kind="cached"}[5m]))
  / sum by (pool) (rate(mesh_route_cache_tokens_total{kind="prompt"}[5m]))

(sum by (pool) (rate(mesh_route_cache_estimate_tokens_total{kind="estimated"}[5m]))
  - sum by (pool) (rate(mesh_route_cache_estimate_tokens_total{kind="actual"}[5m])))
  / sum by (pool) (rate(mesh_route_cache_estimate_tokens_total{kind="prompt"}[5m]))
```

Median prefix match ratio:

```promql
histogram_quantile(0.5,
  sum by (le) (rate(mesh_route_prefix_match_ratio_bucket{policy="cache_aware"}[5m])))
```

With two snapshots instead of Prometheus, apply the same ratios to counter
differences, for example hit rate = increase of `hit` / increase of all results.

## Sample output

An excerpt of `/metrics` from a local run: 2 prefill and 2 decode mock
workers, `--prefill-policy cache_aware --decode-policy dp_sticky`, 3 sessions
of 4 chat turns with a growing prompt.

```text
mesh_route_branch_total{pool="prefill",policy="cache_aware",branch="relocate"} 3
mesh_route_branch_total{pool="prefill",policy="cache_aware",branch="affinity"} 9
mesh_route_branch_total{pool="decode",policy="dp_sticky",branch="new"} 3
mesh_route_branch_total{pool="decode",policy="dp_sticky",branch="hit"} 9
mesh_route_prefix_match_ratio_bucket{pool="prefill",policy="cache_aware",le="0.2"} 3
mesh_route_prefix_match_ratio_bucket{pool="prefill",policy="cache_aware",le="0.8"} 9
mesh_route_prefix_match_ratio_bucket{pool="prefill",policy="cache_aware",le="+Inf"} 12
mesh_route_prefix_match_ratio_sum{pool="prefill",policy="cache_aware"} 7.276585578918457
mesh_route_prefix_match_ratio_count{pool="prefill",policy="cache_aware"} 12
mesh_route_selected_worker_active_requests_sum{pool="prefill",policy="cache_aware",branch="affinity"} 0
mesh_route_selected_worker_active_requests_count{pool="prefill",policy="cache_aware",branch="affinity"} 9
mesh_route_selected_worker_active_requests_count{pool="prefill",policy="cache_aware",branch="relocate"} 3
mesh_selected_dp_rank_total{pool="decode",worker="http://127.0.0.1:20832",dp_rank="none"} 8
mesh_selected_dp_rank_total{pool="decode",worker="http://127.0.0.1:20833",dp_rank="none"} 4
```

Reading it:

- `cache_aware` sent 9 of 12 prefill decisions to the worker holding the
  prefix. The 3 relocations are the first turns, with ratios at or below 0.2;
  the later turns fall between 0.7 and 0.9.
- `mesh_route_selected_worker_active_requests` has one sample per decision, so
  each branch's `_count` matches its decision count. All values are 0 because
  requests ran one at a time.
- The `dp_sticky` hit rate is 9 / 12 = 75%: each session's first turn is `new`.
  No session changed workers, so `mesh_session_worker_change_total` has no
  line.
- Decode sent two sessions, 8 requests, to one worker and the third session to
  the other. The mock workers have no DP ranks, so `dp_rank` is `none`.

[#3141]: https://github.com/ROCm/ATOM/pull/3141

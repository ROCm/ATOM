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
  | grep -E '^mesh_(route|session_affinity|selected_dp_rank)'
```

Values accumulate from router start. Use `rate()` or `increase()` in
Prometheus, or subtract a scrape taken before a benchmark from one taken after
it. Agentic PD CI saves the router port's `/metrics` after each aiperf run as
`metrics/router-metrics.prom` in the run artifact. A series appears once it has
a sample, so a policy or branch that never ran has no line.
`--route-log-sample-rate` logs individual decisions with the same fields
(see [Logging](../README.md#logging)).

## Metric reference

Counters carry the `_total` suffix; histograms expose `_bucket`, `_sum` and
`_count`. `pool` is the pool of the selected worker: `regular` without PD
disaggregation, otherwise `prefill` or `decode`. `policy` is one of `random`,
`round_robin`, `power_of_two`, `cache_aware`, `prefix_hash` and `dp_sticky`.
Decisions and dispatches are counted per attempt: a PD request adds one per
pool, and a retry adds another.

"Since" is the change that starts recording the metric. #3141 already declares
the #3142 metrics as planned, so a build with only #3141 renders no line for
them.

| Metric | Type / unit | Labels | Recorded when | Since |
| --- | --- | --- | --- | --- |
| `mesh_route_branch_total` | Counter, decisions | `pool`, `policy`, `branch` (see [Branches](#branches)) | Every policy selection with at least one candidate. | [#3141] |
| `mesh_route_prefix_match_ratio` | Histogram, ratio 0 to 1 | `pool`, `policy` | `cache_aware` branches `affinity`, `relocate`, `stale_tenant`. | [#3141] |
| `mesh_route_cost` | Histogram, the policy's unit; today active requests | `pool`, `policy` | `cache_aware` `imbalanced`, `relocate`; `prefix_hash` `load_balance_walk`, `fallback_least_load`; `dp_sticky` `no_session`. | [#3141] |
| `mesh_session_affinity_total` | Counter, requests | `pool`, `result`: `hit`, `new`, `rebind`, `expired` | `dp_sticky` selected a worker for a request with a session ID. | [#3141] |
| `mesh_selected_dp_rank_total` | Counter, dispatches | `pool`, `worker` (base URL), `dp_rank` (rank or `none`) | HTTP routers dispatch a request, after DP rank mapping. | [#3142] |
| `mesh_route_cache_tokens_total` | Counter, tokens | `pool` (`regular` or `prefill`), `kind`: `prompt`, `cached` | HTTP routers; the response reported cached tokens. Any policy. | [#3142] |
| `mesh_route_cache_estimate_tokens_total` | Counter, tokens | `pool`, `kind`: `prompt`, `estimated`, `actual` | The same, when the policy also estimated a hit; today only `cache_aware`. | [#3142] |
| `mesh_route_cache_estimate_error_ratio` | Histogram, ratio -1 to 1 | `pool` | Same requests: (estimated - actual) / prompt tokens. | [#3142] |
| `mesh_prefill_overlap_blocks` | Histogram, KV blocks | `pool`, `policy`, `medium`: `gpu`, `cpu`, `disk`, `remote` | Not recorded; reserved for routing on engine KV events. | planned |
| `mesh_prefix_hash_policy_branch_total` | Counter, decisions | `branch` | `prefix_hash`, the same branches without `pool`. Deprecated by #3141 for `mesh_route_branch_total{policy="prefix_hash"}`. | before [#3141] |

Histogram buckets:

- prefix match ratio: 0, 0.1, 0.2, ..., 0.9, 0.95, 1;
- cost: 0, 1, 2, 4, ..., 1024, then 4096, 16384, 65536, 262144, 1048576;
- estimate error ratio: -1, -0.5, -0.25, -0.1, -0.05, -0.01, 0, 0.01, 0.05,
  0.1, 0.25, 0.5, 1;
- overlap blocks: 0, 1, 2, 4, ..., 16384.

## Branches

`no_healthy_workers` exists for every policy: there are candidates but none is
healthy, so no worker is selected. "Samples" lists what the decision adds
beyond `mesh_route_branch_total`. Defaults are the CLI defaults.

| Policy | Branch | Selected worker | Samples |
| --- | --- | --- | --- |
| `random` | `random` | A random healthy worker. | |
| `round_robin` | `round_robin` | The next healthy worker. | |
| `power_of_two` | `single_worker` | The only healthy worker. | |
| | `token_load` | The lower of two random workers by the token load the load monitor last reported. | |
| | `request_load` | The same by active requests, when either worker has no token load. | |
| `cache_aware` | `affinity` | Match ratio above `--cache-threshold` (0.3): the worker holding the longest match. | ratio |
| | `relocate` | Match ratio at or below the threshold: the least-loaded worker, ties broken at random. | ratio, cost |
| | `imbalanced` | The max minus min load exceeds `--balance-abs-threshold` (64) and the max exceeds `--balance-rel-threshold` (1.5) times the min: the least-loaded worker, without a tree lookup. | cost |
| | `stale_tenant` | Affinity, but that worker is gone or unhealthy: it leaves the tree and the first healthy worker is used. | ratio |
| | `no_tree` | No tree for the model and pool: a random healthy worker. | |
| `prefix_hash` | `ring_hit` | The hash ring owner of the first `--prefix-token-count` (256) tokens, when its load is at most `--prefix-hash-load-factor` (1.25) times the average load including this request. | |
| | `load_balance_walk` | The owner is over that bound: the least-loaded worker within it, or the owner if none is. | cost |
| | `fallback_least_load` | No ring, or the owner is not healthy: the least-loaded worker. | cost |
| | `no_tokens` | The request has no tokens: no worker. | |
| `dp_sticky` | `hit`, `new`, `rebind`, `expired` | See [session affinity](#session-affinity). | result |
| | `no_session` | No `x-session-id` header: the least-loaded worker. | cost |

## Reading the values

### Prefix match ratio

`cache_aware` keeps a character tree of the request texts it routed, per model
and pool, and records matched characters over input characters for the best
match. It counts characters of the prompt text, not tokens or KV blocks. The
router prunes the tree on its own schedule (`--eviction-interval`,
`--max-tree-size`) and does not see engine eviction, so the ratio estimates
reuse rather than measuring it. `imbalanced` and `no_tree` skip the lookup, so
the histogram count can be lower than the `cache_aware` decision count. Mass
above the threshold corresponds to `affinity`; mass near 0 comes from first
turns and unrelated prompts.

### Route cost

Every sample today is the selected worker's active request count as tracked by
the router (`Worker::load()`), read before this request is added. Only
branches that pick by load record it. The unit belongs to the policy and the
buckets reach 2^20 for policies that will record tokens or blocks, so compare
cost within one policy only. Many `imbalanced` decisions mean load balancing is
overriding cache affinity; high cost on `relocate` means cache misses land on
busy workers.

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
session `new` again. Each pool that runs `dp_sticky` records its own result.

### Router estimate and engine-reported cached tokens

`kind="cached"` is the engine's own count, read from the response usage
(`usage.prompt_tokens_details.cached_tokens`,
`usage.input_tokens_details.cached_tokens` or `usage.cache_read_input_tokens`).
It is recorded on the pool that ran the prompt, `prefill` or `regular`, never
`decode`. A streaming response carries usage only if the client asks for it,
for example `stream_options.include_usage`; requests without a cached count
are not recorded.

`kind="estimated"` is the router's guess at selection time:
round(prefix match ratio x prompt tokens). It exists only when `cache_aware`
picked the worker holding the best match, which covers every `affinity`
decision and any `relocate` that happened to pick it. The router drops it when
it dispatches to a different DP rank than the policy chose, because ranks keep
separate KV caches. `kind="actual"` repeats the engine count for exactly those
requests, so `estimated` and `actual` compare like for like.

The two differ because the ratio counts characters while the engine counts
tokens in blocks, and the tree does not see engine eviction, chat templates or
KV offload. A positive error means the router overestimated; a negative one
means the engine reused more than predicted. Engine-side cache detail stays on
the engines' own metrics, available through `/engine_metrics` (for ATOM, see
[Prefix cache and KV reuse](../../../docs/metrics_guide.md#prefix-cache-and-kv-reuse)).

### DP ranks

`mesh_selected_dp_rank_total` counts dispatches after the router maps the
selection to a DP rank. `worker` is the base URL, which ranks behind one
endpoint share; `dp_rank` is `none` for workers without ranks. Uneven counts
across ranks of one endpoint show rank skew.

## Example queries

Branch share per policy and pool:

```promql
sum by (pool, policy, branch) (rate(mesh_route_branch_total[5m]))
  / ignoring (branch) group_left
    sum by (pool, policy) (rate(mesh_route_branch_total[5m]))
```

Session affinity hit rate:

```promql
sum by (pool) (rate(mesh_session_affinity_total{result="hit"}[5m]))
  / sum by (pool) (rate(mesh_session_affinity_total[5m]))
```

Engine-reported cache hit rate, and the router's estimate bias:

```promql
sum by (pool) (rate(mesh_route_cache_tokens_total{kind="cached"}[5m]))
  / sum by (pool) (rate(mesh_route_cache_tokens_total{kind="prompt"}[5m]))

(sum(rate(mesh_route_cache_estimate_tokens_total{kind="estimated"}[5m]))
  - sum(rate(mesh_route_cache_estimate_tokens_total{kind="actual"}[5m])))
  / sum(rate(mesh_route_cache_estimate_tokens_total{kind="prompt"}[5m]))
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
of 4 chat turns with a growing prompt. The mock workers report 100 prompt
tokens, 60 of them cached, for every request, so the error values only
illustrate the format.

```text
mesh_route_branch_total{pool="prefill",policy="cache_aware",branch="relocate"} 3
mesh_route_branch_total{pool="prefill",policy="cache_aware",branch="affinity"} 9
mesh_route_branch_total{pool="decode",policy="dp_sticky",branch="new"} 3
mesh_route_branch_total{pool="decode",policy="dp_sticky",branch="hit"} 9
mesh_session_affinity_total{pool="decode",result="hit"} 9
mesh_session_affinity_total{pool="decode",result="new"} 3
mesh_route_prefix_match_ratio_bucket{pool="prefill",policy="cache_aware",le="0.2"} 3
mesh_route_prefix_match_ratio_bucket{pool="prefill",policy="cache_aware",le="0.8"} 9
mesh_route_prefix_match_ratio_bucket{pool="prefill",policy="cache_aware",le="+Inf"} 12
mesh_route_prefix_match_ratio_sum{pool="prefill",policy="cache_aware"} 7.276585578918457
mesh_route_prefix_match_ratio_count{pool="prefill",policy="cache_aware"} 12
mesh_route_cost_sum{pool="prefill",policy="cache_aware"} 0
mesh_route_cost_count{pool="prefill",policy="cache_aware"} 3
mesh_route_cache_tokens_total{pool="prefill",kind="prompt"} 1200
mesh_route_cache_tokens_total{pool="prefill",kind="cached"} 720
mesh_route_cache_estimate_tokens_total{pool="prefill",kind="prompt"} 1100
mesh_route_cache_estimate_tokens_total{pool="prefill",kind="estimated"} 710
mesh_route_cache_estimate_tokens_total{pool="prefill",kind="actual"} 660
mesh_route_cache_estimate_error_ratio_sum{pool="prefill"} 0.5
mesh_route_cache_estimate_error_ratio_count{pool="prefill"} 11
mesh_selected_dp_rank_total{pool="prefill",worker="http://127.0.0.1:21418",dp_rank="none"} 8
mesh_selected_dp_rank_total{pool="prefill",worker="http://127.0.0.1:21419",dp_rank="none"} 4
```

Reading it:

- `cache_aware` sent 9 of 12 prefill decisions to the worker holding the
  prefix. The 3 relocations are the first turns, with ratios at or below 0.2;
  the later turns fall between 0.7 and 0.9.
- `mesh_route_cost` has only the 3 `relocate` samples, all 0 because requests
  ran one at a time.
- The `dp_sticky` hit rate is 9 / 12 = 75%: each session's first turn is `new`.
- The engines reported 720 / 1200 = 60% of prompt tokens cached. 11 decisions
  carried an estimate: the 9 `affinity` decisions and the 2 relocations that
  happened to pick the best-match worker. The router estimated 710 tokens
  where the engines reported 660, a bias of +50 / 1100 = +4.5%.

[#3141]: https://github.com/ROCm/ATOM/pull/3141
[#3142]: https://github.com/ROCm/ATOM/pull/3142

# Cache-aware DP placement with LMCache

`--dp-load-balance cache_affinity` keeps successive requests with the same
session ID on their current owner. When another rank is expected to finish
a later turn sooner by enough to cover the gain margins, it can send that turn
there and let the existing LMCache MP connector
recover its prefix. The target must acknowledge a usable prefix before the
router commits the new owner. This policy is opt-in.

## Configuration

Enable prefix caching, frontend prefix hashes, and the `lmcache_mp` offload
connector. The MP server must be running before model startup. V4.1 uses the
native PAGE/STATE connector, BF16 KV, DP attention, and
`--no-encoder-swa-bounded-replay` to retain the checkpoint images required by
that connector. Apply the same replay setting to both benchmark arms.
PP and PD-only workers
are not supported by this placement policy.

```bash
export ATOM_FRONTEND_PREFIX_HASH=1
python -m atom.entrypoints.openai_server ... \
  --enable_prefix_caching --dp-load-balance cache_affinity \
  --kv-transfer-config '{"kv_connector":"lmcache_mp","kv_role":"offload"}'
```

For attention DP2 sharing one LMCache MP server, add `--max-gpu-workers 2`
to the **LMCache server** command. This existing LMCache option gives the two
GPU clients separate STORE/RETRIEVE handler threads. Each client's operations
remain ordered on its own thread. It does not add model workers or split one
transfer across two threads. For other deployments, size the pool for the
distinct GPU clients connected to that server; it is not necessarily the
global DP size. CPU handler workers are configured independently.

The opt-in benchmark helper `.github/scripts/start_benchmark_lmcache.py`
defaults to two GPU workers for the DP2 preset. Set
`ATOM_BENCHMARK_LMCACHE_GPU_WORKERS=1` to reproduce the previous configuration,
or another positive count for a different client topology. This helper setting
does not reconfigure an externally managed LMCache server.

The OpenAI endpoint accepts `X-Dynamo-Session-ID`, falling back to
`X-Correlation-ID`. Selecting this policy enables session extraction without
the legacy `ATOM_DP_SESSION_AFFINITY` switch. An explicit request
`data_parallel_rank` remains authoritative. Each child session owns its own
placement; the parent header does not force siblings onto one rank.

`cache_affinity` is a value of the existing `--dp-load-balance` argument, not
another switch layered over strict session affinity. Its routing, owner table,
and token-load accounting take precedence over `ATOM_DP_SESSION_AFFINITY`.

| Routing policy | `ATOM_DP_SESSION_AFFINITY` | Session behavior |
| --- | --- | --- |
| Existing policies | `0` / unset | Per-request load balancing |
| Existing policies | `1` | Strict session owner; no automatic migration |
| `cache_affinity` | `0`, `1`, or unset | Owner preference with prepared migration |

For a new deployment, select `--dp-load-balance cache_affinity` and leave the
legacy variable unset. Leaving it at `1` in an existing launch script does
not run both policies or force the new owner to remain fixed. Setting it to
`0` does not disable session extraction for `cache_affinity`. Clients still
need a stable session header to receive affinity across requests; an explicit
`data_parallel_rank` wins under both policies.

The offline `ATOM_DP_DISPATCH_DEPTH` pending pool is inactive with
`cache_affinity`. Requests retain cache-aware placement and admission even
inside an offline batch; a count-only pending drain cannot create or commit
the cache migration state. Other routing policies retain the offline depth cap.

`ATOM_DP_CACHE_BALANCE_CONFIG` accepts a JSON object with options defined in
`CacheBalanceConfig`. Defaults estimate prefill at 32,768 tokens/s and transfer
at 2,000,000 tokens/s plus 10 ms per copy. These are initial heuristics, **not
measured V4.1 latency coefficients**. Tune them for the workload and topology.
The default preparation timeout is 5 s, migration cooldown is 30 s, and limits
are eight concurrent preparations globally and two per target. The checkpoint
hint interval defaults to the engine's state checkpoint interval.

The preparation budget starts when the target scheduler receives the request;
it includes queueing, cache lookup, alignment work, and recovery readiness. It
is not a transfer-only timeout. To override just that budget, use, for example:

```bash
export ATOM_DP_CACHE_BALANCE_CONFIG='{"prepare_timeout_seconds": 1.0}'
```

A shorter budget can improve tail latency by falling back sooner, but it can
also abandon useful migrations. The 1 s setting is a workload-specific tuning
option, not the default.

Placement compares estimated completion times directly. There is no absolute
owner-overload threshold or overload hold period. A move needs a strictly
positive gain of at least both 50 ms and 15% of the estimated stay time by
default. Cooldown, concurrent-preparation limits, session fencing, and target
admission still apply. These margins are safeguards, not a measured confidence
interval for the cost model. Queue/copy overlap and native STATE restore still
need calibration.

The destination's initial GPU-work charge uses the same expected recovered prefix
as the placement comparison. Outstanding restore time is tracked separately and
combined with queued GPU work using a maximum, so the full transfer estimate is
not added to unrelated computation. Scheduler admission corrects the prediction
to actual usable coverage. Completed prefill chunks then reduce the outstanding
work; preemption reserves the recomputation again. These updates are asynchronous
and coalesced per engine step, with execution and migration generations fencing
stale completions. Decode tokens do not produce per-token work updates.

Placement also considers the joint imbalance of in-flight request counts and
their initial prompt lengths. Prompt length is a proxy for decode context size;
it does not track generated tokens or claim to measure sparse-attention runtime.
The normalized squared-imbalance score makes short incoming requests emphasize
batch size and long requests emphasize context load. New sessions use this score
among ranks within the existing minimum useful time-gain margin. A migration must
meet the time-gain requirements. If its joint balance score increases, the router
allows it only when the higher load across the source/target pair increases by
at most `decode_balance_tolerance_ratio` in **each** dimension, compared with
placing the request on the owner. The default is 0.01 (1%). A lower batch-size
peak cannot compensate for exceeding the context limit, or vice versa. The
comparison includes the incoming request in both alternatives and only examines
the affected pair, so an unrelated busy rank cannot hide a destination increase.
This is a per-decision proxy bound, not a guarantee on decode latency or total
imbalance over time. Explicit rank selection remains authoritative.

Set `ATOM_DP_CACHE_BALANCE_CONFIG='{"decode_balance_tolerance_ratio": 0}'` to
restore the strict rule that rejects every joint-score increase. The TTFT-gain
margins, cooldown, preparation limits and prefix-admission acknowledgement apply
with either setting. No calibrated decode-time penalty is assumed.

## Admission, copying, and failure

The router retains bounded, expiring hints from completed requests. A hint
records a hash-validated checkpoint boundary on a rank; it predicts residency
but cannot prove that eviction has not occurred. Tokenizer/hash-cache hits alone
never authorize skipping model computation.

For a proposed move, the target follows normal prefix admission. Existing HBM
coverage remains valid. Native MP loads only the missing PAGE range plus the
complete STATE image at the remote endpoint. Joint transfer completion,
native restore, and checkpoint adoption precede the admission acknowledgement.
The router then corrects its outstanding token charge and changes the owner.

If local HBM coverage ends inside a transport chunk, the target may prefill
only to the next chunk boundary before parking for the incremental load.
This alignment work does not acknowledge or commit the migration. A dropped
handoff or an expired preparation during alignment aborts through normal
postprocess, retaining any buffers still used by a forward or transfer.

If a target cannot recover at least the promised prefix, or the preparation
budget expires while queued, it rejects this placement. The original request
and hash chain are resubmitted to the prior owner. That owner's own scheduler
rechecks residency and recomputes any missing tokens. Cancellation discards
the retry payload; late target replies cannot resurrect it. A concurrent turn
fences an older migration from overwriting a newer owner decision.

A confirmed short recovery also records bounded negative evidence for that
session's exact source, target, prefix hash and checkpoint boundary. The router
avoids promising that same unavailable prefix again for
`recovery_miss_ttl_seconds` (120 s by default). A different boundary or expiration
permits another attempt. Preparation timeouts do not establish missing cache
coverage and do not create this evidence. The record predicts recovery cost;
it never authorizes cache reuse or replaces prior-owner fallback.

Timeouts do not free buffers involved in an uncertain DMA. The existing native
connector holds their leases until terminal completion, including when the
request has already fallen back to another rank.

## Storage and eviction

Use one MP server per host. All GPUs that register IPC memory with that server
must be on that host. Same-host DP shares the server's content-addressed L1.
Across hosts, configure the servers with a common supported L2 backend (for
example Mooncake Store), matching model revision, object groups, and layouts.
ATOM's endpoint map selects GPU-local servers; it does not create the shared
backend:

```json
{
  "kv_connector": "lmcache_mp",
  "kv_role": "offload",
  "kv_connector_extra_config": {
    "lmcache.mp.server_urls_by_rank": {
      "0": "tcp://host-a:5555", "1": "tcp://host-a:5555",
      "2": "tcp://host-b:5555", "3": "tcp://host-b:5555"
    },
    "lmcache.mp.model_revision": "immutable-checkpoint-revision"
  }
}
```

The keys are global engine ranks after DP-attention expansion. The map must
cover every rank. Same-host configurations can keep the existing host/port
options. A host-local MP server alone is not a cross-host cache directory.
Cross-host GPU validation is required before relying on this deployment mode.

Source HBM copies are not forcibly deleted when the owner changes. Existing
PAGE/checkpoint reference counts and LRU policies own reclamation; LMCache
owns its L1/L2 eviction. An evicted or stale hint causes failed preparation
and safe fallback. This version has no global replica-count policy or active
duplicate pruning.

## Measurement

`/debug/dp_router_stats` includes `cache_balance` counters for preparations,
commits, superseded owners, fallbacks, and retry-send failures. Compare matched
AgentX runs with the same trace, concurrency, warmup, cache tiers, model and
speculation settings. Measure TTFT and generated-token throughput, and verify
actual load completions and migrations. A run with no successful migration
measures session placement only; it is not evidence of copy-path performance.

Fallback counters distinguish `preparation_timeout`,
`recoverable_prefix_too_short`, `alignment_load_unavailable`,
`alignment_chunk_unavailable`, and `admission_callback_unavailable`. Unknown
reasons use `other`. These categories have bounded cardinality. Compare deltas
within the profiling window, not lifetime totals that include warmup.

`prefill_work_updates` counts accepted completed-work corrections.
`pending_restore_seconds_per_rank` exposes the estimated restore backlog, not
measured GPU blocking time. `migration_recovery_miss_recorded` and
`migration_recovery_miss_avoided` show whether repeated short-recovery predictions
are being suppressed. These counters need throughput and latency measurements
alongside them: fewer fallbacks alone are not evidence of a faster workload.
`migration_decode_balance_rejected` counts otherwise profitable migration
candidates rejected by the joint request-count/context balance constraint.
`migration_decode_balance_tolerated` counts migrations actually started through
the peak-load tolerance despite a higher joint score. It does not count
unselected candidates, confirmed recoveries, or successful completions.

### Measured scope

The latest incremental comparison against `c4de6c6` tested completed-work
accounting, separate restore backlog, exact-prefix recovery-miss evidence and
the joint count/context balance constraint described above. Both arms used
common worker/dependency recovery fixes, V4.1-Flash, MI355X, attention DP2 /
MoE TP2, DSpark5, memory 0.9, AgentX SA393/seed42 and c128. They ran simultaneously
on separate GPU pairs with a common warmup barrier and 900 s of arrivals.

| Metric | Control | Work accounting + joint balance | Change |
| --- | ---: | ---: | ---: |
| Output tokens/s/GPU | 1091.23 | 1136.58 | +4.16% |
| Total tokens/s/GPU, including cached input | 135063.23 | 139129.66 | +3.01% |
| Mean inter-token latency | 49.58 ms | 45.71 ms | -7.81% |
| P90 inter-token latency | 76.44 ms | 70.54 ms | -7.71% |
| Mean TTFT | 2.266 s | 2.405 s | +6.12% |
| P90 TTFT | 4.032 s | 4.618 s | +14.54% |
| P99 TTFT | 9.492 s | 9.292 s | -2.10% |

This is a throughput/latency tradeoff: the constraint can retain requests at
their owner to protect decode balance. It is not a TTFT improvement across
percentiles. Sampled mean in-flight request-count spread fell from 8.90 to
5.21; actual scheduled prefill tokens were essentially unchanged (14.952M /
14.943M). A preceding work-accounting revision without the joint constraint
reduced output throughput by 2.31%; fixing prefill accounting alone did not
establish a net gain.

Both clients exited 0, with 2569/2600 successful profiling requests and 1/0
profiling errors; the treatment had one warmup error. Among 2506 matched
successful requests, output-length distributions were identical and mean
inter-token latency decreased 7.58%. Completed workload mixes differed, and
one run does not establish statistical significance or isolate each change.
See the [benchmark evidence and dependency overlay](../.github/benchmark/dp_cache_balance_ab.md#incremental-routing-comparison-2026-10-27-utc)
for denominator, failure accounting and reproduction limits.

#### Earlier policy comparisons

A single same-configuration AgentX A/B on V4.1-Flash, two MI355X GPUs,
attention DP2 / MoE TP2, 128 concurrent session slots and 900 s profiling
arrivals compared `least_requests` with the original `cache_affinity` policy.
Both arms already enabled unified prefix hashing and LMCache MP PAGE/STATE;
both used the same image, source, GPU pair, seed, warmup and real DSpark5.

| Metric | `least_requests` | Original `cache_affinity` |
| --- | ---: | ---: |
| P90 TTFT | 7.668 s | 5.324 s |
| P99 TTFT | 23.559 s | 12.012 s |
| Output tokens/s/GPU | 908.43 | 941.53 |
| P90 inter-token latency | 84.24 ms | 86.49 ms |

This is evidence for the combined routing/recovery policy on this workload,
not an isolated measurement of migration, LMCache versus no LMCache, or a
comparison with unmodified main. The original measured policy had overload
gates that were subsequently replaced by the net-gain comparison described
above. Later runs observed lower TTFT, but their different GPU pairs and host
load prevent treating all of that reduction as an isolated policy effect.

Measured transfers were within one host. Cross-host deployment and global
replica eviction still need validation. The policy moves later requests in a
session; it does not live-migrate an active decode.

A later short-recovery reconsideration experiment was removed after a parallel
900 s AgentX comparison on `792d6d6`. Both arms used cache affinity; the control
disabled only reconsideration. On V4.1-Flash, MI355X, attention DP2 / MoE TP2,
c128 and separate LMCache MP servers, the experiment reduced migration fallback
events from 179/578 to 42/554, but output tokens/s/GPU fell from 1089.50 to
1054.61 (-3.20%), total tokens/s/GPU fell 1.89%, and P90 inter-token latency
increased from 72.352 to 78.254 ms (+8.16%). The current code retains the
control's prior-owner fallback behavior.

Throughput uses successful profiling tokens divided by first request start to
last request end (929.675 / 928.065 s including completed drain requests), then
by two GPUs; total includes cached input. Counter samples bracket the 900 s
arrival window within 1.76 s before start and 0.35 s after end. This is one
run on different GPU pairs sharing a host, with 3/1 error records and 23/34
cancelled credits at the finite-window end, not a significance claim.

The 2484 shared successful requests had matching output-length distributions,
but mean input minus reported cached input increased from 5727.17 to 6005.50
tokens (+4.86%) with reconsideration. It also added a source probe, with 27/165
timeouts. A cache snapshot taken before rejection cleanup is not a reservation
for a later resubmission. Future reconsideration must account for actual
retained PAGE/STATE and demonstrate net throughput and latency benefit before
replacing fallback.

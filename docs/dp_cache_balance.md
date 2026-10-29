# Cache-aware DP placement with LMCache

`--dp-load-balance cache_affinity` keeps successive requests with the same
session ID on their current owner. When another rank is expected to finish
a later turn sooner by enough to cover the gain margins, it can send that turn
there and let the existing LMCache MP connector
recover its prefix. The target must acknowledge a usable prefix before the
router commits the new owner. This policy is opt-in.

## Shared prefill cost and progress

Routing and chunk scheduling both use `prefill_cost(c, q, w) = q * (w + c + q/2)`:
`q` is the work still to compute, `c` is the completed prefix, and `w` comes
from `ATOM_DP_PREFILL_BALANCE_TOKEN_WEIGHT`. Equal token counts after different
prefix lengths therefore need not have equal routing cost. Costs add across
chunks: completed work plus remaining work equals the original estimate.

The router sums costs per request, rather than squaring the rank's combined
queue length. At dispatch it uses prefix hints; admission replaces them with
the actual forward start. Only a completed prefill batch reports its remaining
tokens and prefix, using frozen batch offsets. Revisions reject stale reports;
preemption resets the prefix and charges recomputation, including generated
context. Finish/cancellation releases the same token records from which work
is derived. No per-token decode progress messages are added.

`prefill_tokens_per_second` remains a heuristic rate, now explicitly anchored
to a cold chunk of `prefill_reference_tokens` (default: the configured maximum
batched-token budget). The conversion is
`work_per_second = rate * (w + reference_tokens/2)`. Queue and candidate costs
use this same conversion; transfer estimates remain in seconds. This is not
V4.1 sparse-attention calibration, nor does it change the scheduler's current
`min(max, mean * (1 + slack))` target or minimum-token floor.

The router statistics expose `queued_prefill_work_per_rank` separately from
`queued_prefill_tokens_per_rank`, plus the assumed rate and token weight under
`cache_balance`. Work units are predictions, not GPU elapsed time.

## Rejecting an impossible restore early

Before target KV allocation and transfer submission, admission compares the
promised prefix with an optimistic bound from local HBM, a known LMCache
lookup answer, PAGE-only coverage, and any alignment handoff. A confirmed short
bound returns the request to its owner through the existing cleanup path and
increments `migration_fallback_before_allocation`. Lookup/memo reads add no
new RPC. Unknown answers, timeouts, or a composite connector with an unknown
provider do not establish a cache miss. An unavailable lookup at final admission
uses `lookup_unavailable`, which does not populate the negative cache.

A sufficient bound does not acknowledge residency: the existing final
PAGE/STATE completion and admission checks still decide whether migration can
commit. Failed DMA and its buffer/lease lifetime rules remain unchanged.

## Optional prepare / commit protocol

Set `"prepare_migrations": true` in `ATOM_DP_CACHE_BALANCE_CONFIG` to enable
source-owned preparation. It defaults to false while this protocol is being
validated. All coordinator and engine processes must use the same version.

1. The request stays runnable on its source. Once that scheduler confirms it
   is queued, the coordinator sends a non-runnable shadow to the target.
2. The target asynchronously looks up and pins the recoverable prefix, without
   allocating destination KV or submitting a retrieve. With bounded replay it
   pins PAGE alone; STATE is reconstructed by the existing replay path. With
   image-based recovery it pins the grouped PAGE+STATE prefix. Its
   HBM probe helps estimate transfer cost. In this first implementation the
   **entire promised prefix must be pinned in the tier**: HBM alone is not a
   credential, so eviction of an unpinned HBM hint cannot break the promise.
3. READY reports both the pinned transport boundary and the actual forward
   start, computed by the scheduler's `replay_start` rule. Routing charges the
   suffix from this earlier start, including STATE reconstruction, while
   transfer cost covers the PAGE range. The coordinator rechecks current source/target work and the decode
   guard. It asks the source to relinquish the request only if still useful.
   The source scheduler serializes release with allocation: if allocation won,
   it keeps the request and the target preparation is cancelled. No extra
   permission round trip is added to the source's execution path.
4. Only SOURCE_RELEASED permits COMMIT, after a second cost check. The target
   validates its generation, local deadline and held lookup before moving the
   same shadow Sequence into normal admission. Final native transfer/admission
   still gates actual execution. A cancelled or expired uncommitted shadow is
   never runnable. Once committed, ordinary DMA-safe cleanup owns its resources.

Configure `lmcache.mp.prepare_read_ttl_seconds` in the connector extra config
as a confirmed **lower bound** of `--l1-read-ttl-seconds` across all reachable
LMCache servers. Do not assume the installed LMCache default. The target uses
at most half that bound and at most `prepare_timeout_seconds`, measured on its
own clock from before lookup submission. Missing/zero TTL, a synchronous-only
adapter or an unsupported composite provider keeps execution
on the source. A multi connector supports preparation only with one offload
provider. PAGE-only and image-based recovery retain their existing separate
namespaces and read-lock contracts; no LMCache package patch is added by this
protocol. Preparation never changes the model's replay mode.

Generation numbers fence stale messages and survive router resets. Target
expiry during source release waits for the source's acknowledgment before any
requeue. The ordered control channel is required: a missing release ACK or an
ambiguous commit send must never be interpreted as permission to execute at the
other rank. Worker/connection loss uses the existing request failure/abort path;
this is not a durable distributed exactly-once protocol across process crashes.

Counters `migration_prepare_started`, `migration_prepare_source_won`,
`migration_prepare_gain_disappeared`, `migration_prepare_short`,
`migration_prepare_failed`, and `migration_prepare_committed` separate prepare
outcomes from actual migration dispatch. CPU state-machine and connector tests
do not establish an end-to-end throughput benefit; matched GPU measurement is
still required.

## Configuration

Enable prefix caching, frontend prefix hashes, and the `lmcache_mp` offload
connector. The MP server must be running before model startup. V4.1 uses the
native connector, BF16 KV and DP attention. Keep the serving recipe's bounded
replay setting: preparation supports PAGE-only recovery with STATE rebuilt
by replay and does not require `--no-encoder-swa-bounded-replay`.
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
The normalized squared-imbalance score uses the request-count and context means
after including the incoming request, identical for every candidate rank. Short
incoming requests emphasize batch size and long requests emphasize context load.
New sessions use this score
among ranks within the existing minimum useful time-gain margin. A migration must
meet the time-gain requirements and avoid increasing joint decode imbalance
relative to staying with the owner. Explicit rank selection remains authoritative.

## Admission, copying, and failure

The implementation lives in `atom/model_engine/dp_cache/`. `balance.py` owns
the cost policy and coordinator `Session`/`Placement` records; `routing.py`
reconciles placement events with CoreManager's load records through explicit
load-container, lock, and retry-sender dependencies. `admission.py` uses a
declared scheduler interface for queues and cleanup. Its preparation and work
queues are initialized with the component.

`Sequence.dp_cache` is either `None` or a `DPCacheState` from `state.py`. The
router initializes the migration contract, and the worker scheduler updates
its own serialized copy with preparation, admission, and work-report state.
Admission/work events reconcile that copy with the coordinator; it is not a
shared owner record. Retry creates fresh request state at the previous owner.
The scheduler and connector continue to own buffer leases and resource release.

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

## Validation scope

GPU validation covers same-host DP2 placement and prefix recovery. Cross-host
deployment and global replica eviction still need validation. The policy moves
later requests in a session; it does not live-migrate an active decode.

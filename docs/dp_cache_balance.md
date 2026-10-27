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
export ATOM_UNIFIED_PREFIX_CACHE=1
python -m atom.entrypoints.openai_server ... \
  --enable_prefix_caching --dp-load-balance cache_affinity \
  --kv-transfer-config '{"kv_connector":"lmcache_mp","kv_role":"offload"}'
```

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

### Measured scope

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
session; it does not live-migrate an active decode. A target currently needs
to recover the source hint boundary before ownership commits; accepting a
shorter recoverable prefix after reevaluating net benefit is future work.

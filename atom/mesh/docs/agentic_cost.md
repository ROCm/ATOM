# Agentic prefill cost

`policies::agentic` scores eligible workers using externally supplied cache-hit
and load snapshots:

```text
uncached_tokens = max(prompt_tokens - cached_tokens, 0)
cost = inflight_prefill_tokens + uncached_tokens + 512 * inflight_requests
```

The minimum cost wins, with uniform random tie-breaking. The result includes the
original worker index and `uncached_tokens`, the charge for the eventual placement.
Cache hits save their corresponding prefill cost; there is no extra hit bonus.
The request weight is ATOM's default `ATOM_DP_LB_REQ_EQUIV` of 512.

## Integration boundary

This module is a standalone cost function, not yet a selectable router policy.
It does not enable a CLI option or change request routing. Integration and GPU
acceptance wait for the KV-event path led by
[PR #3252](https://github.com/ROCm/ATOM/pull/3252), including the
[prefix-match provider in #3274](https://github.com/ROCm/ATOM/pull/3274).

```text
KV-event index -> unified prefix-match provider --+
                                                  +-> agentic cost -> worker
shared per-worker prefill/request load -----------+
```

The integration should:

1. Filter workers for health, model, pool and any hard affinity constraints.
2. Query the shared provider once for those workers. Map its answer to each
   candidate's reusable leading token count. `None` means unknown and earns no
   cache credit; `Some(0)` means a known miss. Both remain eligible. Event trust,
   prediction and invalidation belong to the provider.
3. Supply per-`url@rank` in-flight prefill tokens and requests from common load
   accounting. These are snapshots, not counters owned by the cost function.
4. Select a worker. If a later rank mapping changes the actual destination,
   recompute the charge using that destination's hit information before accounting
   for it. Account for the final destination and release its prefill charge on
   prefill completion or cancellation.

The provider's `CacheHit.tokens` entries are cumulative tier depths, so they must
not be added together. Lower-tier cache credit must correspond to the worker's
usable load path, not merely to KV stored somewhere. The conversion of tier hits
to reusable tokens, and any transfer-cost weighting, must be agreed during joint
validation. This module takes that resolved token count as input.

## Validation

The unit tests cover cache reuse versus queueing, request pressure, unknown and
zero hits, clamping cache credit, counter overflow, ties and empty candidates:

```sh
cargo test --manifest-path atom/mesh/Cargo.toml --lib policies::agentic
```

Joint validation must also check the provider adapter, final-rank accounting and
the prefill charge's release timing, before measuring AgentX TTFT and throughput.

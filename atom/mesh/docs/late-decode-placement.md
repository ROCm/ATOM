# Experimental independent Decode placement for ATOM P/D

The existing HTTP P/D path selects and reserves both workers before sending
Prefill. Its Decode outstanding count therefore includes requests still waiting
for Prefill. With variable Prefill latency, that placement can be stale when
Decode actually starts.

`--atom-pd-late-decode` defers Decode placement until Prefill returns its handoff
metadata. This is disabled by default and currently supports only ATOM's ordinary
non-chunked HTTP relay with one output sequence per request.

## Selection and lifetime

1. Admit the request and check that a healthy Decode candidate exists, without
   selecting or reserving one. The prototype admits at most 2048 HTTP lifetimes
   per router, returning 429 before Prefill when full.
2. Select a Prefill service through the configured Prefill policy. The service's
   backend selects its DP rank; its response must identify the actual rank.
3. After Prefill completes, filter Decode candidates by model, HTTP protocol,
   health and availability. Across all eligible Decode ranks, minimize the pair
   `(outstanding requests, outstanding context-token budget)`. Rotate exact ties.
4. Select and increment the reservation under the same mutex, before awaiting
   Decode dispatch. This prevents a burst from repeatedly choosing the same
   apparently empty rank.
5. Hold the reservation while waiting for KV, queued at Decode, generating and
   streaming the response. Release it on response completion, cancellation or
   failure. Do not retry the whole P/D request after producing a handoff.

Requests still at Prefill are excluded from Decode reservations. Context budget
is the formatted prompt length returned in `prompt_token_ids`, falling back to
`usage.prompt_tokens`, plus explicit `max_completion_tokens` or `max_tokens`.
Without an explicit output limit, the added output budget is zero. This estimate
is not updated as generation progresses and is not physical KV memory usage.

Decode session affinity and P-to-D rank mapping are bypassed in this mode.
Subsequent turns can select a different Decode rank; an active generation is
never migrated. The configured Prefill policy and backend session routing remain
responsible for Prefill locality.

## Enable

Add these options to an ATOM HTTP P/D router launch with the normal model,
Prefill URLs and Decode URLs:

```text
--backend atom
--pd-disaggregation
--dp-aware
--atom-pd-prefill-backend-routing
--atom-pd-rank-mapping-policy none
--atom-pd-late-decode
--atom-pd-decode-balance request-count
```

`request-count` includes the context-budget tie-breaker. Incompatible backend,
rank-mapping and chunked-transfer configurations are rejected. Multi-sequence
requests (`n > 1`, `best_of > 1`, batch size > 1) are rejected on this path.

### Independent of Prefill cache and routing implementations

This router does not depend on a Store connector or the `store_tokens` backend
policy. P may use local prefix caching, a shared external cache, session routing
or another backend policy. Only the ordinary ATOM handoff is required: the
actual P rank, KV metadata and formatted prompt token IDs or usage.

Keep `--dp-aware` with `--atom-pd-prefill-backend-routing` when P chooses its
own rank: P is represented as a service, while D remains selectable per rank.
Disabling `--dp-aware` also removes rank-level D placement. Without the
backend-routing flag, ordinary rank-aware routing selects P's rank externally.

D's reservation uses the full formatted context plus output budget. Cache hits
reduce Prefill computation but do not reduce the context Decode attends to.
Exact D ties rotate by stable worker/rank identity, independent of registry
order. No P-to-D rank mapping or Decode session affinity is required.

Aggregate gauges expose reservations, labeled by worker and DP rank only:

- `atomesh_pd_decode_reserved_requests`
- `atomesh_pd_decode_reserved_context_tokens`

These are router-local reservations, not scheduler running counts. Multiple
router processes do not share this state. Slow clients can retain a reservation
after backend work completes; disconnect does not prove backend work has stopped.

## Validation and interpretation

The tested prototype used one four-GPU Prefill instance and two four-GPU Decode
instances (DPA enabled, EP disabled), V4.1-Flash-1017 and LMCache MP. The fixed
mixed-request comparison used 768 requests at concurrency 128, identical input
texts/submission order, fresh backends/cache and identical warmup for each arm.
It is a closed-loop workload, so arrival timestamps are not identical.

Both arms completed all requests with zero errors, 7,064,292 input tokens and
2,170,240 output tokens. Output lengths were 64/2048/8192. Dataset SHA256:
`a73e1581171cde4c51b81ac26a44e399fe8510bbd2405ef8131e800f1c31da59`.

| Measurement | Early paired placement | Late Decode placement |
| --- | ---: | ---: |
| Elapsed seconds | 76.920 | 77.649 |
| Mean TTFT seconds | 2.978 | 3.160 |
| P90 TTFT seconds | 5.944 | 6.064 |
| P99 TTFT seconds | 9.593 | 9.480 |
| Mean rank-running CV during sustained submission | 0.192 | 0.112 |
| Median max/min running ratio during sustained submission | 1.750 | 1.364 |
| Completely empty rank samples during sustained submission | 0% | 0% |
| Full-window mean rank-running CV | 0.274 | 0.273 |

The sustained-submission window excludes the first five seconds and the final
drain after the last request starts (50/49 samples). This shows more even Decode
distribution in that window, **not a demonstrated throughput or TTFT benefit**.
Neither arm had idle ranks in that window; full-window idle samples were
4.561%/5.167%. A single short pair is insufficient for a performance claim.

The final draft passed `cargo fmt --all` and release-mode library tests with
filters `routers::http_pd_router` (55), `config::` (91), `backend_routing` (8),
and `late_decode` (16), in a 2-CPU/8-GiB container without GPUs. These test
filters overlap. The published branch removes an unused alternative policy and
fixes nullable output-limit fallback and log correlation. GPU results above
describe the prototype before those cleanups.

## Draft limitations

- No scheduler running/remaining-work or physical KV-capacity feedback, shared
  router state, output-progress accounting, or per-step KV admission pacing.
- No late-placement implementation for the ext-proc selected-pair path.
- Source staged-KV reclamation when Decode disappears or handoff fails relies
  on the existing ATOM cleanup protocol. Router-budget release is covered by
  tests; end-to-end KV fault/cancellation cleanup still needs validation.
- Admission is a fixed router-wide cap, not a rank memory-safety guarantee.
- The non-chunked relay setup is partly duplicated and should be consolidated
  before promoting this prototype to the default path.

The design takes inspiration from independent P/D routing and Decode workload
accounting; it does not claim to reproduce DeepInfer or Dynamo's production
schedulers. Request count is primary, with context as a tie-breaker, rather than
assuming context length alone predicts Decode compute cost.


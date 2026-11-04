# Decode-side P/D admission limits

Mooncake transfers reserve destination KV blocks before asking the producer to
write them. Many queued transfers can therefore hold most of the
decode KV pool while few requests can execute. The optional `pd_admission`
policy limits both pending logical requests and their physical receive targets
for whole-prompt and chunked transfers. Both modes share one N/W budget.

Enable it only on the decode service, in the top-level `--kv-transfer-config`:

```json
{
  "kv_connector": "mooncake",
  "kv_role": "kv_consumer",
  "pd_admission": {
    "enabled": true,
    "max_pending_kv_fraction": 0.15,
    "min_free_kv_fraction": 0.1
  }
}
```

Existing connector options still apply. With `multi`, put `pd_admission` alongside
`connectors`, and list the Mooncake consumer first, followed by the offload tier.
Only the winning P/D receive consumes credits; Store loads retain their existing
admission behavior. To combine with chunked transfer, also enable
`"enable_chunked_transfer": true` on the **producer's** Mooncake connector and
use the existing chunk-capable router, as described in the
[P/D guide](../recipes/pd_disaggregation_guide.md#chunked-kv-transfer-including-pp-prefill).
The router sets each request's transfer mode; the decode admission configuration
above works for either mode without requiring a separate chunk flag on D.

| Option | Default when enabled | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Opt in; absent/false preserves the existing scheduler. |
| `max_pending_requests` | `null` | Logical requests, including received requests awaiting their first scheduled forward. `null` disables this dimension. |
| `max_pending_kv_fraction` | `0.15` | Fraction of each D rank's physical paged KV pool, converted to bytes at startup. `null` disables this dimension. |
| `max_pending_kv_bytes` | unset | Explicit bytes **per D rank**, instead of a fraction. Do not specify both byte options. |
| `min_free_kv_fraction` | `null` | Fraction of D's paged pool that must stay free after a new receive reserves its targets: the headroom kept for the running batch's decode growth. Free includes cached blocks a prefix hit could still reuse. `null` disables this dimension. |

At least one limit must be enabled. For a count-only experiment, set
`max_pending_requests` and explicitly set `max_pending_kv_fraction` to `null`.
Values are experiment starting points, not performance recommendations.

The request count is off by default. A fixed count caps intake at about
`max_pending_requests / receive latency` however much of D is free, and in
whole-prompt mode every request D defers keeps its source KV on P. The byte
limit bounds what non-running receives hold; the free-KV watermark keeps room
for requests that are already decoding.

The policy supports a single paged KV pool, including target,
index and draft regions sharing the same page IDs. D must have PP=1, DCP=1,
PCP=1 and identical page sizes across TP ranks. Separate state/windowed-draft
pools are rejected when this policy is enabled. P can use PP;
PP4→TP4 still counts as one logical receive, not sixteen requests. Startup
reconciles the TP capacity once and reports the actual page and budget bytes.
Existing transfer topology constraints still apply, including D TP being a
multiple of P TP for chunked transfers.

Admission checks run after the local prefix probe, before allocation or any
receive notification. Prefix blocks that are not transferred do not consume
receive-byte credits. The policy uses the connector's real prefix-skip rules,
including full-transfer fallback for incompatible source geometry. It does not
estimate physical occupancy from prompt tokens or network bytes.

In chunked mode, D reserves the complete prompt destination before P allocates
its source KV or starts prefill. The admission estimate preserves a compatible
local prefix even when source block IDs are not yet available. W charges all
reserved receive targets throughout the transfer; receiving an individual
chunk does not reduce W or release N. Neither P's admission-ready signal nor
its final KV chunk is D's compute-ready signal: D waits for the existing
all-stage/all-rank completion and final sampling handoff.

Credits remain held through receive completion until the request is selected
into its first local compute batch. Changing the request status to RUNNING
alone does not release them. Returning credits never frees a running request's
KV. A ready request preempted before its first batch returns its credits when
its received blocks are discarded. Cancellation holds credits until the
existing safe terminal receive/cleanup path; partial stage/rank completion and
logical timeouts do not release them early.

The producer's existing `chunked_admission_timeout_s` continues to bound its
wait for D readiness and source capacity. N/W does not extend that deadline;
if it expires after D has reserved targets, D keeps its credits until the
existing safe failure report. Requests that have not reserved on D hold no
N/W credits.

New requests retain FIFO order. A request that exceeds the soft byte limit or
the free-KV watermark may enter alone once no pending credit remains, so a
long request does not wait forever because of either soft limit. Existing
physical capacity, lookahead and `max_num_seqs` checks remain in force. The
watermark is a fraction of the pool, not a per-request decode growth estimate,
and adds no extra lookahead blocks.

Metrics use `dp_rank` and `engine_role` labels:

- `atom:pd_admission_pending_requests{state="receiving|ready|cancel_draining"}`
- `atom:pd_admission_pending_kv_bytes`
- `atom:pd_admission_deferred_total{reason="requests|bytes|free_kv"}` (requests, each counted once per limit that deferred it)
- `atom:pd_admission_credit_wait_seconds` (from a request's first deferral to its admission; 0 when admitted at once)
- `atom:pd_admission_oversized_total`
- `atom:pd_admission_ready_wait_seconds`

P's retained source KV is exported by every scheduler as
`atom:kv_cache_blocks_deferred_free` (physical blocks) and
`atom:requests_deferred_free`: on P/D prefill, finished requests whose KV decode
has not pulled yet, plus any pending offload save.

Compare baseline (`enabled: false`) → bytes only → bytes plus watermark, and a
count-only arm if the count is of interest, with unchanged model/cache
configuration and chunk/landing flags. Include completed request throughput,
end-to-end latency, preemptions and P's deferred-free blocks. A lower pending
count alone is not evidence of a throughput improvement: D backpressure can
move the queue to P. Real GPU/RDMA validation is required before deployment.

With chunked transfer, D admission also bounds P prefill concurrency, and
credits span the wait for P computation as well as transfer. Retune N/W for
that mode: limits that are too small can underfill P's PP pipeline. Compare
chunked on/off separately, keeping the compute chunk budget fixed within each
N/W comparison; throughput gains are not assumed to add together.

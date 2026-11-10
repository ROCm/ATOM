# Decode-side P/D admission limits

Mooncake transfers reserve destination KV blocks before asking the producer to
write them. Many queued transfers can therefore hold most of the
decode KV pool while few requests can execute. The optional `pd_admission`
policy limits the physical receive targets pending requests hold (W), for
whole-prompt and chunked transfers. Both modes share one budget.

Enable it only on the decode service, in the top-level `--kv-transfer-config`:

```json
{
  "kv_connector": "mooncake",
  "kv_role": "kv_consumer",
  "pd_admission": {
    "enabled": true,
    "max_pending_kv_fraction": 0.15
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
| `max_pending_kv_fraction` | `0.15` | Fraction of each D rank's physical paged KV pool, converted to bytes at startup. |
| `max_pending_kv_bytes` | unset | Explicit bytes **per D rank**, instead of a fraction. Do not specify both byte options. |

Values are experiment starting points, not performance recommendations.

There is no limit on the number of pending requests. A fixed count caps intake
at about the count divided by the receive latency however much of D is free,
and in whole-prompt mode every request D defers keeps its source KV on P. The
byte limit bounds what non-running receives hold.

The policy supports a single paged KV pool, including target,
index and draft regions sharing the same page IDs. D must have PP=1, DCP=1,
PCP=1, SP=1 and identical page sizes across TP ranks. Separate state/windowed-draft
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
chunk does not reduce W. Neither P's admission-ready signal nor
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
wait for D readiness and source capacity. Admission does not extend that
deadline; if it expires after D has reserved targets, D keeps its credits until
the existing safe failure report. Requests that have not reserved on D hold no
credits.

New requests retain FIFO order. A request that exceeds the soft byte limit
may enter alone once no pending credit remains, so a long request does not
wait forever. Existing physical capacity, lookahead and `max_num_seqs` checks
remain in force.

Compare baseline (`enabled: false`) against the byte limit, with unchanged
model/cache configuration and chunk/landing flags. Include completed
request throughput, end-to-end latency and preemptions. A lower pending count alone is not evidence of a throughput
improvement: D backpressure can move the queue to P. Real GPU/RDMA validation
is required before deployment.

With chunked transfer, D admission also bounds P prefill concurrency, and
credits span the wait for P computation as well as transfer. Retune W for
that mode: a limit that is too small can underfill P's PP pipeline. Compare
chunked on/off separately, keeping the compute chunk budget fixed within each
W comparison; throughput gains are not assumed to add together.

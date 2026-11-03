# Decode-side P/D admission limits

Non-chunked Mooncake transfers reserve destination KV blocks before asking the
producer to write them. Many queued transfers can therefore hold most of the
decode KV pool while few requests can execute. The optional `pd_admission`
policy limits both pending logical requests and their physical receive targets.

Enable it only on the decode service, in the top-level `--kv-transfer-config`:

```json
{
  "kv_connector": "mooncake",
  "kv_role": "kv_consumer",
  "pd_admission": {
    "enabled": true,
    "max_pending_requests": 4,
    "max_pending_kv_fraction": 0.15
  }
}
```

Existing connector options still apply. With `multi`, put `pd_admission` alongside
`connectors`, and list the Mooncake consumer first, followed by the offload tier.
Only the winning P/D receive consumes credits; Store loads retain their existing
admission behavior.

| Option | Default when enabled | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Opt in; absent/false preserves the existing scheduler. |
| `max_pending_requests` | `4` | Logical requests, including received requests awaiting their first scheduled forward. `null` disables this dimension. |
| `max_pending_kv_fraction` | `0.15` | Fraction of each D rank's physical paged KV pool, converted to bytes at startup. `null` disables this dimension. |
| `max_pending_kv_bytes` | unset | Explicit bytes **per D rank**, instead of a fraction. Do not specify both byte options. |

At least one limit must be enabled. For a count-only experiment, explicitly set
`max_pending_kv_fraction` to `null`. Values are experiment starting points, not
performance recommendations.

The initial implementation supports a single paged KV pool, including target,
index and draft regions sharing the same page IDs. D must have PP=1, DCP=1,
PCP=1 and identical page sizes across TP ranks. Separate state/windowed-draft
pools and chunked P/D are rejected when this policy is enabled. P can use PP;
PP4→TP4 still counts as one logical receive, not sixteen requests. Startup
reconciles the TP capacity once and reports the actual page and budget bytes.

Admission checks run after the local prefix probe, before allocation or any
receive notification. Prefix blocks that are not transferred do not consume
receive-byte credits. The policy uses the connector's real prefix-skip rules,
including full-transfer fallback for incompatible source geometry. It does not
estimate physical occupancy from prompt tokens or network bytes.

Credits remain held through receive completion until the request is selected
into its first local compute batch. Changing the request status to RUNNING
alone does not release them. Returning credits never frees a running request's
KV. A ready request preempted before its first batch returns its credits when
its received blocks are discarded. Cancellation holds credits until the
existing safe terminal receive/cleanup path; partial stage/rank completion and
logical timeouts do not release them early.

New requests retain FIFO order. A request larger than the soft byte limit may
enter alone once no pending credit remains, so a long request does not wait
forever just because of the byte budget. Existing physical capacity,
lookahead and `max_num_seqs` checks remain in force. This change adds no
cross-request decode growth reservation (G) and no extra lookahead blocks.

Metrics use `dp_rank` and `engine_role` labels:

- `atom:pd_admission_pending_requests{state="receiving|ready|cancel_draining"}`
- `atom:pd_admission_pending_kv_bytes`
- `atom:pd_admission_deferred_total{reason="requests|bytes"}` (attempts, not unique requests)
- `atom:pd_admission_oversized_total`
- `atom:pd_admission_ready_wait_seconds`

Compare baseline → count only → count plus bytes with unchanged model/cache
configuration and chunk/landing flags. Include completed request throughput,
end-to-end latency, preemptions and P's retained source KV. A lower pending
count alone is not evidence of a throughput improvement: D backpressure can
move the queue to P. Real GPU/RDMA validation is required before deployment.

# V4.1 DP cache balance A/B

Select `dp-cache-ab` in the ATOM Agentic Benchmark workflow. It creates two
independent cells, each using attention DP2 / MoE TP2 on two MI355X GPUs,
128 concurrent session slots and 3600 seconds of profiling arrivals by default.
The workflow's `duration_seconds` input can override the duration for both arms.
To run only ON, set `variants` to `-agentic-dspark5-dtp2-cache-on`.
Leaving `variants` empty retains the two-arm comparison. Unknown suffixes fail
matrix generation instead of silently starting extra jobs.

| Arm | DP policy | Legacy session affinity |
| --- | --- | --- |
| OFF | `least_requests` | `0` |
| ON | `cache_affinity` | `0` |

Both arms enable unified prefix caching and LMCache MP with separate PAGE/STATE
object groups, 256 GiB host L1, LRU eviction and identical recovery thresholds.
The ON arm uses the policy defaults, including a 5-second preparation timeout.
This isolates the routing/recovery policy; it does not compare LMCache on/off
or compare this feature branch with unmodified main.

The workload uses the main branch's `agentx` scenario with the SA 393-entry
trace, seed 42, 5 warmup requests per lane, no context truncation and real
DSpark5 acceptance. GPU memory utilization
is 0.9. Pin `image` and `atom_commit` at dispatch to reproduce a run. The image
must include the LMCache MP CLI with native PAGE/STATE object-group support.

The opt-in helper starts LMCache inside each benchmark container before ATOM
and waits for its engine health check. Container teardown also stops LMCache.
The usual benchmark bundles contain TTFT, ITL, token throughput and errors;
`atom-cache-diagnostics-*` adds the LMCache log/status and final DP router
counters. These final counters include warmup and are not profiling-only
deltas. The two cells may run on different dedicated CI hosts, so host effects
remain a limitation of a single A/B run.

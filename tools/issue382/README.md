# Issue #382: first experiment round

Inference source: `1fb68bb34f0f32e24ffc26e813c2740cd5126d49` (PR #2941).
This branch changes experiment configuration, startup helpers, and metrics collection.
Original workload: `Jasen/m31-mxfp4-stack-agentx-20261031`, issue #380 C140.

Use `atomesh-benchmark.yaml`, mounted source, image
`rocm/atom-dev:m31-mxfp4-stack-20261031`, runner `atomesh-cicd`, and
`atomesh_single_node=auto`, `atomesh_1p1d_nodes=auto` for all four cases.
Do not pin a host or supply a candidate node list. Each allocation requests exclusive access to
8 MI355X GPUs. Do not cancel other users' jobs to obtain the node.
Do not publish these smoke/experimental results to the dashboard.

Set `ATOMESH_DROP_HOST_CACHES=0`: the global cache-drop helper requires a
privileged container, which `spur-authz` rejects on managed nodes. Keep the
existing unprivileged model-file cache eviction, NUMA capacity checks, huge-page
preparation, and the full 2 TiB Store. Check actual Store registration and ready
capacity in the runtime logs before treating a run as valid.

The first retry on g08 reached Store startup but failed RDMA registration after
both NUMA compactions timed out. Each NUMA node still held about 1.2 TiB of file
cache after advising only the selected model snapshot. Set
`MOONCAKE_STORE_PAGE_CACHE_DROP_DIRS=${ATOMESH_MODEL_ROOT}` for all four cases,
matching other Store recipes, so clean cached files from other model snapshots
under the mounted model root can also be evicted. This uses unprivileged
`posix_fadvise(DONTNEED)`; the host-wide cache-drop helper stays disabled.
The cache helper reports advised file counts, logical sizes and errors; these
sizes do not measure reclaimed memory. Verify subsequent NUMA free memory,
huge-page preparation and full Store registration in the actual retry.
Container log chunks are written without file buffering so a short startup
message remains visible during long preparation steps.

| Stage | Case | Client concurrency | N | W | Acceptance | Workload |
| --- | --- | ---: | ---: | --- | --- | --- |
| Correctness | issue382-a1-correctness64 | 16 | 4 | disabled | real DSpark7 | GSM8K 20-shot, first 64 samples |
| Correctness | issue382-a2-correctness64 | 16 | 4 | 15% | real DSpark7 | same 64 samples |
| A1 | issue382-a1-c140 | 140 | 4 | disabled | fixed AL=4.04 | AgentX 1.0.12, 3600s profiling after original warmup |
| A2 | issue382-a2-c140 | 140 | 4 | 15% | fixed AL=4.04 | identical AgentX workload |

Run correctness cases in the `accuracy` suite. Review completed sample
counts, errors, output lengths, sample answers, and admission/transfer logs.
The 0.90 score gate is a coarse short-smoke guard (at least 58/64); a 64-sample
result cannot establish the full GSM8K accuracy of the model. The usual 0.94
full-suite threshold must not be interpreted as a statistically precise gate
for this subset. Retried HTTP requests must still be counted as errors in the
analysis even if lm-eval eventually succeeds.

Submit A1 and A2 independently in the `mooncake_store` suite, without waiting for
correctness jobs to finish or for a particular node. This follows the user's
updated scheduling preference. Review correctness before interpreting performance.
Restart the processes and Store for every case. Correctness uses the same
280 scheduler slots and capture sizes as the performance cases to check the
actual server configuration; only client load and synthetic acceptance differ.
Chunk transfer is OFF for this round. Chunk ON performance and tuning wait for
the user's review of this round.

Fixed configuration: P TP1 x PP4, default PP split, eager P with 8192 token
budget, D TP4 with graphs, GPU memory utilization 0.9, block size 128, MXFP4
target KV/index, freq1, landing OFF, no G reservation. Store consists of
four 512 GiB owners on rdma4/5/6/7. Use the exact model snapshot
`0a00e0e39fc00499b23efccc18a8ea63fce5c44c` and its `dspark` subdirectory.
Do not substitute the older `1442d09d` model/chat template or the generic NVFP4
and 1 TiB recipe. The model root is remapped by the standard CI launcher.

The launcher checks mounted ATOM import origin. Record the expanded cell,
source SHA, image digest, actual allocated node, dependency versions, actual D KV pool capacity
and bytes/page from startup logs. Pinning memory utilization alone does not
guarantee identical pool capacity across processes.

`sample_metrics.py` saves both workers' Prometheus output every 5 seconds from
before the workload until cleanup, including warmup. It writes compressed
JSONL under `logs/issue382-metrics.jsonl.gz` (or the execution-phase subdirectory)
with timestamp, role, URL and raw metrics; scrape failures are explicit records.
The standard CI artifact includes this file alongside client records and server
logs. A preflight rejects unreachable metrics or missing admission metric names.
Avoid drawing conclusions from gaps in this sampled series; short excursions
can occur between samples.

Analyze per-minute request completions/output tokens, total/output tok/s/GPU
from client records over their actual span (8 GPUs), TTFT/TPOT/e2e, failures,
timeouts, and P/D KV use/running/waiting/prefix cache/preemptions. Report N states,
W bytes, requests/bytes deferrals, oversized bypasses, and READY wait, plus P
retained KV and Store loads where the existing logs expose them. Check for CQ
errors separately from scheduler congestion. A low parked count is guaranteed
by N and is not evidence of restored throughput by itself.

No A0 is included in this round, so this cannot isolate N against a same-revision
disabled-policy baseline. If W never binds at N=4, A2 measures compatibility
rather than evidence for W's protective value. Do not tune W silently.

Job time limits: 2 hours per correctness case, 4 hours per performance case,
including startup/warmup. Profiling remains 3600 seconds. Queue time is extra.

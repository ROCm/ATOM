# MiniMax M3.1 Flash: native vLLM 1P1D with Mooncake Store

Manual `atomesh-benchmark.yaml` dispatch on `atomesh-cicd`, suite
`mooncake_store`, model `MiniMax-M3.1-Flash-vLLM`. Run
`m31-vllm-1p1d-tp4-mcstore-smoke` first; only after success run
`m31-vllm-1p1d-tp4-mcstore-c32`, `-c48`, and `-c64`.

The image digest and model revision in `experiment.json` come from
https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39929767608/attempts/1.
Those MI355X results used aggregated TP4. This experiment tests the same
image in a new eight-GPU PD layout; it is not a published SA PD score.

One node: prefill GPUs 0–3, decode GPUs 4–7, TP4 each. Both enable GPU prefix
caching and use MultiConnector (Mooncake Transfer + Store). Store embeds
128 GiB per rank, totaling 1 TiB, with 4 GiB of private buffers per rank.
P does Store lookup/load/save; D saves completed decode blocks without Store
lookup. CPU Store stays enabled; `enable_offload: false` disables its disk tier.

The launcher uses installed image packages unchanged. The proxy is vendored
at the image commit; a wrapper adds health/model metadata routes. AgentX
1.0.12 runs in a separate venv. Its supported HF loader class revision is set
to the recorded dataset commit; the exact one-line dataset pin is recorded
in artifacts. Replay behavior is unchanged.

Smoke uses real block rejection, compares greedy PD with independent D
computation, checks GPU reset success, requires Store GET bytes after GPU
reset, and tests a D-only prefix subsequently loaded by P. It emits correctness
artifacts, not performance results. Formal runs use DSpark3 synthetic AL2.88,
393 AgentX traces, seed42, 3600 seconds per concurrency, with fresh engines and
Store for each cell. Throughput normalization uses all eight GPUs.

Artifacts include image/hardware versions, launch arguments, environment,
resolved configuration, per-service logs, Prometheus snapshots, smoke JSON,
and raw AgentX exports. Existing Slurm allocation, GPU busy checks, cleanup,
and result collection are reused. Do not publish dashboard results before
validating smoke and workload metadata.

The `m31-vllm-1p1d-tp4-mcstore-validated-sweep` case runs smoke followed by
C32/C48/C64 in one eight-GPU allocation. Each phase stops its engines and
Store before the next starts, and writes a separate artifact directory.
`suite-status.json` records progress. A failed smoke, benchmark, or teardown
stops the sequence. Each measurement still runs for 3600 seconds.

Automatic allocation currently excludes `pit2-p03-g09` after two unexplained
SIGKILL exits there; explicitly pinned nodes bypass the automatic exclusion.
The read-only inspection case emits schema-filtered status for its configured
Actions ID, never raw logs or arbitrary file content.

Memory preparation follows successful CI `40376309047` (commit
`fc9bf7aa215f33067a8b0728279d076c0da5a313`). The custom
`pd_vllm_worker.StoreMemoryWorker` waits until all eight P/D workers reach
`initialize_from_config`, after model loading/profiling and before native
connector initialization. A file lock elects one worker to evict the pinned
snapshot's clean page cache (including its draft subdirectory) and fault/free
THP with `numa_exec.py` + `compact_huge_pages.py`. There is no preparation or
periodic cache eviction during model loading.

Preparation is best effort: split the eight global Store segments equally over
NUMA nodes with CPUs, cap each node's target at `MemFree + Active(file) +
Inactive(file) - 64 GiB`, and skip targets below 1 GiB. This matches the
successful CI's sizing rule; private buffers are not added to the preparation
target. THP shortfall, a nonzero helper exit, missing measurements or a
600-second per-node timeout only warn, then proceed to the remaining nodes and
native Store registration. Available-memory estimates also only warn. Actual
Store registration determines success; preparation never emits exit 86.
Native Store retains the glibc THP hint and 64 GiB MR cap. The earlier malloc
perturbation, sequential P/D startup and separate native Store PUT/GET preflight
in `57888067b` were reverted.

`store-memory-gate/` records each worker's arrival and completion of the single
preparation attempt. Its `status=completed` allows every worker to initialize
Store even when `preparation_status=completed_with_warnings`. This does not
claim successful Store registration. Unexpected preparation errors still
publish a terminal failure for waiting workers; missing workers still time out.

The common environment explicitly sets `MC_MAX_CONCURRENT_REG_MR=1`,
`NCCL_DMABUF_ENABLE=0`, `MC_GID_INDEX=1`, `MC_IB_TC=104`, and
`MC_IB_PCI_RELAXED_ORDERING=1`. The first variable is parsed by the installed
Mooncake 0.3.13.post1 but does not limit its RDMA registration path. Relaxed
ordering is requested when libibverbs exposes the required symbol; verify the
actual selection in engine logs. This alignment does not change the fixed
image, native Mooncake package, Store buffer sizes or NIC selection.

A smoke-only dispatch uses suite `mooncake_store` and case
`m31-vllm-1p1d-tp4-mcstore-smoke` with `publish_dashboard=false`. It records
`smoke/` and `suite-status.json`, then exits without the C32/C48/C64 measurements.
The workflow's separate `suite=smoke` is an RDMA test, not this model smoke.

On the `atomesh-cicd` Spur runner, actual Store registration ENOMEM produces
a memory-specific exit (86), which first stops and
removes this allocation's container. A job/token/node-validated recovery helper
uses existing host write permissions or noninteractive `sudo -n` to sync, drop
host page cache and compact memory. It never elevates container capabilities.
Successful cleanup permits one fresh local run; old phase artifacts are archived
so result collectors cannot reuse previous partial measurements. Missing cleanup
permission or a second memory failure ends the allocation. Only after Slurm is
terminal, the submitter excludes that node and submits again (at most three
resubmissions, configurable with `ATOMESH_VLLM_MEMORY_RETRIES`). Explicitly pinned
nodes are not silently widened. Generic model and correctness failures do not
trigger this retry path. A host cleanup timeout is fatal, not resubmitted.

`memory-preparation.json` records targets, per-node outcomes and warnings;
`thp-node-*.json` records measured hugepage bytes when the probe finishes;
`vllm-memory-retry.json` binds the reason to the exact allocation;
`host-memory-recovery.json` records whether cleanup succeeded or lacked permission.
The model image, revisions, 1 TiB pool, per-rank 4 GiB buffer, prefix caches,
DSpark3 and benchmark workload remain as specified above.

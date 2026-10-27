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

Memory recovery follows PR #2750 (`e73246e80ec`): evict only the pinned model
snapshot's clean file cache with `drop_page_cache.py`, check NUMA budgets with
128 GiB reserved per node, then use `numa_exec.py` + `compact_huge_pages.py` to
fault/free the Store budget as THP. Private buffers are conservatively budgeted
on each GPU/NIC NUMA node because worker malloc placement is not guaranteed;
this only affects preparation, not the configured Store or buffer capacity. The native launcher checks the actual
`AnonHugePages` bytes; insufficient THP or a 600-second per-node preparation
timeout stops startup. Native Store uses the PR's glibc THP hint and 64 GiB MR
cap. The earlier malloc perturbation, sequential P/D startup and separate
native Store PUT/GET preflight in `57888067b` were reverted.

On the `atomesh-cicd` Spur runner, a memory-specific exit (86) first stops and
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

`memory-preparation.json` and `thp-node-*.json` record measured hugepage bytes;
`vllm-memory-retry.json` binds the reason to the exact allocation;
`host-memory-recovery.json` records whether cleanup succeeded or lacked permission.
The model image, revisions, 1 TiB pool, per-rank 4 GiB buffer, prefix caches,
DSpark3 and benchmark workload remain as specified above.

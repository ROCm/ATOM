# MiniMax M3.1 Flash: native vLLM 1P1D with Mooncake Store

Manual `atomesh-benchmark.yaml` dispatch on `atomesh-cicd`, suite
`mooncake_store`, model `MiniMax-M3.1-Flash-vLLM`. Run
`m31-vllm-1p1d-tp4-mcstore-smoke` first; only after success run
`m31-vllm-1p1d-tp4-mcstore-c32`, `-c48`, and `-c64`.

The active recipe uses main KV `nvfp4`, indexer KV `nvfp4`, and DSpark
draft KV `fp8` with the draft's `TRITON_ATTN` backend. Verify that both P and D
actually log `MiniMax M3 sparse attention selected ROCm NVFP4` before accepting
smoke or performance results. The worker records vLLM's native mixed-precision
and KV-zeroing predicates in `<role>-rank-<rank>-kv-precision.json`; those
correctness guards remain enabled. The earlier uniform-BF16 diagnostic still
failed with HIP 209 in `_apply_write_kernel`, so it is no longer active.
The current target is a passing smoke followed by the C32 case.

The image digest and model revision in `experiment.json` come from
https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39929767608/attempts/1.
Those MI355X results used aggregated TP4. This experiment tests the same
image in a new eight-GPU PD layout; it is not a published SA PD score.

One node: prefill GPUs 0–3, decode GPUs 4–7, TP4 each. Both enable GPU prefix
caching. P uses MultiConnector (`kv_both`): Mooncake Transfer (`kv_producer`)
plus Mooncake Store (`kv_both`, synchronous lookup, asynchronous load). D uses
only Mooncake Transfer (`kv_consumer`) and does not write or allocate Store.
Each P rank embeds 224,873,414,656 bytes (about 209.43 GiB), totaling about
837.72 GiB across four ranks, with 64 MiB of private buffers per P rank
(256 MiB total). CPU Store stays enabled; `enable_offload: false` disables its
disk tier. Per-rank settings match successful CI `40376309047`; that reference
has eight P ranks on a separate node, so its total pool is twice as large.

Mooncake registration settings follow that reference: `MC_MAX_MR_SIZE` is
1 GiB on P and 256 MiB on D; only P gets `GLIBC_TUNABLES=glibc.malloc.hugetlb=1`
and `MOONCAKE_CONFIG_PATH`. Each worker sets `MC_TE_FILTERS=rdma<physical GPU>`
before connector initialization. Store's `device_name` is empty and the
Transfer config omits it, allowing this filter to select one HCA per process.
No `MC_DISABLE_HIP` override is forced. Per-rank selection and Store membership
are recorded in `<role>-rank-<rank>-mooncake.json`; check engine logs for
`Topology discovery complete. Found 1 HCAs.` to confirm actual discovery.

The launcher uses installed image packages unchanged. The proxy is vendored
at the image commit; a wrapper adds health/model metadata routes. AgentX
1.0.12 runs in a separate venv. Its supported HF loader class revision is set
to the recorded dataset commit; the exact one-line dataset pin is recorded
in artifacts. Replay behavior is unchanged.

Smoke uses real block rejection, compares greedy PD with independent D
computation, checks GPU reset success, requires Store GET bytes after GPU
reset on P. D-only Store reuse is marked not applicable in
`smoke-store-scope.json` because D has no Store connector. It emits correctness
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

Automatic model allocations exclude `pit2-p03-g09`, `-g10`, `-g23`, and
`-g25` after previous interrupted allocations. The current no-model diagnostic
is pinned to `pit2-p03-g37`. Explicit dispatch placement overrides case placement.
Read-only inspection emits status and, when enabled, bounded logs/results from
this experiment's namespace into an Actions artifact.

Memory preparation follows successful CI `40376309047` (commit
`fc9bf7aa215f33067a8b0728279d076c0da5a313`). The custom
`pd_vllm_worker.StoreMemoryWorker` waits until all eight P/D workers reach
`initialize_from_config`, after model loading/profiling and before native
connector initialization. All eight wait so D cannot refill the shared node's
model cache during cleanup, while only P contributes Store memory. A file lock
elects one worker to evict the pinned
snapshot's clean page cache (including its draft subdirectory) and fault/free
THP with `numa_exec.py` + `compact_huge_pages.py`. There is no preparation or
periodic cache eviction during model loading.

Preparation is best effort: split the four P global Store segments equally over
NUMA nodes with CPUs, cap each node's target at `MemFree + Active(file) +
Inactive(file) - 64 GiB`, and skip targets below 1 GiB. This matches the
successful CI's sizing rule; private buffers are not added to the preparation
target. THP shortfall, a nonzero helper exit, missing measurements or a
600-second per-node timeout only warn, then proceed to the remaining nodes and
native Store registration. Available-memory estimates also only warn. Actual
Store registration determines success; preparation never emits exit 86.
P uses the glibc THP hint and the role-specific MR limits above. The earlier malloc
perturbation, sequential P/D startup and separate native Store PUT/GET preflight
in `57888067b` were reverted.

`store-memory-gate/` records each worker's arrival and the single preparation
attempt. Workers initialize Store only after hugepage compact finishes on every
NUMA node (`completed`), or that node is skipped because its target is below
1 GiB. A probe failure, a shortfall, or the per-node timeout publishes
`status=failed` and no worker mounts Store. Missing workers still time out.
Each node is given 1800 seconds. The coordinating worker runs pinned to its
GPU's NUMA node, so `numa_exec.py` binds within the container's cpuset rather
than the inherited affinity; inheriting it left node 0 with no allowed CPU and
its compact exited immediately (jobs 10945 and 10969).

The common environment explicitly sets `MC_MAX_CONCURRENT_REG_MR=1`,
`NCCL_DMABUF_ENABLE=0`, `MC_GID_INDEX=1`, and `MC_IB_TC=104`. It does not
set `MC_IB_PCI_RELAXED_ORDERING`. The successful Kimi reference left that
variable unset, and requesting `IBV_ACCESS_RELAXED_ORDERING` made this image's
host MR registration return EINVAL. The first variable is parsed by the installed
Mooncake 0.3.13.post1 but does not limit its RDMA registration path. The
Store/Transfer JSON configs match the successful reference. The MiniMax fixed image and native Mooncake package are
retained; the reference ran Kimi K3 with a custom Mooncake wheel, so this is
alignment of the Store/RDMA settings rather than a copy of its entire stack.

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
The model image, revisions, prefix caches, DSpark3 and benchmark workload
remain as specified above. Store belongs to P only, with the capacities above.

The current `m31-vllm-triton-probe` case selects the `registration_memory`
matrix. Four processes on GPUs 4–7 each allocate a 64 GiB non-KV fixture and
193 GiB KV storage, initialize the image's native Mooncake TransferEngine on
one HCA, and check fresh Triton kernels before/after registration. It loads no
model and starts no Store. Successful cases must also execute the installed
`_zero_kv_blocks_kernel` and verify zeroed and untouched elements exactly.

The default DMA-BUF path was isolated in CI `40801749720`: registering 64 GiB
subtracts another 64 GiB from KFD free VRAM; registering 193 GiB with about
30 GiB free clamps free VRAM to zero, then kernel loading fails with HIP 209.
`MOONCAKE_DISABLE_HIP_DMABUF=1` makes kernels run but reports about 148,833 GiB
free with 772 MR chunks, so it is not a valid fix for this experiment.
`NCCL_DMABUF_ENABLE=0` does not control Mooncake's registration path.

[Mooncake PR #4657](https://github.com/kvcache-ai/Mooncake/pull/4657) addresses
this amdgpu accounting bug by holding one peer-memory pin per allocation while
data MRs remain on DMA-BUF. The image does not contain its
`MC_HIP_KFD_PIN_FIX` implementation. `pd_vllm_kfd_pin.py` uses the image's HIP
and libibverbs ABI for the same allocation-level mechanism without replacing
the installed packages. The model adapter is opt-in through
`ATOMESH_VLLM_KFD_PIN_ADAPTER=1`; it is currently **not enabled** in the recipe.

Before adding anchors, the adapter tests a private 64 MiB HIP allocation to
distinguish affected and fixed drivers. It binds at the outer KV connector,
so P Transfer and Store share one anchor per underlying allocation. It checks
free VRAM after native registration, retaining the fixed KV pool and anchors
for the worker lifetime: a Transfer-only unregister cannot release a pin while
P Store still owns a data MR. This experiment adapter does not support replacing
or resizing the KV pool within a worker. Its diagnostic additionally removes
all native data MRs, releases anchors, and verifies the final memory accounting.

The active diagnostic cases are `anchor-193g`, `anchor-subranges-193g`, and
`adapter-193g`; the second covers multiple regions within one allocation,
including a length not aligned to 2 MiB, and the third exercises the private
driver probe and the model adapter entry point. Per-rank artifacts include
`memory.json`, `registration.json`, stage results, and `kfd-pin.json` for the
adapter. A diagnostic pass is not a model smoke pass. Enable the model adapter
only after GPU validation, then require all existing smoke correctness and
precision checks before starting the full C32 run.

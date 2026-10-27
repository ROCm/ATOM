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

Run 40487190671 attempt 2 failed during Store local-buffer registration on g12
(`EINVAL`, followed by `ENOMEM`), before any CPU pool was mounted. The retry
keeps the 128 GiB segment and 4 GiB private buffer per rank. Process-local glibc
THP hints and malloc perturbation fault allocated pages in before registration;
Mooncake registers NIC MRs sequentially with a 64 GiB MR cap. P becomes ready
before D starts, limiting concurrent pool registration to four engine ranks.
These are startup mitigations; CI must establish whether they resolve the fault.
No host-wide page-cache or hugepage settings are changed.

Before each model load, a disposable native Store client with HIP initialized
allocates the exact per-rank segment and private buffer and verifies a 1 MiB
PUT/GET. It closes before engines start. Its result, memory-lock limit and
allocation settings are saved in `store-preflight.json`; failure stops the
phase. This single-client check does not certify the full eight-rank pool or
GPU KV transfers, which still require the existing PD smoke gates.

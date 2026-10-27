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

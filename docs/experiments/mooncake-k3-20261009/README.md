# K3 + DSpark3: ROCm Mooncake direct PD + CPU Store C16

Prepared 2026-10-09 from conversation `01a0fa86-ce65-7343-b850-a07a2d9d6b9c`.

- vLLM branch `whx/k3-mooncake-rocm-c16-20261009`, pure upstream main `bacbbe187885db62859f5a4a1443f8ec3006e987`; no integration patch.
- Native build from the pinned vLLM source in the existing pinned ROCm image. This is not an image Python overlay.
- Harness branch of the same name at `/app/k3-mooncake-rocm-c16-20261009`, based on `eee56ddc985d74c8ddfd0eb6c6cd949415699ce5`.
- Mooncake main `b140c1a904d53d24ed211d01ce9a87fc161cb432`; official ROCm cp312 artifact `11593557642`, wheel SHA256 `7dfe1f9acec16843868dde28dcb1d3903aafeaf8b92a657239055581a8c44df6`. HIP/multi-protocol/installed RECORD checks run before serving.
- P8/D8, TP8/DCP8, Kimi-K3 + DSpark3, FP8 KV, mamba align, Model Runner V2, FULL_AND_PIECEWISE. Direct RDMA sender uses one thread per rank, retaining C16 clients.
- P: MultiConnector(MooncakeConnector producer, MooncakeStoreConnector both); D: MooncakeConnector consumer. No LMCache. Embedded CPU Store: 224.875 GiB/rank × 8 = 1799 GiB, plus 1 GiB/rank local buffer; no SSD offload. Store is owned by this job only.
- Official Mooncake example proxy with harness health/models adapter. Its rank-0 process and master are scoped to the serving container.
- Same account/QoS `amd-frameworks`/`amd-frameworks-qos`, runner `atomesh-cicd`, partition `amd-spur`; g12/g13 selected from the existing pool based on inspection `37952706738` (15:35 UTC). No cancellation of other jobs or cleanup of other cases.

## Execution and interpretation

1. Native install + package checks + P/D readiness.
2. Exact 7449/98305-token cold, GPU-reset, repeated-request probes. Require actual successful GPU reset and positive P Store load_get bytes plus external hit tokens in at least one probe before spending the C16 window. Both probe outcomes are preserved, including partial-tail misses. One-token outputs only check basic repeat consistency, not model accuracy.
3. Historical agentic replay: inferencex-agentx-mvp / semianalysis_cc_traces_weka_062126, 393 entries, seed 42, 1M context, C16, 10 warmup requests/lane, 3600-second measurement. P standard; D synthetic acceptance [1,1,0] (AL3) for the historical performance protocol.
4. Fresh P/D phase: full 1319-sample GSM8K, 5-shot, C64, max_gen_toks4096, natural DSpark3, threshold0.94.
5. Preserve Actions, Slurm, workload, acceptance and cleanup as separate states. No throughput/accuracy claims before completed raw evidence.

This is a current-main feasibility/performance exploration, not a controlled transport-only A/B: main/model runner, router, loader, nodes and cache implementation differ from the old overlay run. AIPerf is a trace replay, not an agent executing live tools. Report official observation denominator and request errors.

`contract.json`, `inputs.json`, `matrix.json`, both connector JSON files, and `dry-run.log` specify the reviewable experiment. `preflight.json` records successful matrix and shell checks. Harness checks: 8 matrix tests and 4 behavioral probe tests passed; Ruff and shell/YAML syntax passed. Existing vLLM Mooncake tests: 101 passed; 12 worker tests could not initialize the CPU-only accelerator fixture (`torch.accelerator.current_device_index`), not a ROCm runtime result. The full raw log is retained.

Primary-source research and the unmerged DSpark Store/DCP tail risks are in `research.md`. The passing ROCm wheel build does not itself prove K3 + DSpark + C16 works.

## Resume after the infrastructure failure

Run [37906592218](https://github.com/ROCm/ATOM/actions/runs/37906592218), Slurm6095, failed before model startup. Both GPU preflights passed; rank1 aborted when the peer-state reader raised NFS `ESTALE`, and rank0 was stopped during native build setup. Actions failed, Slurm was cancelled after the rank failure, and the workload failed. Both cleanup queries returned zero with empty task-container lists. There is no transfer, Store-reuse, performance, or accuracy result.

The retry adds four bounded read attempts only for `ESTALE` (0.1/0.2/0.4-second delays). Persistent read faults, invalid records, run ownership checks, and real peer failures retain their original exit semantics. CPU fault injection reproduced exit2 before this change and passed after it; cleanup-state, job-result, Store-probe, and matrix suites total 23 passing tests. This validates error handling, not the availability of the remote NFS service.

The retry uses g12/g13 because g10 is now occupied. Local matrix comparison differs only in the node pair. vLLM source, Mooncake wheel, model parameters, account, QoS, GPU count, Store gate, C16 performance window, and natural GSM8K phase are unchanged. Evidence is under `/app/test_scripts/dspark_pd/mooncake-rocm-c16-20261009/retry-estale/`; the original failure is archived separately.

## Dependency alignment after run37953102503

The source built successfully on both ranks, the official ROCm wheel passed HIP/multi-protocol checks, and the Mooncake master started. Both P/D then failed importing the direct connector: the image has huggingface_hub1.30.0 but pinned main requires1.31.0 for its httpx re-export. Both rank cleanup queries succeeded with empty task-container lists. No model requests were run.

A local isolated1.30.0 installation reproduces the exact import failure;1.31.0 passes. Source requirement audit also found xgrammar0.2.6 instead of0.2.8 and missing oss-harmony; xgrammar0.2.8 needs apache-tvm-ffi0.1.11. The pinned runtime requirements include these versions and restore image setuptools79.0.1 after native build tooling installs84. The active dependency imports and declared common/transitive requirements are checked locally. Remote startup now checks common requirements and imports both Mooncake connectors before model launch. Package manifests resolve the active distribution so image metadata cannot overwrite venv provenance.

The next run uses idle g10/g12 per CPU inspection38021019640 at03:34:57 UTC, with identical model, transport, Store, performance and accuracy configuration. The vLLM source remains pure bacbbe187. Current contract and inputs are in this directory; detailed local evidence is in `/app/test_scripts/dspark_pd/mooncake-rocm-c16-20261009/retry-hub131/`.

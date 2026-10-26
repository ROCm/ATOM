# MiniMax M3.1 Flash TP4 P/D with Mooncake Store and DSpark

The dispatch-only `mooncake_store` suite runs AgentX at C32, C48 and C64. The
`profile` suite runs the same C32 configuration and captures all prefill and
decode TP ranks for 180 seconds, starting 1800 seconds after AIPerf enters its
formal Profiling phase. Warmup does not count toward that delay.

| Setting | Value |
| --- | --- |
| Model | `amd/MiniMax-M3.1-Flash-Quark-MXFP4` |
| Hardware | One TW MI355X node, eight GPUs |
| Prefill / decode | TP4 GPUs 0–3 / TP4 GPUs 4–7 |
| Transport | Mooncake Transfer, RDMA, `MC_DISABLE_HIP=1`, one QP per endpoint |
| Prefix offload | Native Mooncake Store on prefill; two 768 GiB owners |
| Store owner placement | NUMA0 on rdma4/5; NUMA1 on rdma6/7 |
| Target / index / draft KV | FP8; block size 128 |
| DSpark | Checkpoint `dspark/`, 3 draft tokens, synthetic mean acceptance length 2.88 |
| Limits | 1,048,576 context tokens; 32,768 tokens per batch; 128 sequences |
| GPU memory utilization | 0.9 |
| Prefix caching | Enabled |
| Workload | `inferencex-agentx-mvp`, `semianalysis_cc_traces_weka_062126`, 393 entries |
| AIPerf | `b7b16cf851885567988a643282266bce74e34437`; seed 42 |
| Warmup | 10 requests per lane; grace period 1800 seconds |
| Formal measurement | 3600 seconds per unprofiled cell |
| Trace | CPU/GPU activity; shapes, stacks and memory recording disabled |

The DSpark K=3 and AL=2.88 match the SA MI355X vLLM high-concurrency run:
[run 39929767608](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/39929767608/attempts/1).
The acceptance value is from the
[MiniMax M3.1 golden AL table](https://github.com/SemiAnalysisAI/InferenceX/blob/65a6dd1978720f65099fea8c4d5ac5eddfdfd8d7/inferencex-e2e/infx/golden_al_distribution/minimaxm31flash_dspark.yaml).
It is a performance control, not measured acceptance or an accuracy result.
ATOM uses its native DSpark implementation with FP8 KV/index cache and no TBO.
SA's C32/C64 vLLM configuration uses NVFP4 target/index caches and DBO; it is
aggregated TP4 without Store. These P/D measurements are not an exact replay of
that serving configuration. Checkpoint-native MXFP8 attention/shared-expert
weights are retained; online quantization and shared-expert fusion are off.

The existing `--gqa-prefill-kv-workspace-mb 8192` bound is retained. It applies
only to eligible dense gather implementations; this target has all sparse
layers, and the DSpark block reads paged KV directly. A configured bound is not
proof that an 8 GiB reservation or grouped gather ran; inspect the memory logs.

Dispatch `atomesh-benchmark.yaml` twice from the test branch, with
`atom_source=mounted`, `atomesh_slurm_submit_runner=atomesh-cicd`, automatic node
selection, and `publish_dashboard=false`:

- `suite=profile`, case `minimax-m31-flash-mxfp4-1p1d-tp4-dspark3-mcstore-agentx-c32-profile`.
- `suite=mooncake_store`, cases `minimax-m31-flash-mxfp4-1p1d-tp4-dspark3-mcstore-agentx-c32`,
  `minimax-m31-flash-mxfp4-1p1d-tp4-dspark3-mcstore-agentx-c48`, and
  `minimax-m31-flash-mxfp4-1p1d-tp4-dspark3-mcstore-agentx-c64`.

Use the validated image for both actions:
`rocm/atom-dev:nightly_202610251529@sha256:9194234e34b71463d80c23c81d3f98d46ac3a60b9e0b621101262b7f8b71e774`.
The Store requires the image's `mooncake_master`, `mooncake_client`, and
`mooncake.store`; the launcher checks these before allocating Store owners.

The model job prints `log_root=...` and `slurm_job_id=...`. Trace files remain
in the shared directory even after the containers exit:

```text
<log_root>/slurm_job-<id>/torch_traces/prefill-rank-0/rank_<0..3>/*.pt.trace.json.gz
<log_root>/slurm_job-<id>/torch_traces/decode-rank-0/rank_<0..3>/*.pt.trace.json.gz
<log_root>/slurm_job-<id>/benchmark_results/aiperf-*/torch_profile.log
```

On the current TW submit host, `log_root` is under
`/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG/`. Other submit hosts may use
`/it-share/ATOMESH_LOG/`; use the job's printed path. Trace export has a 3600-second
rank timeout. Once all traces are complete, the profiling cell stops its load
generator; its throughput is not a benchmark result. Missing traces or an
incomplete 180-second window fail the cell. The unprofiled cells have
`ATOMESH_TORCH_PROFILE_DURATION=0` and run the full measurement.

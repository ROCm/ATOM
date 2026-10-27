# MiniMax-M3.1-Flash: TP4 P/D with the AgentX recipe

This test configuration adapts [recipe PR #2747](https://github.com/ROCm/ATOM/pull/2747)
at `7742552f26e6fe1545202a6cb78f4839ec0563cb` to one prefill TP4 worker and one
decode TP4 worker on eight MI355X GPUs. It uses the recipe's high-concurrency
NVFP4 band. The published recipe measures an aggregated TP4 server through C48;
C64 here extends its `max-num-seqs = 2 * concurrency` rule.

The configuration is in `.github/benchmark/models_atomesh.yaml` under
`MiniMax-M3.1-Flash-Quark-MXFP4`.

| Setting | This P/D configuration |
| --- | --- |
| Checkpoint | `amd/MiniMax-M3.1-Flash-Quark-MXFP4` |
| Served name | `MiniMaxAI/MiniMax-M3.1-Flash` |
| Target sparse KV / index | NVFP4 / FP4 |
| Draft KV | FP8 (`--kv_cache_dtype fp8` remains set) |
| Attention | Vendored SGLang gfx950 NVFP4 sparse attention |
| Target quantization | Online `ptpc_fp8`; recipe exclusions including `*block_sparse_moe` |
| Draft quantization | `ATOM_M3_DSPARK_DRAFT_PTPC=1` |
| Shared expert | `ATOM_FUSE_MXFP8_SHARED_INTO_MXFP4=1` |
| Mono switch | `ATOM_MONO_ENABLE=1`; the NVFP4 target is expected to refuse Mono |
| Fused verify top-k | Enabled; applies to eligible decode calls with at least 3 requests |
| Indexer CP | Unset, as required by FP4 index |
| DSpark | K=7, checkpoint `dspark/` |
| Performance acceptance | Synthetic golden AL 4.04; omitted for accuracy |
| C32 / C48 / C64 max-num-seqs | 64 / 96 / 128 on both roles |
| Capture sizes | Recipe list, truncated at the respective max-num-seqs |
| Context / batch tokens | 1,048,576 / 32,768 |
| Block size / GPU utilization | 128 / 0.9 |
| Prefix cache | Enabled |
| Client | PyPI `agentx-harness==1.0.12`, `agentx profile` defaults |
| Dataset | `semianalysis_cc_traces_weka_062126` |
| Measurement | 3600 seconds for each unprofiled concurrency |
| Accuracy | GSM8K, 5-shot, chat template, full dataset, real acceptance |
| Image | `rocm/atom-dev:nightly_202610261558` |

The old `ATOM_PA_FLYDSL=1` override is removed. FlyDSL PA accepts FP8 KV;
NVFP4 target attention uses the vendored Gluon implementation. This does not
establish that the earlier AITER Gluon GPU fault is fixed: it is a different
kernel and cache layout, and needs a new GPU run.

## P/D-specific settings

- Prefill uses GPUs 0–3; decode uses GPUs 4–7. Both are TP4.
- P→D uses Mooncake Transfer over RDMA, with the existing role/handshake settings.
- Benchmark and profiling use native Mooncake Store on prefill: two 512 GiB
  owners, totalling 1 TiB, retaining the existing NUMA/NIC placement.
- Client requests go through the P/D router. Metrics are scraped from both P
  and D, with `AIPERF_REQUIRED_SERVER_METRIC_PREFIX=atom:` and TCP timeout 900000 ms.
- The profiling case waits 1800 seconds after formal Profiling begins, records
  30 seconds on both roles/all ranks, and stops the load after trace export.
  Trace files remain under the job's shared `torch_traces` directory.
- The former 8 GiB dense-gather workspace flag is omitted to match the recipe;
  it did not reserve 8 GiB for this all-sparse target.

Report P/D throughput per GPU against all eight allocated GPUs. The recipe's
published throughput/GPU divides by four; its results are not measurements of
this P/D setup. The trace case's intentionally stopped load is not a full
3600-second performance result.

## Runtime dependencies brought into the test branch

These were open PRs when the recipe was inspected on 2026-10-27:

| PR | Source head | Function |
| --- | --- | --- |
| [#2686](https://github.com/ROCm/ATOM/pull/2686) | `ac8b0fb7c17cc1a36c2e286ec6bab75b244f4ca4` | M3.1 fixed-DSpark Mono support |
| [#2743](https://github.com/ROCm/ATOM/pull/2743) | `9a928dfe503b49f12967533cfb9b16c26a8e9af3` | Draft PTPC FP8 requantization |
| [#2744](https://github.com/ROCm/ATOM/pull/2744) | `094f4e53442bf518eda277a7ad846f5eb36ef1ca` | Vendored SGLang NVFP4 kernels |
| [#2745](https://github.com/ROCm/ATOM/pull/2745) | `079237727c7ce865ec3e06ee5cbb43dc91d7eb3d` | FP4 index cache |
| [#2746](https://github.com/ROCm/ATOM/pull/2746) | `8c9c4b1ab20c885c38ddc9cb9bee8bbd2f346376` | NVFP4 KV and fused verify top-k |

The environment definitions from #2743 and #2746 are combined. The existing
complete-PAGE transfer/Store codec includes the additional index-scale plane.
The Store namespace also fingerprints sparse NVFP4 and draft PTPC so a server
cannot reuse chunks produced under the old cache/quantization configuration.

## CI inputs

Use `atomesh-benchmark.yaml` from the branch containing these changes, with
`atom_source=mounted`, `atomesh_slurm_submit_runner=atomesh-cicd`,
`model_names=MiniMax-M3.1-Flash-Quark-MXFP4` and `publish_dashboard=false`.
Set `atomesh_image=rocm/atom-dev:nightly_202610261558` for an explicit recipe pin.
The image resolver also selects this model's pin when the model or its cases
are selected without an explicit image override.

- `suite=profile`: the C32 30-second trace case.
- `suite=mooncake_store`: C32/C48/C64, without tracing.
- `suite=accuracy`: the C32 GSM8K 5-shot case, without synthetic acceptance.

Case names now contain `dspark7-nvfp4`; the previous `dspark3` case names no
longer select these tests. The accuracy case also changes `20shot` to `5shot`.

The checks below validate the configuration and CPU-side contracts. GPU
accuracy, benchmark and trace results must be verified from their CI jobs.

## Local validation (2026-10-27)

- 295 CPU tests passed for Mooncake Store, PAGE codec, MHA pool and draft KV.
  This includes NVFP4/FP4 region-copy checks and namespace separation.
- 148 ATOMesh config/launcher tests passed. The Slurm simulations used an empty
  `amd-smi` fixture to avoid this host's unrelated GPU load; their busy-device
  rejection test supplies its own device fixture.
- All five generated cells passed recipe parameter checks and Slurm dry-run.
- Workflow image selection passed model-filter, case-filter, explicit-override
  and other-model fallback checks. Python syntax, Ruff, Black and Bash syntax
  checks passed.
- The AITER-dependent Mono dispatch and index top-k test modules were skipped
  in this CPU environment. No real GPU inference or RDMA transfer was run.

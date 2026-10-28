# MiniMax-M3.1-Flash: TP4 P/D with the AgentX recipe

This test configuration adapts [recipe PR #2747](https://github.com/ROCm/ATOM/pull/2747)
at `7742552f26e6fe1545202a6cb78f4839ec0563cb` to one prefill TP4 worker and one
decode TP4 worker on eight MI355X GPUs. It uses the recipe's high-concurrency
NVFP4 band. The published recipe measures an aggregated TP4 server through C48;
C64 and C88 here extend its `max-num-seqs = 2 * concurrency` rule.

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
| C32 / C48 / C64 / C88 max-num-seqs | 64 / 96 / 128 / 176 on both roles |
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
- The dense cached-prefill workspace feature has been reverted from this
  integration branch. Target sparse attention and DSpark read paged KV directly;
  this recipe does not use a dense-gather workspace budget.

Report P/D throughput per GPU against all eight allocated GPUs. The recipe's
published throughput/GPU divides by four; its results are not measurements of
this P/D setup. The trace case's intentionally stopped load is not a full
3600-second performance result.

## C32 chunked-transfer comparison

Rebase onto a `main` containing [#2425](https://github.com/ROCm/ATOM/pull/2425)
(`8a9253daa`) before using the chunked case. Both the ATOM workers and the
ATOMesh router must include that protocol. With `atom_source=mounted`, this
workflow builds the router from the checked-out branch as well.

Select these two cases explicitly under `suite=mooncake_store`:

- Baseline: `minimax-m31-flash-mxfp4-1p1d-tp4-dspark7-nvfp4-mcstore-agentx-c32`
- Chunked: `minimax-m31-flash-mxfp4-1p1d-tp4-dspark7-nvfp4-mcstore-agentx-chunked-c32`

The second case only adds `"enable_chunked_transfer":true` to prefill's
Mooncake P/D connector configuration. C32, TP4/TP4, NVFP4/FP4 index, draft FP8,
DSpark K7/AL4.04, 1 TiB Store, capture sizes, and the 32768-token prefill batch
budget remain identical. Each case runs 3600 seconds of formal profiling with
no torch trace. Run both on the same branch revision, image and node. The
launcher retains `agentx-harness==1.0.12` and `agentx_recipe_defaults: true` so
the rebase's upstream AgentX installer does not change this workload's CLI.

Chunked transfer sends newly completed pages after each prefill forward's GPU
completion event; it includes target KV, scales, index and draft regions. Decode
reserves destinations early, skips its cached prefix, and waits for all required
KV plus the final first-token/draft handoff before generating. It does not
pipeline Store-to-prefill loading or permit decode to read an incomplete prefix.
A prefill that completes in one compute chunk has limited opportunity for
compute/transfer overlap. Keep the batch budget fixed in this first comparison;
a later smaller-budget experiment needs its own matched off/on baseline.

Compare client/route-level TTFT, request/output throughput, cache hit rates,
errors and memory pressure. Local decode queue, TTFT and KV-wait metrics start
earlier with parallel P/D dispatch and may now include prefill work, so their
raw off/on differences do not measure transfer savings. Earlier reservation
also holds decode KV for longer. If tracing a later experiment, inspect the gap
between the last prefill forward and decode KV readiness separately.

### C32 chunked transfer with an 8192-token budget

To run only the smaller-budget experiment, select
`minimax-m31-flash-mxfp4-1p1d-tp4-dspark7-nvfp4-mcstore-agentx-chunked-mbt8192-c32`
under `suite=mooncake_store`. It inherits the chunked C32 case and sets
`max_num_batched_tokens: 8192` in the shared server arguments for both roles.
C32, TP4/TP4, DSpark K7/AL4.04, 1 TiB Store, graphs, AgentX 1.0.12, pinned
image, and 3600-second formal profiling without torch traces are unchanged.
This selection runs one case. The original off/on cases retain their 32768-token
budget; comparing against them changes the budget as well.

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
- `suite=mooncake_store`: C32/C48/C64/C88, each with 3600 seconds of formal
  profiling after warmup and without torch tracing. Each case starts its own
  P/D service. Use `case_names` to select the sweep or the C32 off/on pair;
  selecting the entire suite also includes the chunked C32 case.
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

### After removing dense-prefill workspace budgeting

- Reverted `2142d542c` in this integration branch; the standalone workspace PR
  remains independent of the M3.1 recipe.
- 338 runtime/metadata/Store tests passed; 59 GPU-stack-dependent tests skipped
  in the CPU environment. All 148 ATOMesh config/launcher tests passed.
- C32/C48/C64/C88 each passed expanded-matrix parameter checks and Slurm dry-run:
  P/D TP4, NVFP4/FP4 index, DSpark K7/AL4.04, 1 TiB Store, no torch trace, and
  3600-second formal profiling. C88 uses max-num-seqs 176 on both roles.
- Ruff, Black, Python AST and diff whitespace checks passed for the reverted files.

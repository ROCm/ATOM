# MiniMax-M3.1: allocate index cache only on computing layers

This branch implements [LLM-Booster #316](https://github.com/wuhuikx/LLM-Booster/issues/316)
on [ATOM #2750](https://github.com/ROCm/ATOM/pull/2750), pinned to
`12107a26089177a4cfd185a4305e5e89b65ad4e3`.

The model's `_should_skip_minimax_m3_index_topk` decision is already stored on
each sparse attention implementation. The pool now assigns index rows only to
implementations with `skip_index_topk=False`. Allocation, binding, P/D PAGE
exports and prefix-cache PAGE views use that same compact row assignment.
Reusing layers still own target K/V and share the per-forward top-k memo.
Explicit patterns, offsets and stage-local model walks retain the model's
layer-selection semantics. This change does not add cross-PP-stage top-k reuse.

Compact index-row allocation is always active and has no separate opt-in flag.
The existing `use_index_cache` / `index_topk_freq` HF overrides choose which
layers reuse top-k; they do not enable the allocation optimization. Every layer
that computes top-k gets an index row, and every reusing layer omits one.

The evaluated M3.1 checkpoint (`0a00e0e39fc00499b23efccc18a8ea63fce5c44c`)
does not declare `index_share` or `indexer_types` shared-layer markers. Its
weight manifest contains index Q/K projections and norms for all 60 layers.
Here, freq4 is an optional inference reuse schedule, with accuracy to be
validated; compact allocation follows that schedule without changing it.

The base recipe rejected top-k reuse with NVFP4 KV. Reusing layers now call
AITER's existing `skip_index_branch` path with no index output/cache, then run
the same NVFP4 target K/V store as computing layers. Both modes use the same
fused Q/K normalization and RoPE implementation. No persistent dummy index
cache or per-layer scratch cache is allocated for skipped layers.

M3 PAGE identity now fingerprints the checkpoint's index schedule, including
explicit patterns and offsets, and the compact layout version. Mooncake Store
uses a new M3 codec identity so objects from the old full-row layout cannot be
loaded into compact pools.

## Capacity accounting

With 60 sparse layers and `use_index_cache=true,index_topk_freq=4`, index rows
fall from 60 to 15. Startup logs report `15/60 sparse layers own rows`; the
baseline reports `60/60`. Target K/V and draft allocations are retained.

| Index storage | Before, bytes/token/rank | After | Saved |
| --- | ---: | ---: | ---: |
| FP8, 128 dimensions | 7,680 | 1,920 | 5,760 |
| FP4 payload + scales, 128 dimensions | 4,320 | 1,080 | 3,240 |

The issue's approximate 7.84M → 10.2M aggregate token capacity uses its FP8
accounting. #2750 uses NVFP4 target KV, FP4 index and FP8 draft KV, so compare
its actual startup token capacity at the same GPU utilization. Do not impose
the old absolute capacity numbers on this different storage recipe. TP4 cache
positions are shared across the four ranks; report per-rank token capacity and,
when quoting the issue's convention, four times that capacity explicitly.

## CI cases

All cases inherit #2750's single-node TP4 prefill (GPUs 0–3) / TP4 decode
(GPUs 4–7), image `rocm/atom-dev:nightly_202610261558`, DSpark7, target NVFP4,
FP4 index, FP8 draft, graphs, 32,768 batch tokens and 1M context limit.

`freq1` explicitly sets `use_index_cache=false,index_topk_freq=1`.
`freq4` sets `use_index_cache=true,index_topk_freq=4`. Each matched pair changes
only these HF overrides and its case identity.

| Suite | Case suffix after `minimax-m31-flash-mxfp4-tp4-` | Workload |
| --- | --- | --- |
| accuracy | `index-freq1-gsm8k-c32`, `index-freq4-gsm8k-c32` | Full GSM8K, 5-shot, max 16,384 generated tokens |
| accuracy | `index-freq1-aime24-c32`, `index-freq4-aime24-c32` | Full AIME24, 0-shot, max 32,768 generated tokens |
| mooncake_store | `index-freq1-mcstore-agentx-c32`, `index-freq4-mcstore-agentx-c32` | C32, AgentX 1.0.12, 3600 seconds after warmup |

Accuracy uses real DSpark acceptance, chat templates and concurrency 32.
Performance retains synthetic AL 4.04, 1 TiB Store (two 512 GiB owners), RDMA
P→D, `semianalysis_cc_traces_weka_062126`, and no torch profiler. Report
throughput/GPU against all eight physical GPUs. Synthetic acceptance cannot
establish accuracy or real acceptance performance.

Both performance cases enable `env.common.ATOMESH_DROP_HOST_CACHES=1`.
After image/mesh preparation and before launching the service container, each
node runs a bounded root Docker helper to `sync` and write `3` to the host's
`/proc/sys/vm/drop_caches` (clean page cache and reclaimable slab). Host and NUMA
memory snapshots are saved in `rank-*/host-memory-<phase>.log`. Cleanup failure
or timeout stops startup; the helper is removed on exit. This is host-wide cache
eviction, so enable it only for benchmark allocations where that is intended.
The existing model-file cache eviction and per-NUMA huge-page compaction still
run before Store owners. The 1 TiB Store allocation is unchanged. Accuracy
cases do not opt into this additional cleanup.

Dispatch `.github/workflows/atomesh-benchmark.yaml` on the new branch with
`atom_source=mounted`, `atomesh_slurm_submit_runner=atomesh-cicd`,
`atomesh_image=rocm/atom-dev:nightly_202610261558`, the appropriate `suite` and
comma-separated `case_names`, and `publish_dashboard=false`. Pin the performance
pair to the same node for comparison. The workflow builds its matching router.

The GPU gate needs paired GSM8K/AIME scores and sample-level comparisons,
startup row/token-capacity logs, and valid completed C32 exports with errors,
TTFT, TPOT, output throughput and total throughput. Local CPU tests establish
layout/launch contracts; they do not establish GPU accuracy or performance.

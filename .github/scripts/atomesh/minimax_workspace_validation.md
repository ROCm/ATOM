# MiniMax-M3 paged cached-prefill validation

Run the two cases in `suite=workspace_validation` against this mounted checkout.
The branch contains the default paged cached-prefill implementation and the cached-KV int64
index fix from ATOM PR #2565. Aiter is pinned to
`2c954755467419d466b55dd15659280420495fe1` (branch `mengqing/paged-prefill-kv`, includes the PR #6519 bounds fix
and paged scale-cache support).
A small GPU preflight checks paged K/V and scale reads against a dense reference
before model loading, including both 2-D and split-K attention, shared pages,
and invalid tail slots. There are no runtime source overlays or `minimax_bounded_gather` hooks.

Both jobs use 1P1D TP4+TP4, EAGLE3 K=3, FP8 KV/index cache, and LMCache
256 GiB per prefill rank. The shared KV pool uses native allocations for HIP
IPC; expandable segments remain enabled for inference allocations on prefill.

- `minimax-m3-fp4-1p1d-tp4-eagle3-gsm8k-20shot-full-c32` runs all 1,319 GSM8K test
  questions, 20-shot, concurrency 32. It uses the model's chat template,
  fewshot-as-multiturn and thinking mode, with a 16,384 output token limit.
  It emits the measured flexible-extract exact-match score and checks sample
  coverage; no prior matching 20-shot score is assumed as a pass threshold.
- `minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32` retains the original
  AgentX dataset, 32 lanes, 10 warmup requests per lane and 3,600 profiling
  seconds. Both cases use the same default paged backend, with no workspace
  or backend switches. The original baseline is run
  `38841844766`; the new branch also contains newer mainline code, so the
  performance difference cannot be attributed solely to removing the hook.

Both retain the eight-hour allocation limit. Host memory preparation runs on
the exclusively allocated node before containers start, with no sudo: up to
three attempts within 900 seconds (120 seconds per step; base-page
release gets 30 additional seconds). It reclaims clean file cache and prepares
THP; unused HugeTLB is released only when permitted and safely restorable.
The previous C32 job spent most of its 300-second budget scanning NFS model
cache and reclaiming base pages, then hit its deadline during healthy THP
progress. The longer budget preserves all capacity/progress checks.
Cleanup restores journaled HugeTLB counts. Previously incompatible nodes g34,
g12 and g53 are excluded; placement otherwise remains automatic.

Use image `rocm/atom-dev:nightly_202610141644`, the successful baseline image.
Its Mesh source tree matches this branch (`55ad15ac25935554c211538e0611bbca26bc567c`),
so reuse its binary and avoid the shared Cargo cache. The ATOM Python package
must use `atom_source=mounted` to test this branch.

Dispatch each case separately with `gh workflow run atomesh-benchmark.yaml`,
`--ref mengqing/draft-prefill-kv-workspace`, `suite=workspace_validation`, the
case name, `atomesh_slurm_partition=amd-spur`, and `publish_dashboard=false`.
Leave Slurm account and node selections at `auto`.

Logs contain `[prefill-paged]` for both target and draft cached prefill. Check
this evidence alongside the GSM8K score and C32 result. If Actions cannot serve
live job logs, dispatch the same workflow with `suite=smoke`,
`run_model_benchmark=false`, `publish_dashboard=false` and
`collect_minimax_run=<run ID>` for a read-only shared-log snapshot without GPU
allocation. Never infer benchmark success from the snapshot job's status.

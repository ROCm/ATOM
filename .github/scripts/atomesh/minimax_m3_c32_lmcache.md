# MiniMax-M3 C32 AgentX benchmark with LMCache memory preparation

Case: `minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32`.
This runs the original AgentX dataset, 32 concurrent lanes, 10 warmup requests per
lane and a 3600-second benchmark. The original 8-hour Slurm limit includes setup,
model loading, warmup and the benchmark. There are no synthetic replay requests,
admission gates or replay instrumentation.

```sh
gh workflow run atomesh-benchmark.yaml --repo ROCm/ATOM \
  --ref mengqing/moe_scatter_reduce_bounds_ci \
  -f suite=nightly \
  -f case_names=minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32 \
  -f atomesh_image=rocm/atom-dev:nightly_202610141644 \
  -f atomesh_slurm_account=amd-atom \
  -f atomesh_slurm_partition=amd-spur \
  -f run_model_benchmark=true -f publish_dashboard=false
```

Only this C32 case enables `benchmark.prepare_lmcache_host_memory`. In
`start_prefill`, after applying the prefill environment and before launching
either model service, the helper directly tries `sync`, `drop_caches=3` and
`compact_memory=1` as the existing CI user. It uses no sudo and changes no
container privileges. If the VM controls are unavailable, it runs the existing
file-cache eviction and NUMA-bound huge-page preparation helpers.

Preparation permits three serial attempts within 300 seconds, at most 90 seconds
per command, plus up to five seconds to reap a timed-out child. A child that cannot
be reaped blocks model startup. The longer step allowance accommodates preparing
1 TiB before the services start. Pool size comes from the prefill GPU list and
LMCache capacity: GPUs 0-3, 256 GiB/rank, currently 1024 GiB on NUMA0.

The pool itself must fit in NUMA free memory; the additional 128 GiB reserve may
include clean active/inactive file cache, excluding dirty/writeback pages. NUMA
total capacity and visible cgroup headroom are checked as well. Unknown cgroup
capacity, persistent insufficient memory, permission errors in the NUMA helper,
or incomplete preparation fail before starting services. File-cache eviction is
advisory. Successful preparation does not establish subsequent pinning latency
and does not release a live LMCache pool; its owning processes must exit first.

Run `38736095232` stopped at the former free-memory check: NUMA0 had about
1056 GiB free and 414 GiB file cache, but the helper required all 1152 GiB to be
free. The direct VM write was read-only, three file-cache passes did not change
this state, and no NUMA compaction or model startup occurred. The updated reserve
check permits compaction in that state without requiring the pool itself to
reclaim file cache.

Artifacts under `benchmark_results/lmcache-host-memory/` retain `summary.json`,
command logs and before/after NUMA, cgroup, buddyinfo, vmstat and THP snapshots.

This case sets `benchmark.mesh_source: image` to avoid the damaged shared Cargo
cache. The pinned image has Mesh source tree
`55ad15ac25935554c211538e0611bbca26bc567c`, identical to this branch; the existing
`/app/ATOM/atom/mesh/target/release/atomesh` binary serves the benchmark. Recheck
compatibility if the image or Mesh source changes.

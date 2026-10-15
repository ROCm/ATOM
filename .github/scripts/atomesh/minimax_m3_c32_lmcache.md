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

Only this C32 case enables `benchmark.prepare_lmcache_host_memory`. The Spur
worker executes preparation on the **allocated host**, after the existing
container cleanup and before launching the service container. It uses the CI
user directly, with no sudo or extra container privileges. The server only
accepts the launch after `ATOMESH_LMCACHE_HOST_PREPARED=true` is set by a
successful host preparation.

`lmcache_hugetlb.py` first checks the prefill NUMA nodes. Where ordinary free
memory is below the LMCache pool plus 128 GiB reserve, it can temporarily shrink
the node's unused 2 MiB HugeTLB pool, only by the calculated shortfall. It requires
the node's pool to be entirely free, no surplus pages and no global 2 MiB page
reservations. Used or reserved pools are not modified. Pool size and the increase
in ordinary free memory are checked after writing. Permission failures fail
before model startup; they cannot be fixed by retries inside a read-only container.

The original counts are journaled before each write. After model containers have
stopped, normal completion, failures and catchable cancellation attempt to restore
the counts. Writes have a 60-second deadline plus up to five seconds to reap the
child. Failed or incomplete restoration fails the job and preserves the journal
for node recovery. A writer that might still be alive blocks automatic recovery.
SIGKILL/node loss can prevent cleanup; recorded counts may require administrator
recovery. The code does not claim to restore node state in those cases.

The host then runs `prepare_lmcache_host_memory.py`: direct sync/drop_caches/
compact_memory where allowed, otherwise file-cache eviction and NUMA-bound THP
preparation. It permits three serial attempts within 300 seconds, at most 90
seconds per command, plus up to five seconds to reap a timed-out child. The pool
itself must fit in ordinary NUMA free memory; the extra reserve may include clean
active/inactive file cache, excluding dirty/writeback pages. Visible cgroup
capacity is also checked. This remains one node, prefill GPUs 0-3 at 256 GiB/rank,
currently 1024 GiB on NUMA0.

Run `38738296856` failed before model startup: NUMA0 had 470.49 GiB ordinary free
memory, 268.30 GiB file cache and 720 GiB idle HugeTLB reservation. The former
container-side script could not write the read-only VM controls and did not
release HugeTLB pages. Three file-cache passes took 1.94 seconds and no THP
compaction ran. `drop_caches` and THP compaction do not shrink HugeTLB pools.

Artifacts under `benchmark_results/lmcache-host-memory/` retain the preparation
`summary.json`, command logs and before/after NUMA, cgroup, buddyinfo, vmstat and
THP snapshots. `host/hugetlb.json` records original counts, targets, observed
results and restoration status, with per-node release/restore logs. File-cache
eviction is advisory; successful preparation still does not prove subsequent
pinning latency. Existing live LMCache arenas must be released by their owners.

This case sets `benchmark.mesh_source: image` to avoid the damaged shared Cargo
cache. The pinned image has Mesh source tree
`55ad15ac25935554c211538e0611bbca26bc567c`, identical to this branch; the existing
`/app/ATOM/atom/mesh/target/release/atomesh` binary serves the benchmark. Recheck
compatibility if the image or Mesh source changes.

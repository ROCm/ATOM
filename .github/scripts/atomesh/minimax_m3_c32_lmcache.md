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
  -f atomesh_slurm_partition=amd-spur \
  -f run_model_benchmark=true -f publish_dashboard=false
```

Slurm account and single-node placement use their defaults (`auto`). The C32
case excludes `pit2-p03-g34`: run `38741996691` confirmed that UID 300152 cannot
write the node's HugeTLB controls. The scheduler selects among the remaining
nodes. This exclusion does not imply other nodes have sufficient memory or
administrator permissions; host preparation still validates the selected node.

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
seconds per command, plus up to five seconds to reap a timed-out child. Before
the bounded allocation, ordinary free memory plus clean active/inactive file
cache must cover the pool and 128 GiB reserve; dirty/writeback pages are excluded.
This lets the preparation reclaim cache itself. After preparation, the pool
must fit in actual ordinary free memory; when the NUMA helper is used, it must
also report full THP coverage. Visible cgroup capacity is checked. This remains
one node, prefill GPUs 0-3 at 256 GiB/rank, currently 1024 GiB on NUMA0.

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

Run `38745966243` selected g19 successfully, but the former free-only pool check
rejected 769 GiB free plus 701 GiB clean cache before running NUMA preparation.
The bounded reclaim path now handles that state and verifies recovered free
memory afterward; g34's 470 GiB free plus 268 GiB cache still fails capacity checks.


Host preparation enables `drop_page_cache.py --stats`: per-directory scanned,
opened and advised file counts, advised file sizes (not reclaimed bytes),
walk/open/advice errors with up to ten examples, and skipped symlink directories.
The parent records NUMA memory immediately before and after this step, so direct
VM cleanup is not counted as file-advice reclamation.

`compact_huge_pages.py --progress` flushes a `[thp-progress]` record at the start,
after each 32 GiB or ten seconds of first-touch progress, and at the end. The
touched address range is distinct from actual RSS and AnonHugePages; one-byte
faults that fall back to 4 KiB pages do not populate a whole 2 MiB interval.
Progress records never satisfy the full-THP completion check. While the child is
running, the parent also samples its state, RSS and kernel wait channel every ten
seconds, including when the child cannot print while blocked in a page fault.
Per-attempt logs are forwarded into Actions and retained in artifacts. These
options do not change allocation size, retry limits or benchmark settings.

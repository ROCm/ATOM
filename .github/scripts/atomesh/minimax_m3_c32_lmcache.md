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
case excludes `pit2-p03-g34` and `pit2-p03-g12`: runs `38741996691` and
`38793498618` confirmed that UID 300152 cannot
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
seconds for ordinary commands or 120 seconds for base-page population and
release, plus up to five seconds to reap a timed-out child. Before
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

When the running Actions job logs cannot be downloaded, dispatch the same
workflow with `collect_minimax_run=<original-run-id>` and
`run_model_benchmark=false`. The separate TW runner job only queries `squeue`
and snapshots existing C32 logs from shared storage. It does not allocate GPUs,
start models or cancel jobs. It uploads at most 64 MiB, with at most 8 MiB from
the tail of each allowed log, and records original sizes and truncation in
`snapshot.json`. Environment files and paths outside the selected run are not
collected. This snapshot has its own CI run and does not change the original
benchmark's outcome.

## Runtime stall evidence

Run 38752033481 prepared 1024 GiB of THP successfully in 116 seconds and
initialized all four LMCache CPU backends in 59–74 seconds. During AgentX warmup,
request 27's 13.36 GiB save then took about 3400 seconds on every rank; later,
all TP ranks timed out at broadcast sequence 95. Pre-start memory preparation
alone therefore does not establish that the runtime pin/copy path is healthy.

The C32 CI service now uses a temporary copy of the same ATOM package with a
Python stack sampler added at import. It snapshots threads every 120 seconds
using `sys._current_frames()` and normal Python traceback formatting. A separate process samples host/NUMA memory,
compaction/THP counters and registered workers' thread wait channels every
30 seconds. It checks process start times before reading a PID. NCCL's bounded
4096-entry trace buffer dumps on timeout. No model, scheduling, concurrency,
cache capacity, warmup, or performance-duration settings are changed.

During AIPerf, `benchmark_guard.py` fails with exit code 86 on a fatal worker
error, or after 900 seconds with requests in flight but no inference progress.
HTTP metrics/health responses and unchanged AIPerf progress lines do not reset
that timer. Successful request returns, scheduled prefill batches, output sends,
and completed offload saves do reset it. This avoids occupying an allocation
for hours after a worker has died. Artifacts include `benchmark-guard.json` and
`benchmark_results/runtime-diagnostics/`; read-only collection supports both.

Run 38788119513 landed on g08, whose prefill NUMA node had only 58 GiB free
and 1404 GiB of clean file cache. Advice on all 160 target and 6 draft files
succeeded but released essentially no memory: the cache belonged to other
files. Three THP attempts timed out after 90 seconds each, recovering only
84 GiB total, and the model was never started.

When advising the selected model directories still leaves less ordinary free
RAM than the LMCache pool, the host preparation now also advises clean cached
files under the existing `/share_nfs/models` mount. This is confined to the
case's exclusive allocation and changes no files or permissions. It is skipped
when the node already has enough free RAM. Both advice passes and their actual
NUMA memory deltas are recorded separately; the existing 300-second total and
90-second step bounds remain in effect.

Run 38793498618 landed on g12, which reserved 720 GiB of idle HugeTLB on NUMA0.
Ordinary free RAM plus clean cache was only about 760 GiB, and UID 300152 could
not lower `nr_hugepages`. No model-cache advice or model startup ran. The case
now excludes this demonstrated incompatible node as well as g34; scheduling
among all other nodes remains automatic, with no Slurm account specified.

Run 38795325170 prepared 1024 GiB on g03 successfully, but a prefill worker
exited with SIGSEGV at 20:36:23 while loading weights. Its first native timeout
stack dump was due at exactly 20:36:22.981 (120 seconds after registration),
and the dump ended in a partial frame with an invalid line number. This strongly
implicates the CI diagnostic's asynchronous native frame walk. That diagnostic
has been removed, including its fatal-signal and fork hooks. The replacement
uses regular Python thread/frame APIs with strong frame references and bounded,
rotated output. If a native call holds the GIL, Python stack sampling may pause;
the external host-memory/thread-wait sampler and NCCL recorder remain available.

Run 38805249199 selected g01. Target/draft file advice recovered 98 GiB;
advising the broader model directory recovered another 579 GiB, raising NUMA0
free RAM to 1110 GiB. All 5917 opened files were advised; sixteen files
were inaccessible, and no symlink directories were skipped. THP preparation
then slowed around 820 GiB. The second attempt reached only 825 GiB of actual
THP in 90 seconds despite 1156 GiB ordinary free RAM at its start. Roughly
300 GiB file cache and many smaller free blocks remained. The third attempt
was bounded by the remaining 27 seconds of the 300-second total budget.
The case excludes this demonstrated preparation failure for subsequent runs;
it does not increase the budget or accept partial THP coverage. No model
service started, so this run provides no runtime validation of the stack
sampler or the earlier NCCL stall.

Run 38809378429 failed preparation on g35. Model-directory advice recovered
450 GiB, but 700 GiB of file cache remained; repeated advice made essentially
no further progress. THP attempts reached only 636/685 GiB before their
90-second deadlines. No model services started.

The bounded preparation now uses temporary base pages when clean file cache
exceeds the reserve and the pool does not fit either ordinary free RAM or the
free blocks of at least 2 MiB. The scratch size is at most ordinary free plus
clean file cache minus 128 GiB, also capped by total NUMA memory and every
visible cgroup's remaining capacity minus the reserve. Dirty/writeback memory
is excluded. A NUMA-bound child uses MADV_NOHUGEPAGE and MADV_POPULATE_WRITE on
disjoint ranges with up to eight threads, so cache reclaim is not coupled to
THP compaction on each initial fault. It checks actual RSS and frees the entire
mapping before the existing THP helper starts. No pinning or VM setting changes
are involved. Failure or timeout prevents model startup; the existing 90-second
ordinary-step and 300-second total bounds still apply; the base-page step
includes an additional 30 seconds for release, as described below. The previous g01 exclusion is
removed so this path can handle that documented fragmentation case as well.
This is a CI preparation change; runtime improvement still requires validation.

Run 38816595852 on the same g35 filled all 1344 GiB of temporary base pages
in 66.8 seconds; RSS verification finished at 71.3 seconds with all memory resident and no huge pages.
The 90-second timer fired during munmap, when RSS had already fallen to 366 GiB,
and the child did not finish release within the five-second kill/reap allowance.
No model was started. The base-page step now has a 120-second bound, including
release, while other steps remain at 90 seconds and the total remains 300 seconds.
The child must still finish before THP preparation or model startup can proceed.

# Decode-node memory fault: an out-of-range cold-pool row

Status: **mechanism reproduced and contained (2026-08-08); the source of the bad
row is still unknown.** Read that literally — the fault can no longer kill the
process, and it is now observable, but nothing here explains what wrote the bad
row in the first place.

## The fault

`results/joint_sizing_m48_r14/decode.log:124150`, 53 minutes into the profiling
phase of the M48 / RATIO=14 / concurrency-48 round:

```
Memory access fault by GPU node-6 (Agent handle: 0x43d29f70) on address 0x71bb2298e000. Reason: Unknown.
```

State at the fault: host cold pool 341,349/343,722 (99.3%), GPU cold tier
24,586/37,918 promoted, index_cache 363,299/382,024 (95% — never full, so not a
joint-sizing overflow), running=30, waiting=26.

## Mechanism (reproduced)

`scripts/check_sparsekv_row_bounds.py` writes a row past the end of the pool
into a translation table and runs the real gather. Before the bound existed it
reproduced the production fault in about thirty seconds, with the same
signature and the same 4 KiB-aligned host address shape:

```
in-range gather moved data: True
Memory access fault by GPU node-6 (Agent handle: 0x181cd1f0) on address 0x765ed6de5000. Reason: Unknown.
```

That closes the diagnosis. The pinned cold pool's VMA is sized to the byte
(247,359,909,888 for 78 × 5,505,696 × 576 fp8, verified against
`/proc/<pid>/maps`), and the anonymous mapping that follows it is host memory
the GPU has no page-table entry for. Any row `>= num_host_pages * page_size`
therefore lands outside the mapping and the HSA runtime kills the agent.

Two independent facts pointed here before the repro:

- **The address granularity.** Every earlier fault in `results/` is on a
  *prefill* node and 2 MiB-aligned — device-memory page granularity, the known
  GEMM class in `docs/pp4pd_gemm_memory_fault.md`. This one is on *decode* and
  4 KiB-aligned: host page granularity. The swap kernels reading the pinned pool
  through `hipHostGetDevicePointer` are the only thing on decode that hands a
  GPU kernel a host pointer.
- **Mapping coverage was eliminated as a cause.**
  `scripts/probe_sparsekv_host_mapping.sh` swept every row of every layer on all
  four decode GPUs with 989 GB pinned in aggregate: all readable, zero pattern
  mismatches, and the mapping is identity (`dev == host VA`). So the pool is
  fully reachable and the bad address had to come from a bad row.

## Containment

The kernels guarded `cold_row >= 0` everywhere but **never bounded it from
above** — an out-of-range positive row was dereferenced unchecked.

`sparsekv_set_pool_rows(cold_rows, gpu_cold_rows)` publishes both pools' row
counts into `__device__` globals once per coordinator (they are per-process
constants), and `cold_row_of` now reports an out-of-range row as unbacked so
every caller's existing `< 0` guard skips it. The plain gather and the two
backup kernels' GPU-tier row get the same treatment. No per-op signature change
and no per-launch cost; the check is a compare against a constant.

Skipping keeps the process alive but makes that token read a stale hot slot, so
`sparsekv_take_oob_row_count()` counts every skip and
`ModelRunner._sparsekv_stage_and_sync` polls it every 1000 steps and logs a
warning. **A non-zero count is a correctness alarm, not back-pressure** — it
means a translation table is being corrupted and answers are quietly wrong.

While bounding, the fused inline gather turned out to be the one gather path
with *no* `cold_row >= 0` guard at all (`sparsekv_swap_and_translate_kernel`);
it now has one. Only reachable with the GPU cold tier off, which is why the
GPU-cold rounds never hit it.

## Related defects fixed in the same pass

`promote_to_gpu` assumed `_req_pages[slot][i]` backs logical page `i`. A partial
promote (GPU tier full) compacts that list, after which the assumption is false:

```
before promote (list pos -> logical page): [(0,0), (1,1), (2,2), (3,3)]
after  promote (list pos -> logical page): [(0,2), (1,3)]
```

A second promote would then map logical pages 0-1 (already on GPU, rows -1),
map nothing, burn two GPU pages, and hand back host pages still serving tokens
32-63. The logical index now travels with the page. It also did one `.item()`
per logical token plus an O(n²) `list.remove` loop — tens of seconds of forward-
loop stall on a 250K-token request — and is now whole-tensor.

Regression tests: `tests/test_sparsekv_coordinator.py::test_second_promote_*`,
`::test_promote_does_not_free_host_pages_it_did_not_move`,
`::test_promote_after_host_growth_past_the_promoted_prefix`,
`::test_promote_keeps_gpu_pages_when_nothing_is_backed`.

## Still open

**What writes the out-of-range row.** Values entering `req_to_host_pool` are
`page_id * page_size + offset` with `page_id < num_host_pages`, and reads are
bounded by `r < max_num_seqs` and `tok < cold_depth`, so on paper no row can be
out of range. Something breaks that on the c48 workload after ~53 minutes. The
counter is the instrument: if it fires, the corruption is real and frequent; if
a full c48 round passes with zero, the trigger is rarer than one round.

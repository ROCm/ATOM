# Phase 0 gate results: is the fully-associative CSA state-group worth building?

Gate from CSA_STATE_GROUP_IMPL_PLAN.md Phase 0: the associative pool only beats
the shipped **direct-mapped** store (`slot = phys % N`) when `phys % N` conflicts
drop hits at the deployable cap N. Measure before building.

## Method (hit-rate by trace replay, not accuracy)

Decision variable is **prefix-boundary hit rate vs cap N**, not GSM8K accuracy
(±2.3% SE at n=150 cannot resolve a hit-rate delta; recompute is a correct
fallback so accuracy barely moves either way — the win is TTFT/throughput, not
correctness).

1. Env-guarded `(phys, hash)` access trace added to `state_pool.py` (publish /
   invalidate / lookup). Flag-off byte-identical (45 block-manager tests pass).
2. One GPU run per workload: DeepSeek-V4-Flash TP4 fp8, GPUs 4–7, prefix caching
   ON, feature ON, **cap=0 (full store, no runtime eviction)** → the trace is the
   ideal access stream. Pool = 22142 blocks, full store = 17.74 GiB.
3. Offline replay (`logs_claude/csa_replay_sim.py`) drives BOTH a
   direct-mapped(N) model (faithful to the shipped slot-owner logic) and a
   fully-associative-LRU(N) model (invalidation guarded by hash-ownership, like
   block_manager.py:131) from the same trace, for N in a sweep. One run → whole
   curve for both policies. `coverage` = granted boundary blocks / primary-cache
   boundary blocks (1.0 = every reusable boundary's snapshot survived at cap N).

## Result depends on access pattern — measured both ends

### v1 — uniform round-robin, distinct suffixes (scan-hostile, NOT realistic)
Working set ~15k boundaries scanned uniformly = the classic LRU worst case.
- N < WS: associative **collapses to 0%** (recency never helps under a scan
  larger than the cache), while direct-mapped degrades gracefully (its
  address-keying acts like scan-resistant random eviction).
- N ≥ WS (20480): DM 97.6% vs LRU 100% → direct-mapped's **pure conflict penalty
  is only ~2.4%** when capacity is ample.

### v2 — Zipf(s=1.1) skew, exact-prefix reuse (realistic) ← the deciding run
Real prefix reuse is skewed (a few hot system prompts / documents). Associative
LRU wins at every deployable cap:

| N     | DM cov | LRU cov | gap     | reqs DM<LRU | HBM (of 17.7 GiB) |
|-------|--------|---------|---------|-------------|-------------------|
| 512   | 30.8%  | 49.0%   | +18.2%  | 1029        | 0.41 GiB |
| 1024  | 45.3%  | 67.1%   | +21.8%  | 1211        | 0.82 GiB |
| 2048  | 62.4%  | 79.7%   | +17.3%  | 1052        | 1.64 GiB |
| 4096  | 79.1%  | 89.5%   | +10.5%  | 639         | 3.28 GiB |
| 8192  | 90.9%  | 94.6%   | +3.7%   | 213         | 6.6 GiB  |
| 12288 | 93.2%  | 94.6%   | +1.4%   | 89          | 9.8 GiB  |
| full  | 94.6%  | 94.6%   | 0.0%    | 0           | 17.7 GiB |

Both converge to the same 94.6% ceiling at full store → the finite-N gap is
purely the eviction policy, not measurement bias. (Full ≠ 100% because ~5% of
reuse terminals were evicted from the *primary* cache under pool churn — equal
for both policies.)

## Verdict: conditional GO

- Under **realistic skewed reuse**, the fully-associative pool recovers **10–22%
  more boundary coverage** than direct-mapped at the caps where a capped store is
  actually deployed (N = 1k–4k = 0.8–3.3 GiB vs 17.7 GiB full). That is a
  material prefix-cache hit-rate (⇒ TTFT/throughput) win under HBM pressure.
- The win is **recency + conflict avoidance combined**, and it is **access-pattern
  dependent**: for a uniform / scan-once document workload (v1) the associative
  pool gives nothing (and LRU can even trail direct-mapped). It pays off exactly
  when reuse is skewed — the common serving case.
- When capacity is ample (N ≥ working set) direct-mapped's pure conflict penalty
  is small (~2.4%); the group is not worth it there. The value is entirely in the
  small-N + skewed-reuse regime.

## Caveats / honesty

- Contiguous FIFO allocation would understate direct-mapped conflicts; the heavy
  transient block churn in these runs scattered phys-ids across the pool, so this
  is not a best-case-for-DM artifact.
- A cheaper middle option exists: making the shipped store **k-way set-associative**
  (small k) would recover most conflict misses without the full content-addressed
  pool + control-plane refactor. Worth weighing against full associativity before
  committing to Phase 1–4.
- Artifacts: `logs_claude/csa_replay_sim.py`, `csa_multiprefix_workload.py` (v1),
  `csa_multiprefix_workload_v2.py`, `csa_trace_v1.txt`, `csa_trace_v2.txt`,
  `phase0_sweeps.txt`.

## Implementation status (Phases 1–4 built)

GO taken → full associative pool built and validated.

- **Phase 1 (CPU control plane):** new `atom/model_engine/csa_state_pool.py`
  `CsaStatePool` — hash-keyed, FIFO free-list + lazy eviction (≈LRU; hits refresh
  recency via pin→unpin→free-to-tail). Replaced the direct-mapped `_csa_slot_owner`
  path in `StatePool`; added `Sequence.csa_page_table`. Per-chunk page claiming
  (`ensure_csa_for_tokens`), publish on hash_blocks, source pinning on a hit,
  release on deallocate. No explicit invalidation (primary hash-eviction gates
  the lookup). 47 block-manager tests; full suite zero regressions vs baseline.
- **Phase 2–3 (GPU):** boundary tensor indexed by page id (staged
  `csa_page_tables` attached to the ratio-4 CompressPlan); capture kernel writes
  `boundary[csa_page_table[batch, pos//128]]` directly (dropped `phys % NUM_PAGES`,
  skips `page < 0`); restore source = page id (unchanged).
- **Phase 4 (validation):** kernel capture→restore round-trip bit-exact; GSM8K
  forced-prefix-hit associative cap=4096 = **94.0% (141/150)** vs shipped
  direct-mapped 94.67% vs recompute 96.67% — parity within noise; 316 restore
  fires, 0 errors; clean server start at **3.28 GiB** (cap=4096) vs 99.5 GiB full.

**Deferred (optional):** (a) budget still charges the boundary per-block
(`compute_block_bytes`); at cap>0 this over-reserves HBM — move to a fixed
allocation to reclaim it. (b) GPU re-measurement of the +10–22% hit-rate win
(already quantified above by trace replay).

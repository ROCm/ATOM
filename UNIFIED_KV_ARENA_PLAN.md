# Unified-KV Chunk Arena — implementation plan & record

**Branch**: `feat/unified-kv-arena` (off current `origin/main`, which already has
`ATOM_SWA_FULL_RETAIN`).
**Container**: `yhl_csa_recompute` (mounts this tree at `/app/ATOM`; editable
install done; CPU tests mock GPU/aiter).
**Flag**: `ATOM_V4_UNIFIED_KV_ARENA` (default `0` = current two-fixed-pool
behaviour, byte-identical).

## Goal

DeepSeek-V4 ships SWA KV and compressed (CSA/HCA) KV as **two fixed-size pools**
carved from each layer's `unified_kv` tensor. The split is frozen at start-up
(`ATOM_SWA_TAIL_BUDGET_FRAC`, default 0.2); an under-used pool wastes its
reserved rows. This feature makes the split **elastic**: SWA and compressed
borrow equal-size physical *chunks* from one arena per ratio group and hand them
back on demand.

**Caveat that gates the whole thing (measure first):** the arena does NOT shrink
data (SWA KV and compressed KV are different data, both needed). It only recovers
the fixed-split waste. Value depends on how much the fixed split actually wastes
— measure SWA-pool peak occupancy on the real workload before/while building.

## Architecture (diagram)

```
DeepSeek-V4 Unified-KV Chunk Arena   (flag: ATOM_V4_UNIFIED_KV_ARENA)

(1) Logical pools (unchanged: hashing / refcount / window-free|LRU; block_id global across layers)
   +---------------------------+        +---------------------------+
   | SWA Pool                  |        | Compressed Pool (CSA/HCA) |
   | swa_block_table (log id)  |        | block_table (log id)      |
   | window-free / full-retain |        | refcount / prefix LRU     |
   +------------+--------------+        +------------+--------------+
     alloc/free_swa                        alloc/free_compressed
                +---------------+-------------------+
                                v
(2) UnifiedKvArena  control plane   (logical id -> per-group physical page; swa_pages=0)
  API: alloc/free_{compressed,swa} . can_alloc_*(gate) . compress_page/swa_page(resolve)
  +----------------------------------+  +----------------------------------+
  | Group C4  (stride=128/4=32 r/blk)|  | Group C128 (stride=128/128=1 r/blk)|
  |  +----------------------------+  |  |  +----------------------------+  |
  |  | ChunkArena shared free-list|  |  |  | ChunkArena shared free-list|  |
  |  | chunk = 128 rows (=1 SWA)  |  |  |  | chunk = 128 rows           |  |
  |  +------+--------------+------+  |  |  +------+--------------+------+  |
  |   +-----v-----+  +-----v------+  |  |   +-----v-----+  +-----v------+  |
  |   | swa_free  |  |compress_free| |  |   | swa_free  |  |compress_free| |
  |   |1 pg/chunk |<>| 4 pg/chunk  | |  |   |1 pg/chunk |<>|128 pg/chunk | |
  |   |(128 rows) |el| (32 rows)   | |  |   |(128 rows) |el| (1 row)     | |
  |   +-----------+  +-------------+ |  |   +-----------+  +-------------+ |
  +----------------------------------+  +----------------------------------+
     ^ SWA logical id takes a page in EVERY group (capacity bounded by tightest)
     | within a group: SWA<->compress elastic borrow (<>); chunk returns to arena only when fully drained
     |
     +-------------> (3) Physical            +-------------> (4) index-build -> kernel
     v
(3) Physical: per-layer arena tensor  [num_chunks x 128, head_dim]  (one buffer)
   +--------+--------+--------+--------+--- ... ---+
   |SWA blk |C4 blkx4|SWA blk |C4 blkx4|           |   <- SWA/compress chunks interleave
   |[.,128, |[.,32,  |[.,128, |[.,32,  |           |   <- reshape per view
   |  hd]   |  hd]   |  hd]   |  hd]   |           |
   +--------+--------+--------+--------+--- ... ---+
   row = physical_page x page_size   (chunk single-owner -> SWA/compress row ranges never alias)

(4) index-build translation -> Triton kernel
   logical id --compress_page/swa_page--> per-group PHYSICAL block_table / swa_block_table
                                          (shared by read gather + write compressor scatter)
                                              |
                                              v
   Triton kernel UNCHANGED: op5 page_size=1 per-token CSR;  row = page x page_size, swa_pages=0

Status:  (1)(2) DONE + tested (chunk_arena.py + unified_kv_arena.py, 15 unit tests)
         (3)(4) + sizing = TODO (5 coupled hot-path edits)
Payoff bound: does NOT shrink data; only recovers the fixed-split waste
              -> measure real SWA-pool occupancy before committing.
```

## Key facts established (from code, verified)

- Per-layer physical: `unified_kv[L] = [swa_pages + compress_pages_L, head_dim]`.
  SWA region `[0, swa_pages)`, compress region `[swa_pages, …)`.
  `swa_pages = num_swa_blocks * block_size`.
- Compressed rows-per-block are **per ratio**: `k1_csa = block_size//4 = 32`
  (C4/CSA), `k2_hca = block_size//128 = 1` (C128/HCA). SWA = `block_size` (128).
  (`deepseek_v4_attn.py:354-355`.)
- Index row formulas (the kernels are parametric, NOT hard-baked to 128):
  - SWA: `row = swa_phys * block_size` (`paged_prefill_indices.py:146`,
    `paged_decode_indices.py:71`).
  - Compress: `row = swa_pages + block_id * stride_L` (HCA stride 1:
    `paged_prefill_indices.py:172` `swa_pages + bt`; CSA stride 32).
- The asm decode kernel `mla_decode_fwd_v4_nm` (op5) is **page_size=1, per-token
  CSR** (`paged_decode.py:931-937`) → block_size-agnostic. Changing SWA page
  granularity does NOT drop the asm fast path; it only affects memory-coalescing
  (SWA reads become as scattered as compressed reads already are in the same
  call — plausibly a small hit, unmeasured).
- `block_id` is **global across layers** (one `block_table` / one
  `swa_block_table` per seq, reused by every layer).

## Why per-group, not one global arena (the architecture constraint)

A single physical chunk of fixed rows maps to a *different number of logical
blocks* in a C4 layer (128/32 = 4) than in a C128 layer (128/1 = 128). Because
`block_id` is global, one global chunk→block assignment cannot be consistent
across ratios: releasing "one SWA block's worth" of rows frees 4 C4 blocks in C4
layers but 128 C128 blocks in C128 layers — inconsistent global block sets.

**Fix:** per-ratio-group arenas. Layers with identical physical layout form a
group (all C4 layers; all C128 layers; dense/SWA-only layers). A logical id
resolves to a **per-group physical page**. Every layer in a group is driven by
the SAME global allocations, so groups stay in lock-step without cross-layer
coordination. SWA lives in EVERY group → an SWA logical id gets a page in every
group's arena (coupled: SWA capacity bounded by the tightest group).

`block_id` stays **logical** (pool hashing/refcount/LRU unchanged); the arena is
a side map `logical id → per-group physical page`. Index-build translates to
physical and passes `swa_pages = 0`, so the kernel formula `page * page_size`
lands correctly (chunks are single-owner, so SWA stride-128 and compressed
stride-32 page ranges never alias).

Chunk = `block_size` (128) rows = one SWA block (the largest page). SWA = 1
page/chunk; C4 = 4 pages/chunk; C128 = 128 pages/chunk.

## DONE (tested, on branch)

- `atom/model_engine/chunk_arena.py` — `ChunkArena` (shared equal-size chunk
  free-list) + `ChunkBackedFreeList` (per-pool: borrow chunk, pack pages, return
  chunk only when fully drained). Page id == arena physical page index.
  Tests: `tests/test_chunk_arena.py` (7).
- `atom/model_engine/unified_kv_arena.py` — `UnifiedKvArena` control plane:
  per-group `ChunkArena` + SWA/compress `ChunkBackedFreeList` + logical→physical
  maps + `alloc/free_{compressed,swa}` (all-or-nothing across groups) +
  `can_alloc_*` gates + `compress_page`/`swa_page` resolution.
  Tests: `tests/test_unified_kv_arena.py` (8) — per-group resolution, SWA backed
  in every group, rollback on group exhaustion, C4 packing, elastic borrow,
  SWA bounded by tightest group, idempotent free.
- `atom/utils/envs.py` — `ATOM_V4_UNIFIED_KV_ARENA` flag.
- Control-plane **v1-simple** (converged correct form): the arena tracks only
  USED vs TRULY-FREE pages; `free_*` = true eviction (chunk returns to the arena
  when fully free); cross-pool lending is POOL-DRIVEN (BlockManager evicts cold
  ref-0 sibling blocks then retries) — single source of truth for cache state,
  in the pool. (Superseded v2's arena-side lendable/surrender/evict-callback,
  which duplicated the pool's lazy cache and could lend a page a cache hit still
  wanted; not a lock/concurrency issue — a dual-state consistency one.)
- Total new CPU tests: **17 passing** (8 chunk_arena + 9 unified_kv_arena).
  Full `pytest tests/` (excl. sglang plugin dep) green (723) with flag off.

## CRITICAL design refinement — arena-page lifecycle vs lazy eviction  [DONE in control-plane v2]

(Found while wiring step 3; the initial control-plane model over-simplified this.
Resolved in control-plane **v2** — `chunk_arena.py` + `unified_kv_arena.py` now
implement lazy lending: fully-free chunks stay resident/reusable and are only
surrendered on sibling demand via `surrender_lru(evict_cb)`; `alloc_*` borrow
across pools on `ArenaEmpty`; `can_alloc_*` count borrowable chunks;
`register_evictors` wires the owner's hash-drop callback. 17 unit tests.
The remaining work is to WIRE the pools' real evict callbacks + alloc/free
calls (step 3 below).)

The compressed pool has a **three-state** block lifecycle, not two:
`used` (ref>0) / `free-but-cached` (ref==0 but hash+KV still valid — lazy
eviction) / `evicted` (only when `_allocate_block` pops+resets the slot). A
cache hit **claims a free-but-cached block** (its KV is still there). So a
block's PHYSICAL page is occupied from `_allocate_block` until the slot is
actually reused — NOT until ref hits 0. In steady state ~all `num_blocks` pages
hold cached KV.

Implications for the arena (correcting `ChunkBackedFreeList`'s "drain → return"):

1. **Borrowing across pools = triggering eviction.** For compressed to lend a
   chunk to SWA, its free-but-cached blocks in that chunk must be evicted
   (drop `hash_to_block_id` entries + reset). A chunk is *lendable* when all its
   pages are ref==0 (cached or free), but lending destroys their cached KV.
2. **Do NOT return a chunk eagerly on drain.** If a ref-0-but-cached chunk goes
   back to the arena and is reassigned to SWA, a later cache hit on one of its
   blocks would read the wrong KV. Returning a chunk MUST invalidate those
   blocks' hash entries first.
3. **On-demand ("lazy") lending, LRU.** A pool keeps its ref-0-but-cached chunks
   for future hits; it only surrenders the coldest (LRU) chunk when the OTHER
   pool is starved and requests a borrow. This matches ATOM's lazy-eviction
   philosophy and preserves hit rate until real pressure. This replaces
   `ChunkBackedFreeList`'s current eager "return chunk when fully pushed".

Required control-plane changes:
- Track per-chunk **ref-occupancy** (pages with ref>0) separately from
  cached-occupancy. A chunk is lendable when ref-occupancy == 0.
- `borrow_from_other(group)` — evict the LRU lendable chunk of the lender
  (invalidate its blocks' hashes via a callback into the owning pool), return it
  to the arena, hand it to the requester.
- Keep ref-0-but-cached chunks resident (do not auto-return); only the lender's
  LRU chunk is surrendered on demand.
- The owning pool must expose an **evict-chunk callback** (drop hash entries +
  reset the blocks) that the arena calls when a chunk is lent.

Until this is implemented, `ChunkBackedFreeList` (eager return) is only correct
for the pure free/alloc case with no lazy-eviction cache — fine for the unit
tests, NOT for the real pools. Wiring step 3 depends on this refinement.

## STATUS

- Control plane (chunk_arena, unified_kv_arena) — v1-simple, DONE + tested.
- **Step 3 (BlockManager + SlidingWindowPool wiring) — DONE at CPU level.**
  BlockManager builds the arena (`_build_arena`, from `config.v4_arena_group_specs`
  + flag), sizes the compressed logical id space to `max_compressed_blocks()` and
  the SWA pool to `max_swa_blocks()`, backs blocks via `_arena_alloc_compressed`
  / SWA `_arena_alloc_swa` at their alloc points, and does POOL-DRIVEN bidirectional
  lending: on `ArenaEmpty` each side evicts the coldest ref-0 sibling block
  (`_evict_cold_compressed` / `swa.evict_cold_for_arena` — drop hash + `free_*`)
  and retries. Admission (`_has_free_compressed`, `swa.has_free`) counts arena
  pages + reclaimable sibling chunks (compressed) / own arena capacity
  conservatively (SWA). Tests: `tests/test_block_manager_arena.py` (6, incl.
  elastic borrow). Full `pytest` 729 green, flag off byte-identical.
- Remaining for a RUNNABLE GPU path: steps 1, 2, 4, 5 below (sizing → arena
  tensor → per-group physical index/scatter translation → GSM8K validation).

## STEP 3 CROSS-PROCESS FINDING (decisive for scope)

BlockManager (which owns the arena) runs in the scheduler/EngineCore process;
ModelRunner + the DSV4 attn metadata build (which stages block_tables to the
kernels) run in the TP WORKER processes (spawn). `model_runner.py` never
references `block_manager`. So the arena's logical→physical mapping is NOT
available where block_tables are staged.

=> The per-group physical translation must happen SCHEDULER-SIDE (where the
arena lives) and the resulting per-group physical block_tables / swa_block_tables
must be shipped to the workers in the batch. This enlarges step 3 beyond the
worker: it touches the scheduler batch build + the ScheduledBatch structure +
IPC + the worker's `_populate_block_tables`/`_populate_swa_block_tables`.

Both shipped V4 models are mixed ratio (C4 + C128 + a few dense), so 3 groups
and their per-group physical tables are all required. The C4-only simplification
(see below) reduces the group count but does NOT remove the cross-process
shipping requirement.

## STEP 2 SHAPE FINDING (must trace on GPU)

Current `unified_kv[layer] = [swa_pages + compress_pages, head_dim]` puts SWA in
the FRONT contiguous region `[0, swa_pages)` and compress in the BACK
`[swa_pages, …)`; `build_kv_cache_tensor` binds `attn.swa_kv = unified[:swa_pages]`
and `compressor.kv_cache = unified[swa_pages:]` by SLICING.

Under the arena, SWA and compress chunks INTERLEAVE in one `[num_chunks *
block_size, head_dim]` tensor (addressed via block_tables), so they can no
longer be bound by front/back slicing. Both views must bind the FULL tensor as
the base ptr; the physical block_table + `page * stride` (swa: ×block_size,
compress: ×compress_stride, `swa_pages = 0`) give the row. This requires
reworking `build_kv_cache_tensor` binding + reshape and re-checking every kernel
base-ptr/stride assumption ON A REAL FORWARD — a wrong shape is silent KV
corruption catchable only by GSM8K. Hence steps 2-3 are GPU-in-the-loop, not
blind edits.

## GPU VALIDATION (2026-07-25) — arena RUNS end-to-end, no accuracy regression

flag-on (`ATOM_V4_UNIFIED_KV_ARENA=1`) server (V4-Flash-latest, TP4, fp8, GPU
4-7) starts, sizes the arena (specs c4/c128/dense; num_kvcache_blocks≈219k vs
365k flag-off), and generates correctly. GPU-in-the-loop fixed 3 bugs: stray
`envs` ref; per-rank-sizing view mismatch (derive rows from
num_physical_kvcache_blocks); decode/prefill compress OOB (swa_pages=0 under
arena). GSM8K (40 Q, same crude harness): flag-on 19/40 (47.5%) vs flag-off
14/40 (35.0%) — flag-on NOT degraded (delta is TP4-fp8-greedy nondeterminism +
small-sample noise, see project_dsv4_greedy_nondeterministic). => arena is
correct end-to-end under NORMAL (no-eviction) load, where the arena mapping is
identity (compress_page(b)==b).

NOT yet validated: non-identity (eviction/borrow) correctness. With ~219k blocks
and short prompts there is no eviction, so scatter/decode/prefill all use
block_id as the physical row (identity) consistently. Under real memory pressure
the mapping diverges and scatter + decode + csa_translate_pack must use per-group
PHYSICAL tables (currently only prefill's paged read does). Remaining work +
a pressure test below.

## REMAINING: full per-group physical for non-identity (eviction/borrow)

Everything else is DONE + committed (control plane, block manager, sizing,
scheduler cross-process shipping of `batch.arena_{block,swa_block}_tables`,
worker arena tensor + interleaved binding). The last piece — the forward index
build must feed PER-GROUP PHYSICAL tables + `swa_pages=0`:

Exact sites (all set `attn_metadata.block_tables` / `swa_block_tables`):
- prefill: `deepseek_v4_attn.py:1775/1779` (`_populate_block_tables` /
  `_populate_swa_block_tables`).
- decode: `:1530`; decode-CG: `:1719`; another decode path: `:3128`.

For each, when arena on, build per-group physical GPU tables from
`batch.arena_block_tables[g]` / `batch.arena_swa_block_tables[g]` (g in c4/c128/
dense) instead of the single logical table, and store per-group on
`attn_metadata` (e.g. dicts). Then the consuming kernels use the layer's group
table + `swa_pages=0`:
- `paged_prefill_indices` / `paged_decode_indices`: HCA compress section uses
  c128 table; SWA prefix section must write c4-swa physical to the CSA buffer
  and c128-swa physical to the HCA buffer (per-group-per-destination) — this is
  the kernel-internal change (or run the index build once per group into
  per-group buffers, reusing the kernels unchanged, and pick per-layer).
- `csa_translate_pack`: CSA topk compress uses c4 table.
- compressor scatter (`deepseek_v4.py`) + `swa_write`: write via the group table.

Validation: only reachable once ALL sites are consistent (big-bang) → start a
server with `ATOM_V4_UNIFIED_KV_ARENA=1` and require GSM8K flag-on == flag-off.
Recommend GPU-in-the-loop (shape-trace each site against a live forward); blind
edits across these hot kernels cannot be validated incrementally.

## TODO — earlier notes (superseded by the section above)

These cannot be CPU-unit-tested in isolation (need the physical tensor); land
them together behind the flag, then start a server and validate.

1. **model_runner sizing** (`get_num_blocks` region, ~line 1690-1740). When flag
   on: compute per-group `num_chunks` instead of the fixed `num_swa_blocks` /
   `num_kvcache_blocks` split. Group C4: `num_chunks_c4 = num_swa + ceil(num_blocks/4)`;
   group C128: `num_swa + ceil(num_blocks/128)`. Keep total bytes ≈ current.
   `num_blocks` (compressed capacity) is elastic, upper-bounded by the tightest
   (C4) group. Put group_specs `(name, stride, num_chunks)` on config for the
   scheduler-side BlockManager.
2. **`allocate_per_req_cache`** (`deepseek_v4_attn.py:628`). When flag on, per
   layer allocate a flat arena tensor `[num_chunks_group * block_size, head_dim]`
   (no fixed swa/compress boundary). Bind `attn.swa_kv` / `compressor.kv_cache`
   as views over the whole arena (they already reshape).
3. **BlockManager + SlidingWindowPool**. Construct `UnifiedKvArena` (flag on).
   - `alloc_compressed(b)` at `_allocate_block` (a truly-free block_id takes a
     physical page); `free_compressed(b)` only at TRUE eviction (the reset in
     `_allocate_block` when a slot is reused, and `clear_cache`) — NOT at
     `_deallocate_block` (ref-0 is lazy-cached; the page stays mapped to `b`).
     Cache-hit-claim does NOT touch the arena (b's page mapping persists).
   - `alloc_swa`/`free_swa` symmetric at the SWA pool's true alloc/evict points.
   - Gate `can_allocate` / `can_append` with `can_alloc_*` (arena-aware).

   **INTEGRATION MODEL CORRECTION (derived while wiring — supersedes v2 surrender).**
   `free_compressed(b)` / `free_swa(s)` must be called ONLY at TRUE eviction
   (hash dropped), NOT at ref-0 `_deallocate_block` — a ref-0 block is still
   lazily cached for hits, so its page must stay owned by `b` and MUST NOT enter
   the arena free-list (else a lend/re-pop corrupts a later cache hit). Since
   ref-0-cached pages are not in the arena free-list, the arena cannot decide
   what is lendable on its own. => **Lending is POOL-DRIVEN (option a):** when a
   pool can't `alloc_*`, BlockManager evicts the coldest ref-0 sibling blocks
   (its own `free_block_ids` LRU) — drop `hash_to_block_id` + `free_*` — until
   the tightest group frees a chunk, then retry. The committed v2
   `ChunkArena` + logical→physical maps + `can_alloc_*` are correct and reused;
   v2's `surrender_lru`/lendable machinery is SUPERSEDED by this pool-driven
   eviction (leave it unused or simplify later). `can_alloc_*` should count
   truly-free pages + pages freeable by evicting the sibling's ref-0 blocks.

   **Cross-group note (still applies).**
   A logical `block_id` has a physical page in EVERY group; evicting it must free
   ALL of them. So lending a chunk in the tightest group (C4) must evict the
   *logical blocks* occupying it, which also frees their (scattered) C128 pages.
   v2's per-group `surrender_lru` is not enough. Two options:
   (a) LRU-driven: evict coldest ref-0 logical blocks (pool's `free_block_ids`)
       via `free_compressed(b)` until the C4 arena yields a free chunk — simple,
       may fragment; pack consecutive block_ids into a chunk at alloc to help.
   (b) reverse-map: keep a per-group `physical page → logical id` map; target a
       fully-ref-0 C4 chunk and evict exactly its blocks — precise, more state.
   Evict callback (`register_evictors`) must drop `hash_to_block_id` for the
   evicted logical ids AND call `free_compressed`/`free_swa` for all groups.
4. **index-build translation** (`_populate_*` in `deepseek_v4_attn.py` +
   `paged_{prefill,decode}_indices`). Produce **per-group physical**
   `block_tables` / `swa_block_tables` by mapping logical ids through
   `arena.compress_page(group, b)` / `arena.swa_page(group, s)`; pass
   `swa_pages = 0`. Kernels unchanged (formula `page * page_size`).
5. **compressor scatter (write side)**. The compress plan / scatter also
   locates writes via `block_tables` → feed it the SAME per-group physical
   table (share the one translation point with the read side).

## GPU VALIDATION 2 (2026-07-25) — non-identity per-group physical CORRECT; borrow livelock found

Completed #8/#9 (per-group PHYSICAL translation on ALL sites so non-identity =
eviction/borrow is correct, not just identity):
- Model forward per-layer consumers route through `_arena_layer_tables(attn_md,
  ratio)` → group table (c4/c128/dense): compressor scatter WRITE, SWA WRITE
  (`_attn_core` block/swa tables), indexer topk gather, `csa_translate_pack`.
- Decode index build (`_attach_v4_paged_decode_meta`): per-group SWA-prefix build
  (3× `write_v4_paged_decode_indices`, dense→swa / c4→csa / c128→hca, scratch for
  the other two) + HCA compress head from c128 physical numpy; `swa_pages=0`.
- CG-safe persistent arena tables: `v4_arena_{bt,swabt}_{c4,c128,dense}` forward_
  vars buffers, staged in prepare_decode + build_for_cudagraph_capture (mirrors
  prepare_prefill). GPU views feed the captured forward; numpy mirrors feed the
  eager decode HCA-head build.

**Decisive cross-process fix (was the reason the arena silently used logical-as-
physical):** `config.v4_arena_group_specs` is computed in the RUNNER subprocess
but the scheduler process (which owns `BlockManager` → the arena → ships per-group
tables) never received it → `block_manager.arena=None` → tables not shipped →
worker fell back to logical block ids as physical rows → correct only under
identity (compress_page(b)==b). Now propagated through the same `get_num_blocks`
→ `block_info` channel as `num_swa_blocks` (`model_runner.py` + `engine_core.py`).

**Validation (forced eviction):** `logs_claude/arena_evict_gsm.py` prepends a
UNIQUE ~2200-tok filler per request so distinct compressed blocks (≈17600 over
800 req) ≫ the pool (util 0.24 → 10080) → pigeonhole page recycling → non-identity.
A gated probe (`ATOM_ARENA_DEBUG=1`) confirmed non-identity was actually exercised
(swa(c4).maxdiff=9, c4.maxdiff=8, c128 compress diverged too). GSM8K held **95.5%
== 0.96 flag-off baseline** over the first 200 req (padding is benign: 6/6 under
no-eviction). => #8/#9 per-group physical translation is CORRECT under real
non-identity. Flag OFF stays byte-identical; 729 CPU tests green.

**Blocker found (#11, control-plane borrow, NOT #8/#9):** under sustained eviction
the run deterministically stalls at ~200-236 req — EngineCore 89% CPU, GPU idle.
py-spy: `busy_loop → schedule (scheduler.py:1058) → can_allocate → compute_hash`
spinning; `can_allocate` returns -1 forever with no running seq to progress. Root
cause: `_evict_cold_compressed` / `swa.evict_cold_for_arena` pop the lent logical
id off the free list but never re-add it → cross-pool borrow permanently LEAKS
logical ids → admission starves. (Also: the eviction scan destructively `popleft`s
skipped unbacked/ref>0 ids without discarding from the set, so a naive re-add
desyncs deque/set; and `_has_free_compressed` is over-conservative for reuse of
already-backed cached blocks.) Needs a free-list-lifecycle redesign + unit tests
(spun off to task #11); the elastic borrow can't run under real pressure until
fixed. Repro: `util 0.24 + arena_evict_gsm.py --limit 800 --pad-tokens 2200`.

## GPU VALIDATION 3 (2026-07-25) — borrow livelock FIXED; arena works under pressure

Fixed the #11 control-plane livelock — the arena now runs correctly under sustained
memory pressure (real cross-pool borrow + eviction). Three coupled bugs in the
borrow control plane (BOTH pools):
1. **Logical-id leak**: `_evict_cold_compressed` / `swa.evict_cold_for_arena` popped
   the lent id off the free list but never re-added it → cross-pool lending
   permanently leaked ids → admission starved → scheduler busy-loops.
2. **Destructive scan**: the eviction scan `popleft`'d skipped (unbacked/ref≠0)
   items, desyncing deque/set.
3. **Over-conservative admission**: `_has_free_compressed` / `swa.has_free` demanded
   NEW arena pages even when reusing an already-backed cached block (whose page is
   held) → when the pool was full of ref-0 cached blocks, `backable≈0` < n → false
   `-1` forever (the (b) livelock).
Plus `compress_pages_per_chunk` returned MAX instead of the binding MIN (one SWA
evict enables only min-group pages/chunk of compressed → over-admission risk).

**Fix — two-list free pool in both pools:** `free_block_ids` = BACKED-free (holds an
arena page; reuse costs 0 pages; the lendable set) and `_unbacked_free_ids` = ref-0
ids whose page was lent (reuse re-borrows a page). `_pop` prefers backed → the
admission accounting is sound. Cross-pool evict moves the id backed→unbacked (never
leaks; the scan only touches backed ids now). Admission credits backed-free reuse:
`total_free >= n AND backed_free + backable >= n`. `compress_pages_per_chunk`→MIN.
Arena-OFF stays byte-identical (single pool, unbacked pool empty).

**Validation:** 4 new unit tests (id conservation under repeated cross-pool evict,
backed-free reuse admission, evicted-id reusability, bounded-evict-no-spin) +
**733 CPU green**, flag-off byte-identical. GPU: the SAME padded forced-eviction
harness (`arena_evict_gsm.py`, util 0.24, 800 req, concurrency 32) that previously
**hard-stalled at ~200 req** now **completes all 800** with non-identity confirmed;
**flag-ON 756/800 = 94.5% vs flag-OFF control 752/800 = 94.0%** (same harness/util,
only the arena differs) — statistically identical → the arena is transparent and
the livelock is gone. Files: `block_manager.py`, `swa_pool.py`, `unified_kv_arena.py`.

## Validation

- Flag OFF: 713 CPU tests + full `pytest tests/` (excl. sglang plugin missing
  dep) stay green; behaviour byte-identical.
- Flag ON: start DSV4 server; **GSM8K flag ON vs OFF must score identically**
  (any logical→physical mapping bug corrupts KV → score drops). Validate with
  GSM8K, NOT token-level A/B (TP4 fp8 greedy is non-deterministic — see memory
  `project_dsv4_greedy_nondeterministic`).
- Measure: arena ON vs OFF `num_kvcache_blocks`, concurrency, SWA occupancy
  elasticity → quantify recovered fixed-split waste.

## Open decision

Granularity chosen: **per-ratio-group** (C4, C128 groups; SWA spans both).
Rationale: layers in a group are identical and driven by the same global
allocations → maps stay in sync; only ~2-3 tables vs per-layer's ~62.

## Risks / rollback

- Flag default off → zero change to the shipping path.
- Highest risk = the logical↔physical map (steps 4-5): a wrong translation reads/
  writes the wrong KV. GSM8K flag on/off parity is the backstop.
- Fragmentation: a chunk returns to the arena only when fully drained; a few
  sticky LRU blocks pin a chunk (C4: ≤3 blocks pinned = mild; C128: many blocks
  but ~0 bytes). Cross-pool elasticity is best-effort. Optional compaction
  (migrate the last live block, update block_table + refcount) only if measured
  fragmentation is bad — NOT in scope for the first landing.

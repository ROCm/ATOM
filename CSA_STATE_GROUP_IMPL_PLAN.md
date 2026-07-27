# Impl plan: fully-associative paged CSA-state group

Scope: upgrade the shipped CSA boundary-snapshot store from a **direct-mapped**
evictable array (`slot = phys % N`) to a **fully-associative, content-addressed,
evictable paged pool** keyed by the 128-block hash. Ring stays for live
execution; a hit still restores the tail page into the fresh ring. This is the
only realizable delta over the shipped snapshot (see CSA_STATE_GROUP_CORRECTED.md
for why "no ring / no restore" is out).

Base branch: `feat/csa-snapshot-evictable` (ring + capture/restore validated,
direct-mapped store in place). Reuse the generic `SlidingWindowPool` /
`group_block_tables` groundwork from `feat/csa-state-group`.

## Phase 0 — Gate: prove the group is needed (no code changes)

The group only beats direct-mapped when N is small enough that `phys % N`
conflicts start dropping hits. Measure before building.

- On a representative workload (or a synthetic one with many distinct long
  cached prefixes), sweep the shipped store `ATOM_V4_CSA_BOUNDARY_SLOTS` ∈
  {512, 1k, 2k, 4k, 8k, full} and record prefix-hit rate + accuracy.
- Decision:
  - If the N that fits the HBM budget shows no hit-rate loss → STOP, ship
    direct-mapped. Group not worth it.
  - If hit-rate degrades at that N (direct-mapped conflicts) → proceed; the
    knee quantifies the win the group must recover.

Deliverable: a hit-rate-vs-N curve; go/no-go.

## Phase 1 — Associative csa page pool + control plane (CPU-testable)

- `CsaStatePool` (or `SlidingWindowPool` used as a non-windowed content store):
  `num_csa_pages` slots, `hash_to_page: dict[int,int]`, free-list, per-page
  refcount, LRU / lazy eviction (mirror the primary block pool).
- `Sequence.csa_page_table: list[int]` (block_idx → page id, -1 = none). Reuse
  `group_block_tables["csa_state"]`.
- BlockManager lockstep (guarded by the flag; no-op otherwise):
  - `allocate` (+ chunk scheduling): for each 128-block this seq will produce,
    claim a csa page (evict LRU if full), refcount++, record in csa_page_table.
    Must precede the forward so capture can write into the page.
  - `hash_blocks` (postprocess, prefill only): publish the finalized block's page
    under its 128-block hash; refcount-- (page now shared/cacheable).
  - `can_allocate`: for the candidate hit boundary, look up the terminal block's
    128-hash → page; present ⇒ hit, `csa_boundary_source = page id`; else shrink
    boundary (reuse the existing bound_hit intersection loop).
  - `deallocate`: release the seq's still-held pages (refcount--).
  - Eviction: `_pop` drops the LRU free page's `hash_to_page` entry (lazy), like
    the primary/SWA pools; a refcounted (live) page is never evicted.
- Replace the direct-mapped `_csa_slot_owner` path in StatePool with the pool.
- CPU tests: claim/publish/lookup/evict/refcount lifecycle; associative hit
  where direct-mapped would conflict-miss; cap-0 legacy still full-coverage.

## Phase 2 — Paged storage + budget + staging (GPU)

- Builder `allocate_per_req_cache`: allocate the pool tensors
  `[n_csa, num_csa_pages, 4, dim]` (main+idx, kv+score); startup log (slots, GiB).
- model_runner budget: reserve the pool as a **fixed allocation** (not per-block),
  so `num_kvcache_blocks` reclaims the HBM the shipped version over-reserves
  (the follow-up noted in CSA_SNAPSHOT_EVICTABLE.md). One number:
  `n_csa * num_csa_pages * 4 * (2*head_dim + 2*index_head_dim) * 2 * 4B`.
- `prepare_prefill`: stage the per-seq `csa_page_table` to GPU (like block_tables).

## Phase 3 — Wire capture/restore to the pool (GPU)

- Capture: pass `csa_page_table` (block_idx → page) as the capture kernel's
  table; write `boundary[csa_page_table[batch, pos//128]]`. Small tweak: when a
  real page table is supplied, use its id directly (drop the `phys % NUM_PAGES`
  modulo the direct-mapped path uses).
- Restore: `source = page id` from the hash lookup (already just an index);
  reuse `restore_compressor_boundary` unchanged. One-shot source threading and
  the postprocess clear are already in the scheduler.
- Everything else (ring, `Compressor.forward` restore-before-read /
  capture-after-write) unchanged.

## Phase 4 — Validation

- Kernel: capture→restore round-trip still bit-exact with the page-table path.
- GSM8K forced-prefix-hit: group == shipped == recompute within noise (parity).
- The Phase-0 regime: at the small N where direct-mapped lost hits, the group
  recovers them (the actual win). Report the hit-rate delta.
- CPU suite green; flag-off byte-identical.

## Reuse map (already built + validated)

- Capture write (every-block-tail) — `capture_compressor_boundary`. ✓
- Restore into ring — `restore_compressor_boundary`. ✓
- Paged state read (if ever going ringless) — `fused_compress_attn` PAGED_STATE
  on `feat/csa-state-group`. ✓ (not needed for this ring-kept design)
- Generic `SlidingWindowPool` + `group_block_tables` — `feat/csa-state-group`. ✓

## Risks / watch-items

- Page must be claimed **before** the forward (capture writes it); a live page
  must be refcount-protected from eviction.
- Refcount correctness across chunked prefill, decode growth, concurrent
  requests sharing a prefix (claim once, share by hash), preemption/release.
- Decode: pages produced during generation aren't published (prefill-only, like
  the snapshot) — decode-cached prefixes stay non-reusable in v1.
- Budget reservation must match the actual pool size, or OOM / under-use.

## Effort / value

- Effort: moderate multi-file control plane + GPU e2e, several debug rounds.
  Write/read kernels already done, so the new work is the associative pool +
  per-seq page table + budget + staging.
- Value: recovers direct-mapped conflict misses **only** in the small-N +
  many-distinct-long-prefixes regime (quantified by Phase 0). Zero at N≈full or
  under heavy prefix sharing. Plus a fixed-pool budget that converts saved HBM
  into KV capacity.
- Gate on Phase 0: build only if the curve shows the loss.

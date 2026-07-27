# CSA snapshot: evictable direct-mapped boundary store

Localized optimization of the validated snapshot (branch
`feat/csa-snapshot-evictable`, off `feat/csa-prefix-snapshot`). Replaces the
fixed `[num_blocks, 4, dim]` boundary tensor with a **direct-mapped, evictable**
store of `num_csa_boundary_slots` slots, so the snapshot HBM is tunable instead
of scaling with the whole KV block count.

## Design

- A snapshot for compressed physical block `phys` lives in slot
  `phys % N` (`N = num_csa_boundary_slots`, config /
  `ATOM_V4_CSA_BOUNDARY_SLOTS`; `0` = legacy full store, `N = num_blocks`,
  slot == phys, no eviction).
- `StatePool._csa_slot_owner[slot] = phys` records the current owner; a newer
  block finalizing on the same slot evicts the older (owner overwrite). A hit
  is offered only if the block still owns its slot; otherwise the boundary
  shrinks (same as a missing SWA page).
- `csa_boundary_source(phys)` returns the **slot** (`phys % N`); the restore
  kernel reads `boundary[slot]`. The capture kernel writes
  `boundary[phys % NUM_PAGES]` (`NUM_PAGES` = store dim 0). Legacy `N =
  num_blocks` ⇒ `phys % num_blocks == phys`, byte-identical to the old path.
- Ring + capture/restore kernels otherwise unchanged. This is *not* the state
  group refactor — see CSA_STATE_GROUP_CORRECTED.md for why that isn't worth it.

## Validation (GPU, DeepSeek-V4-Pro TP4 fp8)

Startup log with `ATOM_V4_CSA_BOUNDARY_SLOTS=4096` (num_blocks=18534):
`30 CSA layers x 4096 slots x 4 rows -> 4.69 GiB HBM` (vs 21.5 GiB full = 4.6x
smaller).

GSM8K forced-prefix-hit (fixed 8-shot prefix, restore fires every question,
n=150):

| Store | HBM | accuracy | restore fires |
|---|---|---|---|
| full snapshot (legacy) | 21.5 GiB | 95.33% (143/150) | every q |
| **evictable cap=4096**  | **4.69 GiB** | **94.67% (142/150)** | 132+ |
| recompute baseline | — | 96.67% (145/150) | — |

1-question spread across all three — within the ~2.3% SE at n=150 and the
model's run-to-run nondeterminism. The capped store fired restore on every
question and held accuracy at 4.6x less HBM.

Tests: 87 CPU (incl. direct-mapped eviction + cap-0 full-coverage); GPU
capture/restore round-trip bit-exact under `phys % NUM_PAGES`.

## Notes / follow-ups

- The KV budget (`compute_block_bytes`) still charges the boundary term as the
  legacy per-block (num_blocks) amount — conservative: with a cap the *footprint*
  drops but `num_blocks` is not increased to reclaim the freed reservation.
  Tightening the budget to charge only `N` slots would convert the saved HBM into
  more KV capacity. Left as a follow-up (needs a fixed-pool reservation in the
  model_runner budget rather than a per-block charge).
- Choose `N` for the deployment's working set: too small raises conflict
  evictions (shorter hits); `N ≈ num_blocks` is full coverage.

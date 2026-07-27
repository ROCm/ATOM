# CSA prefix boundary-state snapshot — implementation & validation

Branch `feat/csa-prefix-state-snapshot`. Feature flag
`enable_v4_csa_prefix_state_cache` (env `ATOM_V4_CSA_PREFIX_STATE_CACHE=1`),
default **off** (legacy 4-token recompute/replay stays the fallback).

## What it does

On a native HBM prefix-cache hit at a 128-aligned boundary `B`, instead of
replaying the 4 warmup tokens `B-4..B-1` to reseed a fresh CSA compressor ring
(the recompute path, which has the documented SWA-window gap), the producer
**captures** the terminal block's last-4 full compressor ring rows into an
immutable per-block snapshot at prefill time, and a hit **restores** those rows
into the new request's fresh ring — no token replay, no dependency on the
(partly reclaimed) SWA window → bit-exact by construction.

- Capture: `capture_compressor_boundary` (state_writes.py), after
  `update_compressor_states`, writes exactly what the ring got (score fused
  `+ape`), keyed by physical block id.
- Restore: `restore_compressor_boundary`, before `fused_compress_attn`, copies
  the 4 rows into ring slots `(B-4..B-1) % STATE_SIZE`.
- Readiness: `BlockManager.hash_blocks` (prefill publish path — capture ran in
  that forward) marks blocks restorable; decode-finalized blocks (never hashed)
  are never offered as sources; physical-id reuse invalidates.
- Snapshot pool charged in `compute_block_bytes` so `num_blocks` accounts for
  the HBM. Both CSA main and CSA indexer rings are snapshotted; HCA is clean at
  an aligned boundary and is not.

## Validation status

### PROVEN — kernel bit-exactness (GPU)
`tests/test_csa_boundary_snapshot.py` (standalone, `python tests/...`; skips
under bare pytest due to a pre-existing atom.config circular import, same as
`test_csa_prefix_recompute.py`). ALL_PASS:
- capture kernel == pure-python reference
- restore kernel == pure-python reference
- **capture→restore == the producer's `update_compressor_states` ring rows for
  B-4..B-1** (the core invariant — restore reproduces the exact producer state)
- cross-chunk capture accumulation into one physical block

### PROVEN — integration fires (GPU, TP4, DeepSeek-V4-Pro, fp8)
- Server boots stably with the flag on; snapshot pool allocated and logged
  (30 CSA layers × 18770 blocks × 4 rows ≈ 21.5 GiB/rank, fp32 full rows).
- On a prefix hit the restore path executes on every rank with the correct
  source block id and no errors:
  `DSV4 CSA snapshot: restoring 1 seq(s) from boundary sources [2] ...`
- Output is coherent and correct on the Alice prompt (answers "A White Rabbit
  with pink eyes"), i.e. the reseeded ring is not corrupt.

### PROVEN — GSM8K accuracy under heavy restore (GPU, TP4, fp8)
The decisive scale test. `lm_eval` gsm8k as-is does NOT hit the prefix cache
(its per-doc prompts don't share a cacheable prefix here — 0 hits at
num_concurrent 1 and 32), so a custom harness prepends a fixed 8-shot prefix
(1024 tokens = 8 blocks) to every test question and sends them sequentially,
forcing `cached:[1024]` on every question.

| Config | Accuracy (n=150) | Restore fires |
|---|---|---|
| Recompute (flag off), forced hits | 96.67% (145/150) | n/a (4-tok replay) |
| **Snapshot (flag on), forced hits** | **95.33% (143/150)** | 36+, every question |
| Snapshot, stock lm_eval (no hits)  | 94.69% (1319) | 0 |

The snapshot restore fired on every question (`DSV4 CSA snapshot: restoring 1
seq(s)...`, `cached:[1024]`) and accuracy stayed at 95.3% — statistically
indistinguishable from the recompute baseline (Δ = 2 questions, < 1 SE ≈ 2.3%
at n=150, and below the model's own run-to-run nondeterminism). A corrupted CSA
restore would collapse reasoning accuracy, not hold it flat. This validates the
snapshot path end-to-end at scale.

### NOT a valid signal here — greedy token-exact A/B
Greedy (temp=0) output on this build is **not reproducible**: the identical
request run twice on the same server produces different completions (MoE routing
/ atomic-reduction nondeterminism; fp8 KV). Confirmed on the recompute (flag
off) server too, so it is an environment property, not a snapshot defect.
Consequently prefix-ON-vs-OFF token comparison cannot distinguish a correct from
a buggy cache here. (The recompute branch's "token-identical" claim in
`CSA_RECOMPUTE_KNOWN_GAP.md` was not reproducible in this environment.)

### Correctness argument (grounded, not hand-wavy)
The accepted recompute fix seeds the ring for B-4..B-1 by re-forwarding those 4
tokens through `update_compressor_states`. The snapshot seeds the same 4 ring
rows by copy, and the GPU test proves the captured rows equal exactly what
`update_compressor_states` writes for those positions. Therefore the snapshot
produces the same ring as recompute **minus** the recompute's SWA-window
shortfall (B-4's window loses 3 reclaimed positions) — which the snapshot avoids
by construction. So snapshot is at least as correct as the shipped recompute
fix, and strictly better on that gap.

## Remaining / follow-ups
- **HBM**: full fp32 ring rows cost ~21.5 GiB/rank at this cache size. Two easy
  reductions: (a) store bf16 (halves it; a round-trip error, re-check bit-exact
  claim if taken); (b) store only the first-half (`head_dim`) if a read-side
  audit confirms the second half is never used from state for the B-side
  overlap. Neither done yet.
- **Decode-finalized blocks** are not snapshotted (capture is prefill-only), so
  prefixes that were generated (not prompted) are not CSA-restorable. Correct
  but a missed caching opportunity; add decode-path capture if needed.
- **Deterministic logits A/B**: to get a bit-level e2e check, disable the
  nondeterministic kernels (or compare the compressor ring tensor directly after
  a hit vs a full compute). Not run.

# Kimi-K3 gfx1250: the column-sliced AttnRes seam inside K1

Status: design + skeleton (`k1_sliced_seam_layout.py`, `k1_sliced_seam_gfx1250.py`),
not wired. The standalone kernel it reuses is `attn_res_sliced_gfx1250`
(`ATOM_KIMI_MONO_ATTN_RES_SLICED`).

## Where we are

| seam path (40 rows, serving order) | relative cost |
|---|---|
| external `attn_res_gfx1250` + K1 (`_xs`) | baseline |
| in-kernel K1 seam, 56 CTAs, device mailboxes, fence-free | about 1.2 us slower |
| external `attn_res_sliced_gfx1250` (4/8 CTAs) + K1 | about 0.5 us faster per seam (one rep) |

The in-kernel seam loses because it crosses CTAs at device scope four times
(slice partials to the gate CTA, gate weights back, mean squares, x ready), at
about 0.8 us a hop, from only 56 CTAs. The sliced kernel crosses at cluster
(SE) scope twice, about 0.3 to 0.7 us each, with no tags or fences, from up to
256 CTAs.

## Design

**Clusters.** With the TDM projection, K1 already launches in clusters of
`TDM_CLUSTER = 4` (token-slab multicast). The seam uses those clusters: rank
`bid % 4` owns columns `[1792 rank, 1792 (rank + 1))`, home `bid // 4` (64 homes)
owns token `home`. One pass covers 64 tokens, so every K1 width is a single
pass. The attention CTAs (homes 0 to 13) take part; they wait for projections
anyway.

**When.** Inside `gemv_tdm.project`'s `prologue`: the first `DEPTH` (BF16 3,
FP8 5) weight slabs of every projection CTA are already in flight, so about
200 CTAs times 3 to 5 slabs of 16 KiB stream while the seam runs. The attention
CTAs call the same seam code without a projection around it.

**Exchanges.** Inside a cluster only, as in the standalone kernel: slice sums
to scratch at SCOPE_SE, `s_wait_storecnt(0)`, cluster barrier, SE-scope reads.
The records need no generation tags: the kernel boundary orders launches, and
within a launch each record is written once. Every CTA of a cluster runs the
same number of passes (homes past `tokens` run none), so the cluster barriers
stay matched; the projection's own per-slab cluster barriers come after.

**LDS.** `smem.red` (2048 FP32), outside the role union the TDM copies fill
during the prologue. The largest Gram build needs 1376 words (`K1SeamPlan.lds_words`).

**Writes.** Each thread stores only its own columns: x (BF16, scratch `x`),
`prefix + delta` back into `prefix` in place, and the `write_idx` block row.
Then one device-scope ready pair per CTA: `x_ready[rank * 64 + home] =
(0, generation)`, after `s_wait_storecnt(0)` and a CTA barrier
(`publish_slice`).

**Per-slab polling.** The projection walks 14 slabs of 512 columns. Before
the token copy of the first slab that reads a rank's columns (slabs 0, 3, 7, 10
for ranks 0 to 3; `K1SeamPlan.polls`), the copying wave polls that rank's pair
of every home holding a token, a lane a home, all in flight at once
(`gate_slab`). The token copy reads x at device scope (`x_cache_policy=CM_DEV`,
as `k2_tail` already does), so there is no acquire fence. Slab 0 starts as soon
as rank 0's slices are out instead of after the slowest CTA, and the other
ranks' polls overlap slabs 0 to 9 of compute.

**Hook needed in `gemv_tdm`.** One optional `x_gate(slab)` callback, emitted
on the token-copy wave right before each `issue_tokens(slab)`, in both the
unrolled and the rolled (`_rolled_token_step`) slab loops. `slab` is a Python
int in the unrolled loop and a runtime index in the rolled one; `gate_slab`
handles both. When `x_gate` is None, the emitted code is unchanged.

**Numerics.** Same as `attn_res_sliced_gfx1250`. The score weight is
`ares_nw * ares_qk` formed per column in FP32, equal to the runner's
precomputed `score_weight` (`norm.weight.float() * proj.weight.float()`)
when both weights are BF16, since a BF16 product is exact in FP32, and the output gain is
`in_nw`. The Gram variant removes the second exchange (about 0.3 to 0.4 us),
but its error grows with the mix's cancellation ratio (see the
`build_attn_res_sliced` docstring), so it stays opt-in until that ratio is
measured on real activations.

## Estimate (40 rows, per KDA layer, unmeasured)

Against the external pair (`attn_res` kernel, kernel boundary, K1):

- Kernel boundary and the separate seam kernel's ramp and tail are gone:
  about 1 us.
- The seam overlaps the first `DEPTH` weight slabs: about 0.3 to 0.6 us.
- Per-slab polling hides the slices' finish skew (standalone stamps: median
  3.5 us, max 4.0 us): about 0.2 to 0.4 us.
- Added: one device hop from the last slice to the first token copy, about
  0.8 us, partly hidden by per-slab polling.

Net: about 1 to 2 us better than the external pair, versus 1.2 us worse for
today's in-kernel seam. The Gram variant would add about 0.3 to 0.4 us. The
2 us target stays uncertain until it is measured.

## TP8 note

At TP8, `OWN_COLS = 896` is exactly the 8-CTA slice width. If o_proj
reduce-scatters to column owners, each rank holds one 896-column slice of
every token. The seam's slice sums would then cross ranks instead of a
cluster's CTAs; that is step 2's design, not this one's.

## Order of work once GPUs are back

1. Accuracy and two-rep serving order for the standalone 4/8-CTA kernels
   (`ATOM_KIMI_MONO_ATTN_RES_SLICED=4` / `8`).
2. Extract `seam_pass` from `attn_res_sliced_gfx1250` (bitwise equal to
   the standalone kernel), add the `x_gate` hook, and wire it behind a K1 build
   key flag.
3. Bitwise check against the external sliced seam, then serving order at rows
   8/24/40/48 against the external pair, two reps.

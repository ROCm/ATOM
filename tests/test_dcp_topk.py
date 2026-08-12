# SPDX-License-Identifier: MIT
"""DCP decode top-k: deterministic merge, and the candidate exchange built on it.

Two layers of the same change, both against the shipped implementations in
``atom/model_ops/dcp_ops.py``:

  * ``dcp_stable_topk``          -- the merge itself: given candidates, is the
    kept set a pure FUNCTION of (scores, gids, k)?
  * ``dcp_pack_topk_candidates`` + the merge -- the substitution: does
    "local top-k, exchange W*topk (score, gid) pairs, merge" reproduce the
    global top-k it replaced?

Why determinism is the property under test rather than a nicety: each rank runs
the merge independently, so an ambiguous choice resolved differently on two
ranks breaks the disjoint-partition premise ``cp_lse_ag_out_rs`` needs.
Ambiguity can only arise among candidates whose score exactly equals the
selection threshold, which is why the tie-heavy cases below matter more than the
random-float ones -- on real workloads only ~0.06% of rows have such a tie, so a
random-data-only test would almost never exercise the path that matters.

Note what the exchange does NOT promise. If a rank's own top-k boundary lands on
a tie, aiter's kernel may keep either tied token, so the exchanged candidate set
can differ from a gid-stable local top-k and the merged answer can be a
DIFFERENT valid top-k. It is still valid (same score multiset) and still
identical on every rank -- every rank merges the same gathered buffer with the
same total order -- and cross-rank agreement is what the partition needs. The
assertions below are written to that weaker, real contract.
"""

import pytest
import torch

try:
    from aiter.ops.topk import top_k_per_row_decode

    from atom.model_ops.dcp_ops import (
        DCP_TOPK_TIE_CAP,
        dcp_pack_topk_candidates,
        dcp_stable_topk,
    )
except ImportError as _e:  # triton/aiter absent on a CPU-only runner
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)

DEV = "cuda"
TOPK = 2048
# The decode indexer sizes its local plane from max_model_len, not from the live
# context, so every rank's shard is padded with uninitialised memory.
MAX_MODEL_LEN = 1 << 20


def _make_case(rows, n, seed, tie_frac=0.0):
    """tie_frac: how much of the score range is collapsed, to force exact ties."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    sc = torch.randn(rows, n, generator=g, device=DEV, dtype=torch.float32)
    if tie_frac > 0:
        levels = max(2, int(n * (1 - tie_frac)))
        sc = (sc * levels).round() / levels
    gid = torch.randperm(n, generator=g, device=DEV).to(torch.int32)
    gid = gid.unsqueeze(0).repeat(rows, 1).contiguous()
    return sc.contiguous(), gid


def _stable_reference(scores, gids, k):
    """Exact stable top-k: score desc, ties broken by smallest gid."""
    # Sort by gid ascending first; a STABLE sort by -score then keeps, within
    # each equal-score group, the gid-ascending order -- i.e. the total order.
    order = torch.argsort(gids.to(torch.int64), dim=-1, stable=True)
    sc = torch.gather(scores, 1, order)
    gd = torch.gather(gids, 1, order)
    idx = torch.argsort(-sc.double(), dim=-1, stable=True)[:, :k]
    return torch.gather(gd, 1, idx)


# ────────────────────────────────────────────────── dcp_stable_topk in itself ──


@pytest.mark.parametrize(
    "rows, n, tie_frac",
    # tie_frac collapses the score range; the resulting tie set at the selection
    # threshold must stay under DCP_TOPK_TIE_CAP or the kernel legitimately gives
    # up (see test_threshold_tie_cap_boundary). Measured max tie set: 0.9 -> 6,
    # 0.99 -> 26, 0.999 -> 249, i.e. 0.999 sits 7 short of the 256 cap and would
    # start failing on any RNG change. Keep the cases well below it.
    [
        (8, 16384, 0.0),
        (8, 16384, 0.99),
        (32, 16384, 0.9),
        (16, 16384, 0.99),
        (4, 8192, 0.0),
    ],
)
def test_stable_topk_matches_reference(rows, n, tie_frac):
    sc, gd = _make_case(rows, n, seed=rows * 31 + n, tie_frac=tie_frac)
    got, ovf = dcp_stable_topk(sc, gd, TOPK)
    ref = _stable_reference(sc, gd, TOPK)

    # Ties at the threshold are capped (DCP_TOPK_TIE_CAP); above the cap the
    # kernel drops the extras and flags it. These cases must stay under it.
    assert int(ovf.sum()) == 0, "tie buffer overflowed; case exceeds DCP_TOPK_TIE_CAP"
    assert torch.equal(got.sort(-1).values, ref.sort(-1).values.int()), (
        f"kept set differs from the exact stable reference "
        f"(rows={rows} n={n} tie_frac={tie_frac})"
    )


@pytest.mark.parametrize(
    "delta, overflows", [(-56, False), (+44, True)], ids=["under_cap", "over_cap"]
)
def test_threshold_tie_cap_boundary(delta, overflows):
    """The documented limit: more than DCP_TOPK_TIE_CAP ties must be FLAGGED.

    The kernel buffers the tied gids in a fixed CAP-wide scratch. Under the cap
    the selection is exact; over it the extras are dropped and that row's top-k
    is genuinely wrong -- `overflow` is the only signal, and the hot path cannot
    read it back (a D2H copy would break CUDAGraph capture). So what is pinned
    here is "flags rather than hides", and that the boundary sits exactly where
    DCP_TOPK_TIE_CAP says it does. The over-cap output is deliberately NOT
    asserted: it is documented as wrong.

    Sizes are derived from the constant so the test cannot go stale if it moves.
    """
    rows, n = 4, 8192
    n_tied = DCP_TOPK_TIE_CAP + delta
    need = 148  # how many of the tied candidates the selection has to take
    n_strict = TOPK - need
    assert need < n_tied, "case must force a choice among the tied candidates"

    scores = torch.zeros(rows, n, device=DEV)
    scores[:, :n_strict] = 10.0  # strict winners
    scores[:, n_strict : n_strict + n_tied] = 5.0  # all exactly at the threshold
    scores = scores.contiguous()
    g = torch.Generator(device=DEV).manual_seed(17)
    gid = torch.randperm(n, generator=g, device=DEV).to(torch.int32)
    gid = gid.unsqueeze(0).repeat(rows, 1).contiguous()

    got, ovf = dcp_stable_topk(scores, gid, TOPK)

    assert int(ovf.sum()) == (rows if overflows else 0)
    if not overflows:
        ref = _stable_reference(scores, gid, TOPK)
        assert torch.equal(
            got.sort(-1).values, ref.sort(-1).values.int()
        ), "tie-break must take the `need` smallest gids among the tied set"


def test_stable_topk_is_deterministic():
    """Same input, 20 merges: the ranks only agree if this is a function."""
    sc, gd = _make_case(32, 16384, seed=7, tie_frac=0.99)
    first = dcp_stable_topk(sc, gd, TOPK)[0].clone()
    for i in range(20):
        again = dcp_stable_topk(sc, gd, TOPK)[0]
        assert torch.equal(again, first), f"merge {i} differs from the first"


def test_stable_topk_is_permutation_invariant():
    """Candidate ORDER must not matter -- ranks receive them in gather order."""
    sc, gd = _make_case(16, 16384, seed=11, tie_frac=0.99)
    a = dcp_stable_topk(sc, gd, TOPK)[0].sort(-1).values
    perm = torch.randperm(16384, device=sc.device)
    b = dcp_stable_topk(sc[:, perm].contiguous(), gd[:, perm].contiguous(), TOPK)[0]
    assert torch.equal(a, b.sort(-1).values), "kept set changed when candidates moved"


@pytest.mark.parametrize("rows, W, k_loc, tie_frac", [
    (8, 8, 2048, 0.0),
    (8, 8, 2048, 0.99),
    (32, 4, 4096, 0.9),
])
def test_stable_topk_accepts_noncontiguous_3d_gids(rows, W, k_loc, tie_frac):
    """Chapter-9 optimization: gids passed as a 3D AllGather VIEW (no
    `.contiguous()`) must give the bit-identical result as the plain 2D
    contiguous path -- this is the whole point of the g_s1/g_k_loc kernel
    parameters: same answer, one fewer materializing copy.

    Shape mimics the real candidate-exchange buffer exactly: recv is
    `[W, 2, rows, k_loc]` contiguous (from an AllGather along dim 0), plane 1
    is the gid plane, and the production call site does
    `recv[:, 1].permute(1, 0, 2)` -- reproduced here without the
    `torch.stack`+`.contiguous()` a naive test would otherwise reach for.
    """
    n = W * k_loc
    sc, gd_2d = _make_case(rows, n, seed=rows * 1009 + W * 97 + k_loc, tie_frac=tie_frac)

    # Build recv exactly as the AllGather does: [W, 2, rows, k_loc] contiguous,
    # where recv[w, 1, row, j] == gd_2d[row, w*k_loc + j] (the (score, gid) pack
    # order the production code relies on).
    gd_3d_src = gd_2d.view(rows, W, k_loc).permute(1, 0, 2).contiguous()  # [W, rows, k_loc]
    recv = torch.empty(W, 2, rows, k_loc, dtype=torch.int32, device=DEV)
    recv[:, 1].copy_(gd_3d_src)
    recv[:, 0].copy_(0)  # score plane unused by this test

    gathered_gid = recv[:, 1].permute(1, 0, 2)  # [rows, W, k_loc], NOT contiguous
    assert not gathered_gid.is_contiguous()
    assert gathered_gid.stride(2) == 1

    got_3d, ovf_3d = dcp_stable_topk(sc, gathered_gid, TOPK if n >= TOPK else n)
    got_2d, ovf_2d = dcp_stable_topk(sc, gd_2d, TOPK if n >= TOPK else n)
    assert torch.equal(ovf_3d, ovf_2d)
    assert torch.equal(got_3d, got_2d), (
        f"3D non-contiguous gid path diverged from the 2D contiguous baseline "
        f"(rows={rows} W={W} k_loc={k_loc} tie_frac={tie_frac})"
    )


# ───────────────────────────────────────── pack + merge == global top-k ──


def _build_gathered(global_logits, ctx, world, k=TOPK):
    """Everything up to and including the all-gather, for all ranks.

    Single process: the all-gather is a concat, and each rank's local shard is
    carved out of the global plane by the round-robin rule (position p lives on
    rank p % W, at local index p // W).
    """
    rows = global_logits.shape[0]
    dev = global_logits.device
    l_max = (MAX_MODEL_LEN + world - 1) // world
    sc_planes, gid_planes = [], []

    for r in range(world):
        local_ctx = (ctx - r + world - 1) // world  # #positions p<ctx, p%W==r
        # torch.empty in the real path: everything past local_ctx is garbage, so
        # seed it with garbage that would WIN if it were ever read.
        local = torch.rand(rows, l_max, device=dev, dtype=torch.float32) * 1e4
        if local_ctx > 0:
            local[:, :local_ctx] = global_logits[:, r:ctx:world]
        lens = torch.full((rows,), local_ctx, dtype=torch.int32, device=dev)

        idx = torch.empty(rows, k, dtype=torch.int32, device=dev)
        top_k_per_row_decode(
            local, 1, lens, idx, rows, local.stride(0), local.stride(1), k
        )
        send = torch.empty(2, rows, k, dtype=torch.float32, device=dev)
        dcp_pack_topk_candidates(local, idx, lens, r, world, send)
        # all_gather puts rank on dim 0; the merge wants it on the candidate dim
        sc_planes.append(send[0].clone())
        gid_planes.append(send.view(torch.int32)[1].clone())

    return (
        torch.cat(sc_planes, dim=1).contiguous(),
        torch.cat(gid_planes, dim=1).contiguous(),
    )


def _simulate(global_logits, ctx, world, k=TOPK):
    sc_all, gid_all = _build_gathered(global_logits, ctx, world, k)
    return dcp_stable_topk(sc_all, gid_all, k)


def _global_reference(global_logits, ctx, k=TOPK):
    """Exact gid-stable global top-k: score desc, ties by smallest position."""
    sc = global_logits[:, :ctx].double()
    n_keep = min(k, ctx)
    # stable argsort on -score keeps ascending position order within a tie
    idx = torch.argsort(-sc, dim=-1, stable=True)[:, :n_keep]
    return idx.to(torch.int32)


def _make_logits(rows, ctx, seed, tie_frac=0.0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    gl = torch.randn(rows, ctx, generator=g, device=DEV, dtype=torch.float32)
    if tie_frac > 0:
        levels = max(2, int(ctx * (1 - tie_frac)))
        gl = (gl * levels).round() / levels
    return gl


@pytest.mark.parametrize(
    "name, rows, ctx, world, tie_frac, seed",
    [
        ("long ctx", 8, 131072, 8, 0.0, 1),
        ("ctx = W*topk", 8, 16384, 8, 0.0, 2),
        ("ctx not div by W", 8, 131071, 8, 0.0, 3),
        ("heavy ties", 8, 131072, 8, 0.99, 4),
        # ctx < topk: every candidate is selected and the local top-k returns
        # fewer ids than k_loc -- the padding path short prompts hit.
        ("ctx < topk (padding)", 8, 1000, 8, 0.0, 5),
        ("ctx just over topk", 8, 2500, 8, 0.0, 6),
        ("W=2", 8, 65536, 2, 0.0, 7),
        ("large batch", 64, 32768, 8, 0.0, 8),
    ],
)
def test_candidate_exchange_reproduces_global_topk(
    name, rows, ctx, world, tie_frac, seed
):
    gl = _make_logits(rows, ctx, seed, tie_frac)
    out, _ = _simulate(gl, ctx, world)
    ref = _global_reference(gl, ctx)
    n_keep = min(TOPK, ctx)
    got = out[:, :n_keep]

    assert bool(((got >= 0) & (got < ctx)).all()), f"[{name}] id out of range"
    for row in got.cpu():
        assert len(set(row.tolist())) == n_keep, f"[{name}] duplicate id in a row"
    # -1 padding only where candidates genuinely ran out
    assert int((out < 0).sum()) == rows * (TOPK - n_keep), f"[{name}] bad padding"

    # The invariant that survives a local boundary tie reshuffling WHICH of two
    # equal-scored tokens was exchanged: the selected score multiset is still
    # exactly the global top-k's.
    sc_got = torch.gather(gl, 1, got.long()).sort(-1).values
    sc_ref = torch.gather(gl, 1, ref.long()).sort(-1).values
    assert torch.equal(sc_got, sc_ref), f"[{name}] selected scores are not the top-k"

    # With no tie AT the global threshold there is nothing to choose, so the ids
    # themselves must match the gid-stable reference.
    thr = gl[:, :ctx].topk(n_keep, dim=-1).values[:, -1:]
    if int((gl[:, :ctx] == thr).sum(-1).max()) == 1:
        assert torch.equal(
            got.sort(-1).values, ref.sort(-1).values
        ), f"[{name}] ids differ from the reference with no threshold tie"


def test_merge_agrees_across_ranks_on_a_fixed_buffer():
    """The property the partition actually needs.

    NOT "the pipeline returns the same answer every run" -- it cannot, because
    aiter's local top-k picks arbitrarily among tied candidates (measured: with
    ties, 18/20 repeats swap one id on some row; with no ties, 0/20). What must
    hold is that every rank merging the SAME gathered buffer returns the same
    answer, which is what all-gather actually hands them.
    """
    gl = _make_logits(16, 65536, seed=9, tie_frac=0.99)
    sc, gid = _build_gathered(gl, 65536, 8)
    first = dcp_stable_topk(sc, gid, TOPK)[0].clone()
    for i in range(20):
        assert torch.equal(
            dcp_stable_topk(sc, gid, TOPK)[0], first
        ), f"merge {i} disagrees -- ranks would build overlapping candidate sets"


def test_reruns_stay_valid_even_when_the_set_shifts():
    """End to end, with ties: the chosen set may move, but never off the top-k."""
    gl = _make_logits(16, 65536, seed=9, tie_frac=0.99)
    ref_sc = torch.gather(gl, 1, _global_reference(gl, 65536).long()).sort(-1).values
    for i in range(10):
        out, _ = _simulate(gl, 65536, 8)
        # Range-check BEFORE the gather. An out-of-range index would trip a
        # device-side assert, and on ROCm that leaves the HIP context unusable --
        # every later GPU test in the same process fails with an error that
        # points nowhere near here. (Clamping instead would hide the regression.)
        assert bool(((out >= 0) & (out < 65536)).all()), f"rerun {i}: id out of range"
        got_sc = torch.gather(gl, 1, out.long()).sort(-1).values
        assert torch.equal(got_sc, ref_sc), f"rerun {i} selected outside the top-k"

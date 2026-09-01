# SPDX-License-Identifier: MIT
"""DCP decode candidate exchange: does it reproduce the global top-k?

Tests the substitution ``dcp_pack_topk_candidates`` + merge performs: does
"local top-k, exchange W*topk (score, gid) pairs, merge" reproduce the global
top-k it replaced, and is the merged set a pure FUNCTION of the gathered buffer?

The merge is aiter's ``top_k_per_row_decode(..., stable=True)`` plus a gid
gather, mirrored here by ``_merge`` from `_dcp_decode_candidate_exchange`.
(Until 2026-08-13 it was a hand-written ``dcp_stable_topk`` in
``atom/model_ops/dcp_ops.py``; that kernel and its dedicated tests were removed
once aiter's stable path proved both faster and sufficient.)

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
        dcp_global_pos,
        dcp_local_context_lens,
        dcp_merge_candidates,
        dcp_pack_topk_candidates,
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


# ───────────────────────────────────────────── pack kernel == reference ──
#
# `dcp_pack_topk_candidates` became a single Triton kernel on 2026-08-24 (it was
# 19 eager ops; see DCP_Further_Optimization.md ch.6). The end-to-end tests below
# only ever run it at S == 1, so pin it against the eager formulation directly,
# across interleave sizes and including the padding / out-of-range slots.


def _pack_reference(local_logits, local_idx, local_lens, rank, world, s_itl):
    """The pre-2026-08-24 eager implementation, verbatim."""
    rows, _k = local_idx.shape
    valid = (local_idx >= 0) & (local_idx < local_lens.view(rows, 1))
    safe = torch.where(valid, local_idx, torch.zeros_like(local_idx))
    sc = torch.gather(local_logits, 1, safe.to(torch.int64))
    gid = torch.where(
        valid,
        dcp_global_pos(local_idx, rank, world, s_itl),
        torch.full_like(local_idx, -1),
    )
    return (
        torch.where(valid, sc, torch.full_like(sc, -float("inf"))),
        gid,
    )


@pytest.mark.parametrize("s_itl", [1, 4, 16])
@pytest.mark.parametrize("rank, world", [(0, 8), (3, 8), (7, 8), (1, 2)])
@pytest.mark.parametrize("rows, k, l_max", [(8, TOPK, 4096), (1, 37, 64)])
def test_pack_matches_eager_reference(s_itl, rank, world, rows, k, l_max):
    g = torch.Generator(device=DEV).manual_seed(1000 + rank * 17 + s_itl)
    logits = torch.randn(rows, l_max, generator=g, device=DEV, dtype=torch.float32)
    # Mix in-range ids with the two padding conventions the bound check exists
    # for: negative slots and ids past the live local length.
    idx = torch.randint(
        -3, l_max, (rows, k), generator=g, device=DEV, dtype=torch.int32
    )
    lens = torch.randint(0, l_max, (rows,), generator=g, device=DEV, dtype=torch.int32)

    out = torch.empty(2, rows, k, dtype=torch.float32, device=DEV)
    dcp_pack_topk_candidates(logits, idx, lens, rank, world, out, s_itl)
    ref_sc, ref_gid = _pack_reference(logits, idx, lens, rank, world, s_itl)

    assert torch.equal(out[0], ref_sc), "score plane differs from the eager reference"
    assert torch.equal(
        out.view(torch.int32)[1], ref_gid
    ), "gid plane differs from the eager reference"


# ───────────────────────────────────────── pack + merge == global top-k ──


def _build_gathered(global_logits, ctx, world, k=TOPK):
    """Everything up to and including the all-gather, for all ranks.

    Single process: the all-gather is a concat, and each rank's local shard is
    carved out of the global plane by the round-robin rule (position p lives on
    rank p % W, at local index p // W).

    Returns the buffer in the layout production hands the merge: the int32 view
    of ``[W, 2, rows, k]`` (rank outermost, plane 0 score / plane 1 gid).
    """
    rows = global_logits.shape[0]
    dev = global_logits.device
    l_max = (MAX_MODEL_LEN + world - 1) // world
    sends = []

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
        sends.append(send.clone())

    return torch.stack(sends, dim=0).view(torch.int32)


def _merge(recv, k=TOPK):
    """The production merge -- the real one, not a copy of it.

    This used to be a hand-written mirror of the block inlined in
    `dcp_decode_candidate_exchange`, and the two silently diverged the moment
    that function was moved between modules (the fused gid map was replaced by
    a reshape+gather and no test noticed). Call the shared helper instead so a
    divergence is impossible by construction.
    """
    rows = recv.shape[2]
    out = torch.empty(rows, k, dtype=torch.int32, device=recv.device)
    dcp_merge_candidates(recv, out)
    return out


def _simulate(global_logits, ctx, world, k=TOPK):
    return _merge(_build_gathered(global_logits, ctx, world, k), k), None


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

    # Assert on the SET, not the layout. aiter's stable top-k returns candidate
    # ARRAY-index order, so when ctx < topk the real ids are interleaved with the
    # (-inf, -1) padding across the full width instead of packed into the first
    # n_keep columns. Nothing downstream needs the packed layout -- the filter
    # kernels scan all NUM_TOPK_TOKENS columns and gate on `tok >= 0` -- so
    # asserting it would only re-freeze an obsolete convention.
    valid = out >= 0
    assert int(valid.sum()) == rows * n_keep, f"[{name}] wrong number of valid ids"
    assert bool((out[valid] < ctx).all()), f"[{name}] id out of range"
    for r in range(rows):
        row = out[r][valid[r]].cpu().tolist()
        assert len(set(row)) == n_keep, f"[{name}] duplicate id in a row"
    # Compact the valid ids per row so the score comparison below is layout-free.
    got = torch.stack([out[r][valid[r]] for r in range(rows)])

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
    recv = _build_gathered(gl, 65536, 8)
    first = _merge(recv).clone()
    for i in range(20):
        assert torch.equal(
            _merge(recv), first
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


# ─────────────────────────────── filter: disjoint union == global top-k ──
#
# The gap this closes: every test above stops at `topk_indices`, so the
# `triton_filter_and_convert_dcp_index` kernels -- which turn that into each
# rank's owned slot list -- had NO coverage. A 2026-08-13 regression lived
# exactly there: the count/compact kernels masked columns on `indice_id <
# g_kv_len`, which silently assumed the merge packs valid ids into the first
# min(ctx, topk) columns. When ctx < topk and aiter's stable top-k interleaves
# ids with (-inf, -1) padding, 875 of 1000 candidates were dropped -- invisible
# to a test that only inspects topk_indices.


def _filter_owned(token_indices, ctx, rank, world, block_size=16):
    """Run the production filter for one rank; return its owned GLOBAL positions.

    The kernel emits physical slots, so this rebuilds the slot->global mapping
    with an identity block_table, letting the test assert on positions.
    """
    from atom.model_ops.dcp_ops import triton_filter_and_convert_dcp_index

    rows = token_indices.shape[0]
    dev = token_indices.device
    vbs = block_size * world
    n_blocks = (ctx + vbs - 1) // vbs
    block_table = torch.arange(rows * n_blocks, dtype=torch.int32, device=dev).reshape(
        rows, n_blocks
    )
    qo_indptr = torch.arange(rows + 1, dtype=torch.int32, device=dev)
    g_kv_indptr = torch.arange(rows + 1, dtype=torch.int32, device=dev) * ctx
    out_kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
    owned_counts = torch.zeros(rows, dtype=torch.int32, device=dev)
    out = torch.zeros(rows * TOPK, dtype=torch.int32, device=dev)

    triton_filter_and_convert_dcp_index(
        qo_indptr,
        g_kv_indptr,
        block_table,
        token_indices,
        rank,
        world,
        block_size,
        out_kv_indptr,
        owned_counts,
        NUM_TOPK_TOKENS=TOPK,
        out=out,
    )

    # slot = block_table[req, g // vbs] * block_size + (g % vbs) // W, and the
    # block_table is the identity, so invert it back to g.
    per_row = []
    for r in range(rows):
        s, e = int(out_kv_indptr[r]), int(out_kv_indptr[r + 1])
        slots = out[s:e]
        blk, off = slots // block_size, slots % block_size
        g = (blk - r * n_blocks) * vbs + off * world + rank
        per_row.append(g)
    return per_row


@pytest.mark.parametrize(
    "name, rows, ctx, world",
    [
        ("ctx < topk (the regression case)", 4, 1000, 8),
        ("ctx just over topk", 4, 2500, 8),
        ("long ctx", 4, 32768, 8),
        ("W=2", 4, 1000, 2),
    ],
)
def test_filter_partitions_topk_disjointly(name, rows, ctx, world):
    gl = _make_logits(rows, ctx, seed=11)
    out, _ = _simulate(gl, ctx, world)
    expected = [set(out[r][out[r] >= 0].cpu().tolist()) for r in range(rows)]

    union = [set() for _ in range(rows)]
    for rank in range(world):
        for r, g in enumerate(_filter_owned(out, ctx, rank, world)):
            ids = set(g.cpu().tolist())
            assert all(
                i % world == rank for i in ids
            ), f"[{name}] rank {rank} kept a foreign position"
            assert not (union[r] & ids), f"[{name}] rank {rank} overlaps another rank"
            union[r] |= ids

    for r in range(rows):
        assert union[r] == expected[r], (
            f"[{name}] row {r}: union of the ranks' kept sets != the global top-k "
            f"(missing {len(expected[r] - union[r])}, extra {len(union[r] - expected[r])})"
        )


# ─────────────────────────────────── prefill filter: layout independence ──
#
# `triton_filter_and_convert_dcp_index_prefill` carried the same truncation
# (`col_id < kv_len`) as the decode twin. It never broke in production because
# aiter's prefill kernels short-circuit `row_len <= k` and emit every candidate
# in order with the tail filled to -1 -- on both the one-block (stable=True,
# GLM-5.2) and multi-block (stable=False, V3.2) paths. But that is aiter's
# layout contract, not an invariant of our kernel, and the decode path proved
# what happens when such a contract silently changes.
#
# So these tests do NOT assert the layout. They feed the SAME candidate set in
# two placements -- compact (what aiter emits today) and scattered across the
# full 2048 width (what broke decode) -- and require an identical partition from
# both. That is the property we actually depend on.


def _build_prefill_topk(kv_lens, bases, layout, k=TOPK, seed=7):
    """Emulate a `top_k_per_row_prefill` output plane: FLAT KV indices, -1 pad.

    kv_len <= k mirrors the short-circuit (every candidate kept, ascending);
    kv_len > k picks a k-subset, which is what the real radix top-k returns.
    """
    gen = torch.Generator().manual_seed(seed)
    ti = torch.full((len(kv_lens), k), -1, dtype=torch.int32)
    for row, (kv_len, base) in enumerate(zip(kv_lens, bases)):
        n = min(kv_len, k)
        if kv_len <= k:
            sel = torch.arange(n, dtype=torch.int32)
        else:
            sel = (
                torch.randperm(kv_len, generator=gen)[:k].sort().values.to(torch.int32)
            )
        cols = (
            torch.arange(n)
            if layout == "compact"
            else torch.randperm(k, generator=gen)[:n].sort().values
        )
        ti[row, cols] = sel + base
    return ti.to(DEV)


def _filter_owned_prefill(
    ti, kv_lens, token_req, cu_seqlens_k, ctx_max, rank, world, block_size=16
):
    """Run the production prefill filter for one rank; return owned positions."""
    from atom.model_ops.dcp_ops import triton_filter_and_convert_dcp_index_prefill

    rows = ti.shape[0]
    vbs = block_size * world
    n_blocks = (ctx_max + vbs - 1) // vbs
    num_req = cu_seqlens_k.numel() - 1
    block_table = torch.arange(
        num_req * n_blocks, dtype=torch.int32, device=DEV
    ).reshape(num_req, n_blocks)
    dsa_kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=DEV)
    dsa_kv_indptr[1:] = torch.tensor(kv_lens, device=DEV).cumsum(0).to(torch.int32)
    out_kv_indptr = torch.zeros(rows + 1, dtype=torch.int32, device=DEV)
    owned_counts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    out = torch.zeros(rows * TOPK, dtype=torch.int32, device=DEV)

    triton_filter_and_convert_dcp_index_prefill(
        dsa_kv_indptr,
        token_req,
        ti,
        cu_seqlens_k,
        block_table,
        rank,
        world,
        block_size,
        out_kv_indptr,
        owned_counts,
        NUM_TOPK_TOKENS=TOPK,
        out=out,
    )

    per_row = []
    for t in range(rows):
        s = int(out_kv_indptr[t])
        # The persistent metadata reserves one dummy slot for a rank-local
        # zero-length row. Only the true owned count belongs to the partition
        # being checked here; the dummy is neutralized by the DCP LSE merge.
        e = s + int(owned_counts[t])
        slots = out[s:e]
        blk, off = slots // block_size, slots % block_size
        req = int(token_req[t])
        per_row.append((blk - req * n_blocks) * vbs + off * world + rank)
    return per_row


@pytest.mark.parametrize("layout", ["compact", "scattered"])
@pytest.mark.parametrize(
    "name, ctxs, token_req, kv_lens, world",
    [
        # Two requests so the flat->position mapping (`indice - cu_seqlens_k[req]`)
        # is exercised with a non-zero base, and short rows so kv_len << topk.
        ("short rows, 2 reqs", [1000, 300], [0, 0, 0, 1, 1], [1, 17, 1000, 1, 300], 8),
        ("rows longer than topk", [5000], [0, 0], [3000, 5000], 8),
        ("W=2", [1000], [0], [999], 2),
    ],
)
def test_prefill_filter_is_layout_independent(
    name, ctxs, token_req, kv_lens, world, layout
):
    cu = [0]
    for c in ctxs:
        cu.append(cu[-1] + c)
    cu_seqlens_k = torch.tensor(cu, dtype=torch.int32, device=DEV)
    bases = [int(cu_seqlens_k[r]) for r in token_req]
    treq = torch.tensor(token_req, dtype=torch.int32, device=DEV)
    ti = _build_prefill_topk(kv_lens, bases, layout)

    # Ground truth straight off the input plane, so it holds for either layout.
    expected = [
        set((ti[t][ti[t] >= 0] - bases[t]).cpu().tolist()) for t in range(len(kv_lens))
    ]

    union = [set() for _ in kv_lens]
    for rank in range(world):
        owned = _filter_owned_prefill(
            ti, kv_lens, treq, cu_seqlens_k, max(ctxs), rank, world
        )
        for t, pos in enumerate(owned):
            ids = set(pos.cpu().tolist())
            assert all(
                i % world == rank for i in ids
            ), f"[{name}/{layout}] rank {rank} token {t} kept a foreign position"
            assert not (
                union[t] & ids
            ), f"[{name}/{layout}] rank {rank} token {t} overlaps another rank"
            union[t] |= ids

    for t in range(len(kv_lens)):
        assert union[t] == expected[t], (
            f"[{name}/{layout}] token {t}: union of the ranks' kept sets != the "
            f"candidate set (missing {len(expected[t] - union[t])}, "
            f"extra {len(union[t] - expected[t])})"
        )


# ---------------------------------------------------------------------------
# Regression guards for the fusions themselves.
#
# The two checks below exist because both existing safety nets are blind to a
# production path that stops *using* a fused helper: the correctness tests drove
# a hand-written copy of the merge, and bench_dcp_indexer_fuse.py times the
# primitives directly. When `dcp_decode_candidate_exchange` was moved between
# modules its fused local_ctx read and fused gid map were replaced by the old
# eager code, and every test and the benchmark still passed.
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, ctx, published=None):
        self.context_lens = ctx
        if published is not None:
            self.dcp_local_context_lens = published


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 16])
def test_local_context_lens_prefers_published_buffer(world, interleave):
    """The published host buffer must be returned as-is, not recomputed.

    Identity, not equality: returning an equal-but-recomputed tensor is exactly
    the regression this guards (8 elementwise kernels per full-index layer).
    """
    rows = 37
    ctx = torch.randint(1, 100000, (rows,), dtype=torch.int32, device=DEV)
    published = torch.arange(rows, dtype=torch.int32, device=DEV)
    got = dcp_local_context_lens(_FakeMeta(ctx, published), 0, world, interleave, rows)
    assert got is published


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 16])
def test_local_context_lens_fallback_matches_reference(world, interleave):
    """No published buffer (or a stale one) -> derive, and match the split."""
    rows = 37
    ctx = torch.randint(1, 100000, (rows,), dtype=torch.int32, device=DEV)
    ref = torch.stack(
        [
            torch.tensor(
                sum(1 for p in range(int(c)) if (p // interleave) % world == 0),
                dtype=torch.int32,
                device=DEV,
            )
            for c in ctx
        ]
    )
    for meta in (
        _FakeMeta(ctx),  # attribute absent
        _FakeMeta(ctx, None),  # published but None (non-DCP metadata builder)
        _FakeMeta(ctx, torch.zeros(rows + 1, dtype=torch.int32, device=DEV)),  # stale
    ):
        got = dcp_local_context_lens(meta, 0, world, interleave, rows)
        assert got.dtype == torch.int32
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 8])
def test_merge_launches_the_fused_kernel_count(world):
    """The merge must stay at 3 device kernels (score copy, top-k, gid map).

    The pre-fusion form of this block was 6+ (reshape copy, int64 index cast,
    gather, copy_ on top of the same three), so this trips the moment the fused
    gid map is swapped back out for a reshape+gather.
    """
    rows, k_loc = 8, 256
    recv = torch.empty(world, 2, rows, k_loc, dtype=torch.int32, device=DEV)
    recv[:, 0] = torch.randn(world, rows, k_loc, device=DEV).view(torch.int32)
    recv[:, 1] = torch.randint(
        0, 1 << 20, (world, rows, k_loc), dtype=torch.int32, device=DEV
    )
    out = torch.empty(rows, k_loc, dtype=torch.int32, device=DEV)

    dcp_merge_candidates(recv, out)  # warm up any JIT before counting
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        dcp_merge_candidates(recv, out)
        torch.cuda.synchronize()
    launches = sum(
        e.count
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total
    )
    assert launches <= 4, f"merge launched {launches} kernels, expected <=4"

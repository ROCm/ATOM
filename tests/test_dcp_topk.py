# SPDX-License-Identifier: MIT
"""DCP decode candidate exchange: does it reproduce the global top-k?

The decode indexer replaces a global top-k over the whole context with "local
top-k per rank, exchange the W*topk scores, merge". These tests drive that
substitution end to end -- from a real global logits plane, through each rank's
shard, to the physical KV slots the rank ends up attending over -- and ask
whether the answer is still the global top-k.

The merge is ``aiter.flydsl_dcp_topk_merge``, which selects the global threshold
and emits only THIS rank's owned slots, already localized and compacted. The
exchange therefore carries the score plane alone: ownership is positional
(candidate column c came from rank c // k_loc), so global ids never travel.

Why determinism is the property under test rather than a nicety: each rank runs
the merge independently, so an ambiguous choice resolved differently on two ranks
breaks the disjoint-partition premise ``cp_lse_ag_out_rs`` needs. Ambiguity can
only arise among candidates whose score exactly equals the selection threshold,
which is why the tie-heavy cases below matter more than the random-float ones --
on real workloads only ~0.06% of rows have such a tie, so a random-data-only test
would almost never exercise the path that matters.

Note what the exchange does NOT promise. If a rank's own top-k boundary lands on
a tie, aiter's kernel may keep either tied token, so the exchanged candidate set
can differ from a gid-stable local top-k and the merged answer can be a DIFFERENT
valid top-k. It is still valid (same score multiset) and the ranks still partition
the KV disjointly, which is what the partition needs. The assertions below are
written to that weaker, real contract.
"""

import pytest
import torch

try:
    from aiter.ops.topk import flydsl_dcp_topk_merge, top_k_per_row_decode

    from atom.model_ops.dcp_ops import dcp_local_context_lens
except ImportError as _e:  # triton/aiter absent on a CPU-only runner
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)

DEV = "cuda"
TOPK = 2048
# The decode indexer sizes its local plane from max_model_len, not from the live
# context, so every rank's shard is padded with uninitialised memory.
MAX_MODEL_LEN = 1 << 20
# Page size for the identity block_table these tests build.
PAGE = 16


def _build_gathered(global_logits, ctx, world, k=TOPK):
    """Everything up to and including the all-gather, for all ranks.

    Single process: the all-gather is a concat, and each rank's local shard is
    carved out of the global plane by the round-robin rule (position p lives on
    rank p % W, at local index p // W).

    Returns what production hands the merge: the [rows, W*k] score plane with
    column block r owned by rank r, plus each rank's local_idx.
    """
    rows = global_logits.shape[0]
    dev = global_logits.device
    l_max = (MAX_MODEL_LEN + world - 1) // world
    scores, idxs = [], []

    for r in range(world):
        local_ctx = (ctx - r + world - 1) // world  # #positions p<ctx, p%W==r
        # torch.empty in the real path: everything past local_ctx is garbage, so
        # seed it with garbage that would WIN if it were ever read.
        local = torch.rand(rows, l_max, device=dev, dtype=torch.float32) * 1e4
        if local_ctx > 0:
            local[:, :local_ctx] = global_logits[:, r:ctx:world]
        lens = torch.full((rows,), local_ctx, dtype=torch.int32, device=dev)

        idx = torch.empty(rows, k, dtype=torch.int32, device=dev)
        val = torch.empty(rows, k, dtype=torch.float32, device=dev)
        # values= is what lets the exchange drop the separate score gather. Short
        # rows pad the index with -1 and the score with -inf, so the merge sinks
        # them with no masking here.
        top_k_per_row_decode(
            local,
            1,
            lens,
            idx,
            rows,
            local.stride(0),
            local.stride(1),
            k,
            stable=True,
            values=val,
        )
        scores.append(val.clone())
        idxs.append(idx.clone())

    # all_gather -> [W, rows, k] -> [rows, W*k], rank-major within a row.
    gathered = torch.stack(scores, dim=0).permute(1, 0, 2).reshape(rows, world * k)
    return gathered.contiguous(), idxs


def _identity_block_table(rows, ctx, world, dev, page=PAGE):
    """A block_table whose slot arithmetic inverts back to a local index.

    The merge emits physical slots; an identity table lets the tests assert on
    positions instead. Local index j maps to
    ``block_table[j // page] * page + j % page``, which with this table is
    ``row * n_blocks * page + j``.
    """
    l_max = (ctx + world - 1) // world
    n_blocks = (l_max + page - 1) // page + 1
    bt = torch.arange(rows * n_blocks, dtype=torch.int32, device=dev).reshape(
        rows, n_blocks
    )
    return bt, n_blocks


def _merge_owned(gathered, local_idx, block_table, rank, world, k_loc, topk=TOPK):
    """The production merge for one rank -- the real op, not a copy of it."""
    rows = gathered.shape[0]
    dev = gathered.device
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=dev)
    counts = torch.zeros(rows, dtype=torch.int32, device=dev)
    out = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32, device=dev)
    staging = torch.empty(rows, k_loc, dtype=torch.int32, device=dev)
    flydsl_dcp_topk_merge(
        gathered,
        local_idx,
        block_table,
        out,
        indptr,
        counts,
        staging,
        rank,
        world,
        topk,
        PAGE,
    )
    return out, indptr


def _owned_global_positions(gathered, idxs, ctx, world, k=TOPK):
    """Merge on every rank; return each rank's owned GLOBAL positions per row.

    Inverts the slot arithmetic through the identity block_table, then undoes the
    round-robin shard: local index j on rank r is global position j*W + r.
    """
    rows = gathered.shape[0]
    bt, n_blocks = _identity_block_table(rows, ctx, world, gathered.device)
    per_rank = []
    for r in range(world):
        out, indptr = _merge_owned(gathered, idxs[r], bt, r, world, k)
        rows_out = []
        for row in range(rows):
            s, e = int(indptr[row]), int(indptr[row + 1])
            j = out[s:e].to(torch.int64) - row * n_blocks * PAGE
            rows_out.append(j * world + r)
        per_rank.append(rows_out)
    return per_rank


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
    """The union over ranks of what each one owns == the global top-k.

    No rank sees the whole set any more -- the merge emits only its own share --
    so the invariant lives in the union, not in any single rank's output.
    """
    gl = _make_logits(rows, ctx, seed, tie_frac)
    gathered, idxs = _build_gathered(gl, ctx, world)
    per_rank = _owned_global_positions(gathered, idxs, ctx, world)
    ref = _global_reference(gl, ctx)
    n_keep = min(TOPK, ctx)

    # Check the counts BEFORE stacking: a merge that drops tokens gives ragged
    # rows, and torch.stack would raise a shape error that says nothing about
    # what actually went wrong.
    unions = [torch.cat([per_rank[w][r] for w in range(world)]) for r in range(rows)]
    for r, union in enumerate(unions):
        assert (
            union.numel() == n_keep
        ), f"[{name}] row {r}: {union.numel()} owned tokens, expected {n_keep}"
        assert bool(
            ((union >= 0) & (union < ctx)).all()
        ), f"[{name}] row {r}: position out of range"
        assert (
            union.unique().numel() == n_keep
        ), f"[{name}] row {r}: the same token is owned twice"
    got = torch.stack(unions)

    # The invariant that survives a local boundary tie reshuffling WHICH of two
    # equal-scored tokens was exchanged: the selected score multiset is still
    # exactly the global top-k's.
    sc_got = torch.gather(gl, 1, got).sort(-1).values
    sc_ref = torch.gather(gl, 1, ref.long()).sort(-1).values
    assert torch.equal(sc_got, sc_ref), f"[{name}] selected scores are not the top-k"

    # With no tie AT the global threshold there is nothing to choose, so the ids
    # themselves must match the gid-stable reference.
    thr = gl[:, :ctx].topk(n_keep, dim=-1).values[:, -1:]
    if int((gl[:, :ctx] == thr).sum(-1).max()) == 1:
        assert torch.equal(
            got.sort(-1).values, ref.sort(-1).values.long()
        ), f"[{name}] ids differ from the reference with no threshold tie"


# Re-merging the SAME gathered buffer must return the same answer, or two ranks
# could claim the same token. That is a property of the op alone, and aiter owns
# it: op_tests/test_flydsl_dcp_topk_merge.py::check_deterministic_across_runs
# re-runs it 200 times per rank on tie-heavy and random inputs. Not duplicated
# here. What this file still has to prove is the weaker end-to-end statement
# below: the PIPELINE may return a different valid answer between runs, because
# aiter's local top-k picks arbitrarily among tied candidates before the merge
# ever sees them.


def test_reruns_stay_valid_even_when_the_set_shifts():
    """End to end, with ties: the chosen set may move, but never off the top-k."""
    rows, world, ctx = 16, 8, 65536
    gl = _make_logits(rows, ctx, seed=9, tie_frac=0.99)
    ref_sc = torch.gather(gl, 1, _global_reference(gl, ctx).long()).sort(-1).values
    for i in range(5):
        gathered, idxs = _build_gathered(gl, ctx, world)
        per_rank = _owned_global_positions(gathered, idxs, ctx, world)
        got = torch.stack(
            [torch.cat([per_rank[w][r] for w in range(world)]) for r in range(rows)]
        )
        # Range-check BEFORE the gather. An out-of-range index would trip a
        # device-side assert, and on ROCm that leaves the HIP context unusable --
        # every later GPU test in the same process fails with an error that
        # points nowhere near here. (Clamping instead would hide the regression.)
        assert bool(((got >= 0) & (got < ctx)).all()), f"rerun {i}: id out of range"
        got_sc = torch.gather(gl, 1, got).sort(-1).values
        assert torch.equal(got_sc, ref_sc), f"rerun {i} selected outside the top-k"


# ─────────────────────────────── the ranks partition the KV disjointly ──
#
# What the merge promises beyond "the union is the global top-k": no token is
# owned twice. ``cp_lse_ag_out_rs`` combines the per-rank partial attentions by
# summing them, so a token counted on two ranks is silently double-weighted --
# no crash, just a wrong answer.


@pytest.mark.parametrize(
    "name, rows, ctx, world",
    [
        ("ctx < topk (the padding case)", 4, 1000, 8),
        ("ctx just over topk", 4, 2500, 8),
        ("long ctx", 4, 32768, 8),
        ("heavy ties", 4, 32768, 8),
    ],
)
def test_filter_partitions_topk_disjointly(name, rows, ctx, world):
    tie = 0.99 if "ties" in name else 0.0
    gl = _make_logits(rows, ctx, seed=11, tie_frac=tie)
    gathered, idxs = _build_gathered(gl, ctx, world)
    per_rank = _owned_global_positions(gathered, idxs, ctx, world)

    for r in range(rows):
        seen = torch.cat([per_rank[w][r] for w in range(world)])
        assert (
            seen.unique().numel() == seen.numel()
        ), f"[{name}] row {r}: ranks overlap -- a token would be double-counted"
        # Ownership is positional: rank w may only ever emit positions p with
        # p % W == w (round-robin shard, S=1).
        for w in range(world):
            owned = per_rank[w][r]
            if owned.numel():
                assert bool(
                    ((owned % world) == w).all()
                ), f"[{name}] row {r}: rank {w} emitted a token it does not own"


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
# These exist because the correctness tests above are blind to a production path
# that stops *using* a fused helper: they drive the ops directly, so swapping a
# fused read back out for eager code keeps every assertion green. That has
# happened before -- a move between modules silently replaced the fused local_ctx
# read, and every test still passed.
# ---------------------------------------------------------------------------


class _FakeMeta:
    def __init__(self, ctx, published=None):
        self.context_lens = ctx
        self.dcp_local_context_lens = published


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("interleave", [1, 16])
def test_local_context_lens_prefers_published_buffer(world, interleave):
    """The published host buffer must be returned as-is, not recomputed.

    Identity, not equality: returning an equal-but-recomputed tensor is exactly
    the regression this guards (7 elementwise kernels per full-index layer).
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
        _FakeMeta(ctx),  # non-DCP or non-sparse metadata builder
        _FakeMeta(ctx, torch.zeros(rows - 1, dtype=torch.int32, device=DEV)),  # stale
    ):
        got = dcp_local_context_lens(meta, 0, world, interleave, rows)
        assert got.dtype == torch.int32
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("world", [2, 8])
def test_merge_launches_the_fused_kernel_count(world):
    """The merge must stay at 2 device kernels (select, pack).

    The sequence it replaced was ~16, and every correctness assertion above would
    happily accept a return to it -- they drive the op, not the op count.

    It was 3 until the cross-row prefix sum moved into pack: that scan used to be
    its own grid=(1,) kernel walking the rows on one thread, which cost 37% of the
    op at rows=128. Each pack block now recomputes the prefix it needs in
    parallel, so a regression back to a separate scan kernel trips this.
    """
    rows, k_loc, topk = 8, 256, 256
    gathered = torch.randn(rows, world * k_loc, device=DEV)
    local_idx = torch.stack(
        [torch.randperm(k_loc, device=DEV)[:k_loc] for _ in range(rows)]
    ).to(torch.int32)
    bt = torch.randint(0, 500, (rows, 512), dtype=torch.int32, device=DEV)
    # Allocate the outputs OUTSIDE the profiled region: _merge_owned's zeros()
    # are the test harness, not the op, and would be counted as launches.
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=DEV)
    counts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    out = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32, device=DEV)
    staging = torch.empty(rows, k_loc, dtype=torch.int32, device=DEV)

    def merge():
        flydsl_dcp_topk_merge(
            gathered,
            local_idx,
            bt,
            out,
            indptr,
            counts,
            staging,
            0,
            world,
            topk,
            PAGE,
        )

    merge()  # warm up JIT
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        merge()
        torch.cuda.synchronize()
    launches = sum(
        e.count
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total
    )
    assert launches == 2, f"merge launched {launches} kernels, expected 2"

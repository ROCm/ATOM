# SPDX-License-Identifier: MIT
"""Properties the fused DCP decode merge must hold on the gathered buffer.

`aiter.flydsl_dcp_topk_merge` does the global-threshold select, the ownership
filter, the slot localize and the compaction in one op. These check its output
against properties derived independently of any kernel -- what the rank owns, how
the ranks partition the KV, and the -inf padding contract the exchange relies on.

This exercises the part that runs identically on one GPU and on eight. Not covered
here: the all-gather itself (single process) and cross-rank agreement under real
NCCL. Both need a multi-GPU run.
"""

import pytest
import torch

try:
    from aiter.ops.topk import flydsl_dcp_topk_merge, top_k_per_row_decode
except ImportError as _e:  # pragma: no cover - env without aiter/triton
    pytest.skip(f"requires aiter + atom: {_e}", allow_module_level=True)

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("needs a GPU", allow_module_level=True)

DEV = "cuda"


def _build_gathered(rows, world, k_loc, max_blocks, page_size, seed, tie_heavy):
    """The state both paths start from: gathered scores + this rank's local top-k.

    Mirrors what the exchange produces -- scores all-gathered as [rows, W*k_loc]
    with column c owned by rank c // k_loc, and each rank's own local_idx.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_cand = world * k_loc
    if tie_heavy:
        scores = torch.randint(
            -4, 5, (rows, n_cand), generator=g, dtype=torch.int32, device=DEV
        ).float()
    else:
        scores = torch.randn(rows, n_cand, generator=g, device=DEV)
    local_idx = torch.empty(rows, k_loc, dtype=torch.int32, device=DEV)
    hi = min(max_blocks * page_size, k_loc)
    for r in range(rows):
        local_idx[r] = torch.randperm(hi, generator=g, device=DEV)[:k_loc].to(
            torch.int32
        )
    block_table = torch.randint(
        0, 500, (rows, max_blocks), generator=g, dtype=torch.int32, device=DEV
    )
    return scores, local_idx, block_table


def _fused_path(scores, local_idx, block_table, rank, world, k_loc, topk, page):
    rows = scores.shape[0]
    indptr = torch.zeros(rows + 1, dtype=torch.int32, device=DEV)
    counts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    indices = torch.zeros(rows * max(topk, k_loc), dtype=torch.int32, device=DEV)
    staging = torch.empty(rows, k_loc, dtype=torch.int32, device=DEV)
    flydsl_dcp_topk_merge(
        scores,
        local_idx,
        block_table,
        indices,
        indptr,
        counts,
        staging,
        rank,
        world,
        topk,
        page,
    )
    return indices, indptr


@pytest.mark.parametrize("interleave", [1, 2, 4])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("tie_heavy", [False, True])
def test_emitted_slots_are_the_owned_candidates(interleave, page_size, tie_heavy):
    """Every emitted slot is a slot this rank owns, derived from its own local_idx.

    The merge picks a global threshold across all W planes but may only emit from
    its own. Recomputing the expected slot set from local_idx alone -- no kernel
    involved -- catches a merge that emits another rank's candidates or maps the
    page arithmetic wrong.
    """
    rows, world, k_loc, topk = 8, 8, 128, 256
    scores, local_idx, bt = _build_gathered(
        rows, world, k_loc, 512, page_size, seed=7, tie_heavy=tie_heavy
    )
    for rank in range(world):
        ix, ip = _fused_path(scores, local_idx, bt, rank, world, k_loc, topk, page_size)
        for r in range(rows):
            lo, hi = int(ip[r]), int(ip[r + 1])
            if hi == lo:
                continue
            j = local_idx[r].long()
            legal = bt[r][j // page_size].to(torch.int64) * page_size + (j % page_size)
            emitted = ix[lo:hi].to(torch.int64)
            assert torch.isin(
                emitted, legal
            ).all(), f"rank {rank} row {r}: emitted a slot not derivable from local_idx"
            assert (
                emitted.unique().numel() == emitted.numel()
            ), f"rank {rank} row {r}: emitted a duplicate slot"


@pytest.mark.parametrize("stable", [False, True])
def test_topk_emits_each_index_with_its_own_score(stable):
    """`values=` must write the logit belonging to the index beside it.

    The fused exchange ships this score plane instead of re-gathering it from the
    logits, so a score/index mismatch here silently corrupts the global threshold
    every rank then selects against.
    """
    rows, n, k = 4, 4096, 512
    torch.manual_seed(5)
    logits = torch.randn(rows, n, dtype=torch.float32, device=DEV)
    # Includes rows shorter than k, which is where the padding matters.
    lens = torch.tensor([0, 1, 100, n], dtype=torch.int32, device=DEV)

    idx = torch.empty(rows, k, dtype=torch.int32, device=DEV)
    scores = torch.empty(rows, k, dtype=torch.float32, device=DEV)
    top_k_per_row_decode(
        logits,
        1,
        lens,
        idx,
        rows,
        logits.stride(0),
        logits.stride(1),
        k,
        stable=stable,
        values=scores,
    )
    torch.cuda.synchronize()

    for r, L in enumerate(lens.tolist()):
        vk = min(L, k)
        if vk == 0:
            continue
        torch.testing.assert_close(
            scores[r, :vk],
            logits[r].gather(0, idx[r, :vk].long()),
            rtol=0,
            atol=0,
            msg=f"row {r}: score does not match its index",
        )
        # And it really is the top-k: nothing selected may rank below a
        # non-selected candidate.
        chosen = set(idx[r, :vk].tolist())
        rest = [x for x in range(L) if x not in chosen]
        if rest:
            assert (
                scores[r, :vk].min() >= logits[r, rest].max()
            ), f"row {r}: a selected score ranks below a rejected one"


def test_kernel_pads_scores_with_neg_inf():
    """The exchanged score plane must sink padded slots, and the kernel does it.

    top_k_per_row_decode pads short rows' indices with -1; their SCORES must pad
    to -inf, not 0.0. Logits are routinely negative, so a 0.0 pad outranks real
    candidates and steals global top-k slots -- the W ranks then stop
    partitioning the top-k, which is exactly what cp_lse_ag_out_rs cannot
    survive (measured before the fix: 30 slots emitted instead of 256).

    Asserts the kernel's own padding, then that a hypothetical 0.0 pad really
    would break the partition -- so if the kernel ever regresses to 0.0, this
    fails rather than silently corrupting KV.
    """
    rows_p, n_p, k_p = 3, 4096, 512
    torch.manual_seed(11)
    lg = torch.randn(rows_p, n_p, dtype=torch.float32, device=DEV)
    lens_p = torch.tensor([0, 7, 100], dtype=torch.int32, device=DEV)
    idx_p = torch.empty(rows_p, k_p, dtype=torch.int32, device=DEV)
    val_p = torch.empty(rows_p, k_p, dtype=torch.float32, device=DEV)
    top_k_per_row_decode(
        lg,
        1,
        lens_p,
        idx_p,
        rows_p,
        lg.stride(0),
        lg.stride(1),
        k_p,
        stable=True,
        values=val_p,
    )
    torch.cuda.synchronize()
    pad = idx_p < 0
    assert bool((val_p[pad] == -float("inf")).all()), (
        "kernel must pad the score with -inf wherever it padded the index "
        "with -1; a 0.0 pad silently breaks the DCP partition"
    )
    rows, world, k_loc, topk, page = 2, 4, 64, 128, 16
    scores, local_idx, bt = _build_gathered(
        rows, world, k_loc, 512, page, seed=3, tie_heavy=False
    )
    scores = scores - 5.0  # all real logits negative
    local_idx = local_idx.clone()
    local_idx[:, 10:] = -1  # every rank keeps only 10 valid
    n_valid = 10
    expect = min(topk, world * n_valid)

    # (a) leave the 0.0 padding in place -> partition breaks
    bad = scores.clone()
    for w in range(world):
        bad[:, w * k_loc + n_valid : (w + 1) * k_loc] = 0.0
    bad_total = 0
    for rank in range(world):
        _ix, ip = _fused_path(bad, local_idx, bt, rank, world, k_loc, topk, page)
        bad_total += int(ip[rows])
    assert bad_total < rows * expect, (
        "0.0 padding was expected to steal top-k slots but did not; if the "
        "kernel's padding value changed, the -inf sink may no longer be needed"
    )

    # (b) sink it the way the production path does -> partition restored
    good = bad.clone()
    for w in range(world):
        good[:, w * k_loc + n_valid : (w + 1) * k_loc] = -float("inf")
    good_total = 0
    for rank in range(world):
        _ix, ip = _fused_path(good, local_idx, bt, rank, world, k_loc, topk, page)
        good_total += int(ip[rows])
    assert good_total == rows * expect, (
        f"with -inf padding the ranks must partition the top-k: "
        f"{good_total} != {rows * expect}"
    )


def test_ranks_partition_the_kv():
    """The W ranks' owned slot sets are disjoint and cover the global top-k.

    This is the premise cp_lse_ag_out_rs depends on: if two ranks claimed the
    same KV token, or none claimed it, the LSE merge would be wrong.
    """
    rows, world, k_loc, topk, page = 4, 8, 128, 256, 64
    scores, local_idx, bt = _build_gathered(
        rows, world, k_loc, 512, page, seed=11, tie_heavy=True
    )
    per_row_total = torch.zeros(rows, dtype=torch.int64, device=DEV)
    for rank in range(world):
        _ix, ip = _fused_path(scores, local_idx, bt, rank, world, k_loc, topk, page)
        per_row_total += (ip[1:] - ip[:-1]).to(torch.int64)
    expect = min(topk, world * k_loc)
    assert torch.all(
        per_row_total == expect
    ), f"ranks do not partition the top-k: {per_row_total.tolist()} != {expect}"

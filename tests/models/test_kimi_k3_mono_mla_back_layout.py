# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only geometry of the gfx1250 MLA back half: every W_UV block of every
head on one wave, every merged row on one merging wave, a row's lines
disjoint in the bank and addressed alike by the pushing lane and the merging
lane, the banks and flags sized for the widest step."""

import pytest

from atom.models.kimi_k3.mono.config import HIDDEN, MAX_ROWS
from atom.models.kimi_k3.mono.kernels.stages import mla_back_layout as lay

SERVING = (8, 8, 12)  # TP8, DCP8, 12 heads a rank
TRAY = (4, 4, 24)  # one tray: TP4, DCP4
LOCAL = (1, 1, 12)  # one rank, no merge
SHAPES = [SERVING, TRAY, LOCAL]


def test_widths_are_kimi_k3s_mla_back():
    assert (lay.V_DIM, lay.KV_LORA, lay.N_BLOCKS, lay.K_STEPS) == (128, 512, 8, 16)
    assert lay.ROW_LINES == 32 == lay.LANES and lay.LINE_VALUES * lay.ROW_LINES == 128
    assert lay.OPROJ_CTAS == 224 and lay.GRID == 256
    assert lay.WUV_WAVES + lay.MERGE_WAVES == lay.WAVES == 8


@pytest.mark.parametrize(("tp", "dcp", "hl"), SHAPES)
def test_every_w_uv_block_is_one_waves(tp, dcp, hl):
    plan = lay.BackPlan(8, tp, dcp, hl)
    seen = {}
    for cta in range(lay.GRID):
        for wave in range(lay.WAVES):
            job = lay.wuv_job(plan, cta, wave)
            if job is not None:
                assert wave < lay.WUV_WAVES and job not in seen
                seen[job] = (cta, wave)
    assert sorted(seen) == [
        (head, block) for head in range(plan.heads) for block in range(lay.N_BLOCKS)
    ]


@pytest.mark.parametrize(("tp", "dcp", "hl"), SHAPES)
@pytest.mark.parametrize("rows", [1, 7, 16, 53, 54, MAX_ROWS])
def test_every_merged_row_is_one_merging_waves(tp, dcp, hl, rows):
    plan = lay.BackPlan(rows, tp, dcp, hl)
    seen = []
    for cta in range(lay.GRID):
        for wave in range(lay.WAVES):
            for round_index in range(plan.merge_rounds):
                row = lay.merge_row(plan, cta, wave, round_index)
                if row is not None:
                    assert wave >= lay.WUV_WAVES
                    seen.append(row)
    assert sorted(seen) == list(range(plan.merge_rows))
    assert plan.merge_rows == rows * hl
    # no wave both pushes and merges
    assert all(
        lay.wuv_job(plan, cta, wave) is None
        or lay.merge_row(plan, cta, wave, 0) is None
        for cta in range(lay.GRID)
        for wave in range(lay.WAVES)
    )


def test_a_wave_merges_one_row_through_the_serving_widths():
    assert lay.BackPlan(MAX_ROWS, *SERVING).merge_rounds == 1
    assert lay.BackPlan(53, *TRAY).merge_rounds == 1
    assert lay.BackPlan(54, *TRAY).merge_rounds == 2


@pytest.mark.parametrize(("tp", "dcp", "hl"), SHAPES)
def test_a_rows_lines_are_its_own_and_fill_the_bank(tp, dcp, hl):
    plan = lay.BackPlan(3, tp, dcp, hl)
    words = [
        plan.line_word(source, row, line)
        for source in range(dcp)
        for row in range(plan.merge_rows)
        for line in range(lay.ROW_LINES)
    ]
    assert words == list(range(0, plan.merge_bank_words, lay.LINE_WORDS))
    with pytest.raises(ValueError, match="line"):
        plan.line_word(dcp, 0, 0)


def test_the_pushing_lane_and_the_merging_lane_meet_on_a_line():
    """A W_UV lane holds values ``block * 16 + (lane // 16) * 8 + i`` of token
    ``lane % 16``: its two lines are the ones lanes ``value // 4`` of the
    merging wave poll."""
    for block in range(lay.N_BLOCKS):
        for lane in range(lay.LANES):
            first = block * 16 + lane // 16 * 8
            lines = [block * 4 + lane // 16 * 2 + half for half in range(2)]
            assert lines == [first // 4, first // 4 + 1]
    assert sorted(
        block * 4 + group * 2 + half
        for block in range(lay.N_BLOCKS)
        for group in range(2)
        for half in range(2)
    ) == list(range(lay.ROW_LINES))


@pytest.mark.parametrize(("tp", "dcp", "hl"), SHAPES)
def test_the_banks_and_flags_cover_the_widest_step(tp, dcp, hl):
    widest = lay.BackPlan(MAX_ROWS, tp, dcp, hl)
    for rows in range(1, MAX_ROWS + 1, 9):
        plan = lay.BackPlan(rows, tp, dcp, hl)
        assert plan.peer_bytes <= widest.peer_bytes
        assert plan.flag_words <= widest.flag_words
        assert plan.merge_rows <= plan.flag_words
        assert plan.peer_bytes == 8 * (plan.merge_bank_words + plan.ar_bank_words)
        assert plan.peer_bytes % 16 == 0
    assert widest.proj == hl * 128 and widest.heads == dcp * hl


def test_the_all_reduce_bank_is_the_one_shot_o_projs():
    pytest.importorskip("flydsl", reason="k2_oproj_ar_gfx1250 builds with FlyDSL")
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    for tp, dcp, hl in SHAPES:
        for rows in (1, 8, 47, MAX_ROWS):
            plan = lay.BackPlan(rows, tp, dcp, hl)
            assert plan.ar_bank_words == oar.oneshot_bank_words(tp, rows)
    assert lay.OPROJ_CTAS == oar.CTAS and lay.LINE_WORDS == oar.LINE_WORDS
    assert lay.AR_LINE_TOKENS == oar.LINE_TOKENS


def test_the_serving_push_is_the_merge_kernels_volume():
    plan = lay.BackPlan(8, *SERVING)
    # 8 rows x 84 heads of other ranks x 32 lines of 16 B: 344 KB
    assert lay.pushed_lines(plan) == 8 * 84 * 32
    assert lay.pushed_lines(plan) // lay.GRID == 84  # lines a CU


def test_the_plan_refuses_what_the_kernel_cannot_place():
    with pytest.raises(ValueError, match="rows"):
        lay.BackPlan(MAX_ROWS + 1, *SERVING)
    with pytest.raises(ValueError, match="heads"):
        lay.BackPlan(8, 8, 8, 13)  # 104 heads: more W_UV blocks than waves
    with pytest.raises(ValueError, match="whole CTAs"):
        lay.BackPlan(8, 3, 3, 32)
    assert HIDDEN % (8 * lay.OPROJ_COLUMNS) == 0


# ------------------------------------------------------------ address audit
# Every address the kernel stores to or polls on a peer comes from the
# functions below (``mla_back_gfx1250`` calls them on device integers). The
# audit enumerates, as the kernel does, every rank / CTA / wave / lane / line
# and proves on the host: each lies inside the destination rank's receive
# memory (``BackPlan.peer_bytes``, what the runner and the tools map), what
# the ranks push into a rank is exactly what that rank polls, each line once.
np = pytest.importorskip("numpy")
REAL = [TRAY, SERVING]


def _uv_pushed(plan, epoch):
    """``{destination rank: words}`` of every W_UV line every rank pushes,
    by the kernel's enumeration: each wave with a job, each token tile, each
    lane of a live token, its two lines."""
    jobs = [
        lay.wuv_job(plan, cta, wave)
        for cta in range(lay.GRID)
        for wave in range(lay.WAVES)
    ]
    jobs = np.array([job for job in jobs if job is not None])
    head = jobs[:, 0].reshape(-1, 1, 1, 1)
    block = jobs[:, 1].reshape(-1, 1, 1, 1)
    tile = np.arange(plan.tiles).reshape(1, -1, 1, 1)
    lane = np.arange(lay.LANES).reshape(1, 1, -1, 1)
    half = np.arange(2).reshape(1, 1, 1, -1)
    token = tile * 16 + lane % 16
    live = np.broadcast_to(token < plan.rows, (len(jobs), plan.tiles, lay.LANES, 2))
    owner = np.broadcast_to(head // plan.local_heads, live.shape)
    out = {rank: [] for rank in range(plan.dcp)}
    for source in range(plan.dcp):
        word = lay.bank_base(plan, epoch) + lay.uv_line_word(
            plan, source, token, head % plan.local_heads, block, lane // 16, half
        )
        word = np.broadcast_to(word, live.shape)
        for rank in range(plan.dcp):
            out[rank].append(word[live & (owner == rank)])
    return {rank: np.concatenate(words) for rank, words in out.items()}


def _uv_polled(plan, epoch):
    """The words a rank's merging waves poll: every merged row's wave, every
    lane, every shard."""
    rows = [
        lay.merge_row(plan, cta, wave, round_index)
        for cta in range(lay.GRID)
        for wave in range(lay.WAVES)
        for round_index in range(plan.merge_rounds)
    ]
    row = np.array([r for r in rows if r is not None]).reshape(-1, 1, 1)
    lane = np.arange(lay.LANES).reshape(1, -1, 1)
    source = np.arange(plan.dcp).reshape(1, 1, -1)
    return (
        lay.bank_base(plan, epoch) + lay.merge_line_word(plan, source, row, lane)
    ).reshape(-1)


def _ar_items(plan):
    """``(triple, column pair)`` of every all-reduce line the o_proj CTAs'
    threads take, by the kernel's enumeration."""
    cta = np.arange(lay.OPROJ_CTAS).reshape(-1, 1)
    item = np.arange(plan.ar_rounds * lay.THREADS).reshape(1, -1)
    live = np.broadcast_to(item < plan.ar_triples * 16, (lay.OPROJ_CTAS, item.size))
    triple = np.broadcast_to(item // 16, live.shape)[live]
    pair = np.broadcast_to(cta * 16 + item % 16, live.shape)[live]
    return triple, pair


def _inside(plan, words):
    """Whether every 16 B line at ``words`` lies in a rank's receive memory."""
    offsets = 4 * words.astype(np.int64)
    return bool(
        words.size
        and offsets.min() >= 0
        and offsets.max() + 16 <= plan.peer_bytes
        and (offsets % 16 == 0).all()
    )


@pytest.mark.parametrize(("tp", "dcp", "hl"), REAL)
@pytest.mark.parametrize("rows", range(1, MAX_ROWS + 1))
def test_every_line_pushed_is_inside_the_owners_bank_and_polled_once(tp, dcp, hl, rows):
    plan = lay.BackPlan(rows, tp, dcp, hl)
    assert plan.peer_bytes < 1 << 31  # the device's word arithmetic is i32
    for epoch in (1, 2):
        base = lay.bank_base(plan, epoch)
        assert base == (epoch & 1) * plan.bank_words
        pushed = _uv_pushed(plan, epoch)
        polled = np.sort(_uv_polled(plan, epoch))
        assert _inside(plan, polled)
        assert polled[0] == base and polled[-1] < base + plan.merge_bank_words
        assert np.unique(polled).size == polled.size
        for rank in range(dcp):
            assert _inside(plan, pushed[rank])
            # what the ranks push into a rank is what it polls, each line once
            assert np.array_equal(np.sort(pushed[rank]), polled)
        triple, pair = _ar_items(plan)
        ar_polled = np.sort(
            np.concatenate(
                [base + lay.ar_line_word(plan, p, triple, pair) for p in range(tp)]
            )
        )
        assert _inside(plan, ar_polled)
        assert ar_polled[0] == base + plan.merge_bank_words
        assert ar_polled[-1] == base + plan.bank_words - lay.LINE_WORDS
        assert np.unique(ar_polled).size == ar_polled.size
        # every rank pushes its line of every pair to every rank: the union
        # over the sources landing in a rank is what that rank polls
        ar_pushed = np.sort(
            np.concatenate(
                [base + lay.ar_line_word(plan, r, triple, pair) for r in range(tp)]
            )
        )
        assert np.array_equal(ar_pushed, ar_polled)


@pytest.mark.parametrize(("tp", "dcp", "hl"), [LOCAL, (1, 8, 12), (1, 4, 24)])
@pytest.mark.parametrize("rows", [1, 7, 16, 17, 48, MAX_ROWS])
def test_the_one_gpu_shapes_address_alike(tp, dcp, hl, rows):
    plan = lay.BackPlan(rows, tp, dcp, hl)
    pushed = _uv_pushed(plan, 1)
    polled = np.sort(_uv_polled(plan, 1))
    for rank in range(dcp):
        assert _inside(plan, pushed[rank])
        assert np.array_equal(np.sort(pushed[rank]), polled)


@pytest.mark.parametrize(("tp", "dcp", "hl"), [TRAY, SERVING, LOCAL])
@pytest.mark.parametrize("rows", [1, 7, 16, 17, 53, 54, MAX_ROWS])
def test_every_local_read_and_write_is_inside_its_tensor(tp, dcp, hl, rows):
    plan = lay.BackPlan(rows, tp, dcp, hl)
    # core / gate: every merged row's lanes tile [rows, proj] BF16 exactly
    row = np.arange(plan.merge_rows).reshape(-1, 1)
    lane = np.arange(lay.LANES).reshape(1, -1)
    core = np.sort(lay.core_word(plan, row, lane).reshape(-1))
    assert np.array_equal(core, np.arange(0, rows * plan.proj // 2, 2))
    # flags: a word a merged row; every thread's polled words are the buffer's
    tid = np.arange(lay.THREADS).reshape(-1, 1, 1)
    vector = np.arange(plan.flag_vectors).reshape(1, -1, 1)
    index = tid * lay.FLAG_VECTOR + vector * lay.THREADS * lay.FLAG_VECTOR
    index = index + np.arange(lay.FLAG_VECTOR).reshape(1, 1, -1)
    assert index.max() < plan.flag_words and plan.merge_rows <= plan.flag_words
    assert np.unique(index).size == plan.flag_words
    # W_UV's weights and tokens: the last word read is the tensor's last
    head, n = plan.heads - 1, lay.V_DIM - 1
    last = lay.uv_weight_word(head, n, lay.K_STEPS - 1, 1, 1) + 2
    assert last == plan.heads * lay.V_DIM * lay.KV_LORA // 4
    assert lay.uv_weight_word(0, 0, 0, 0, 0) == 0
    last = lay.uv_token_word(plan, rows - 1, head, lay.K_STEPS - 1, 1, 1) + 4
    assert last == rows * plan.heads * lay.KV_LORA // 2
    assert lay.uv_token_word(plan, 0, 0, 0, 0, 0) == 0


def test_the_line_words_are_the_plans():
    plan = lay.BackPlan(5, *TRAY)
    for source, row, line in ((0, 0, 0), (3, 119, 31), (2, 57, 13)):
        assert lay.merge_line_word(plan, source, row, line) == plan.line_word(
            source, row, line
        )
    # a W_UV lane's two lines are the lines of its values
    assert lay.uv_line_word(plan, 1, 4, 23, 7, 1, 1) == plan.line_word(
        1, 4 * 24 + 23, 31
    )
    assert lay.uv_line_word(plan, 0, 0, 0, 0, 0, 0) == 0
    assert plan.ar_rounds == 1 and lay.BackPlan(49, *TRAY).ar_rounds == 2


def test_the_all_reduce_rounds_are_the_o_proj_kernels():
    pytest.importorskip("flydsl", reason="k2_oproj_ar_gfx1250 builds with FlyDSL")
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    for rows in range(1, MAX_ROWS + 1):
        plan = lay.BackPlan(rows, *SERVING)
        assert plan.ar_rounds == oar.line_rounds(rows)
        assert plan.ar_triples == oar.triples(rows)


def test_the_runners_banks_sit_line_aligned_inside_one_mapping():
    """The runner maps one buffer a rank: the K2 all-reduce's banks, the
    o_proj all-reduce's, then the back half's. Each region starts on a line
    boundary and the widest back half ends where the mapping ends."""
    pytest.importorskip("flydsl", reason="the all-reduce kernels build with FlyDSL")
    pytest.importorskip("aiter", reason="the kernel under test needs aiter")
    from atom.models.kimi_k3.mono.kernels.stages import k2_allreduce_gfx1250 as ar
    from atom.models.kimi_k3.mono.kernels.stages import k2_oproj_ar_gfx1250 as oar

    for tp, dcp, hl in REAL:
        reduce_bytes = ar.allreduce_peer_bytes(tp, MAX_ROWS)
        oproj_bytes = oar.oproj_peer_bytes(tp, MAX_ROWS)
        widest = lay.BackPlan(MAX_ROWS, tp, dcp, hl).peer_bytes
        assert reduce_bytes % 16 == 0 and oproj_bytes % 16 == 0 and widest % 16 == 0
        mapped = reduce_bytes + oproj_bytes + widest
        for rows in range(1, MAX_ROWS + 1):
            plan = lay.BackPlan(rows, tp, dcp, hl)
            # the last byte any launch of this width touches in a peer
            last = reduce_bytes + oproj_bytes + plan.peer_bytes
            assert last <= mapped

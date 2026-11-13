# SPDX-License-Identifier: MIT
"""Self-consistency of the probabilistic verification's variants.

On the same inputs `prob_rejection_sample` must give the same tokens and
verdict

- reading the target rows of the full logits in place (`in_place=True`) as
  reading a compacted copy of them;
- over several vocab chunks as over a single chunk (cross-chunk combine);
- with q sharded over ranks, each shard spanning several vocab chunks, as
  with the whole q on one rank;
- splitting an overflowing top-p interval over the whole row as finishing it
  from the chunked levels, including an interval of tied tokens;

and, against an independent sort-based reference (`_nucleus_probs`), the
top-p nucleus and the draft token's p(x) of every top-p row, tied rows
included.

Needs a GPU, or TRITON_INTERPRET=1 (then runs on the CPU).
"""

import os

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
if torch.cuda.is_available():
    DEV = "cuda"
elif os.environ.get("TRITON_INTERPRET") == "1":
    DEV = "cpu"
else:
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

from fake_tp_group import run_ranks

import atom.model_ops.dspark_prob_sample as dps

# Three vocab chunks of the split verification, the last one partial.
V = 2 * dps._CHUNK + 3616
R, N = 4, 3
PER_ROW_TOP_P = (0.95, 0.5, 1.0, 0.8)


def _case(seed, greedy_rows=(), no_q_rows=()):
    g = torch.Generator().manual_seed(seed)
    temps = torch.full((R,), 0.8)
    for i in greedy_rows:
        temps[i] = 1e-10
    per = N + 1
    full = torch.randn(R * per + 3, V, generator=g) * 3.0
    full[1] *= 0.05  # a flat row: top-p keeps many tokens
    drafts = torch.randint(0, V, (R, N), generator=g, dtype=torch.int32)
    q = torch.randn(R, N, V, generator=g) * 3.0
    for i in range(R):
        for k in range(0, N, 2):  # some drafts likely under both models
            full[i * per + k, drafts[i, k]] += 12.0
            q[i, k, drafts[i, k]] += 12.0
    input_ids = torch.zeros(R * per, dtype=torch.int32)
    rows = torch.zeros(R * N, dtype=torch.int32)
    bonus_rows = torch.zeros(R, dtype=torch.int32)
    for i in range(R):
        input_ids[i * per + 1 : (i + 1) * per] = drafts[i]
        rows[i * N : (i + 1) * N] = torch.arange(N, dtype=torch.int32) + i * per
        bonus_rows[i] = i * per + N
    src = torch.arange(R, dtype=torch.int32)
    src[list(no_q_rows)] = -1
    to = {"device": DEV}
    return {
        "full": full.to(**to),
        "rows": rows.to(**to),
        "input_ids": input_ids.to(**to),
        "cu": (torch.arange(1, R + 1, dtype=torch.int32) * N).to(**to),
        "bonus": torch.randint(0, V, (R,), generator=g, dtype=torch.int32).to(**to),
        "bonus_rows": bonus_rows.to(**to),
        "temps": temps.to(**to),
        "src": src.to(**to),
        "q": q.to(**to),
        "drafts": drafts.to(**to),
    }


def _state(c, vocab_start=0, cols=V, tp=1):
    """The draft state of the rank holding q's columns
    ``[vocab_start, vocab_start + cols)``."""
    state = dps.ProbDraftState(
        q=c["q"][:, :, vocab_start : vocab_start + cols].clone(),
        ids=c["drafts"].clone(),
        temps=c["temps"].clone(),
        seed=torch.zeros(1, dtype=torch.int64, device=DEV),
    )
    state.set_layout(vocab_start, cols, tp)
    state.verify_step = 12345
    return state


def _verify(c, top_p, use_block, in_place=False, state=None, tp_group=None):
    if state is None:
        state = _state(c)
    if torch.is_tensor(top_p):
        top_p = top_p.to(DEV)
    logits = c["full"] if in_place else c["full"].index_select(0, c["rows"])
    out, verdict = dps.prob_rejection_sample(
        c["input_ids"],
        c["rows"],
        N,
        c["cu"],
        logits,
        c["bonus"],
        c["bonus_rows"],
        c["temps"],
        top_p,
        c["src"],
        state,
        use_block=use_block,
        tp_group=tp_group,
        in_place=in_place,
    )
    return out.cpu(), verdict.cpu()


CASES = [
    # seed, top_p, block verification, greedy rows, rows without q
    (1, None, False, (), ()),
    (2, 0.9, False, (), ()),
    (3, 0.9, True, (2,), (1,)),
    (4, None, True, (), (3,)),
    (5, torch.tensor(PER_ROW_TOP_P), False, (0,), ()),
    (6, torch.tensor(PER_ROW_TOP_P), True, (), (2,)),
]


def _assert_same(a, b):
    assert torch.equal(a[0], b[0]), (a[0], b[0])
    assert torch.equal(a[1], b[1]), (a[1], b[1])


@pytest.mark.parametrize("seed,top_p,use_block,greedy,no_q", CASES)
def test_rows_read_in_place_match_compacted_rows(seed, top_p, use_block, greedy, no_q):
    c = _case(seed, greedy, no_q)
    _assert_same(
        _verify(c, top_p, use_block, in_place=True), _verify(c, top_p, use_block)
    )


def test_in_place_refuses_a_strided_source():
    """The kernels step rows by the vocab width, so a strided view of the
    logits must be refused rather than silently copied back out."""
    c = _case(1, (), ())
    strided = torch.cat((c["full"], c["full"]), dim=1)[:, :V]
    assert not strided.is_contiguous()
    state = dps.ProbDraftState(
        q=c["q"].clone(),
        ids=c["drafts"].clone(),
        temps=c["temps"].clone(),
        seed=torch.zeros(1, dtype=torch.int64, device=DEV),
    )
    state.set_layout(0, V, 1)
    with pytest.raises(AssertionError):
        dps.prob_rejection_sample(
            c["input_ids"],
            c["rows"],
            N,
            c["cu"],
            strided,
            c["bonus"],
            c["bonus_rows"],
            c["temps"],
            None,
            c["src"],
            state,
            in_place=True,
        )


@pytest.mark.parametrize("seed,top_p,use_block,greedy,no_q", CASES)
def test_vocab_chunks_match_a_single_chunk(
    monkeypatch, seed, top_p, use_block, greedy, no_q
):
    c = _case(seed, greedy, no_q)
    split = _verify(c, top_p, use_block)
    assert triton.cdiv(V, dps._CHUNK) > 2
    monkeypatch.setattr(dps, "_CHUNK", 4 * dps._CHUNK)  # one chunk per row
    _assert_same(split, _verify(c, top_p, use_block))


@pytest.mark.parametrize("seed,top_p,use_block,greedy,no_q", CASES)
def test_sharded_q_spanning_chunks_matches_one_rank(
    seed, top_p, use_block, greedy, no_q
):
    """Two ranks, each holding half of q's columns: every shard spans more than
    one vocab chunk, so the stats chunks past a rank's columns carry empty q
    records and the residual passes clip and offset per shard."""
    tp, cols = 2, V // 2
    assert cols > dps._CHUNK and V % tp == 0
    c = _case(seed, greedy, no_q)
    states = [_state(c, rk * cols, cols, tp) for rk in range(tp)]
    ranks = run_ranks(
        tp,
        lambda rk, grp: _verify(c, top_p, use_block, state=states[rk], tp_group=grp),
    )
    for other in ranks[1:]:
        _assert_same(other, ranks[0])
    _assert_same(ranks[0], _verify(c, top_p, use_block))


@pytest.mark.parametrize("seed,top_p,use_block,greedy,no_q", CASES[1:3] + CASES[4:])
def test_whole_row_top_p_splits_match_chunked_search(
    monkeypatch, seed, top_p, use_block, greedy, no_q
):
    c = _case(seed, greedy, no_q)
    chunked = _verify(c, top_p, use_block)
    # A tiny compaction buffer: every top-p interval overflows the chunked
    # levels and keeps splitting over the whole row until it fits.
    monkeypatch.setattr(dps, "_TOPP_RB", 16)
    _assert_same(chunked, _verify(c, top_p, use_block))


@pytest.mark.parametrize("use_block", [False, True])
def test_tied_cutoff_interval_is_kept_whole(monkeypatch, use_block):
    # Logits on a coarse grid: the cutoff falls inside many exactly
    # tied tokens, which no split separates. They are kept whole, as the
    # bisection keeps a tie at the boundary.
    c = _case(7)
    c["full"] = torch.round(c["full"] * 2.0) / 2.0
    top_p = torch.tensor(PER_ROW_TOP_P)
    default = _verify(c, top_p, use_block)
    monkeypatch.setattr(dps, "_TOPP_RB", 16)
    _assert_same(default, _verify(c, top_p, use_block))


def _row_stats(monkeypatch, c, top_p, use_block):
    """`_verify`, also returning the per-row statistics [N, 8] handed to
    the decide kernel (amax, m, tau, zk, p(x) | q fields)."""
    seen = {}
    kernel = dps._prob_decide_kernel
    rs_arg = kernel.arg_names.index("rs_ptr")

    class Capture:
        def __getitem__(self, grid):
            launch = kernel[grid]

            def run(*args, **kwargs):
                seen["rs"] = args[rs_arg].clone()
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(dps, "_prob_decide_kernel", Capture())
    out = _verify(c, top_p, use_block)
    return out, seen["rs"].cpu()


@pytest.mark.parametrize(
    "seed,top_p,grid,rb",
    [
        (11, 0.9, None, None),
        (12, torch.tensor(PER_ROW_TOP_P), None, None),
        # Logits on a coarse grid: the cutoff falls on a run of tied tokens.
        (13, torch.tensor(PER_ROW_TOP_P), 2.0, None),
        (14, 0.8, 2.0, None),
        # Every interval overflows: the whole-row splits find the cutoff.
        (15, torch.tensor(PER_ROW_TOP_P), None, 16),
        (16, torch.tensor(PER_ROW_TOP_P), 2.0, 16),
    ],
)
def test_top_p_nucleus_matches_sorted_reference(monkeypatch, seed, top_p, grid, rb):
    c = _case(seed)
    if grid is not None:
        c["full"] = torch.round(c["full"] * grid) / grid
    if rb is not None:
        monkeypatch.setattr(dps, "_TOPP_RB", rb)
    _, rs = _row_stats(monkeypatch, c, top_p, use_block=False)
    target = c["full"].index_select(0, c["rows"]).cpu().float()
    temps = c["temps"].cpu()
    drafts = c["input_ids"].cpu()[c["rows"].cpu().long() + 1].long()
    tps = top_p if torch.is_tensor(top_p) else torch.full((R,), top_p)
    checked = 0
    for i in range(R):
        if float(tps[i]) >= 1.0:
            continue
        for k in range(N):
            r = i * N + k
            ref = dps._nucleus_probs(
                target[r : r + 1], temps[i : i + 1, None], float(tps[i])
            )[0]
            y = target[r] * (1.0 / temps[i])
            nucleus = y >= rs[r, 2]
            assert torch.equal(nucleus, ref > 0), (r, int(nucleus.sum()))
            x = int(drafts[r])
            assert float(rs[r, 4]) == pytest.approx(float(ref[x]), rel=1e-4, abs=1e-12)
            checked += 1
    assert checked > 0

# SPDX-License-Identifier: MIT
"""The two kernels lossless speculative sampling adds, against their laws.

* `dspark_markov_sample`: each row stores the top of ``softmax((base + bias) /
  T)`` -- ids, renormalized probabilities, and the slot of the id it drew --
  at its own row block, nowhere else, and draws ``x`` from exactly that q.
* The probability-ratio verify (`rejection_sample` with ``draft_probs``): the
  emitted tokens follow the target's p whatever q the drafts came from, read
  from the sparse store exactly as a dense q would be read; greedy rows are
  exact match; a request without q falls back to p. A request with top-k /
  top-p is served `Sampler`'s filtered p: the same ids kept, the same law.

Statistical checks use fixed seeds and tolerances several standard errors
wide, so they are deterministic on a given platform and fail only on a real
bias. The properties that hold for every coin (destination rows, residual
support, greedy outputs, rank-consistent draws) are asserted exactly.
"""

import pytest
import torch

# The CPU unit job has neither aiter nor triton; both imports stay below these
# guards, or the job fails to collect this file and aborts the whole run.
pytest.importorskip("aiter", reason="the samplers are AITER/Triton ops")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

import triton
import triton.language as tl
from spec_lossless_reference import dspark_markov_sample_reference, sparse_q_reference

from atom.model_ops.dspark_markov_sample import (
    dspark_markov_argmax,
    dspark_markov_sample,
    dspark_markov_sample_scratch,
)
from atom.model_ops.rejection_sampler import (
    _RATIO_ACCEPT_KEY,
    _RATIO_BLOCK_V,
    _RATIO_RESIDUAL_KEY,
    _draft_token,
    _finish_request,
    _ratio_accept_kernel,
    _ratio_densify_clear_kernel,
    _ratio_densify_scatter_kernel,
    _ratio_residual_pick_kernel,
    _ratio_residual_race_kernel,
    _ratio_scratch_shapes,
    _ratio_stats_kernel,
    _ratio_target_p,
    ratio_verify_scratch,
    rejection_sample,
    warmup_ratio_verify,
)
from atom.model_ops.sampler import SAMPLER_EPS, Sampler
from atom.model_ops.sampling_filter import (
    _ratio_filter_count_kernel,
    _ratio_filter_cut_kernel,
    _ratio_filter_floor_kernel,
    _ratio_filter_pass_kernel,
    _request_of_row,
)
from atom.model_ops.sampling_scratch import take
from atom.model_ops.softmax_tiles import softmax_combine_kernel
from atom.model_ops.sparse_q_store import RatioDraftQ, SparseQStore

DEV = "cuda"
ANCHOR = 0


def tables(rows, vocab, rank, seed=1201):
    torch.manual_seed(seed)
    kw = {"dtype": torch.bfloat16, "device": DEV}
    return (
        torch.randn(rows, vocab, **kw),
        torch.randint(0, vocab, (rows,), dtype=torch.int64, device=DEV),
        torch.randn(vocab, rank, **kw) * 0.1,
        torch.randn(vocab, rank, **kw) * 0.1,
    )


def draft_args(rows, temperature=1.0, greedy=False, step=0):
    return (
        torch.full((rows,), temperature, device=DEV),
        torch.full((rows,), int(greedy), dtype=torch.int32, device=DEV),
        torch.tensor([step], dtype=torch.int32, device=DEV),
    )


# ---- draft: dspark_markov_sample --------------------------------------------


def filters(rows, top_k=0, top_p=1.0):
    return (
        torch.full((rows,), top_k, dtype=torch.int32, device=DEV),
        torch.full((rows,), top_p, device=DEV),
    )


def empty_store(rows, width):
    """A store every slot of which reads as "never written"."""
    store = SparseQStore.allocate(rows, width, DEV)
    for t in store:
        t.fill_(-7)
    return store


def sample(
    base,
    prev,
    w1,
    w2,
    store,
    row_base,
    position,
    temps,
    greedy,
    seed,
    top_k=None,
    top_p=None,
    noise_offset=0,
    scratch=None,
):
    """One `dspark_markov_sample` call; ``(out, s_scratch)``."""
    rows, vocab = base.shape
    if top_k is None:
        top_k, _ = filters(rows)
    if top_p is None:
        _, top_p = filters(rows)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    s = torch.full((rows, vocab), float("nan"), device=DEV)
    dspark_markov_sample(
        base,
        prev,
        w1,
        w2,
        out,
        *store,
        s,
        row_base,
        position,
        temps,
        greedy,
        top_k,
        top_p,
        seed,
        noise_offset,
        None if scratch is None else scratch.f32,
        None if scratch is None else scratch.i32,
    )
    return out, s


def stored(store, row):
    """``(ids, probs, xslot)`` of one store row, as written."""
    n, xslot = store.meta[row].tolist()
    return store.ids[row, :n], store.probs[row, :n], xslot


def assert_store_row_is_q(store, row, s_row, out_id, width, top_k=None, top_p=None):
    """The store contract for one sampled row whose scores were ``s_row``."""
    ids, probs, xslot = stored(store, row)
    n = ids.numel()
    assert 1 <= n <= width
    # Distinct, ascending, in range.
    assert (ids[1:] > ids[:-1]).all() and ids[0] >= 0 and ids[-1] < s_row.numel()
    assert ids[xslot].item() == out_id, "xslot does not name the emitted id"
    assert (probs > 0).all()
    assert abs(probs.double().sum().item() - 1.0) < 1e-5
    gap = s_row.max() - s_row[ids.long()]
    assert (gap <= 19.0).all(), "an id more than 19 nats down was stored"
    # Exactly the set the (level, id) oracle picks, at its values.
    ref = sparse_q_reference(
        s_row[None],
        width,
        None if top_k is None else top_k[:1],
        None if top_p is None else top_p[:1],
    )[0]
    assert torch.equal(ids.long(), ref.nonzero().view(-1))
    torch.testing.assert_close(probs, ref[ids.long()], rtol=1e-5, atol=1e-7)


def test_each_row_writes_its_q_at_its_own_block_and_nowhere_else():
    rows, vocab, rank, t, width = 6, 3000, 64, 4, 64
    base, prev, w1, w2 = tables(rows, vocab, rank)
    store = empty_store(10 * t, width)
    # Two rows without a destination, the others scattered and out of order.
    row_base = torch.tensor([8, -1, 0, 4, -1, 6], dtype=torch.int32, device=DEV) * t
    row_base = torch.where(row_base < 0, -1, row_base).to(torch.int32)
    position = 2
    temps, greedy, seed = draft_args(rows, temperature=0.8)
    out, s = sample(base, prev, w1, w2, store, row_base, position, temps, greedy, seed)
    written = set()
    for r, b in enumerate(row_base.tolist()):
        if b >= 0:
            assert_store_row_is_q(store, b + position, s[r], out[r].item(), width)
            written.add(b + position)
    untouched = [i for i in range(store.ids.shape[0]) if i not in written]
    assert all((t[untouched] == -7).all() for t in store), "written outside a block"
    assert ((out >= 0) & (out < vocab)).all()


@pytest.mark.parametrize("rows", [1, 32])
def test_at_temperature_zero_it_drafts_what_the_argmax_kernel_drafts(rows):
    """The flag-on draft at T=0 is the flag-off draft, chained position by
    position, and the q it publishes is the delta on that id."""
    vocab, rank, steps = 2048, 128, 4
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows, temperature=SAMPLER_EPS, greedy=True)
    store = empty_store(rows * steps, 16)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV) * steps
    drawn, ref = prev, prev
    for k in range(steps):
        ref_ids = torch.empty(rows, dtype=torch.int64, device=DEV)
        dspark_markov_argmax(base, ref, w1, w2, ref_ids)
        ids, _ = sample(base, drawn, w1, w2, store, row_base, k, temps, greedy, seed)
        assert torch.equal(ids, ref_ids), k
        rows_k = row_base.long() + k
        assert store.meta[rows_k].tolist() == [[1, 0]] * rows
        assert torch.equal(store.ids[rows_k, 0].long(), ids)
        assert (store.probs[rows_k, 0] == 1.0).all()
        drawn, ref = ids, ref_ids


def test_a_greedy_row_on_a_tie_publishes_the_delta_its_id_came_from():
    """softmax(s / EPS) on an exact tie is 1/2 per tied id, but the id is the
    lowest of them, not a draw; the accept rule must see the delta."""
    rows, vocab, rank, low, high = 2, 3000, 32, 40, 2900
    base = torch.randn(rows, vocab, device=DEV).to(torch.bfloat16)
    base[:, low] = 20.0
    base[:, high] = 20.0
    prev = torch.zeros(rows, dtype=torch.int64, device=DEV)
    w = torch.zeros(vocab, rank, dtype=torch.bfloat16, device=DEV)
    temps, greedy, seed = draft_args(rows, temperature=SAMPLER_EPS, greedy=True)
    store = empty_store(rows, 8)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    out, _ = sample(base, prev, w, w, store, row_base, 0, temps, greedy, seed)
    assert out.tolist() == [low, low]
    for r in range(rows):
        ids, probs, xslot = stored(store, r)
        assert ids.tolist() == [low] and probs.tolist() == [1.0] and xslot == 0


@pytest.mark.parametrize("temperature", [SAMPLER_EPS, 1.0], ids=["greedy", "sampled"])
def test_a_row_without_a_destination_drafts_the_argmax_and_writes_nothing(
    temperature,
):
    """Greedy requests claim no q slot, and pad, dummy and align-only rows
    write no q; all draft the argmax rather than id 0, so a misclassified step
    would still propose sensibly."""
    rows, vocab, rank = 16, 2048, 64
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(
        rows, temperature=temperature, greedy=temperature == SAMPLER_EPS, step=3
    )
    store = empty_store(4, 32)
    row_base = torch.full((rows,), -1, dtype=torch.int32, device=DEV)
    ref = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_argmax(base, prev, w1, w2, ref)
    out, _ = sample(base, prev, w1, w2, store, row_base, 0, temps, greedy, seed)
    assert torch.equal(out, ref)
    assert all((t == -7).all() for t in store)


def test_the_preallocated_scratch_drafts_exactly_what_allocation_does():
    rows, vocab, rank, width = 24, 3000, 64, 100
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows, temperature=0.9, step=5)
    greedy[::3] = 1
    scratch = dspark_markov_sample_scratch(64, vocab, DEV)  # sized above rows
    results = []
    for buffers in (None, scratch):
        store = empty_store(rows, width)
        row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
        out, _ = sample(
            base, prev, w1, w2, store, row_base, 0, temps, greedy, seed, scratch=buffers
        )
        results.append((out, *store))
    for a, b in zip(*results):
        assert torch.equal(a, b)


def test_the_drafts_combine_without_1_over_z_keeps_the_max_and_argmax():
    """The sampled draft reads only the row max and the raw argmax (its q is
    normalized over the stored entries), so it combines without the sum: the
    two it reads must be bit for bit what the full combine gives, on finite,
    partly -inf and all -inf rows, and the sum buffers are never touched."""
    torch.manual_seed(181)
    tiles, rows = 37, 5
    part_max = torch.randn(tiles, rows, device=DEV) * 4
    part_max[::3, 1] = float("-inf")
    part_max[:, 2] = float("-inf")
    part_sum = torch.rand(tiles, rows, device=DEV)
    part_val = torch.randn(tiles, rows, device=DEV).round()
    part_val[:, 2] = float("-inf")
    part_idx = torch.randint(0, 1 << 20, (tiles, rows), device=DEV).to(torch.int32)
    outs = []
    for with_inv_z in (True, False):
        row_max = torch.full((rows,), -7.0, device=DEV)
        row_argmax = torch.full((rows,), -7, dtype=torch.int32, device=DEV)
        row_inv_z = torch.full((rows,), -7.0, device=DEV)
        softmax_combine_kernel[(rows,)](
            part_max,
            part_sum if with_inv_z else None,
            row_max,
            row_inv_z if with_inv_z else None,
            part_val,
            part_idx,
            row_argmax,
            rows,
            tiles,
            BLOCK_TILE=triton.next_power_of_2(tiles),
            WITH_ARGMAX=True,
            WITH_INV_Z=with_inv_z,
            num_warps=4,
        )
        outs.append((row_max, row_argmax))
    (full_max, full_argmax), (max_only, argmax_only) = outs
    assert torch.equal(full_max.view(torch.int32), max_only.view(torch.int32))
    assert torch.equal(full_argmax, argmax_only)


@pytest.mark.parametrize("width", [1, 7, 1024, 4096, 163840])
def test_the_store_holds_exactly_the_top_of_q_at_any_width(width):
    """At the real vocab, K from one entry to all of V, with and without the
    request's filters: every invariant of the store contract, and the set the
    (level, id) oracle picks."""
    rows, vocab, rank = 6, 163840, 64
    base, prev, w1, w2 = tables(rows, vocab, rank, seed=7)
    # Wide rows (a flat T) and peaked ones, so the cut lands at every depth.
    temps = torch.tensor([1.0, 0.3, 0.7, 2.0, 1.0, 0.7], device=DEV)
    _, greedy, seed = draft_args(rows, step=9)
    top_k = torch.tensor([0, 0, 50, 0, 3000, 0], dtype=torch.int32, device=DEV)
    top_p = torch.tensor([1.0, 1.0, 1.0, 0.9, 0.5, 0.95], device=DEV)
    store = empty_store(rows, width)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    out, s = sample(
        base, prev, w1, w2, store, row_base, 0, temps, greedy, seed, top_k, top_p
    )
    for r in range(rows):
        assert_store_row_is_q(
            store, r, s[r], out[r].item(), width, top_k[r : r + 1], top_p[r : r + 1]
        )
        if top_p[r] == 1.0:
            # n is the whole budget, wherever it binds.
            reach = int(((s[r].max() - s[r]) <= 19.0).sum())
            limit = width if top_k[r] == 0 else min(width, int(top_k[r]))
            assert store.meta[r, 0].item() == min(limit, reach)


def test_a_draft_top_k_of_one_stores_and_drafts_the_lowest_id_max():
    rows, vocab, rank = 4, 3000, 32
    base = torch.randn(rows, vocab, device=DEV).to(torch.bfloat16)
    base[:, 77] = 9.0
    base[:, 2500] = 9.0
    prev = torch.zeros(rows, dtype=torch.int64, device=DEV)
    w = torch.zeros(vocab, rank, dtype=torch.bfloat16, device=DEV)
    temps, greedy, seed = draft_args(rows, step=2)
    top_k, top_p = filters(rows, top_k=1)
    store = empty_store(rows, 64)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    out, _ = sample(
        base, prev, w, w, store, row_base, 0, temps, greedy, seed, top_k, top_p
    )
    assert out.tolist() == [77] * rows
    assert store.meta.tolist() == [[1, 0]] * rows
    assert store.ids[:, 0].tolist() == [77] * rows
    assert store.probs[:, 0].tolist() == [1.0] * rows


def test_heavy_ties_and_non_finite_rows_store_what_was_counted():
    """At K >= V the store is every id within reach, however many tie on a
    level or at a tile's edge; a row with no finite max stores the delta on
    its raw argmax rather than anything the levels would make of NaN or inf."""
    vocab, rank = 20000, 32
    torch.manual_seed(11)
    rows = 6
    # Few distinct values, so whole levels (and the max) tie across tiles.
    base = torch.randint(0, 6, (rows, vocab), device=DEV).float().to(torch.bfloat16)
    base[2] = float("nan")
    base[3] = float("-inf")
    base[4, 123] = float("inf")
    base[4, 9000] = float("inf")
    base[5, ::7] = float("nan")
    prev = torch.zeros(rows, dtype=torch.int64, device=DEV)
    w = torch.zeros(vocab, rank, dtype=torch.bfloat16, device=DEV)
    temps, greedy, seed = draft_args(rows, temperature=0.4, step=6)
    store = empty_store(rows, vocab)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    out, s = sample(base, prev, w, w, store, row_base, 0, temps, greedy, seed)
    for r in (0, 1):
        assert_store_row_is_q(store, r, s[r], out[r].item(), vocab)
        assert store.meta[r, 0].item() == vocab  # 5 / 0.4 nats: all in reach
    # Rows whose max is not finite: the delta on the raw argmax.
    for r, expected in ((2, 0), (3, 0), (4, 123)):
        assert out[r].item() == expected, r
        assert store.meta[r].tolist() == [1, 0], r
        assert store.ids[r, 0].item() == expected and store.probs[r, 0] == 1.0
    # NaN entries are never stored; the finite rest is.
    ids, _, _ = stored(store, 5)
    assert torch.isfinite(s[5, ids.long()]).all()
    finite = torch.isfinite(s[5])
    assert ids.numel() == int(((s[5][finite].max() - s[5][finite]) <= 19).sum())


def test_two_runs_and_a_graph_replay_store_and_draft_bit_for_bit_alike():
    """What keeps TP ranks in agreement: no atomics anywhere, so the same key
    stores and drafts the same bits, eager or replayed."""
    rows, vocab, rank, width = 64, 163840, 64, 1024
    base, prev, w1, w2 = tables(rows, vocab, rank, seed=13)
    temps, greedy, seed = draft_args(rows, temperature=0.8, step=21)
    greedy[::5] = 1
    top_k, top_p = filters(rows)
    top_k[1::4] = 40
    top_p[2::4] = 0.8
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    row_base[3::7] = -1
    scratch = dspark_markov_sample_scratch(rows, vocab, DEV)
    out = torch.empty(rows, dtype=torch.int64, device=DEV)
    s = torch.empty(rows, vocab, device=DEV)
    store = empty_store(rows + 1, width)  # position 1

    def call():
        dspark_markov_sample(
            base,
            prev,
            w1,
            w2,
            out,
            *store,
            s,
            row_base,
            1,
            temps,
            greedy,
            top_k,
            top_p,
            seed,
            rows * vocab,
            scratch.f32,
            scratch.i32,
        )

    snapshots = []
    for _ in range(2):
        for t in store:
            t.fill_(-7)
        call()
        snapshots.append([t.clone() for t in (out, *store)])
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call()  # warm on the capture stream
        with torch.cuda.graph(graph, stream=stream):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    for t in (out, *store):
        t.fill_(-7)
    graph.replay()
    torch.cuda.synchronize()
    snapshots.append([t.clone() for t in (out, *store)])
    for a, b, c in zip(*snapshots):
        assert torch.equal(a, b) and torch.equal(a, c)


@pytest.mark.parametrize("width", [7, 48])
def test_sampled_ids_follow_the_q_they_publish(width):
    """Chi-square of the drafted ids against the stored probabilities, which
    at width 7 are a truncation of the softmax: the draw must follow what was
    stored, not the softmax it was cut from, and never leave the stored set."""
    rows, vocab, rank = 16384, 48, 16
    base, prev, w1, w2 = tables(1, vocab, rank, seed=5)
    base, prev = base.expand(rows, -1).contiguous(), prev.expand(rows).contiguous()
    temps, greedy, seed = draft_args(rows, temperature=1.3, step=11)
    store = empty_store(rows, width)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    out, _ = sample(base, prev, w1, w2, store, row_base, 0, temps, greedy, seed)
    ids, probs, _ = stored(store, 0)
    assert torch.equal(store.meta[:, 0], store.meta[:1, 0].expand(rows))
    assert torch.equal(store.ids, store.ids[:1].expand_as(store.ids))
    assert torch.equal(store.probs, store.probs[:1].expand_as(store.probs))
    counts = torch.bincount(out, minlength=vocab).double()
    assert counts.sum() == counts[ids.long()].sum(), "a draw left the stored set"
    expected = probs.double() * rows
    chi2 = ((counts[ids.long()] - expected) ** 2 / expected).sum().item()
    dof = ids.numel() - 1
    # ~6 standard deviations of the statistic above its mean.
    assert chi2 < dof + 6 * (2 * dof) ** 0.5, (chi2, dof)


def test_the_draws_depend_on_the_key_alone():
    """TP ranks run this independently and must draft the same ids."""
    rows, vocab, rank = 64, 1000, 32
    base, prev, w1, w2 = tables(rows, vocab, rank)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    results = []
    for step in (3, 3, 4):
        temps, greedy, seed = draft_args(rows, step=step)
        store = empty_store(rows, 256)
        out, _ = sample(base, prev, w1, w2, store, row_base, 0, temps, greedy, seed)
        results.append(out)
    assert torch.equal(results[0], results[1])
    assert not torch.equal(results[0], results[2])


def test_rows_past_two_gib_of_elements_are_addressed_correctly():
    """A `[2 * 512 * 16, V]` store at K = V passes 2**31 elements; so does the
    base logits' row stride at large batch. Both must be int64 in the kernel."""
    vocab, rank = 163840, 64
    big = 2**31 // vocab + 4  # first row past 2**31 elements, plus margin
    store = empty_store(big + 1, vocab)
    torch.manual_seed(3)
    stride = 2**31 + 64
    flat = torch.randn(stride + vocab, dtype=torch.bfloat16, device=DEV)
    base = flat.as_strided((2, vocab), (stride, 1))
    prev = torch.tensor([1, 2], dtype=torch.int64, device=DEV)
    w1 = torch.randn(vocab, rank, dtype=torch.bfloat16, device=DEV) * 0.1
    w2 = torch.randn(vocab, rank, dtype=torch.bfloat16, device=DEV) * 0.1
    temps, greedy, seed = draft_args(2)
    row_base = torch.tensor([big, 0], dtype=torch.int32, device=DEV)
    out, s = sample(base, prev, w1, w2, store, row_base, 0, temps, greedy, seed)
    assert_store_row_is_q(store, big, s[0], out[0].item(), vocab)
    assert_store_row_is_q(store, 0, s[1], out[1].item(), vocab)
    # And the scores themselves are the reference's, read at the far stride.
    _, ref_q = dspark_markov_sample_reference(
        base.contiguous(),
        prev,
        w1,
        w2,
        temps,
        greedy,
        torch.ones(2, vocab, device=DEV),
        vocab,
    )
    ids, probs, _ = stored(store, big)
    torch.testing.assert_close(probs, ref_q[0, ids.long()], rtol=1e-3, atol=1e-7)


def test_unsupported_weights_raise_rather_than_drafting_off_the_global_rng():
    rows, vocab, rank = 4, 64, 16
    base, prev, w1, w2 = tables(rows, vocab, rank)
    temps, greedy, seed = draft_args(rows)
    with pytest.raises(ValueError, match="cannot run"):
        sample(
            base,
            prev,
            w1.float(),
            w2.float(),
            empty_store(rows, 8),
            torch.arange(rows, dtype=torch.int32, device=DEV),
            0,
            temps,
            greedy,
            seed,
        )


def test_the_block_sampler_chains_the_op_through_its_bound_buffers():
    """K3's sampled block, the head's shared `sample_block_q`: position k
    drafts from x_{k-1} into row base + k, with the per-row inputs sliced to
    the running batch and the noise counter strided by the buffers' max batch,
    exactly as calling the op directly."""
    from types import SimpleNamespace

    from atom.models.dspark_draft import DraftSampling
    from atom.models.kimi_k3_dspark import DSparkMarkovHead, KimiK3DSpark

    max_bs, bs, steps, vocab, rank, width = 4, 3, 3, 3000, 32, 64
    head = DSparkMarkovHead(vocab, rank).to(DEV, torch.bfloat16)
    torch.manual_seed(19)
    base = torch.randn(bs, steps, vocab, device=DEV).to(torch.bfloat16)
    anchors = torch.randint(0, vocab, (bs,), dtype=torch.int32, device=DEV)
    temps, greedy, seed = draft_args(max_bs, temperature=0.9, step=4)
    top_k, top_p = filters(max_bs)
    top_k[1] = 20
    row_base = torch.tensor([6, -1, 0, -1], dtype=torch.int32, device=DEV)
    sampling = DraftSampling(
        q_store=empty_store(3 * steps, width),
        s_scratch=torch.empty(max_bs, vocab, device=DEV),
        q_row_base=row_base,
        temperatures=temps,
        greedy_mask=greedy,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        scratch=dspark_markov_sample_scratch(max_bs, vocab, DEV),
    )
    model = SimpleNamespace(markov_head=head, draft_sampling=sampling)
    drafts = KimiK3DSpark._sample_block(model, base, anchors)

    store = empty_store(3 * steps, width)
    prev = anchors
    for k in range(steps):
        out, _ = sample(
            base[:, k],
            prev,
            head.markov_w1.weight,
            head.markov_w2.weight,
            store,
            row_base[:bs],
            k,
            temps[:bs],
            greedy[:bs],
            seed,
            top_k[:bs],
            top_p[:bs],
            noise_offset=k * max_bs * vocab,
        )
        assert torch.equal(drafts[:, k].long(), out), k
        prev = out
    for a, b in zip(sampling.q_store, store):
        assert torch.equal(a, b)


# ---- verify: the probability-ratio rule ---------------------------------------


def verify_batch(drafts, width, logits, temperatures, row_base=None):
    """`rejection_sample` arguments for equal-width requests whose row r
    scores ``logits[r]`` and drafts ``drafts[r]``."""
    rows = drafts.numel()
    batch = rows // width
    ids, logit_rows = [], []
    for b in range(batch):
        start = len(ids)
        ids.append(ANCHOR)
        ids.extend(drafts[b * width : (b + 1) * width].tolist())
        logit_rows.extend(range(start, start + width))
    return {
        "input_ids": torch.tensor(ids, device=DEV, dtype=torch.int32),
        "target_logits_indices": torch.tensor(logit_rows, device=DEV),
        "num_spec_steps": width,
        "cu_num_draft_tokens": torch.arange(
            width, rows + 1, width, device=DEV, dtype=torch.int32
        ),
        "target_logits": logits.contiguous(),
        "bonus_token_ids": torch.zeros(batch, device=DEV, dtype=torch.int32),
        "bonus_logits_indices": torch.arange(
            width, (width + 1) * batch, width + 1, device=DEV
        ),
        "temperatures": temperatures,
        # Flat ragged q unless told otherwise: request b's rows start at b * width.
        "draft_probs_row_base": (
            torch.arange(batch, device=DEV, dtype=torch.int32) * width
            if row_base is None
            else row_base
        ),
    }


def sparse(vocab, support):
    """A distribution over ``vocab`` ids with ``support = {id: mass}``."""
    p = torch.zeros(vocab, device=DEV)
    for i, m in support.items():
        p[i] = m
    return p


def draw(dist, n):
    return torch.multinomial(dist, n, replacement=True).to(torch.int32)


def draft_q(q, drafts=None, *, dense_rows=None):
    """The `RatioDraftQ` of dense q rows: each row's support, ascending, at
    its q, with xslot naming ``drafts[r]`` (slot 0 where that is not stored,
    or with no drafts). A 1-D ``q`` is one row, or every draft's row when
    ``drafts`` is given. The dense scratch starts `stale`."""
    rows = q.view(-1, q.shape[-1])
    vocab = rows.shape[1]
    present = rows > 0
    order = torch.where(present, torch.arange(vocab, device=DEV), vocab).sort(1)[0]
    n = present.sum(1)
    ids = order[:, : max(int(n.max()), 1)]
    kept = ids < vocab
    probs = torch.where(kept, rows.gather(1, ids.clamp(max=vocab - 1)), 0.0)
    ids = torch.where(kept, ids, -1)
    if q.ndim == 1 and drafts is not None:
        count = drafts.numel()
        ids, probs, n = ids.expand(count, -1), probs.expand(count, -1), n.expand(count)
    xslot = torch.zeros_like(n)
    if drafts is not None:
        hit = ids[: drafts.numel()] == drafts.view(-1, 1)
        xslot[: drafts.numel()] = hit.int().argmax(1)
    store = SparseQStore(
        ids.to(torch.int32).contiguous(),
        probs.float().contiguous(),
        torch.stack((n, xslot), 1).to(torch.int32).contiguous(),
    )
    return RatioDraftQ(store, stale(dense_rows or ids.shape[0], vocab))


def stale(rows, vocab):
    """A dense scratch as the verify finds it: leftovers, every one of which
    reads as a plausible q (in [0, 1), unlike NaN, which the race maps to 0),
    so a residual that read an id the verify did not clear would show. Its
    own generator, so the test's stream is left as it was."""
    leftovers = torch.Generator(device=DEV).manual_seed(rows)
    return torch.rand(rows, vocab, device=DEV, generator=leftovers)


# Supports spread over three vocab tiles, the last one ragged.
VOCAB = 2 * _RATIO_BLOCK_V + 1000
P = {5: 0.5, _RATIO_BLOCK_V + 7: 0.3, VOCAB - 1: 0.2}
Q = {5: 0.2, _RATIO_BLOCK_V + 7: 0.1, VOCAB - 2: 0.7}


@triton.jit
def _rows_to_requests(cu_ptr, out_ptr, batch_size, steps):
    row = tl.program_id(0)
    tl.store(out_ptr + row, _request_of_row(cu_ptr, row, batch_size, steps))


@pytest.mark.parametrize(
    "kernel",
    [
        _ratio_stats_kernel,
        _ratio_filter_pass_kernel,
        _ratio_filter_floor_kernel,
        _ratio_filter_cut_kernel,
        _ratio_filter_count_kernel,
        _ratio_residual_race_kernel,
        _ratio_residual_pick_kernel,
        softmax_combine_kernel,
    ],
    ids=lambda k: k.fn.__name__,
)
def test_no_verify_kernel_specializes_on_the_batch(kernel):
    """The verify launches eagerly at whatever batch arrives; a row or request
    count Triton specialized on (== 1, % 16) would compile a new variant the
    first time such a batch arrived, mid-serve."""
    counts = {"num_tokens", "batch_size", "num_rows"}
    params = [p for p in kernel.params if p.name in counts]
    assert params
    assert all(p.do_not_specialize for p in params), [p.name for p in params]


@pytest.mark.parametrize(
    "widths", [[3], [2, 0, 3, 1, 0, 4], [1] * 37, [0, 5, 0], [7, 7, 7, 7, 7]]
)
def test_each_draft_row_finds_its_request(widths):
    """The stats kernel's row -> request lookup, against searchsorted."""
    cu = torch.tensor(widths, device=DEV).cumsum(0).to(torch.int32)
    rows = int(cu[-1])
    out = torch.empty(rows, dtype=torch.int32, device=DEV)
    _rows_to_requests[(rows,)](cu, out, len(widths), len(widths).bit_length() + 1)
    expected = torch.searchsorted(cu, torch.arange(rows, device=DEV), right=True)
    assert torch.equal(out.long(), expected)


def test_the_emitted_token_follows_p_across_vocab_tiles():
    torch.manual_seed(91)
    n = 40000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(q, n)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        draft_probs=draft_q(q, drafts),
        ratio_step=5,
    )
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)
    # Accepted at the rate sum min(p, q).
    rate = torch.minimum(p, q).sum().item()
    assert abs(accepted.float().mean().item() - rate) < 0.01


def test_a_residual_draw_never_lands_on_an_id_without_residual_mass():
    """Holds for every coin: a rejection's replacement has p > q."""
    torch.manual_seed(17)
    n = 20000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(q, n)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        draft_probs=draft_q(q, drafts),
        ratio_step=6,
    )
    replaced = out[accepted == 0, 0].long()
    assert replaced.numel() > 1000
    assert ((p - q)[replaced] > 0).all()


def test_a_residual_without_mass_falls_back_to_p():
    """p == q, and the draft is an id with no mass in either: it is rejected
    (q(x) = 0 is not the row x came from), relu(p - q) is 0 everywhere, and
    the replacement must still follow p rather than land on a fixed id."""
    torch.manual_seed(29)
    n = 30000
    p = sparse(VOCAB, P)
    drafts = torch.full((n,), 7, dtype=torch.int32, device=DEV)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, p.log().expand(n, -1), torch.ones(n, device=DEV)),
        draft_probs=draft_q(p, drafts),
        ratio_step=8,
    )
    assert (accepted == 0).all()
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)


def test_a_row_without_a_finite_z_emits_its_argmax_not_id_0():
    """One +inf logit, or one NaN logit, makes Z NaN and every p 0: the
    draft is rejected, the residual and the fallback race have no positive
    key, and the emitted id must be the row's raw argmax (the +inf id, or the
    finite max beside the NaN), never id 0, which carries no mass."""
    n = 64
    hot = _RATIO_BLOCK_V + 7
    torch.manual_seed(37)
    logits = torch.randn(2, VOCAB, device=DEV)
    logits[:, 0] = -30.0
    logits[0, hot] = float("inf")
    logits[1, hot] = 10.0
    logits[1, 20] = float("nan")
    q = sparse(VOCAB, {3: 0.5, 11: 0.5})
    drafts = draw(q, 2 * n)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, logits.repeat(n, 1), torch.ones(2 * n, device=DEV)),
        draft_probs=draft_q(q, drafts),
        ratio_step=9,
    )
    assert (accepted == 0).all()
    assert (out[:, 0] == hot).all()


def test_a_request_without_q_rejects_and_draws_from_p():
    torch.manual_seed(23)
    n = 30000
    p, q = sparse(VOCAB, P), sparse(VOCAB, Q)
    drafts = draw(p, n * 2)  # even drafts p itself would accept, given q
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(
            drafts,
            2,
            p.log().expand(n * 2, -1),
            torch.ones(n, device=DEV),
            row_base=torch.full((n,), -1, dtype=torch.int32, device=DEV),
        ),
        draft_probs=draft_q(q, dense_rows=n),
        ratio_step=7,
    )
    assert (accepted == 0).all()
    assert (out[:, 1:] == -1).all()
    freq = torch.bincount(out[:, 0].long(), minlength=VOCAB).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)


def test_q_is_read_through_the_row_base_not_the_batch_position():
    """Scattered blocks give exactly the flat layout's answer: same coins
    (keyed by verify row and step), same q values."""
    torch.manual_seed(31)
    batch_n, width, vocab = 64, 3, 500
    logits = torch.randn(batch_n * width, vocab, device=DEV)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    temps = torch.full((batch_n,), 0.9, device=DEV)
    args = verify_batch(drafts, width, logits, temps)
    flat = draft_q(q_rows, drafts)
    flat_out, flat_verdict = rejection_sample(**args, draft_probs=flat, ratio_step=9)
    slots = torch.randperm(2 * batch_n, device=DEV)[:batch_n]
    store = empty_store(2 * batch_n * width, flat.store.ids.shape[1])
    base = (slots * width).to(torch.int32)
    for k in range(width):
        for dst, src in zip(store, flat.store):
            dst[base.long() + k] = src.view(batch_n, width, -1)[:, k]
    args["draft_probs_row_base"] = base
    out, verdict = rejection_sample(
        **args, draft_probs=RatioDraftQ(store, flat.dense), ratio_step=9
    )
    assert torch.equal(out, flat_out) and torch.equal(verdict, flat_verdict)


@triton.jit
def _uniforms(out_ptr, seed, base, WIDE: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)
    if WIDE:
        u = tl.rand(seed, base + offs.to(tl.int64))
    else:
        u = tl.rand(seed, offs)
    tl.store(out_ptr + offs, u)


@pytest.mark.parametrize("wide", [False, True], ids=["int32-offset", "int64-offset"])
def test_the_key_purpose_word_separates_streams_at_any_offset_width(wide):
    """Draft noise, accept coins and residual race differ only in the key's
    high word (0, 1, 2); that word must reach Philox whether the offset is
    32-bit (the coins) or 64-bit (the draft noise and the residual race)."""
    from atom.model_ops.rejection_sampler import (
        _RATIO_ACCEPT_KEY,
        _RATIO_RESIDUAL_KEY,
    )

    step, n = 12345, 1024
    base = 5 * 2**31 if wide else 0
    streams = []
    for purpose in (0, _RATIO_ACCEPT_KEY, _RATIO_RESIDUAL_KEY):
        out = torch.empty(n, device=DEV)
        _uniforms[(1,)](out, purpose | step, base, WIDE=wide, N=n)
        streams.append(out)
    for i in range(3):
        for j in range(i + 1, 3):
            assert not torch.equal(streams[i], streams[j]), (i, j)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("filters", ["none", "per-row", "scalar"])
def test_the_preallocated_scratch_verifies_exactly_what_allocation_does(dtype, filters):
    """Production reuses one scratch, holding whatever the last step left in
    it; no buffer of it, the filter's included, may be read before written."""
    torch.manual_seed(43)
    batch_n, width, vocab = 48, 5, 3000
    logits = torch.randn(batch_n * width, vocab, device=DEV).to(dtype)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    temps = torch.ones(batch_n, device=DEV)
    temps[::4] = SAMPLER_EPS
    top_ks = torch.randint(1, 300, (batch_n,), device=DEV, dtype=torch.int32)
    top_ks[1::5] = -1
    top_ps = torch.rand(batch_n, device=DEV) * 0.9 + 0.05
    top_ps[2::5] = 1.0
    ks, ps = {
        "none": (None, None),
        "per-row": (top_ks, top_ps),
        "scalar": (int(top_ks[0]), float(top_ps[0])),
    }[filters]
    args = verify_batch(drafts, width, logits, temps)
    q = draft_q(q_rows, drafts)
    plain = rejection_sample(**args, draft_probs=q, ratio_step=11, top_ks=ks, top_ps=ps)
    scratch = ratio_verify_scratch(64, width, vocab, DEV)  # sized above the batch
    for garbage in (float("nan"), -1.0):
        scratch.f32.fill_(garbage)
        scratch.i32.random_(-(2**31), 2**31 - 1)
        reused = rejection_sample(
            **args,
            draft_probs=q,
            ratio_step=11,
            top_ks=ks,
            top_ps=ps,
            ratio_scratch=scratch,
        )
        assert torch.equal(plain[0], reused[0]) and torch.equal(plain[1], reused[1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_startup_warmup_compiles_every_variant_a_serving_verify_launches(
    dtype,
):
    """No warmup or capture forward verifies, so `warmup_ratio_verify` is what
    keeps the first sampled steps from compiling mid-serve: after it, a
    serving-shaped verify in each filter form `prepare_sample` hands over, at
    another batch, store contents and row bases, compiles nothing."""
    kernels = (
        _ratio_stats_kernel,
        softmax_combine_kernel,
        _ratio_filter_count_kernel,
        _ratio_filter_pass_kernel,
        _ratio_filter_floor_kernel,
        _ratio_filter_cut_kernel,
        _ratio_accept_kernel,
        _ratio_densify_clear_kernel,
        _ratio_densify_scatter_kernel,
        _ratio_residual_race_kernel,
        _ratio_residual_pick_kernel,
    )
    max_bs, batch_n, width, q_width, vocab = 16, 5, 4, 8, 3000
    scratch = ratio_verify_scratch(max_bs, width, vocab, DEV)
    caches = [k.device_caches[torch.cuda.current_device()][0] for k in kernels]
    saved = [dict(cache) for cache in caches]
    for cache in caches:
        cache.clear()
    try:
        warmup_ratio_verify(width, vocab, q_width, dtype, scratch)
        torch.cuda.synchronize()
        warmed = [len(cache) for cache in caches]
        assert all(warmed), [k.fn.__name__ for k, n in zip(kernels, warmed) if not n]

        torch.manual_seed(53)
        rows = batch_n * width
        q_rows = torch.softmax(torch.randn(rows, vocab, device=DEV), -1)
        drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
        # Sliced out of the per-row buffers, as `prepare_sample` hands them.
        top_ks = torch.randint(1, 300, (max_bs,), device=DEV, dtype=torch.int32)
        top_ps = torch.rand(max_bs, device=DEV) * 0.9 + 0.05
        forms = (
            (None, None),
            (20, None),
            (300, None),
            (None, 0.8),
            (top_ks[:batch_n], None),
            (None, top_ps[:batch_n]),
            (top_ks[:batch_n], 0.8),
            (top_ks[:batch_n], top_ps[:batch_n]),
        )
        store = SparseQStore.allocate(max_bs * width, q_width, DEV)
        for t in store:
            t.zero_()
        for top_ks_form, top_ps_form in forms:
            # A step's target rows are gathered out of its whole logits.
            logits = torch.randn(rows + batch_n, vocab, device=DEV).to(dtype)
            args = verify_batch(
                drafts,
                width,
                logits[:rows],
                torch.ones(max_bs, device=DEV)[:batch_n],
                row_base=torch.arange(batch_n, device=DEV, dtype=torch.int32) * width,
            )
            rejection_sample(
                **args,
                draft_probs=RatioDraftQ(store, stale(max_bs, vocab)),
                ratio_step=3,
                top_ks=top_ks_form,
                top_ps=top_ps_form,
                ratio_scratch=scratch,
            )
        torch.cuda.synchronize()
        grew = [
            k.fn.__name__
            for k, cache, n in zip(kernels, caches, warmed)
            if len(cache) != n
        ]
        assert not grew, f"compiled while serving: {grew}"
    finally:
        for cache, entries in zip(caches, saved):
            cache.clear()
            cache.update(entries)


def test_the_same_key_gives_the_same_verdict():
    """What keeps TP ranks in agreement without exchanging coins."""
    torch.manual_seed(37)
    batch_n, width, vocab = 128, 4, 777
    logits = torch.randn(batch_n * width, vocab, device=DEV)
    q_rows = torch.softmax(torch.randn(batch_n * width, vocab, device=DEV), -1)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    args = verify_batch(drafts, width, logits, torch.ones(batch_n, device=DEV))
    q = draft_q(q_rows, drafts)
    a = rejection_sample(**args, draft_probs=q, ratio_step=40)
    b = rejection_sample(**args, draft_probs=q, ratio_step=40)
    c = rejection_sample(**args, draft_probs=q, ratio_step=41)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def tied_logits(rows, vocab, low, high):
    """bf16 logits whose max is an exact tie between ids ``low < high``."""
    logits = torch.randn(rows, vocab, device=DEV).to(torch.bfloat16)
    logits[:, low] = 30.0
    logits[:, high] = 30.0
    return logits


@pytest.mark.parametrize("draft_first", ["tie-low", "tie-high", "other"])
def test_a_greedy_row_is_exact_match_whatever_its_batch_mates(draft_first):
    """On a bf16 tie the greedy row's p is 0.5/0.5, not a delta, so the ratio
    rule would accept the high tie half the time. Exact match never does, and
    must not care who else is in the batch."""
    torch.manual_seed(41)
    width, vocab, low, high = 3, 1024, 3, 700
    first = {"tie-low": low, "tie-high": high, "other": 11}[draft_first]
    greedy_drafts = torch.tensor([first, low, low], dtype=torch.int32, device=DEV)
    greedy_logits = tied_logits(width, vocab, low, high)

    # Reference: the flag-off kernel on the greedy request alone.
    ref_out, ref_verdict = rejection_sample(
        **verify_batch(
            greedy_drafts,
            width,
            greedy_logits,
            torch.full((1,), SAMPLER_EPS, device=DEV),
        ),
        draft_probs=None,
    )
    expected_accept = {"tie-low": 3, "tie-high": 0, "other": 0}[draft_first]
    assert ref_verdict[0, 0].item() == expected_accept
    if expected_accept == 0:
        assert ref_out[0, 0].item() == low

    # The same request among sampled ones, at several positions and keys.
    sampled_n = 7
    for slot in (0, 3, sampled_n):
        for step in range(5):
            others = torch.randint(0, vocab, (sampled_n * width,), device=DEV)
            drafts = torch.cat(
                (others[: slot * width], greedy_drafts, others[slot * width :])
            ).to(torch.int32)
            logits = torch.cat(
                (
                    torch.randn(slot * width, vocab, device=DEV).to(torch.bfloat16),
                    greedy_logits,
                    torch.randn((sampled_n - slot) * width, vocab, device=DEV).to(
                        torch.bfloat16
                    ),
                )
            )
            temps = torch.ones(sampled_n + 1, device=DEV)
            temps[slot] = SAMPLER_EPS
            # q = p for the greedy row too, the case the ratio rule gets wrong.
            q = torch.softmax(logits.float(), -1)
            out, verdict = rejection_sample(
                **verify_batch(drafts, width, logits, temps),
                draft_probs=draft_q(q, drafts),
                ratio_step=100 + step,
            )
            assert torch.equal(out[slot, :width], ref_out[0, :width]), (slot, step)
            assert torch.equal(verdict[:2, slot], ref_verdict[:2, 0]), (slot, step)


def test_the_target_law_holds_at_the_real_vocab():
    torch.manual_seed(53)
    vocab, n = 163840, 6000
    support_p = {7: 0.4, 81920: 0.35, vocab - 1: 0.25}
    support_q = {7: 0.6, 81920: 0.1, vocab - 3: 0.3}
    p, q = sparse(vocab, support_p), sparse(vocab, support_q)
    drafts = draw(q, n)
    logits = p.log().to(torch.bfloat16).expand(n, -1)
    out, _ = rejection_sample(
        **verify_batch(
            drafts,
            1,
            logits,
            torch.ones(n, device=DEV),
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        ),
        draft_probs=draft_q(q, dense_rows=n),
        ratio_step=3,
    )
    p_bf16 = torch.softmax(p.log().to(torch.bfloat16).float(), -1)
    freq = torch.bincount(out[:, 0].long(), minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_bf16, rtol=0, atol=0.025)


# ---- verify: the sparse store reads as the dense q it describes ----------------


@triton.jit(do_not_specialize=["num_spec_steps", "accept_seed"])
def _dense_accept_kernel(
    output_token_ids_ptr,
    verdict_ptr,
    residual_row_ptr,
    cu_num_draft_tokens_ptr,
    input_ids_ptr,
    target_logits_indices_ptr,
    logits_ptr,
    temperatures_ptr,
    row_max_ptr,
    row_inv_z_ptr,
    row_argmax_ptr,
    q_ptr,
    q_row_base_ptr,
    bonus_token_ids_ptr,
    bonus_logits_indices_ptr,
    num_spec_steps,
    vocab_size,
    logits_row_stride,
    greedy_temperature,
    accept_seed,
):
    """The accept walk as it was over a dense ``[rows, V]`` q, kept as the
    reference the sparse walk must reproduce verdict for verdict."""
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx
    temp = tl.load(temperatures_ptr + req_idx).to(tl.float32)
    greedy = temp <= greedy_temperature
    q_base = tl.load(q_row_base_ptr + req_idx)
    has_q = q_base >= 0
    out_row = output_token_ids_ptr + req_idx * (num_spec_steps + 1)
    INVALID_TOKEN: tl.constexpr = -1
    rejected = False
    reject_pos = -1
    last_checked = -1
    for pos in range(num_draft_tokens):
        if rejected:
            tl.store(out_row + pos, INVALID_TOKEN)
        else:
            row = start_idx + pos
            x = _draft_token(input_ids_ptr, target_logits_indices_ptr, row)
            target_argmax = tl.load(row_argmax_ptr + row)
            if greedy:
                accept = x == target_argmax
            else:
                logit = tl.load(logits_ptr + row.to(tl.int64) * logits_row_stride + x)
                p = tl.exp(
                    logit.to(tl.float32) / temp - tl.load(row_max_ptr + row)
                ) * tl.load(row_inv_z_ptr + row)
                q = tl.load(
                    q_ptr + (q_base + pos).to(tl.int64) * vocab_size + x,
                    mask=has_q,
                    other=0.0,
                )
                u = tl.rand(accept_seed, row)
                accept = (q > 0.0) & (q <= 1.0) & (u * q < p)
            if accept:
                tl.store(out_row + pos, tl.cast(x, tl.int32))
            else:
                rejected = True
                reject_pos = pos
                if greedy:
                    tl.store(out_row + pos, tl.cast(target_argmax, tl.int32))
            last_checked += 1
    residual_row = -1
    if rejected and not greedy:
        residual_row = start_idx + reject_pos
    tl.store(residual_row_ptr + req_idx, residual_row)
    _finish_request(
        output_token_ids_ptr,
        verdict_ptr,
        bonus_token_ids_ptr,
        bonus_logits_indices_ptr,
        req_idx,
        num_draft_tokens,
        last_checked,
        rejected,
        num_spec_steps,
    )


def dense_verify(args, q_dense, ratio_step):
    """The verify over a dense q: the same stats, race and pick kernels the
    sparse path runs, around `_dense_accept_kernel`, the race reading the
    rejecting position's row of ``q_dense`` copied to the request's row, where
    the sparse path densifies it."""
    logits = args["target_logits"]
    cu = args["cu_num_draft_tokens"]
    steps = args["num_spec_steps"]
    temps = args["temperatures"].float().contiguous()
    base = args["draft_probs_row_base"]
    num_tokens, vocab = logits.shape
    batch_n = cu.numel()
    tiles = triton.cdiv(vocab, _RATIO_BLOCK_V)

    def f32(*shape):
        return torch.empty(shape, device=DEV)

    def i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=DEV)

    part_max, part_sum, part_val = (f32(tiles, num_tokens) for _ in range(3))
    part_idx, row_argmax = i32(tiles, num_tokens), i32(num_tokens)
    row_max, row_inv_z = f32(num_tokens), f32(num_tokens)
    out, verdict, residual = i32(batch_n, steps + 1), i32(3, batch_n), i32(batch_n)
    keys, key_idx = f32(tiles, batch_n), i32(tiles, batch_n)
    _ratio_stats_kernel[(num_tokens, tiles)](
        logits,
        cu,
        temps,
        part_max,
        part_sum,
        part_val,
        part_idx,
        num_tokens,
        vocab,
        vocab,
        batch_n,
        batch_n.bit_length() + 1,
        BLOCK_V=_RATIO_BLOCK_V,
        num_warps=8,
    )
    softmax_combine_kernel[(num_tokens,)](
        part_max,
        part_sum,
        row_max,
        row_inv_z,
        part_val,
        part_idx,
        row_argmax,
        num_tokens,
        tiles,
        BLOCK_TILE=triton.next_power_of_2(tiles),
        WITH_ARGMAX=True,
        num_warps=4,
    )
    _dense_accept_kernel[(batch_n,)](
        out,
        verdict,
        residual,
        cu,
        args["input_ids"],
        args["target_logits_indices"],
        logits,
        temps,
        row_max,
        row_inv_z,
        row_argmax,
        q_dense,
        base,
        args["bonus_token_ids"],
        args["bonus_logits_indices"],
        steps,
        vocab,
        vocab,
        SAMPLER_EPS,
        _RATIO_ACCEPT_KEY | ratio_step,
        num_warps=1,
    )
    # The rejecting position's own q row at its request's row, as the sparse
    # path densifies it. A request without one gets some row the race masks
    # off, since it decides from ``residual`` and ``base`` alone.
    start = torch.cat((cu.new_zeros(1), cu[:-1]))
    has_row = (residual >= 0) & (base >= 0)
    q_row = torch.where(has_row, base + residual - start, 0)
    _ratio_residual_race_kernel[(batch_n, tiles)](
        residual,
        logits,
        temps,
        row_max,
        row_inv_z,
        None,  # no floor: unfiltered
        q_dense[q_row.long()],
        base,
        keys,
        key_idx,
        batch_n,
        vocab,
        vocab,
        _RATIO_RESIDUAL_KEY | ratio_step,
        BLOCK_V=_RATIO_BLOCK_V,
        FILTER=False,
        num_warps=8,
    )
    _ratio_residual_pick_kernel[(batch_n,)](
        out,
        residual,
        cu,
        keys,
        key_idx,
        logits,
        temps,
        row_max,
        row_inv_z,
        row_argmax,
        None,
        batch_n,
        tiles,
        steps,
        vocab,
        vocab,
        _RATIO_RESIDUAL_KEY | ratio_step,
        BLOCK_TILE=triton.next_power_of_2(tiles),
        BLOCK_V=_RATIO_BLOCK_V,
        FILTER=False,
        num_warps=1,
    )
    return out, verdict


def top_of(logits, k):
    """softmax over each row's k largest logits, 0 elsewhere."""
    kept = logits.topk(k, dim=-1)
    q = torch.zeros_like(logits)
    return q.scatter_(-1, kept.indices, torch.softmax(kept.values, -1))


def test_the_sparse_store_verifies_bit_for_bit_as_its_dense_q():
    """Densified, the store is a dense q; the verify over the store and the
    verify over that q must emit the same tokens and verdicts, coin for coin:
    greedy rows, requests without q, an emitted slot that does not hold the
    draft, and a draft the store does not hold all included."""
    torch.manual_seed(61)
    batch_n, width, vocab = 96, 4, 2 * _RATIO_BLOCK_V + 500
    rows = batch_n * width
    q_logits = torch.randn(rows, vocab, device=DEV) * 2
    q_rows = top_of(q_logits, 40)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    drafts[5::23] = vocab - 1  # not stored: q(x) = 0 whatever xslot says
    logits = (q_logits + torch.randn(rows, vocab, device=DEV)).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.6, SAMPLER_EPS, 1.3], device=DEV).repeat(batch_n // 4)
    args = verify_batch(drafts, width, logits, temps)
    args["draft_probs_row_base"][7::11] = -1
    q = draft_q(q_rows, drafts, dense_rows=batch_n)
    n, xslot = q.store.meta.unbind(1)
    # Slots that name another stored id: the walk must find x's true q.
    xslot[1::9] = (xslot[1::9] + 1) % n[1::9]
    for step in (0, 1, 2):
        out, verdict = rejection_sample(**args, draft_probs=q, ratio_step=step)
        ref_out, ref_verdict = dense_verify(args, q_rows, step)
        assert torch.equal(out, ref_out), step
        assert torch.equal(verdict, ref_verdict), step
    sampled = temps > SAMPLER_EPS
    accepted = verdict[0][sampled]
    assert (accepted > 0).any() and (accepted < width).any()


@pytest.mark.parametrize("others", [1, 1500], ids=["first-chunk", "past-a-chunk"])
def test_a_slot_that_does_not_hold_the_draft_still_reads_its_true_q(others):
    """The walk reads q(x) at the emitted slot; where that slot holds another
    id it must look x up -- the rate is min(1, p(x) / q(x)) for x's own q, not
    the slot's, and a draft the store does not hold has q 0 and never passes.
    The lookup walks the stored entries a chunk at a time, so x also sits
    past the first chunk."""
    torch.manual_seed(67)
    n, vocab = 20000, 3000
    x, outside = 2999, 5
    p = sparse(vocab, {x: 0.3, outside: 0.7})
    store = empty_store(1, 2048)
    # `others` ids of 0.01 between them, then x at slot `others`.
    store.ids[0, : others + 1] = torch.tensor(
        [*range(1000, 1000 + others), x], device=DEV
    )
    store.probs[0, :others] = 0.01 / others
    store.probs[0, others] = 0.99
    store.meta[0] = torch.tensor([others + 1, 0], device=DEV)  # names id 1000
    for draft, rate in ((x, 0.3 / 0.99), (outside, 0.0)):
        drafts = torch.full((n,), draft, dtype=torch.int32, device=DEV)
        args = verify_batch(
            drafts,
            1,
            p.log().expand(n, -1),
            torch.ones(n, device=DEV),
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        )
        out, (accepted, _, _) = rejection_sample(
            **args, draft_probs=RatioDraftQ(store, stale(n, vocab)), ratio_step=13
        )
        assert abs(accepted.float().mean().item() - rate) < 0.015, draft
        # relu(p - q) is all on `outside`: x's 0.3 is below its q of 0.99.
        assert (out[accepted == 0, 0] == outside).all(), draft


@pytest.mark.parametrize("temperature", [0.3, 1.0])
@pytest.mark.parametrize("width", [1, 7, 1024, 4096, 6000])
def test_the_first_token_follows_p_at_any_store_width(width, temperature):
    """End to end -- the sampled draft writes the store, the verify reads it --
    the first emitted token follows the target's p, whatever K cut q to, and
    never lands on an id p gives no mass."""
    torch.manual_seed(71)
    n, vocab, rank = 40000, 6000, 32
    target = torch.randn(vocab, device=DEV) * 3
    target[torch.rand(vocab, device=DEV) < 0.3] = float("-inf")
    base = (target.nan_to_num(neginf=-20.0) + torch.randn(vocab, device=DEV)).to(
        torch.bfloat16
    )
    _, prev, w1, w2 = tables(1, vocab, rank, seed=73)
    temps, greedy, seed = draft_args(n, temperature=temperature, step=17)
    store = empty_store(n, width)
    row_base = torch.arange(n, dtype=torch.int32, device=DEV)
    drafts, _ = sample(
        base.expand(n, -1).contiguous(),
        prev.expand(n).contiguous(),
        w1,
        w2,
        store,
        row_base,
        0,
        temps,
        greedy,
        seed,
    )
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts.to(torch.int32), 1, target.expand(n, -1), temps),
        draft_probs=RatioDraftQ(store, stale(n, vocab)),
        ratio_step=19,
    )
    p = torch.softmax(target / temperature, -1)
    first = out[:, 0].long()
    assert (p[first] > 0).all(), "emitted an id the target gives no mass"
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p, rtol=0, atol=0.01)
    assert 0 < accepted.float().mean().item() < 1


@pytest.mark.parametrize("width", [7, 50])
def test_a_chained_block_follows_the_autoregressive_target(width):
    """Three chained positions, each drafted from the last and scored by a
    target conditioned on it: every emitted token follows p(. | the token
    emitted before it) -- position 0 given the anchor, later positions among
    the requests that reached them -- and none lands where p is 0."""
    torch.manual_seed(79)
    n, vocab, rank, steps, anchor = 60000, 50, 16, 3, 3
    base, _, w1, w2 = tables(1, vocab, rank, seed=83)
    # The target near the draft (the draft's own logits given each previous
    # id, perturbed), so enough requests reach the last position.
    _, draft_logits = sample(
        base.expand(vocab, -1).contiguous(),
        torch.arange(vocab, device=DEV),
        w1,
        w2,
        empty_store(vocab, 1),
        torch.arange(vocab, dtype=torch.int32, device=DEV),
        0,
        *draft_args(vocab),
    )
    table = draft_logits + torch.randn(vocab, vocab, device=DEV) * 0.5
    table[torch.rand(vocab, vocab, device=DEV) < 0.2] = float("-inf")
    table[:, 0] = 1.0  # every row keeps some mass
    temps, greedy, seed = draft_args(n, step=23)
    store = empty_store(n * steps, width)
    row_base = torch.arange(0, n * steps, steps, dtype=torch.int32, device=DEV)
    base = base.expand(n, -1).contiguous()
    prev = torch.full((n,), anchor, dtype=torch.int64, device=DEV)
    drafts, scored = [], []
    for k in range(steps):
        scored.append(table[prev])
        out, _ = sample(
            base,
            prev,
            w1,
            w2,
            store,
            row_base,
            k,
            temps,
            greedy,
            seed,
            noise_offset=k * n * vocab,
        )
        drafts.append(out)
        prev = out
    drafts = torch.stack(drafts, 1).view(-1).to(torch.int32)
    logits = torch.stack(scored, 1).view(n * steps, vocab)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, steps, logits, temps),
        draft_probs=RatioDraftQ(store, stale(n, vocab)),
        ratio_step=29,
    )
    cond = torch.softmax(table, -1)
    given = torch.full((n,), anchor, dtype=torch.int64, device=DEV)
    for k in range(steps):
        reached = accepted >= k
        y = out[reached, k].long()
        prior = given[reached] if k == 0 else out[reached, k - 1].long()
        assert (y >= 0).all()
        assert (cond[prior, y] > 0).all(), f"position {k} emitted a zero-mass id"
        assert y.numel() > 2000, f"too few requests reached position {k}"
        # Chi-square against the mixture of the rows' conditionals.
        expected = cond[prior].double().sum(0)
        counts = torch.bincount(y, minlength=vocab).double()
        live = expected > 0
        chi2 = ((counts - expected)[live] ** 2 / expected[live]).sum().item()
        dof = int(live.sum()) - 1
        assert chi2 < dof + 6 * (2 * dof) ** 0.5, (k, chi2, dof)


# ---- verify: top-k / top-p, the target's filtered p ---------------------------


def filtered_p(logits, temps, top_ks=None, top_ps=None, widths=None):
    """`_ratio_target_p` over requests of ``widths`` draft rows each (one by
    default), filters and temperatures per request: ``(kept, p_f)`` per row."""
    rows, vocab = logits.shape
    widths = torch.ones(rows, dtype=torch.int64) if widths is None else widths
    cu = widths.cumsum(0).to(device=DEV, dtype=torch.int32)
    assert int(cu[-1]) == rows
    (
        (row_max, row_inv_z, row_floor, part_max, part_sum, part_val, _, *filt_f32),
        (row_argmax, part_idx, _, _, *filt_i32),
    ) = take(*_ratio_scratch_shapes(rows, len(widths), vocab), DEV, None, None)
    temps = temps.float().contiguous()
    assert _ratio_target_p(
        logits,
        cu,
        temps,
        top_ks,
        top_ps,
        (row_max, row_inv_z, row_argmax, row_floor),
        (
            part_max,
            part_sum,
            part_val,
            part_idx,
            filt_f32[0],
            filt_i32[0],
            filt_f32[1],
            filt_i32[1],
        ),
    )
    x = logits.float()
    kept = x >= row_floor[:, None]
    temps = temps.repeat_interleave(widths.to(DEV))
    p = torch.exp(x / temps[:, None] - row_max[:, None]) * row_inv_z[:, None]
    return kept, torch.where(kept, p, 0.0)


def sampler_kept(logits, temps, top_ks=None, top_ps=None):
    """The ids `Sampler`'s aiter kernels may draw, and its filtered p: fewer
    than top_k ids strictly above, whose p sums to less than top_p, both over
    the unfiltered p `Sampler` computes (`_topk_topp_sample`). Per-row tensors
    or None; -1 and 1.0 are off. Sums in double, so a row whose cut is within
    rounding of its top_p is no check of the mass rule."""
    rows, vocab = logits.shape
    probs = (logits / temps[:, None]).softmax(dim=-1, dtype=torch.float32)
    sorted_p, order = probs.sort(-1, descending=True)
    starts = torch.ones_like(sorted_p, dtype=torch.bool)
    starts[:, 1:] = sorted_p[:, 1:] != sorted_p[:, :-1]
    slot = torch.arange(vocab, device=DEV).expand(rows, vocab)
    # Ids strictly above each id: the slot its tie group starts at.
    higher = torch.where(starts, slot, 0).cummax(-1).values
    mass = (sorted_p.double().cumsum(-1) - sorted_p.double()).gather(1, higher)
    k = vocab + 1 if top_ks is None else torch.where(top_ks > 0, top_ks, vocab + 1)
    p = 2.0 if top_ps is None else torch.where(top_ps < 1, top_ps.double(), 2.0)
    k = torch.as_tensor(k, device=DEV).view(-1, 1)
    p = torch.as_tensor(p, device=DEV).view(-1, 1)
    keep = torch.zeros_like(starts).scatter_(1, order, (higher < k) & (mass < p))
    p_f = torch.where(keep, probs.double(), 0.0)
    return keep, (p_f / p_f.sum(-1, keepdim=True)).float()


def tied_rows(rows, vocab, seed, dtype=torch.bfloat16):
    """Logits, every other row on a half-integer grid (ties everywhere, so
    every cut lands in a tie group), none near 0 (where `Sampler`'s fp32 p
    could round two distinct logits onto one value)."""
    torch.manual_seed(seed)
    logits = torch.randn(rows, vocab, device=DEV) * 2
    logits[::2] = (logits[::2] * 2).round() / 2
    logits = torch.where(logits.abs() < 0.05, 0.05, logits)
    return logits.to(dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_the_kept_set_is_samplers_bit_for_bit(dtype):
    """Top-k alone is a count, so the set is exact on every row (`tied_rows`
    keeps clear of 0; see the test after next): cuts inside a tie group keep
    the group whole, top_k = 1 keeps the max's group, top_k >=
    V and -1 keep everything. Top-p is compared on the rows whose cut is not
    within rounding of top_p; top-p 1.0 keeps everything. fp32 logits take
    the 32-bit descent."""
    rows, vocab = 96, 3 * _RATIO_BLOCK_V + 123
    logits = tied_rows(rows, vocab, 113, dtype)
    temps = torch.rand(rows, device=DEV) + 0.3
    top_ks = torch.randint(1, 400, (rows,), device=DEV, dtype=torch.int32)
    top_ks[::8], top_ks[1::8], top_ks[2::8], top_ks[3::8] = 1, vocab, vocab + 9, -1
    top_ps = torch.rand(rows, device=DEV) * 0.98 + 0.01
    top_ps[4::8] = 1.0
    nudge = torch.full_like(top_ps, 1e-5)
    for ks, ps in (
        (top_ks, None),
        (None, top_ps),
        (top_ks, top_ps),
        (int(top_ks[5]), None),
        (None, float(top_ps[5])),
        # `prepare_sample` picks scalar or per-row for each independently.
        (top_ks, float(top_ps[5])),
        (int(top_ks[5]), top_ps),
    ):
        kept, p_f = filtered_p(logits, temps, ks, ps)
        k_rows = ks if ks is None or isinstance(ks, torch.Tensor) else top_ks * 0 + ks
        p_rows = ps if ps is None or isinstance(ps, torch.Tensor) else top_ps * 0 + ps
        ref, ref_p = sampler_kept(logits, temps, k_rows, p_rows)
        robust = torch.ones(rows, dtype=torch.bool, device=DEV)
        if p_rows is not None:
            low = sampler_kept(logits, temps, k_rows, p_rows - nudge)[0]
            high = sampler_kept(logits, temps, k_rows, p_rows + nudge)[0]
            robust = (low == high).all(1)
            assert robust.float().mean() > 0.8
        assert torch.equal(kept[robust], ref[robust]), (ks, ps)
        torch.testing.assert_close(p_f[robust], ref_p[robust], rtol=1e-4, atol=1e-7)
        torch.testing.assert_close(
            p_f.double().sum(-1),
            torch.ones(rows, device=DEV).double(),
            rtol=0,
            atol=1e-5,
        )
    # `Sampler`'s own top-k kernel keeps that set too, tie for tie.
    kept, _ = filtered_p(logits, temps, top_ks, None)
    on = top_ks > 0
    probs = (logits / temps[:, None]).softmax(dim=-1, dtype=torch.float32)
    renorm = torch.ops.aiter.top_k_renorm_probs(
        probs[on].contiguous(), top_ks[on].contiguous(), 0
    )
    assert torch.equal(renorm > 0, kept[on])


def test_ragged_requests_each_keep_their_own_set_at_the_real_vocab():
    """Requests of 0 to 7 draft rows, each with its own temperature, top-k and
    top-p, at V=163840: every row keeps its own request's set. 64 rows split
    the last count into programs of three tiles each, the last one ragged."""
    vocab = 163840
    widths = torch.tensor([3, 0, 7, 1, 5, 0, 2, 7, 4, 6, 0, 7, 7, 6, 2, 7])
    rows, batch_n = int(widths.sum()), widths.numel()
    logits = tied_rows(rows, vocab, 173)
    temps = torch.rand(batch_n, device=DEV) + 0.3
    top_ks = torch.randint(1, 400, (batch_n,), device=DEV, dtype=torch.int32)
    top_ks[::5], top_ks[1::7] = -1, vocab
    top_ps = torch.rand(batch_n, device=DEV) * 0.98 + 0.01
    top_ps[2::5] = 1.0
    nudge = torch.full((rows,), 1e-5, device=DEV)

    def per_row(x):
        return None if x is None else x.repeat_interleave(widths.to(DEV))

    for ks, ps in ((top_ks, None), (None, top_ps), (top_ks, top_ps)):
        kept, p_f = filtered_p(logits, temps, ks, ps, widths)
        k_rows, p_rows = per_row(ks), per_row(ps)
        ref, ref_p = sampler_kept(logits, per_row(temps), k_rows, p_rows)
        robust = torch.ones(rows, dtype=torch.bool, device=DEV)
        if p_rows is not None:
            low = sampler_kept(logits, per_row(temps), k_rows, p_rows - nudge)[0]
            high = sampler_kept(logits, per_row(temps), k_rows, p_rows + nudge)[0]
            # A request's rows stand or fall together, and the real vocab's
            # tie groups are heavy: fewer cuts clear the nudge than above.
            robust = (low == high).all(1)
            assert robust.float().mean() > 0.5
        assert torch.equal(kept[robust], ref[robust]), (ks, ps)
        torch.testing.assert_close(p_f[robust], ref_p[robust], rtol=1e-4, atol=1e-7)


def test_a_cut_inside_a_tie_group_keeps_the_whole_group():
    """Five ids tie for second: top-k 2 keeps all six, and top-p just past
    the max's mass keeps them too (their mass above is the max's alone)."""
    logits = torch.full((1, 64), -4.0, device=DEV)
    logits[0, 9] = 3.0
    logits[0, [2, 17, 33, 40, 63]] = 1.0
    logits = logits.to(torch.bfloat16)
    temps = torch.ones(1, device=DEV)
    top = torch.softmax(logits.float(), -1)[0, 9].item()
    for ks, ps in ((2, None), (None, top + 1e-3)):
        kept, _ = filtered_p(logits, temps, ks, ps)
        assert kept[0].nonzero().view(-1).tolist() == [2, 9, 17, 33, 40, 63]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_a_counted_top_k_keeps_what_the_descent_keeps(monkeypatch, dtype):
    """A top-k of at most 256 alone is counted from the stats pass's run
    maxima, not descended: the same floor bit for bit, and p_f to rounding,
    over tied rows, top_k 1 to 256 (cuts past one gather of runs), masked,
    NaN, +-inf and all--inf rows, an allow-list of 10 ids in -inf and
    plateaus that tie thousands of runs at the k-th largest max, scalar and
    per request, at the real vocab and at one whose last tile is ragged. A
    top_p that is off (1.0, scalar or per request) changes nothing, the
    +inf row included: it gets its top-k floor whatever its batch passes."""
    import atom.model_ops.rejection_sampler as rs
    import atom.model_ops.sampling_filter as sf

    for vocab in (163840, 3 * _RATIO_BLOCK_V + 123):
        rows = 40
        logits = tied_rows(rows, vocab, 191, torch.float32)
        logits[0, torch.rand(vocab, device=DEV) < 0.3] = float("-inf")
        logits[1, :7] = float("nan")
        logits[2] = float("nan")
        logits[3] = float("-inf")
        logits[4, 11] = float("inf")
        logits[5, 1::2] = 0.0
        logits[5, ::2] = -0.0
        logits[6, : vocab // 2] = logits[6].max()
        logits[7] = float("-inf")
        logits[7, torch.randperm(vocab, device=DEV)[:10]] = torch.randn(10, device=DEV)
        logits[8] = 0.0
        logits[8, 5] = 10.0
        logits[9] = -1.0
        logits[9, torch.randperm(vocab, device=DEV)[:30]] = torch.arange(
            30, device=DEV, dtype=torch.float32
        )
        logits = logits.to(dtype)
        temps = torch.rand(rows, device=DEV) + 0.3
        top_ks = torch.randint(1, 257, (rows,), device=DEV, dtype=torch.int32)
        top_ks[::7], top_ks[1::9], top_ks[2::11] = 1, 256, -1
        for ks in (top_ks, 1, 20, 50, 256):
            counted = filtered_p(logits, temps, ks)
            with monkeypatch.context() as m:
                for module in (rs, sf):
                    m.setattr(module, "_FILTER_COUNT_MAX_K", 0)
                descended = filtered_p(logits, temps, ks)
            assert torch.equal(counted[0], descended[0]), (vocab, ks)
            torch.testing.assert_close(
                counted[1], descended[1], rtol=1e-5, atol=1e-9, equal_nan=True
            )
            for ps in (1.0, torch.ones(rows, device=DEV)):
                p_off = filtered_p(logits, temps, ks, ps)
                assert torch.equal(p_off[0], counted[0]), (vocab, ks, ps)
                assert torch.equal(p_off[1].isnan(), counted[1].isnan())


def test_the_count_rule_splits_only_what_samplers_p_rounds_together():
    """The descent ranks ids by logit, aiter by `Sampler`'s fp32 p, whose
    ``logits / T - max`` rounds the distinct logits 2e-8 and 1e-8 onto one
    value: aiter's top-k 3 keeps both as a tie, the floor only the higher.
    Pinned so the gap stays the one the notes allow: every id the two
    disagree on has the p of the lowest id the floor keeps."""
    logits = torch.full((1, 4096), -20.0, device=DEV)
    logits[0, :4] = torch.tensor([1.0, 3e-8, 2e-8, 1e-8], device=DEV)
    logits = logits.to(torch.bfloat16)
    temps = torch.ones(1, device=DEV)
    ks = torch.tensor([3], dtype=torch.int32, device=DEV)
    kept, _ = filtered_p(logits, temps, ks)
    probs = (logits / temps[:, None]).softmax(dim=-1, dtype=torch.float32)
    aiter_kept = torch.ops.aiter.top_k_renorm_probs(probs, ks, 0) > 0
    assert kept[0].nonzero().view(-1).tolist() == [0, 1, 2]
    assert aiter_kept[0].nonzero().view(-1).tolist() == [0, 1, 2, 3]
    assert not (kept & ~aiter_kept).any()
    assert (probs[aiter_kept & ~kept] == probs[kept].min()).all()


def assert_draws_are(draws, kept, p_f):
    """Draws from one row follow p_f, both ways: none lands outside the kept
    set, every kept id worth 20 draws turns up (a set missing a tie group
    would not), and each at its rate."""
    n = draws.numel()
    assert kept[draws].all(), "Sampler drew an id the verify filters out"
    freq = torch.bincount(draws, minlength=kept.numel()).float() / n
    assert (freq[p_f * n > 20] > 0).all(), "a kept id Sampler never draws"
    torch.testing.assert_close(freq, p_f, rtol=0, atol=0.01)


@pytest.mark.parametrize(
    "top_k, top_p", [(30, None), (None, 0.7), (100, 0.6), (1000, 0.95)]
)
def test_samplers_draws_stay_in_the_kept_set_and_follow_p_f(top_k, top_p):
    """Against `Sampler` itself (its aiter kernels, not a re-derivation), on
    tied and untied rows: Sampler's draws follow p_f. aiter has no top-p mask
    to compare sets with, so this is the top-p check against aiter itself."""
    torch.manual_seed(127)
    rows, n, vocab = 4, 40000, 4000
    logits = tied_rows(rows, vocab, 131).repeat_interleave(n, 0)
    temps = torch.full((rows * n,), 0.8, device=DEV)
    ks = None if top_k is None else torch.full((rows * n,), top_k, device=DEV)
    ps = None if top_p is None else torch.full((rows * n,), top_p, device=DEV)
    ks = None if ks is None else ks.to(torch.int32)
    draws = Sampler()(logits, temps, ks, ps).long().view(rows, n)
    kept, p_f = filtered_p(
        logits[::n],
        temps[:rows],
        None if ks is None else ks[:rows],
        None if ps is None else ps[:rows],
    )
    for row in range(rows):
        assert_draws_are(draws[row], kept[row], p_f[row])


def test_the_bonus_draw_keeps_a_top_k_tie_group_whole_at_the_real_vocab():
    """For a scalar top-k (small k, large vocab) with a top-p, aiter's joint
    op takes a top-k-first path that keeps exactly k ids, splitting the tie
    group at the cut. The ratio verify's p_f keeps the group whole, and so
    does every draw `Sampler` makes with the flag set (``keep_top_k_ties``):
    from the scalar a direct caller passes, and from the per-row k the
    runner hands it (`prepare_sample`)."""
    torch.manual_seed(163)
    n, vocab = 10000, 163840
    row = ((torch.randn(vocab, device=DEV) * 12).round() / 4).to(torch.bfloat16)
    logits = row.expand(n, -1).contiguous()
    temps = torch.full((n,), 0.8, device=DEV)
    kept, p_f = filtered_p(logits[:1], temps[:1], 50, 0.95)
    assert kept[0].sum() > 50, "the cut is not inside a tie group"
    sampler = Sampler()
    sampler.keep_top_k_ties = True
    per_row = torch.full((n,), 50, dtype=torch.int32, device=DEV)
    for top_k in (50, per_row):
        draws = sampler(logits, temps, top_k, 0.95).long()
        assert_draws_are(draws, kept[0], p_f[0])


def first_token_law(target, temps, top_ks, top_ps, width=1024, step=41):
    """Draft with each row's own filters (the draft spells "off" 0 and 1.0),
    verify against the target's filtered p; ``(first token, accepted)``."""
    n = temps.numel()
    vocab = target.shape[-1]
    _, prev, w1, w2 = tables(1, vocab, 32, seed=137)
    base = (target.float() + torch.randn(vocab, device=DEV)).to(torch.bfloat16)
    greedy = (temps <= SAMPLER_EPS).to(torch.int32)
    store = empty_store(n, width)
    drafts, _ = sample(
        base.expand(n, -1).contiguous(),
        prev.expand(n).contiguous(),
        w1,
        w2,
        store,
        torch.arange(n, dtype=torch.int32, device=DEV),
        0,
        temps,
        greedy,
        torch.tensor([step], dtype=torch.int32, device=DEV),
        top_k=top_ks.clamp(min=0),
        top_p=top_ps,
    )
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts.to(torch.int32), 1, target.expand(n, -1), temps),
        draft_probs=RatioDraftQ(store, stale(n, vocab)),
        ratio_step=step,
        top_ks=top_ks,
        top_ps=top_ps,
    )
    return out[:, 0].long(), accepted


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "top_k, top_p", [(20, 1.0), (-1, 0.8), (50, 0.7)], ids=["top_k", "top_p", "both"]
)
def test_the_first_token_follows_samplers_filtered_p(top_k, top_p, dtype):
    """End to end, draft and verify each filtering per row: the first token
    follows p_f, never lands on a filtered-out id, and some drafts pass. fp32
    logits take the longer, 32-bit descent."""
    torch.manual_seed(139)
    n, vocab = 40000, 6000
    target = (torch.randn(vocab, device=DEV) * 3).to(dtype)
    temps = torch.full((n,), 0.7, device=DEV)
    ks = torch.full((n,), top_k, dtype=torch.int32, device=DEV)
    ps = torch.full((n,), top_p, device=DEV)
    first, accepted = first_token_law(target, temps, ks, ps)
    kept, p_f = sampler_kept(target[None], temps[:1], ks[:1], ps[:1])
    assert kept[0, first].all(), "emitted an id the filter removes"
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)
    assert 0 < accepted.float().mean().item() < 1


def test_a_mixed_batch_serves_each_request_its_own_p():
    """Greedy, unfiltered, top-k, top-p and both, interleaved in one batch of
    per-request filters: greedy requests get the argmax, the rest each their
    own p_f."""
    torch.manual_seed(149)
    kinds = [(SAMPLER_EPS, -1, 1.0), (0.9, -1, 1.0), (0.9, 15, 1.0)]
    kinds += [(0.9, -1, 0.75), (0.9, 40, 0.6)]
    n, vocab = 60000, 3000
    target = (torch.randn(vocab, device=DEV) * 3).to(torch.bfloat16)
    temps, ks, ps = (
        torch.tensor(column, device=DEV).repeat(n // len(kinds))
        for column in zip(*kinds)
    )
    first, _ = first_token_law(target, temps, ks.to(torch.int32), ps)
    for i, (temp, k, p) in enumerate(kinds):
        got = first[i :: len(kinds)]
        if temp <= SAMPLER_EPS:
            assert (got == target.float().argmax()).all()
            continue
        kept, p_f = sampler_kept(
            target[None],
            temps[i : i + 1],
            ks[i : i + 1].to(torch.int32),
            ps[i : i + 1],
        )
        assert kept[0, got].all(), (temp, k, p)
        freq = torch.bincount(got, minlength=vocab).float() / got.numel()
        torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.015)


def test_each_position_of_a_filtered_block_is_served_its_own_p_f():
    """Three draft positions per request, each scoring its own logits, under
    interleaved per-request filters (one kind unfiltered): every position a
    request reaches emits an id its own row keeps, at that row's p_f."""
    torch.manual_seed(179)
    kinds = [(-1, 1.0), (15, 1.0), (-1, 0.75), (40, 0.6)]
    n, width, vocab, support = 40000, 3, 3000, 64
    target = (torch.randn(width, vocab, device=DEV) * 3).to(torch.bfloat16)
    temps = torch.full((n,), 0.9, device=DEV)
    ks, ps = (
        torch.tensor(column, device=DEV).repeat(n // len(kinds))
        for column in zip(*kinds)
    )
    ks = ks.to(torch.int32)
    # Position k's q: the top of a noisy copy of its target, renormalized.
    noise = torch.randn(width, vocab, device=DEV) * 0.3
    noisy = torch.softmax((target.float() + noise) / 0.9, -1)
    q_probs, q_ids = noisy.topk(support, -1)
    q_probs = (q_probs / q_probs.sum(-1, keepdim=True)).repeat(n, 1)
    q_ids = q_ids.repeat(n, 1).to(torch.int32)
    slots = torch.multinomial(q_probs, 1)
    drafts = q_ids.gather(1, slots).view(-1)
    meta = torch.cat((torch.full_like(slots, support), slots), 1)
    store = SparseQStore(q_ids, q_probs.contiguous(), meta.to(torch.int32))
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, width, target.repeat(n, 1), temps),
        draft_probs=RatioDraftQ(store, stale(n, vocab)),
        ratio_step=53,
        top_ks=ks,
        top_ps=ps,
    )
    for i, kind in enumerate(kinds):
        kept, p_f = sampler_kept(target, temps[:width], ks[i].expand(width), ps[i])
        for k in range(width):
            reached = accepted[i :: len(kinds)] >= k
            got = out[i :: len(kinds), k][reached].long()
            assert got.numel() > 2000, (kind, k)
            assert kept[k, got].all(), (kind, k)
            freq = torch.bincount(got, minlength=vocab).float() / got.numel()
            torch.testing.assert_close(freq, p_f[k], rtol=0, atol=0.02)


def test_a_filtered_out_draft_is_always_rejected():
    """q all on an id top-k removes: p_f there is 0, so no coin accepts it,
    and the replacement follows p_f (relu(p_f - q) is p_f)."""
    torch.manual_seed(151)
    n, vocab = 30000, 3000
    target = (torch.randn(vocab, device=DEV) * 2).to(torch.bfloat16)
    temps = torch.ones(n, device=DEV)
    ks = torch.full((n,), 10, dtype=torch.int32, device=DEV)
    kept, p_f = sampler_kept(target[None], temps[:1], ks[:1])
    outside = int((~kept[0]).nonzero()[0])
    store = empty_store(1, 4)
    store.ids[0, 0], store.probs[0, 0] = outside, 1.0
    store.meta[0] = torch.tensor([1, 0], device=DEV)
    drafts = torch.full((n,), outside, dtype=torch.int32, device=DEV)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(
            drafts,
            1,
            target.expand(n, -1),
            temps,
            row_base=torch.zeros(n, dtype=torch.int32, device=DEV),
        ),
        draft_probs=RatioDraftQ(store, stale(n, vocab)),
        ratio_step=43,
        top_ks=ks,
    )
    assert (accepted == 0).all()
    first = out[:, 0].long()
    assert kept[0, first].all()
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)


def test_a_filtered_residual_without_mass_falls_back_to_p_f():
    """q a hair above p_f on every id top-k keeps (so no rounding of the
    kernel's p_f leaves a residual; q need not sum to 1 for that), and the
    draft on an id it removes: rejected, relu(p_f - q) is 0 everywhere, and
    the fallback must still follow p_f, never landing where the filter cuts."""
    torch.manual_seed(181)
    n, vocab = 30000, 3000
    target = (torch.randn(vocab, device=DEV) * 2).to(torch.bfloat16)
    temps = torch.ones(n, device=DEV)
    ks = torch.full((n,), 10, dtype=torch.int32, device=DEV)
    kept, p_f = sampler_kept(target[None], temps[:1], ks[:1])
    outside = int((~kept[0]).nonzero()[0])
    q = torch.where(kept[0], p_f[0] * 1.001, 0.0)
    q[outside] = 0.01
    drafts = torch.full((n,), outside, dtype=torch.int32, device=DEV)
    out, (accepted, _, _) = rejection_sample(
        **verify_batch(drafts, 1, target.expand(n, -1), temps),
        draft_probs=draft_q(q, drafts),
        ratio_step=59,
        top_ks=ks,
    )
    assert (accepted == 0).all()
    first = out[:, 0].long()
    assert kept[0, first].all(), "the fallback emitted an id the filter removes"
    freq = torch.bincount(first, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_masked_and_nan_target_rows_filter_and_emit_inside_the_vocab(dtype):
    """Filtered target rows with masked (-inf) ids, and an all-NaN one: the
    masked row keeps `Sampler`'s set and its first token follows p_f, never
    landing on an id p_f gives no mass; the NaN row emits an id inside the
    vocab."""
    torch.manual_seed(167)
    n, vocab = 30000, 3000
    target = torch.randn(2, vocab, device=DEV) * 3
    target[0, torch.rand(vocab, device=DEV) < 0.3] = float("-inf")
    target[1] = float("nan")
    target = target.to(dtype)
    temps = torch.full((2,), 0.8, device=DEV)
    ks = torch.tensor([20, 20], dtype=torch.int32, device=DEV)
    ps = torch.tensor([0.8, 0.8], device=DEV)
    kept, p_f = filtered_p(target, temps, ks, ps)
    ref, ref_p = sampler_kept(target[:1], temps[:1], ks[:1], ps[:1])
    assert torch.equal(kept[0], ref[0])
    torch.testing.assert_close(p_f[0], ref_p[0], rtol=1e-4, atol=1e-7)
    q = sparse(vocab, {i: 1 / 50 for i in range(50)})
    drafts = draw(q, 2 * n)
    out, _ = rejection_sample(
        **verify_batch(drafts, 1, target.repeat(n, 1), temps.repeat(n)),
        draft_probs=draft_q(q, drafts),
        ratio_step=47,
        top_ks=ks.repeat(n),
        top_ps=ps.repeat(n),
    )
    first = out[:, 0].long()
    assert ((first >= 0) & (first < vocab)).all()
    masked = first[0::2]
    assert (p_f[0, masked] > 0).all(), "emitted an id p_f gives no mass"
    freq = torch.bincount(masked, minlength=vocab).float() / n
    torch.testing.assert_close(freq, p_f[0], rtol=0, atol=0.01)


def test_filters_that_keep_everything_leave_the_verify_bit_for_bit():
    """top_k >= V or -1 and top_p 1.0, as one value or per request: the same
    tokens and verdicts as no filter at all, coin for coin."""
    torch.manual_seed(157)
    batch_n, width, vocab = 64, 4, 2 * _RATIO_BLOCK_V + 77
    rows = batch_n * width
    q_logits = torch.randn(rows, vocab, device=DEV) * 2
    q_rows = top_of(q_logits, 30)
    drafts = torch.multinomial(q_rows, 1).view(-1).to(torch.int32)
    logits = (q_logits + torch.randn(rows, vocab, device=DEV)).to(torch.bfloat16)
    temps = torch.tensor([1.0, 0.6, SAMPLER_EPS, 1.3], device=DEV).repeat(batch_n // 4)
    args = verify_batch(drafts, width, logits, temps)
    q = draft_q(q_rows, drafts, dense_rows=batch_n)
    ref = rejection_sample(**args, draft_probs=q, ratio_step=3)
    every_k = torch.tensor([-1, vocab, vocab + 5, -1], device=DEV, dtype=torch.int32)
    for ks, ps in (
        (every_k.repeat(batch_n // 4), torch.ones(batch_n, device=DEV)),
        (vocab, None),
        (None, 1.0),
    ):
        got = rejection_sample(
            **args, draft_probs=q, ratio_step=3, top_ks=ks, top_ps=ps
        )
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1]), (ks, ps)

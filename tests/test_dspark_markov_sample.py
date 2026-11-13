# SPDX-License-Identifier: MIT
"""The sampled DSpark Markov op (`dspark_markov_sample`): each row stores the
top of ``softmax((base + bias) / T)`` -- ids, renormalized probabilities, and
the slot of the id it drew -- at its own row block, nowhere else, and draws
``x`` from exactly that q; and its input contract.

Statistical checks use fixed seeds and tolerances several standard errors
wide, so they are deterministic on a given platform and fail only on a real
bias. The properties that hold for every coin are asserted exactly.
"""

import pytest
import torch

# The CPU unit job has neither aiter nor triton; both imports stay below this
# guard. The kernel tests need a GPU and say so one by one.
pytest.importorskip("aiter", reason="the samplers are AITER/Triton ops")

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)

import triton
from spec_lossless_reference import (
    DEV,
    assert_store_row_is_q,
    draft_args,
    dspark_markov_sample_reference,
    empty_store,
    filters,
    greedy_at_eps,
    sample,
    stored,
    tables,
)

from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax
from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.dspark_markov_sample import (
    dspark_markov_sample,
    dspark_markov_sample_scratch,
)
from atom.model_ops.spec_sampling.softmax_tiles import softmax_combine_kernel
from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore
from tests.test_dspark_markov_argmax import _ids, _tables


def test_the_sampled_op_checks_its_tables_as_the_greedy_one_does():
    """Both ops index W2 by the logits' column: a table of another vocab must
    raise in the sampled op too, not read the wrong rows."""
    rows, vocab = 5, 64
    base, prev, w1, w2 = _tables(rows, vocab, 16, "cpu")
    store = SparseQStore.allocate(rows, 8, "cpu")
    scratch = dspark_markov_sample_scratch(rows, vocab, "cpu")
    i32 = {"dtype": torch.int32}
    rest = (
        *store,
        torch.zeros(rows, **i32),
        0,
        torch.ones(rows),
        torch.zeros(rows, **i32),
        torch.ones(rows),
        torch.zeros(1, **i32),
        0,
        scratch,
    )
    with pytest.raises(ValueError, match="markov_w2"):
        dspark_markov_sample(base, prev, w1, w2[:-1], _ids(rows, "cpu"), *rest)
    with pytest.raises(ValueError, match="markov_w1"):
        dspark_markov_sample(base, prev, w1[:-1], w2, _ids(rows, "cpu"), *rest)


@requires_gpu
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


@requires_gpu
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


@requires_gpu
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


@requires_gpu
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


@requires_gpu
def test_an_oversized_scratch_drafts_exactly_what_an_exact_one_does():
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


@requires_gpu
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


@requires_gpu
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


@requires_gpu
@pytest.mark.parametrize("width", [7, 1024, 20000])
def test_every_stored_row_is_strictly_ascending(width):
    """The `SparseQStore` contract the residual race's binary search relies
    on, over every row the writer touches: tied levels, a cut inside a level,
    and filters, at widths below, at and past the reach."""
    rows, vocab, rank = 32, 20000, 32
    torch.manual_seed(5)
    base = torch.randint(0, 9, (rows, vocab), device=DEV).float().to(torch.bfloat16)
    base[::2] = torch.randn(rows // 2, vocab, device=DEV).to(torch.bfloat16)
    prev = torch.zeros(rows, dtype=torch.int64, device=DEV)
    w = torch.zeros(vocab, rank, dtype=torch.bfloat16, device=DEV)
    temps, greedy, seed = draft_args(rows, temperature=0.8, step=4)
    top_k = torch.tensor([0, 5, 300, 0] * (rows // 4), dtype=torch.int32, device=DEV)
    top_p = torch.tensor([1.0, 1.0, 0.9, 0.6] * (rows // 4), device=DEV)
    store = empty_store(rows, width)
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV)
    sample(base, prev, w, w, store, row_base, 0, temps, greedy, seed, top_k, top_p)
    for r in range(rows):
        ids, _, _ = stored(store, r)
        assert ids.numel() >= 1, r
        assert (ids[1:] > ids[:-1]).all(), f"row {r} is not strictly ascending"


@requires_gpu
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


@requires_gpu
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


@requires_gpu
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
    store = empty_store(rows + 1, width)  # position 1

    def call():
        dspark_markov_sample(
            base,
            prev,
            w1,
            w2,
            out,
            *store,
            row_base,
            1,
            greedy_at_eps(temps, greedy),
            top_k,
            top_p,
            seed,
            rows * vocab,
            scratch,
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


@requires_gpu
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


@requires_gpu
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


@requires_gpu
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


@requires_gpu
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


@requires_gpu
def test_the_block_sampler_chains_the_op_through_its_bound_buffers():
    """K3's sampled block, Kimi-K3's `sample_block`: position k
    drafts from x_{k-1} into row base + k, with the per-row inputs sliced to
    the running batch and the noise counter strided by the buffers' max batch,
    exactly as calling the op directly."""
    from types import SimpleNamespace

    from atom.models.dspark_draft import DraftSampling, DSparkMarkovHead
    from atom.models.kimi_k3_dspark import KimiK3DSpark

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
        q_row_base=row_base,
        seed=seed,
        scratch=dspark_markov_sample_scratch(max_bs, vocab, DEV),
        temperatures=greedy_at_eps(temps, greedy),
        top_k=top_k,
        top_p=top_p,
    )
    model = SimpleNamespace(markov_head=head, draft_sampling=sampling)
    drafts = KimiK3DSpark._sample_block(model, base, anchors, False)

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


@requires_gpu
def test_the_greedy_block_ignores_the_bound_sampling():
    """K3's ``greedy`` block, the second pass an armed drafter declares:
    the argmax chain whatever ``draft_sampling`` holds, and no q written."""
    from types import SimpleNamespace

    from atom.models.dspark_draft import DraftSampling, DSparkMarkovHead
    from atom.models.kimi_k3_dspark import KimiK3DSpark

    max_bs, bs, steps, vocab, rank, width = 4, 3, 3, 3000, 32, 64
    head = DSparkMarkovHead(vocab, rank).to(DEV, torch.bfloat16)
    torch.manual_seed(23)
    base = torch.randn(bs, steps, vocab, device=DEV).to(torch.bfloat16)
    anchors = torch.randint(0, vocab, (bs,), dtype=torch.int32, device=DEV)
    temps, _, seed = draft_args(max_bs, temperature=0.9, step=4)
    top_k, top_p = filters(max_bs)
    sampling = DraftSampling(
        q_store=empty_store(3 * steps, width),
        q_row_base=torch.tensor([6, 3, 0, -1], dtype=torch.int32, device=DEV),
        seed=seed,
        scratch=dspark_markov_sample_scratch(max_bs, vocab, DEV),
        temperatures=temps,
        top_k=top_k,
        top_p=top_p,
    )
    before = [t.clone() for t in sampling.q_store]
    model = SimpleNamespace(markov_head=head, draft_sampling=sampling)
    drafts = KimiK3DSpark._sample_block(model, base, anchors, True)

    assert torch.equal(drafts, head.sample_block(base, anchors, None))
    for a, b in zip(sampling.q_store, before):
        assert torch.equal(a, b)

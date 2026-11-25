# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`dspark_markov_sample_tp` against the whole-vocab op: every rank of a
vocab-sharded LM head must store and draw exactly what `dspark_markov_sample`
stores and draws over the gathered logits, bit for bit.

The ranks run as threads on one GPU over a group whose all-gather is a
barrier, so the cross-rank data paths are the real ones without a process
group.
"""

import threading

import pytest
import torch

# The CPU unit job has neither aiter nor triton; both imports stay below these
# guards, or the job fails to collect this file and aborts the whole run.
pytest.importorskip("aiter", reason="the samplers are AITER/Triton ops")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from atom.model_ops.sampler import SAMPLER_EPS
from atom.model_ops.spec_sampling.dspark_markov_sample import (
    dspark_markov_sample,
    dspark_markov_sample_scratch,
    dspark_markov_sample_tp,
)
from atom.model_ops.spec_sampling.sparse_q_store import SparseQStore

DEV = "cuda"


class ThreadGroup:
    """``all_gather(dim=0)`` across threads, one thread per rank. One rank
    runs at a time (each holds the group's lock but while it waits in a
    collective), so kernel launches never interleave across ranks."""

    def __init__(self, world_size):
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size)
        self._parts = [None] * world_size
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def rank_in_group(self):
        return self._local.rank

    def run(self, fn):
        errors = []

        def body(rank):
            self._local.rank = rank
            self._lock.acquire()
            try:
                fn(rank)
            except BaseException as e:  # noqa: BLE001 -- re-raised below
                errors.append(e)
                self._barrier.abort()
            finally:
                self._lock.release()

        threads = [
            threading.Thread(target=body, args=(r,)) for r in range(self.world_size)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]

    def all_gather(self, t, use_custom=False, dim=0):
        assert dim == 0
        torch.cuda.synchronize()
        self._parts[self.rank_in_group] = t.clone()
        self._wait()
        out = torch.cat(self._parts, dim=0)
        self._wait()
        return out

    def _wait(self):
        self._lock.release()
        try:
            self._barrier.wait()
        finally:
            self._lock.acquire()


def _inputs(rows, vocab, rank, seed, dtype):
    torch.manual_seed(seed)
    kw = {"dtype": dtype, "device": DEV}
    base = torch.randn(rows, vocab, **kw) * 3
    # A sharp peak in the last shard, and a tie across two shards.
    base[1, vocab - 7] = 40.0
    base[2, 11] = base[2, vocab // 2 + 11] = 30.0
    prev = torch.randint(0, vocab, (rows,), dtype=torch.int64, device=DEV)
    w1 = torch.randn(vocab, rank, **kw) * 0.1
    w2 = torch.randn(vocab, rank, **kw) * 0.1
    temps = torch.linspace(0.3, 1.5, rows, device=DEV)
    temps[3] = SAMPLER_EPS  # greedy: drafts the argmax, stores its delta
    top_k = torch.full((rows,), -1, dtype=torch.int32, device=DEV)
    top_k[4] = 20
    top_p = torch.ones(rows, device=DEV)
    top_p[5] = 0.7
    row_base = torch.arange(rows, dtype=torch.int32, device=DEV) * 2
    row_base[6] = -1  # no destination: drafts the argmax, stores nothing
    seed_t = torch.tensor([17], dtype=torch.int32, device=DEV)
    return base, prev, w1, w2, temps, top_k, top_p, row_base, seed_t


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=str)
@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("vocab", [4 * 5000, 4 * 8192, 129280])
def test_every_rank_stores_and_draws_the_whole_vocab_ops_bits(world, vocab, dtype):
    """Shards that end mid-tile (5000, 32320 ids) and on one; a greedy row,
    a row without a destination, top-k and top-p rows among them."""
    rows, rank, width, position = 8, 64, 256, 1
    noise_offset = position * rows * vocab
    base, prev, w1, w2, temps, top_k, top_p, row_base, seed = _inputs(
        rows, vocab, rank, world + vocab, dtype
    )
    store_rows = 2 * rows + 2

    def fresh_store():
        store = SparseQStore.allocate(store_rows, width, DEV)
        for t in store:
            t.fill_(-7)
        return store

    ref_store = fresh_store()
    ref_out = torch.empty(rows, dtype=torch.int64, device=DEV)
    dspark_markov_sample(
        base,
        prev,
        w1,
        w2,
        ref_out,
        *ref_store,
        row_base,
        position,
        temps,
        top_k,
        top_p,
        seed,
        noise_offset,
        dspark_markov_sample_scratch(rows, vocab, DEV),
    )

    shard = vocab // world
    group = ThreadGroup(world)
    stores = [fresh_store() for _ in range(world)]
    outs = [torch.empty(rows, dtype=torch.int64, device=DEV) for _ in range(world)]
    scratches = [
        dspark_markov_sample_scratch(rows, vocab, DEV, tp_size=world)
        for _ in range(world)
    ]

    def rank_fn(r):
        dspark_markov_sample_tp(
            base[:, r * shard : (r + 1) * shard].contiguous(),
            prev,
            w1,
            w2,
            outs[r],
            *stores[r],
            row_base,
            position,
            temps,
            top_k,
            top_p,
            seed,
            noise_offset,
            scratches[r],
            r * shard,
            group,
        )

    group.run(rank_fn)
    torch.cuda.synchronize()

    written = row_base[row_base >= 0].long() + position
    for r in range(world):
        assert torch.equal(outs[r], ref_out), r
        assert torch.equal(stores[r].meta[written], ref_store.meta[written]), r
        for row in written.tolist():
            n = ref_store.meta[row, 0].item()
            assert torch.equal(stores[r].ids[row, :n], ref_store.ids[row, :n])
            assert torch.equal(stores[r].probs[row, :n], ref_store.probs[row, :n])
        # Rows nobody drafts into stay untouched.
        untouched = torch.ones(store_rows, dtype=torch.bool, device=DEV)
        untouched[written] = False
        assert (stores[r].meta[untouched] == -7).all()
    # The filters still bite: the top-k row stores at most 20, the greedy row
    # its delta.
    assert ref_store.meta[row_base[4].long() + position, 0].item() <= 20
    assert ref_store.meta[row_base[3].long() + position, 0].item() == 1
    sampled = ref_store.meta[written, 0] > 1
    assert sampled.sum().item() >= 4


def test_the_scratch_holds_either_op_at_its_tp_degree():
    """`dspark_markov_sample_scratch` with a TP degree holds the sharded op's
    regions even where they outgrow the whole-vocab op's (a small vocab, where
    the `[2, B, K]` staging pair is no longer small beside the scores)."""
    from atom.model_ops.spec_sampling.dspark_markov_sample import (
        _sample_scratch,
        _sample_tp_scratch,
    )
    from atom.model_ops.spec_sampling.sampling_scratch import carve
    from atom.model_ops.spec_sampling.sparse_q_store import store_width

    rows, vocab, world = 4, 2048, 2
    buf = dspark_markov_sample_scratch(rows, vocab, DEV, tp_size=world)
    carve(buf, _sample_scratch(rows, vocab))
    carve(buf, _sample_tp_scratch(rows, vocab, world, store_width(vocab)))

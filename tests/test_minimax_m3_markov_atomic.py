# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 draft's atomic sharded Markov block
(`dspark._markov_step_atomic`, `_markov_last_atomic`) against the whole-vocab
`dspark_markov_argmax` position by position, the TP4 exchange emulated on one
GPU: the same ids, with ties across ranks and the packed buffers left at their
reset value."""

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

W, VS, R, T = 4, 50016, 256, 7


pack_max_id = pytest.importorskip("atom.models.minimax_m3.packed_argmax").pack_max_id


@triton.jit
def _pack_kernel(val_ptr, idx_ptr, out_ptr):
    i = tl.program_id(0)
    tl.store(out_ptr + i, pack_max_id(tl.load(val_ptr + i), tl.load(idx_ptr + i)))


def _tables(dev):
    w1 = (torch.randn(W * VS, R, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(W * VS, R, device=dev) * 0.05).to(torch.bfloat16)
    return w1, w2


def _whole_vocab_ids(base, anchors, w1, w2):
    from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax

    ref = torch.empty(base.shape[0], T + 1, dtype=torch.int32, device=base.device)
    ref[:, 0] = anchors
    for k in range(T):
        dspark_markov_argmax(base[:, k], ref[:, k], w1, w2, ref[:, k + 1])
    return ref[:, 1:]


def _atomic_ids(base, anchors, w1, w2, packed):
    """One block, each rank's launches run in turn; ``packed`` is
    ``[W, T, max_bs]`` and is the buffers' state across blocks."""
    import triton

    from atom.model_ops.dspark_markov_stage1 import (
        BLOCK_K,
        BLOCK_V,
        markov_launch_geometry,
    )
    from atom.models.minimax_m3.dspark import (
        _markov_last_atomic,
        _markov_step_atomic,
    )

    bs = base.shape[0]
    out = torch.empty(bs, T, dtype=torch.int32, device=base.device)
    geometry = markov_launch_geometry(bs, VS)
    gathered = None
    for k in range(T):
        words = []
        for r in range(W):
            shard = base[:, k, r * VS : (r + 1) * VS]
            mine = packed[r, k, :bs]
            _markov_step_atomic[(geometry.v_tiles, geometry.row_tiles)](
                shard, anchors, mine if gathered is None else gathered,
                packed[r, max(k - 1, 0), :bs], out[:, max(k - 1, 0)], w1, w2, mine,
                bs, VS, R, shard.stride(0), anchors.stride(0), out.stride(0),
                w1.stride(0), w2.stride(0), r * VS, FIRST=k == 0, W=W,
                BLOCK_ROW=geometry.block_row, BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K,
                num_warps=geometry.num_warps, num_stages=2,
            )  # fmt: skip
            words.append(mine.clone())
        gathered = torch.cat(words)
    block_row = min(1024, triton.next_power_of_2(bs))
    for r in range(W):
        _markov_last_atomic[(triton.cdiv(bs, block_row),)](
            gathered, packed[r, T - 1, :bs], out[:, T - 1], bs, out.stride(0),
            W=W, BLOCK_ROW=block_row, num_warps=1,
        )  # fmt: skip
    return out


@pytest.mark.parametrize("bs", [1, 3, 16, 64])
def test_atomic_matches_whole_vocab_over_two_blocks(bs):
    from atom.models.minimax_m3.packed_argmax import EMPTY

    torch.manual_seed(bs)
    dev = torch.device("cuda")
    w1, w2 = _tables(dev)
    packed = torch.full((W, T, 64), EMPTY, dtype=torch.int64, device=dev)
    # Two blocks in a row on the same buffers: the second reads whatever the
    # first left behind.
    for _ in range(2):
        base = (torch.randn(bs, T, W * VS, device=dev) * 2).to(torch.bfloat16)
        anchors = torch.randint(0, W * VS, (bs,), device=dev, dtype=torch.int32)
        out = _atomic_ids(base, anchors, w1, w2, packed)
        torch.testing.assert_close(
            out, _whole_vocab_ids(base, anchors, w1, w2), atol=0, rtol=0
        )
        assert bool((packed == EMPTY).all())


def test_ties_across_ranks_pick_the_lowest_id():
    from atom.models.minimax_m3.packed_argmax import EMPTY

    dev = torch.device("cuda")
    bs = 5
    w1, w2 = _tables(dev)
    w2.zero_()  # no bias: the base logits alone decide
    base = torch.zeros(bs, T, W * VS, device=dev, dtype=torch.bfloat16)
    # The same maximum in every rank's shard (and twice in one tile of rank 2).
    for r in range(W):
        base[:, :, r * VS + 7 + r] = 3.0
    base[:, :, 2 * VS + 8] = 3.0
    anchors = torch.zeros(bs, device=dev, dtype=torch.int32)
    packed = torch.full((W, T, 64), EMPTY, dtype=torch.int64, device=dev)
    out = _atomic_ids(base, anchors, w1, w2, packed)
    assert bool((out == 7).all())
    torch.testing.assert_close(
        out, _whole_vocab_ids(base, anchors, w1, w2), atol=0, rtol=0
    )


def test_pack_max_id_signed_zero_is_one_value():
    dev = torch.device("cuda")
    vals = torch.tensor([-0.0, 0.0, -0.0, 0.0], device=dev)
    idx = torch.tensor([5, 5, 7, 7], device=dev, dtype=torch.int32)
    out = torch.empty(4, dtype=torch.int64, device=dev)
    _pack_kernel[(4,)](vals, idx, out)
    assert out[0] == out[1] and out[2] == out[3]
    assert out[0] > out[3]  # equal values: the lower id packs larger


def test_signed_zero_ties_pick_the_lowest_id():
    """-0.0 and +0.0 are equal: the lowest id wins across ranks, whichever sign."""
    from atom.models.minimax_m3.packed_argmax import EMPTY

    dev = torch.device("cuda")
    bs = 4
    w1, w2 = _tables(dev)
    w2.zero_()
    # The row maximum is an exact zero; rank 0 holds -0.0 at id 5, rank 1 +0.0
    # at its id 100000 (and the other way round for the last two rows).
    base = torch.full((bs, T, W * VS), -1.0, device=dev, dtype=torch.bfloat16)
    neg_zero = -torch.zeros((), device=dev, dtype=torch.bfloat16)
    base[:2, :, 5] = neg_zero
    base[:2, :, VS + 50] = 0.0
    base[2:, :, 5] = 0.0
    base[2:, :, VS + 50] = neg_zero
    anchors = torch.zeros(bs, device=dev, dtype=torch.int32)
    packed = torch.full((W, T, 64), EMPTY, dtype=torch.int64, device=dev)
    out = _atomic_ids(base, anchors, w1, w2, packed)
    assert bool((out == 5).all())
    torch.testing.assert_close(
        out, _whole_vocab_ids(base, anchors, w1, w2), atol=0, rtol=0
    )


def test_sharded_head_dispatch_layout(monkeypatch):
    """`_head_and_sample_sharded` end to end on a one-rank group whose
    all_gather is a copy: the [bs, 2] float32 view of the packed words
    and the int64 view back must round-trip to the whole-vocab ids."""
    from types import SimpleNamespace

    from atom.models.minimax_m3 import dspark as mod
    from atom.models.minimax_m3.packed_argmax import EMPTY

    dev = torch.device("cuda")
    bs, vs = 7, 4096
    w1 = (torch.randn(vs, R, device=dev) * 0.05).to(torch.bfloat16)
    w2 = (torch.randn(vs, R, device=dev) * 0.05).to(torch.bfloat16)
    base = (torch.randn(bs, T, vs, device=dev) * 2).to(torch.bfloat16)
    anchors = torch.randint(0, vs, (bs,), device=dev, dtype=torch.int32)

    group = SimpleNamespace(
        world_size=1, all_gather=lambda x, dim, use_custom: x.clone()
    )
    monkeypatch.setattr(mod, "get_tp_group", lambda: group)
    monkeypatch.setattr(mod.tgemm, "mm", lambda x, w: x)
    model = object.__new__(mod.MiniMaxM3DSpark)
    model.emit_confidence = False
    model._mk_packed = torch.full((T, 64), EMPTY, dtype=torch.int64, device=dev)
    model.lm_head = SimpleNamespace(weight=base.view(bs * T, vs), vocab_start_idx=0)
    model.markov_head = SimpleNamespace(
        markov_w1=SimpleNamespace(weight=w1), markov_w2=SimpleNamespace(weight=w2)
    )
    out, confidence = model._head_and_sample_sharded(base.view(bs * T, vs), anchors, T)
    assert confidence is None
    torch.testing.assert_close(
        out, _whole_vocab_ids(base, anchors, w1, w2), atol=0, rtol=0
    )
    assert bool((model._mk_packed == EMPTY).all())

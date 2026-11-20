# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_DRAFT_VOCAB on the CPU: the range spec, the column -> vocabulary maps
(the sampler's token, its records' strided column lanes, filler columns), the draft
presence mask (a view of the first range, or each step's rows gathered), and the
TP1 argmax path of the draft LM head over the subset. The sampler, the FP8 head
and graph capture are tests/test_draft_vocab_gpu.py's."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.draft_vocab import DraftVocab, parse_vocab_ranges

V, H = 1024, 8
# the default spec's shape: a large low range and a small high tail (specials)
SPEC = "0:960,1000:1012"  # 972 ids, a 12-id tail (<= 1/64)
WIDE = "0:256,512:768"  # a tail half the subset: gathered a step


def _vocab(spec=SPEC, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(V, H, generator=g).to(torch.bfloat16)
    return w, DraftVocab(w, parse_vocab_ranges(spec, V), fp8=False)


def test_ranges():
    assert parse_vocab_ranges("5:9, 0:3", 10) == [(0, 3), (5, 9)]
    assert parse_vocab_ranges("0:98278,248044:248070", 248320) == [
        (0, 98278),
        (248044, 248070),
    ]
    for bad in ("0:4,3:6", "0:11", "4:4", ""):
        with pytest.raises(AssertionError):
            parse_vocab_ranges(bad, 10)


def test_subset_rows_and_maps():
    w, dv = _vocab()
    assert dv.size == 972 and dv.ranges == [(0, 960, 0), (1000, 1012, 960)]
    assert torch.equal(dv.weight, w[dv.ids.long()])
    cols = torch.tensor([0, 959, 960, 971], dtype=torch.int32)
    out = torch.empty_like(cols)
    assert dv.to_vocab(cols, out=out).tolist() == [0, 959, 1000, 1011]
    dv.to_vocab(cols, out=cols)  # in place: one gather
    assert cols.tolist() == [0, 959, 1000, 1011]
    # the sampler's filler columns (size + lane) map to -1, never out of bounds
    fill = torch.tensor([972, 972 + 127], dtype=torch.int32)
    assert dv.to_vocab(fill, out=fill).tolist() == [-1, -1]


def test_record_columns_map_in_place():
    """The k column lanes of `[rows, 3 * KB + 2]` sampler records: only those."""
    _, dv = _vocab()
    k, dw = 5, 3 * 16 + 2
    rec = torch.randint(0, 972, (3, dw), dtype=torch.int32)
    before = rec.clone()
    dv.map_cols_(rec[:, :k])
    assert torch.equal(rec[:, :k], dv.ids[before[:, :k].long()])
    assert torch.equal(rec[:, k:], before[:, k:])


def test_presence_view_rule():
    _, dv = _vocab()
    mask = torch.zeros(4, V, dtype=torch.uint8)
    view = dv.presence_view(mask)
    assert view.shape == (4, 972) and view.data_ptr() == mask.data_ptr()
    assert view.stride() == mask.stride()
    _, wide = _vocab(WIDE)
    assert wide.presence_view(mask) is None  # the tail is no small share
    _, top = _vocab("100:1012")
    assert top.presence_view(mask).data_ptr() == mask[:, 100:].data_ptr()
    _, past = _vocab("200:900,1000:1024")  # 724 wide from 200 fits; tail 24 > 724/64
    assert past.presence_view(mask) is None


def test_gather_columns():
    _, dv = _vocab(WIDE)
    src = torch.randint(0, 2, (6, V), dtype=torch.uint8)
    rows = torch.tensor([4, 1, 5], dtype=torch.int32)
    out = torch.zeros(3, dv.size, dtype=torch.uint8)
    dv.gather_columns(src, rows, out)
    assert torch.equal(out, src[rows.long()][:, dv.ids.long()])


def _presence(spec):
    pp = pytest.importorskip("atom.model_ops.presence_penalty", exc_type=ImportError)
    _, dv = _vocab(spec)
    state = pp.PresencePenaltyState(V, max_num_seqs=4, max_extra=3, device="cpu")
    dp = pp.DraftPresencePenalty(state, max_rows=4, max_drafts=3, device="cpu")
    return state, dp, dv


def test_presence_mask_is_a_view_of_the_first_range():
    state, dp, dv = _presence(SPEC)
    assert dp.draft_mask is state.mask and dp.draft_slots is dp.slots
    dp.set_vocab(dv)
    assert dp.mask_vocab is None
    assert dp.draft_slots is dp.slots
    assert dp.draft_mask.data_ptr() == state.mask.data_ptr()
    assert dp.draft_mask.shape == (state.num_slots, dv.size)
    state.mask[2, [7, 965, 1001]] = 1
    dp.slots[0] = 2
    dp.stage_vocab(1)  # a no-op
    row = dp.draft_mask[dp.draft_slots[0].long()]
    # column 7 is id 7; column 965 is id 1005 in the subset but reads id 965's bit;
    # id 1001 (column 961) is not read
    assert row.nonzero().view(-1).tolist() == [7, 965]


def test_presence_rows_gathered_a_step():
    state, dp, dv = _presence(WIDE)
    dp.set_vocab(dv)
    assert dp.mask_vocab is not None and dp.draft_mask is dp.mask_vocab
    assert dp.draft_slots.tolist() == [0, 1, 2, 3]
    state.mask[3, [10, 300, 600]] = 1  # ids 10 and 600 are in the subset
    dp.slots[:2] = torch.tensor([3, 0], dtype=torch.int32)
    dp.stage_vocab(2)
    assert dp.mask_vocab[0].nonzero().view(-1).tolist() == [10, 256 + (600 - 512)]
    assert int(dp.mask_vocab[1].sum()) == 0


@pytest.fixture
def head(monkeypatch):
    embed_head = pytest.importorskip("atom.model_ops.embed_head", exc_type=ImportError)
    h = embed_head.ParallelLMHead.__new__(embed_head.ParallelLMHead)
    torch.nn.Module.__init__(h)
    h.tp_size = 1
    h.bias = None
    h._argmax_row_lens = {}
    seen = {}

    def topk_select(input, topk, *, end=None, tie=None, output_idx=None, **_):
        seen.setdefault("end", []).append(end)
        output_idx.copy_(input.float().argmax(-1, keepdim=True).to(torch.int32))

    monkeypatch.setattr(embed_head, "topk_select", topk_select)
    monkeypatch.setattr(
        embed_head,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(is_draft=True)),
    )
    return embed_head, h, seen


def test_draft_argmax_over_the_subset(head, monkeypatch):
    """The TP1 argmax over the subset's logits: a vocabulary id comes out, the
    presence state records the column, and the row lengths are the subset's."""
    embed_head, h, seen = head
    w, dv = _vocab()
    h.weight = torch.nn.Parameter(w, requires_grad=False)
    h.draft_vocab = dv
    x = torch.randn(3, H).to(torch.bfloat16)
    logits = (x.float() @ w.float().t())[:, dv.ids.long()]
    # the highest subset column of row 0 in the tail, of the others in the low range
    logits[0, 965] = 1e4
    monkeypatch.setattr(embed_head.tgemm, "mm", lambda a, b, bias=None: logits)
    recorded = []
    h.draft_presence = SimpleNamespace(
        apply=lambda lg: lg, record=lambda ids: recorded.append(ids.clone())
    )
    out = torch.empty(3, dtype=torch.int32)
    h.compute_argmax_token(x, out=out)
    cols = logits.argmax(-1)
    assert out.tolist() == dv.ids[cols].tolist() and out[0].item() == 1005
    assert recorded[0].tolist() == cols.tolist()
    assert seen["end"][0].tolist() == [dv.size] * 3
    assert list(h._argmax_row_lens) == [dv.size]

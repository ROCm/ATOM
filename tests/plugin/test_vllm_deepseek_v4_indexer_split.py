# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 indexer split on prefill-classified steps with no prefill rows.

A prompt of 128*m + 1 tokens re-sent after a full prefix hit, a fresh prompt of
8192*k + 1 tokens, or a 1-token prompt has a last prefill step of one token. The
bridge classifies that step as prefill, ``_populate_indexer`` peels every
1-token row onto the paged decode path, and nothing is left for the dense path.
Calling the dense path anyway gathered one committed row through an empty block
table on the GPU (memory access fault).
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the V4 model module reaches AITER ops")

from atom.plugin.vllm import deepseek_v4_bridge as bridge
from atom.plugin.vllm.models import deepseek_v4 as plugin_v4
from atom.utils.forward_context import AttnState

TOPK = 8


def _indexer(monkeypatch, meta, num_blocks_rows):
    idx = object.__new__(plugin_v4.IndexerVllm)
    calls = {"decode": [], "prefill": []}

    def decode(q, w, bt, indexer_meta, topk, **kw):
        calls["decode"].append((q.size(0), bt.size(0)))
        return torch.full((q.size(0), topk), 1, dtype=torch.int32)

    def prefill(q, w, bt, indexer_meta, topk):
        calls["prefill"].append((q.size(0), bt.size(0)))
        assert q.size(0) > 0 and bt.size(0) > 0, "dense path on an empty prefill slice"
        return torch.full((q.size(0), topk), 2, dtype=torch.int32)

    idx._score_topk_decode = decode
    idx._score_topk_prefill = prefill
    fc = NS(
        context=NS(is_prefill=True),
        attn_metadata=NS(
            indexer_meta=meta,
            block_tables=torch.zeros(num_blocks_rows, 16, dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(plugin_v4, "get_forward_context", lambda: fc)
    return idx, calls


def test_short_extend_alone_is_a_prefill_step():
    # The lone last prompt token at 896 of an 897-token prompt.
    common = NS(
        num_reqs=1,
        is_prefilling=np.array([True]),
        max_query_len=1,
        num_actual_tokens=1,
        _num_computed_tokens_cpu=torch.tensor([896]),
    )
    assert bridge._infer_atom_attn_state(common) == AttnState.PREFILL_PREFIX


def test_prefilling_row_of_decode_length_ends_the_decode_group():
    q1 = np.ones(64, dtype=np.int32)
    # 63 real decodes then the 1-token extend: like native, the extend is a
    # prefill row, so the dense group is not empty.
    pref = torch.zeros(64, dtype=torch.bool)
    pref[63] = True
    assert bridge._indexer_decode_group(q1, 1, pref) == (63, 1)
    # The extend alone.
    assert bridge._indexer_decode_group(
        np.ones(1, dtype=np.int32), 1, torch.tensor([True])
    ) == (0, 1)
    # Extend first in the decode-length run: everything goes dense.
    pref = torch.zeros(4, dtype=torch.bool)
    pref[0] = True
    assert bridge._indexer_decode_group(np.ones(4, dtype=np.int32), 1, pref) == (0, 1)
    # Unchanged when nothing is prefilling: run of equal decode lengths, then a
    # longer prefill row.
    lens = np.array([1, 1, 1, 37], dtype=np.int32)
    assert bridge._indexer_decode_group(
        lens, 1, torch.tensor([False, False, False, True])
    ) == (3, 1)
    assert bridge._indexer_decode_group(
        np.array([4, 4, 1, 4], dtype=np.int32), 4, None
    ) == (2, 4)
    assert bridge._indexer_decode_group(np.array([37, 1], dtype=np.int32), 1, None) == (
        0,
        1,
    )


def test_prefill_step_with_only_decode_length_rows_skips_dense_path(monkeypatch):
    # 63 decodes + the 1-token extend: every row goes to the paged path.
    meta = {
        "num_decode_tokens": 64,
        "num_decodes": 64,
        "decode_next_n": 1,
        "n_committed_per_seq_gpu": torch.full((64,), 224, dtype=torch.int32),
        "total_committed": 1,
    }
    idx, calls = _indexer(monkeypatch, meta, 64)
    q = torch.zeros(64, 4, 8)
    out = idx.indexer_score_topk(q, torch.zeros(64, 4), None, TOPK)
    assert calls["prefill"] == []
    assert calls["decode"] == [(64, 64)]
    assert out.shape == (64, TOPK) and bool((out == 1).all())


def test_mixed_step_still_splits(monkeypatch):
    meta = {
        "num_decode_tokens": 2,
        "num_decodes": 2,
        "decode_next_n": 1,
        "n_committed_per_seq_gpu": torch.full((3,), 224, dtype=torch.int32),
        "total_committed": 300,
    }
    idx, calls = _indexer(monkeypatch, meta, 3)
    q = torch.zeros(5, 4, 8)
    out = idx.indexer_score_topk(q, torch.zeros(5, 4), None, TOPK)
    assert calls["decode"] == [(2, 2)]
    assert calls["prefill"] == [(3, 1)]
    assert out[:2].eq(1).all() and out[2:].eq(2).all()

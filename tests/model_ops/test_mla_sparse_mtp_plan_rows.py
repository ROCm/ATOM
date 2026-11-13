# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A sparse MTP decode must not plan more query rows than its q holds.

The persistent MLA kernel reads q and writes o for every work item, zero-KV
padding included. The MTP drafter's step 0 reuses the target's verify metadata
with the target's unpadded token stream, so a plan over the cudagraph-padded
token count read q and wrote o past their ends (LLM-Booster #287 / #342).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops import attention_mla
from atom.model_ops.attention_mla import MLAAttention
from atom.utils.forward_context import AttentionMetaData

HEADS, RANK, ROPE, Q_LEN = 16, 512, 64, 4


def _layer():
    # Only the decode portion: projections are outside this regression.
    layer = MLAAttention.__new__(MLAAttention)
    torch.nn.Module.__init__(layer)
    layer.num_heads = HEADS
    layer.head_pad = 0
    layer.dcp_world_size = 1
    layer.dcp_rank = 0
    layer.use_seg_mla = False
    layer.use_triton_mla = False
    layer.kv_shuffled = False
    layer.is_sparse_mla = True
    layer.kv_lora_rank = RANK
    layer.qk_rope_head_dim = ROPE
    layer.scale = (RANK + ROPE) ** -0.5
    layer.dtype = torch.bfloat16
    layer._q_scale = torch.ones(())
    layer._k_scale = torch.ones(())
    layer._dpa_persistent_supported = True
    layer.dcp_persistent_supported = True
    layer.sparse_dcp_metadata_rebuild = False
    layer._pad_decode_query_heads = lambda q: q
    layer._restore_decode_query_heads = lambda o, heads: o
    layer._v_up_proj_and_o_proj = lambda o: o
    return layer


def _metadata(planned_tokens):
    metadata = AttentionMetaData(max_seqlen_q=Q_LEN)
    metadata.sparse_cu_seqlens_q = torch.zeros(planned_tokens + 1, dtype=torch.int32)
    metadata.sparse_kv_indptr = torch.zeros(planned_tokens + 1, dtype=torch.int32)
    metadata.sparse_kv_last_page_lens = torch.ones(planned_tokens, dtype=torch.int32)
    metadata.sparse_kv_indices = torch.zeros(64, dtype=torch.int32)
    for name in (
        "work_meta_data",
        "work_indptr",
        "work_info_set",
        "reduce_indptr",
        "reduce_final_map",
        "reduce_partial_map",
    ):
        setattr(metadata, "sparse_mtp_" + name, torch.zeros(1, dtype=torch.int32))
    metadata.sparse_mtp_num_tokens = planned_tokens
    return metadata


def _stub_decode(monkeypatch, seen):
    monkeypatch.setattr(
        attention_mla, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(
        attention_mla, "should_use_persistent_mode", lambda **kwargs: True
    )

    def fake_mla_decode_fwd(q, kv, o, *args, **kwargs):
        seen.append((q.shape[0], o.shape[0]))
        return None, None

    monkeypatch.setattr(attention_mla, "mla_decode_fwd", fake_mla_decode_fwd)


def test_drafter_step0_rows_below_the_padded_plan_are_refused(monkeypatch):
    """39 real requests x 4 query rows, planned as the cudagraph's 40 x 4."""
    seen = []
    _stub_decode(monkeypatch, seen)
    q = torch.zeros(39 * Q_LEN, HEADS, RANK + ROPE)
    cache = torch.zeros(8, RANK + ROPE)
    with pytest.raises(
        AssertionError, match="plan covers 160 query rows but q holds 156"
    ):
        _layer()._forward_decode(q, cache, _metadata(40 * Q_LEN))
    assert seen == [], "the kernel must not be reached with a plan past q"


def test_dcp_rebuild_plans_only_the_rows_q_holds(monkeypatch):
    """Under DCP the sparse_mtp_ plan is rebuilt per layer; its inputs must be
    cut to the drafter's unpadded rows, not the padded running_tokens."""
    _stub_decode(monkeypatch, [])
    layer = _layer()
    layer.dcp_world_size = 4
    layer.sparse_dcp_metadata_rebuild = True
    layer.owns_sparse_indexer = True
    layer.dcp_kernel_num_heads = HEADS
    metadata = _metadata(40 * Q_LEN)
    metadata.dcp_sparse_kv_indptr = torch.zeros(64 * Q_LEN, dtype=torch.int32)
    seen = []

    def fake_rebuild(md, q, kv, cu_seqlens_q, kv_indptr, last_page_lens, *a, **k):
        seen.append((len(cu_seqlens_q), len(kv_indptr), len(last_page_lens)))
        raise StopIteration

    layer._rebuild_sparse_dcp_persistent_metadata = fake_rebuild
    q = torch.zeros(39 * Q_LEN, HEADS, RANK + ROPE)
    with pytest.raises(StopIteration):
        layer._forward_decode(q, torch.zeros(8, RANK + ROPE), metadata)
    assert seen == [(39 * Q_LEN + 1, 39 * Q_LEN + 1, 39 * Q_LEN)]


@pytest.mark.parametrize("q_rows", [39 * Q_LEN, 40 * Q_LEN])
def test_a_plan_within_q_reaches_the_kernel(monkeypatch, q_rows):
    """Plan over the real tokens (#287): the drafter's unpadded stream and the
    target's padded one both hold at least the planned rows."""
    seen = []
    _stub_decode(monkeypatch, seen)
    q = torch.zeros(q_rows, HEADS, RANK + ROPE)
    cache = torch.zeros(8, RANK + ROPE)
    _layer()._forward_decode(q, cache, _metadata(39 * Q_LEN))
    assert seen == [(q_rows, q_rows)]

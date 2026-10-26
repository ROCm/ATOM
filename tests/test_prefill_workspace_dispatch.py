# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exercise real per-layer dispatch and Eagle metadata transitions on CPU tensors.

The production attention/proposer modules import AITER/Triton even though these
routing checks do not launch GPU kernels. Keep that dependency visible in CPU CI.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="real attention/proposer imports require AITER")
pytest.importorskip("triton", reason="real attention/proposer imports require Triton")

from atom.model_engine.sequence import new_block_table
from atom.model_ops import attention_mha
from atom.model_ops.attention_mha import PagedAttentionImpl, SparseMHAPagedAttentionImpl
from atom.model_ops.prefill_workspace import PrefillRequestLengths
from atom.spec_decode import eagle_proposer
from atom.spec_decode.eagle_proposer import EagleProposer
from atom.utils.forward_context import AttentionMetaData


def layer(cls=PagedAttentionImpl):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.gqa_prefill_kv_workspace_bytes = 1024
    obj.use_flash_layout = False
    obj._can_use_prefill_sink_asm = lambda *args: False
    obj._prefill_attention = lambda *args: "dense"
    obj.prefill_attention_triton = lambda *args: "paged"
    obj._dispatch_decode = lambda *args: lambda *args: "decode"
    obj._sparse_prefill = lambda *args: "sparse"
    return obj


@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize(
    "path", ["dense", "paged", "unified", "sink", "sparse", "decode"]
)
def test_grouping_follows_current_layer_dispatch(monkeypatch, is_draft, path):
    monkeypatch.setattr(attention_mha.envs, "ATOM_USE_UNIFIED_ATTN", path == "unified")
    called = []
    monkeypatch.setattr(
        attention_mha,
        "run_grouped_prefill",
        lambda *args: called.append(args) or "grouped",
    )
    obj = layer(SparseMHAPagedAttentionImpl if path == "sparse" else PagedAttentionImpl)
    obj.use_flash_layout = path in ("paged", "sink")
    obj._can_use_prefill_sink_asm = lambda *args: path == "sink"
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=path != "decode", is_draft=is_draft),
        attn_metadata=AttentionMetaData(has_cached=True, max_seqlen_q=1),
    )
    q = torch.empty(2, 4, 8)
    backend = obj.dispatch_backend(ctx, q, q, q)
    result = backend(q, q, q, q, q, None, None, ctx)
    expected = "grouped" if path in ("dense", "sink") else path
    if path == "unified":
        expected = "paged"
    assert result == expected
    assert len(called) == (1 if expected == "grouped" else 0)


@pytest.mark.parametrize("budget,has_cached", [(0, True), (1024, False)])
def test_dense_fast_paths_do_not_group(monkeypatch, budget, has_cached):
    monkeypatch.setattr(
        attention_mha,
        "run_grouped_prefill",
        lambda *args: pytest.fail("fast path must not group"),
    )
    obj = layer()
    obj.gqa_prefill_kv_workspace_bytes = budget
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_draft=True),
        attn_metadata=AttentionMetaData(has_cached=has_cached),
    )
    q = torch.empty(2, 4, 8)
    assert obj.prefill_attention(q, q, q, q, q, None, None, ctx) == "dense"


def test_eagle_clears_prefill_snapshot_when_reusing_metadata_for_decode(monkeypatch):
    md = AttentionMetaData(
        max_seqlen_q=3,
        prefill_request_lengths=PrefillRequestLengths((0, 2, 5), (0, 7, 16)),
    )
    ctx = SimpleNamespace(context=SimpleNamespace(is_prefill=True), attn_metadata=md)
    monkeypatch.setattr(eagle_proposer, "get_forward_context", lambda: ctx)
    var = {
        "slot_mapping": SimpleNamespace(gpu=torch.zeros(2, dtype=torch.int64)),
        "cu_seqlens_q": SimpleNamespace(gpu=torch.zeros(3, dtype=torch.int32)),
        "context_lens": SimpleNamespace(gpu=torch.tensor([7, 9], dtype=torch.int32)),
        "block_tables": SimpleNamespace(gpu=torch.zeros(2, 4, dtype=torch.int32)),
    }
    builder = SimpleNamespace(
        row_ids=torch.arange(3, dtype=torch.int32),
        _publish_dcp_token_block_tables=lambda *args: None,
    )
    obj = SimpleNamespace(
        runner=SimpleNamespace(
            forward_vars=var,
            attn_metadata_builder=builder,
            attn_family=SimpleNamespace(is_mla=False),
        )
    )
    EagleProposer._enter_decode_metadata(
        obj,
        2,
        2,
        torch.arange(5),
        torch.tensor([1, 4]),
        torch.zeros(2, dtype=torch.int32),
    )
    assert ctx.attn_metadata is md
    assert not ctx.context.is_prefill
    assert md.prefill_request_lengths is None
    assert md.cu_seqlens_q.tolist() == [0, 1, 2]


@pytest.mark.parametrize("cached", [False, True])
def test_tbo_straddle_snapshot_matches_final_visible_context(monkeypatch, cached):
    from atom.model_ops.attentions import aiter_attention
    from atom.utils.tbo.ubatch_splitting import UBatchSlice, split_attn_metadata

    cu_q = np.array([0, 2, 5, 9], dtype=np.int32)
    prefix = np.array([5, 6, 4] if cached else [0, 0, 0], dtype=np.int32)
    k_lens = np.diff(cu_q) + prefix
    cu_k = np.concatenate(([0], np.cumsum(k_lens))).astype(np.int32)
    md = AttentionMetaData(
        has_cached=cached,
        cu_seqlens_q=torch.from_numpy(cu_q),
        cu_seqlens_k=torch.from_numpy(cu_k),
        context_lens=torch.from_numpy(k_lens),
        batch_id_per_k_token=torch.repeat_interleave(
            torch.arange(3), torch.from_numpy(k_lens)
        ),
        prefill_request_lengths=PrefillRequestLengths(
            tuple(map(int, cu_q)), tuple(map(int, cu_k))
        ),
    )
    ub_slice = UBatchSlice(slice(1, 3), slice(3, 9))
    ub = split_attn_metadata(md, ub_slice, running_bs=4)
    builder = SimpleNamespace(
        device=torch.device("cpu"),
        _tbo_prefill_state=SimpleNamespace(
            cu_tokens=cu_q,
            num_cached=prefix,
            block_tables=[new_block_table([i]) for i in range(3)],
        ),
    )
    monkeypatch.setattr(
        aiter_attention,
        "upload_numpy",
        lambda data, device: torch.from_numpy(data.copy()),
    )
    aiter_attention.AiterAttentionMetadataBuilder._attach_tbo_token_split_straddle_prefix(
        builder, ub, ub_slice
    )
    expected_k = (0, 9, 17) if cached else (0, 3, 7)
    assert ub.has_cached
    assert ub.prefill_request_lengths == PrefillRequestLengths((0, 2, 6), expected_k)
    assert ub.cu_seqlens_k.tolist() == list(expected_k)
    assert ub.total_kv == expected_k[-1]
    assert ub.block_tables.tolist() == [[1], [2]]
    assert ub.batch_id_per_k_token.tolist() == [0] * expected_k[1] + [1] * (
        expected_k[2] - expected_k[1]
    )
    assert md.prefill_request_lengths.cu_q_lens == (0, 2, 5, 9)

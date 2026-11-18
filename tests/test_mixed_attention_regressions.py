# SPDX-License-Identifier: MIT
"""Default DSpark preparation and mixed attention buffer ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module", autouse=True)
def kernel_imports():
    with atom_without_aiter():
        yield


@pytest.mark.parametrize(
    "width,explicit,deferred,expected",
    [
        (1, False, True, [20, 24]),
        (4, False, True, [19, 22]),
        (1, True, True, [19, 22]),
        (4, False, False, [19, 23]),
        (1, True, False, [19, 23]),
    ],
)
def test_decode_rollback_and_positions(width, explicit, deferred, expected):
    from atom.model_engine.model_runner import tokenIDProcessor
    from atom.model_ops.attentions.deepseek_v4_attn import (
        DeepseekV4AttentionMetadataBuilder,
    )

    class Prepared(Exception):
        pass

    def stop_after_positions(*args):
        raise Prepared

    variables = {
        "positions": SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
        "context_lens": SimpleNamespace(np=np.zeros(2, dtype=np.int32)),
        "batch_id_per_q_token": SimpleNamespace(np=np.repeat(np.arange(2), width)),
    }
    lens = np.full(2, width, dtype=np.int32)
    processor = SimpleNamespace(num_rejected=np.array([1, 2]), is_deferred_out=deferred)
    processor.verify_ctx_lens = tokenIDProcessor.verify_ctx_lens.__get__(processor)
    builder = SimpleNamespace(
        model_runner=SimpleNamespace(
            forward_vars=variables,
            arange_np=np.arange(8, dtype=np.int32),
            tokenID_processor=processor,
        ),
        decode_spans=lambda batch: (2, lens, np.array([0, width, 2 * width])),
        prepare_block_tables=stop_after_positions,
        _check_metadata_writable=lambda *a: None,
        _pad_tokens=lambda *a: None,
    )
    batch = SimpleNamespace(
        context_lens=[20, 24],
        is_dummy_run=False,
        num_spec_step=3,
        total_tokens_num_decode=2 * width,
    )
    counts = np.array([1, 2]) if explicit else None
    with pytest.raises(Prepared):
        DeepseekV4AttentionMetadataBuilder.stage_decode(
            builder, batch, 2, 2 * width, width, None, num_rejected=counts
        )
    assert variables["context_lens"].np.tolist() == expected
    positions = np.concatenate(
        [np.arange(ctx - 4, ctx - 4 + width) for ctx in expected]
    )
    np.testing.assert_array_equal(variables["positions"].np[: 2 * width], positions)
    with pytest.raises(ValueError, match="aligned"):
        DeepseekV4AttentionMetadataBuilder.stage_decode(
            builder, batch, 2, 2 * width, width, None, num_rejected=np.array([1, 2, 3])
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_default_first_decode_keeps_ragged_shrink(mixed):
    from atom.model_engine.model_runner import ModelRunner

    runner = SimpleNamespace(
        config=SimpleNamespace(
            enable_mixed_prefill_decode=mixed,
            dspark=SimpleNamespace(ragged_graph_sizes="4"),
        ),
        tokenID_processor=SimpleNamespace(
            prev_batch=SimpleNamespace(
                req_ids=[3, 4],
                total_seqs_num_prefill=2,
                is_dummy_run=False,
            )
        ),
        enforce_eager=True,
    )
    batch = SimpleNamespace(
        req_ids=[3, 4],
        total_seqs_num_prefill=0,
        is_mixed=False,
        num_scheduled_tokens=np.array([4, 4]),
        num_bonus=np.array([0, 1]),
        scheduled_tokens=np.arange(8),
        total_tokens_num_decode=8,
        total_tokens_num_prefill=0,
        total_tokens_num=8,
    )
    ModelRunner._dspark_apply_ragged(runner, batch, 2, 4, {3: 0, 4: 2})
    assert batch.num_scheduled_tokens.tolist() == ([4, 4] if mixed else [1, 3])


@pytest.mark.parametrize("bf16", [True, False])
def test_attention_output_reuse_keeps_decode_input_intact(monkeypatch, bf16):
    import atom.models.deepseek_v4 as model
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut

    original = torch.arange(24, dtype=torch.bfloat16).reshape(6, 2, 2)
    query = original.clone()
    qkn = QKNormRopeOut(q_sa=query) if bf16 else QKNormRopeOut(q_packed=query)
    metadata = SimpleNamespace(
        num_prefill_tokens=4,
        num_decode_tokens=2,
        prefill_attn_metadata=object(),
        decode_attn_metadata=object(),
    )
    ctx = SimpleNamespace(is_prefill=True, input_ids=torch.arange(6))
    fc = SimpleNamespace(context=ctx, attn_metadata=metadata)
    monkeypatch.setattr(model, "get_forward_context", lambda: fc)
    calls = []

    def attend(segment, positions, *, out, **kwargs):
        sl = slice(0, 4) if ctx.is_prefill else slice(4, 6)
        incoming = segment.q_sa if bf16 else segment.q_packed
        torch.testing.assert_close(incoming, original[sl])
        calls.append(ctx.is_prefill)
        out.copy_(incoming + 10)

    attention = SimpleNamespace(
        n_local_heads=2,
        head_dim=2,
        _qk_norm_rope=lambda *args: qkn,
        _launch_compressors=lambda x: None,
        _sparse_attention=attend,
    )
    out = model.DeepseekV4Attention._attn_mixed(
        attention, metadata, query, query, query, None, query, torch.arange(6)
    )
    assert calls == [True, False]
    assert (out.data_ptr() == query.data_ptr()) == bf16
    torch.testing.assert_close(out, (original + 10).reshape(6, 4))
    assert fc.attn_metadata is metadata and ctx.is_prefill


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_sparse_attention_passes_caller_slice_to_kernel(
    monkeypatch, decode, fp8, ratio
):
    import atom.models.deepseek_v4 as model
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut
    from atom.utils.forward_context import AttnState

    query = torch.ones(3, 2, 2, dtype=torch.bfloat16)
    qkn = (
        QKNormRopeOut(q_packed=query, q_rope=query, k_packed=query, k_rope=query)
        if fp8
        else QKNormRopeOut(q_sa=query, kv=query[:, 0])
    )
    parent = torch.full((5, 2, 2), -9.0, dtype=torch.bfloat16)
    output = parent[1:4]
    metadata = SimpleNamespace(
        state=AttnState.DECODE if decode else AttnState.PREFILL_NATIVE,
        state_slot_out=torch.arange(3),
        qo_indptr=torch.arange(4),
        empty_kv_indptr=torch.zeros(4, dtype=torch.int32),
        cu_seqlens_q=torch.arange(4),
        max_seqlen_q=1,
        kv_indices_extend=object(),
        kv_indptr_extend=object(),
    )
    for region in ("swa", "csa", "hca"):
        for name in (
            "kv_indices",
            "kv_indptr",
            "split_plan",
            "kv_indices_prefix",
            "kv_indptr_prefix",
        ):
            setattr(metadata, f"{name}_{region}", object())
    monkeypatch.setattr(
        model,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False), attn_metadata=metadata
        ),
    )
    monkeypatch.delenv("ATOM_V4_BYPASS_ATTN", raising=False)
    monkeypatch.setattr(model, "_pcp_active", lambda: False)
    calls = []

    def attend(*args, out, **kwargs):
        assert out is output
        assert kwargs["q_packed_in" if decode else "q_packed"] is qkn.q_packed
        region = {0: "swa", 4: "csa", 128: "hca"}[ratio]
        indices = "kv_indices" if decode else "kv_indices_prefix"
        assert args[2] is getattr(metadata, f"{indices}_{region}")
        calls.append("attention")
        return out.fill_(7)

    def write(*args, **kwargs):
        calls.append("swa_write")

    monkeypatch.setattr(model, "sparse_attn_v4_paged_decode", attend)
    monkeypatch.setattr(model, "sparse_attn_v4_paged_2src", attend)
    monkeypatch.setattr(model, "swa_write", write)
    layer = SimpleNamespace(
        n_local_heads=2,
        head_dim=2,
        compress_ratio=ratio,
        _join_compressors=lambda: None,
        indexer=None,
        skip_topk=True,
        unified_kv=object(),
        unified_kv_rope=object() if fp8 else None,
        attn_sink=object(),
        softmax_scale=1.0,
        layer_name="test",
        swa_plane=object(),
        swa_window=object(),
        swa_plane_rope=None,
        window_size=4,
    )
    result = model.DeepseekV4Attention._sparse_attention(
        layer, qkn, torch.arange(3), out=output
    )
    assert calls == (["attention"] if decode else ["attention", "swa_write"])
    assert result.data_ptr() == output.data_ptr() and result.shape == (3, 4)
    assert torch.all(output == 7) and torch.all(parent[[0, -1]] == -9)
    assert torch.all(query == 1)

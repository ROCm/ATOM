# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Default DSpark preparation and mixed attention buffer ownership."""

from functools import partial
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
    processor = SimpleNamespace(
        num_rejected=np.array([9, 9, 1, 2] if explicit else [1, 2]),
        is_deferred_out=deferred,
    )
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


@pytest.mark.parametrize("mixed", [False, True])
def test_attention_constructor_binds_mixed_only_when_enabled(monkeypatch, mixed):
    import atom.models.deepseek_v4 as model

    config = SimpleNamespace(
        enable_mixed_prefill_decode=mixed,
        kv_cache_dtype="auto",
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(model, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(model, "get_gfx_runtime", lambda: "gfx950")
    for name in (
        "MergedReplicatedLinear",
        "ColumnParallelLinear",
        "RowParallelLinear",
        "RMSNorm",
        "_V4RoPE",
    ):
        monkeypatch.setattr(model, name, lambda *a, **kw: torch.nn.Identity())
    args = SimpleNamespace(
        dim=4,
        n_heads=2,
        o_groups=1,
        q_lora_rank=4,
        o_lora_rank=2,
        head_dim=4,
        rope_head_dim=2,
        window_size=4,
        compress_ratios=[0],
        norm_eps=1e-6,
        scale_fmt=None,
        quant_config=None,
        rope_theta=10000,
        max_seq_len=8,
        rope_factor=1,
        beta_fast=32,
        beta_slow=1,
    )
    layer = model.DeepseekV4Attention(0, args, prefix="test.attention")
    assert config.compilation_config.static_forward_context["test.attention"] is layer
    if mixed:
        assert layer.forward_impl.__self__ is layer
        assert (
            layer.forward_impl.__func__ is model.DeepseekV4Attention._forward_impl_mixed
        )
        assert (
            layer.forward_impl_default.__func__
            is model.DeepseekV4Attention.forward_impl
        )
    else:
        assert "forward_impl" not in vars(layer)
        assert "forward_impl_default" not in vars(layer)
        assert layer.forward_impl.__func__ is model.DeepseekV4Attention.forward_impl


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
    decode_outputs = []

    def decode_kernel(query, *, out=None):
        assert out is not None
        decode_outputs.append(out)
        return out.copy_(query + 10)

    monkeypatch.setattr(model, "sparse_attn_v4_paged_decode", decode_kernel)

    def attend(segment, positions, *, out, decode, **kwargs):
        sl = slice(0, 4) if ctx.is_prefill else slice(4, 6)
        incoming = segment.q_sa if bf16 else segment.q_packed
        torch.testing.assert_close(incoming, original[sl])
        calls.append(ctx.is_prefill)
        if ctx.is_prefill:
            out.copy_(incoming + 10)
        else:
            assert decode(incoming).data_ptr() == out.data_ptr()

    def norm(q, kv, positions, mixed=False):
        assert mixed is True
        return qkn

    attention = SimpleNamespace(
        n_local_heads=2,
        head_dim=2,
        indexer=None,
        _qk_norm_rope=norm,
        _launch_compressors=lambda x: None,
        _sparse_attention=attend,
    )
    out = model.DeepseekV4Attention._attn_mixed(
        attention, metadata, query, query, query, None, query, torch.arange(6)
    )
    assert calls == [True, False]
    assert (out.data_ptr() == query.data_ptr()) == bf16
    assert len(decode_outputs) == 1
    assert decode_outputs[0].data_ptr() == out[4:].data_ptr()
    assert (
        decode_outputs[0].untyped_storage().data_ptr()
        == out.untyped_storage().data_ptr()
    )
    torch.testing.assert_close(out, (original + 10).reshape(6, 4))
    assert fc.attn_metadata is metadata and ctx.is_prefill


@pytest.mark.parametrize("fp4_index", [False, True])
def test_mixed_projects_opaque_qr_scales_before_slicing(monkeypatch, fp4_index):
    import atom.models.deepseek_v4 as model
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import QKNormRopeOut

    torch.manual_seed(41)
    rows, split, groups, heads, dim = 9, 5, 2, 2, 4
    qr = torch.randn(rows, 256)
    # Exactly the column-major storage q_norm exposes as shape [M, G].
    logical_scale = torch.rand(rows, groups) + torch.arange(rows)[:, None]
    scale = logical_scale.T.contiguous().view(rows, groups)
    weight = torch.randn(heads * dim, 256)
    keys = torch.randn(64, dim)
    projected_rows = []

    def project(qr, *, x_scale):
        n = qr.shape[0]
        projected_rows.append(n)
        dequant = qr * x_scale.reshape(groups, n).T.repeat_interleave(128, 1)
        return dequant @ weight.T

    def quantize(q, *args, **kwargs):
        # The indexer's output scales, unlike qr_scale, ARE token-major.
        s = q.abs().amax(-1, keepdim=True).clamp_min(1e-4)
        return q / s, s

    monkeypatch.setattr(model, "quantize_query", quantize)
    monkeypatch.setattr(
        model, "fold_query_scale", lambda w, s, *a, **kw: w * s.squeeze(-1)
    )
    indexer = SimpleNamespace(
        wq_b=project,
        weights_proj=lambda x: x,
        n_heads=heads,
        head_dim=dim,
        rope_head_dim=0,
        rotary_emb=SimpleNamespace(cos_cache=None, sin_cache=None),
        _indexer_fp4=fp4_index,
        _layout=None,
        _weights_scale=1,
        prefix="test",
    )
    indexer.forward_pre = model.Indexer.forward_pre.__get__(indexer)
    indexer.forward_batched = model.Indexer.forward_batched.__get__(indexer)

    def score(q, weights, q_scale):
        q = q if q_scale is None else q * q_scale
        logits = torch.einsum("thd,kd->thk", q, keys).relu()
        return (logits * weights[:, :, None]).sum(1).topk(8).indices

    indexer.score_topk_from = score
    x = torch.rand(rows, heads)
    pos = torch.arange(rows)
    expected = []
    for sl in (slice(0, split), slice(split, rows)):
        # Separate forwards get independently packed q_norm scales.
        packed = logical_scale[sl].T.contiguous().view(-1, groups)
        expected.append(score(*indexer.forward_pre(x[sl], qr[sl], pos[sl], packed)))
    projected_rows.clear()
    metadata = SimpleNamespace(
        num_prefill_tokens=split,
        num_decode_tokens=rows - split,
        prefill_attn_metadata=object(),
        decode_attn_metadata=object(),
    )
    ctx = SimpleNamespace(is_prefill=True, input_ids=pos)
    fc = SimpleNamespace(context=ctx, attn_metadata=metadata)
    monkeypatch.setattr(model, "get_forward_context", lambda: fc)
    selections = []

    def attend(
        qkn,
        positions,
        *,
        out,
        idx_q_quant=None,
        idx_weights=None,
        idx_q_scale=None,
        **kw,
    ):
        selections.append(
            model.Indexer.topk(
                indexer,
                kw["x"],
                kw.get("qr"),
                positions,
                kw.get("qr_scale"),
                pre_q_quant=idx_q_quant,
                pre_weights=idx_weights,
                pre_q_scale=idx_q_scale,
            )
        )
        out.zero_()

    query = torch.zeros(rows, heads, dim)
    attention = SimpleNamespace(
        n_local_heads=heads,
        head_dim=dim,
        indexer=indexer,
        skip_topk=False,
        _qk_norm_rope=lambda *a, **kw: QKNormRopeOut(q_sa=query),
        _launch_compressors=lambda *a: None,
        _sparse_attention=attend,
    )
    model.DeepseekV4Attention._attn_mixed(
        attention, metadata, query, query, qr, scale, x, pos
    )
    torch.testing.assert_close(
        torch.cat(selections), torch.cat(expected), rtol=0, atol=0
    )
    assert projected_rows == [rows]


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
        layer, qkn, torch.arange(3), out=output, decode=partial(attend, out=output)
    )
    assert calls == (["attention"] if decode else ["attention", "swa_write"])
    assert result.data_ptr() == output.data_ptr() and result.shape == (3, 4)
    assert torch.all(output == 7) and torch.all(parent[[0, -1]] == -9)
    assert torch.all(query == 1)


@pytest.mark.parametrize("mixed", [False, True])
def test_full_forward_selects_mixed_only_when_needed(monkeypatch, mixed):
    import atom.models.deepseek_v4 as model

    x = torch.ones(2, 4)
    metadata = SimpleNamespace(is_mixed=mixed)
    monkeypatch.setattr(
        model,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False), attn_metadata=metadata
        ),
    )
    monkeypatch.delenv("ATOM_V4_BYPASS_ATTN", raising=False)
    calls = []
    layer = SimpleNamespace(
        forward_impl_default=lambda *a: calls.append("default") or x,
        _attn_pre=lambda *a, **kw: (x, x, x, None),
        _attn_mixed=lambda *a: calls.append("mixed") or x,
        _attn_post=lambda o, p: o,
    )
    assert model.DeepseekV4Attention._forward_impl_mixed(layer, x, torch.arange(2)) is x
    assert calls == (["mixed"] if mixed else ["default"])


@pytest.mark.parametrize("mixed,decode", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("fp8", [False, True])
def test_qkn_routes_only_decode_rows_to_fused_swa(monkeypatch, mixed, decode, fp8):
    import atom.models.deepseek_v4 as model
    from atom.utils.forward_context import AttnState

    mask = torch.tensor([-1, 0, 1]) if mixed else torch.arange(3)
    rows = torch.tensor([9, 10, 11])
    md = SimpleNamespace(
        state=AttnState.DECODE if decode else AttnState.PREFILL_NATIVE,
        swa_dest_rows={4: rows},
        batch_id_per_q_token=mask,
    )
    monkeypatch.setattr(
        model,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False), attn_metadata=md
        ),
    )
    monkeypatch.delenv("ATOM_V4_BYPASS_ATTN", raising=False)
    monkeypatch.setattr(model, "_V4_USE_REF_QUANT", False)
    calls = []
    monkeypatch.setattr(
        model, "qk_norm_rope_maybe_quant", lambda *a, **kw: calls.append(kw)
    )
    layer = SimpleNamespace(
        rope_head_dim=2,
        compress_ratio=4,
        kv_norm=SimpleNamespace(weight=None),
        rotary_emb=SimpleNamespace(cos_cache=None, sin_cache=None),
        n_local_heads=2,
        head_dim=4,
        eps=1e-6,
        kv_fp8=fp8,
        swa_plane=object(),
        swa_plane_rope=object(),
        layer_name="test",
    )
    model.DeepseekV4Attention._qk_norm_rope(
        layer, torch.zeros(3, 8), None, torch.arange(3), mixed
    )
    kw = calls[0]
    enabled = mixed or decode
    assert kw["batch_id_per_q_token"] is (mask if enabled else None)
    assert kw["swa_dest_rows"] is (rows if enabled else None)
    key = "swa_nope_scale_buff" if fp8 else "swa_kv"
    assert kw[key] is (layer.swa_plane if enabled else None)

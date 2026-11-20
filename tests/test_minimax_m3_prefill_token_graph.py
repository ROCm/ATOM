# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill token graphs: its attention region on a padded prefill
against the unpadded one (M3's sparse impl, kernels replaced by references)
and, from the attention builder's side, the
deployments that record them, the dummy prefill a bucket is recorded on and
which live prefills still run eagerly.

The builders are made with ``__new__`` and given only the state each method
reads, so no device is needed; their tests still need triton and aiter, which
the builder's module imports."""

from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context

_BUDGET = 8192
H, D = 2, 4  # query heads, head dim; one KV head and one index head


def _m3_impl(attention_mha):
    """M3's sparse attention impl with what its prefill reads: an MXFP4 KV
    cache, whose attention is an attribute here, and an fp8 index cache."""
    impl = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    impl.num_heads, impl.num_kv_heads, impl.head_dim, impl.scale = H, 1, D, 1.0
    impl.num_idx_heads, impl.index_q_size, impl.index_head_dim = 1, D, D
    impl.kv_cache_dtype, impl.index_cache_dtype = "fp4", "fp8"
    impl.skip_index_topk, impl.indexer_cp_group = False, None
    impl.topk = impl.init_blocks = impl.local_blocks = 1
    impl.index_topk_cache_state = impl._index_q = impl._index_q_cache_key_info = None
    impl._select_page_kwargs, impl.layer_num = {}, 0
    norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
    impl.q_norm = impl.k_norm = impl.index_q_norm = impl.index_k_norm = norm
    impl.rotary_emb = SimpleNamespace(rotary_dim=D)
    return impl


def _kernels(monkeypatch, attention_mha, impl, slots):
    """The fused norm/RoPE/cache write, the index top-k and the sparse
    attention as references, each refusing rows the metadata does not cover.
    The top-k's per-row weight stands in for its sparse block table."""
    from atom.model_ops.minimax_m3 import index_topk
    from atom.models.minimax_m3 import model

    def write(qkv, *args, **kwargs):
        position, slot_mapping, k_cache, v_cache, index_cache = args[3], *args[11:15]
        q_out, index_q = args[16], args[17]
        assert qkv.shape[0] == position.shape[-1] == slot_mapping.shape[0]
        q, k, v, iq, ik = (qkv * (position[:, None] + 1)).split(
            [H * D, D, D, D, D], dim=-1
        )
        q_out.copy_(q)
        index_q.copy_(iq)
        k_cache[slot_mapping], v_cache[slot_mapping] = k, v
        index_cache[slot_mapping] = ik

    def top_k(index_q, *args, **kwargs):
        assert index_q.shape[0] == slots.shape[0]
        return None, index_q.float().sum((-2, -1)), None

    def attend(q, k_cache, v_cache, weight, *args, out):
        assert q.shape[0] == slots.shape[0]
        kv = (k_cache[slots] + v_cache[slots]).cumsum(0)  # causal over the cache
        out.copy_(q.float() * weight[:, None, None] + kv[:, None, :])

    monkeypatch.setattr(attention_mha.aiter, "fused_qknorm_idxrqknorm", write, False)
    monkeypatch.setattr(model, "_minimax_m3_cos_sin_cache", lambda *args: None)
    monkeypatch.setattr(index_topk, "minimax_m3_index_topk", top_k)
    impl._sparse_pa_mxfp4 = attend


@pytest.mark.parametrize("bucket", [5, 8])
def test_a_padded_m3_prefill_attends_and_writes_like_the_eager_one(monkeypatch, bucket):
    """M3's attention region on a prefill padded to its token-graph bucket,
    against the same prefill unpadded: the same output rows, zeros past them,
    the same KV and index cache; its kernels see the scheduled rows only."""
    attention_mha = pytest.importorskip("atom.model_ops.attention_mha")
    base = pytest.importorskip("atom.model_ops.base_attention")
    rows, slots = 5, torch.tensor([9, 2, 7, 4, 11])
    impl = _m3_impl(attention_mha)
    _kernels(monkeypatch, attention_mha, impl, slots)
    layers = {"layer": SimpleNamespace(impl=impl)}
    compilation = SimpleNamespace(static_forward_context=layers)
    monkeypatch.setattr(
        base,
        "get_current_atom_config",
        lambda: SimpleNamespace(compilation_config=compilation),
    )
    prefill = SimpleNamespace(
        cu_seqlens_q=None,
        seq_lens=torch.tensor([rows]),
        context_lens=None,
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        max_query_len=rows,
        max_seq_len=rows,
        qo_indptr=torch.arange(rows),
    )
    metadata = SimpleNamespace(
        slot_mapping=slots, prefill=prefill, n_valid_column_per_row=None
    )
    torch.manual_seed(3)
    qkv = torch.randn(bucket, (H + 4) * D)
    qkv[rows:] = 1e4  # pad rows: they must reach neither the output nor a cache
    positions = torch.arange(bucket) + 7

    def run(n):
        caches = [torch.zeros(16, D) for _ in range(3)]
        layer = SimpleNamespace(k_cache=caches[0], v_cache=caches[1])
        layer.k_scale = layer.v_scale = None
        impl.index_cache = caches[2]
        step = SimpleNamespace(
            is_prefill=True, is_dummy_run=False, scheduled_tokens=rows
        )
        ctx = SimpleNamespace(
            context=step,
            attn_metadata=metadata,
            kv_cache_data={"layer_0": layer},
            tr_graph=None,
            in_hipgraph=False,
            main_stream=None,
        )
        monkeypatch.setattr(forward_context, "_forward_context", ctx)
        q, k, v, _, _ = qkv[:n].split([H * D, D, D, D, D], dim=-1)
        out = base._attention_region(
            q, None, k, v, positions[:n], "layer", False, qkv[:n]
        )
        return out, caches

    eager, eager_caches = run(rows)
    padded, padded_caches = run(bucket)

    assert padded.shape == (bucket, H * D) and eager.abs().sum() > 0
    torch.testing.assert_close(padded[:rows], eager, rtol=0, atol=0)
    assert torch.count_nonzero(padded[rows:]) == 0
    for got, want in zip(padded_caches, eager_caches):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.fixture
def aa():
    pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
    pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")
    return pytest.importorskip("atom.model_ops.attentions.aiter_attention")


def _builder(aa, sparse=True, dp=1, sp=1):
    builder = aa.AiterAttentionMetadataBuilder.__new__(aa.AiterAttentionMetadataBuilder)
    builder._has_sparse_attention = sparse
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=dp),
            sequence_parallel_size=sp,
            max_num_batched_tokens=_BUDGET,
        ),
        forward_vars={"positions": SimpleNamespace(gpu=torch.arange(_BUDGET))},
    )
    return builder


@pytest.mark.parametrize(
    "builder, enabled",
    [
        ({}, True),
        ({"dp": 2}, False),
        ({"sp": 2}, False),
        # Every other model this builder serves has no sparse attention.
        ({"sparse": False}, False),
    ],
)
def test_token_graphs_are_on_for_m3_outside_dp_and_sp(aa, builder, enabled):
    assert _builder(aa, **builder).supports_token_graph is enabled


@pytest.mark.parametrize("tokens", [1, 256, _BUDGET])
def test_a_bucket_records_on_a_dummy_prefill_of_its_height(aa, tokens):
    builder = _builder(aa)
    attn_metadata, context = builder.build_for_token_graph_capture(tokens)

    assert context.is_prefill and context.is_dummy_run
    assert (context.scheduled_bs, context.running_bs) == (1, 1)
    assert (context.scheduled_tokens, context.running_tokens) == (tokens, tokens)
    assert (attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k) == (
        tokens,
        tokens,
    )
    # The recording reads the persistent positions a live step fills.
    buffer = builder.model_runner.forward_vars["positions"].gpu
    assert context.positions.data_ptr() == buffer.data_ptr()
    assert context.positions.shape == (tokens,) and not context.positions.any()


@pytest.mark.parametrize("tokens", [0, _BUDGET + 1])
def test_a_bucket_outside_the_token_budget_is_refused(aa, tokens):
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _builder(aa).build_for_token_graph_capture(tokens)


@pytest.mark.parametrize(
    "enabled, prefill_tokens, images, eager",
    [
        (True, 512, {7: {"pixel_values": None}}, True),
        (True, 512, {}, False),
        # Decode and a builder without token graphs keep the base answer.
        (True, 0, {7: {"pixel_values": None}}, False),
        (False, 512, {7: {"pixel_values": None}}, False),
    ],
)
def test_a_prefill_carrying_images_stays_eager(
    aa, enabled, prefill_tokens, images, eager
):
    batch = SimpleNamespace(
        total_tokens_num_prefill=prefill_tokens, multimodal_data=images
    )
    assert _builder(aa, sparse=enabled).step_needs_eager(batch) is eager

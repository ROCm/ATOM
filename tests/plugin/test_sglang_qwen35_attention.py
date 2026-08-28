from types import SimpleNamespace

import torch

from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
    ATOMAttnBackendForSgl,
)
from atom.plugin.sglang.models import qwen3_5_attention as bridge
from atom.plugin.sglang.register import (
    _register_tc_piecewise_attention_split_ops,
)
from atom.utils.forward_context import AttnState


class _Mode:
    def __init__(self, *, decode: bool):
        self.decode = decode

    def is_decode_or_idle(self):
        return self.decode

    def is_prefill(self):
        return not self.decode


class _TokenPool:
    page_size = 1

    def __init__(self):
        self.k = torch.empty(8, 2, 16, dtype=torch.bfloat16)
        self.v = torch.empty_like(self.k)

    def get_kv_buffer(self, layer_id):
        assert layer_id == 3
        return self.k, self.v


def test_page_one_cache_does_not_use_asm_shuffle_layout():
    k_cache = torch.empty(8, 2, 8, 1, 16, dtype=torch.uint8)
    v_cache = torch.empty(8, 2, 128, 1, dtype=torch.uint8)

    assert (
        bridge._use_sglang_asm_cache_layout(k_cache, v_cache, use_triton_attn=False)
        is False
    )
    assert (
        PagedAttentionImpl._use_asm_cache_layout(
            None, k_cache, v_cache, use_triton_attn=False
        )
        is True
    )


def _forward_batch(*, decode: bool, seq_lens, extend_lens=None):
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    batch_size = int(seq_lens.numel())
    total_tokens = batch_size if decode else int(sum(extend_lens))
    return SimpleNamespace(
        batch_size=batch_size,
        forward_mode=_Mode(decode=decode),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        extend_seq_lens=(
            None
            if extend_lens is None
            else torch.tensor(extend_lens, dtype=torch.int32)
        ),
        extend_seq_lens_cpu=extend_lens,
        req_pool_indices=torch.arange(batch_size),
        out_cache_loc=torch.arange(total_tokens, dtype=torch.int64),
    )


def test_qwen35_construction_context_selects_native_attention():
    from atom.model_ops import base_attention
    from atom.models import qwen3_next

    previous_attention = base_attention.Attention
    previous_base_linear_attention = base_attention.LinearAttention
    previous_qwen_linear_attention = qwen3_next.LinearAttention
    with bridge.qwen35_native_attention_construction():
        assert base_attention.Attention is bridge.SGLangATOMQwen35Attention
        assert base_attention.LinearAttention is bridge.SGLangATOMQwen35LinearAttention
        assert qwen3_next.LinearAttention is bridge.SGLangATOMQwen35LinearAttention
    assert base_attention.Attention is previous_attention
    assert base_attention.LinearAttention is previous_base_linear_attention
    assert qwen3_next.LinearAttention is previous_qwen_linear_attention


def test_qwen35_split_ops_write_into_stable_output_buffers(monkeypatch):
    mha_output = torch.empty(2, 3)
    gdn_output = torch.empty(2, 3)

    class _MHAImpl:
        def forward(self, **kwargs):
            return torch.full_like(kwargs["query"], 7)

    class _GDNImpl:
        def forward(self, _mixed_qkv, _b, _a, output, _layer_name):
            output.fill_(11)
            return output

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "mha": SimpleNamespace(impl=_MHAImpl()),
                "gdn": SimpleNamespace(impl=_GDNImpl()),
            }
        )
    )
    monkeypatch.setattr(bridge, "get_current_atom_config", lambda: config)

    mha_address = mha_output.data_ptr()
    bridge.sglang_qwen35_attention_with_stable_output(
        torch.empty_like(mha_output),
        None,
        torch.empty_like(mha_output),
        torch.empty_like(mha_output),
        None,
        "mha",
        None,
        mha_output,
    )
    gdn_address = gdn_output.data_ptr()
    bridge.sglang_qwen35_linear_attention_with_stable_output(
        torch.empty(2),
        torch.empty(2),
        torch.empty(2),
        gdn_output,
        "gdn",
    )

    assert mha_output.data_ptr() == mha_address
    assert mha_output.tolist() == [[7, 7, 7], [7, 7, 7]]
    assert gdn_output.data_ptr() == gdn_address
    assert gdn_output.tolist() == [[11, 11, 11], [11, 11, 11]]


def test_qwen35_compile_only_prefill_uses_native_returning_ops(monkeypatch):
    monkeypatch.setattr(bridge, "is_compile_only_prefill_active", lambda: True)
    calls = []

    def native_mha(*args):
        calls.append(("mha", args[5]))
        return torch.full_like(args[0], 7)

    def native_gdn(*args):
        calls.append(("gdn", args[4]))
        return torch.full_like(args[3], 11)

    monkeypatch.setattr(
        torch.ops.aiter, "unified_attention_with_output_base", native_mha
    )
    monkeypatch.setattr(
        torch.ops.aiter, "linear_attention_with_output_base", native_gdn
    )

    mha = bridge.SGLangATOMQwen35Attention.__new__(bridge.SGLangATOMQwen35Attention)
    torch.nn.Module.__init__(mha)
    mha.layer_name = "mha"
    query = torch.empty(2, 3)
    mha_output = mha.forward(query, query, query)

    gdn = bridge.SGLangATOMQwen35LinearAttention.__new__(
        bridge.SGLangATOMQwen35LinearAttention
    )
    torch.nn.Module.__init__(gdn)
    gdn.layer_name = "gdn"
    core_attn_out = torch.empty(2, 3)
    gdn_output = gdn.forward(
        torch.empty(2),
        torch.empty(2),
        torch.empty(2),
        core_attn_out,
    )

    assert calls == [("mha", "mha"), ("gdn", "gdn")]
    assert mha_output.tolist() == [[7, 7, 7], [7, 7, 7]]
    assert gdn_output.tolist() == [[11, 11, 11], [11, 11, 11]]
    assert gdn_output.data_ptr() != core_attn_out.data_ptr()


def test_qwen35_piecewise_prefill_selects_stable_output_ops(monkeypatch):
    monkeypatch.setattr(bridge, "is_compile_only_prefill_active", lambda: False)
    monkeypatch.setattr(bridge, "is_in_tc_piecewise_cuda_graph", lambda: True)

    assert bridge._use_stable_piecewise_output() is True


def test_qwen35_eager_paths_select_native_returning_ops(monkeypatch):
    monkeypatch.setattr(bridge, "is_compile_only_prefill_active", lambda: False)
    monkeypatch.setattr(bridge, "is_in_tc_piecewise_cuda_graph", lambda: False)

    assert bridge._use_stable_piecewise_output() is False


def test_qwen35_compile_only_overrides_piecewise_context(monkeypatch):
    monkeypatch.setattr(bridge, "is_compile_only_prefill_active", lambda: True)
    monkeypatch.setattr(bridge, "is_in_tc_piecewise_cuda_graph", lambda: True)

    assert bridge._use_stable_piecewise_output() is False


def test_qwen35_prefill_split_ops_trim_padded_tokens(monkeypatch):
    padded_num_tokens = 5
    real_num_tokens = 3
    forward_batch = SimpleNamespace(out_cache_loc=torch.tensor([5, 6, 7, 0, 0]))
    monkeypatch.setattr(
        bridge,
        "_resolve_tc_piecewise_real_tokens",
        lambda _padded: (forward_batch, real_num_tokens),
    )

    class _MHAImpl:
        def forward(self, **kwargs):
            assert forward_batch.out_cache_loc.tolist() == [5, 6, 7]
            assert kwargs["query"].shape[0] == real_num_tokens
            assert kwargs["key"].shape[0] == real_num_tokens
            assert kwargs["value"].shape[0] == real_num_tokens
            assert kwargs["position"].shape[0] == real_num_tokens
            assert kwargs["q_scale"].shape[0] == real_num_tokens
            assert kwargs["qkv"].shape[0] == real_num_tokens
            return torch.full_like(kwargs["query"], 7)

    class _GDNImpl:
        def forward(self, mixed_qkv, b, a, output, _layer_name):
            assert mixed_qkv.shape[0] == real_num_tokens
            assert b.shape[0] == real_num_tokens
            assert a.shape[0] == real_num_tokens
            assert output.shape[0] == real_num_tokens
            output.fill_(11)

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={
                "mha": SimpleNamespace(impl=_MHAImpl()),
                "gdn": SimpleNamespace(impl=_GDNImpl()),
            }
        )
    )
    monkeypatch.setattr(bridge, "get_current_atom_config", lambda: config)

    mha_output = torch.full((padded_num_tokens, 3), -1.0)
    bridge.sglang_qwen35_attention_with_stable_output(
        torch.empty_like(mha_output),
        torch.empty(padded_num_tokens),
        torch.empty_like(mha_output),
        torch.empty_like(mha_output),
        torch.empty(padded_num_tokens, dtype=torch.int64),
        "mha",
        torch.empty_like(mha_output),
        mha_output,
    )
    assert forward_batch.out_cache_loc.tolist() == [5, 6, 7, 0, 0]
    assert torch.all(mha_output[:real_num_tokens] == 7)
    assert torch.all(mha_output[real_num_tokens:] == 0)

    gdn_output = torch.full((padded_num_tokens, 2, 3), -1.0)
    bridge.sglang_qwen35_linear_attention_with_stable_output(
        torch.empty(padded_num_tokens, 4),
        torch.empty(padded_num_tokens, 2),
        torch.empty(padded_num_tokens, 2),
        gdn_output,
        "gdn",
    )
    assert torch.all(gdn_output[:real_num_tokens] == 11)
    assert torch.all(gdn_output[real_num_tokens:] == 0)


def test_qwen35_prefill_metadata_matches_atom_contract(monkeypatch):
    monkeypatch.setattr("atom.utils.envs.ATOM_USE_UNIFIED_ATTN", False)
    forward_batch = _forward_batch(
        decode=False,
        seq_lens=[3, 2],
        extend_lens=[3, 2],
    )
    token_pool = SimpleNamespace(page_size=2)
    req_pool = SimpleNamespace(
        req_to_token=torch.tensor(
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int64
        )
    )

    metadata = bridge.build_qwen35_attention_metadata(
        forward_batch,
        torch.arange(5),
        token_pool=token_pool,
        req_pool=req_pool,
    )

    assert metadata.state is AttnState.PREFILL_NATIVE
    assert metadata.has_cached is False
    assert metadata.cu_seqlens_q.tolist() == [0, 3, 5]
    assert metadata.cu_seqlens_k.tolist() == [0, 3, 5]
    assert metadata.context_lens.tolist() == [3, 2]
    assert metadata.block_tables.tolist() == [[0, 1, 2], [3, 4, 5]]


def test_qwen35_prefix_prefill_separates_total_and_cached_lengths(monkeypatch):
    monkeypatch.setattr("atom.utils.envs.ATOM_USE_UNIFIED_ATTN", False)
    forward_batch = _forward_batch(
        decode=False,
        seq_lens=[5, 4],
        extend_lens=[2, 3],
    )
    token_pool = SimpleNamespace(page_size=1)
    req_pool = SimpleNamespace(
        req_to_token=torch.tensor(
            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]], dtype=torch.int64
        )
    )

    metadata = bridge.build_qwen35_attention_metadata(
        forward_batch,
        torch.arange(5),
        token_pool=token_pool,
        req_pool=req_pool,
    )

    assert metadata.state is AttnState.PREFILL_PREFIX
    assert metadata.context_lens.tolist() == [5, 4]
    assert metadata.num_cached_tokens.tolist() == [3, 1]
    assert metadata.seq_starts.tolist() == [0, 0]
    assert metadata.block_tables.tolist() == [
        [0, 1, 2, 0, 1],
        [6, 2, 3, 4, 10],
    ]


def test_qwen35_prefill_metadata_uses_real_token_count(monkeypatch):
    forward_batch = _forward_batch(
        decode=False,
        seq_lens=[2, 1],
        extend_lens=[2, 1],
    )
    forward_batch.out_cache_loc = torch.tensor([5, 6, 7, 0, 0, 0, 0, 0])
    monkeypatch.setattr(
        bridge,
        "_resolve_tc_piecewise_real_tokens",
        lambda _padded: (forward_batch, 3),
    )

    metadata = bridge.build_qwen35_attention_metadata(
        forward_batch,
        torch.arange(8),
        token_pool=SimpleNamespace(page_size=2),
        req_pool=SimpleNamespace(req_to_token=torch.empty(2, 16)),
    )
    assert metadata.slot_mapping.tolist() == [5, 6, 7]
    assert metadata.total_kv == 3


def test_qwen35_attention_ops_are_tc_piecewise_split_boundaries():
    from sglang.srt.compilation.compilation_config import SPLIT_OPS

    original_split_ops = list(SPLIT_OPS)
    try:
        _register_tc_piecewise_attention_split_ops()
        _register_tc_piecewise_attention_split_ops()

        assert SPLIT_OPS.count("aiter.sglang_qwen35_attention_with_stable_output") == 1
        assert (
            SPLIT_OPS.count("aiter.sglang_qwen35_linear_attention_with_stable_output")
            == 1
        )
        assert SPLIT_OPS.count("aiter.unified_attention_with_output_base") == 1
        assert SPLIT_OPS.count("aiter.linear_attention_with_output_base") == 1
    finally:
        SPLIT_OPS[:] = original_split_ops


def test_qwen35_decode_reuses_graph_stable_backend_metadata(monkeypatch):
    forward_batch = _forward_batch(decode=True, seq_lens=[4, 6])
    page_table = torch.tensor([[1, 2, 0], [3, 4, 5]], dtype=torch.int32)
    kv_lens = torch.tensor([4, 6], dtype=torch.int32)
    monkeypatch.setattr(
        bridge,
        "_full_attention_backend",
        lambda _batch: SimpleNamespace(
            forward_metadata=SimpleNamespace(
                page_table=page_table,
                kv_lens=kv_lens,
            )
        ),
    )
    token_pool = SimpleNamespace(page_size=2)
    req_pool = SimpleNamespace(req_to_token=torch.empty(2, 16, dtype=torch.int64))

    metadata = bridge.build_qwen35_attention_metadata(
        forward_batch,
        torch.arange(2),
        token_pool=token_pool,
        req_pool=req_pool,
    )

    assert metadata.state is AttnState.DECODE
    assert metadata.block_tables.data_ptr() == page_table.data_ptr()
    assert metadata.context_lens.data_ptr() == kv_lens.data_ptr()
    assert metadata.max_seqlen_k == page_table.shape[1] * token_pool.page_size
    assert metadata.slot_mapping.tolist() == [5, 11]


def test_qwen35_decode_padding_rows_do_not_write_kv(monkeypatch):
    forward_batch = _forward_batch(decode=True, seq_lens=[4, 0])
    page_table = torch.tensor([[1, 2, 0], [0, 0, 0]], dtype=torch.int32)
    kv_lens = torch.tensor([4, 0], dtype=torch.int32)
    monkeypatch.setattr(
        bridge,
        "_full_attention_backend",
        lambda _batch: SimpleNamespace(
            forward_metadata=SimpleNamespace(
                page_table=page_table,
                kv_lens=kv_lens,
            )
        ),
    )
    token_pool = SimpleNamespace(page_size=2)
    req_pool = SimpleNamespace(req_to_token=torch.empty(2, 16, dtype=torch.int64))

    metadata = bridge.build_qwen35_attention_metadata(
        forward_batch,
        torch.arange(2),
        token_pool=token_pool,
        req_pool=req_pool,
    )

    assert metadata.slot_mapping.tolist() == [5, -1]


def test_decode_graph_replay_clears_padding_kv_lens(monkeypatch):
    backend = ATOMAttnBackendForSgl.__new__(ATOMAttnBackendForSgl)

    def replay_metadata(*_args):
        backend.forward_metadata = SimpleNamespace(
            kv_lens=torch.tensor([4, 1, 1, 1], dtype=torch.int32),
            page_table=torch.tensor(
                [[1, 2, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                dtype=torch.int32,
            ),
        )

    backend.init_forward_metadata_replay_cuda_graph = replay_metadata
    forward_batch = SimpleNamespace(
        batch_size=4,
        num_padding=3,
        forward_mode=_Mode(decode=True),
        req_pool_indices=torch.tensor([0, 0, 0, 0]),
        seq_lens=torch.tensor([4, 1, 1, 1]),
        seq_lens_sum=7,
        encoder_lens=None,
        spec_info=None,
        seq_lens_cpu=torch.tensor([4, 1, 1, 1]),
        out_cache_loc=None,
    )

    backend.init_forward_metadata_out_graph(forward_batch)

    assert backend.forward_metadata.kv_lens.tolist() == [4, 0, 0, 0]
    monkeypatch.setattr(bridge, "_full_attention_backend", lambda _batch: backend)
    metadata = bridge.build_qwen35_attention_metadata(
        forward_batch,
        torch.arange(4),
        token_pool=SimpleNamespace(page_size=2),
        req_pool=SimpleNamespace(req_to_token=torch.empty(1, 16)),
    )
    assert metadata.slot_mapping.tolist() == [5, -1, -1, -1]


def test_qwen35_forward_metadata_combines_mha_and_gdn(monkeypatch):
    from atom.plugin.sglang.attention_backend.attention_gdn import (
        SGLangGDNForwardContext,
    )
    from atom.plugin.sglang.runtime import SGLangForwardBatchMetadata

    forward_batch = _forward_batch(decode=True, seq_lens=[4])
    positions = torch.arange(1)
    mha_metadata = SimpleNamespace(slot_mapping=torch.tensor([7]))
    gdn_metadata = object()
    monkeypatch.setattr(bridge, "_mha_pools", lambda _batch: (object(), object()))
    monkeypatch.setattr(
        bridge,
        "build_qwen35_attention_metadata",
        lambda *_args, **_kwargs: mha_metadata,
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_resolve_attn_backend",
        lambda _batch: object(),
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_linear_attn_backend",
        lambda _backend: object(),
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_build_gdn_metadata",
        lambda _batch, _backend: gdn_metadata,
    )

    metadata = bridge.build_qwen35_forward_metadata(
        SGLangForwardBatchMetadata(
            forward_batch=forward_batch,
            save_kv_cache=False,
        ),
        positions,
    )

    assert metadata is mha_metadata
    assert metadata.gdn_metadata is gdn_metadata
    assert metadata.slot_mapping.tolist() == [-1]


def test_qwen35_cache_binding_uses_standard_views_when_page_unaligned(monkeypatch):
    attention = SimpleNamespace(
        layer_num=3,
        k_scale=None,
        v_scale=None,
        impl=SimpleNamespace(use_flash_layout=True),
    )
    token_pool = _TokenPool()
    monkeypatch.setattr(bridge, "_native_attention_layers", lambda: [attention])
    monkeypatch.setattr(
        bridge,
        "_mha_pools",
        lambda _batch: (token_pool, SimpleNamespace()),
    )

    cache = bridge.bind_qwen35_cache_views(SimpleNamespace())["layer_3"]

    assert cache.k_cache.shape == (8, 2, 2, 1, 8)
    assert cache.v_cache.shape == (8, 2, 16, 1)
    assert (
        cache.k_cache.untyped_storage().data_ptr()
        == token_pool.k.untyped_storage().data_ptr()
    )
    assert (
        cache.v_cache.untyped_storage().data_ptr()
        == token_pool.v.untyped_storage().data_ptr()
    )
    assert attention.impl.use_flash_layout is False


def test_qwen35_cache_binding_uses_native_shuffle_views_when_page_aligned(
    monkeypatch,
):
    attention = SimpleNamespace(
        layer_num=3,
        k_scale=None,
        v_scale=None,
        impl=SimpleNamespace(use_flash_layout=True),
    )
    token_pool = SimpleNamespace(
        page_size=16,
        k=torch.empty(32, 1, 256, dtype=torch.uint8),
        v=torch.empty(32, 1, 256, dtype=torch.uint8),
    )
    token_pool.get_kv_buffer = lambda layer_id: (
        token_pool.k,
        token_pool.v,
    )
    monkeypatch.setattr(bridge, "_native_attention_layers", lambda: [attention])
    monkeypatch.setattr(
        bridge,
        "_mha_pools",
        lambda _batch: (token_pool, SimpleNamespace()),
    )

    cache = bridge.bind_qwen35_cache_views(SimpleNamespace())["layer_3"]

    assert cache.k_cache.shape == (2, 1, 16, 16, 16)
    assert cache.v_cache.shape == (2, 1, 1, 256, 16)
    assert cache.k_cache.data_ptr() == token_pool.k.data_ptr()
    assert cache.v_cache.data_ptr() == token_pool.v.data_ptr()
    assert (
        bridge._use_sglang_asm_cache_layout(
            cache.k_cache,
            cache.v_cache,
            use_triton_attn=True,
        )
        is True
    )
    assert attention.impl.use_flash_layout is False


def test_qwen35_cache_install_reuses_pool_stable_views(monkeypatch):
    import atom.utils.forward_context as atom_forward_context
    from atom.plugin.sglang.attention_backend import backend_resolver
    from atom.plugin.sglang.attention_backend.attention_gdn import (
        SGLangGDNForwardContext,
    )

    token_pools = [object()]
    linear_backend = object()
    mamba_map = {1: 0}
    mamba_pool = SimpleNamespace(mamba_map=mamba_map)
    bind_calls = []
    gdn_calls = []
    installed = []
    forward_context = SimpleNamespace(kv_cache_data=None)

    monkeypatch.setattr(
        bridge,
        "_mha_pools",
        lambda _batch: (token_pools[0], SimpleNamespace()),
    )

    def bind_mha(_batch):
        bind_calls.append(token_pools[0])
        return {"layer_0": object()}

    monkeypatch.setattr(bridge, "bind_qwen35_cache_views", bind_mha)
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_resolve_attn_backend",
        lambda _batch: object(),
    )
    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_linear_attn_backend",
        lambda _backend: linear_backend,
    )

    def build_gdn(_batch, _backend):
        gdn_calls.append(mamba_pool)
        return {"layer_1": object()}

    monkeypatch.setattr(
        SGLangGDNForwardContext,
        "_build_kv_cache_tensors",
        build_gdn,
    )
    monkeypatch.setattr(
        backend_resolver,
        "resolve_mamba_req_pool",
        lambda _batch, _backend: mamba_pool,
    )
    monkeypatch.setattr(
        atom_forward_context,
        "set_kv_cache_data",
        installed.append,
    )
    monkeypatch.setattr(
        atom_forward_context,
        "get_forward_context",
        lambda: forward_context,
    )

    owner = SimpleNamespace()
    batch = SimpleNamespace()
    bridge.install_qwen35_cache_views(batch, cache_owner=owner)
    bridge.install_qwen35_cache_views(batch, cache_owner=owner)

    assert len(bind_calls) == 1
    assert len(gdn_calls) == 1
    assert installed[0] is installed[1]
    assert forward_context.kv_cache_data is installed[1]

    token_pools[0] = object()
    bridge.install_qwen35_cache_views(batch, cache_owner=owner)

    assert len(bind_calls) == 2
    assert len(gdn_calls) == 2
    assert installed[2] is not installed[1]

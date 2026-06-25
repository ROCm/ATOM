"""Tests for GLM5 RTP MLA sparse topk consumption."""

import builtins
import importlib
import inspect
import sys
from types import SimpleNamespace

import torch

_SPARSE_BACKEND_MODULE = "atom.plugin.rtpllm.attention_backend.rtp_sparse_mla_backend"
_FORBIDDEN_CUDA_SPARSE_MODULES = (
    "flashmla_sparse",
    "flash_mla",
    "sparse_mla",
    "attention_mla_sparse",
)


def _guard_sparse_kernel_imports(monkeypatch):
    original_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if any(part in _FORBIDDEN_CUDA_SPARSE_MODULES for part in name.split(".")):
            raise AssertionError(
                f"GLM5 RTP sparse tests must not import CUDA sparse kernel: {name}"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _load_sparse_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    importlib.invalidate_caches()
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    return module.RTPSparseMlaBackend


def _forward_context_module():
    module = sys.modules.get("atom.utils.forward_context")
    if module is None:
        module = type(sys)("atom.utils.forward_context")
        module.get_forward_context = lambda: None
        sys.modules["atom.utils.forward_context"] = module
    return module


def test_rtp_sparse_attn_indexer_bridge_forwards_to_main_indexer(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    calls = []
    expected = torch.empty(1)

    def fake_sparse_attn_indexer(*args):
        calls.append(args)
        return expected

    fake_deepseek = type(sys)("atom.models.deepseek_v2")
    fake_deepseek.sparse_attn_indexer = fake_sparse_attn_indexer
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v2", fake_deepseek)

    tensor = torch.empty(1)
    output = module.rtp_sparse_attn_indexer(
        tensor,
        "indexer.prefix",
        tensor,
        tensor,
        tensor,
        tensor,
        128,
        None,
        2048,
        64,
        4096,
        1,
        tensor,
        tensor,
        tensor,
        1e-6,
        tensor,
        tensor,
        tensor,
        1.0,
        True,
        False,
    )

    assert output is expected
    assert len(calls) == 1
    assert calls[0][0] is tensor
    assert calls[0][1] == "indexer.prefix"
    assert calls[0][6:12] == (128, None, 2048, 64, 4096, 1)


def test_rtp_sparse_attn_indexer_uses_rtp_topk_path_when_context_exists(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = _forward_context_module()
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False, is_dummy_run=False, batch_size=1),
        attn_metadata=SimpleNamespace(max_seqlen_q=1),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )

    def _unexpected_call(*args, **kwargs):
        raise AssertionError("RTP context path must not call deepseek sparse indexer")

    fake_deepseek = type(sys)("atom.models.deepseek_v2")
    fake_deepseek.sparse_attn_indexer = _unexpected_call
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v2", fake_deepseek)

    expected = torch.empty(1)
    calls = []

    def _fake_topk_only(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(
        module, "_run_rtp_sparse_attn_indexer_topk_only", _fake_topk_only
    )
    tensor = torch.empty(1)

    output = module.rtp_sparse_attn_indexer(
        tensor,
        "indexer.prefix",
        tensor,
        tensor,
        tensor,
        tensor,
        128,
        None,
        2048,
        64,
        4096,
        1,
        torch.empty(1, 2048, dtype=torch.int32),
        tensor,
        tensor,
        1e-6,
        tensor,
        tensor,
        tensor,
        1.0,
        True,
        False,
    )

    assert output is expected
    assert len(calls) == 1
    assert calls[0][-2:] == (
        fake_forward_context.context,
        fake_forward_context.attn_metadata,
    )


def test_rtp_sparse_attn_indexer_fake_bridge_forwards_to_main_fake(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    calls = []
    expected = torch.empty(1)

    def fake_sparse_attn_indexer_fake(*args):
        calls.append(args)
        return expected

    fake_deepseek = type(sys)("atom.models.deepseek_v2")
    fake_deepseek.sparse_attn_indexer_fake = fake_sparse_attn_indexer_fake
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v2", fake_deepseek)

    tensor = torch.empty(1)
    output = module.rtp_sparse_attn_indexer_fake(
        tensor,
        "indexer.prefix",
        tensor,
        tensor,
        tensor,
        tensor,
        128,
        None,
        2048,
        64,
        4096,
        1,
        tensor,
        tensor,
        tensor,
        1e-6,
        tensor,
        tensor,
        tensor,
        1.0,
        True,
        False,
    )

    assert output is expected
    assert len(calls) == 1
    assert calls[0][0] is tensor
    assert calls[0][1] == "indexer.prefix"
    assert calls[0][6:12] == (128, None, 2048, 64, 4096, 1)


def test_rtp_sparse_attn_indexer_short_prefill_fills_causal_topk(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = _forward_context_module()
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=True, is_dummy_run=False),
        attn_metadata=SimpleNamespace(max_seqlen_k=4),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )

    def _unexpected_call(*args, **kwargs):
        raise AssertionError(
            "short prefill path should not call deepseek sparse_attn_indexer"
        )

    fake_deepseek = type(sys)("atom.models.deepseek_v2")
    fake_deepseek.sparse_attn_indexer = _unexpected_call
    monkeypatch.setitem(sys.modules, "atom.models.deepseek_v2", fake_deepseek)

    topk_buffer = torch.full((3, 8), -99, dtype=torch.int32)
    positions = torch.tensor([0, 1, 3], dtype=torch.int32)
    tensor = torch.empty(3, 2)
    weights = torch.randn(3, 4)
    out = module.rtp_sparse_attn_indexer(
        tensor,
        "indexer.prefix",
        tensor,
        tensor,
        tensor,
        weights,
        128,
        None,
        6,
        64,
        4096,
        3,
        topk_buffer,
        tensor,
        tensor,
        1e-6,
        positions,
        tensor,
        tensor,
        1.0,
        True,
        False,
    )

    assert out is weights
    assert topk_buffer[:3, :6].tolist() == [
        [0, -1, -1, -1, -1, -1],
        [0, 1, -1, -1, -1, -1],
        [0, 1, 2, 3, -1, -1],
    ]


class _FakeSparseImpl:
    def __init__(self, v_head_dim: int = 5):
        self.v_head_dim = v_head_dim
        self.calls = []

    def forward(
        self,
        q,
        compressed_kv,
        k_pe,
        kv_cache,
        layer_id,
        *,
        topk_indices,
        attn_metadata,
    ):
        self.calls.append(
            {
                "q": q,
                "compressed_kv": compressed_kv,
                "k_pe": k_pe,
                "kv_cache": kv_cache,
                "layer_id": layer_id,
                "topk_indices": topk_indices,
                "attn_metadata": attn_metadata,
            }
        )
        return q.new_full((q.shape[0], q.shape[1], self.v_head_dim), 7)


def _build_backend(backend_cls, sparse_impl):
    params = inspect.signature(backend_cls).parameters
    kwargs = {}

    if "sparse_impl" in params:
        kwargs["sparse_impl"] = sparse_impl
    else:
        raise AssertionError(
            "RTPSparseMlaBackend must accept an injected sparse implementation"
        )

    if "v_head_dim" in params:
        kwargs["v_head_dim"] = int(getattr(sparse_impl, "v_head_dim", 5))
    return backend_cls(**kwargs)


def _make_inputs():
    return (
        torch.randn(3, 2, 4),
        torch.randn(3, 8),
        torch.randn(3, 3),
        SimpleNamespace(name="kv-cache"),
        11,
    )


def test_sparse_backend_passes_topk_through_unchanged(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[4, 1], [3, 0], [2, 1]], dtype=torch.int32)

    output = backend.forward(
        q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk
    )

    assert output.shape == (3, 2, sparse_impl.v_head_dim)
    assert len(sparse_impl.calls) == 1
    assert sparse_impl.calls[0]["topk_indices"] is topk
    assert sparse_impl.calls[0]["topk_indices"].dtype == torch.int32
    assert sparse_impl.calls[0]["topk_indices"].shape == (3, 2)


def test_sparse_backend_prefill_without_topk_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    forward_context_mod = _forward_context_module()
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(
            plugin_metadata=SimpleNamespace(num_prefills=1, is_dummy_warmup=False)
        ),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()

    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    try:
        backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None)
    except module._SparseUnavailable as exc:
        assert "requires topk_indices" in str(exc)
    else:
        raise AssertionError("Expected missing prefill topk_indices to raise")
    assert sparse_impl.calls == []


def test_sparse_backend_decode_without_topk_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = _forward_context_module()
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=SimpleNamespace(
            plugin_metadata=SimpleNamespace(num_prefills=0, is_dummy_warmup=False)
        ),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()

    try:
        backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None)
    except module._SparseUnavailable as exc:
        assert "requires topk_indices" in str(exc)
    else:
        raise AssertionError("Expected missing decode topk_indices to raise")
    assert sparse_impl.calls == []


def test_sparse_backend_threads_kv_cache_and_layer_id_to_sparse_impl(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)

    call = sparse_impl.calls[0]
    assert call["q"] is q
    assert call["compressed_kv"] is compressed_kv
    assert call["k_pe"] is k_pe
    assert call["kv_cache"] is kv_cache
    assert call["layer_id"] == layer_id


def test_sparse_backend_pulls_attn_metadata_from_forward_context(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    forward_context_mod = _forward_context_module()

    attn_metadata = SimpleNamespace(block_table="block-table", seq_lens="seq-lens")
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=attn_metadata,
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)

    assert sparse_impl.calls[0]["attn_metadata"] is attn_metadata


def test_sparse_backend_prefill_missing_sparse_kernel_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = _forward_context_module()

    attn_metadata = SimpleNamespace(
        plugin_metadata=SimpleNamespace(num_prefills=1, is_dummy_warmup=False)
    )
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=attn_metadata,
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )

    class _MissingPrefillSparse:
        def forward(self, *args, **kwargs):
            raise module._SparseUnavailable("flash_mla_sparse_fwd unavailable")

    sparse_impl = _MissingPrefillSparse()
    backend = _build_backend(backend_cls, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    try:
        backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)
    except module._SparseUnavailable:
        pass
    else:
        raise AssertionError(
            "prefill sparse unavailability must not fall back to dense"
        )


def test_sparse_backend_decode_missing_sparse_kernel_still_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = _forward_context_module()

    attn_metadata = SimpleNamespace(
        plugin_metadata=SimpleNamespace(num_prefills=0, is_dummy_warmup=False)
    )
    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=False),
        attn_metadata=attn_metadata,
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )

    class _MissingDecodeSparse:
        def forward(self, *args, **kwargs):
            raise module._SparseUnavailable("flash_mla_sparse_fwd unavailable")

    backend = _build_backend(backend_cls, _MissingDecodeSparse())
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    try:
        backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)
    except module._SparseUnavailable:
        pass
    else:
        raise AssertionError("decode sparse unavailability must not fall back to dense")


def test_sparse_backend_forward_signature_matches_dense_boundary(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)

    signature = inspect.signature(backend_cls.forward)
    params = signature.parameters

    assert list(params) == [
        "self",
        "q",
        "compressed_kv",
        "k_pe",
        "kv_cache",
        "layer_id",
        "topk_indices",
        "positions",
    ]
    assert params["topk_indices"].default is None


def test_sparse_backend_converts_request_local_topk_to_global_slots(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    convert = module._RealSparseMlaImpl._convert_topk_to_global
    plugin_metadata = SimpleNamespace(
        block_table=torch.tensor([[7, 8], [20, 21]], dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(plugin_metadata=plugin_metadata)
    topk = torch.tensor(
        [
            [0, 1029, -1],
            [1024, 2048, 5],
        ],
        dtype=torch.int32,
    )

    del backend_cls
    global_topk = convert(
        topk_indices=topk,
        attn_metadata=attn_metadata,
        block_size=1024,
    )

    assert global_topk.cpu().tolist() == [
        [7 * 1024, 8 * 1024 + 5, 0],
        [21 * 1024, 0, 20 * 1024 + 5],
    ]


def test_real_sparse_decode_uses_atom_aiter_metadata(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    calls = {}

    aiter = type(sys)("aiter")
    aiter.dtypes = SimpleNamespace(
        fp8=torch.float8_e4m3fnuz,
        d_dtypes={"fp16": torch.float16, "bf16": torch.bfloat16},
    )
    monkeypatch.setitem(sys.modules, "aiter", aiter)

    def fake_metadata_info(*args, **kwargs):
        calls["metadata_info"] = (args, kwargs)
        return (
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
        )

    def fake_metadata_v1(*args, **kwargs):
        calls["metadata_v1"] = (args, kwargs)

    monkeypatch.setattr(
        aiter, "get_mla_metadata_info_v1", fake_metadata_info, raising=False
    )
    monkeypatch.setattr(aiter, "get_mla_metadata_v1", fake_metadata_v1, raising=False)

    fake_mla = type(sys)("aiter.mla")

    def fake_mla_decode_fwd(
        q,
        kv,
        output,
        qo_indptr,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
        *args,
        **kwargs,
    ):
        calls["mla_decode_fwd"] = {
            "q": q,
            "kv": kv,
            "output": output,
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "args": args,
            "kwargs": kwargs,
        }
        output.fill_(3)

    fake_mla.mla_decode_fwd = fake_mla_decode_fwd
    monkeypatch.setitem(sys.modules, "aiter.mla", fake_mla)

    fake_sparse_helpers = type(sys)("atom.plugin.attention_mla_sparse")

    def fake_generate_sparse_seqlen(
        query_lens, seq_lens, query_start_loc, topk, num_tokens, max_query_len
    ):
        return torch.tensor([3, 2], dtype=torch.int32, device=query_lens.device)

    def fake_convert(
        req_id,
        block_table,
        token_indices,
        cu_seqlens,
        out,
        BLOCK_SIZE=1,
        NUM_TOPK_TOKENS=0,
        BLOCK_N=128,
    ):
        out[:5] = torch.tensor([0, 1, 2, 4, 5], dtype=torch.int32, device=out.device)

    fake_sparse_helpers.generate_sparse_seqlen_triton = fake_generate_sparse_seqlen
    fake_sparse_helpers.triton_convert_req_index_to_global_index = fake_convert
    monkeypatch.setitem(
        sys.modules,
        "atom.plugin.attention_mla_sparse",
        fake_sparse_helpers,
    )

    impl = module._RealSparseMlaImpl(
        mla_modules=SimpleNamespace(
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            num_heads=2,
            rotary_emb=None,
            kv_b_proj=SimpleNamespace(weight=torch.empty(0)),
        ),
        v_head_dim=3,
    )
    q_latent = torch.randn(2, 2, 5, dtype=torch.bfloat16)
    kv_cache = torch.empty(8, 1, 5, dtype=torch.uint8)
    topk = torch.tensor([[0, 1, 2], [0, 1, -1]], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        plugin_metadata=SimpleNamespace(
            query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
            seq_lens=torch.tensor([3, 2], dtype=torch.int32),
            req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
            block_table=torch.tensor([[0], [1]], dtype=torch.int32),
        )
    )

    output = impl._run_aiter_sparse_decode(
        q_latent=q_latent,
        kv_cache_base=kv_cache,
        topk_indices=topk,
        attn_metadata=attn_metadata,
        block_size=4,
    )

    assert output.shape == (2, 2, 4)
    assert output.dtype == torch.bfloat16
    assert torch.all(output == 3)
    decode_call = calls["mla_decode_fwd"]
    assert decode_call["q"].shape == (2, 16, 5)
    assert decode_call["q"].dtype == aiter.dtypes.fp8
    assert decode_call["output"].shape == (2, 16, 4)
    assert decode_call["output"].dtype == torch.bfloat16
    assert decode_call["paged_kv_indptr"].tolist() == [0, 3, 5]
    assert decode_call["paged_kv_indices"][:5].tolist() == [0, 1, 2, 4, 5]
    assert decode_call["kwargs"]["page_size"] == 1
    assert decode_call["kwargs"]["q_scale"] is not None
    assert decode_call["kwargs"]["kv_scale"] is not None
    assert decode_call["kwargs"]["work_meta_data"] is not None
    assert decode_call["kwargs"]["reduce_final_map"] is not None


def test_real_sparse_cache_dtype_uses_aiter_fp8_layout():
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    impl = module._RealSparseMlaImpl(
        mla_modules=SimpleNamespace(
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            num_heads=2,
            rotary_emb=None,
            kv_b_proj=SimpleNamespace(weight=torch.empty(0)),
        ),
        v_head_dim=128,
    )

    assert impl._cache_dtype_name(torch.empty(1, 576, dtype=torch.uint8)) == "fp8"
    assert impl._cache_dtype_name(torch.empty(1, 576, dtype=torch.bfloat16)) == "auto"


def test_sparse_index_converter_resolves_current_refactored_path(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    old_module_name = "atom.plugin.attention_mla_sparse"
    new_module_name = "atom.plugin.vllm.attention.layer_sparse_mla"
    monkeypatch.delitem(sys.modules, old_module_name, raising=False)

    fake_new_helpers = type(sys)(new_module_name)

    def fake_convert():
        return None

    fake_new_helpers.triton_convert_req_index_to_global_index = fake_convert
    monkeypatch.setitem(sys.modules, new_module_name, fake_new_helpers)

    assert module._resolve_plugin_sparse_index_converter() is fake_convert


def test_real_sparse_eager_metadata_workspace_skips_refill(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    metadata_calls = []

    fake_aiter = type(sys)("aiter")
    fake_aiter.dtypes = SimpleNamespace(d_dtypes={"bf16": "bf16", "fp16": "fp16"})

    def fake_metadata_info(*args, **kwargs):
        return (
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
            (4, torch.int32),
        )

    def fake_metadata_v1(*args, **kwargs):
        metadata_calls.append((args, kwargs))

    fake_aiter.get_mla_metadata_info_v1 = fake_metadata_info
    fake_aiter.get_mla_metadata_v1 = fake_metadata_v1
    monkeypatch.setitem(sys.modules, "aiter", fake_aiter)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: False, raising=False
    )

    fake_sparse_helpers = type(sys)("atom.plugin.attention_mla_sparse")

    def fake_convert(
        req_id,
        block_table,
        token_indices,
        cu_seqlens,
        out,
        BLOCK_SIZE=1,
        NUM_TOPK_TOKENS=0,
        BLOCK_N=128,
    ):
        del req_id, block_table, token_indices, BLOCK_SIZE, NUM_TOPK_TOKENS, BLOCK_N
        out[: int(cu_seqlens[-1].item())].zero_()

    fake_sparse_helpers.triton_convert_req_index_to_global_index = fake_convert
    monkeypatch.setitem(
        sys.modules,
        "atom.plugin.attention_mla_sparse",
        fake_sparse_helpers,
    )

    impl = module._RealSparseMlaImpl(
        mla_modules=SimpleNamespace(
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            num_heads=2,
            rotary_emb=None,
            kv_b_proj=SimpleNamespace(weight=torch.empty(0)),
        ),
        v_head_dim=3,
    )
    q_latent = torch.randn(2, 2, 5)
    kv_cache = torch.randn(8, 1, 5)
    topk = torch.tensor([[0, 1, 2], [0, 1, -1]], dtype=torch.int32)
    plugin_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1], dtype=torch.int32),
        block_table=torch.tensor([[0], [1]], dtype=torch.int32),
    )
    attn_metadata = SimpleNamespace(plugin_metadata=plugin_metadata)

    first = impl._build_atom_sparse_metadata(
        q_latent=q_latent,
        kv_cache_base=kv_cache,
        topk_indices=topk,
        attn_metadata=attn_metadata,
        block_size=4,
    )
    second = impl._build_atom_sparse_metadata(
        q_latent=q_latent,
        kv_cache_base=kv_cache,
        topk_indices=topk,
        attn_metadata=attn_metadata,
        block_size=4,
    )

    assert len(metadata_calls) == 1
    assert second.work_meta_data is first.work_meta_data
    assert plugin_metadata._rtp_sparse_eager_meta_workspace["metadata_ready"] is True


def test_real_sparse_decode_rejects_oob_paged_kv_indices(monkeypatch):
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    decode_called = {"value": False}
    monkeypatch.setenv("ATOM_RTP_GLM5_SPARSE_VALIDATE", "1")

    fake_mla = type(sys)("aiter.mla")

    def fake_mla_decode_fwd(*args, **kwargs):
        decode_called["value"] = True

    fake_mla.mla_decode_fwd = fake_mla_decode_fwd
    monkeypatch.setitem(sys.modules, "aiter.mla", fake_mla)

    impl = module._RealSparseMlaImpl(
        mla_modules=SimpleNamespace(
            kv_lora_rank=4,
            qk_nope_head_dim=2,
            qk_rope_head_dim=1,
            num_heads=2,
            rotary_emb=None,
            kv_b_proj=SimpleNamespace(weight=torch.empty(0)),
        ),
        v_head_dim=3,
    )
    q_latent = torch.randn(2, 2, 5)
    kv_cache = torch.randn(8, 1, 5)
    topk = torch.tensor([[0, 1, 2], [0, 1, -1]], dtype=torch.int32)
    attn_metadata = SimpleNamespace(plugin_metadata=SimpleNamespace())

    oob_meta = module._AtomSparseMetadata(
        qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 3, 6], dtype=torch.int32),
        # kv_buffer has 8 slots, index=8 is out of range.
        paged_kv_indices=torch.tensor([0, 1, 2, 3, 4, 8], dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(2, dtype=torch.int32),
        work_meta_data=torch.zeros(1, dtype=torch.int32),
        work_indptr=torch.zeros(1, dtype=torch.int32),
        work_info_set=torch.zeros(1, dtype=torch.int32),
        reduce_indptr=torch.zeros(1, dtype=torch.int32),
        reduce_final_map=torch.zeros(1, dtype=torch.int32),
        reduce_partial_map=torch.zeros(1, dtype=torch.int32),
        padded_num_heads=2,
        head_repeat_factor=1,
        page_size=1,
    )
    monkeypatch.setattr(impl, "_build_atom_sparse_metadata", lambda **kwargs: oob_meta)

    try:
        impl._run_aiter_sparse_decode(
            q_latent=q_latent,
            kv_cache_base=kv_cache,
            topk_indices=topk,
            attn_metadata=attn_metadata,
            block_size=4,
        )
    except module._SparseUnavailable as exc:
        assert "out-of-range paged_kv_indices" in str(exc)
    else:
        raise AssertionError(
            "Expected OOB paged_kv_indices to raise _SparseUnavailable"
        )
    assert decode_called["value"] is False


def _load_rtp_mla_attention():
    module = importlib.import_module(
        "atom.plugin.rtpllm.attention_backend.rtp_mla_attention"
    )
    return module.RTPMLAAttention


class _FakeSparseBackend:
    def __init__(self, v_head_dim: int):
        self.v_head_dim = v_head_dim
        self.calls = []

    def forward(self, q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None):
        self.calls.append(
            {
                "q": q,
                "compressed_kv": compressed_kv,
                "k_pe": k_pe,
                "kv_cache": kv_cache,
                "layer_id": layer_id,
                "topk_indices": topk_indices,
            }
        )
        return q.new_empty((q.shape[0], q.shape[1], self.v_head_dim))


class _FakeIndexer:
    def __init__(self, topk_values):
        self.calls = []
        self.index_topk = topk_values.shape[1]
        self.topk_indices_buffer = torch.full(
            (topk_values.shape[0], topk_values.shape[1] + 2),
            -1,
            dtype=torch.int32,
        )
        self.topk_indices_buffer[: topk_values.shape[0], : topk_values.shape[1]].copy_(
            topk_values
        )
        self.weights = torch.full(topk_values.shape, 99.0, dtype=torch.float32)

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.weights


class _FakeQProj:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def __call__(self, query, q_scale=None):
        self.calls.append((query, q_scale))
        return self.output


class _FakeOProj:
    def __init__(self):
        self.calls = []

    def __call__(self, tensor):
        self.calls.append(tensor)
        return tensor


def _make_attention(topk_values):
    token_count = topk_values.shape[0]
    num_heads = 2
    qk_head_dim = 4
    v_head_dim = 3
    projected_q = torch.arange(
        token_count * num_heads * qk_head_dim, dtype=torch.float32
    ).reshape(token_count, num_heads * qk_head_dim)
    backend = _FakeSparseBackend(v_head_dim=v_head_dim)
    indexer = _FakeIndexer(topk_values)
    modules = SimpleNamespace(
        q_proj=_FakeQProj(projected_q),
        o_proj=_FakeOProj(),
        kv_b_proj=object(),
        indexer=indexer,
        v_head_dim=v_head_dim,
        qk_head_dim=qk_head_dim,
        num_heads=num_heads,
        num_local_heads=num_heads,
        index_topk=topk_values.shape[1],
    )
    attention = _load_rtp_mla_attention()(
        mla_modules=modules,
        sparse_backend=backend,
        layer_num=7,
        kv_cache="kv-cache",
    )
    return attention, modules, backend


def _run_attention(attention, token_count: int):
    query = torch.empty(token_count, 6)
    compressed_kv = torch.empty(token_count, 8)
    k_rope = torch.empty(token_count, 3)
    positions = torch.arange(token_count, dtype=torch.int32)
    return attention.forward(
        query,
        compressed_kv,
        k_rope,
        positions=positions,
    )


def _patch_forward_context(monkeypatch, *, is_dummy_run, is_prefill, max_seqlen_k):
    forward_context_mod = sys.modules["atom.utils.forward_context"]

    fake_forward_context = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=is_dummy_run, is_prefill=is_prefill),
        attn_metadata=SimpleNamespace(max_seqlen_k=max_seqlen_k),
    )
    monkeypatch.setattr(
        forward_context_mod,
        "get_forward_context",
        lambda: fake_forward_context,
        raising=False,
    )


def test_constructor_injects_indexer_and_topk_indices_buffer_owner_path():
    topk_buffer = torch.tensor([[4, 1, 3, 0]], dtype=torch.int32)
    indexer = SimpleNamespace(topk_indices_buffer=topk_buffer, index_topk=4)
    modules = SimpleNamespace(
        q_proj=object(),
        o_proj=object(),
        kv_b_proj=object(),
        indexer=indexer,
        v_head_dim=3,
    )
    attention = _load_rtp_mla_attention()(mla_modules=modules)

    assert attention.indexer is indexer
    assert attention.topk_indices_buffer is topk_buffer


def test_constructor_swaps_indexer_to_rtp_sparse_indexer_op(monkeypatch):
    default_op = object()
    rtp_op = object()
    monkeypatch.setattr(
        torch.ops.aiter, "rtp_sparse_attn_indexer", rtp_op, raising=False
    )
    topk_buffer = torch.tensor([[4, 1, 3, 0]], dtype=torch.int32)
    indexer = SimpleNamespace(
        topk_indices_buffer=topk_buffer,
        index_topk=4,
        sparse_attn_indexer_impl=default_op,
    )
    modules = SimpleNamespace(
        q_proj=object(),
        o_proj=object(),
        kv_b_proj=object(),
        indexer=indexer,
        v_head_dim=3,
    )

    attention = _load_rtp_mla_attention()(mla_modules=modules, sparse_backend=object())

    assert attention.indexer is indexer
    assert indexer.sparse_attn_indexer_impl is rtp_op


def test_constructor_patches_indexer_forward_to_own_topk_buffer(monkeypatch):
    default_op = object()
    rtp_op = object()
    monkeypatch.setattr(
        torch.ops.aiter, "rtp_sparse_attn_indexer", rtp_op, raising=False
    )

    class _ForwardIndexer:
        def __init__(self):
            self.topk_tokens = 4
            self.sparse_attn_indexer_impl = default_op
            self.sparse_kv_indices_buffer = torch.empty(0, dtype=torch.int32)
            self.seen_sparse_buffer = None

        def forward(self, hidden_states):
            self.seen_sparse_buffer = self.sparse_kv_indices_buffer
            return hidden_states

    indexer = _ForwardIndexer()
    modules = SimpleNamespace(
        q_proj=object(),
        o_proj=object(),
        kv_b_proj=object(),
        indexer=indexer,
        v_head_dim=3,
    )

    _load_rtp_mla_attention()(mla_modules=modules, sparse_backend=object())
    hidden_states = torch.empty(2, 8)
    indexer.forward(hidden_states)

    assert indexer.sparse_attn_indexer_impl is rtp_op
    assert indexer.topk_indices_buffer.shape == (2, 4)
    assert indexer.topk_indices_buffer.dtype == torch.int32
    assert indexer.seen_sparse_buffer is indexer.topk_indices_buffer


def test_indexer_buffer_topk_is_passed_to_sparse_backend_when_emit_allowed(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    topk_values = torch.tensor([[4, 1, 3, 0], [2, 0, 1, 3]], dtype=torch.int32)
    attention, modules, backend = _make_attention(topk_values)

    _run_attention(attention, token_count=topk_values.shape[0])

    assert modules.indexer.calls == []
    topk_indices = backend.calls[0]["topk_indices"]
    assert topk_indices is not None
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape == topk_values.shape
    assert torch.equal(topk_indices, topk_values)
    assert topk_indices is not modules.indexer.weights
    assert not torch.equal(topk_indices.to(torch.float32), modules.indexer.weights)


def test_dummy_run_does_not_emit_topk_to_sparse_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    _patch_forward_context(
        monkeypatch,
        is_dummy_run=True,
        is_prefill=False,
        max_seqlen_k=4096,
    )
    topk_values = torch.tensor([[4, 1, 3, 0], [2, 0, 1, 3]], dtype=torch.int32)
    attention, modules, backend = _make_attention(topk_values)

    _run_attention(attention, token_count=topk_values.shape[0])

    assert modules.indexer.calls == []
    assert backend.calls[0]["topk_indices"] is None


def test_short_prefill_emits_topk_to_sparse_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    _patch_forward_context(
        monkeypatch,
        is_dummy_run=False,
        is_prefill=True,
        max_seqlen_k=4,
    )
    topk_values = torch.tensor([[4, 1, 3, 0], [2, 0, 1, 3]], dtype=torch.int32)
    attention, modules, backend = _make_attention(topk_values)

    _run_attention(attention, token_count=topk_values.shape[0])

    assert modules.indexer.calls == []
    topk_indices = backend.calls[0]["topk_indices"]
    assert topk_indices is not None
    assert torch.equal(topk_indices, topk_values)


def test_prefill_within_topk_buffer_padding_still_emits_topk(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    _patch_forward_context(
        monkeypatch,
        is_dummy_run=False,
        is_prefill=True,
        max_seqlen_k=5,
    )
    topk_values = torch.tensor([[4, 1, 3, 0], [2, 0, 1, 3]], dtype=torch.int32)
    attention, modules, backend = _make_attention(topk_values)

    _run_attention(attention, token_count=topk_values.shape[0])

    assert modules.indexer.index_topk == 4
    assert modules.indexer.topk_indices_buffer.shape[1] == 6
    assert modules.indexer.calls == []
    topk_indices = backend.calls[0]["topk_indices"]
    assert topk_indices is not None
    assert torch.equal(topk_indices, topk_values)

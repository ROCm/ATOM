"""Contract-executable tests for GLM5 RTP MLA M2 sparse topk consumption."""

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
                f"M2 sparse contract must not import CUDA sparse kernel: {name}"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _load_sparse_backend(monkeypatch):
    _guard_sparse_kernel_imports(monkeypatch)
    importlib.invalidate_caches()
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    return module.RTPSparseMlaBackend


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


class _FakeDenseBackend:
    def __init__(self, v_head_dim: int = 5):
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
        return q.new_full((q.shape[0], q.shape[1], self.v_head_dim), -1)


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


def _build_backend(backend_cls, dense_backend, sparse_impl):
    params = inspect.signature(backend_cls).parameters
    kwargs = {}
    if "dense_backend" not in params:
        raise AssertionError(
            "RTPSparseMlaBackend must accept dense_backend= for dense fallback"
        )
    kwargs["dense_backend"] = dense_backend

    if "sparse_impl" in params:
        kwargs["sparse_impl"] = sparse_impl
    else:
        raise AssertionError(
            "RTPSparseMlaBackend must accept a mock sparse impl injection"
        )

    if "v_head_dim" in params:
        kwargs["v_head_dim"] = dense_backend.v_head_dim
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
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[4, 1], [3, 0], [2, 1]], dtype=torch.int32)

    output = backend.forward(
        q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk
    )

    assert output.shape == (3, 2, sparse_impl.v_head_dim)
    assert dense_backend.calls == []
    assert len(sparse_impl.calls) == 1
    assert sparse_impl.calls[0]["topk_indices"] is topk
    assert sparse_impl.calls[0]["topk_indices"].dtype == torch.int32
    assert sparse_impl.calls[0]["topk_indices"].shape == (3, 2)


def test_sparse_backend_falls_back_to_dense_when_topk_is_none(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    forward_context_mod = sys.modules["atom.utils.forward_context"]
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
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()

    output = backend.forward(
        q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None
    )

    assert output.shape == (3, 2, dense_backend.v_head_dim)
    assert len(dense_backend.calls) == 1
    assert sparse_impl.calls == []
    assert dense_backend.calls[0]["q"] is q
    assert dense_backend.calls[0]["compressed_kv"] is compressed_kv
    assert dense_backend.calls[0]["k_pe"] is k_pe
    assert dense_backend.calls[0]["kv_cache"] is kv_cache
    assert dense_backend.calls[0]["layer_id"] == layer_id
    assert dense_backend.calls[0]["topk_indices"] is None


def test_sparse_backend_decode_without_topk_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = sys.modules["atom.utils.forward_context"]
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
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()

    try:
        backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=None)
    except module._SparseUnavailable as exc:
        assert "decode requires topk_indices" in str(exc)
    else:
        raise AssertionError("Expected missing decode topk_indices to raise")
    assert dense_backend.calls == []
    assert sparse_impl.calls == []


def test_sparse_backend_threads_kv_cache_and_layer_id_to_sparse_impl(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
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
    forward_context_mod = sys.modules["atom.utils.forward_context"]

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
    dense_backend = _FakeDenseBackend()
    sparse_impl = _FakeSparseImpl()
    backend = _build_backend(backend_cls, dense_backend, sparse_impl)
    q, compressed_kv, k_pe, kv_cache, layer_id = _make_inputs()
    topk = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.int32)

    backend.forward(q, compressed_kv, k_pe, kv_cache, layer_id, topk_indices=topk)

    assert sparse_impl.calls[0]["attn_metadata"] is attn_metadata


def test_sparse_backend_prefill_missing_sparse_kernel_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = sys.modules["atom.utils.forward_context"]

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

    dense_backend = _FakeDenseBackend()
    backend = _build_backend(backend_cls, dense_backend, _MissingPrefillSparse())
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

    assert len(dense_backend.calls) == 0


def test_sparse_backend_decode_missing_sparse_kernel_still_raises(monkeypatch):
    backend_cls = _load_sparse_backend(monkeypatch)
    module = importlib.import_module(_SPARSE_BACKEND_MODULE)
    forward_context_mod = sys.modules["atom.utils.forward_context"]

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

    backend = _build_backend(backend_cls, _FakeDenseBackend(), _MissingDecodeSparse())
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

    import aiter

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
    q_latent = torch.randn(2, 2, 5)
    kv_cache = torch.randn(8, 1, 5)
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
    assert torch.all(output == 3)
    decode_call = calls["mla_decode_fwd"]
    assert decode_call["q"].shape == (2, 16, 5)
    assert decode_call["output"].shape == (2, 16, 4)
    assert decode_call["paged_kv_indptr"].tolist() == [0, 3, 5]
    assert decode_call["paged_kv_indices"][:5].tolist() == [0, 1, 2, 4, 5]
    assert decode_call["kwargs"]["page_size"] == 1
    assert decode_call["kwargs"]["work_meta_data"] is not None
    assert decode_call["kwargs"]["reduce_final_map"] is not None


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

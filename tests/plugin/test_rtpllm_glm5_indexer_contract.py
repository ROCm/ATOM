"""Contract-executable tests for GLM5 RTP MLA M1.5 indexer behavior."""

import builtins
import sys
from types import SimpleNamespace

import torch

from atom.plugin.rtpllm.attention_backend.rtp_mla_attention import RTPMLAAttention

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
                f"M1.5 tests must not import sparse MLA kernels: {name}"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


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
    attention = RTPMLAAttention(
        mla_modules=modules,
        sparse_backend=backend,
        layer_num=7,
        kv_cache="kv-cache",
    )
    return attention, modules, backend


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
    attention = RTPMLAAttention(mla_modules=modules)

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

    attention = RTPMLAAttention(mla_modules=modules, sparse_backend=object())

    assert attention.indexer is indexer
    assert indexer.sparse_attn_indexer_impl is rtp_op


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

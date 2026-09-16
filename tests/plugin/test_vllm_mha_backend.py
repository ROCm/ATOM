from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from atom.plugin.vllm.attention.backend import (
    AiterMhaBackendForVllm,
    AiterMhaFlexibleBlockBackendForVllm,
    AiterMlaBackendForVllm,
    AtomAiterMLAPrefillBackend,
    GDNAttentionBackend,
    MiniMaxM3SparseAttentionBackend,
)
from atom.plugin.vllm.attention.layer_mha import (
    _mha_backend_for_layer,
)


def test_target_mha_keeps_physical_kernel_block_size_16():
    assert AiterMhaBackendForVllm.get_supported_kernel_block_sizes() == [16]
    assert (
        _mha_backend_for_layer(
            47,
            SimpleNamespace(num_hidden_layers=48),
        )
        is AiterMhaBackendForVllm
    )


def test_draft_mha_accepts_hybrid_logical_page_size():
    assert (
        _mha_backend_for_layer(
            48,
            SimpleNamespace(num_hidden_layers=48),
        )
        is AiterMhaFlexibleBlockBackendForVllm
    )
    assert AiterMhaFlexibleBlockBackendForVllm.get_supported_kernel_block_sizes() != [
        16
    ]


@pytest.mark.parametrize(
    "backend",
    [
        AiterMhaBackendForVllm,
        AiterMlaBackendForVllm,
        MiniMaxM3SparseAttentionBackend,
        GDNAttentionBackend,
    ],
)
def test_duck_typed_backends_keep_vllm_028_kv_spec(backend):
    spec = object()

    assert backend.customize_spec(spec) is spec
    assert not backend.supports_device_cpu_query_lens_mismatch()


def test_mla_prefill_backend_accepts_vllm_028_context_chunk_contract():
    seen = {}
    expected = (object(), object())

    class Layer:
        def _flash_attn_varlen_diff_headdims(self, **kwargs):
            seen.update(kwargs)
            return expected

    backend = object.__new__(AtomAiterMLAPrefillBackend)
    backend._layer = Layer()
    backend.scale = 0.5
    chunk = SimpleNamespace(
        query_start_loc=object(),
        cu_seq_lens=object(),
        max_query_len=3,
        max_seq_len=17,
    )
    q, k, v = object(), object(), object()

    assert backend.run_prefill_context_chunk(chunk, q, k, v) is expected
    assert seen == {
        "q": q,
        "k": k,
        "v": v,
        "cu_seqlens_q": chunk.query_start_loc,
        "cu_seqlens_k": chunk.cu_seq_lens,
        "max_seqlen_q": 3,
        "max_seqlen_k": 17,
        "softmax_scale": 0.5,
        "causal": False,
        "return_softmax_lse": True,
    }

    with pytest.raises(NotImplementedError, match="output buffer"):
        backend.run_prefill_context_chunk(chunk, q, k, v, out=object())

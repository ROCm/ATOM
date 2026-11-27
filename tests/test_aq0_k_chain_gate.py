"""aq0's ATOM gate (CPU): the first layer's QKV goes onto AQ in the tuned GEMM's K order only when the switch is on,
aiter's ``flydsl_aq1`` takes ``k_chain``, and the step qualifies for AQ (``_aq1_decode``). Every other case returns
None, so the caller runs the QKV linear on the same MXFP8 input: production's quantize + GEMM."""

import sys
import types

import pytest
import torch

am = pytest.importorskip("atom.model_ops.attention_mha", exc_type=ImportError)
pytest.importorskip("atom.models.minimax_m3.model", exc_type=ImportError)

H, NH, HD, NKV, IQ, IHD = 6144, 16, 128, 1, 512, 128


def _impl(aq1):
    linear = types.SimpleNamespace(
        weight=torch.zeros(NH * HD + 2 * NKV * HD + IQ + IHD, 4, dtype=torch.float8_e4m3fn),
        native_a8_group_rows=1,
    )
    linear.weight.is_shuffled = True
    calls = []
    impl = types.SimpleNamespace(
        fused_qkv_linear=(linear,), kv_nvfp4=True, index_scale=object(), skip_index_topk=False,
        num_heads=NH, head_dim=HD, num_kv_heads=NKV, index_q_size=IQ, index_head_dim=IHD, num_idx_heads=4,
        rotary_emb=types.SimpleNamespace(rotary_dim=64), qkv_production_k_order=True,
        _aq1_decode=lambda n: calls.append(n) or aq1,
    )
    return impl, calls


@pytest.fixture
def chain_ok(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250",
        types.SimpleNamespace(mxfp8_qkv_chain_supported=lambda *a, **k: True,
                              mxfp8_qkv_norm_rope_nvfp4_insert=None),
    )
    monkeypatch.setattr(am, "AQ_K_CHAIN", True)


def _call(impl):
    x = torch.zeros(8, H, dtype=torch.float8_e4m3fn)
    xs = torch.zeros(8, H // 32, dtype=torch.uint8)
    return am.SparseMHAPagedAttentionImpl._fused_qkv_chain(impl, x, xs, None, None, None, None, None, None, None)


def test_switch_off_returns_to_the_qkv_linear(chain_ok, monkeypatch):
    monkeypatch.setattr(am, "AQ_K_CHAIN", False)
    impl, calls = _impl(aq1=("md", 500000))
    assert _call(impl) is None and calls == []  # declined before AQ is even asked


def test_a_step_aq_does_not_take_returns_to_the_qkv_linear(chain_ok):
    impl, calls = _impl(aq1=None)  # e.g. a step of more than one request
    assert _call(impl) is None and calls == [8]


def test_an_aiter_without_k_chain_returns_to_the_qkv_linear(chain_ok, monkeypatch):
    am._aq_takes_k_chain.cache_clear()
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.mono.gfx1250.aq", types.SimpleNamespace(flydsl_aq1=lambda q, *, exchange=None: q))
    impl, calls = _impl(aq1=("md", 500000))
    try:
        assert not am._aq_takes_k_chain()
        assert _call(impl) is None and calls == [8]
    finally:
        am._aq_takes_k_chain.cache_clear()


def test_k_chain_detected_from_the_signature(monkeypatch):
    am._aq_takes_k_chain.cache_clear()
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.mono.gfx1250.aq",
                        types.SimpleNamespace(flydsl_aq1=lambda q, *, exchange=None, k_chain=False: q))
    try:
        assert am._aq_takes_k_chain()
    finally:
        am._aq_takes_k_chain.cache_clear()

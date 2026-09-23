"""MiMo MTP preserves its attention contract with either SGLang KV pool."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("sglang")
pytest.importorskip("aiter")

from atom.plugin.sglang.attention_backend.full_attention.full_attention_backend import (
    ATOMAttnBackendForSgl,
)


@pytest.mark.parametrize("method", ["forward_extend", "forward_decode"])
@pytest.mark.parametrize("hybrid_pool", [False, True])
def test_mimo_mtp_preserves_learned_sinks(monkeypatch, method, hybrid_pool):
    backend = object.__new__(ATOMAttnBackendForSgl)
    backend._is_mimo_mtp = True
    backend.use_sliding_window_kv_pool = hybrid_pool
    layer = SimpleNamespace(sinks=torch.tensor([2.0, -1.0], dtype=torch.bfloat16))
    observed = {}

    def native_call(self, q, k, v, received_layer, batch, **kwargs):
        assert received_layer is layer
        observed.update(kwargs)
        return "native-draft"

    monkeypatch.setattr(ATOMAttnBackendForSgl.__bases__[0], method, native_call)
    result = getattr(backend, method)(
        None, None, None, layer, SimpleNamespace(), save_kv_cache=False
    )
    assert result == "native-draft"
    assert observed["save_kv_cache"] is False
    assert observed["sinks"].dtype == torch.float32
    torch.testing.assert_close(observed["sinks"], layer.sinks.float())


@pytest.mark.parametrize("method", ["forward_extend", "forward_decode"])
def test_mimo_mtp_keeps_explicit_sink_argument(monkeypatch, method):
    backend = object.__new__(ATOMAttnBackendForSgl)
    backend._is_mimo_mtp = True
    layer = SimpleNamespace(sinks=torch.tensor([2.0]))
    explicit = torch.tensor([3.0])

    def native_call(self, *args, **kwargs):
        return kwargs["sinks"]

    monkeypatch.setattr(ATOMAttnBackendForSgl.__bases__[0], method, native_call)
    assert (
        getattr(backend, method)(
            None, None, None, layer, SimpleNamespace(), sinks=explicit
        )
        is explicit
    )


@pytest.mark.parametrize(
    "method", ["init_forward_metadata", "init_forward_metadata_out_graph"]
)
def test_mimo_mtp_ordinary_pool_uses_native_metadata(monkeypatch, method):
    backend = object.__new__(ATOMAttnBackendForSgl)
    backend._is_mimo_mtp = True
    backend._is_mimo_v2_family = True
    backend.use_sliding_window_kv_pool = False
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(is_target_verify=lambda: False)
    )

    def native_call(self, received_batch, **kwargs):
        assert received_batch is batch
        return "native-metadata"

    monkeypatch.setattr(ATOMAttnBackendForSgl.__bases__[0], method, native_call)
    assert getattr(backend, method)(batch) == "native-metadata"

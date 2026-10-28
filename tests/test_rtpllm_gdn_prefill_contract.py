"""CPU contracts for generic hooks used by RTP prefix checkpoints."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_rtp_native_snapshot_keeps_shared_prefill_operations(monkeypatch):
    pytest.importorskip("triton", reason="GDN prefill requires Triton")
    from atom.model_ops.fla_ops import gdn_flydsl
    from atom.plugin import prepare

    # Import this plugin-only module without importing RTP's optional bindings.
    path = (
        Path(__file__).resolve().parents[1]
        / "atom/plugin/rtpllm/utils/gdn_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("rtp_gdn_checkpoint_contract", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    calls = []
    q = torch.ones((1, 2, 1, 2))
    h = torch.arange(6, dtype=torch.float32).reshape(1, 1, 1, 2, 3)
    final = torch.ones((1, 1, 2, 3))

    def prepare_kernel(**kwargs):
        calls.append(("prepare", tuple(kwargs), kwargs["BT"], kwargs["use_exp2"]))
        return q, q, q

    def state_kernel(**kwargs):
        calls.append(
            (
                "state",
                tuple(kwargs),
                kwargs["snapshot_dtype"],
                kwargs["state_dtype"],
            )
        )
        return h, q, final

    def output_kernel(*args, **kwargs):
        calls.append(("output", len(args), tuple(kwargs), kwargs["head_major_vk"]))
        return q * 2

    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", "rtpllm")
    monkeypatch.setattr(gdn_flydsl, "prefill_supported", lambda *args: True)
    monkeypatch.setattr(gdn_flydsl, "_log_dispatch", lambda *args: None)
    monkeypatch.setattr(gdn_flydsl, "l2norm_fwd", lambda value: value)
    monkeypatch.setattr(gdn_flydsl, "prepare_prefill_state", lambda state, *_: state)
    monkeypatch.setattr(gdn_flydsl, "ops", lambda: (None, prepare_kernel, state_kernel))
    monkeypatch.setattr(gdn_flydsl, "chunk_fwd_o", output_kernel)
    original = getattr(gdn_flydsl.prefill, "_rtp_original", gdn_flydsl.prefill)
    monkeypatch.setattr(gdn_flydsl, "prefill", original)
    module.install_rtp_native_prefill_snapshot()
    metadata = (object(), object())
    inputs = (q, q, q, q, q, final, None, metadata, True)

    baseline = original(*inputs)
    baseline_calls = calls.copy()
    calls.clear()
    optimized = gdn_flydsl.prefill(*inputs)

    assert calls == baseline_calls
    assert torch.equal(optimized[0], baseline[0])
    assert torch.equal(optimized[1], baseline[1])
    assert torch.equal(optimized[2].transpose(-1, -2), baseline[2])
    assert optimized[2]._rtp_gdn_snapshot_vk is True


def test_shared_ple_forward_calls_optional_checkpoint_writer(monkeypatch):
    pytest.importorskip("aiter", reason="Qwen4Exp PLE requires AITER")
    from atom.model_ops.qwen4_exp import ple_layer

    monkeypatch.setattr(ple_layer, "advance_ngram_state", lambda *args, **kwargs: None)
    writes = []
    layer = SimpleNamespace(
        ple_embedding=SimpleNamespace(eos_token_id=0, ngram_size=3),
        conv_state_len=3,
        norm_conv=None,
        gated_memory=lambda hidden, *_: hidden.unsqueeze(1),
        _apply_norm=lambda _norm, gated: gated,
        _short_conv=lambda normalized, *_: torch.full_like(normalized, 3),
        _state_checkpoint_writer=lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    metadata = SimpleNamespace(
        query_start_loc=None,
        ngram_state=None,
        state_indices_in=None,
        state_indices_out=None,
        has_initial_state=None,
        conv_state=None,
        num_accepted_tokens=None,
    )
    hidden = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    ids = torch.arange(3)
    result = ple_layer.Qwen4ExpPLELayer.forward_with_state(layer, hidden, ids, metadata)
    assert torch.equal(result, hidden + 3)
    assert len(writes) == 1
    assert torch.equal(writes[0][0][0], ids)
    assert torch.equal(writes[0][0][1], hidden)
    assert writes[0][1] == {"history_width": 2, "conv_state_len": 3}
    del layer._state_checkpoint_writer
    assert torch.equal(
        ple_layer.Qwen4ExpPLELayer.forward_with_state(layer, hidden, ids, metadata),
        result,
    )
    assert len(writes) == 1

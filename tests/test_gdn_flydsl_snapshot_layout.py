"""CPU contracts for the layout of the GDN prefill's chunk-state snapshot."""

import pytest
import torch


def test_kv_snapshot_is_the_kernel_states_viewed_as_kv(monkeypatch):
    pytest.importorskip("triton", reason="GDN prefill requires Triton")
    from atom.model_ops.fla_ops import gdn_flydsl

    q = torch.ones((1, 2, 1, 2))
    # The K5 kernel's chunk states, stored VK: [B, NT, H, V, K].
    h = torch.arange(24, dtype=torch.float32).reshape(1, 2, 1, 3, 4)
    final = torch.ones((1, 1, 3, 4))
    monkeypatch.setattr(gdn_flydsl, "prefill_supported", lambda *args: True)
    monkeypatch.setattr(gdn_flydsl, "_log_dispatch", lambda *args: None)
    monkeypatch.setattr(gdn_flydsl, "l2norm_fwd", lambda value: value)
    monkeypatch.setattr(gdn_flydsl, "prepare_prefill_state", lambda state, *_: state)
    monkeypatch.setattr(
        gdn_flydsl,
        "ops",
        lambda: (None, lambda **kwargs: (q, q, q), lambda **kwargs: (h, q, final)),
    )
    monkeypatch.setattr(gdn_flydsl, "chunk_fwd_o", lambda *args, **kwargs: q)
    prefill = getattr(gdn_flydsl.prefill, "_rtp_original", gdn_flydsl.prefill)
    inputs = (q, q, q, q, q, final, None, (object(), object()))

    _, _, kv = prefill(*inputs, True)
    _, _, vk = prefill(*inputs, True, snapshot_layout="vk")
    _, _, none = prefill(*inputs, False)
    assert vk is h and none is None
    # The KV contract without a copy: the kernel's own storage, by strides.
    assert kv.shape == (1, 2, 1, 4, 3)
    assert kv.data_ptr() == h.data_ptr() and kv.stride() == (24, 12, 12, 1, 4)
    assert torch.equal(kv, h.transpose(-1, -2))


def test_checkpoint_writer_rejects_an_unknown_snapshot_layout():
    pytest.importorskip("triton", reason="GDN checkpoints require Triton")
    from atom.model_ops.fla_ops.state_checkpoint import write_state_checkpoints

    pool = torch.zeros(2, 1, 4, 3)
    # Neither [K, V] nor [V, K] contiguous per state: every other column.
    strided = torch.zeros(1, 2, 1, 4, 6)[..., ::2]
    one = torch.zeros(1, dtype=torch.int32)
    with pytest.raises(ValueError, match="neither a KV nor a VK layout"):
        write_state_checkpoints(
            strided,
            pool,
            torch.zeros(64, 2),
            torch.zeros(2, 2, 3),
            one,
            one,
            one,
            one,
            one,
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 64], dtype=torch.int32),
            64,
            3,
        )

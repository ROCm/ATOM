# SPDX-License-Identifier: MIT
"""V4.1 DSpark input taps write the drafter's capture rows themselves
(`AuxCaptureSpec.writes_into`): the fused post + mean lands in the buffer, and
the hook has nothing left to copy."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


def _modules():
    pytest.importorskip("aiter", reason="Single-Pass mHC calls AITER kernels")
    from atom.model_ops.deepseek_v41 import mhc
    from atom.models.deepseek_v41 import dspark
    from atom.spec_decode import drafter

    return mhc, dspark, drafter


def _owed_state(mhc, tokens, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    residual = torch.randn(tokens, 4, hidden, device="cuda", generator=g).to(
        torch.bfloat16
    )
    pending = (torch.randn(tokens, hidden, device="cuda", generator=g) * 3).to(
        torch.bfloat16
    )
    post = torch.rand(tokens, 4, 1, device="cuda", generator=g) * 2
    comb = torch.softmax(torch.randn(tokens, 4, 4, device="cuda", generator=g), -1)
    pre = torch.sigmoid(torch.randn(tokens, 4, device="cuda", generator=g))
    state = mhc.SinglePassHCState(residual, pre, pending, post, comb)
    expected = mhc.expand_residual(pending, residual, post, comb).mean(dim=-2)
    return state, expected


def test_the_tap_returns_the_offered_rows():
    mhc, dspark, _ = _modules()
    state, expected = _owed_state(mhc, 9, 2048, 1)
    tap = dspark.DeepseekV41DSpark._target_layer_input
    buffer = torch.full((16, 2048), float("nan"), device="cuda", dtype=torch.bfloat16)
    rows = tap((state,), None, out_for=lambda n: buffer[5 : 5 + n])
    assert rows.data_ptr() == buffer[5].data_ptr() and rows.shape == (9, 2048)
    assert torch.equal(buffer[5:14], expected)
    assert buffer[:5].isnan().all() and buffer[14:].isnan().all()
    # Offered nothing, the tap returns a new tensor of the same bits.
    fresh = tap((state,), None)
    assert fresh.data_ptr() != buffer.data_ptr() and torch.equal(fresh, expected)


def test_the_spec_hook_writes_the_buffer_without_a_copy(monkeypatch):
    mhc, dspark, drafter = _modules()
    model = dspark.DeepseekV41DSpark.__new__(dspark.DeepseekV41DSpark)
    model.config = SimpleNamespace(engram_layer_ids=[])
    spec = model.target_aux_capture_spec((37, 38, 39), 2048)
    assert spec.writes_into and spec.capture == "input"

    state, expected = _owed_state(mhc, 9, 2048, 2)
    buffer = torch.full((16, 2048), float("nan"), device="cuda", dtype=torch.bfloat16)
    ctx = SimpleNamespace(
        is_draft=False, ubatch_token_offset=5, ubatch_token_order=None
    )
    monkeypatch.setattr(
        drafter, "get_forward_context", lambda: SimpleNamespace(context=ctx)
    )
    returned = []

    def extract(*args, **kwargs):
        returned.append(spec.extract(*args, **kwargs))
        return returned[-1]

    owner = SimpleNamespace(_aux_buffers=[buffer])
    hook = drafter.Drafter._make_aux_hook(
        owner, 0, extract, writes_into=spec.writes_into
    )
    hook(None, (), (state,))  # what the input pre-hook hands it
    assert returned[0].data_ptr() == buffer[5].data_ptr()
    assert torch.equal(buffer[5:14], expected)
    assert buffer[:5].isnan().all() and buffer[14:].isnan().all()

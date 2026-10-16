# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DP pad rows on the fused MoRI v2 transport (ATOM_MORI_V2_FUSED=1) under
ATOM_MEGA_MASK_PAD_ROWS: routed to -1, returned as zeros. The MegaMoE op is a
fake that records the ids it was handed and returns NaN, so a row that is not
zeroed fails loudly.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
pytest.importorskip("triton", reason="MegaMoE imports its Triton row-zeroing kernel")

import torch

import atom.model_ops.fused_moe.flydsl_mega_experts as fme
import atom.model_ops.fused_moe.mori_v2_prepare_finalize as mv2
import atom.utils.forward_context as fc

TOPK = 9
DIM = 16


def test_prepare_forwards_exact_device_row_count(monkeypatch):
    class FakeOp:
        @staticmethod
        def dispatch(a1, weights, _scales, ids, *, return_routing):
            assert return_routing
            return (
                a1,
                weights,
                None,
                ids,
                torch.tensor([a1.shape[0]], dtype=torch.int32),
                object(),
            )

    monkeypatch.setattr(mv2.envs, "ATOM_USE_TRITON_MOE", False)
    pf = mv2.MoriV2PrepareAndFinalize.__new__(mv2.MoriV2PrepareAndFinalize)
    pf._op = FakeOp()
    pf._routing = None
    hidden = torch.ones((2, DIM), dtype=torch.bfloat16)
    ids = torch.tensor([[0, 33, 66, 99, 32], [1, 34, 67, 100, 32]])
    weights = torch.ones_like(ids, dtype=torch.float32)
    expert_map = torch.full((130,), -1, dtype=torch.int32)

    _, _, metadata, dispatched_ids, dispatched_weights = pf.prepare(
        hidden,
        weights,
        ids,
        num_experts=132,
        expert_map=expert_map,
        apply_router_weight_on_input=False,
        quant_config=None,
    )

    assert torch.equal(dispatched_ids, ids.to(torch.int32))
    assert torch.equal(dispatched_weights, weights)
    assert metadata.expert_num_tokens.item() == hidden.shape[0]
    assert pf.needs_dispatch_output_trim() is False


@pytest.fixture
def run(monkeypatch):
    seen = {}

    class FakeMega:
        max_tokens_per_rank = 256

        def __call__(self, x, weights, ids, **kwargs):
            seen["ids"] = ids.clone()
            return torch.full_like(x, float("nan"))

    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(fc, "_row_index_device", None)
    monkeypatch.setattr(fc, "_real_requests_device", None)
    fc.enable_pad_rows_device(256, torch.device("cpu"))

    def zero_pad_rows_(out, pad_rows):
        out.masked_fill_(pad_rows, 0)
        return out

    monkeypatch.setattr(fme, "zero_pad_rows_", zero_pad_rows_)
    monkeypatch.setattr(
        mv2.MoriV2ModularKernel, "_assert_recipe_matches", lambda *a: None
    )
    kernel = mv2.MoriV2ModularKernel.__new__(mv2.MoriV2ModularKernel)
    kernel._recv_bound = lambda arena_rows: None
    pf = SimpleNamespace(
        mega=FakeMega(),
        mask_pad_rows=True,
        num_dispatchers=lambda: 4,
        combine_quant_for_step=lambda: "bf16",
    )
    object.__setattr__(kernel, "prepare_finalize", pf)

    def _run(*, scheduled, running, rows=None, capturing=False, mask=True):
        rows = running if rows is None else rows
        pf.mask_pad_rows = mask
        # A target pass of one-token requests: `running` slots, `scheduled` real.
        cu = torch.arange(running + 1, dtype=torch.int32).clamp(max=scheduled)
        context = SimpleNamespace(
            is_draft=False,
            scheduled_tokens=scheduled,
            running_tokens=running,
            running_bs=running,
        )
        fwd = SimpleNamespace(
            context=context,
            attn_metadata=SimpleNamespace(cu_seqlens_q=cu),
            pad_rows_done=set(),
        )
        monkeypatch.setattr(fc, "get_forward_context", lambda: fwd)
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: capturing
        )
        ids = (torch.arange(rows * TOPK) % 260).reshape(rows, TOPK)
        out = kernel.forward(
            torch.ones(rows, DIM, dtype=torch.bfloat16),
            torch.zeros(1),
            torch.zeros(1),
            torch.ones(rows, TOPK),
            ids,
        )
        return ids.to(torch.int32), seen["ids"], out

    return _run


@pytest.mark.parametrize("capturing", [False, True])
def test_pad_rows_are_dropped_whole_and_come_back_zero(run, capturing):
    ids, sent, out = run(scheduled=37, running=64, capturing=capturing)
    assert torch.equal(sent[:37], ids[:37])
    assert (sent[37:] == -1).all()  # every slot, shared expert included
    assert (out[37:] == 0).all()
    assert out[:37].isnan().all()  # real rows are the op's, untouched


def test_unpadded_step_is_passed_through(run):
    ids, sent, out = run(scheduled=40, running=40)
    assert torch.equal(sent, ids)
    assert out.isnan().all()


@pytest.mark.parametrize("rows", [7, 48])
def test_rows_that_are_not_the_step_width_are_left_alone(run, rows):
    ids, sent, out = run(scheduled=5, running=32, rows=rows)
    assert torch.equal(sent, ids)
    assert out.isnan().all()


def test_disabled_is_identity(run):
    ids, sent, out = run(scheduled=5, running=32, mask=False)
    assert torch.equal(sent, ids)
    assert out.isnan().all()


def test_bind_enables_the_mask_for_the_transport_rows(monkeypatch):
    import atom.model_ops.fused_moe.flydsl_mega_experts as fme

    asked = []
    monkeypatch.setattr(
        fme, "_enable_mega_pad_row_mask", lambda rows: asked.append(rows) or True
    )
    monkeypatch.setattr(mv2, "init_mega_transport", lambda **kw: object())
    pf = mv2.MoriV2PrepareAndFinalize.__new__(mv2.MoriV2PrepareAndFinalize)
    pf._mega_geometry = {}
    pf.mega = None
    pf.max_tokens_per_rank = 12288
    pf.mask_pad_rows = False
    quant = SimpleNamespace(
        intermediate_size=2048,
        is_guinterleave=True,
        quant_type=None,
        hidden_pad=0,
        intermediate_pad=0,
    )
    pf.bind_mega_transport(SimpleNamespace(activation=None), quant)
    assert asked == [12288]
    assert pf.mask_pad_rows is True

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DP pad rows on the MegaMoE backend (ATOM_MEGA_MASK_PAD_ROWS): routed to -1,
returned as zeros. The MegaMoEV2 op is a fake that records the ids it was handed
and returns NaN, so a row that is not zeroed fails loudly. The GPU-only row
zeroing helper is replaced with an in-place CPU test double.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="MegaMoE imports its Triton row-zeroing kernel")

import atom.utils.forward_context as fc
from atom.model_ops.fused_moe import flydsl_mega_experts as mega

TOPK = 7
DIM = 16


@pytest.fixture
def run(monkeypatch):
    seen = {}

    class FakeMegaMoEV2:
        supports_combine_mask = False

        def __init__(self, **kwargs):
            pass

        def forward(self, x, weights, ids, **kwargs):
            seen["ids"] = ids.clone()
            seen["kwargs"] = kwargs
            return torch.full_like(x, float("nan"))

    def stub_module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    manager = SimpleNamespace(rank=0, world_size=8)
    ep_group = SimpleNamespace(
        device_communicator=SimpleNamespace(all2all_manager=manager)
    )
    stub_module("aiter.dist.parallel_state", get_ep_group=lambda: ep_group)
    stub_module("aiter.ops.flydsl.kernels.mega_moe", MegaMoEV2=FakeMegaMoEV2)
    monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", "0")
    monkeypatch.setattr(mega, "_MEGA_CACHE", {})
    monkeypatch.setattr(fc, "_pad_rows_device", None)
    monkeypatch.setattr(fc, "_row_index_device", None)
    monkeypatch.setattr(fc, "_real_requests_device", None)
    fc.enable_pad_rows_device(256, torch.device("cpu"))

    def zero_pad_rows_(out, pad_rows):
        seen["zero_pad_calls"] = seen.get("zero_pad_calls", 0) + 1
        out.masked_fill_(pad_rows, 0)
        return out

    monkeypatch.setattr(mega, "zero_pad_rows_", zero_pad_rows_)
    layer = SimpleNamespace(
        **{
            name: torch.zeros(48, 4, dtype=torch.uint8)
            for name in ("_mega_w1", "_mega_w1_scale", "_mega_w2", "_mega_w2_scale")
        }
    )

    def _run(*, scheduled, running, rows=None, capturing=False, mask=True):
        rows = running if rows is None else rows
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
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        ids = (torch.arange(rows * TOPK) % 384).reshape(rows, TOPK)

        def call():
            return mega.run_mega_moe(
                layer,
                torch.ones(rows, DIM, dtype=torch.bfloat16),
                torch.ones(rows, TOPK),
                ids,
                model_dim=DIM,
                inter_dim=8,
                experts=384,
                topk=TOPK,
                mtpr=256,
                swiglu_limit=0.0,
                mask_pad_rows=mask,
            )

        if capturing:
            call()  # eager warmup builds the instance; a capture may only reuse it
            monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        out = call()
        return ids.to(torch.int32), seen["ids"], out

    _run.mega_cls = FakeMegaMoEV2
    _run.seen = seen
    return _run


@pytest.mark.parametrize("capturing", [False, True])
def test_pad_rows_are_dropped_whole_and_come_back_zero(run, capturing):
    ids, sent, out = run(scheduled=5 * 7, running=8 * 7, capturing=capturing)
    assert torch.equal(sent[: 5 * 7], ids[: 5 * 7])
    assert (sent[5 * 7 :] == -1).all()  # every slot, shared expert included
    assert (out[5 * 7 :] == 0).all()
    assert out[: 5 * 7].isnan().all()  # real rows are the op's, untouched


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


def test_switch_follows_the_env(monkeypatch):
    enabled = []
    monkeypatch.setattr(
        fc, "enable_pad_rows_device", lambda rows, device: enabled.append(rows)
    )
    monkeypatch.setenv("ATOM_MEGA_MASK_PAD_ROWS", "0")
    assert mega._enable_mega_pad_row_mask(128) is False
    monkeypatch.setenv("ATOM_MEGA_MASK_PAD_ROWS", "1")
    assert mega._enable_mega_pad_row_mask(128) is True
    assert enabled == [128]


def test_masking_combine_output_is_not_reselected(run, monkeypatch):
    # An aiter whose combine skips -1 slots returns the pad rows itself; the
    # output select would be a redundant elementwise pass per layer.
    monkeypatch.setattr(run.mega_cls, "supports_combine_mask", True)
    _, sent, out = run(scheduled=5 * 7, running=8 * 7)
    assert (sent[5 * 7 :] == -1).all()
    assert run.seen["kwargs"] == {"mask_invalid_slots": True}
    assert out.isnan().all()


def test_masking_combine_is_only_requested_for_padded_passes(run, monkeypatch):
    monkeypatch.setattr(run.mega_cls, "supports_combine_mask", True)
    run(scheduled=40, running=40)
    assert run.seen["kwargs"] == {"mask_invalid_slots": False}

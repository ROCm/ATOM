# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 launch-A ``mf`` kernel module (CPU): the MoE-front launch is built with the real-row count production's
router reads (``pass_scheduled_tokens_gpu``), one table a (MoE, count tensor); anything it cannot hand that count
(a count tensor first met inside a capture, a non-int32 count) runs Stage O instead."""

import sys
import types

import pytest
import torch

from atom.models.minimax_m3.launch_a_kernels import mf, stage_o


class _FakeAiterMf(types.ModuleType):
    def __init__(self):
        super().__init__("atom.models.minimax_m3.mono.gfx1250.mf")
        self.tables = []  # num_valid_rows of each mf_table call
        self.launches = []  # table of each flydsl_mf call
        self.roles = []  # roles of each flydsl_mf call

    class MfOutputs:
        def __init__(self, device):
            self.device = device
            self.shared = torch.zeros(8, 4)  # the launch's shared output (the glue epilogue's addend)

    def mf_scratch(self, device):
        return {}

    def mf_table(self, gate_w, bias, gu_w, gu_s, dn_w, dn_s, outs, scratch, num_valid_rows=None):
        self.tables.append(num_valid_rows)
        return ("table", len(self.tables), outs)

    def flydsl_mf(self, x, x_scale, w, w_scale, res, norm_w, eps, inbox, ctr, slot, rank, tab, **kw):
        self.launches.append(tab)
        self.roles.append(kw.get("roles", 7))
        return torch.zeros(8, 4), torch.zeros(8, 4)


@pytest.fixture
def env(monkeypatch):
    fake = _FakeAiterMf()
    state = {"nv": None, "capturing": False, "fronts": []}
    launch_a = types.SimpleNamespace(
        k5_maxit=lambda: 1000,
        set_moe_front=lambda moe, out, outs: state["fronts"].append(outs),
    )
    fwd = types.SimpleNamespace(pass_scheduled_tokens_gpu=lambda rows: state["nv"])
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.mono.gfx1250.mf", fake)
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.launch_a", launch_a)
    monkeypatch.setitem(sys.modules, "atom.utils.forward_context", fwd)
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.module_dispatch_ops",
        types.SimpleNamespace(_dual_stream_is_active=lambda moe, m: True),
    )
    monkeypatch.setattr(mf, "_takes", lambda moe: True)
    monkeypatch.setattr(mf, "ENABLED", True)
    monkeypatch.setattr(mf, "_TABLES", {})
    monkeypatch.setattr(mf, "_OUTPUTS", {}, raising=False)
    monkeypatch.setattr(stage_o, "run", lambda call: "stage_o")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["capturing"])
    return fake, state


def _call(rows=8):
    u8 = torch.zeros(4, dtype=torch.uint8)
    moe = types.SimpleNamespace(
        prefix="model.layers.3.block_sparse_moe",
        gate=types.SimpleNamespace(weight=torch.zeros(4, 4, dtype=torch.bfloat16)),
        e_score_correction_bias=torch.zeros(4),
        shared_experts=types.SimpleNamespace(
            gate_up_proj=types.SimpleNamespace(weight=u8, weight_scale=u8),
            down_proj=types.SimpleNamespace(weight=u8, weight_scale=u8),
        ),
    )
    seam = types.SimpleNamespace(moe=moe, o_proj=types.SimpleNamespace(weight=u8, weight_scale=u8))
    k5 = types.SimpleNamespace(inbox=[0] * 4, ctr=0, slot=0, rank=0, max_rows=32)
    return types.SimpleNamespace(
        seam=seam,
        x=torch.zeros(rows, 2048, dtype=torch.uint8),
        x_scale=torch.zeros(rows, 64, dtype=torch.uint8),
        residual=torch.zeros(rows, 4),
        norm_w=torch.zeros(4),
        eps=1e-6,
        k5=k5,
    )


def test_table_takes_production_row_count(env):
    fake, state = env
    nv = torch.tensor([5], dtype=torch.int32)
    state["nv"] = nv
    call = _call()
    out = mf.run(call)
    assert out != "stage_o"
    assert fake.tables == [nv]  # the launch's router masks rows >= nv, as production's
    assert len(fake.launches) == 1 and len(state["fronts"]) == 1
    mf.run(call)  # same count tensor: same table
    assert len(fake.tables) == 1 and len(fake.launches) == 2


def test_one_table_per_count_tensor_same_outputs(env):
    fake, state = env
    call = _call()
    state["nv"] = torch.tensor([8], dtype=torch.int32)
    mf.run(call)
    state["nv"] = torch.tensor([3], dtype=torch.int32)
    mf.run(call)
    state["nv"] = None
    mf.run(call)
    assert [None if t is None else int(t) for t in fake.tables] == [8, 3, None]
    outs = {id(t[2]) for t in fake.launches}
    assert len(outs) == 1  # one output set for the layer, whichever count it routed with


def test_unknown_count_inside_capture_runs_stage_o(env):
    fake, state = env
    state["nv"] = torch.tensor([8], dtype=torch.int32)
    state["capturing"] = True
    assert mf.run(_call()) == "stage_o"
    assert fake.tables == [] and fake.launches == [] and state["fronts"] == []


def test_non_int32_count_runs_stage_o(env):
    fake, state = env
    state["nv"] = torch.tensor([8], dtype=torch.int64)
    assert mf.run(_call()) == "stage_o"
    assert fake.launches == []


@pytest.mark.parametrize("rows", [1, 7, 16, 32])
def test_other_rows_run_stage_o(env, rows):
    fake, state = env
    assert mf.run(_call(rows)) == "stage_o"
    assert fake.launches == []


def test_switch_off_runs_stage_o(env, monkeypatch):
    fake, state = env
    monkeypatch.setattr(mf, "ENABLED", False)
    assert mf.run(_call()) == "stage_o"
    assert fake.launches == []


def test_whole_shared_expert_without_tail_down(env, monkeypatch):
    fake, state = env
    monkeypatch.setattr(mf, "shared_down_registry", lambda: {})
    state["nv"] = torch.tensor([8], dtype=torch.int32)
    mf.run(_call())
    assert fake.roles == [7]  # router + gate_up + down: the launch's shared output is the addend


class _Registered(dict):
    def __contains__(self, key):
        return True


def test_router_gate_up_when_tail_runs_down(env, monkeypatch):
    fake, state = env
    monkeypatch.setattr(mf, "shared_down_registry", lambda: _Registered())
    state["nv"] = torch.tensor([8], dtype=torch.int32)
    mf.run(_call())
    assert fake.roles == [3]  # router + gate_up: the shared down runs in P6's tail launch

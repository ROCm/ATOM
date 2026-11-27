# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 DMF seam (CPU): a dense layer's post-attention op runs aiter's one launch only at a C4 decode call of
8 rows whose AR is production's k5 Gluon one-shot and whose MLP is the shape the launch takes, with one table a
layer built outside any capture; every other call runs production's calls in the layer's order (o_proj, the AR +
norm the layer's MLP input takes, the MLP)."""

import sys
import types

import pytest
import torch

from atom.models.minimax_m3 import dense_mf


class _FakeDmf(types.ModuleType):
    def __init__(self):
        super().__init__("atom.models.minimax_m3.mono.gfx1250.dmf")
        self.tables = 0
        self.launches = []

    class DmfOutputs:
        def __init__(self, device):
            self.down = torch.full((8, 6144), 3.0, dtype=torch.bfloat16)
            self.x2 = torch.zeros((8, 6144), dtype=torch.uint8)
            self.x2s = torch.zeros((8, 192), dtype=torch.uint8)
            self.r2 = torch.zeros((8, 6144), dtype=torch.bfloat16)

    def dmf_scratch(self, device, build=None):
        self.builds = getattr(self, "builds", []) + [build]
        return {}

    def dmf_table(self, gu_w, gu_s, dn_w, dn_s, outs, scratch, next_norm_w=None):
        self.tables += 1
        self.next_norms = getattr(self, "next_norms", []) + [next_norm_w]
        return ("table", self.tables)

    def dmf_supported(self, rows, inter, k):
        return None

    def flydsl_dmf(self, x, x_scale, w, w_scale, res, norm_w, eps, inbox, ctr, slot, rank, tab, **kw):
        self.launches.append((tab, kw))
        return torch.zeros(8, 6144, dtype=torch.bfloat16), torch.full((8, 6144), 5.0, dtype=torch.bfloat16)


class _Mlp:
    swiglu_alpha, swiglu_beta, swiglu_limit = 1.702, 1.0, 7.0

    def __init__(self, calls):
        self.calls = calls
        w = torch.zeros(4, dtype=torch.uint8)
        self.gate_up_proj = types.SimpleNamespace(weight=w, weight_scale=w, kind="gu")
        self.down_proj = types.SimpleNamespace(weight=w, weight_scale=w)

    def __call__(self, h, x_scale=None):
        self.calls.append(("mlp", h, x_scale))
        return torch.full((h.shape[0], 6144), 7.0, dtype=torch.bfloat16)


@pytest.fixture
def env(monkeypatch):
    fake = _FakeDmf()
    calls = []
    state = {"capturing": False, "decode": True, "route": True, "takes": True, "pt": False, "mx": True}
    k5 = types.SimpleNamespace(inbox=[1, 2, 3, 4], ctr=5, slot=6, rank=0, max_rows=256)
    launch_a = types.SimpleNamespace(_k5=lambda: k5, k5_maxit=lambda: 1000)
    m3 = types.SimpleNamespace(
        _fast_oproj_resolved=lambda o_proj: True,
        _linear_consumes_per_token_fp8=lambda lin: state["pt"],
        _linear_consumes_mxfp8=lambda lin: state["mx"],
        _o_proj_mxfp8=lambda o_proj, x, s: calls.append(("o_proj",)) or torch.ones(x.shape[0], 6144),
    )

    def ar(kind, quant):
        def f(h, residual, norm):
            calls.append((kind,))
            return (h, "scale", residual) if quant else (h, residual)

        return f

    ln = types.SimpleNamespace(
        fused_allreduce_gemma_rms_norm=ar("ar", False),
        fused_allreduce_gemma_rms_norm_mxfp8_quant=ar("ar_mxfp8", True),
        fused_allreduce_gemma_rms_norm_quant=ar("ar_fp8", True),
        sp_fused_allreduce_gemma_rms_norm=ar("sp_ar", False),
        sp_fused_allreduce_gemma_rms_norm_mxfp8_quant=ar("sp_ar_mxfp8", True),
    )
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.mono.gfx1250.dmf", fake)
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250",
        types.SimpleNamespace(mxfp8_decode_gemm_supported=lambda m, n, k: True),
    )
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.launch_a", launch_a)
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.model", m3)
    monkeypatch.setitem(sys.modules, "atom.model_ops.layernorm", ln)
    monkeypatch.setitem(
        sys.modules, "atom.utils.envs", types.SimpleNamespace(ATOM_MINIMAX_M3_GLUON_AR_MXFP8=True)
    )
    import atom.models.minimax_m3 as pkg
    import atom.utils as utils_pkg

    monkeypatch.setattr(pkg, "launch_a", launch_a, raising=False)
    monkeypatch.setattr(pkg, "model", m3, raising=False)
    monkeypatch.setattr(utils_pkg, "envs", sys.modules["atom.utils.envs"], raising=False)
    monkeypatch.setattr(dense_mf, "_decode_step", lambda: state["decode"])
    monkeypatch.setattr(dense_mf, "_k5_gluon_mxfp8_route", lambda r, w, mode=0: state["route"])
    monkeypatch.setattr(dense_mf, "_mlp_takes", lambda mlp: state["takes"])
    monkeypatch.setattr(dense_mf, "ENABLED", True)
    monkeypatch.setattr(dense_mf, "_TABLES", {})
    monkeypatch.setattr(dense_mf, "_SEAMS", {})
    monkeypatch.setattr(dense_mf, "STATS", {"dmf": 0, "dmf_production": 0, "dmf_ar2": 0, "ar2_taken": 0,
                                            "ar2_missed": 0})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["capturing"])
    mlp = _Mlp(calls)
    layer = _Layer()
    layer.self_attn = types.SimpleNamespace(
        layer_num=1, o_proj=types.SimpleNamespace(weight=torch.zeros(6144, 2048), weight_scale=None)
    )
    layer.mlp = mlp
    layer.post_attention_layernorm = types.SimpleNamespace(variance_epsilon=1e-6)
    dense_mf._SEAMS["l1"] = dense_mf.Seam("l1", layer)
    monkeypatch.setattr(dense_mf, "_DONE", {})
    monkeypatch.setattr(dense_mf, "_HOOK", {"ok": True})
    return fake, state, calls


class _Layer:  # weakly referenced, as an nn.Module is
    pass


def _op(rows=8, mode=0):
    x = torch.zeros(rows, 2048, dtype=torch.float8_e4m3fn)
    xs = torch.zeros(rows, 64, dtype=torch.uint8)
    res = torch.zeros(rows, 6144, dtype=torch.bfloat16)
    return dense_mf.minimax_m3_dense_tail(x, xs, res, torch.zeros(6144), 1e-6, "l1", mode)


def test_decode_call_runs_the_one_launch_with_one_table(env):
    fake, _, calls = env
    for _ in range(3):
        out, res_out = _op()
        assert torch.equal(out, torch.full((8, 6144), 3.0, dtype=torch.bfloat16))  # the launch's down output
        assert torch.equal(res_out, torch.full((8, 6144), 5.0, dtype=torch.bfloat16))
    assert fake.tables == 1 and len(fake.launches) == 3 and calls == []
    kw = fake.launches[0][1]
    assert (kw["alpha"], kw["beta"], kw["limit"], kw["max_rows"]) == (1.702, 1.0, 7.0, 256)
    assert dense_mf.stats()["dmf"] == 3 and dense_mf.stats()["dmf_production"] == 0
    assert dense_mf.stats()["dmf_ar2"] == 0 and fake.next_norms == [None]  # an aiter launch without ar2


@pytest.mark.parametrize("what", ["rows", "prefill", "route", "takes", "capture"])
def test_other_calls_run_production_in_order(env, what):
    fake, state, calls = env
    rows = 16 if what == "rows" else 8
    state["decode"] = what != "prefill"
    state["route"] = what != "route"
    state["takes"] = what != "takes"
    state["capturing"] = what == "capture"  # no table met outside a capture
    out, res_out = _op(rows)
    assert not fake.launches
    assert [c[0] for c in calls] == ["o_proj", "ar_mxfp8", "mlp"]
    assert calls[2][2] == "scale"  # the MLP takes the AR's MXFP8 output
    assert torch.equal(out, torch.full((rows, 6144), 7.0, dtype=torch.bfloat16))
    assert dense_mf.stats()["dmf"] == 0 and dense_mf.stats()["dmf_production"] == 1


@pytest.mark.parametrize("pt, mx, want", [(True, False, "ar_fp8"), (False, False, "ar")])
def test_production_follows_the_layers_ar_choice(env, pt, mx, want):
    _, state, calls = env
    state["decode"] = False
    state["pt"], state["mx"] = pt, mx
    _op()
    assert [c[0] for c in calls] == ["o_proj", want, "mlp"]
    assert calls[2][2] == ("scale" if want == "ar_fp8" else None)


def test_ar2_outputs_go_to_the_next_layers_ar_call(env, monkeypatch):
    fake, _, calls = env
    launches = []

    def flydsl_dmf(x, x_scale, w, w_scale, res, norm_w, eps, inbox, ctr, slot, rank, tab, *, ar2=0, **kw):
        launches.append(ar2)
        return torch.zeros(8, 6144, dtype=torch.bfloat16), torch.full((8, 6144), 5.0, dtype=torch.bfloat16)

    fake.flydsl_dmf = flydsl_dmf
    nw2 = torch.ones(6144, dtype=torch.bfloat16)
    monkeypatch.setattr(dense_mf, "_next_input_norm", lambda seam: types.SimpleNamespace(weight=nw2))
    prod = []

    class Comm:
        def fused_allreduce_rmsnorm_mxfp8_quant(self, input_, res_inp_, weight_, eps, prefill_support=False,
                                                gemma_norm=False):
            prod.append(input_)
            return "production"

    comm = Comm()
    tp = types.SimpleNamespace(device_communicator=comm)
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", types.SimpleNamespace(get_tp_group=lambda: tp))
    monkeypatch.setattr(dense_mf, "_HOOK", {})
    monkeypatch.setattr(dense_mf, "AR2", True)  # off by default; the served arm turns it on
    down, res_out = _op()
    assert launches == [1] and len(fake.next_norms) == 1 and fake.next_norms[0].data_ptr() == nw2.data_ptr() and fake.builds[-1][-1] == 1
    assert dense_mf.stats()["dmf_ar2"] == 1
    # the next layer's AR call on the launch's outputs takes them, production's does not run
    got = comm.fused_allreduce_rmsnorm_mxfp8_quant(down, res_out, nw2, 1e-6, gemma_norm=True)
    assert got[0].dtype == torch.float8_e4m3fn and got[1].dtype == torch.bfloat16 and got[2].shape == (8, 192)
    assert not prod and dense_mf.stats()["ar2_taken"] == 1
    # taken once: a second call (or any other) is production's
    assert comm.fused_allreduce_rmsnorm_mxfp8_quant(down, res_out, nw2, 1e-6, gemma_norm=True) == "production"
    # a call with another residual does not take the entry
    _op()
    other = torch.zeros(8, 6144, dtype=torch.bfloat16)
    assert comm.fused_allreduce_rmsnorm_mxfp8_quant(down, other, nw2, 1e-6, gemma_norm=True) == "production"
    assert dense_mf.stats()["ar2_missed"] == 1 and len(prod) == 2
    assert calls == []


@pytest.mark.parametrize("mx, want", [(True, "sp_ar_mxfp8"), (False, "sp_ar")])
def test_the_sequence_parallel_forward_falls_back_to_its_own_ops(env, mx, want):
    fake, state, calls = env
    state["decode"] = False
    state["mx"] = mx
    _op(mode=1)
    assert not fake.launches
    assert [c[0] for c in calls] == ["o_proj", want, "mlp"]
    assert calls[2][2] == ("scale" if mx else None)


def test_the_sequence_parallel_forward_takes_the_launch_at_decode(env):
    fake, _, calls = env
    out, _ = _op(mode=1)
    assert len(fake.launches) == 1 and calls == []
    assert torch.equal(out, torch.full((8, 6144), 3.0, dtype=torch.bfloat16))

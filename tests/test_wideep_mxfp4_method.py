# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing

try:
    import atom.model_ops.moe as moe_mod
except ImportError as exc:
    skip_if_dependency_missing(exc, "requires full atom import env")


def _moe_parallel(**overrides):
    values = {
        "use_ep": True,
        "ep_size": 16,
        "local_ep_size": 8,
        "use_mori_kernels": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _moe(**overrides):
    values = {
        "moe_parallel_config": _moe_parallel(),
        "expert_layout": SimpleNamespace(mode=moe_mod.SharedExpertMode.NONE),
        "has_bias": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_explicit_wideep_backend_selects_wideep_method(monkeypatch):
    standard = object()
    wideep = object()
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(moe_backend="wideep"),
    )
    monkeypatch.setattr(moe_mod, "Mxfp4MoEMethod", lambda *_args: standard)
    monkeypatch.setattr(moe_mod, "WideEpMxfp4MoEMethod", lambda *_args: wideep)

    assert moe_mod._make_mxfp4_moe_method(object(), object()) is wideep


def test_wideep_method_requires_ep16_mori(monkeypatch):
    monkeypatch.setattr(moe_mod.Mxfp4MoEMethod, "__init__", lambda *_args: None)
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(eplb_enable=False),
    )

    method = moe_mod.WideEpMxfp4MoEMethod(SimpleNamespace(is_dynamic=True), _moe())
    assert method.use_triton is False
    assert method.use_triton_decode is False

    with pytest.raises(ValueError, match="requires EP16"):
        moe_mod.WideEpMxfp4MoEMethod(
            SimpleNamespace(is_dynamic=True),
            _moe(moe_parallel_config=_moe_parallel(ep_size=8)),
        )

    with pytest.raises(ValueError, match="requires the MORI"):
        moe_mod.WideEpMxfp4MoEMethod(
            SimpleNamespace(is_dynamic=True),
            _moe(moe_parallel_config=_moe_parallel(use_mori_kernels=False)),
        )


def test_wideep_run_calls_aiter_and_copies_combine_output(monkeypatch):
    from aiter.dist import parallel_state

    from atom.model_ops.fused_moe import wideep_experts

    manager = SimpleNamespace(rank=3, world_size=16)
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(all2all_manager=manager)
    )
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: group)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    arena_output = torch.ones((2, 4), dtype=torch.bfloat16)
    calls = []

    class FakeOp:
        def forward(self, hidden, weights, ids):
            calls.append((hidden, weights, ids))
            return arena_output

    captured = {}

    def fake_get(**kwargs):
        captured.update(kwargs)
        return FakeOp()

    monkeypatch.setattr(wideep_experts, "_get_wideep_op", fake_get)
    layer = SimpleNamespace(swiglu_limit=10.0)
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    weights = torch.randn(2, 2, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    output = wideep_experts.run_wideep_moe(
        layer,
        hidden,
        weights,
        ids,
        model_dim=4,
        inter_dim=8,
        experts=16,
        mtpr=32,
    )

    assert captured["rank"] == 3
    assert captured["world_size"] == 16
    assert captured["topk"] == 2
    assert calls[0][1].dtype == torch.float32
    assert calls[0][2].dtype == torch.int32
    assert torch.equal(output, arena_output)
    assert output.data_ptr() != arena_output.data_ptr()


def test_wideep_rejects_torch_compile(monkeypatch):
    from atom.model_ops.fused_moe import wideep_experts

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(RuntimeError, match="--level 0"):
        wideep_experts.run_wideep_moe(
            SimpleNamespace(),
            torch.empty(1, 4),
            torch.empty(1, 1),
            torch.empty(1, 1, dtype=torch.int64),
            model_dim=4,
            inter_dim=8,
            experts=16,
            mtpr=32,
        )

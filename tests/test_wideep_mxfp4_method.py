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
    assert method.is_guinterleave is True

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


def test_wideep_reuses_mori_prepare_finalize_around_fused_moe(monkeypatch):
    from aiter import ActivationType, QuantType
    from aiter.dist import parallel_state

    from atom.model_ops.fused_moe import (
        modular_kernel,
        mori_prepare_finalize,
        wideep_experts,
    )

    manager = SimpleNamespace(rank=3, world_size=16)
    group = SimpleNamespace(
        device_communicator=SimpleNamespace(all2all_manager=manager)
    )
    monkeypatch.setattr(parallel_state, "get_ep_group", lambda: group)
    monkeypatch.setattr(mori_prepare_finalize, "MORI_AVAILABLE", True)
    monkeypatch.delenv("AITER_BF16_FP8_MOE_BOUND", raising=False)

    quantized = torch.zeros(3, 4, dtype=torch.uint8)
    quant_scale = torch.ones(3, 1, dtype=torch.uint8)
    recv_x = torch.zeros(8, 4, dtype=torch.uint8)
    recv_scale = torch.ones(8, 1, dtype=torch.uint8)
    recv_weights = torch.randn(8, 2, dtype=torch.float32)
    recv_ids = torch.zeros(8, 2, dtype=torch.int32)
    recv_count = torch.tensor([2], dtype=torch.int32)
    fused_output = torch.full((3, 4), 7, dtype=torch.bfloat16)
    combined = torch.full((3, 4), 9, dtype=torch.bfloat16)
    calls = {}

    def fake_quantize(hidden_states):
        calls["quantize"] = hidden_states
        return quantized, quant_scale

    class FakeOp:
        def dispatch(self, hidden, weights, scale, ids, block_num, warp_num):
            calls["dispatch"] = (
                hidden,
                weights,
                scale,
                ids,
                block_num,
                warp_num,
            )
            return recv_x, recv_weights, recv_scale, recv_ids, recv_count

        def combine(self, output, bias, ids, block_num, warp_num):
            calls["combine"] = (output, bias, ids, block_num, warp_num)
            return combined, None

    monkeypatch.setattr(wideep_experts, "_init_wideep_op", lambda **_kwargs: FakeOp())
    monkeypatch.setattr(wideep_experts, "_quantize_wideep_dispatch", fake_quantize)
    monkeypatch.setattr(
        modular_kernel,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(running_tokens_across_dp=(1, 2))
        ),
    )

    def fake_fused_moe(*args, **kwargs):
        calls["fused_moe"] = (args, kwargs)
        return fused_output

    monkeypatch.setattr(modular_kernel, "fused_moe", fake_fused_moe)

    prepare_finalize = wideep_experts.make_wideep_prepare_finalize(
        model_dim=4,
        experts=16,
        experts_per_rank=1,
        topk=2,
        mtpr=32,
    )
    assert isinstance(prepare_finalize, mori_prepare_finalize.MoriPrepareAndFinalize)
    kernel = modular_kernel.FusedMoEModularKernel(prepare_finalize, quant_config=None)
    hidden = torch.randn(3, 4, dtype=torch.bfloat16)
    topk_weights = torch.randn(3, 2, dtype=torch.bfloat16)
    topk_ids = torch.zeros(3, 2, dtype=torch.int64)
    expert_mask = torch.tensor([1] + [0] * 16, dtype=torch.int32)

    result = kernel(
        hidden,
        torch.empty(1, 8, 2),
        torch.empty(1, 4, 4),
        topk_weights,
        topk_ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        global_num_experts=16,
        expert_mask=expert_mask,
        w1_scale=torch.ones(1),
        w2_scale=torch.ones(1),
        moe_extra_args={"gate_mode": "interleave", "swiglu_limit": 10.0},
    )

    assert calls["quantize"] is hidden
    dispatch = calls["dispatch"]
    assert dispatch[0] is quantized
    assert dispatch[1].dtype == torch.float32
    assert dispatch[2] is quant_scale
    assert dispatch[3].dtype == torch.int32
    assert dispatch[4:] == (96, 8)

    args, kwargs = calls["fused_moe"]
    assert args[0].shape[0] == 3
    assert torch.equal(args[0], recv_x[:3])
    assert torch.equal(args[3], recv_weights[:3])
    assert torch.equal(args[4], recv_ids[:3])
    assert args[5] is expert_mask
    assert torch.equal(kwargs["a1_scale"], recv_scale[:3])
    assert kwargs["num_local_tokens"] is recv_count
    assert kwargs["gate_mode"] == "interleave"

    combine = calls["combine"]
    assert combine[0] is fused_output
    assert combine[1] is None
    assert combine[2].dtype == torch.int32
    assert combine[3:] == (96, 8)
    assert torch.equal(result, combined)


def test_wideep_method_installs_modular_kernel_and_sentinel_mask(monkeypatch):
    from atom.model_ops.fused_moe import wideep_experts
    from atom.model_ops.fused_moe.modular_kernel import FusedMoEModularKernel

    method = object.__new__(moe_mod.WideEpMxfp4MoEMethod)
    method.moe = SimpleNamespace(
        num_experts=16,
        num_local_experts=1,
        experts_per_token=2,
        max_num_tokens=32,
    )
    method.hidden_size = 4
    method.intermediate_size = 8
    method.topk_indices_dtype = None
    method.fused_experts = None
    quant_config = object()
    method.get_fused_moe_quant_config = lambda _layer: quant_config

    class FakePrepareFinalize:
        def topk_indices_dtype(self):
            return torch.int32

    prepare_finalize = FakePrepareFinalize()
    monkeypatch.setattr(
        wideep_experts,
        "make_wideep_prepare_finalize",
        lambda **_kwargs: prepare_finalize,
    )
    layer = SimpleNamespace(expert_mask=torch.ones(16, dtype=torch.int32))

    method.init_prepare_finalize(layer)

    assert method.topk_indices_dtype == torch.int32
    assert layer.expert_mask.tolist() == [1] * 16 + [0]
    assert isinstance(method.fused_experts, FusedMoEModularKernel)
    assert method.fused_experts.prepare_finalize is prepare_finalize
    assert method.fused_experts.quant_config is quant_config

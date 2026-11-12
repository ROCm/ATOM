# SPDX-License-Identifier: MIT
"""Exercise real MoE entry points with only the GPU/collective boundaries mocked."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from conftest import atom_config_double

from atom.moe_config import (
    MoEBackendConfig,
    MoEParallelMapping,
    resolve_moe_execution_plan,
)


def _plan(backend="standard", all2all="none"):
    mapping = MoEParallelMapping.make(
        tp_size=1, dp_size=2, local_dp_size=2, enable_ep=True, flatten_dp=True
    )
    return resolve_moe_execution_plan(
        mapping,
        MoEBackendConfig(backend=backend, all2all_backend=all2all, mori_available=True),
    )


@pytest.mark.parametrize(
    "backend, all2all, gather",
    [("standard", "none", True), ("standard", "auto", False), ("mega", "none", False)],
)
def test_forward_uses_plan_for_dp_collectives(
    monkeypatch, atom_modules, backend, all2all, gather
):
    moe = atom_modules.moe
    plan = _plan(backend, all2all)
    x = torch.arange(12).reshape(3, 4).float()
    w = torch.ones(3, 1)
    ids = torch.zeros(3, 1, dtype=torch.int32)
    run = Mock(side_effect=lambda hidden, *_: hidden * 2)
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(execution_plan=plan, dp_logical_ratio=1),
        quant_method=SimpleNamespace(
            select_experts=Mock(return_value=(w, ids)), routed_rows_wire=None
        ),
        _run_experts=run,
        reduce_results=True,
    )
    # A plugin may flatten DP without the native DP-attention flag. The plan
    # must still gather; conversely Mega must always receive only local rows.
    monkeypatch.setattr(
        moe,
        "get_current_atom_config",
        lambda: atom_config_double(enable_dp_attention=False),
    )
    ctx = SimpleNamespace(context=SimpleNamespace(running_tokens_are_unified=True))
    monkeypatch.setattr(moe, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(moe, "get_dp_group", lambda: object())
    tp_reduce = Mock(side_effect=lambda out, **_: out)
    monkeypatch.setattr(
        moe, "get_tp_group", lambda: SimpleNamespace(all_reduce=tp_reduce)
    )
    monkeypatch.setattr(moe, "tbo_active", lambda: False)
    gather_rows = Mock(
        side_effect=lambda rows, *_: (tuple(t.repeat(2, 1) for t in rows), 3, [3, 3])
    )
    scatter = Mock(side_effect=lambda out, _: out[:3])
    monkeypatch.setattr(moe, "dp_gather_routed_rows", gather_rows)
    monkeypatch.setattr(moe, "reduce_scatter_with_unpadding", scatter)
    monkeypatch.setattr(moe, "reassign_gathered_shared_ids", Mock())

    out = moe.FusedMoE.forward_impl_graph(layer, x, torch.empty(3, 8))

    torch.testing.assert_close(out, x * 2)
    assert run.call_args.args[0].shape[0] == (6 if gather else 3)
    assert run.call_args.args[-1] is gather
    assert gather_rows.call_count == scatter.call_count == int(gather)
    assert tp_reduce.call_count == int(gather)


def test_mega_skips_standard_prepare_finalize(monkeypatch, atom_modules):
    moe = atom_modules.moe
    build = Mock(
        side_effect=AssertionError("Mega must not allocate modular communication")
    )
    monkeypatch.setattr(moe.FusedMoEMethodBase, "_maybe_make_prepare_finalize", build)
    method = SimpleNamespace(
        moe=SimpleNamespace(
            moe_parallel_config=SimpleNamespace(execution_plan=_plan("mega"))
        )
    )
    assert moe.FusedMoEMethodBase.maybe_make_prepare_finalize(method) is None
    build.assert_not_called()


def test_parallel_facade_cannot_diverge_from_its_plan(atom_modules):
    plan = _plan()
    parallel = atom_modules.moe.FusedMoEParallelConfig(
        **asdict(plan.mapping), requested_all2all_backend="none", execution_plan=plan
    )
    for overrides in (
        {"ep_rank": 1},
        {"low_latency": True},
        {"requested_all2all_backend": "rccl"},
    ):
        with pytest.raises(ValueError, match="must agree with its execution plan"):
            replace(parallel, **overrides)


@pytest.mark.parametrize(
    "backend, expected", [("standard", "standard"), ("mega", "mega")]
)
def test_weight_method_factory_consumes_resolved_plan(
    monkeypatch, atom_modules, backend, expected
):
    moe = atom_modules.moe
    standard = Mock(return_value="standard")
    mega = Mock(return_value="mega")
    monkeypatch.setattr(moe, "Mxfp4MoEMethod", standard)
    monkeypatch.setattr(moe, "MegaMxfp4MoEMethod", mega)
    config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(execution_plan=_plan(backend))
    )
    assert moe._make_mxfp4_moe_method(object(), config) == expected
    assert standard.call_count == int(backend == "standard")
    assert mega.call_count == int(backend == "mega")


def test_parallel_facade_consumes_plan_without_runtime_resolution(
    monkeypatch, atom_modules
):
    moe = atom_modules.moe
    plan = _plan()
    refuse = Mock(side_effect=AssertionError("Layers must consume the saved plan"))
    monkeypatch.setattr(moe, "resolve_moe_execution_plan", refuse)
    monkeypatch.setattr(moe, "get_tp_group", refuse)
    monkeypatch.setattr(moe, "get_dp_group", refuse)
    parallel = moe.FusedMoEParallelConfig.from_plan(plan)
    assert parallel.execution_plan is plan
    assert parallel.ep_size == 2
    refuse.assert_not_called()


@pytest.mark.parametrize("transport", ["auto", "rccl"])
def test_routed_combine_needs_no_outer_pcp_or_tp_reduction(
    monkeypatch, atom_modules, transport
):
    moe = atom_modules.moe
    mapping = MoEParallelMapping.make(
        tp_size=1,
        dp_size=2,
        local_dp_size=2,
        enable_ep=True,
        flatten_dp=True,
        pcp_size=2,
        pcp_rank=1,
        merge_pcp=True,
    )
    plan = resolve_moe_execution_plan(
        mapping, MoEBackendConfig(all2all_backend=transport, mori_available=True)
    )
    assert not plan.needs_pcp_collectives and not plan.needs_tp_all_reduce
    x = torch.randn(3, 4)
    w, ids = torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int32)
    method = SimpleNamespace(
        routed_rows_wire=None,
        place_experts=Mock(
            side_effect=lambda layer, weights, ids, gathered: (weights, ids)
        ),
        apply=Mock(return_value=x * 2),
    )
    layer = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(execution_plan=plan),
        quant_method=method,
        global_num_experts=8,
        expert_map=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="experts",
    )
    monkeypatch.setattr(
        moe,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False, pcp_split=True)
        ),
    )
    monkeypatch.setattr(moe, "pass_scheduled_tokens_gpu", lambda _: None)
    gather = Mock(side_effect=AssertionError("dispatch already owns PCP rows"))
    monkeypatch.setattr(moe, "all_gatherv", gather)
    scatter = Mock(side_effect=AssertionError("combine already returns local rows"))
    monkeypatch.setattr(moe, "pcp_reduce_scatter", scatter)
    monkeypatch.setattr(moe, "pcp_all_reduce", scatter)
    out = moe.FusedMoE._run_experts(layer, x, w, ids)
    torch.testing.assert_close(out, x * 2)
    assert method.apply.call_args.kwargs["x"] is x
    gather.assert_not_called()
    scatter.assert_not_called()


def test_shared_expert_gate_uses_supplied_plan(monkeypatch, atom_modules):
    from atom.model_ops import topK

    quant = SimpleNamespace(exclude_layers=[], quant_dtype=None)
    config = atom_config_double(
        parallel_config=SimpleNamespace(data_parallel_size=2),
        enable_dp_attention=True,
        quant_config=quant,
    )
    monkeypatch.setattr(topK, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(topK.envs, "ATOM_FUSE_SHARED_EXPERT", False)
    # The current request disagrees with the supplied, frozen layer plan.
    config.moe_all2all_backend = "rccl"
    plan = _plan(all2all="auto")
    assert not topK.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
        quant, execution_plan=plan
    )
    assert topK.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
        quant, execution_plan=_plan(all2all="rccl")
    )

# SPDX-License-Identifier: MIT
"""One reduction, destinations and input views through production MoE paths."""

from types import SimpleNamespace

import pytest
import test_kimi_k3_moe_front_merge as front_tests
import torch
from torch.library import Library

make_block = front_tests.make_block
modules = front_tests.modules


class _Method:
    apply_takes_out = False

    def __init__(self, config):
        self.calls = []

    def create_weights(self, **kwargs):
        pass

    def select_experts(self, layer, x, logits):
        return logits, logits.long()

    def place_experts(self, layer, weights, ids, gathered):
        return weights, ids

    def apply(self, *, x, topk_weights, **kwargs):
        self.calls.append((x, kwargs))
        y = x + topk_weights.sum(-1, keepdim=True)
        return y if kwargs.get("out") is None else kwargs["out"].copy_(y)


@pytest.fixture
def real_experts(modules, monkeypatch):
    """Construct FusedMoE normally; substitute only quantized kernel execution."""
    moe = modules.moe
    config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        fake_eplb=False,
        enable_dp_attention=False,
        max_num_batched_tokens=128,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(moe, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe, "UnquantizedFusedMoEMethod", _Method)
    monkeypatch.setattr(
        moe,
        "is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        moe.FusedMoEParallelConfig,
        "make",
        lambda tp, dp, cfg: moe.FusedMoEParallelConfig(tp, dp, 1, 0, 0, 0, False, 1),
    )
    step = SimpleNamespace(is_dummy_run=False, pcp_split=False)
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=step)
    )
    monkeypatch.setattr(moe, "pass_scheduled_tokens_gpu", lambda rows: None)
    layer = moe.FusedMoE(6, 2, 8, 16, tp_size=2, dp_size=1, prefix="cpu.experts")
    return layer


@pytest.mark.parametrize("tokens", [1, 16, 128])
@pytest.mark.parametrize("tp", [1, 2])
def test_one_reduction_preserves_nonlinear_tail(
    make_block, real_experts, modules, monkeypatch, tokens, tp
):
    block = make_block(tp=tp)
    block.load()
    block.moe.experts = real_experts
    block.moe.routed_expert_norm.fn = lambda x: (x.float().square() / 64 + 1).bfloat16()
    reductions = []

    def reduce(x):
        reductions.append(x.clone())
        return x * 2

    monkeypatch.setattr(modules.kimi, "tensor_model_parallel_all_reduce", reduce)
    x = torch.arange(tokens * 16).reshape(tokens, 16).remainder(3).bfloat16()
    actual = block.moe(x)
    one = list(reductions)
    reductions.clear()
    monkeypatch.setattr(
        block.moe.shared_experts.down_proj, "supports_out", lambda: False
    )
    expected = block.moe(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if tp == 1:
        assert not one and not reductions
    else:
        assert len(one) == 1 and len(reductions) == 2
        torch.testing.assert_close(
            one[0], torch.cat([x.flatten() for x in reductions]), rtol=0, atol=0
        )


@pytest.mark.parametrize("takes_out", [False, True])
@pytest.mark.parametrize("path", ["plain", "repeated", "dp", "sp"])
def test_destination_handles_direct_and_fallback_results(
    real_experts, modules, monkeypatch, takes_out, path
):
    layer = real_experts
    layer.quant_method.apply_takes_out = takes_out
    if path == "repeated":
        layer.moe_parallel_config.dp_logical_ratio = 2
    elif path == "dp":
        monkeypatch.setattr(
            modules.moe, "get_dp_group", lambda: SimpleNamespace(world_size=2)
        )
        layer.forward_impl_graph = lambda x, logits, **kwargs: x * 3
    elif path == "sp":
        layer.sp_mega_moe = SimpleNamespace(
            runs=lambda rows: True, forward=lambda layer, x, logits: x * 4
        )
    x, logits = torch.ones(3, 8).bfloat16(), torch.ones(3, 2).bfloat16()
    expected = layer(x, logits)
    out = torch.full_like(x, float("nan"))
    assert layer(x, logits, out=out) is out
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    if path in ("plain", "repeated"):
        assert layer.quant_method.calls[-1][1].get("out") is (
            out if takes_out and path == "plain" else None
        )


@pytest.mark.parametrize("bad", ["shape", "dtype", "strided", "input", "router"])
def test_destination_rejects_mismatch_and_overlap(real_experts, bad):
    x = torch.ones(3, 8).bfloat16()
    logits = torch.ones(3, 8).bfloat16()
    out = {
        "shape": torch.empty(3, 9).bfloat16(),
        "dtype": torch.empty(3, 8),
        "strided": torch.empty(8, 3).bfloat16().T,
        "input": x,
        "router": logits,
    }[bad]
    with pytest.raises(ValueError, match="MoE output"):
        real_experts(x, logits, out=out)
    assert not real_experts.quant_method.calls


_LIB = Library("atom_kimi_moe_output_test", "FRAGMENT")


@pytest.mark.parametrize("view", [False, True])
def test_destination_op_survives_aot_functionalization(
    real_experts, modules, monkeypatch, view
):
    from atom.utils.custom_register import direct_register_custom_op

    moe = modules.moe
    schema = torch.ops.aiter.moe_forward_into_output.default._schema
    mutates = [
        arg.name
        for arg in schema.arguments
        if arg.alias_info and arg.alias_info.is_write
    ]
    assert mutates == ["out"]
    op = direct_register_custom_op(
        "into_output",
        moe.moe_forward_into_output,
        mutates,
        fake_impl=moe.moe_forward_into_output_fake,
        target_lib=_LIB,
        dispatch_key="CPU",
    )
    monkeypatch.setattr(moe, "moe_forward_into_output_op", op)
    x = torch.ones(3, 24 if view else 8).bfloat16()
    logits = torch.ones(3, 2).bfloat16()
    seen = _direct_mxfp4(real_experts, modules, monkeypatch) if view else None

    def run(x, logits):
        if view:
            x = x[:, 8:16]
        out = torch.empty_like(x)
        return real_experts(x, logits, out=out) + 1

    compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
    try:
        torch.testing.assert_close(compiled(x, logits), run(x, logits), rtol=0, atol=0)
        if view:
            assert (
                seen[0][0].untyped_storage().data_ptr()
                == x.untyped_storage().data_ptr()
            )
    finally:
        torch._dynamo.reset()


def _direct_mxfp4(layer, modules, monkeypatch):
    """Keep the production MXFP4 apply method and observe its external kernel."""
    moe = modules.moe
    method = object.__new__(moe.Mxfp4MoEMethod)
    method.use_triton = method.use_triton_decode = method.use_triton_ep = False
    method.use_flydsl_herd = method.is_guinterleave = method.fp8_activations = False
    method.fused_experts = None
    method.quant_type = moe.QuantType.per_1x32
    method.select_experts = lambda layer, x, logits: (logits, logits.long())
    method.place_experts = lambda layer, weights, ids, gathered: (weights, ids)
    layer.quant_method = method
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale",
        "w2_weight_scale",
        "w13_bias",
        "w2_bias",
    ):
        setattr(layer, name, None)
    seen = []

    def fused(x, w1, w2, weights, ids, **kwargs):
        seen.append((x, kwargs))
        value = x + weights.sum(-1, keepdim=True)
        out = kwargs.get("output")
        return value if out is None else out.copy_(value)

    monkeypatch.setattr(moe, "fused_moe", fused)
    return seen


@pytest.mark.parametrize(
    "layout", ["aligned", "misaligned", "row_stride", "columns", "overlap"]
)
def test_only_supported_input_views_reach_direct_mxfp4(
    real_experts, modules, monkeypatch, layout
):
    seen = _direct_mxfp4(real_experts, modules, monkeypatch)
    owner = torch.arange(3 * 24).reshape(3, 24).bfloat16()
    x = {
        "aligned": owner[:, 8:16],
        "misaligned": owner[:, 1:9],
        "row_stride": owner.flatten()[:69].view(3, 23)[:, 8:16],
        "columns": owner[:, :16:2],
        "overlap": owner.as_strided((3, 8), (4, 1)),
    }[layout]
    logits = torch.ones(3, 2).bfloat16()
    out = torch.empty(3, 8).bfloat16()
    actual = real_experts(x, logits, out=out)
    torch.testing.assert_close(actual, x + 2, rtol=0, atol=0)
    assert seen[0][1]["output"] is out
    assert (seen[0][0] is x) == (layout == "aligned")
    if layout != "aligned":
        assert seen[0][0].is_contiguous()


def test_merged_front_view_reaches_production_mxfp4(
    make_block, real_experts, modules, monkeypatch
):
    block = make_block(tp=2, experts=8)
    block.load()
    block.moe.experts = real_experts
    seen = _direct_mxfp4(real_experts, modules, monkeypatch)
    x = torch.ones(16, 16).bfloat16()
    actual = block.moe(x)
    latent = seen[0][0]
    assert not latent.is_contiguous()
    assert (
        latent.untyped_storage().data_ptr()
        == block.seen["gate_up"].untyped_storage().data_ptr()
    )
    block.moe.front_proj.merged_weight = None
    torch.testing.assert_close(actual, block.moe(x), rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["modular", "mega"])
def test_other_mxfp4_receivers_keep_contiguous_input(
    real_experts, modules, monkeypatch, backend
):
    _direct_mxfp4(real_experts, modules, monkeypatch)
    method = real_experts.quant_method
    method.hidden_pad = method.intermediate_pad = 0
    if backend == "mega":
        method.__class__ = modules.moe.MegaMxfp4MoEMethod
    inputs = []

    def experts(hidden_states, **kwargs):
        inputs.append(hidden_states)
        return hidden_states * 3

    method.fused_experts = experts
    x = torch.ones(3, 24).bfloat16()[:, 8:16]
    out = torch.empty(3, 8).bfloat16()
    real_experts(x, torch.ones(3, 2).bfloat16(), out=out)
    torch.testing.assert_close(out, x * 3, rtol=0, atol=0)
    assert inputs[0] is not x and inputs[0].is_contiguous()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The caller returned by `direct_register_custom_op` runs the op's Python
implementation directly in eager mode, and goes through torch.ops wherever a
tracer or a dispatch mode handles the op as an opaque call. FusedMoE.forward
calls moe_forward through such a caller, and the FP8 MoE methods call
rocm_aiter_fused_moe, nested inside it, through another.

Which path ran is read off the profiler: a dispatched op is recorded as an op,
a direct call as a user annotation of the same name. The profiler is not a
dispatch mode, so it does not change the path.
"""

import importlib.util
import inspect
from types import SimpleNamespace

import pytest
import torch
from torch.library import Library
from torch.utils._python_dispatch import TorchDispatchMode

from atom.utils.custom_register import direct_register_custom_op, eager_direct_call

_LIB = Library("atom_moe_eager_test", "FRAGMENT")
_PY_CALLS = []


def _scale(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    _PY_CALLS.append(factor)
    return x * factor


def _scale_fake(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    return torch.empty_like(x)


# Without a fake impl, Dynamo traces into the op's Python kernel.
_scale_op = direct_register_custom_op(
    "scale", _scale, [], fake_impl=_scale_fake, target_lib=_LIB, dispatch_key="CPU"
)


class Passthrough(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **(kwargs or {}))


def _paths(fn, name):
    """`fn()`'s result and how `name` ran: "op" (dispatched through torch.ops)
    and/or "direct" (the user annotation of a direct call)."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        out = fn()
    return out, {
        "direct" if e.is_user_annotation else "op" for e in p.events() if e.name == name
    }


def test_caller_keeps_the_signature_and_registers_once():
    assert inspect.signature(_scale_op) == inspect.signature(_scale)
    again = direct_register_custom_op(
        "scale", _scale, [], target_lib=_LIB, dispatch_key="CPU"
    )
    x = torch.ones(2)
    assert torch.equal(again(x, 3.0), torch.full((2,), 3.0))
    assert torch.equal(torch.ops.atom_moe_eager_test.scale(x), torch.full((2,), 2.0))
    with pytest.raises(TypeError):
        _scale_op(x, 2.0, 1)


def test_caller_runs_directly_in_eager_and_dispatches_otherwise():
    x = torch.arange(4.0)
    name = "atom_moe_eager_test::scale"
    _PY_CALLS.clear()
    out, ran = _paths(lambda: _scale_op(x, factor=3.0), name)
    assert ran == {"direct"} and _PY_CALLS == [3.0]
    assert torch.equal(out, x * 3)

    _PY_CALLS.clear()
    assert torch.equal(_scale_op(x, 3.0), x * 3) and _PY_CALLS == [3.0]

    with Passthrough():
        ref, ran = _paths(lambda: _scale_op(x, factor=3.0), name)
    assert ran == {"op"} and torch.equal(ref, out)


def test_caller_dispatches_for_a_non_tensor_first_argument():
    def first(xs: list[torch.Tensor]) -> torch.Tensor:
        return xs[0].clone()

    caller = direct_register_custom_op(
        "first", first, [], target_lib=_LIB, dispatch_key="CPU"
    )
    out, ran = _paths(lambda: caller([torch.ones(2)]), "atom_moe_eager_test::first")
    assert ran == {"op"} and torch.equal(out, torch.ones(2))


def test_caller_stays_an_op_under_torch_compile():
    targets = []

    def recording_backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    x = torch.arange(4.0)
    compiled = torch.compile(lambda t: _scale_op(t, 3.0), backend=recording_backend)
    try:
        out = compiled(x)
    finally:
        torch._dynamo.reset()
    assert any("atom_moe_eager_test.scale" in t for t in targets), targets
    assert torch.equal(out, x * 3)


def test_eager_direct_call_only_for_real_eager_tensors():
    from torch._subclasses.fake_tensor import FakeTensorMode

    x = torch.zeros(2)
    assert eager_direct_call(x)
    with Passthrough():
        assert not eager_direct_call(x)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(x)
        assert not eager_direct_call(fake)
    assert not eager_direct_call(fake)  # a fake tensor outside its mode


needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("aiter") is None,
    reason="the MoE ops are registered for CUDA and run AITER kernels",
)


@pytest.fixture
def moe_mod():
    from atom.model_ops import moe

    return moe


def _layer(moe_mod, monkeypatch, forward_impl):
    """A FusedMoE whose forward_impl is `forward_impl`, registered under its name."""
    # The plugin-mode lazy wrapper subclasses FusedMoE under the same name; take
    # the class that defines forward.
    cls = next(c for c in moe_mod.FusedMoE.__mro__ if "forward" in vars(c))
    assert cls.__name__ == "FusedMoE"
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    layer.balance_router_logits = None
    layer.layer_name = "model.layers.0.mlp.experts"
    layer.forward_impl = forward_impl
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={layer.layer_name: layer}
        )
    )
    monkeypatch.setattr(moe_mod, "get_current_atom_config", lambda: config)
    return layer


@needs_gpu
def test_fused_moe_forward_calls_moe_forward_directly(moe_mod, monkeypatch):
    calls = []

    def forward_impl(hidden_states, router_logits):
        calls.append(router_logits)
        return hidden_states + router_logits.sum(-1, keepdim=True)

    layer = _layer(moe_mod, monkeypatch, forward_impl)
    x = torch.randn(4, 8, device="cuda")
    logits = torch.randn(4, 3, device="cuda")
    out, ran = _paths(lambda: layer(x, logits), "aiter::moe_forward")
    assert ran == {"direct"}
    assert len(calls) == 1 and calls[0] is logits
    torch.testing.assert_close(out, x + logits.sum(-1, keepdim=True), rtol=0, atol=0)

    ref, ran = _paths(
        lambda: torch.ops.aiter.moe_forward(x, logits, layer.layer_name),
        "aiter::moe_forward",
    )
    assert ran == {"op"}
    assert torch.equal(out, ref)


@needs_gpu
def test_fused_moe_forward_stays_an_op_under_torch_compile(moe_mod, monkeypatch):
    layer = _layer(moe_mod, monkeypatch, lambda h, r: h * 3)
    targets = []

    def recording_backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    x = torch.randn(4, 8, device="cuda")
    compiled = torch.compile(lambda h, r: layer(h, r), backend=recording_backend)
    try:
        out = compiled(x, torch.randn(4, 3, device="cuda"))
    finally:
        torch._dynamo.reset()
    assert any("moe_forward" in t for t in targets), targets
    assert torch.equal(out, x * 3)


def _fp8_moe_inputs(M=300, E=64, D=2048, inter=512, topk=8):
    from aiter import dtypes
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(0)

    def weight(e, n, k):
        w = (torch.randn(e, n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16)
        scale = w.abs().amax(-1, keepdim=True).float() / torch.finfo(dtypes.fp8).max
        q = shuffle_weight((w / scale.to(w.dtype)).to(dtypes.fp8), layout=(16, 16))
        q.is_shuffled = True
        return q, scale

    w13, w13_s = weight(E, 2 * inter, D)
    w2, w2_s = weight(E, D, inter)
    x = torch.randn(M, D, device="cuda", generator=g).to(torch.bfloat16)
    weights, ids = torch.topk(
        torch.randn(M, E, device="cuda", generator=g).softmax(-1), topk, dim=-1
    )
    # One nonzero top-k weight per token: the atomic adds of the other routes
    # are exact zeros, so the output is reproducible bit for bit.
    weights[:, 1:] = 0
    return x, w13, w2, weights, ids.to(torch.int32), w13_s, w2_s


@needs_gpu
def test_rocm_aiter_fused_moe_direct_call_matches_the_op(moe_mod):
    from aiter import ActivationType, QuantType

    x, w13, w2, weights, ids, w13_s, w2_s = _fp8_moe_inputs()

    def run(fused_moe):
        return fused_moe(
            x,
            w13,
            w2,
            weights,
            ids,
            expert_mask=None,
            activation=ActivationType.Silu.value,
            quant_type=QuantType.per_Token.value,
            w1_scale=w13_s,
            w2_scale=w2_s,
            a1_scale=None,
            a2_scale=None,
            doweight_stage1=False,
            gate_mode="separated",
            swiglu_limit=0.0,
            scheduled_tokens_gpu=None,
        )

    name = "aiter::rocm_aiter_fused_moe"
    out, ran = _paths(lambda: run(moe_mod.rocm_aiter_fused_moe_op), name)
    assert ran == {"direct"}
    ref, ran = _paths(lambda: run(torch.ops.aiter.rocm_aiter_fused_moe), name)
    assert ran == {"op"}
    torch.cuda.synchronize()
    assert torch.equal(out, ref)

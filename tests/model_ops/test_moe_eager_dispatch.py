# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FusedMoE runs its two custom ops (moe_forward, rocm_aiter_fused_moe) without
the torch.ops dispatch in eager mode, and through it wherever a tracer or a
dispatch mode handles them as opaque calls.

Which path ran is read off the profiler: it records an op only when the op is
dispatched, and it is not a dispatch mode, so it does not change the path.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch.library import Library

from atom.utils.custom_register import direct_register_custom_op, eager_direct_call

_LIB = Library("atom_moe_eager_test", "FRAGMENT")


def _double(x: torch.Tensor) -> torch.Tensor:
    return x * 2


def test_register_reports_whether_it_registered():
    kw = {"target_lib": _LIB, "dispatch_key": "CPU"}
    assert direct_register_custom_op("double", _double, [], **kw)
    assert not direct_register_custom_op("double", _double, [], **kw)
    assert torch.equal(
        torch.ops.atom_moe_eager_test.double(torch.ones(2)), torch.full((2,), 2.0)
    )


def test_eager_direct_call_only_for_real_eager_tensors(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import TorchDispatchMode

    x = torch.zeros(2)
    assert eager_direct_call(x)

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with Passthrough():
        assert not eager_direct_call(x)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(x)
        assert not eager_direct_call(fake)
    assert not eager_direct_call(fake)  # a fake tensor outside its mode
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not eager_direct_call(x)


needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("aiter") is None,
    reason="the MoE ops are registered for CUDA and run AITER kernels",
)


@pytest.fixture
def moe_mod():
    from atom.model_ops import moe

    return moe


def _dispatched(fn):
    """`fn()`'s result and the aiter ops it dispatched through torch.ops."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        out = fn()
    names = {e.name for e in p.events()}
    return out, {
        n for n in ("aiter::moe_forward", "aiter::rocm_aiter_fused_moe") if n in names
    }


def _layer(moe_mod, monkeypatch, forward_impl):
    """A FusedMoE whose forward_impl is `forward_impl`, registered under its name."""
    cls = moe_mod.FusedMoE.__mro__[1]  # the class under the plugin-mode lazy wrapper
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
    out, ops = _dispatched(lambda: layer(x, logits))
    assert ops == set()
    assert len(calls) == 1 and calls[0] is logits
    torch.testing.assert_close(out, x + logits.sum(-1, keepdim=True), rtol=0, atol=0)

    monkeypatch.setattr(moe_mod, "_MOE_FORWARD_IS_OURS", False)
    ref, ops = _dispatched(lambda: layer(x, logits))
    assert ops == {"aiter::moe_forward"}
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
def test_rocm_aiter_fused_moe_direct_call_matches_the_op(moe_mod, monkeypatch):
    from aiter import ActivationType, QuantType

    x, w13, w2, weights, ids, w13_s, w2_s = _fp8_moe_inputs()

    def run():
        return moe_mod.rocm_aiter_fused_moe(
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

    out, ops = _dispatched(run)
    assert "aiter::rocm_aiter_fused_moe" not in ops
    monkeypatch.setattr(moe_mod, "_ROCM_AITER_FUSED_MOE_IS_OURS", False)
    ref, ops = _dispatched(run)
    assert "aiter::rocm_aiter_fused_moe" in ops
    torch.cuda.synchronize()
    assert torch.equal(out, ref)

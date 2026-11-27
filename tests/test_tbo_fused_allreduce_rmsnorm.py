# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The fused TP all-reduce + RMSNorm (+ FP8 quant) under pure-TP TBO: which
configurations take the TBO-aware ops (`communication_op`), which steps hand
the all-reduce to the comm stream and which run aiter's fused kernel
(`module_dispatch_ops`), and what a split step reduces into and normalizes.
CPU only: the kernels, the communicator and the TBO hooks are faked."""

import importlib
import sys
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

ub = pytest.importorskip("atom.utils.tbo.ubatching")

TP = 4
EPS = 1e-6
FP8 = torch.float8_e4m3fnuz
NORM = "tbo_fused_allreduce_rmsnorm"
QUANT = "tbo_fused_allreduce_rmsnorm_quant"


@pytest.fixture(scope="module")
def mods():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            ln=load("atom.model_ops.layernorm"),
            mdo=load("atom.model_ops.module_dispatch_ops"),
            co=load("atom.model_ops.communication_op"),
        )


def _inputs():
    return torch.randn(3, 256), torch.randn(3, 256), torch.randn(256)


def _same(args, expected):
    """Tensors by identity, anything else by value."""
    return len(args) == len(expected) and all(
        a is e if isinstance(e, torch.Tensor) else a == e
        for a, e in zip(args, expected)
    )


@pytest.fixture
def fused(mods, monkeypatch):
    """aiter's fused all-reduce + RMSNorm kernels, logged in `calls`; they
    return fixed tensors."""
    comm_op = importlib.import_module("aiter.dist.communication_op")
    calls = []
    norm_out = (torch.zeros(1), torch.zeros(1))
    quant_out = tuple(torch.zeros(1) for _ in range(4))

    def norm(x, residual, weight, eps, **kwargs):
        calls.append(("norm", (x, residual, weight, eps), kwargs))
        return norm_out

    def quant(x, residual, weight, eps, **kwargs):
        calls.append(("quant", (x, residual, weight, eps), kwargs))
        return quant_out if kwargs["emit_bf16"] else quant_out[:3]

    patch = monkeypatch.setattr
    patch(comm_op, "tensor_model_parallel_fused_allreduce_rmsnorm", norm, raising=False)
    patch(
        comm_op,
        "tensor_model_parallel_fused_allreduce_rmsnorm_quant",
        quant,
        raising=False,
    )
    return SimpleNamespace(calls=calls, norm_out=norm_out, quant_out=quant_out)


@pytest.fixture
def ops(mods, monkeypatch):
    """The TBO-aware ops as `torch.ops.aiter` resolves them, logged in `calls`;
    they return fixed tensors."""
    calls = []
    outs = {
        NORM: (torch.zeros(1), torch.zeros(1)),
        QUANT: tuple(torch.zeros(1) for _ in range(4)),
    }
    for name, out in outs.items():

        def op(*args, name=name, out=out):
            calls.append((name, args))
            return out

        monkeypatch.setattr(torch.ops.aiter, name, op)
    return SimpleNamespace(calls=calls, outs=outs)


@pytest.fixture
def deployment(mods, monkeypatch):
    """Sets the TP world size and the TBO / DP config the wrappers read."""
    parallel_state = importlib.import_module("aiter.dist.parallel_state")

    def deploy(tp=TP, enable_tbo=True, dp=1):
        group = SimpleNamespace(world_size=tp)
        monkeypatch.setattr(
            parallel_state, "get_tp_group", lambda: group, raising=False
        )
        config = SimpleNamespace(
            enable_tbo=enable_tbo,
            parallel_config=SimpleNamespace(data_parallel_size=dp),
        )
        monkeypatch.setattr("atom.config.get_current_atom_config", lambda: config)

    return deploy


@pytest.fixture
def kernels(mods, monkeypatch):
    """The norm and quant kernels a split step runs after the all-reduce,
    logged in order. Both norms write `x + residual` as the residual; the
    Gemma one normalizes to half of it, the plain one to a third."""
    calls = []

    def group_quant(*args, **kwargs):  # fused_qk_rmsnorm_group_quant
        calls.append(("group_quant", args, kwargs))
        x = args[2] if args else kwargs["q"]
        summed = x + kwargs["q_residual"]
        kwargs["q_res_out"].copy_(summed)
        if kwargs.get("q_out_unquantized") is not None:
            kwargs["q_out_unquantized"].copy_(summed / 2)

    def add_rms_norm(x, weight, residual, eps, dim):  # rmsnorm2d_fwd_with_add_
        calls.append(("add_rms_norm", (x, weight, residual, eps, dim)))
        return (x + residual) / 3, x + residual

    def get_hip_quant(quant_type):
        def quant(x, quant_dtype):
            calls.append(("quant", quant_type, x, quant_dtype))
            return x.sign().to(quant_dtype), x.abs().amax(-1, keepdim=True)

        return quant

    ln = mods.ln
    monkeypatch.setattr(ln, "fused_qk_rmsnorm_group_quant", group_quant)
    monkeypatch.setattr(ln, "rmsnorm2d_fwd_with_add_", add_rms_norm)
    monkeypatch.setattr(sys.modules["aiter"], "get_hip_quant", get_hip_quant)
    dtypes = importlib.import_module("aiter.utility.dtypes")
    monkeypatch.setattr(dtypes, "fp8", FP8, raising=False)
    return calls


@pytest.fixture
def split_step(monkeypatch):
    """Micro-batch 1 of a split step, `ATOM_TBO_TP_AR_MODE=overlap`. Logs, in
    order: the switches to comm and back to compute, every `torch.empty_like`
    with the stream current at that point, the collective and any hold."""
    log = []
    current = ["compute"]

    def switch_to(stream):
        def switch():
            log.append(stream)
            current[0] = stream

        return switch

    def all_reduce(x, out_tensor=None, stream=None):
        log.append(("all_reduce", x, out_tensor, stream, current[0]))
        out_tensor.copy_(TP * x)

    empty_like = torch.empty_like

    def logged_empty_like(t, **kwargs):
        out = empty_like(t, **kwargs)
        log.append(("alloc", out, current[0]))
        return out

    comm = SimpleNamespace(all_reduce=all_reduce, disabled=False)
    monkeypatch.setenv("ATOM_TBO_TP_AR_MODE", "overlap")
    monkeypatch.setattr(ub, "tbo_active", lambda: True)
    monkeypatch.setattr(ub, "tbo_current_ubatch_id", lambda: 1)
    monkeypatch.setattr(ub, "tbo_get_ubatch_tp_comm", {1: comm}.__getitem__)
    monkeypatch.setattr(ub, "tbo_get_comm_stream", lambda: "comm stream")
    monkeypatch.setattr(
        ub, "_tbo_yield_and_switch_from_compute_to_comm", switch_to("comm")
    )
    monkeypatch.setattr(ub, "tbo_switch_to_compute_sync", switch_to("compute"))
    ctx = SimpleNamespace(compute_stream="compute stream")
    monkeypatch.setattr(ub, "_get_current_tbo_context", lambda: ctx)
    monkeypatch.setattr(
        torch.Tensor, "record_stream", lambda t, s: log.append(("held", t, s))
    )
    monkeypatch.setattr(torch, "empty_like", logged_empty_like)
    return SimpleNamespace(log=log, comm=comm)


def _handed_off(log, x):
    """The tensor `x` was reduced into: allocated on the compute stream before
    the hand-off, so the partner's next hand-off is not given it while the
    norm still reads it, and written on the comm stream. The comm phase
    returns nothing, so nothing is held."""
    (alloc, reduced, allocated_on), comm, collective, compute = log[:4]
    assert (alloc, allocated_on, comm, compute) == (
        "alloc",
        "compute",
        "comm",
        "compute",
    )
    assert _same(collective, ("all_reduce", x, reduced, "comm stream", "comm"))
    assert not [e for e in log if isinstance(e, tuple) and e[0] == "held"]
    torch.testing.assert_close(reduced, TP * x)
    return reduced


def _views(a, b):
    return a.data_ptr() == b.data_ptr() and a.shape == b.shape


GATES = [
    pytest.param({}, True, id="pure TP TBO"),
    pytest.param({"tp": 1}, False, id="TP1"),
    pytest.param({"enable_tbo": False}, False, id="TBO off"),
    pytest.param({"dp": 2}, False, id="TBO with DP"),
]


@pytest.mark.parametrize("gate, routed", GATES)
def test_only_pure_tp_tbo_takes_the_ops(mods, deployment, ops, fused, gate, routed):
    deployment(**gate)
    x, residual, weight = _inputs()
    co = mods.co

    out = co.tensor_model_parallel_fused_allreduce_rmsnorm(
        x, residual, weight, EPS, gemma_norm=True
    )
    per_group = co.tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, quant_type="per_group", transpose_scale=True
    )
    per_token = co.tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, emit_bf16=True, gemma_norm=True
    )

    if routed:
        assert not fused.calls
        (norm, norm_args), (quant, group_args), (quant2, token_args) = ops.calls
        assert (norm, quant, quant2) == (NORM, QUANT, QUANT)
        assert _same(norm_args, (x, residual, weight, EPS, True))
        assert _same(
            group_args, (x, residual, weight, EPS, "per_group", 128, False, True, False)
        )
        assert _same(
            token_args, (x, residual, weight, EPS, "per_token", 128, True, False, True)
        )
        assert out is ops.outs[NORM]
        # aiter's arity: `(out, residual_out, scale)`, the bf16 4th if asked for
        assert _same(per_group, ops.outs[QUANT][:3])
        assert _same(per_token, ops.outs[QUANT])
    else:
        assert not ops.calls
        assert [c[0] for c in fused.calls] == ["norm", "quant", "quant"]
        assert out is fused.norm_out
        assert per_group is not None and len(per_group) == 3
        assert len(per_token) == 4
        _, args, kwargs = fused.calls[0]
        assert _same(args, (x, residual, weight, EPS))
        assert kwargs == {"gemma_norm": True}
        assert fused.calls[1][2]["quant_type"] == "per_group"
        assert fused.calls[1][2]["transpose_scale"] is True


def test_any_other_quant_type_keeps_the_fused_kernel(mods, deployment, ops, fused):
    deployment()
    x, residual, weight = _inputs()
    out = mods.co.tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, quant_type="mxfp4"
    )
    assert not ops.calls
    assert _same(out, fused.quant_out[:3])
    assert fused.calls[0][2]["quant_type"] == "mxfp4"


@pytest.mark.parametrize(
    "split, mode", [(False, "overlap"), (True, "inline")], ids=["unsplit", "inline"]
)
def test_without_overlap_the_ops_run_the_fused_kernels(
    mods, fused, monkeypatch, split, mode
):
    monkeypatch.setenv("ATOM_TBO_TP_AR_MODE", mode)
    monkeypatch.setattr(ub, "tbo_active", lambda: split)
    monkeypatch.setattr(ub, "tbo_on_comm", lambda *a, **k: pytest.fail("handed off"))
    x, residual, weight = _inputs()
    mdo = mods.mdo

    out = mdo.tbo_fused_allreduce_rmsnorm(x, residual, weight, EPS, True)
    assert out is fused.norm_out
    out = mdo.tbo_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, "per_group", 128, False, True, True
    )
    # no bf16 asked for: the 4th output is an empty placeholder
    assert _same(out[:3], fused.quant_out[:3]) and out[3].numel() == 0
    out = mdo.tbo_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, "per_token", 128, True, False, False
    )
    assert _same(out, fused.quant_out)

    for _, args, _ in fused.calls:
        assert _same(args, (x, residual, weight, EPS))
    assert [kwargs for *_, kwargs in fused.calls] == [
        {"gemma_norm": True},
        {
            "quant_type": "per_group",
            "group_size": 128,
            "emit_bf16": False,
            "transpose_scale": True,
            "gemma_norm": True,
        },
        {
            "quant_type": "per_token",
            "group_size": 128,
            "emit_bf16": True,
            "transpose_scale": False,
            "gemma_norm": False,
        },
    ]


@pytest.mark.parametrize("gemma_norm", [True, False], ids=["gemma", "plain"])
def test_a_split_step_norms_the_reduction(mods, kernels, split_step, gemma_norm):
    x, residual, weight = _inputs()
    out, residual_out = mods.mdo.tbo_fused_allreduce_rmsnorm(
        x, residual, weight, EPS, gemma_norm
    )
    reduced = _handed_off(split_step.log, x)

    (call,) = kernels
    if gemma_norm:
        name, args, kwargs = call
        assert (name, args) == ("group_quant", ())
        assert _views(kwargs["q"], reduced) and _views(kwargs["q_residual"], residual)
        assert kwargs["q_weight"] is weight and kwargs["q_epsilon"] == EPS
        assert kwargs["gemma_norm"] is True
        assert _views(out, kwargs["q_out_unquantized"])
        assert _views(residual_out, kwargs["q_res_out"])
    else:
        assert _same(call, ("add_rms_norm", (reduced, weight, residual, EPS, 256)))
    summed = TP * x + residual
    torch.testing.assert_close(residual_out, summed)
    torch.testing.assert_close(out, summed / (2 if gemma_norm else 3))


@pytest.mark.parametrize("emit_bf16", [False, True], ids=["fp8", "fp8 and bf16"])
@pytest.mark.parametrize("gemma_norm", [True, False], ids=["gemma", "plain"])
def test_a_split_per_token_quant_norms_then_quantizes(
    mods, kernels, split_step, gemma_norm, emit_bf16
):
    x, residual, weight = _inputs()
    out, residual_out, scale, bf16 = mods.mdo.tbo_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, "per_token", 128, emit_bf16, False, gemma_norm
    )
    _handed_off(split_step.log, x)

    norm, (name, quant_type, quantized, quant_dtype) = kernels
    assert norm[0] == ("group_quant" if gemma_norm else "add_rms_norm")
    assert name == "quant" and quant_type is mods.ln.QuantType.per_Token
    assert quant_dtype is FP8
    summed = TP * x + residual
    normed = summed / (2 if gemma_norm else 3)
    torch.testing.assert_close(quantized, normed)
    torch.testing.assert_close(out.float(), normed.sign())
    torch.testing.assert_close(scale, normed.abs().amax(-1, keepdim=True))
    torch.testing.assert_close(residual_out, summed)
    if emit_bf16:
        assert bf16 is quantized
    else:
        assert bf16.numel() == 0


@pytest.mark.parametrize("emit_bf16", [False, True], ids=["fp8", "fp8 and bf16"])
@pytest.mark.parametrize("transpose", [False, True], ids=["row-major", "column-major"])
@pytest.mark.parametrize("gemma_norm", [True, False], ids=["gemma", "plain"])
def test_a_split_per_group_quant_is_one_kernel(
    mods, kernels, split_step, gemma_norm, transpose, emit_bf16
):
    x, residual, weight = _inputs()
    out, residual_out, scale, bf16 = mods.mdo.tbo_fused_allreduce_rmsnorm_quant(
        x, residual, weight, EPS, "per_group", 128, emit_bf16, transpose, gemma_norm
    )
    reduced = _handed_off(split_step.log, x)

    ((name, args, kwargs),) = kernels
    assert name == "group_quant"
    assert _same(args, (out, scale, reduced, weight, EPS))
    assert kwargs["q_residual"] is residual and kwargs["q_res_out"] is residual_out
    assert kwargs["group_size"] == 128
    assert kwargs["transpose_scale"] is transpose
    assert kwargs["gemma_norm"] is gemma_norm
    assert out.dtype == FP8 and out.shape == x.shape
    assert scale.dtype == torch.float32 and scale.shape == (3, 2)
    # column-major: allocated (num_groups, M), viewed as (M, num_groups)
    assert (scale._base is not None and scale._base.shape == (2, 3)) is transpose
    if emit_bf16:
        assert bf16 is kwargs["q_out_unquantized"] and bf16.shape == x.shape
    else:
        assert kwargs["q_out_unquantized"] is None and bf16.numel() == 0
    torch.testing.assert_close(residual_out, TP * x + residual)


@pytest.mark.parametrize("quant", [False, True], ids=["norm", "quant"])
def test_a_split_step_refuses_a_disabled_communicator(mods, kernels, split_step, quant):
    # A disabled pynccl communicator returns without writing `out_tensor`.
    split_step.comm.disabled = True
    x, residual, weight = _inputs()
    with pytest.raises(RuntimeError, match="ATOM_TBO_TP_AR_MODE=inline"):
        if quant:
            mods.mdo.tbo_fused_allreduce_rmsnorm_quant(
                x, residual, weight, EPS, "per_token", 128, False, False, True
            )
        else:
            mods.mdo.tbo_fused_allreduce_rmsnorm(x, residual, weight, EPS, True)
    assert split_step.log == [] and kernels == []


@pytest.mark.parametrize(
    "quant_type, emit_bf16, transpose",
    [
        ("per_token", False, False),
        ("per_token", True, False),
        ("per_group", False, False),
        ("per_group", True, True),
    ],
)
def test_the_fakes_match_a_split_step(
    mods, kernels, split_step, quant_type, emit_bf16, transpose
):
    x, residual, weight = _inputs()
    mdo = mods.mdo
    args = (x, residual, weight, EPS, quant_type, 128, emit_bf16, transpose, True)
    pairs = [
        (
            mdo.tbo_fused_allreduce_rmsnorm_quant(*args),
            mdo._tbo_fused_allreduce_rmsnorm_quant_fake(*args),
        ),
        (
            mdo.tbo_fused_allreduce_rmsnorm(x, residual, weight, EPS, True),
            mdo._tbo_fused_allreduce_rmsnorm_fake(x, residual, weight, EPS, True),
        ),
    ]
    for real, fake in pairs:
        assert [(t.shape, t.dtype, t.stride()) for t in real] == [
            (t.shape, t.dtype, t.stride()) for t in fake
        ]


@pytest.mark.parametrize("transpose", [False, True])
def test_a_per_group_scale_has_the_fused_kernels_strides(
    mods, kernels, split_step, transpose
):
    # aiter's fused kernel, which unsplit steps run, returns a transposed scale
    # as torch.empty((num_groups, M)).transpose(0, 1).
    x, residual, weight = _inputs()
    m, n = x.shape
    strides = (1, m) if transpose else (n // 128, 1)
    mdo = mods.mdo
    args = (x, residual, weight, EPS, "per_group", 128, False, transpose, True)
    for scale in (
        mdo.tbo_fused_allreduce_rmsnorm_quant(*args)[2],
        mdo._tbo_fused_allreduce_rmsnorm_quant_fake(*args)[2],
    ):
        assert (scale.shape, scale.stride()) == ((m, n // 128), strides)


def test_every_fused_allreduce_norm_takes_the_wrapper(mods):
    # RMSNorm, GemmaRMSNorm and the MiniMax-M3 helpers call these names.
    for name in (
        "tensor_model_parallel_fused_allreduce_rmsnorm",
        "tensor_model_parallel_fused_allreduce_rmsnorm_quant",
    ):
        assert getattr(mods.ln, name) is getattr(mods.co, name)


@pytest.mark.parametrize(
    "quant, name", [("per_1x128", "per_group"), ("per_Token", "per_token")]
)
def test_rmsnorm_names_its_quant_type(mods, monkeypatch, quant, name):
    # The wrapper routes by name; comparing `QuantType` members graph-breaks
    # Dynamo.
    ln = mods.ln
    seen = {}

    def fused_quant(x, residual, weight, eps, **kwargs):
        seen.update(kwargs)
        return torch.zeros(1), torch.zeros(1), torch.zeros(1)

    monkeypatch.setattr(ln, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(
        ln, "tensor_model_parallel_fused_allreduce_rmsnorm_quant", fused_quant
    )
    layer = SimpleNamespace(quant_type=getattr(ln.QuantType, quant), quant_dtype=FP8)
    quant_config = SimpleNamespace(
        get_layer_quant_config=lambda prefix: layer, blockscale_e8m0_scale=False
    )
    norm = ln.RMSNorm(
        256, fused_allreduce=True, fused_quant=True, quant_config=quant_config
    )
    norm(torch.randn(3, 256), torch.randn(3, 256))
    assert seen["quant_type"] == name

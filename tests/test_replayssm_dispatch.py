# SPDX-License-Identifier: MIT
"""ReplaySSM backend routing, state ownership and fork/verify regression tests."""

import builtins

import pytest
import torch
import torch.nn.functional as F
from conftest import gpu_stack_import

dispatch = gpu_stack_import("atom.model_ops.fla_ops.replayssm_dispatch")


def _args(heads=12, window=5, device="cpu", width=128):
    tokens, slots, cap = 2 * window, 3, 16
    args = {
        name: torch.randn(1, tokens, heads, width, device=device, dtype=torch.bfloat16)
        for name in ("q", "k", "v", "a")
    }
    args.update(
        # Match the model's non-contiguous beta slice.
        b=torch.randn(1, tokens, heads + 128, device=device, dtype=torch.bfloat16)[
            ..., :heads
        ],
        A_log=torch.zeros(heads, device=device),
        dt_bias=torch.zeros(heads * width, device=device),
        ckpt=torch.randn(slots, heads, width, width, device=device) * 0.1,
        buf_k=F.normalize(
            torch.randn(slots, heads, cap, width, device=device), dim=-1
        ).bfloat16(),
        buf_u=torch.randn(slots, heads, cap, width, device=device).bfloat16() * 0.1,
        buf_g=-torch.rand(slots, heads, cap, width, device=device),
        write_pos=torch.tensor([11, 0, -1], dtype=torch.int32, device=device),
        slot_idx=torch.tensor([0, 2], dtype=torch.int32, device=device),
        cu_seqlens=torch.tensor([0, window, tokens], dtype=torch.int32, device=device),
        max_query_len=window,
        o=torch.zeros(tokens, heads, width, device=device, dtype=torch.bfloat16),
        use_qk_l2norm_in_kernel=True,
        lower_bound=-5.0,
    )
    return args


@pytest.mark.parametrize("case", ["enabled", "disabled", "unsupported", "old_aiter"])
def test_routing(case, monkeypatch):
    args = _args()
    monkeypatch.setenv(
        "ATOM_USE_AITER_REPLAYSSM_KDA", "0" if case == "disabled" else "1"
    )
    monkeypatch.setattr(dispatch, "_supported", lambda *a: case != "unsupported")
    calls = []

    def backend(**kw):
        calls.append(kw)
        return kw["o"].unsqueeze(0)

    monkeypatch.setattr(
        dispatch,
        "_get_aiter_replayssm",
        lambda: None if case == "old_aiter" else backend,
    )
    out = dispatch.try_aiter_replayssm(**args)
    assert len(calls) == (case == "enabled")
    if calls:
        assert out.shape == (1, *args["o"].shape)
        assert out.data_ptr() == args["o"].data_ptr()
        assert all(calls[0][k] is v for k, v in args.items())
    else:
        assert out is None


@pytest.mark.parametrize("returns_none", [False, True])
def test_launch_failure_propagates(returns_none, monkeypatch):
    monkeypatch.setenv("ATOM_USE_AITER_REPLAYSSM_KDA", "1")
    monkeypatch.setattr(dispatch, "_supported", lambda *a: True)
    calls = []

    def backend(**kw):
        calls.append(1)
        if not returns_none:
            raise RuntimeError("failed after state update")

    monkeypatch.setattr(dispatch, "_get_aiter_replayssm", lambda: backend)
    with pytest.raises(RuntimeError, match="after"):
        dispatch.try_aiter_replayssm(**_args())
    assert calls == [1]


def test_old_aiter_import(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "aiter.ops.triton.attention.kda_replayssm":
            raise ImportError("no ReplaySSM in this version")
        return original(name, *args, **kwargs)

    dispatch._get_aiter_replayssm.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(builtins, "__import__", missing)
            assert dispatch._get_aiter_replayssm() is None
    finally:
        dispatch._get_aiter_replayssm.cache_clear()


def _entry():
    from conftest import gpu_stack_import

    return gpu_stack_import("atom.model_ops.fla_ops.replayssm")


def test_public_entry_seeds_fork_before_dispatch(monkeypatch):
    native = _entry()
    args = _args()
    parent = args["ckpt"][1].clone()
    calls = []
    seed = native.seed_forked_slots

    def seed_once(*values):
        calls.append("seed")
        seed(*values)

    def backend(*values, **kwargs):
        calls.append("backend")
        torch.testing.assert_close(args["ckpt"][2], parent)
        return args["o"].unsqueeze(0)

    monkeypatch.setattr(native, "seed_forked_slots", seed_once)
    monkeypatch.setattr(native, "try_aiter_replayssm", backend)
    result = native.replayssm_sigmoid_gating_delta_rule(
        **args, fork_slots=torch.tensor([[2], [1]])
    )
    assert calls == ["seed", "backend"]
    assert result.data_ptr() == args["o"].data_ptr()


def _gpu():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("gfx1250 required")
    if dispatch._get_aiter_replayssm() is None:
        pytest.skip("AITER ReplaySSM API required")


def _reference(args, fork_slots=None):
    # Independent token-by-token FP32 recurrence, including replay and flush.
    ckpt, bk, bu, bg = (args[k].clone() for k in ("ckpt", "buf_k", "buf_u", "buf_g"))
    if fork_slots is not None:
        ckpt[fork_slots[0]] = ckpt[fork_slots[1]]
    out = args["o"].clone()
    heads, width = args["v"].shape[2:]
    head_of = torch.arange(heads, device=out.device) // (heads // args["q"].shape[2])
    for n, slot in enumerate(args["slot_idx"].tolist()):
        start, end = args["cu_seqlens"][n : n + 2].tolist()
        if slot < 0 or start == end:
            continue
        h = max(0, int(args["write_pos"][slot]))
        state = ckpt[slot].clone()
        for j in range(h):
            state = state * bg[slot, :, j].exp()[:, None, :]
            state += (
                bu[slot, :, j].float()[:, :, None] * bk[slot, :, j].float()[:, None, :]
            )
        flush = h + 2 * args["max_query_len"] > bk.shape[2]
        if flush:
            ckpt[slot] = state
        for i in range(start, end):
            q, k = (args[name][0, i, head_of].float() for name in ("q", "k"))
            if args["use_qk_l2norm_in_kernel"]:
                q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
                k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
            x = args["a"][0, i].float() + args["dt_bias"].view(heads, width)
            strength = args["A_log"].exp()[:, None]
            bound = args["lower_bound"]
            g = (
                -strength * F.softplus(x)
                if bound is None
                else bound * torch.sigmoid(strength * x)
            )
            state = state * g.exp()[:, None, :]
            u = (args["v"][0, i].float() - (state @ k[:, :, None]).squeeze(-1)) * args[
                "b"
            ][0, i].float().sigmoid()[:, None]
            state += u[:, :, None] * k[:, None, :]
            scale = args.get("scale")
            out[i] = (
                state @ (q * (width**-0.5 if scale is None else scale))[:, :, None]
            ).squeeze(-1)
            pos = (0 if flush else h) + i - start
            bk[slot, :, pos], bu[slot, :, pos], bg[slot, :, pos] = k, u, g
    return out, ckpt, bk, bu, bg


@pytest.mark.parametrize("heads", [12, 24, 48, 96])
@pytest.mark.parametrize(
    "window,fork,graph",
    [(1, False, False), (5, False, False), (5, True, False), (5, True, True)],
)
def test_gpu_recurrence_and_fork(heads, window, fork, graph, monkeypatch):
    _gpu()
    native = _entry()
    monkeypatch.setenv("ATOM_USE_AITER_REPLAYSSM_KDA", "1")
    torch.manual_seed(7)
    args = _args(heads=heads, window=window, device="cuda")
    fork_slots = torch.tensor([[2], [1]], device="cuda") if fork else None
    if fork:
        for key in ("buf_k", "buf_u", "buf_g"):
            args[key][2].fill_(float("nan"))
    expected = _reference(args, fork_slots)
    metadata = {k: args[k].clone() for k in ("write_pos", "slot_idx", "cu_seqlens")}
    saved = {k: args[k].clone() for k in ("ckpt", "buf_k", "buf_u", "buf_g")}
    assert dispatch._supported(
        *(
            args[k]
            for k in (
                "q",
                "k",
                "v",
                "a",
                "b",
                "A_log",
                "dt_bias",
                "ckpt",
                "buf_k",
                "buf_u",
                "buf_g",
                "write_pos",
                "slot_idx",
                "cu_seqlens",
                "max_query_len",
                "o",
            )
        )
    )
    calls = []
    backend = dispatch._get_aiter_replayssm()

    def counted(**kw):
        calls.append(1)
        return backend(**kw)

    monkeypatch.setattr(dispatch, "_get_aiter_replayssm", lambda: counted)
    result = native.replayssm_sigmoid_gating_delta_rule(**args, fork_slots=fork_slots)
    assert calls == [1]
    if graph:
        for key, value in saved.items():
            args[key].copy_(value)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            native.replayssm_sigmoid_gating_delta_rule(**args, fork_slots=fork_slots)
        # Restore state so replay sees exactly the same checkpoint/history.
        for key, value in saved.items():
            args[key].copy_(value)
        g.replay()
    assert result.shape == (1, *args["o"].shape)
    assert result.data_ptr() == args["o"].data_ptr()
    for key, want, atol, rtol in zip(
        ("o", "ckpt", "buf_k", "buf_u", "buf_g"),
        expected,
        (2e-3, 1e-5, 1e-3, 2e-3, 1e-5),
        (2e-2, 1e-4, 1e-2, 2e-2, 1e-5),
    ):
        torch.testing.assert_close(
            args[key], want, atol=atol, rtol=rtol, equal_nan=True
        )
    for key, value in metadata.items():
        assert torch.equal(args[key], value)


@pytest.mark.parametrize(
    "case", ["disabled", "old_aiter", "unsupported", "fork_disabled"]
)
def test_gpu_native_fallback(case, monkeypatch):
    _gpu()
    native = _entry()
    args = _args(device="cuda", width=64 if case == "unsupported" else 128)
    # No flush: this checks routing without depending on native multi-CTA flush ordering.
    args["write_pos"].zero_()
    fork_slots = (
        torch.tensor([[2], [1]], device="cuda") if case == "fork_disabled" else None
    )
    expected = _reference(args, fork_slots)
    monkeypatch.setenv(
        "ATOM_USE_AITER_REPLAYSSM_KDA",
        "0" if case in ("disabled", "fork_disabled") else "1",
    )
    monkeypatch.setattr(
        dispatch,
        "_get_aiter_replayssm",
        lambda: None if case == "old_aiter" else pytest.fail("AITER must not launch"),
    )
    native.replayssm_sigmoid_gating_delta_rule(**args, fork_slots=fork_slots)
    for key, value in zip(("o", "ckpt", "buf_k", "buf_u", "buf_g"), expected):
        torch.testing.assert_close(args[key], value, atol=2e-3, rtol=2e-2)


@pytest.mark.parametrize(
    "case",
    [
        "width",
        "checkpoint_dtype",
        "checkpoint_layout",
        "records_layout",
        "metadata_dtype",
        "metadata_layout",
        "capacity",
        "gate_size",
        "output_alias",
    ],
)
def test_gpu_unsupported_metadata(case):
    _gpu()
    args = _args(device="cuda", width=64 if case == "width" else 128)
    if case == "checkpoint_dtype":
        args["ckpt"] = args["ckpt"].bfloat16()
    elif case == "checkpoint_layout":
        args["ckpt"] = args["ckpt"].transpose(-1, -2)
    elif case == "records_layout":
        args["buf_u"] = args["buf_u"].transpose(-1, -2)
    elif case == "metadata_dtype":
        args["slot_idx"] = args["slot_idx"].long()
    elif case == "metadata_layout":
        args["write_pos"] = torch.zeros(6, dtype=torch.int32, device="cuda")[::2]
    elif case == "capacity":
        args["max_query_len"] = 9
    elif case == "gate_size":
        args["dt_bias"] = args["dt_bias"][:-1]
    elif case == "output_alias":
        args["o"] = args["v"].squeeze(0)
    assert not dispatch._supported(
        *(
            args[k]
            for k in (
                "q",
                "k",
                "v",
                "a",
                "b",
                "A_log",
                "dt_bias",
                "ckpt",
                "buf_k",
                "buf_u",
                "buf_g",
                "write_pos",
                "slot_idx",
                "cu_seqlens",
                "max_query_len",
                "o",
            )
        )
    )

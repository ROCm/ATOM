# SPDX-License-Identifier: MIT

"""Kimi-KDA routing contract and optional gfx1250 integration coverage."""

import pytest
import torch

from atom.model_ops.fla_ops import kda_decode_dispatch as dispatch


def _arguments(n=8, device="cpu", padded=False):
    values = {
        name: torch.randn((1, n, 12, 128), device=device, dtype=torch.bfloat16)
        for name in ("q", "k", "v", "a")
    }
    state = torch.randn(n + 2, 12 * 128 * 128 + (128 if padded else 0), device=device)
    indices = torch.arange(n, device=device, dtype=torch.int32)
    values.update(
        b=torch.randn(
            1, n, 6284 if padded else 12, device=device, dtype=torch.bfloat16
        )[..., :12],
        A_log=torch.randn(12, device=device) * 0.1,
        dt_bias=torch.randn(12, 128, device=device) * 0.1,
        initial_state=state[:, : 12 * 128 * 128].view(n + 2, 12, 128, 128),
        o=torch.full((n, 12, 128), 7.0, device=device, dtype=torch.bfloat16),
        cu_seqlens=torch.arange(n + 1, device=device, dtype=torch.int32),
        ssm_state_indices=indices[:],
        ssm_state_indices_in=indices[:],
        use_qk_l2norm_in_kernel=True,
        is_kda=True,
        lower_bound=-5.0,
    )
    return values


def test_index_views():
    indices = torch.arange(8, dtype=torch.int32)
    assert dispatch._same_index_view(None, indices)
    assert dispatch._same_index_view(indices[:4], indices[:4])
    assert not dispatch._same_index_view(indices[:4], indices[1:5])
    assert not dispatch._same_index_view(indices, indices.clone())
    assert not dispatch._same_index_view(indices, None)


@pytest.mark.parametrize(
    "case",
    [
        "enabled",
        "disabled",
        "fork",
        "spec",
        "norm",
        "lower_bound",
        "scale",
        "not_kda",
        "non_inplace",
        "softplus",
        "threshold",
        "old_aiter",
        "unsupported",
    ],
)
def test_routing_contract(case, monkeypatch):
    args = _arguments()
    monkeypatch.setenv(
        "ATOM_USE_AITER_KDA_WRITEBACK", "0" if case == "disabled" else "1"
    )
    changes = {
        "fork": {"ssm_state_indices_in": args["ssm_state_indices"].clone()},
        "spec": {"num_accepted_tokens": torch.ones(8, dtype=torch.int32)},
        "norm": {"use_qk_l2norm_in_kernel": False},
        "lower_bound": {"lower_bound": None},
        "scale": {"scale": 1.0},
        "not_kda": {"is_kda": False},
        "non_inplace": {"inplace_final_state": False},
        "softplus": {"beta": 2.0},
        "threshold": {"threshold": 10.0},
    }
    args.update(changes.get(case, {}))
    monkeypatch.setattr(dispatch, "_supported", lambda *a: case != "unsupported")
    calls = []

    def aiter(**kw):
        calls.append(kw)
        return kw["out"], kw["initial_state"]

    monkeypatch.setattr(
        dispatch, "_get_aiter_decode", lambda: None if case == "old_aiter" else aiter
    )
    fallback = []
    monkeypatch.setattr(
        dispatch,
        "_atom_decode",
        lambda **kw: fallback.append(kw) or (kw["o"], kw["initial_state"]),
    )
    out, state = dispatch.fused_sigmoid_gating_delta_rule_update(**args)
    assert out is args["o"] and state is args["initial_state"]
    assert len(calls) == (case == "enabled")
    assert len(fallback) == (case != "enabled")
    if calls:
        kw = calls[0]
        assert kw["g"] is args["a"] and kw["beta"] is args["b"]
        assert kw["out"].shape == (1, 8, 12, 128)
        assert kw["out"].data_ptr() == args["o"].data_ptr()
        assert kw["state_v_first"] and not kw["pad_slot_guard"]
    else:
        assert all(fallback[0][key] is value for key, value in args.items())


def test_launch_failure_never_retries(monkeypatch):
    monkeypatch.setenv("ATOM_USE_AITER_KDA_WRITEBACK", "1")
    monkeypatch.setattr(dispatch, "_supported", lambda *a: True)

    def fail(**kw):
        raise RuntimeError("launch failed after possible state update")

    monkeypatch.setattr(dispatch, "_get_aiter_decode", lambda: fail)
    monkeypatch.setattr(dispatch, "_atom_decode", lambda **kw: pytest.fail("retried"))
    with pytest.raises(RuntimeError, match="launch failed"):
        dispatch.fused_sigmoid_gating_delta_rule_update(**_arguments())


def test_missing_aiter_import(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("aiter."):
            raise ImportError("older AITER without write-back")
        return original_import(name, *args, **kwargs)

    dispatch._get_aiter_decode.cache_clear()
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", missing)
        assert dispatch._get_aiter_decode() is None
    dispatch._get_aiter_decode.cache_clear()


def test_gpu_unsupported_layouts_fall_back(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("gfx1250 required")
    monkeypatch.setenv("ATOM_USE_AITER_KDA_WRITEBACK", "1")
    base = _arguments(8, "cuda")
    invalid = [
        {"initial_state": base["initial_state"].bfloat16()},
        {"q": base["q"][:, :, :6]},
        {"a": base["a"].float()},
        {
            "a": torch.empty(1, 8, 13, 128, device="cuda", dtype=torch.bfloat16)[
                :, :, :12
            ]
        },
        {
            "ssm_state_indices": base["ssm_state_indices"].long(),
            "ssm_state_indices_in": None,
        },
        {
            "ssm_state_indices": base["ssm_state_indices"].view(8, 1),
            "ssm_state_indices_in": None,
        },
        {"cu_seqlens": base["cu_seqlens"].long()},
        {"dt_bias": base["dt_bias"].bfloat16()},
        {"o": base["q"].squeeze(0)},  # shared storage is not a safe output
        {"o": base["o"].transpose(1, 2)},
    ]
    calls = []
    monkeypatch.setattr(
        dispatch,
        "_atom_decode",
        lambda **kw: calls.append(kw) or (kw["o"], kw["initial_state"]),
    )
    monkeypatch.setattr(
        dispatch,
        "_get_aiter_decode",
        lambda: pytest.fail("unsupported input routed to AITER"),
    )
    for change in invalid:
        args = {**base, **change}
        out, state = dispatch.fused_sigmoid_gating_delta_rule_update(**args)
        assert out is args["o"] and state is args["initial_state"]
    assert len(calls) == len(invalid)


def _reference(args, state, out):
    starts = args["cu_seqlens"][:-1]
    slots = args["ssm_state_indices"]
    valid = (slots >= 0) & (args["cu_seqlens"][1:] > starts)
    tokens, slots = starts[valid].long(), slots[valid].long()
    q, k, v, a = (args[key][0, tokens].float() for key in ("q", "k", "v", "a"))
    q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * 128**-0.5
    k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    decay = (
        -5.0 * torch.sigmoid(args["A_log"].exp()[:, None] * (a + args["dt_bias"]))
    ).exp()
    s = state[slots] * decay[:, :, None, :]
    error = (v - (s * k[:, :, None, :]).sum(-1)) * args["b"][
        0, tokens
    ].float().sigmoid()[:, :, None]
    s += error[:, :, :, None] * k[:, :, None, :]
    state[slots] = s
    out[tokens] = (s * q[:, :, None, :]).sum(-1).to(out.dtype)


@pytest.mark.parametrize("n", [1, 8, 72])
@pytest.mark.parametrize("padded", [False, True])
def test_gpu_writeback_and_graph(n, padded, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("gfx1250 required")
    from aiter.ops.triton.attention import kda

    dispatch._get_aiter_decode.cache_clear()
    assert dispatch._get_aiter_decode() is not None, "install AITER PR #7916"
    monkeypatch.setenv("ATOM_USE_AITER_KDA_WRITEBACK", "1")
    monkeypatch.setattr(
        dispatch, "_atom_decode", lambda **kw: pytest.fail("unexpected fallback")
    )
    selected = []
    original = kda.try_writeback

    def observe(*a, **kw):
        result = original(*a, **kw)
        selected.append(result is not None)
        return result

    monkeypatch.setattr(kda, "try_writeback", observe)
    torch.manual_seed(42)
    args = _arguments(n, "cuda", padded)
    if n > 1:
        args["ssm_state_indices"][1] = -1
        args["cu_seqlens"][-1] = n - 1  # trailing empty sequence
    expected_state, expected_out = args["initial_state"].clone(), args["o"].clone()
    for step in range(3):
        if step:
            for key in ("q", "k", "v", "a", "b"):
                args[key].normal_()
        _reference(args, expected_state, expected_out)
        out, state = dispatch.fused_sigmoid_gating_delta_rule_update(**args)
        assert out is args["o"] and state is args["initial_state"]
        torch.testing.assert_close(out, expected_out, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)
    # Warm up on a side stream before capture; reset after capture so replay
    # starts from exactly the same state/output as the reference step.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dispatch.fused_sigmoid_gating_delta_rule_update(**args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        dispatch.fused_sigmoid_gating_delta_rule_update(**args)
    args["initial_state"].copy_(expected_state)
    args["o"].copy_(expected_out)
    _reference(args, expected_state, expected_out)
    graph.replay()
    torch.testing.assert_close(args["o"], expected_out, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(
        args["initial_state"], expected_state, atol=1e-5, rtol=1e-4
    )
    assert selected and all(selected), "public AITER wrapper must select write-back"

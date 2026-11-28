# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The measured gfx942 entry preserves records with eight V-slice arrivals."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("requires a gfx942 GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx942":
    pytest.skip("exercises the gfx942-specific launch rule", allow_module_level=True)

from atom.model_ops.fla_ops import replayssm as replay

SLOTS, H, HV, K, V, T, CAP = 3, 8, 24, 128, 128, 4, 16


def _fixture(h, seed=3943):
    torch.manual_seed(seed)
    bf16 = {"dtype": torch.bfloat16, "device": "cuda"}
    state = {
        "ckpt": torch.randn(SLOTS, HV, K, V, device="cuda") * 0.03,
        "buf_k": torch.randn(SLOTS, HV, CAP, K, **bf16) * 0.1,
        "buf_u": torch.randn(SLOTS, HV, CAP, V, **bf16) * 0.03,
        "buf_g": -torch.rand(SLOTS, HV, CAP, device="cuda") * 0.1,
        "write_pos": torch.tensor([h, 4, 8], dtype=torch.int32, device="cuda"),
        "flush_arrivals": torch.zeros(SLOTS, HV, dtype=torch.int32, device="cuda"),
    }
    inputs = {
        "q": torch.randn(1, T, H, K, **bf16),
        "k": torch.randn(1, T, H, K, **bf16),
        "v": torch.randn(1, T, HV, V, **bf16) * 0.1,
        "g": -torch.rand(1, T, HV, device="cuda") * 0.1,
        "beta": torch.rand(1, T, HV, **bf16) * 0.5,
        "slot_idx": torch.tensor([0], dtype=torch.int32, device="cuda"),
        "cu_seqlens": torch.tensor([0, T], dtype=torch.int32, device="cuda"),
        "max_query_len": T,
        "use_qk_l2norm_in_kernel": True,
        "flush_wait": replay.replayssm_flush_wait(),
    }
    return state, inputs


def _clone(state):
    return {name: value.clone() for name, value in state.items()}


def _close(got, want):
    assert got.shape == want.shape and got.dtype == want.dtype
    assert torch.isfinite(got).all() and torch.isfinite(want).all()
    error = (got.double() - want.double()).abs()
    rel = float(error.norm() / want.double().norm().clamp_min(1e-12))
    rms = float(want.double().square().mean().sqrt().clamp_min(1e-12))
    peak = float(error.max()) / rms
    # Frozen tiling comparison bounds; these do not establish model accuracy.
    assert rel <= 0.005 and peak <= 0.05, (rel, peak)
    return {"rel_l2": rel, "max_over_rms": peak, "exact": bool(torch.equal(got, want))}


def _oracle(original, inputs, h):
    """FP64 sequential recurrence from the checkpoint and committed records."""
    expected = _clone(original)
    state = original["ckpt"][0].double()
    for pos in range(h):
        state = state * original["buf_g"][0, :, pos].double().exp()[:, None, None]
        state = state + (
            original["buf_k"][0, :, pos].double()[:, :, None]
            * original["buf_u"][0, :, pos].double()[:, None, :]
        )
    flush = h + 2 * T > CAP
    if flush:
        expected["ckpt"][0].copy_(state)
    base = 0 if flush else h
    output = torch.empty((1, T, HV, V), dtype=torch.bfloat16, device="cuda")
    for pos in range(T):
        q, k = [inputs[name][0, pos].double() for name in ("q", "k")]
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        q, k = (value.repeat_interleave(HV // H, dim=0) for value in (q, k))
        q = q * K**-0.5
        g = inputs["g"][0, pos].double()
        state = state * g.exp()[:, None, None]
        u = inputs["beta"][0, pos].double()[:, None] * (
            inputs["v"][0, pos].double() - (state * k[:, :, None]).sum(1)
        )
        state = state + k[:, :, None] * u[:, None, :]
        output[0, pos].copy_((state * q[:, :, None]).sum(1))
        expected["buf_k"][0, :, base + pos].copy_(k)
        expected["buf_u"][0, :, base + pos].copy_(u)
        expected["buf_g"][0, :, base + pos].copy_(g)
    return output, expected


class _LaunchObserver:
    """Record one real JIT launch and delegate its arguments without changes."""

    def __init__(self, kernel, rows):
        self.kernel, self.rows = kernel, rows

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def observed(*args, **kwargs):
            compiled = launch(*args, **kwargs)
            assert compiled is not None
            self.rows.append(
                {
                    "grid": list(grid),
                    "BV": kwargs["BV"],
                    "num_warps": compiled.metadata.num_warps,
                    "kernel_name": compiled.name,
                    "WAIT": kwargs["WAIT"],
                }
            )
            return compiled

        return observed


@pytest.mark.parametrize("h", [0, 8, 12])
@pytest.mark.parametrize("graph", [False, True])
def test_gfx942_entry_and_flush_arrivals(h, graph, monkeypatch, record_property):
    original, inputs = _fixture(h)
    work = _clone(original)
    destination = torch.empty((T, HV, V), dtype=torch.bfloat16, device="cuda")
    launches = []
    monkeypatch.setattr(
        replay,
        "_replayssm_fwd_kernel",
        _LaunchObserver(replay._replayssm_fwd_kernel, launches),
    )

    def step():
        for name, value in work.items():
            # The kernel must retire its own arrivals, including graph replay.
            if name != "flush_arrivals":
                value.copy_(original[name])
        return replay.replayssm_gated_delta_rule(**inputs, **work, o=destination)

    step()
    torch.cuda.synchronize()
    assert not work["flush_arrivals"].any()
    if graph:
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            result = step()
        # Change contents at the captured addresses, then evaluate that input.
        _, changed = _fixture(h, seed=503)
        for name in ("q", "k", "v", "g", "beta"):
            inputs[name].copy_(changed[name])

        def run():
            captured.replay()
            return result

    else:
        run = step
    first = {"output": run().clone(), **_clone(work)}
    differs = {
        name: torch.zeros((), dtype=torch.int32, device="cuda") for name in first
    }
    for _ in range(63):
        current = {"output": run(), **work}
        for name, value in current.items():
            differs[name] += (value != first[name]).any()
    assert {name: int(value) for name, value in differs.items()} == dict.fromkeys(
        first, 0
    )
    assert not work["flush_arrivals"].any()
    want_output, want = _oracle(original, inputs, h)
    metrics = {"output": _close(first["output"], want_output)}
    metrics["live_checkpoint"] = _close(work["ckpt"][0], want["ckpt"][0])
    base = 0 if h + 2 * T > CAP else h
    for pos in range(CAP):
        if base <= pos < base + T:
            metrics[f"u_position_{pos}"] = _close(
                work["buf_u"][0, :, pos], want["buf_u"][0, :, pos]
            )
            _close(work["buf_k"][0, :, pos], want["buf_k"][0, :, pos])
            assert torch.equal(work["buf_g"][0, :, pos], want["buf_g"][0, :, pos])
        else:
            for name in ("buf_k", "buf_u", "buf_g"):
                assert torch.equal(work[name][0, :, pos], original[name][0, :, pos])
    for name in original:
        assert torch.equal(work[name][1:], original[name][1:]), name
    assert torch.equal(work["write_pos"], original["write_pos"])
    if h + 2 * T <= CAP:
        assert torch.equal(work["ckpt"], original["ckpt"])
    assert launches and all(
        row["grid"] == [8, HV]
        and row["BV"] == 16
        and row["num_warps"] == 1
        and "replayssm_fwd_kernel" in row["kernel_name"]
        and row["WAIT"] == inputs["flush_wait"]
        for row in launches
    )
    record_property("actual_launch", json.dumps(launches[-1], sort_keys=True))
    record_property("numerical_metrics", json.dumps(metrics, sort_keys=True))
    record_property("repeat_count", 64)
    record_property("arrivals_zero", True)

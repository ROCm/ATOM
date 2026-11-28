# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Recurrent padding writes +0, preserving real rows and all unowned state."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Recurrent kernels require a GPU", allow_module_level=True)

from atom.model_ops.fla_ops.fused_sigmoid_gating import (
    fused_sigmoid_gating_delta_rule_update,
)
from atom.model_ops.fla_ops.replayssm import (
    replayssm_buffer_shapes,
    replayssm_gated_delta_rule,
    replayssm_sigmoid_gating_delta_rule,
)

# V=48 uses two KDA BV=32 tiles (last partial) and one partial UT BV=64 tile.
# Serial uses one partial BV=64 tile on gfx942 or at T=1; for T>1 on gfx950
# these small batches use three full BV=16 tiles.
H, HV, K, V = 1, 2, 32, 48
NUM_SLOTS = 8
REAL_SLOTS = [3, 0, 5, 2]
CASES = [
    ("gdn", 1, "serial"),
    ("gdn", 4, "serial"),
    ("gdn", 4, "ut"),
    ("kda", 1, None),
    ("kda", 4, None),
]
STATE_NAMES = ("ckpt", "buf_k", "buf_u", "buf_g", "write_pos")


def _inputs(kind, t, lengths, device, dtype):
    gen = torch.Generator(device=device).manual_seed(20261119)
    total = sum(lengths)
    cap = 4 * t
    kda = kind == "kda"

    def rnd(*shape, dtype=dtype):
        return torch.randn(*shape, generator=gen, device=device).to(dtype)

    state = (NUM_SLOTS, HV, V, K) if kda else (NUM_SLOTS, HV, K, V)
    shapes = replayssm_buffer_shapes(cap, HV, K, V, kda)
    # Slot 5 flushes; slot 3 appends without flushing.
    x = {
        "q": rnd(1, total, H, K),
        "k": rnd(1, total, H, K),
        "v": rnd(1, total, HV, V),
        "ckpt": rnd(*state),
        "buf_k": rnd(NUM_SLOTS, *shapes[0]),
        "buf_u": rnd(NUM_SLOTS, *shapes[1]),
        "buf_g": -rnd(NUM_SLOTS, *shapes[2], dtype=torch.float32).abs() * 0.1,
        "write_pos": torch.tensor(
            [1, 2, 0, t, 3, cap - t, 1, 0], device=device, dtype=torch.int32
        ),
    }
    if kda:
        x.update(a=rnd(1, total, HV, K), b=rnd(1, total, HV))
        x["A_log"] = rnd(HV, dtype=torch.float32)
        x["dt_bias"] = rnd(HV, K, dtype=torch.float32)
    else:
        x["g"] = -rnd(1, total, HV, dtype=torch.float32).abs() * 0.1
        x["beta"] = rnd(1, total, HV, dtype=torch.float32).sigmoid()
    cu = torch.tensor([0, *lengths], device=device, dtype=torch.int32).cumsum(0)
    return x, cu


def _case(kind, t, route, slots, lengths, device, dtype=torch.bfloat16):
    x, cu = _inputs(kind, t, lengths, device, dtype)
    pad = (
        torch.tensor(slots, device=device)
        .lt(0)
        .repeat_interleave(torch.tensor(lengths, device=device))
    )
    for name in ("q", "k", "v", "a", "b", "g", "beta"):
        if name in x:
            x[name][0, pad] = float("nan")
    # Red zones catch writes past a partial last V tile or sequence boundary.
    arena = torch.full((sum(lengths) + 2, HV, V), 17.0, device=device, dtype=dtype)
    out = arena[1:-1]
    out.fill_(float("nan"))
    common = {name: x[name] for name in STATE_NAMES}
    common.update(
        slot_idx=torch.tensor(slots, device=device, dtype=torch.int32),
        cu_seqlens=cu,
        max_query_len=t,
        use_qk_l2norm_in_kernel=True,
        o=out,
    )

    def launch():
        if kind == "kda":
            return replayssm_sigmoid_gating_delta_rule(
                x["q"],
                x["k"],
                x["v"],
                x["a"],
                x["b"],
                x["A_log"],
                x["dt_bias"],
                **common,
            )
        return replayssm_gated_delta_rule(
            x["q"], x["k"], x["v"], x["g"], x["beta"], route=route, **common
        )

    return launch, out, x, arena, pad


def check(kind, t, route, layout, all_pad, device, dtype=torch.bfloat16):
    lengths = {
        "uniform": [t] * 4,
        "ragged": [t, 0, max(1, t - 1), t],
        "empty": [0] * 4,
    }[layout]
    slots = [-1] * 4 if all_pad else [3, -1, 5, -1]
    init, _ = _inputs(kind, t, lengths, device, dtype)
    run, out, padded, arena, pad = _case(kind, t, route, slots, lengths, device, dtype)
    ref_run, ref, reference, _, _ = _case(
        kind, t, route, REAL_SLOTS, lengths, device, dtype
    )
    run()
    ref_run()
    assert torch.equal(out[pad], torch.zeros_like(out[pad])), "pad rows not zero"
    assert not torch.signbit(out[pad]).any(), "pad rows contain negative zero"
    assert torch.equal(out[~pad], ref[~pad]), "real rows differ"
    assert torch.all(arena[[0, -1]] == 17), "output red zone overwritten"
    live = [s for s, n in zip(slots, lengths) if s >= 0 and n > 0]
    for name in STATE_NAMES:
        want = init[name].clone()
        want[live] = reference[name][live]
        assert torch.equal(padded[name], want), f"{name} differs"


@pytest.mark.parametrize("kind, t, route", CASES)
@pytest.mark.parametrize("layout", ["uniform", "ragged", "empty"])
@pytest.mark.parametrize("all_pad", [False, True], ids=["mixed", "all_pad"])
def test_pad_entries_write_zeros_and_touch_no_state(kind, t, route, layout, all_pad):
    check(kind, t, route, layout, all_pad, "cuda")


@pytest.mark.parametrize(
    "kind, t, state_dtype",
    [
        ("gdn", 1, torch.float32),
        ("gdn", 4, torch.bfloat16),
        ("kda", 1, torch.bfloat16),
        ("kda", 4, torch.float32),
    ],
)
@pytest.mark.parametrize("layout", ["uniform", "ragged", "empty"])
@pytest.mark.parametrize("all_pad", [False, True], ids=["mixed", "all_pad"])
@pytest.mark.parametrize("supplied_output", [False, True], ids=["allocated", "caller"])
def test_fused_sigmoid_pad_entries_write_zeros_and_touch_no_state(
    kind, t, state_dtype, layout, all_pad, supplied_output
):
    lengths = {
        "uniform": [t] * 4,
        "ragged": [t, 0, max(1, t - 1), t],
        "empty": [0] * 4,
    }[layout]
    slots = [-1] * 4 if all_pad else [3, -1, 5, -1]
    # Both fused gate variants use [slot, HV, V, K] state. Reuse the KDA
    # fixture, reducing a/dt_bias to headwise scalars only for the GDN gate.
    initial, cu = _inputs("kda", t, lengths, "cuda", torch.bfloat16)
    initial["ckpt"] = initial["ckpt"].to(state_dtype)
    if kind == "gdn":
        initial["a"] = initial["a"][..., 0].contiguous()
        initial["dt_bias"] = initial["dt_bias"][:, 0].contiguous()
    pad = (
        torch.tensor(slots, device="cuda")
        .lt(0)
        .repeat_interleave(torch.tensor(lengths, device="cuda"))
    )
    writes = (
        torch.tensor(REAL_SLOTS, device="cuda", dtype=torch.int32)
        .view(-1, 1)
        .expand(-1, t)
        .contiguous()
    )

    def launch(read_slots):
        x = {
            name: initial[name].clone()
            for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias", "ckpt")
        }
        reads = (
            torch.tensor(read_slots, device="cuda", dtype=torch.int32)
            .view(-1, 1)
            .expand(-1, t)
            .contiguous()
        )
        masked = (reads[:, 0] < 0).repeat_interleave(
            torch.tensor(lengths, device="cuda")
        )
        for name in ("q", "k", "v", "a", "b"):
            x[name][0, masked] = float("nan")
        arena = torch.full(
            (sum(lengths) + 2, HV, V), 17.0, device="cuda", dtype=torch.bfloat16
        )
        destination = arena[1:-1].unsqueeze(0)
        destination.fill_(float("nan"))
        output, state = fused_sigmoid_gating_delta_rule_update(
            x["A_log"],
            x["a"],
            x["b"],
            x["dt_bias"],
            x["q"],
            x["k"],
            x["v"],
            o=destination if supplied_output else None,
            initial_state=x["ckpt"],
            cu_seqlens=cu,
            ssm_state_indices=writes,
            ssm_state_indices_in=reads,
            use_qk_l2norm_in_kernel=True,
            is_kda=kind == "kda",
            lower_bound=-5.0 if kind == "kda" else None,
        )
        assert state.data_ptr() == x["ckpt"].data_ptr()
        if supplied_output:
            assert output.data_ptr() == destination.data_ptr()
        assert output.shape == (1, sum(lengths), HV, V)
        assert torch.all(arena[[0, -1]] == 17), "output red zone overwritten"
        return output[0], state

    out, state = launch(slots)
    reference, reference_state = launch(REAL_SLOTS)
    assert torch.equal(out[pad], torch.zeros_like(out[pad])), "pad rows not zero"
    assert not torch.signbit(out[pad]).any(), "pad rows contain negative zero"
    assert torch.equal(out[~pad], reference[~pad]), "real rows differ"
    live = [s for s, n in zip(slots, lengths) if s >= 0 and n > 0]
    want = initial["ckpt"].clone()
    want[live] = reference_state[live]
    # Even a valid write-side slot must remain untouched when its read side is
    # padding. FP32/BF16 state also exercises BV=32/64 with this partial V tile.
    assert torch.equal(state, want), "unowned or live state differs"


@pytest.mark.parametrize(
    "kind, route", [("gdn", "serial"), ("gdn", "ut"), ("kda", None)]
)
def test_pad_rows_on_graph_replay_with_changed_inputs(kind, route):
    run, out, state, arena, pad = _case(
        kind, 4, route, [3, -1, 5, -1], [4, 0, 3, 4], "cuda"
    )
    initial = {name: state[name].clone() for name in STATE_NAMES}
    # Complete JIT outside capture. Keep all captured pointers stable.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for delta in (0.125, -0.375):
        state["q"].add_(delta)
        for name in STATE_NAMES:
            state[name].copy_(initial[name])
        out.fill_(float("nan"))
        run()
        expected_out = out.clone()
        expected_state = {name: state[name].clone() for name in STATE_NAMES}
        for name in STATE_NAMES:
            state[name].copy_(initial[name])
        out.fill_(float("nan"))
        graph.replay()
        assert torch.equal(out, expected_out), "graph output differs from eager"
        assert torch.equal(out[pad], torch.zeros_like(out[pad]))
        assert not torch.signbit(out[pad]).any()
        assert torch.all(arena[[0, -1]] == 17)
        for name in STATE_NAMES:
            assert torch.equal(state[name], expected_state[name]), name

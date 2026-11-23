# SPDX-License-Identifier: MIT
"""ReplaySSM padding writes +0, preserving real rows and all unowned state."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton", reason="the ReplaySSM kernels are Triton")
pytest.importorskip("aiter", reason="replayssm imports aiter's arch helpers")

from atom.model_ops.fla_ops.replayssm import (
    replayssm_buffer_shapes,
    replayssm_gated_delta_rule,
    replayssm_sigmoid_gating_delta_rule,
)

# V spans multiple serial/KDA tiles and a partial last tile.
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("kind, t, route", CASES)
@pytest.mark.parametrize("layout", ["uniform", "ragged", "empty"])
@pytest.mark.parametrize("all_pad", [False, True], ids=["mixed", "all_pad"])
def test_pad_entries_write_zeros_and_touch_no_state(kind, t, route, layout, all_pad):
    check(kind, t, route, layout, all_pad, "cuda")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
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

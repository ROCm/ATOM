# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Where a splitting op's output buffer gets allocated.

A splitting graph is the one piece PIECEWISE never compiles and never captures:
it runs eagerly between two captured pieces. If it *returns* its output, that is
a fresh allocation every step and the captured piece downstream reads the one
address it baked at capture. If instead the caller allocates the buffer and the
op only mutates it, the allocation is a node in the piece *upstream* of the
split -- captured, so its address is fixed by construction.

These tests pin that placement: they assert the `torch.empty` really does land
in the upstream submodule, which is the whole reason Kimi-K3's
`kda_attention_with_output` takes an `output` argument.
"""

from __future__ import annotations

import torch
from torch import fx

from atom.utils.backends import split_graph


def _empty_like_shape(x: torch.Tensor, out: torch.Tensor) -> None:
    return None


torch.library.define(
    "atom_test::split_with_output",
    "(Tensor x, Tensor(a1!) out) -> ()",
)
torch.library.impl("atom_test::split_with_output", "CompositeExplicitAutograd")(
    lambda x, out: out.copy_(x * 2)
)
torch.library.register_fake("atom_test::split_with_output")(_empty_like_shape)
torch.ops.atom_test.split_with_output.spliting_op = True


class _Model(torch.nn.Module):
    """up -> empty -> split op (mutates) -> down, the Kimi-K3 KDA shape."""

    def forward(self, x):
        up = x + 1
        out = torch.empty_like(up)
        torch.ops.atom_test.split_with_output(up, out)
        return out * 3


def _split(model):
    gm = fx.symbolic_trace(model)
    return split_graph(gm, [])


def _node_targets(submod):
    return [n.target for n in submod.graph.nodes if n.op == "call_function"]


def test_output_buffer_is_allocated_in_the_upstream_piece():
    """The `torch.empty` must be a node of the submodule *before* the split, so
    the cudagraph pool that captures that piece owns the address."""
    split_gm, items = _split(_Model())

    upstream = next(item for item in items if not item.is_splitting_graph)
    submod = getattr(split_gm, upstream.submod_name)

    assert torch.empty_like in _node_targets(submod), (
        "the output allocation escaped the captured piece; a splitting op that "
        "allocates its own output is exactly the PIECEWISE replay bug"
    )


def test_the_splitting_piece_only_calls_the_op():
    """Nothing but the opaque op lives in the eager piece -- in particular no
    allocation, which is what would move between steps."""
    split_gm, items = _split(_Model())

    splitting = next(item for item in items if item.is_splitting_graph)
    targets = _node_targets(getattr(split_gm, splitting.submod_name))

    assert targets == [torch.ops.atom_test.split_with_output]


def test_the_mutated_buffer_crosses_the_split_as_a_graph_edge():
    """The buffer is produced upstream and read downstream, so both ends are
    captured pieces and the address is stable without any runtime pinning."""
    split_gm, items = _split(_Model())
    names = [item.submod_name for item in items]

    calls = [n for n in split_gm.graph.nodes if n.op == "call_module"]
    assert [n.target for n in calls] == names

    # The splitting submodule takes the buffer as an argument, and the piece
    # after it takes the splitting submodule's *predecessor* output -- not the
    # splitting submodule's return value, which is empty.
    splitting_idx = next(i for i, item in enumerate(items) if item.is_splitting_graph)
    downstream = calls[splitting_idx + 1]
    assert calls[splitting_idx] not in downstream.args


class _PrologueModel(torch.nn.Module):
    """The Kimi-K3 KDA shape after the boundary shrank to conv1d-onward:
    in_proj GEMM -> f_b_proj GEMM -> empty -> split op (mutates) -> o-side."""

    def forward(self, x, w_in, w_fb):
        fused_in = torch.mm(x, w_in)
        gate = torch.mm(fused_in, w_fb)
        out = torch.empty_like(fused_in)
        torch.ops.atom_test.split_with_output(gate, out)
        return out * 3


def test_the_prologue_gemms_land_in_the_captured_piece():
    """Both projections must be upstream of the split, not inside the eager
    submodule. That is the whole point of shrinking the KDA boundary: a
    splitting submodule is excluded from compilation, so anything left in it
    pays eager dispatch on every one of ~69 layers."""
    split_gm, items = _split(_PrologueModel())

    upstream = next(item for item in items if not item.is_splitting_graph)
    targets = _node_targets(getattr(split_gm, upstream.submod_name))

    assert targets.count(torch.mm) == 2, f"a prologue GEMM escaped: {targets}"
    assert torch.empty_like in targets

    splitting = next(item for item in items if item.is_splitting_graph)
    assert _node_targets(getattr(split_gm, splitting.submod_name)) == [
        torch.ops.atom_test.split_with_output
    ]


def test_split_graph_still_handles_a_returning_op():
    """The old shape must keep working: base_attention.py's ops still return."""

    torch.library.define("atom_test::split_returning", "(Tensor x) -> Tensor")
    torch.library.impl("atom_test::split_returning", "CompositeExplicitAutograd")(
        lambda x: x * 2
    )
    torch.library.register_fake("atom_test::split_returning")(
        lambda x: torch.empty_like(x)
    )
    torch.ops.atom_test.split_returning.spliting_op = True

    class _Returning(torch.nn.Module):
        def forward(self, x):
            return torch.ops.atom_test.split_returning(x + 1) * 3

    _, items = _split(_Returning())

    assert sum(item.is_splitting_graph for item in items) == 1

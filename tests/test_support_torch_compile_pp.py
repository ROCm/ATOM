# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A non-first pipeline stage's first compiled forward (jgong5/ATOM#1321).

Qwen3.5 and GLM-4 MoE list `intermediate_tensors` in `dynamic_arg_dims`. On
a non-first stage that argument is the `IntermediateTensors` the previous stage
sent, and `support_torch_compile` must mark the tensors it carries, not refuse
the container.
"""

from types import SimpleNamespace

import torch
from torch import nn

from atom.config import CompilationConfig, CompilationLevel
from atom.models.utils import IntermediateTensors
from atom.utils.decorators import support_torch_compile


# The declaration `Qwen3_5Model` and `Glm4MoeModel` carry.
@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class _Stage(nn.Module):
    def __init__(self, *, atom_config, prefix: str = ""):
        super().__init__()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]
        return hidden_states + residual


def _stage():
    level = CompilationLevel.PIECEWISE
    atom_config = SimpleNamespace(compilation_config=CompilationConfig(level=level))
    stage = _Stage(atom_config=atom_config)
    assert not stage.do_not_compile
    # Dynamo traces the real forward; only the backend is a recorder, so no
    # graph is lowered.
    graphs = []

    def record(gm, _example_inputs):
        graphs.append(gm)
        return gm.forward

    stage.compiled_callable = torch.compile(stage.forward, backend=record)
    return stage, graphs


def test_received_tensors_are_marked_dynamic_on_the_first_forward():
    stage, graphs = _stage()
    num_tokens, hidden = 8, 4
    received = IntermediateTensors(
        {
            "hidden_states": torch.randn(num_tokens, hidden),
            "residual": torch.randn(num_tokens, hidden),
        }
    )

    out = stage(
        torch.zeros(num_tokens, dtype=torch.long),
        torch.arange(num_tokens),
        intermediate_tensors=received,
    )

    torch.testing.assert_close(out, received["hidden_states"] + received["residual"])
    for name in ("hidden_states", "residual"):
        assert received[name]._dynamo_dynamic_indices == {0}, name
    # The traced graph took the token count as a symbol, not as 8.
    (graph,) = graphs
    shapes = [
        node.meta["example_value"].shape
        for node in graph.graph.nodes
        if node.op == "placeholder"
        and isinstance(node.meta.get("example_value"), torch.Tensor)
        and node.meta["example_value"].dim() == 2
    ]
    assert len(shapes) == 2
    assert all(isinstance(shape[0], torch.SymInt) for shape in shapes)

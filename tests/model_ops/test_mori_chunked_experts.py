# SPDX-License-Identifier: MIT
"""Ragged chunking through the production MoRI prepare and routed-row wire."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("wire_kind", ["fp4", "fp8", "bf16"])
@pytest.mark.parametrize("prepacked", [False, True])
@pytest.mark.parametrize("local_rows", [0, 3, 6])
def test_ragged_chunks_keep_the_dispatch_width_and_restore_context(
    atom_modules, monkeypatch, wire_kind, prepacked, local_rows
):
    routed = importlib.import_module("atom.model_ops.fused_moe.routed_rows")
    mpf = importlib.import_module("atom.model_ops.fused_moe.mori_prepare_finalize")
    chunked = importlib.import_module("atom.model_ops.fused_moe.mori_chunked_experts")
    internode = importlib.import_module(
        "atom.model_ops.fused_moe.megamoe_internode_experts"
    )
    fc = importlib.import_module("atom.utils.forward_context")
    hidden, cap, group_rows = 5120, 4, 10
    wire = routed.resolve_routed_rows_wire(
        torch.bfloat16,
        hidden,
        fp4=wire_kind == "fp4",
        mxfp8_activations=wire_kind == "fp8",
    )
    width = hidden // 2 if wire_kind == "fp4" else hidden
    calls = []
    across = [local_rows, group_rows]
    context = SimpleNamespace(running_tokens_across_dp=across)
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(internode, "_group_rows", lambda rows: (group_rows, False))
    monkeypatch.setattr(mpf, "MORI_AVAILABLE", True)

    # The external quantizer and GPU dispatch/GEMMs are the only stand-ins.
    quantizer = Mock(
        side_effect=lambda x, **kwargs: (
            torch.empty(x.shape[0], width, dtype=wire.dtype),
            torch.empty(x.shape[0], wire.scale_dim, dtype=wire.scale_dtype),
        )
    )
    monkeypatch.setattr(routed, "get_hip_quant", lambda quant_type: quantizer)

    def dispatch(x, weights, scale, ids, **kwargs):
        calls.append(
            (
                x.shape,
                x.dtype,
                scale.shape if scale is not None else None,
                list(context.running_tokens_across_dp),
            )
        )
        assert x.shape[1] == width
        return x, weights, scale, ids, torch.zeros(16, dtype=torch.int32)

    handle = SimpleNamespace(
        dispatch=dispatch,
        config=SimpleNamespace(rank=0, max_num_inp_token_per_rank=cap),
        _cpp_config=SimpleNamespace(max_num_tokens_to_recv=lambda: cap * 16),
    )
    pf = mpf.MoriPrepareAndFinalize(
        [handle],
        max_tokens_per_rank=cap,
        num_dispatchers=16,
        dispatch_format=wire,
        low_latency=False,
        internode=True,
    )
    monkeypatch.setattr(pf, "_launch_geometry", lambda rows: (-1, -1))

    class Experts(torch.nn.Module):
        shared_experts = None
        prepare_finalize = pf

        def forward(self, hidden_states, topk_weights, topk_ids, x_scale=None):
            pf.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                256,
                None,
                False,
                quant_config=None,
                x_scale=x_scale,
            )
            return torch.ones(hidden_states.shape[0], hidden, dtype=torch.bfloat16)

    x = torch.empty(
        local_rows,
        width if prepacked else hidden,
        dtype=wire.dtype if prepacked else torch.bfloat16,
    )
    scale = (
        torch.empty(local_rows, wire.scale_dim, dtype=wire.scale_dtype)
        if prepacked and wire.quant_type is not None
        else None
    )
    result = chunked.MoriChunkedExperts(Experts(), cap)(
        hidden_states=x,
        topk_weights=torch.ones(local_rows, 2),
        topk_ids=torch.zeros(local_rows, 2, dtype=torch.int32),
        x_scale=scale,
    )
    expected_rows = [min(local_rows, cap), max(0, local_rows - cap), 0]
    assert [c[0] for c in calls] == [torch.Size([n, width]) for n in expected_rows]
    assert all(c[1] == wire.dtype for c in calls)
    expected_scale = (
        (lambda n: torch.Size([n, hidden // 32])) if wire.quant_type else lambda n: None
    )
    assert [c[2] for c in calls] == [expected_scale(n) for n in expected_rows]
    assert [c[3] for c in calls] == [
        [min(local_rows, cap), 4],
        [max(0, local_rows - cap), 4],
        [0, 2],
    ]
    assert context.running_tokens_across_dp is across
    assert result.shape == (local_rows, hidden) and result.dtype == torch.bfloat16
    assert (result == 1).all()
    assert quantizer.call_count == (
        sum(n > 0 for n in expected_rows) if not prepacked and wire.quant_type else 0
    )

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU contracts at the production candidate-exchange collective boundary."""

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def exchange():
    # Execute the real exchange and gather adapter without importing GPU kernels.
    path = Path(__file__).parents[2] / "atom/model_ops/dcp_ops.py"
    names = {
        "_ag_custom_view_dtype",
        "dcp_all_gather",
        "dcp_local_logits_width",
        "dcp_decode_candidate_exchange_fused",
    }
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {
        "torch": torch,
        "PRESHUFFLE": "preshuffle",
        "MFMA": "mfma",
        "_AG_CUSTOM_DTYPES": (torch.float32, torch.float16, torch.bfloat16),
    }
    exec(  # noqa: S102 -- exercises production host code without GPU imports
        compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("rows", [0, 1, 6, 16, 128])
@pytest.mark.parametrize("backend", ["device", "group"])
def test_candidates_reach_merge_in_rank_order_without_a_copy(
    monkeypatch, exchange, world, rows, backend
):
    k = 8
    rank = world - 1
    columns = torch.arange(k).expand(rows, k)
    lengths = [(torch.arange(rows) + r) % (k + 1) for r in range(world)]
    sent = []
    for r in range(world):
        scores = torch.arange(rows * k, dtype=torch.float32).reshape(rows, k) + r * 4096
        scores[columns >= lengths[r][:, None]] = -torch.inf
        sent.append(scores)
    if rows:
        sent[1][0, 0] = -0.0
    local_indices = columns.to(torch.int32).masked_fill(
        columns >= lengths[rank][:, None], -1
    )
    seen = {}

    def topk(*args, values, **kwargs):
        args[3].copy_(local_indices)
        values.copy_(sent[rank])
        assert kwargs["stable"]

    def gather(tensor, dim):
        assert torch.equal(tensor.view(torch.int32), sent[rank].view(torch.int32))
        seen["dim"] = dim
        seen["gathered"] = torch.cat(sent, dim=dim)
        return seen["gathered"]

    def merge(scores, indices, tables, out, indptr, counts, staging, r, w, *args):
        seen.update(scores=scores, indices=indices, tables=tables, rank=r, world=w)
        assert out is output and indptr is offsets and counts is owned

    group = SimpleNamespace(all_gather=gather)
    if backend == "device":
        group.device_communicator = SimpleNamespace(all_gather=gather)
        group.all_gather = lambda *a, **kw: pytest.fail(
            "device communicator was bypassed"
        )
    exchange.update(
        get_dcp_world_size=lambda: world,
        get_dcp_group=lambda: group,
        dcp_local_context_lens=lambda *args: lengths[rank].to(torch.int32),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.topk",
        SimpleNamespace(top_k_per_row_decode=topk, flydsl_dcp_topk_merge=merge),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.triton.pa_mqa_logits",
        SimpleNamespace(deepgemm_fp8_paged_mqa_logits=lambda *a, **kw: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.fp4_mqa_ragged_metadata",
        SimpleNamespace(Fp4MqaRaggedMetadata=object),
    )
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        return empty(*args, **dict(kwargs, device="cpu"))

    monkeypatch.setattr(torch, "empty", cpu_empty)
    tables = torch.arange(rows, dtype=torch.int32).reshape(rows, 1)
    output = torch.empty(rows * k, dtype=torch.int32)
    offsets = torch.empty(rows + 1, dtype=torch.int32)
    owned = torch.empty(rows, dtype=torch.int32)
    exchange["dcp_decode_candidate_exchange_fused"](
        SimpleNamespace(token_block_tables=tables),
        torch.empty(rows, 1, 2, 8),
        kv_cache=torch.empty(0),
        weights=torch.empty(rows, 2),
        dcp_rank=rank,
        num_decode_tokens=rows,
        topk_tokens=k,
        max_model_len=128,
        runner_block_size=16,
        stable_topk=True,
        cp_kv_cache_interleave_size=1,
        out_kv_indices=output,
        out_kv_indptr=offsets,
        owned_counts=owned,
    )
    expected = torch.stack(sent).permute(1, 0, 2).reshape(rows, world * k)
    assert seen["dim"] == -1
    assert seen["scores"] is seen["gathered"]
    assert seen["scores"].is_contiguous()
    assert torch.equal(seen["scores"].view(torch.int32), expected.view(torch.int32))
    assert torch.equal(seen["indices"], local_indices)
    assert torch.equal(seen["tables"], tables)
    assert (seen["rank"], seen["world"]) == (rank, world)

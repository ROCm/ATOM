# SPDX-License-Identifier: MIT
"""QSA's all-visible FlyDSL decode shortcut and graph replay."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.model_ops.qwen4_exp.ops.qsa import qsa_select_paged_tokens


def _case(context: int, rows: int):
    if not torch.cuda.is_available():
        pytest.skip("needs a ROCm GPU")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
        pytest.skip("the shortcut targets gfx950")
    page = 64
    pages = context // page
    torch.manual_seed(context + rows)
    table = (
        torch.randperm(pages * rows, device="cuda", dtype=torch.int64)
        .to(torch.int32)
        .view(rows, pages)
    )
    q = torch.randn(rows, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(
        pages * rows, page // 4, 1, 128, device="cuda", dtype=torch.bfloat16
    )
    requests = torch.arange(rows, device="cuda", dtype=torch.int32)
    lengths = torch.full((rows,), context, device="cuda", dtype=torch.int32)
    positions = lengths.to(torch.int64) - 1
    return q, cache, table, requests, positions, lengths


def _select(case, context: int, shortcut: bool):
    q, cache, table, requests, positions, lengths = case
    return qsa_select_paged_tokens(
        q,
        cache,
        table,
        requests,
        positions,
        lengths,
        2048,
        max_seq_len=context,
        allow_shortcut=shortcut,
    )


@pytest.mark.parametrize("context", [64, 512, 2048])
def test_all_visible_exact_for_ragged_and_padded_rows(context):
    case = _case(context, 8)
    _, _, _, requests, positions, lengths = case
    lengths[1] = 1
    lengths[2] = min(context, 3)
    lengths[3] = min(context, 63)
    positions[:4] = lengths[:4].to(torch.int64) - 1
    positions[4] = -1
    positions[5] = lengths[5]
    requests[6] = -1
    positions[7] = 2**32 + 1

    torch.testing.assert_close(
        _select(case, context, True), _select(case, context, False), atol=0, rtol=0
    )


def test_all_visible_graph_replay_reads_changed_metadata():
    context = 2048
    case = _case(context, 4)
    _, _, _, requests, positions, lengths = case
    _select(case, context, True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _select(case, context, True)

    lengths.copy_(torch.tensor([1023, 2048, 1, 2048], device="cuda", dtype=torch.int32))
    positions.copy_(torch.tensor([1022, 2047, 0, 7], device="cuda", dtype=torch.int64))
    requests[3] = -1
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, _select(case, context, False), atol=0, rtol=0)

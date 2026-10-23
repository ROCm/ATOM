# SPDX-License-Identifier: MIT
"""The stacked-head QSA decode scorer against the CUDA-core one and torch."""

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

HEADS, DIM, RATIO = 4, 128, 4


def _case(rows_per_request, page_size, contexts, padded_rows=3, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width = max(-(-(ctx // RATIO) // page_size) for ctx in contexts) + 2
    pages = len(contexts) * width + 5
    table = torch.full((len(contexts), width), -1, dtype=torch.int32, device="cuda")
    order = torch.randperm(pages, generator=g, device="cuda").to(torch.int32)
    for req, ctx in enumerate(contexts):
        used = -(-(ctx // RATIO) // page_size)
        table[req, :used] = order[req * width : req * width + used]
    keys = torch.randn(pages, page_size, 1, DIM, generator=g, device="cuda")
    keys = keys.to(torch.bfloat16)
    requests, positions = [], []
    for req, ctx in enumerate(contexts):
        requests += [req] * rows_per_request
        positions += range(ctx - rows_per_request, ctx)
    requests += [-1] * padded_rows
    positions += [0] * padded_rows
    rows = len(requests)
    q = torch.randn(rows, HEADS, DIM, generator=g, device="cuda").to(torch.bfloat16)
    token_to_request = torch.tensor(requests, dtype=torch.int32, device="cuda")
    query_positions = torch.tensor(positions, dtype=torch.int64, device="cuda")
    context_lens = torch.tensor(contexts, dtype=torch.int32, device="cuda")
    return q, keys, table, token_to_request, query_positions, context_lens


def _reference(q, keys, table, token_to_request, query_positions, context_lens):
    page_size = keys.shape[1]
    columns = table.shape[1] * page_size
    logits = torch.full((q.shape[0], columns), -float("inf"), device="cuda")
    visible = torch.zeros(q.shape[0], dtype=torch.int32, device="cuda")
    for row in range(q.shape[0]):
        req = int(token_to_request[row])
        if req < 0:
            continue
        n = min(
            (int(query_positions[row]) + 1) // RATIO, int(context_lens[req]) // RATIO
        )
        visible[row] = n
        groups = torch.arange(n, device="cuda")
        k = keys[table[req, groups // page_size].long(), groups % page_size, 0]
        scores = torch.relu(q[row].float() @ k.float().T).sum(0) / DIM**0.5
        logits[row, :n] = scores
    return logits, visible


def _seen(logits, visible):
    # Top-k reads a row only up to its visible count; nothing past it is kept.
    columns = torch.arange(logits.shape[1], device="cuda")
    return logits[columns[None, :] < visible[:, None]]


@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("rows_per_request", [1, 4])
def test_stacked_scorer_matches_cuda_core(monkeypatch, page_size, rows_per_request):
    from atom.model_ops.qwen4_exp.ops import qsa

    # Uneven contexts: a block of stacked rows spans requests and the widest
    # row sets the tile count, the narrower ones must still mask their tails.
    case = _case(rows_per_request, page_size, [4103, 37, 1990, 8, 12288, 515])
    ref, ref_visible = _reference(*case)
    out = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("ATOM_QSA_STACKED_DECODE_SCORER", flag)
        row_starts = torch.full(
            (case[0].shape[0],), 7, dtype=torch.int32, device="cuda"
        )
        logits, visible = qsa.qsa_paged_mqa_logits(
            *case, compress_ratio=RATIO, row_starts=row_starts
        )
        assert torch.equal(visible, ref_visible)
        assert not row_starts.any()
        stacked = qsa._qsa_paged_mqa_logits_stacked_kernel.device_caches
        assert flag == "0" or stacked[torch.cuda.current_device()][0]
        out[flag] = _seen(logits, visible)
        torch.testing.assert_close(out[flag], _seen(ref, visible), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out["1"], out["0"], rtol=5e-5, atol=5e-5)


def test_stacked_scorer_respects_max_columns(monkeypatch):
    from atom.model_ops.qwen4_exp.ops import qsa

    monkeypatch.setenv("ATOM_QSA_STACKED_DECODE_SCORER", "1")
    case = _case(4, 16, [9000, 600])
    ref, _ = _reference(*case)
    logits, visible = qsa.qsa_paged_mqa_logits(
        *case, compress_ratio=RATIO, max_columns=1000
    )
    assert logits.shape[1] == 1000
    visible = visible.clamp(max=1000)
    torch.testing.assert_close(
        _seen(logits, visible), _seen(ref[:, :1000], visible), rtol=1e-4, atol=1e-4
    )

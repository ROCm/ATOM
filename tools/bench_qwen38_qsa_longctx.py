# SPDX-License-Identifier: MIT
"""Synthetic 128k/256k Qwen3.8 QSA decode correctness and graph benchmark.

This exercises one unsplit QSA layer's compressed-index selection and sparse
paged GQA at the real 4-head/128-dim index and 24-head/256-dim query geometry.
It does not load a checkpoint or exercise context parallelism.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
import triton.testing

from atom.model_ops.qwen4_exp.ops.qsa import (
    qsa_select_paged_tokens,
    qsa_sparse_paged_gqa,
)


def _time(fn, repeats: int) -> float:
    fn()
    torch.cuda.synchronize()
    return round(
        statistics.median(
            triton.testing.do_bench_cudagraph(fn, rep=1) * 1000 for _ in range(repeats)
        ),
        3,
    )


def run(context: int, rows: int, repeats: int) -> dict:
    if context % 64 or context < 2048 or rows < 1:
        raise ValueError("context must be a multiple of 64 and >= 2048; rows >= 1")
    device, block, ratio, budget = "cuda", 64, 4, 2048
    pages = context // block
    torch.manual_seed(2026 + context)
    # A nontrivial physical page order tests both index and K/V translation.
    page_table = torch.randperm(pages, device=device, dtype=torch.int64).to(
        torch.int32
    )[None]
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.arange(context - rows, context, device=device, dtype=torch.int64)
    lengths = torch.tensor([context], device=device, dtype=torch.int32)
    index_q = torch.randn(rows, 4, 128, dtype=torch.bfloat16, device=device)
    compressed = torch.randn(
        pages, block // ratio, 1, 128, dtype=torch.bfloat16, device=device
    )
    query = torch.randn(rows, 24, 256, dtype=torch.bfloat16, device=device)
    keys = torch.randn(pages, block, 2, 256, dtype=torch.bfloat16, device=device)
    values = torch.randn_like(keys)

    def select():
        return qsa_select_paged_tokens(
            index_q,
            compressed,
            page_table,
            requests,
            positions,
            lengths,
            budget,
            ratio,
            max_seq_len=context,
        )

    selected = select()
    for row in range(rows):
        visible = selected[row][selected[row] >= 0]
        expected = budget + (int(positions[row]) + 1) % ratio
        if visible.numel() != expected:
            raise AssertionError(
                f"row {row}: got {visible.numel()} tokens, expected {expected}"
            )
        if visible.unique().numel() != expected or int(visible.max()) > int(
            positions[row]
        ):
            raise AssertionError(f"row {row}: duplicate or future selected token")
    output = qsa_sparse_paged_gqa(
        query, keys, values, selected, page_table, requests, num_decode_requests=1
    )
    # Independently score every compressed group for the last query. This
    # checks the paged index selection itself, including page translation.
    group_ids = torch.arange(context // ratio, device=device)
    index_pages = page_table[0, group_ids // (block // ratio)].long()
    compressed_keys = compressed[index_pages, group_ids % (block // ratio), 0].float()
    index_scores = torch.sum(
        compressed_keys[:, None, :] * index_q[-1].float()[None, :, :],
        dim=-1,
    ).clamp_min(0).sum(dim=-1) * (128**-0.5)
    expected_groups = torch.topk(index_scores, budget // ratio).indices.sort().values
    selected_groups = (selected[-1, :budget:ratio].long() // ratio).sort().values
    torch.testing.assert_close(selected_groups, expected_groups, atol=0, rtol=0)

    # Every row's selected K/V gathers provide an independent attention oracle.
    errors = []
    for row in range(rows):
        token_ids = selected[row][selected[row] >= 0].long()
        physical = page_table[0, token_ids // block].long()
        gathered_k = keys[physical, token_ids % block].float()
        gathered_v = values[physical, token_ids % block].float()
        q = query[row].float().reshape(2, 12, 256)
        scores = torch.einsum("ghd,lgd->ghl", q, gathered_k) * (256**-0.5)
        probabilities = torch.softmax(scores, dim=-1)
        reference = torch.einsum("ghl,lgd->ghd", probabilities, gathered_v).reshape(
            24, 256
        )
        errors.append(
            float(
                torch.linalg.norm(output[row].float() - reference)
                / torch.linalg.norm(reference)
            )
        )
    error = max(errors)
    if error > 0.01:
        raise AssertionError(f"QSA output NRMSE {error:.5f} exceeds 0.01")

    # Capture the selection and sparse attention together. Replay after new Q/K
    # values must consume the new values, including AITER's top-k scratch.
    select()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_selected = select()
        captured_output = qsa_sparse_paged_gqa(
            query,
            keys,
            values,
            captured_selected,
            page_table,
            requests,
            num_decode_requests=1,
        )
    index_q.copy_(torch.randn_like(index_q))
    query.copy_(torch.randn_like(query))
    graph.replay()
    torch.cuda.synchronize()
    expected_selected = select()
    expected_output = qsa_sparse_paged_gqa(
        query,
        keys,
        values,
        expected_selected,
        page_table,
        requests,
        num_decode_requests=1,
    )
    torch.testing.assert_close(captured_selected, expected_selected, atol=0, rtol=0)
    torch.testing.assert_close(captured_output, expected_output, atol=0, rtol=0)
    return {
        "context": context,
        "rows": rows,
        "selected_width": selected.shape[1],
        "selected_valid_last": selected[-1].ge(0).sum().item(),
        "independent_index_topk": True,
        "attention_max_nrmse": round(error, 6),
        "changed_input_graph_replay": True,
        "select_us": _time(select, repeats),
        "sparse_attention_us": _time(
            lambda: qsa_sparse_paged_gqa(
                query,
                keys,
                values,
                selected,
                page_table,
                requests,
                num_decode_requests=1,
            ),
            repeats,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[131072, 262144])
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "cases": [run(ctx, args.rows, args.repeats) for ctx in args.contexts],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

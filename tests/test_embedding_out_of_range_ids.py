# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A token id outside the table reads a zero row; it never reaches a raw gather.

A speculative decode can leave the -1 truncation sentinel in `input_ids`.
`F.embedding` hands it to index_select, whose bounds assert takes the rank
down: a Kimi-K3 DP8EP8 decode (attention TP1) with DSpark died that way, where
the TP>1 path, which masks, could not. Every lookup goes through the masked
kernel instead, which is bit-identical for every id inside the table.
"""

import pathlib
import re
from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import

ATOM = pathlib.Path(__file__).resolve().parents[1] / "atom"
RAW_GATHER = re.compile(r"\b(F|functional|torch)\.embedding\(")


def test_no_lookup_is_a_raw_gather():
    sources = sorted(ATOM.rglob("*.py"))
    assert sources, f"no sources under {ATOM}"
    offenders = [
        f"{path.relative_to(ATOM)}:{number}"
        for path in sources
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if RAW_GATHER.search(line.split("#", 1)[0])
    ]
    assert offenders == [], f"route these through the masked kernel: {offenders}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel needs a GPU")
def test_tp1_reads_a_zero_row_for_an_id_outside_the_table(monkeypatch):
    embed_head = gpu_stack_import("atom.model_ops.embed_head")
    monkeypatch.setattr(
        embed_head,
        "get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0, world_size=1),
    )
    vocab, hidden = 1000, 7168
    embedding = embed_head.VocabParallelEmbedding(vocab, hidden)
    weight = torch.randn(vocab, hidden, dtype=torch.bfloat16, device="cuda")
    embedding.weight.data = weight
    ids = torch.tensor(
        [5, -1, vocab, 7, vocab + 1000, 0, vocab - 1], dtype=torch.int32, device="cuda"
    )

    out = embedding(ids)

    inside = (ids >= 0) & (ids < vocab)
    assert torch.equal(out[inside], weight[ids[inside].long()])
    assert not out[~inside].any()

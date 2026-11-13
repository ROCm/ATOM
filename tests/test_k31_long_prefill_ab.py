"""CPU tests for `tools/k31_long_prefill_ab.py` on synthetic runs."""

import json
import math
import sys
from pathlib import Path

import pytest
import torch

from atom.model_engine.token_logprobs import pack_token_logprobs, unpack_token_logprobs

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import k31_long_prefill_ab as ab

TOP = {i: -0.25 * i for i in range(20)}


def run(*tops, sha="p"):
    prompts = [
        {"offset": i, "sha256": f"{sha}{i}", "token": min(top, key=lambda t: -top[t])}
        | {"top_ids": list(top), "top_logprobs": list(top.values())}
        for i, top in enumerate(tops)
    ]
    return {"commit": "c", "prompts": prompts}


def test_top_of_reads_the_engine_logprob_record():
    ids = torch.tensor([[[7, 3, 11]]], dtype=torch.int32)
    values = torch.tensor([[[-0.5, -0.5, -1.25, -2.0]]])
    payload = pack_token_logprobs(values, ids, torch.tensor([[1]], dtype=torch.int32))
    record = unpack_token_logprobs(payload, [0], {0: [7]})[0][0]
    assert ab.top_of(record) == {
        "top_ids": [7, 3, 11],
        "top_logprobs": [-0.5, -1.25, -2.0],
    }


def test_compare_refuses_runs_over_different_prompts():
    with pytest.raises(ValueError, match="different prompts"):
        ab.compare(run(TOP), run(TOP, sha="q"))


def test_a_logprob_delta_counts_as_its_probability_delta():
    tail = TOP | {19: TOP[19] - 10}
    head = TOP | {1: TOP[1] - 0.5}
    rows = ab.compare(run(TOP, TOP), run(tail, head))
    assert [delta for *_, delta in rows] == pytest.approx(
        [math.exp(-4.75) - math.exp(-14.75), math.exp(-0.25) - math.exp(-0.75)]
    )


@pytest.mark.parametrize("below, status", [(0.0, 0), (1e-9, 1)])
def test_compare_fails_only_above_the_threshold(tmp_path, below, status):
    a, b = run(TOP, TOP), run(TOP, TOP | {5: TOP[5] - 0.5})
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    paths[0].write_text(json.dumps(a))
    paths[1].write_text(json.dumps(b))
    threshold = repr(ab.compare(a, b)[1][2] - below)
    assert ab.main(["compare", *map(str, paths), "--threshold", threshold]) == status


def test_a_token_missing_from_one_list_counts_only_the_proven_gap():
    # B swaps ids 18, 19 for 20, 21; each missing id is at most the other floor.
    swapped = {i: TOP[i] for i in range(18)} | {20: -4.5, 21: -4.75}
    gap = math.exp(-4.5) - math.exp(-4.75)
    assert ab.compare(run(TOP), run(swapped)) == [(True, 18, pytest.approx(gap))]


def test_disjoint_top_lists_fail_on_the_greedy_token():
    other = {100 + i: v for i, v in TOP.items()}
    gap = 1 - math.exp(-4.75)
    assert ab.compare(run(TOP), run(other)) == [(False, 0, pytest.approx(gap))]

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


def run(*tops):
    prompts = [
        {"offset": i, "sha256": f"p{i}", "token": min(top, key=lambda t: -top[t])}
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
    a, b = run(TOP, TOP), run(TOP, TOP)
    b["prompts"][1]["sha256"] = "q1"
    with pytest.raises(ValueError, match="different prompts"):
        ab.compare(a, b)


def test_a_logprob_delta_counts_as_its_probability_delta():
    tail = TOP | {19: TOP[19] - 10}
    head = TOP | {1: TOP[1] - 0.5}
    rows = ab.compare(run(TOP, TOP), run(tail, head))
    assert [row[2] for row in rows] == pytest.approx([10, 0.5])
    assert [delta for *_, delta in rows] == pytest.approx(
        [math.exp(-4.75) - math.exp(-14.75), math.exp(-0.25) - math.exp(-0.75)]
    )


@pytest.mark.parametrize("below, status", [(0.0, 0), (1e-9, 1)])
def test_compare_fails_only_above_the_threshold(tmp_path, below, status):
    a, b = run(TOP, TOP), run(TOP, TOP | {5: TOP[5] - 0.5})
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    paths[0].write_text(json.dumps(a))
    paths[1].write_text(json.dumps(b))
    threshold = repr(ab.compare(a, b)[1][-1] - below)
    assert ab.main(["compare", *map(str, paths), "--threshold", threshold]) == status


@pytest.mark.parametrize("ids", [range(20), range(100, 120)])
def test_a_nan_logprob_fails_the_prompt(tmp_path, ids):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(run(TOP)))
    b.write_text(json.dumps(run(dict.fromkeys(ids, math.nan))))
    assert ab.main(["compare", str(a), str(b)]) == 1


def test_a_token_missing_from_one_list_counts_only_the_proven_gap():
    # B swaps ids 18, 19 for 20, 21 and sits on a lower floor; each missing id
    # is at most the other list's floor.
    swapped = {i: TOP[i] for i in range(18)} | {20: -6.0, 21: -8.0}
    gap = math.exp(-4.5) - math.exp(-8.0)
    row = (True, 18, pytest.approx(3.5), pytest.approx(gap))
    assert ab.compare(run(TOP), run(swapped)) == [row]
    assert ab.compare(run(swapped), run(TOP)) == [row]


def test_disjoint_top_lists_fail_on_the_greedy_token():
    other = {100 + i: v for i, v in TOP.items()}
    gap = 1 - math.exp(-4.75)
    row = (False, 0, pytest.approx(4.75), pytest.approx(gap))
    assert ab.compare(run(TOP), run(other)) == [row]

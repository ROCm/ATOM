import json
from types import SimpleNamespace as NS

import torch

from atom.benchmarks.dspark_capture import install


def test_collector_pairs_by_request_after_reorder_and_keeps_zero_accept(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "atom.benchmarks.dspark_capture.calibration_identity", lambda *_: {"test": True}
    )
    block = NS(run=lambda: (None, torch.tensor([[0.1] * 5, [0.9] * 5])))
    builder = NS(commit_speculative_state=lambda *_: None)
    runner = NS(
        rank=0,
        device="cpu",
        config=NS(
            dspark=NS(confidence_schedule=False),
            hf_config=NS(model_type="deepseek_v41_text"),
            max_num_seqs=2,
        ),
        drafter=NS(block=block),
        attn_metadata_builder=builder,
    )

    def forward(batch):
        spans = [NS(request_id=r, length=6) for r in batch.req_ids]
        step = NS(decode=True, requests=spans, cu_seqlens_q=torch.tensor([0, 6, 12]))
        builder.commit_speculative_state(
            NS(step=step, cache=NS(pending=True)), torch.tensor([0, 11])
        )
        return block.run()

    runner.forward = forward
    install(runner, tmp_path)
    for _ in range(8):
        runner.forward(NS(is_dummy_run=False, total_seqs_num=2, req_ids=["a", "b"]))
    runner.forward(NS(is_dummy_run=False, total_seqs_num=2, req_ids=["b", "a"]))
    records = [
        json.loads(s) for s in (tmp_path / "confidence.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert records[0]["accepted"] == 0
    assert records[1]["accepted"] == 5
    assert abs(records[0]["confidence"][0] - 0.9) < 1e-6
    assert abs(records[1]["confidence"][0] - 0.1) < 1e-6
    assert all(set(row) == {"request", "confidence", "accepted"} for row in records)
    assert records[0]["request"] != records[1]["request"]


def test_nonzero_rank_does_not_create_output(tmp_path):
    runner = NS(
        rank=1,
        config=NS(
            dspark=NS(confidence_schedule=False),
            hf_config=NS(model_type="deepseek_v41_text"),
        ),
    )
    install(runner, tmp_path / "rank1")
    assert not (tmp_path / "rank1").exists()

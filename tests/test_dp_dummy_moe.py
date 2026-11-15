# SPDX-License-Identifier: MIT
"""An idle DP step keeps its execution shape but publishes no expert rows.

Load the runner's two methods from source to exercise them on CPU without
importing the GPU attention stack. The batches and token arrays are real.
"""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.sequence import Sequence


def _method(owner, name):
    path = Path(__file__).resolve().parents[1] / "atom/model_engine/model_runner.py"
    module = ast.parse(path.read_text())
    cls = next(
        n for n in module.body if isinstance(n, ast.ClassDef) and n.name == owner
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    method.decorator_list = []
    namespace = {
        "np": np,
        "torch": torch,
        "ScheduledBatch": ScheduledBatch,
    }
    exec(  # noqa: S102 -- execute the repository method without its GPU imports
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


dummy_execution = _method("ModelRunner", "dummy_execution")
prepare_input_ids = _method("tokenIDProcessor", "prepare_input_ids")


def _dummy(*, mtp_k=None, producer=False):
    runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(is_pd_producer=producer),
        forward=lambda batch: batch,
    )
    if mtp_k is not None:
        runner.drafter = SimpleNamespace(mtp_k=mtp_k)
    return dummy_execution(runner)


def _processor(draft_width):
    # The stage/publish boundary stands in for the H2D, using one persistent
    # tensor just as a captured graph's input does.
    counts = SimpleNamespace(
        np=np.full(3, -99, dtype=np.int32),
        gpu=torch.full((3,), -99, dtype=torch.int32),
    )

    def stage(batch, max_seqlen_q, group):
        counts.gpu.copy_(torch.from_numpy(counts.np))
        return torch.from_numpy(batch.scheduled_tokens.copy()), None

    return SimpleNamespace(
        batch_info=counts,
        draft_tokens_per_seq=draft_width,
        _stage_input_ids=stage,
        recv_spec_status=Mock(return_value=(None, None)),
    )


@pytest.mark.parametrize("max_seqlen_q", [1, 4, 16])
@pytest.mark.parametrize("mtp_k", [None, 0, 3])
@pytest.mark.parametrize("producer", [False, True])
def test_dp_dummy_schedules_nothing_and_publishes_zero_expert_rows(
    max_seqlen_q, mtp_k, producer
):
    batch = _dummy(mtp_k=mtp_k, producer=producer)
    assert not batch.is_dummy_run  # Reserved for startup profiling/warmup.
    assert batch.total_seqs_num == batch.total_tokens_num == 0
    assert batch.req_ids == []
    assert batch.num_scheduled_tokens.tolist() == []
    assert batch.num_spec_step == ((mtp_k or 0) if not producer else 0)

    processor = _processor(4)
    # A peer's padded query width must not turn local empty rows into work.
    ids = prepare_input_ids(processor, batch, max_seqlen_q, None)
    assert ids.shape == (0,)
    assert processor.batch_info.gpu.tolist() == [0, 0, 0]


@pytest.mark.parametrize("warmup", [False, True])
def test_real_dummy_real_reuses_counts_without_losing_real_or_warmup_rows(warmup):
    seqs = [Sequence([11, 12, 13], block_size=16) for _ in range(2)]
    batch = ScheduledBatch(
        seqs={seq.id: seq for seq in seqs},
        num_scheduled_tokens=[3, 3],
        total_tokens_num=6,
        total_tokens_num_prefill=6,
        total_seqs_num=2,
        total_seqs_num_prefill=2,
        is_dummy_run=warmup,
    )
    processor = _processor(4)
    tensor = processor.batch_info.gpu
    # Target uses entry 1, Eagle mid-steps entry 0, DSpark entry 2.
    pass_counts = [tensor[i : i + 1] for i in range(3)]
    for current, expected in (
        (batch, [2, 6, 8]),
        (_dummy(), [0, 0, 0]),
        (batch, [2, 6, 8]),
    ):
        ids = prepare_input_ids(processor, current, 1, None)
        assert torch.equal(ids, torch.from_numpy(current.scheduled_tokens))
        assert processor.batch_info.gpu is tensor
        assert [count.item() for count in pass_counts] == expected

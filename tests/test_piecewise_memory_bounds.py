# SPDX-License-Identifier: MIT
"""Live AF storage is bounded independently of the shapes seen while serving."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.metadata import BatchStep, RequestSpan
from atom.models.deepseek_v41 import bounded_replay
from atom.utils.cuda_graph import StableOutputs


def _step(lengths, buffers, namespace="main", value=0, device="cpu"):
    ends = np.cumsum(lengths)
    width = int(ends[-1])
    offsets = np.r_[0, ends[:-1]]
    step = BatchStep(
        tuple(
            RequestSpan(i, 20 + i, int(o), n, i)
            for i, (o, n) in enumerate(zip(offsets, lengths))
        ),
        torch.arange(width, dtype=torch.int64),
        torch.tensor(np.r_[0, ends], dtype=torch.int32),
        torch.arange(len(lengths), dtype=torch.int32),
        torch.repeat_interleave(
            torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
        ),
        torch.arange(len(lengths) * 3, dtype=torch.int32).view(-1, 3),
        scheduled=width,
        max_q_len=max(lengths),
        is_prefill=True,
        result_buffers=buffers,
        buffer_namespace=namespace,
        request_finals=np.ones(len(lengths), dtype=np.bool_),
        visible={1: torch.arange(width, dtype=torch.int32) + value},
    )
    # The builder publishes its allocation bounds; active lengths never set them.
    step.token_capacity, step.request_capacity = 64, 4
    step.selected[3] = (
        torch.arange(width * 2, dtype=torch.int32).view(1, width, 2) + value
    )
    step.candidates[3] = (
        torch.arange(width * 3, dtype=torch.int32).view(width, 3) + value
    )
    for name in ("positions", "cu_seqlens_q", "slots", "batch_ids", "block_tables"):
        setattr(step, name, getattr(step, name).to(device))
    for values in (step.visible, step.selected, step.candidates):
        for name, tensor in values.items():
            values[name] = tensor.to(device)
    return step


@pytest.fixture
def cpu_gather(monkeypatch):
    # Replace only the Triton transport; all planning, offsets and storage are real.
    monkeypatch.setattr(
        bounded_replay,
        "_gather_rows",
        lambda rows, *values, dim=0: tuple(
            v.index_select(dim, rows.long()) for v in values
        ),
    )


def _tail_values(plan):
    return (
        plan.tail_rows,
        plan.last_rows,
        plan.tail.positions,
        plan.tail.cu_seqlens_q,
        plan.tail.slots,
        plan.tail.batch_ids,
        plan.tail.block_tables,
        plan.tail.visible[1],
        plan.tail.swa_replay_start,
        plan.tail.swa_cached_until,
        plan.q39.rows,
        plan.q39.positions,
        plan.q39.index.batch_ids,
        plan.q39.ptr_rows,
        plan.tail.selected[3],
        plan.tail.candidates[3],
    )


def test_tail_shapes_reuse_storage_and_fixed_field_offsets(cpu_gather):
    buffers = StableOutputs()
    specs = [SimpleNamespace(layer_id=10, topk_owner=3, candidate_source=3)]
    pointers, footprints = [], []
    for i, lengths in enumerate(
        ((8,), (3, 5, 9), (1, 2), (16, 16), (7, 2, 3, 8), (8,))
    ):
        step = _step(lengths, buffers, value=i * 100)
        plan = bounded_replay.build_replay_plan(step, 6)
        reference = bounded_replay.build_replay_plan(
            replace(step, result_buffers=None), 6
        )
        bounded_replay.carry_selections(step, plan, specs)
        bounded_replay.carry_selections(step, reference, specs)
        for actual, expected in zip(_tail_values(plan), _tail_values(reference)):
            torch.testing.assert_close(actual, expected)
            assert actual.is_contiguous()
        pointers.append(tuple(t.data_ptr() for t in _tail_values(plan)))
        footprints.append(
            sum(t.untyped_storage().nbytes() for t in buffers._slots.values())
        )
    assert len(set(footprints)) == 1
    assert len(set(pointers)) == 1


def test_tail_namespaces_do_not_overwrite_each_other(cpu_gather):
    buffers = StableOutputs()
    first = bounded_replay.build_replay_plan(_step((8,), buffers, "first", 10), 6)
    saved = first.tail.visible[1].clone()
    second = bounded_replay.build_replay_plan(_step((8,), buffers, "second", 30), 6)
    assert first.tail.visible[1].data_ptr() != second.tail.visible[1].data_ptr()
    torch.testing.assert_close(first.tail.visible[1], saved)


def test_tail_refuses_an_undersized_capacity(cpu_gather):
    step = _step((8,), StableOutputs())
    step.token_capacity = 3
    with pytest.raises(ValueError, match="capacity"):
        bounded_replay.build_replay_plan(step, 6)


def test_one_token_tails_remain_prefill(cpu_gather):
    plan = bounded_replay.build_replay_plan(_step((1, 1), StableOutputs()), 6)
    assert not plan.tail.decode
    assert not plan.q39.index.decode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_recorded_tail_consumers_read_current_values_across_shapes():
    buffers = StableOutputs()
    specs = [SimpleNamespace(layer_id=10, topk_owner=3, candidate_source=3)]
    recorded = {}
    footprints = []
    for i, lengths in enumerate(((3, 5), (8,), (2, 3, 6), (3, 5), (8,))):
        step = _step(lengths, buffers, value=100 * i, device="cuda")
        plan = bounded_replay.build_replay_plan(step, 6)
        bounded_replay.carry_selections(step, plan, specs)
        values = _tail_values(plan)
        expected = torch.cat([t.flatten() for t in values])
        if lengths not in recorded:
            result = torch.empty_like(expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result.copy_(torch.cat([t.flatten() for t in values]))
            recorded[lengths] = graph, result
        graph, result = recorded[lengths]
        graph.replay()
        torch.testing.assert_close(result, expected)
        footprints.append(
            sum(t.untyped_storage().nbytes() for t in buffers._slots.values())
        )
    assert len(set(footprints)) == 1

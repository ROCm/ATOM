"""PAGE accounting consumes current main's actual native plan publications."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.models.deepseek_v41.owner_service.owner_serving_plan import written_page_units

PLAN = (
    Path(__file__).resolve().parents[3] / "atom/model_ops/v4_kernels/compress_plan.py"
)
spec = importlib.util.spec_from_file_location("_owner_native_compress_plan", PLAN)
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class Buffer:
    def __init__(self):
        self.np = np.full((128, 4), 999, dtype=np.int32)
        self.gpu = torch.from_numpy(self.np)

    def copy_to_gpu(self, count):
        return self.gpu[:count]


def publication(lengths, ends, *, buffers=None, floors=None):
    if buffers is None:
        buffers = {
            ratio: {"compress": Buffer(), "write": Buffer()} for ratio in (1, 2, 4)
        }
    plans = native.make_compress_plans(
        np.asarray(lengths, dtype=np.int32),
        np.asarray(ends, dtype=np.int32),
        ((1, False), (2, False), (4, True)),
        plan_buffers=buffers,
        running_bs=8,
        max_q_len=6,
        extra_write=5,
        write_from=None if floors is None else np.asarray(floors, dtype=np.int32),
    )
    step = SimpleNamespace(
        requests=[SimpleNamespace(request_id=10 + i) for i in range(len(lengths))],
        plans=plans,
    )
    return step, buffers


def pages(step, buffers):
    tables = [
        tuple(100 * (i + 1) + j for j in range(64)) for i in range(len(step.requests))
    ]
    return written_page_units(
        step,
        tables,
        SimpleNamespace(rows_per_page=lambda ratio: 8 // ratio),
        compress_buffers={ratio: row["compress"] for ratio, row in buffers.items()},
    )


def test_current_native_plan_tracks_ragged_page_boundaries_without_cpu_plan_field():
    step, buffers = publication([3, 1, 6], [10, 16, 27])
    assert all(not hasattr(plan, "compress_plan_cpu") for plan in step.plans.values())
    assert pages(step, buffers) == {10: (100, 101), 11: (201,), 12: (302, 303)}


def test_reused_publication_ignores_old_rows_and_empty_request_padding():
    _, buffers = publication([6] * 8, [64] * 8)
    step, _ = publication([1, 0], [1, 99], buffers=buffers)
    assert pages(step, buffers) == {10: (100,), 11: ()}
    empty, _ = publication([], [], buffers=buffers)
    assert pages(empty, buffers) == {}


def test_native_write_floor_excludes_recomputed_prefix_pages():
    step, buffers = publication([6, 6], [18, 34], floors=[16, 34])
    assert pages(step, buffers) == {10: (102,), 11: ()}


def test_wrong_publication_buffer_is_rejected_before_page_accounting():
    step, buffers = publication([3], [10])
    buffers[1]["compress"] = Buffer()
    with pytest.raises(ValueError, match="this step's native"):
        pages(step, buffers)

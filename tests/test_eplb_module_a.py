# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-A statistics only)

import pytest

torch = pytest.importorskip("torch")

import atom.model_ops.eplb as eplb


class _FakeTPGroup:
    def __init__(self, world_size: int = 1):
        self.world_size = world_size

    def all_reduce(self, tensor, ca_fp8_quant=False):  # pragma: no cover
        # For unit tests we only need deterministic pass-through semantics.
        _ = ca_fp8_quant
        return tensor


def test_count_physical_load_filters_invalid_ids():
    topk = torch.tensor(
        [
            [0, 1, 2],
            [2, -1, 8],  # -1 and 8 are invalid for num_physical=4
        ],
        dtype=torch.int32,
    )
    counts = eplb.count_physical_load(topk, num_physical=4)
    assert counts.tolist() == [1, 1, 2, 0]


def test_monitor_window_accumulate_and_skip_dummy(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=3)

    # pass-1 (real): [2,1,1,0]
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 0], [1, 2]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)
    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out.shape == (1, 4)
    assert out[0].tolist() == [2, 1, 1, 0]

    # pass-2 (dummy): should not be appended into window.
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[3, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=True)
    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out[0].tolist() == [2, 1, 1, 0]

    # pass-3 (real): add [0,3,0,1] => total [2,4,1,1]
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[1, 1], [1, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)
    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out[0].tolist() == [2, 4, 1, 1]


def test_monitor_capacity_growth_preserves_existing_window(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)

    # first real pass on layer-0, width=2
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)

    # second real pass grows to layer-2 and width=4
    monitor.on_forward_start()
    monitor.record(
        layer_id=2,
        topk_physical=torch.tensor([[3, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)

    out = monitor.dump_global_physical_load()
    assert out is not None
    # layer-0 keeps its previous record after growth.
    assert out[0].tolist() == [1, 1, 0, 0]
    # layer-2 has new counts.
    assert out[2].tolist() == [0, 0, 0, 2]

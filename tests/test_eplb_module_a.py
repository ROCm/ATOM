# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py (Module-A: ExpertLoadMonitor)

import pytest

torch = pytest.importorskip("torch")

# Import atom.config first so it is fully initialized before atom.model_ops's
# __init__ chain references get_current_atom_config (avoids a mainline circular
# import that only surfaces when atom.model_ops is the entry-point import).
try:
    import atom.config  # noqa: F401
    import atom.model_ops.eplb as eplb
except Exception as _e:  # aiter/triton absent under bare non-GPU pytest
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)


class _FakeTPGroup:
    def __init__(self, world_size: int = 1):
        self.world_size = world_size

    def all_reduce(self, tensor, ca_fp8_quant=False):  # pragma: no cover
        # For unit tests we only need deterministic pass-through semantics.
        _ = ca_fp8_quant
        return tensor


def _init_monitor(monitor, *, num_layers=1, num_physical=4, device=None):
    if device is None:
        device = torch.device("cpu")
    monitor.initialize(num_layers=num_layers, num_physical=num_physical, device=device)
    return monitor


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
    _init_monitor(monitor, num_layers=1, num_physical=4)

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


def test_monitor_preallocated_capacity_covers_all_layers(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))

    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    _init_monitor(monitor, num_layers=3, num_physical=4)

    # first real pass on layer-0
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)

    # second real pass uses the preallocated layer-2 slot.
    monitor.on_forward_start()
    monitor.record(
        layer_id=2,
        topk_physical=torch.tensor([[3, 3]], dtype=torch.int32),
        num_physical=4,
    )
    monitor.on_forward_end(is_dummy_run=False)

    out = monitor.dump_global_physical_load()
    assert out is not None
    assert out.shape == (3, 4)
    assert out[0].tolist() == [1, 1, 0, 0]
    assert out[2].tolist() == [0, 0, 0, 2]


def test_monitor_record_rejects_uninitialized():
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    with pytest.raises(AssertionError, match="before initialization"):
        monitor.record(
            layer_id=0,
            topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
            num_physical=2,
        )


def test_monitor_initialize_rejects_runtime_resize():
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    _init_monitor(monitor, num_layers=1, num_physical=2)
    with pytest.raises(RuntimeError, match="already initialized"):
        monitor.initialize(num_layers=2, num_physical=2, device=torch.device("cpu"))


def test_count_physical_load_rejects_float_dtype():
    bad = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    with pytest.raises(AssertionError):
        eplb.count_physical_load(bad, num_physical=4)


def test_monitor_record_rejects_new_layer_after_initialization(monkeypatch):
    monkeypatch.setattr(eplb, "get_tp_group", lambda: _FakeTPGroup(world_size=1))
    monitor = eplb.ExpertLoadMonitor(enabled=True, window_size=2)
    _init_monitor(monitor, num_layers=1, num_physical=2)
    monitor.on_forward_start()
    monitor.record(
        layer_id=0,
        topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
        num_physical=2,
    )
    monitor.on_forward_end(is_dummy_run=False)
    with pytest.raises(AssertionError, match="outside initialized capacity"):
        monitor.record(
            layer_id=1,
            topk_physical=torch.tensor([[0, 1]], dtype=torch.int32),
            num_physical=2,
        )


# NOTE: Module-B (EPLBManager scheduler) tests live in test_eplb_module_b.py.
# The previous duplicates here were removed: they predated the warm-start
# scheduler (`first_window = interval // 4`) and the removal of the legacy
# owner-hook rebalance scaffold, so they no longer reflect the code.

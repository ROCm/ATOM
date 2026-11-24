import importlib.util
import pathlib

_SPEC = importlib.util.spec_from_file_location(
    "dp_sync_gate",
    pathlib.Path(__file__).resolve().parents[1] / "atom/model_engine/dp_sync_gate.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
DPSyncGate = _MOD.DPSyncGate

# The loop that drives the gate is `tests/test_dp_sync_interval.py`'s; these pin
# the gate's own arithmetic.


def test_syncs_until_a_sync_sees_work():
    gate = DPSyncGate(4)
    assert gate.should_sync()
    gate.synced(has_unfinished=False, shutdown=False, held=False)
    assert gate.should_sync()


def test_after_busy_sync_skips_until_n_busy_rounds():
    gate = DPSyncGate(4)
    gate.synced(has_unfinished=True, shutdown=False, held=False)
    seen = []
    for _ in range(8):
        seen.append(gate.should_sync())
        if seen[-1]:
            gate.synced(has_unfinished=True, shutdown=False, held=False)
        gate.busy_round_done()
    assert seen == [True, False, False, False] * 2


def test_shutdown_or_hold_syncs_every_round():
    for kw in ({"shutdown": True, "held": False}, {"shutdown": False, "held": True}):
        gate = DPSyncGate(8)
        gate.synced(has_unfinished=True, **kw)
        gate.busy_round_done()
        assert gate.should_sync()

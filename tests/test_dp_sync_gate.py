import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "dp_sync_gate",
    pathlib.Path(__file__).resolve().parents[1] / "atom/model_engine/dp_sync_gate.py",
)
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
DPSyncGate = _MOD.DPSyncGate


def simulate(interval, ranks, rounds, unfinished, held_cmd_at=None, shutdown_at=None):
    """Mirror DPEngineCoreProc.busy_loop's control flow for `ranks` lockstep ranks.

    unfinished(rank, round) -> local work. A held command queued at
    `held_cmd_at` is applied, like a utility command, on the next round that
    syncs. Returns per-round records; asserts the ranks never disagree on a
    collective.
    """
    gates = [DPSyncGate(interval) for _ in range(ranks)]
    pending_held = [False] * ranks
    held = [False] * ranks
    shutdown = [False] * ranks
    log = []
    for t in range(rounds):
        sync_now = [g.should_sync() for g in gates]
        assert len(set(sync_now)) == 1, f"ranks disagree on syncing at round {t}"
        if held_cmd_at is not None and t >= held_cmd_at:
            pending_held = [True] * ranks
        if shutdown_at is not None and t >= shutdown_at:
            shutdown = [True] * ranks
        if sync_now[0]:
            for r in range(ranks):
                if pending_held[r]:
                    held[r] = True
            local = [
                (unfinished(r, t) and not held[r], shutdown[r], held[r])
                for r in range(ranks)
            ]
            g_state = tuple(any(x[i] for x in local) for i in range(3))
            for g in gates:
                g.synced(*g_state)
        else:
            g_state = (True, False, False)
        has_unf, g_shut, g_held = g_state
        if g_shut and not has_unf:
            log.append((t, sync_now[0], "exit"))
            break
        if g_held:
            log.append((t, sync_now[0], "held"))
            continue
        if not has_unf:
            log.append((t, sync_now[0], "idle"))
            continue
        assert not any(held), f"a held rank ran a forward at round {t}"
        for g in gates:
            g.busy_round_done()
        log.append((t, sync_now[0], "busy"))
    return log


def test_interval_one_syncs_every_round():
    log = simulate(1, 4, 50, lambda r, t: t % 7 != 0)
    assert all(s for _, s, _ in log)


def test_busy_syncs_every_n_rounds():
    log = simulate(4, 16, 40, lambda r, t: True)
    assert [t for t, s, _ in log if s] == list(range(0, 40, 4))
    assert all(k == "busy" for *_, k in log)


def test_idle_syncs_every_round():
    log = simulate(8, 4, 30, lambda r, t: False)
    assert all(s and k == "idle" for _, s, k in log)


def test_busy_idle_busy_stays_aligned():
    # rank r busy on its own windows; ranks drop out at different rounds
    def unf(r, t):
        return t < 10 + 3 * r or 60 <= t < 75 + r

    log = simulate(16, 4, 120, unf)
    kinds = [k for *_, k in log]
    assert "busy" in kinds and "idle" in kinds
    # after going idle, the first busy round is a sync round
    for i in range(1, len(log)):
        if log[i][2] == "busy" and log[i - 1][2] == "idle":
            assert log[i][1]


@pytest.mark.parametrize("cmd_round", [0, 1, 5, 7, 13])
def test_held_applies_on_sync_round_and_stops_forwards(cmd_round):
    log = simulate(8, 4, 40, lambda r, t: True, held_cmd_at=cmd_round)
    first_held = next(t for t, _, k in log if k == "held")
    assert first_held - cmd_round < 8
    assert all(k == "held" for t, _, k in log if t >= first_held)


def test_shutdown_waits_for_sync_and_idle():
    log = simulate(8, 4, 60, lambda r, t: t < 20, shutdown_at=3)
    assert log[-1][2] == "exit"
    assert log[-1][1]


def test_gate_clamps_interval():
    assert DPSyncGate(0).interval == 1
    assert DPSyncGate(-5).should_sync()

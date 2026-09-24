"""CPU regression for SP collective input registration during graph capture."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from atom.distributed import ulysses_sp


def test_registration_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("ATOM_SP_REGISTER_GRAPH_INPUTS", raising=False)
    assert not ulysses_sp.envs.environment_variables["ATOM_SP_REGISTER_GRAPH_INPUTS"]()


def _group(actions, *, disabled=False, already_capturing=False):
    ca = SimpleNamespace(disabled=disabled, _IS_CAPTURING=already_capturing)

    @contextmanager
    def capture(context):
        assert not ca._IS_CAPTURING
        ca._IS_CAPTURING = True
        actions.append(("enter", context))
        try:
            yield context
        finally:
            ca._IS_CAPTURING = False
            actions.append(("flush", context))

    return SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=ca),
                           graph_capture=capture), ca


def _enable(monkeypatch, group):
    monkeypatch.setenv("ATOM_SP_REGISTER_GRAPH_INPUTS", "1")
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", 4)
    monkeypatch.setattr(ulysses_sp, "get_sp_group", lambda: group)


@pytest.mark.parametrize("enabled,world_size", [(False, 4), (True, 1)])
def test_disabled_paths_do_not_resolve_group(monkeypatch, enabled, world_size):
    monkeypatch.setenv("ATOM_SP_REGISTER_GRAPH_INPUTS", "1" if enabled else "0")
    monkeypatch.setattr(ulysses_sp, "_SP_WORLD_SIZE", world_size)

    def forbidden():
        raise AssertionError("disabled path must not initialize or resolve a communicator")

    monkeypatch.setattr(ulysses_sp, "get_sp_group", forbidden)
    with ulysses_sp.sp_graph_capture(object()):
        pass


@pytest.mark.parametrize("group", [SimpleNamespace(),
                                  SimpleNamespace(device_communicator=SimpleNamespace(ca_comm=None))])
def test_unavailable_communicator_is_noop(monkeypatch, group):
    _enable(monkeypatch, group)
    with ulysses_sp.sp_graph_capture(object()):
        pass


@pytest.mark.parametrize("disabled,active", [(True, False), (False, True)])
def test_inactive_or_already_enclosed_communicator_is_noop(monkeypatch, disabled, active):
    actions = []
    group, ca = _group(actions, disabled=disabled, already_capturing=active)
    _enable(monkeypatch, group)
    with ulysses_sp.sp_graph_capture(object()):
        assert ca._IS_CAPTURING == active
    assert not actions
    assert ca._IS_CAPTURING == active


def test_sp_reuses_outer_stream_context_and_flushes_once(monkeypatch):
    actions = []
    group, ca = _group(actions)
    _enable(monkeypatch, group)
    context = object()
    with ulysses_sp.sp_graph_capture(context) as received:
        assert received is context
        assert ca._IS_CAPTURING
        # A nested user must neither capture twice nor flush the outer context.
        with ulysses_sp.sp_graph_capture(context):
            assert ca._IS_CAPTURING
        assert actions == [("enter", context)]
    assert not ca._IS_CAPTURING
    assert actions == [("enter", context), ("flush", context)]


def test_capture_failure_exits_registration_context(monkeypatch):
    actions = []
    group, ca = _group(actions)
    _enable(monkeypatch, group)
    context = object()
    with pytest.raises(RuntimeError, match="capture aborted"):
        with ulysses_sp.sp_graph_capture(context):
            raise RuntimeError("capture aborted")
    assert not ca._IS_CAPTURING
    assert actions == [("enter", context), ("flush", context)]

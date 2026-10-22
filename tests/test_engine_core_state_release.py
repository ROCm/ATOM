# SPDX-License-Identifier: MIT
"""A P/D consumer whose state slots are all held must not stall.

`BlockManager.advance_state_release_lag` holds a finished request's slot until
an output-reading postprocess confirms the forward that may still write it.
When admission fails because every slot is held, the scheduler returns an
empty batch: no forward runs, so the engine step itself has to drain the
workers and release the slots, or the queued request never starts.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
# The engine core reaches AITER through the model runner it imports.
pytest.importorskip("aiter", reason="atom.model_engine.engine_core imports AITER")

from atom.model_engine.engine_core import EngineCore


class _BlockManager:
    state_release_lag = True

    def __init__(self):
        self.held = True
        self.released_with = []

    @property
    def has_held_state_slots(self):
        return self.held

    def advance_state_release_lag(self, confirmed):
        self.released_with.append(confirmed)
        if confirmed is None:
            self.held = False


def _engine(empty_batch):
    core = EngineCore.__new__(EngineCore)
    calls = []
    bm = _BlockManager()
    core.scheduler = SimpleNamespace(
        block_manager=bm,
        schedule=lambda: (empty_batch, {}),
        take_rejected=list,
        compute_detailed_aggregates=lambda *a: None,
        metrics=SimpleNamespace(record_forward=lambda *a: None),
        prefill_delayer=None,
    )
    core.runner_mgr = SimpleNamespace(
        call_func=lambda name, *a, **k: calls.append(name) or True
    )
    core.kv_transfer_enabled = False
    core.label = "test"
    core._poll_kv_transfer_progress = lambda: None
    core._advance_idle_kv_transfer = lambda *a, **k: None
    return core, bm, calls


def test_an_empty_batch_drains_and_releases_held_slots():
    batch = SimpleNamespace(req_ids=[], connector_meta_output=None)
    core, bm, calls = _engine(batch)
    assert core._process_engine_step_inner() is False
    assert calls == ["drain_device"]
    assert bm.released_with == [None] and not bm.has_held_state_slots


def test_nothing_held_means_no_drain():
    batch = SimpleNamespace(req_ids=[], connector_meta_output=None)
    core, bm, calls = _engine(batch)
    bm.held = False
    core._process_engine_step_inner()
    assert calls == []

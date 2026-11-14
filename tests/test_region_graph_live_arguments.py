# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""What a token graph's recording keeps of an eager region's arguments.

A replay calls the region with the arguments its ``live`` reads afresh and
the rest as recorded. DeepSeek-V4.1's regions take the step live; a recording
that kept the recorded step kept that bucket's step tables alive beside the
shared graph pool, one set per bucket. Runs without a GPU: the segment
captures around the call are stubbed; the region call, what is recorded of
it and its replay are the real code.
"""

import contextlib
import gc
import weakref

import pytest
import torch

from atom.utils import region_graph

# What the live step reads at the time: the recorded forward's step, then
# the step being served.
LIVE = {}


class Step:
    def __init__(self, name):
        self.name = name


class Owner:
    def __init__(self):
        self.calls = []

    @region_graph.eager_region(live=lambda: {"step": LIVE["step"]})
    def takes_step_live(self, step, scale):
        self.calls.append((step.name, scale))

    @region_graph.eager_region()
    def takes_nothing_live(self, step, scale):
        self.calls.append((step.name, scale))


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setattr(region_graph._Recorder, "begin", lambda self: None)
    monkeypatch.setattr(region_graph._Recorder, "end", lambda self: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    recording = region_graph._Recorder(pool=None)
    monkeypatch.setattr(region_graph._local, "recorder", recording, raising=False)
    yield recording
    LIVE.clear()


def _record(owner, region):
    """One recorded call of ``region`` on a step that only the recording
    can still hold afterwards; a weak reference to that step."""
    step = LIVE["step"] = Step("recorded")
    getattr(owner, region)(step, 2.0)
    LIVE["step"] = Step("served")
    held = weakref.ref(step)
    del step
    gc.collect()
    return held


def test_a_recording_keeps_no_argument_its_region_takes_live(recorder):
    owner = Owner()
    held = _record(owner, "takes_step_live")
    assert held() is None
    recorder._steps[-1]()
    assert owner.calls == [("recorded", 2.0), ("served", 2.0)]


def test_a_region_without_live_arguments_replays_what_it_recorded(recorder):
    owner = Owner()
    held = _record(owner, "takes_nothing_live")
    assert held() is not None
    recorder._steps[-1]()
    assert owner.calls == [("recorded", 2.0), ("recorded", 2.0)]

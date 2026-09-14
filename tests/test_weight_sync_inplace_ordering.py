# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An FP8 weight is not overwritten before its readers are done.

The update writes the parameter buffer in place so a captured decode graph
keeps reading a valid address, and the price is writing into a buffer that may
still be in use. What matters is that the wait sits with the write: waiting
once per update instead still lost five of seven weight syncs in a DAPO smoke.

So assert the ordering, not the wait. Both in-place writers -- the requantize
and the layout post-process -- must wait first, and the wait must name the
parameter's own device, because a colocated replica is not always on device 0.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.rollout.weight_updater import WeightUpdaterMixin


def _updater():
    class _Updater(WeightUpdaterMixin):
        device = torch.device("cpu")
        label = "test"
        rank = 0
        world_size = 1

    return _Updater()


def _fp8_module(quant_type=None):
    """A module shaped like the FP8 linear the update path recognises."""
    param = nn.Parameter(torch.zeros(4, 4, dtype=torch.float32), requires_grad=False)

    class _Module(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = param
            self.weight_scale = nn.Parameter(torch.ones(1), requires_grad=False)
            self.quant_type = quant_type

    return _Module(), param


def test_post_process_waits_before_rewriting_the_layout(monkeypatch):
    """`_post_process_fp8_weight` is the one every call site reaches."""
    events = []
    updater = _updater()
    module, param = _fp8_module()

    import atom.rollout.weight_updater as wu

    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_await_readers_of",
        lambda self, p: events.append("wait"),
        raising=True,
    )
    # quant_type None returns before any shuffle, which is enough: the wait is
    # supposed to happen before the function decides anything.
    updater._post_process_fp8_weight(module, param)

    assert events == ["wait"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the quant_type dispatch imports aiter, which reads the chip arch "
    "out of rocminfo",
)
def test_requantize_waits_before_the_first_write(monkeypatch):
    events = []
    updater = _updater()
    module, param = _fp8_module()

    import atom.rollout.weight_updater as wu

    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_await_readers_of",
        lambda self, p: events.append("wait"),
        raising=True,
    )
    monkeypatch.setattr(
        wu.WeightUpdaterMixin,
        "_post_process_fp8_weight",
        lambda self, m, p: events.append("post"),
        raising=True,
    )

    def _copy_(self, other, *a, **k):
        events.append("write")
        return self

    monkeypatch.setattr(torch.Tensor, "copy_", _copy_, raising=True)

    updater._requantize_fp8_weight(module, "weight", param, torch.zeros(4, 4))

    assert events and events[0] == "wait"
    assert "write" not in events[: events.index("wait")]


def test_the_wait_is_a_no_op_off_device():
    """The mixin's methods run unbound on CPU stand-ins throughout the tests,
    and a host that never allocated on a device has nothing to wait for."""
    calls = []
    torch_sync = torch.cuda.synchronize
    try:
        torch.cuda.synchronize = lambda *a, **k: calls.append(a)
        WeightUpdaterMixin._await_readers_of(
            SimpleNamespace(), nn.Parameter(torch.zeros(2), requires_grad=False)
        )
    finally:
        torch.cuda.synchronize = torch_sync

    assert calls == []


def test_the_wait_targets_the_parameters_own_device(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda d=None: calls.append(d))

    param = SimpleNamespace(device=torch.device("cuda", 3))
    WeightUpdaterMixin._await_readers_of(SimpleNamespace(), param)

    assert calls == [torch.device("cuda", 3)]

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A rank-local allocation failure in `Glm5MonoKernel.__init__` is agreed with
the peers, never unwound past them.

`SymmetricPeerBuffer`'s constructor all-gathers IPC handles and the agreement
below it all-gathers error reports: two collectives inside one `__init__`. A
rank that raises before reaching them leaves for the runner's
`tp_uniform_local_validation` while its peers sit inside one of the two, and
a collective ORDER mismatch hangs -- it does not fall back.

`scratch` and `step` were already inside the agreed attempt. The weight
packing, the index params and the identity block tables were not: they ran
earlier and straight through, so an OOM in any of them was exactly that hang,
a few statements further up. They allocate from the same pool on the same
device, and `pack_fp8` of the indexer weights is much the largest of them, so
they are where an OOM is most likely rather than least.

These are ordering tests, which is why they assert on *which collective ran*
rather than on the exception alone. An implementation that catches the failure
and re-raises it without reporting passes an exception-only test and still
hangs the group.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("flydsl", reason="the GLM mono op is FlyDSL source")
pytest.importorskip("aiter", reason="the GLM mono op imports aiter")

from atom.models.glm52.mono.kernels import op
from atom.models.glm52.mono.kernels.geometry_config import glm5_tp_config
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.weights.layer import LayerWeights

INDEXER_WEIGHTS = (
    "w_index_k",
    "s_index_k",
    "w_index_w",
    "w_index_q",
    "s_index_q",
    "g_index_k",
    "b_index_k",
)


class _Boom(Exception):
    """Stands in for `torch.OutOfMemoryError`, which is what actually happens
    here. A distinct type so the shared-runtime test can prove the ORIGINAL
    exception survives rather than being flattened into `MonoUnsupported`."""


@pytest.fixture
def events(monkeypatch):
    """Record every collective `__init__` reaches, in order.

    Nothing here touches a device: `torch.device("cuda", 0)` is only a
    descriptor, so pinning `current_device` is enough to get through the
    constructor's preamble on a box with no GPU, and every test below fails
    its allocation before anything is really allocated.
    """
    seen = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def _agree(reports, obj, group=None):
        seen.append("agree")
        reports[:] = [obj] + [None] * (len(reports) - 1)

    def _peer_buffer(*args, **kwargs):
        seen.append("peer_buffer")
        return SimpleNamespace(storage=None, local_address=0, addresses=[])

    monkeypatch.setattr(torch.distributed, "all_gather_object", _agree)
    monkeypatch.setattr(op, "SymmetricPeerBuffer", _peer_buffer)
    return seen


def _weights():
    """A TP4 rank's `LayerWeights`.

    The three packed indexer tensors carry dtypes and a shape their packer
    really accepts (one 16-row, 64-K MFMA tile), so that each step of the span
    is independently reachable: stubbing `pack_bf16` alone has to fail AT
    `pack_bf16`, not at a `pack_fp8` that was never given a packable tensor.
    `w_ug` is read for its dtype only, the rest for their `data_ptr`.
    """
    bf16 = torch.zeros(16, 64, dtype=torch.bfloat16)
    fp8 = torch.zeros(16, 64, dtype=torch.float8_e4m3fnuz)
    tensors = {"w_ug": bf16, "w_index_k": fp8, "w_index_q": fp8, "w_index_w": bf16}
    tensors.update({name: bf16 for name in INDEXER_WEIGHTS if name not in tensors})
    return LayerWeights(heads=16, t=tensors, config=glm5_tp_config(4), rank=0, npes=4)


def _build(**overrides):
    kwargs = {
        "W": _weights(),
        "samples": 4,
        "rank": 0,
        "npes": 4,
        "group": None,
        "topk": 2048,
        "with_indexer": True,
        "index_max_seq": 16384,
        "prepared_weights": {},
    }
    kwargs.update(overrides)
    return op.Glm5MonoKernel(**kwargs)


@pytest.mark.parametrize(
    "where",
    [
        # the whole span that used to run outside the attempt, first to last:
        # the dense requantization, then each indexer pack
        "prepare_glm5_weights",
        "pack_fp8",
        "pack_bf16",
    ],
)
def test_an_allocation_failure_reaches_the_agreement(events, monkeypatch, where):
    def boom(*args, **kwargs):
        raise _Boom("HIP out of memory")

    monkeypatch.setattr(op, where, boom)
    prepared = None if where == "prepare_glm5_weights" else {}
    with pytest.raises(MonoUnsupported) as raised:
        _build(prepared_weights=prepared)

    # The rank reported into the agreement instead of leaving for the runner's
    # own collective, and never went on to the peer buffer -- which is the
    # second collective its peers would otherwise still be waiting in.
    assert events == ["agree"]
    assert "rank 0" in str(raised.value) and "HIP out of memory" in str(raised.value)


def test_a_peer_failure_stops_this_rank_before_the_peer_buffer(events, monkeypatch):
    # The other direction: a peer reports a failure. This rank must raise on
    # that report alone -- whatever its OWN allocation did, which is why
    # nothing here stubs it -- so that nobody is left holding a peer buffer the
    # others never built.
    def _agree(reports, obj, group=None):
        events.append("agree")
        reports[:] = [obj, "OutOfMemoryError: HIP out of memory", None, None]

    monkeypatch.setattr(torch.distributed, "all_gather_object", _agree)
    with pytest.raises(MonoUnsupported) as raised:
        _build()
    assert events == ["agree"]
    assert "rank 1" in str(raised.value)


def test_a_shared_runtime_raises_plainly_because_nothing_collective_is_pending(
    events, monkeypatch
):
    # With a runtime handed in there is no peer-buffer constructor and no
    # agreement in this `__init__` at all, so no peer can be inside either:
    # every rank's next collective is the runner's own
    # `tp_uniform_local_validation`, which it reaches whether this raised or
    # returned. Reporting here instead would ADD a collective that the ranks
    # which did not fail never execute -- the very mismatch being fixed.
    def boom(*args, **kwargs):
        raise _Boom("HIP out of memory")

    monkeypatch.setattr(op, "pack_fp8", boom)
    with pytest.raises(_Boom):
        _build(runtime=SimpleNamespace(scr_layout=None, sym_layout=None))
    assert events == []

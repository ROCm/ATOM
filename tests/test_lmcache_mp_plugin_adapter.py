# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The two vLLM scheduler contracts the LMCache MP worker does not implement.

``LMCacheMPConnector`` is a bare ``KVConnectorBase``: unlike the in-process
workers it does not inherit ``OffloadWorkerMixin``, so it has neither
``take_load_error_blocks`` nor ``wait_for_requests``. Both are vLLM scheduler
contracts rather than transport properties, so the plugin supplies them in an
adapter instead of growing ATOM's shared MP worker.

Both defaults are silent and both corrupt:

* an empty ``take_load_error_blocks`` tells vLLM that a failed load's
  destination blocks hold valid KV, and vLLM serves the never-written bytes;
* a no-op ``wait_for_requests`` lets a preempting step hand blocks to another
  request while a transfer is still reading them.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload.metadata import LMCacheOffloadMetadata, LMCacheReqMeta
from atom.kv_transfer.offload.mp.transfer import _PendingSave
from atom.plugin.vllm.kv_transfer.mp_worker_adapter import MPOffloadWorkerAdapter


@dataclass
class _Future:
    """The ``query``/``result`` pair ``_terminal_future_result`` probes."""

    done: bool = False

    def query(self) -> bool:
        return self.done

    def result(self, timeout: float | None = None) -> Any:
        return None


class _FakeMPConnector:
    """Only what the adapter forwards to or reads off the real worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending_saves: dict[str, Any] = {}
        self._pending_loads: dict[str, Any] = {}
        self.output = KVConnectorOutput()
        self.started: list[Any] = []
        self.registered: list[tuple] = []
        self.transport_only_attribute = "forwarded"

    def register_kv_caches(self, kv_caches, transfer_tensors=None, num_blocks=None):
        self.registered.append((kv_caches, transfer_tensors, num_blocks))

    def start_load_kv(self, metadata) -> None:
        self.started.append(metadata)

    def get_finished(self) -> KVConnectorOutput:
        return self.output


def _metadata_with_load(req_id: str, block_ids: list[int]) -> LMCacheOffloadMetadata:
    metadata = LMCacheOffloadMetadata()
    metadata.add_request(
        LMCacheReqMeta(
            req_id=req_id,
            token_ids=[1, 2, 3],
            block_ids=block_ids,
            load_spec=object(),
        )
    )
    return metadata


def test_unknown_attributes_reach_the_real_worker():
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    assert adapter.transport_only_attribute == "forwarded"


def test_failed_load_surfaces_its_destination_blocks():
    """vLLM invalidates exactly the blocks this returns; missing one is served."""
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    adapter.start_load_kv(_metadata_with_load("r1", [7, 8, 9]))
    assert inner.started  # still dispatched
    inner.output.failed_loading = [LoadOperationId("r1", 0)]
    adapter.get_finished()
    assert adapter.take_load_error_blocks() == {7, 8, 9}
    # Taking is draining: a second step must not re-invalidate live blocks that
    # have since been handed to another request.
    assert adapter.take_load_error_blocks() == set()


def test_successful_load_reports_no_error_blocks():
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    adapter.start_load_kv(_metadata_with_load("r1", [7, 8, 9]))
    inner.output.finished_loading = [LoadOperationId("r1", 0)]
    adapter.get_finished()
    assert adapter.take_load_error_blocks() == set()


def test_bare_request_id_completions_still_match():
    """Legacy connectors report request ids where newer ones report generations."""
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    adapter.start_load_kv(_metadata_with_load("r1", [4]))
    inner.output.failed_loading = ["r1"]
    adapter.get_finished()
    assert adapter.take_load_error_blocks() == {4}


def test_fence_returns_once_the_transfer_is_terminal():
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    future = _Future(done=False)
    inner._pending_saves["save:r1:0"] = _PendingSave(
        completion=SaveOperationId("r1", 0), future=future, start=0, end=3
    )

    def finish() -> None:
        future.done = True

    threading.Timer(0.02, finish).start()
    adapter.wait_for_requests(["r1"])
    assert future.done


def test_fence_ignores_other_requests():
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    inner._pending_saves["save:other:0"] = _PendingSave(
        completion=SaveOperationId("other", 0),
        future=_Future(done=False),
        start=0,
        end=3,
    )
    adapter.wait_for_requests(["r1"])  # must not block on someone else's transfer


def test_fence_does_not_consume_the_pending_entry():
    """The completion it carries belongs to ``get_finished``.

    Draining here would swallow the report that releases the block lease, and
    the blocks would stay pinned for the life of the engine.
    """
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    inner._pending_saves["save:r1:0"] = _PendingSave(
        completion=SaveOperationId("r1", 0),
        future=_Future(done=True),
        start=0,
        end=3,
    )
    adapter.wait_for_requests(["r1"])
    assert "save:r1:0" in inner._pending_saves


def test_fence_treats_an_unsubmitted_transfer_as_not_reading_blocks():
    """``future=None`` is a submission that has not reached the server yet."""
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    inner._pending_loads["load:r1:0"] = _PendingSave(
        completion=LoadOperationId("r1", 0), future=None, start=0, end=3
    )
    adapter.wait_for_requests(["r1"])


def test_empty_fence_is_a_no_op():
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    adapter.wait_for_requests([])


def test_register_forwards_the_published_layout():
    inner = _FakeMPConnector()
    adapter = MPOffloadWorkerAdapter(inner)
    sentinel = object()
    adapter.register_kv_caches({"0": object()}, sentinel, 64)
    assert inner.registered[0][1] is sentinel and inner.registered[0][2] == 64


def test_fence_timeout_gives_up_loudly_rather_than_wedging_the_engine(
    monkeypatch, caplog
):
    """The one path that proceeds while a transfer may still be reading blocks.

    It is reached only when the server process is wedged, and what follows is a
    single wrong cached prefix with no other symptom -- so it must leave a
    record. Blocking forever instead would hang the engine inside
    ``execute_model``, which is the worse of the two.
    """
    import logging

    from atom.plugin.vllm.kv_transfer import mp_worker_adapter

    monkeypatch.setattr(mp_worker_adapter, "_FENCE_TIMEOUT_S", 0.01)
    inner = _FakeMPConnector()
    adapter = mp_worker_adapter.MPOffloadWorkerAdapter(inner)
    inner._pending_saves["save:r1:0"] = _PendingSave(
        completion=SaveOperationId("r1", 0),
        future=_Future(done=False),
        start=0,
        end=3,
    )
    with caplog.at_level(logging.ERROR, logger="atom"):
        adapter.wait_for_requests(["r1"])
    assert any("preemption fence timed out" in r.message for r in caplog.records)

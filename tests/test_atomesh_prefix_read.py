# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU checks for the experiment's observation and evidence gates."""

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest


def load_script(name):
    path = Path(__file__).parents[1] / ".github/scripts/atomesh" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trace_reads_only_complete_lines_and_requires_all_kinds(tmp_path):
    module = load_script("pd_prefix_workload")
    path = tmp_path / "rank.jsonl"
    records = [
        {"event": "plan", "request_id": "cmpl-prefix-probe"},
        {"event": "admission", "request_id": "cmpl-prefix-probe"},
    ]
    records += [
        {
            "event": "read_complete",
            "request_id": "cmpl-prefix-probe",
            "rank": rank,
            "kind": kind,
            "success": True,
        }
        for rank in range(8)
        for kind in ("attention", "kda")
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    trace = module.Trace(tmp_path)
    trace.poll()
    assert len(trace.records) == len(records) - 1
    with path.open("a") as stream:
        stream.write("\n")
    plan, admission, completed = asyncio.run(trace.request(["prefix-probe"]))
    assert len(completed) == 16
    assert plan["request_id"] == admission["request_id"]
    trace.poll()
    assert len(trace.records) == len(records)


@pytest.mark.parametrize("success", [False, True])
def test_observer_preserves_retry_and_completion_semantics(
    tmp_path, monkeypatch, success
):
    module = load_script("pd_prefix_observer")
    monkeypatch.setenv("MORI_PREFIX_TRACE_DIR", str(tmp_path))

    class Status:
        def __init__(self, failed):
            self.failed = failed

        def Failed(self):
            return self.failed

        def Message(self):
            return "SQ full" if self.failed else ""

    class Wrapper:
        def __init__(self):
            self.calls = 0

        def read_remote_data(self, sizes, local, remote, session):
            self.calls += 1
            return Status(self.calls == 1)

    class Scheduler:
        def update_state_after_alloc(self, *args):
            pass

    class Worker:
        tp_rank = 3

        def __init__(self):
            self.moriio_wrapper = Wrapper()
            self._recving_transfers_callback_addr = {"r": ("host", 1, "tx")}
            self._recving_transfers = {"r": {}}

        def _is_mamba_layer(self, layer):
            return layer == "kda"

        def _post_read_with_backoff(
            self, session, sizes, local, remote, request, layer, deadline
        ):
            while True:
                status = self.moriio_wrapper.read_remote_data(
                    sizes, local, remote, session
                )
                if not status.Failed():
                    return status

        def _pop_done_transfers(self):
            self._recving_transfers.clear()
            self._recving_transfers_callback_addr.clear()
            return {"tx"} if success else set()

        def _await_reads_issued_this_step(self):
            return "waited"

        def wait_for_layer_load(self, layer):
            return "waited"

    module.install(
        {
            "MoRIIOConnectorScheduler": Scheduler,
            "MoRIIOConnectorWorker": Worker,
            "MoRIIOWrapper": Wrapper,
        }
    )
    worker = Worker()
    status = worker._post_read_with_backoff(
        None, [10, 20], [1, 2], [3, 4], "r", "attention", 1
    )
    assert not status.Failed()
    assert worker.moriio_wrapper.calls == 2
    assert worker.wait_for_layer_load("attention") == "waited"
    assert worker._pop_done_transfers() == ({"tx"} if success else set())
    records = [
        json.loads(line)
        for path in tmp_path.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    completed = [r for r in records if r["event"] == "read_complete"]
    assert len(completed) == 1
    assert completed[0]["payload_bytes"] == 30
    assert completed[0]["attempts"] == 2
    assert completed[0]["success"] is success
    assert len([r for r in records if r["event"] == "read_rejected"]) == 1
    assert not worker._prefix_observed_reads

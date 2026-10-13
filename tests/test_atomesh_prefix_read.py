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


@pytest.mark.parametrize("page_tokens", [128, 1024, 12288])
def test_workload_geometry_uses_complete_pages(page_tokens):
    module = load_script("pd_prefix_workload")
    geometry = module.workload_geometry(page_tokens)
    assert geometry["mechanism_end"] >= 8192
    assert geometry["mechanism_hit"] >= 6 * page_tokens
    assert geometry["mechanism_hit"] % page_tokens == 0
    assert geometry["performance_length"] >= 32769
    assert geometry["performance_hit"] % page_tokens == 0
    assert (
        0.8
        <= geometry["performance_hit"] / (geometry["performance_length"] - 1)
        <= 0.95
    )
    if page_tokens == 12288:
        assert geometry["mechanism_end"] == 98304
        assert geometry["mechanism_hit"] == 73728
        assert geometry["performance_length"] == 122881
        assert geometry["performance_hit"] == 110592


def test_trace_reads_only_complete_lines_and_requires_all_kinds(tmp_path):
    module = load_script("pd_prefix_workload")
    path = tmp_path / "rank.jsonl"
    records = [
        {
            "event": "plan",
            "request_id": "cmpl-prefix-probe",
            "external_tokens": 128,
        },
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


def test_trace_accepts_full_attention_hit_only_after_all_kda_reads(
    tmp_path, monkeypatch
):
    """A full attention hit still transfers KDA, with no attention READs."""
    module = load_script("pd_prefix_workload")
    request_id = "cmpl-prefix-repeat"
    plan = {
        "event": "plan",
        "request_id": request_id,
        "local": [[], [200]],
        "external_tokens": 0,
    }
    admission = {
        "event": "admission",
        "request_id": request_id,
        "external_tokens": 0,
    }
    records = [plan, admission] + [
        {
            "event": "read_complete",
            "request_id": request_id,
            "rank": rank,
            "kind": "kda",
            "success": True,
        }
        for rank in range(8)
    ]
    path = tmp_path / "rank.jsonl"

    async def no_sleep(_):
        pass

    monkeypatch.setattr(module.asyncio, "sleep", no_sleep)
    path.write_text("".join(json.dumps(row) + "\n" for row in records[:-1]))
    trace = module.Trace(tmp_path)
    with pytest.raises(AssertionError, match="eight-rank READ completion"):
        asyncio.run(trace.request(["prefix-repeat"]))
    with path.open("a") as stream:
        stream.write(json.dumps(records[-1]) + "\n")
    actual_plan, actual_admission, completed = asyncio.run(
        trace.request(["prefix-repeat"])
    )
    assert actual_plan == plan
    assert actual_admission == admission
    assert len(completed) == 8
    assert {row["kind"] for row in completed} == {"kda"}


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

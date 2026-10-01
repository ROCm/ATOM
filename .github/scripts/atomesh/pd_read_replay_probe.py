"""Opt-in, bounded MRV2 READ/replay evidence; never instruments model functions."""

import argparse
import json
import os
import threading
import time
from functools import wraps
from pathlib import Path


class Recorder:
    def __init__(self, root, role, rank, limit=128):
        self.root, self.role, self.rank, self.limit = root, role, rank, limit
        self.step = 0
        self.active = False
        self.wait_context = False
        self.requests = []

    def begin(self, requests, sync_load):
        self.step += 1
        self.active = self.step <= self.limit
        self.requests = list(requests)
        self.emit("step", sync_load=sync_load)

    def emit(self, event, **fields):
        if not self.active:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f"{self.role}-{self.rank}-{os.getpid()}.jsonl").open(
            "a"
        ) as f:
            f.write(
                json.dumps(
                    {
                        "role": self.role,
                        "rank": self.rank,
                        "pid": os.getpid(),
                        "thread": threading.get_ident(),
                        "step": self.step,
                        "event": event,
                        "time_ns": time.monotonic_ns(),
                        "request_ids": self.requests,
                        **fields,
                    }
                )
                + "\n"
            )


def qualifying_ranks(events):
    states, passed = {}, set()
    for event in events:
        key = tuple(event[k] for k in ("role", "rank", "pid", "thread", "step"))
        if event["event"] == "step":
            states[key] = {
                "sync": event.get("sync_load"),
                "wait": False,
                "failed": False,
                "replay": False,
            }
            continue
        state = states.get(key)
        if state is None:
            continue
        if event["event"] == "error":
            state["failed"] = True
        elif event["event"] == "wait_done":
            state["wait"] = event.get("count", 0) > 0 and event.get("mode") == "FULL"
        elif (
            event["event"] == "replay_done"
            and state["sync"]
            and state["wait"]
            and not state["failed"]
            and event.get("preexisting_graph")
            and event.get("mode") == "FULL"
        ):
            state["replay"] = True
    for key, state in states.items():
        if state["replay"] and not state["failed"]:
            passed.add(key[:2])
    return passed


_restore = None


def install():
    global _restore
    import torch.distributed as dist
    from vllm.config import CUDAGraphMode
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
        MoRIIOConnectorWorker,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_engine import (
        MoRIIOWrapper,
    )
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
    from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector

    root = Path(os.environ["ATOMESH_READ_REPLAY_ROOT"])
    enabled = root / "enabled"
    local = threading.local()

    def recorder():
        if not hasattr(local, "recorder"):
            local.recorder = Recorder(
                root, os.environ["ATOMESH_RUNTIME_ROLE"], dist.get_rank()
            )
        return local.recorder

    original_pre = ActiveKVConnector.pre_forward

    @wraps(original_pre)
    def pre(self, scheduler_output, **kwargs):
        rec = recorder()
        if enabled.exists():
            rec.begin(
                scheduler_output.num_scheduled_tokens,
                scheduler_output.has_sync_kv_loads,
            )
        else:
            rec.active = False
        return original_pre(self, scheduler_output, **kwargs)

    original_await = MoRIIOConnectorWorker._await_reads_issued_this_step

    @wraps(original_await)
    def await_reads(self):
        rec = recorder()
        rec.wait_context = rec.active
        try:
            return original_await(self)
        finally:
            rec.wait_context = False

    original_wait = MoRIIOWrapper.waiting_for_transfer_complete

    @wraps(original_wait)
    def wait(self, transfer_statuses=None):
        rec = recorder()
        if not rec.wait_context:
            return original_wait(self, transfer_statuses)
        count = len(transfer_statuses) if transfer_statuses is not None else 0
        mode = get_forward_context().cudagraph_runtime_mode.name
        try:
            result = original_wait(self, transfer_statuses)
        except Exception:
            rec.emit("error", operation="read_wait", count=count, mode=mode)
            raise
        rec.emit("wait_done", count=count, mode=mode)
        return result

    original_replay = CudaGraphManager.run_fullgraph

    @wraps(original_replay)
    def replay(self, desc):
        rec = recorder()
        if not rec.active:
            return original_replay(self, desc)
        existing = desc in self.graphs and self.graphs[desc] is not None
        mode = desc.cg_mode.name
        assert desc.cg_mode == CUDAGraphMode.FULL and existing
        try:
            result = original_replay(self, desc)
        except Exception:
            rec.emit("error", operation="replay", mode=mode)
            raise
        rec.emit("replay_done", mode=mode, preexisting_graph=existing)
        return result

    ActiveKVConnector.pre_forward = pre
    MoRIIOConnectorWorker._await_reads_issued_this_step = await_reads
    MoRIIOWrapper.waiting_for_transfer_complete = wait
    CudaGraphManager.run_fullgraph = replay

    def restore():
        ActiveKVConnector.pre_forward = original_pre
        MoRIIOConnectorWorker._await_reads_issued_this_step = original_await
        MoRIIOWrapper.waiting_for_transfer_complete = original_wait
        CudaGraphManager.run_fullgraph = original_replay

    _restore = restore


class ReadReplayWorkerExtension:
    """Loading this extension installs hooks in each spawned native worker."""

    def pr58968_disable_probe(self):
        if _restore is not None:
            _restore()
        return {"probe_disabled": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    events = [
        json.loads(line)
        for path in args.root.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    passed = qualifying_ranks(events)
    expected = {("decode", rank) for rank in range(8)}
    result = {
        "full_sync_read_replay_ranks": sorted(passed),
        "required_decode_ranks": sorted(expected),
        "passed": expected <= passed,
        "physical_block_reuse_proven": False,
    }
    (args.root / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise SystemExit(
            "Missing same-step synchronous nonempty READ wait -> FULL replay evidence"
        )
elif os.environ.get("ATOMESH_READ_REPLAY_ROOT"):
    install()

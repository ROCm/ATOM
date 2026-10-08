# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Symmetric, experiment-only MoRI observations; no GPU synchronization."""

import functools
import itertools
import json
import os
import socket
import threading
import time
from pathlib import Path

_fd = None
_sequence = itertools.count()
_context = threading.local()


def emit(event, **fields):
    global _fd
    directory = os.environ.get("MORI_PREFIX_TRACE_DIR")
    if not directory:
        return
    if _fd is None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        path = Path(directory) / f"{socket.gethostname()}-{os.getpid()}.jsonl"
        _fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    record = dict(event=event, time=time.time(), pid=os.getpid(), **fields)
    os.write(_fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())


def install(namespace):
    scheduler = namespace["MoRIIOConnectorScheduler"]
    worker = namespace["MoRIIOConnectorWorker"]
    wrapper = namespace["MoRIIOWrapper"]
    update = scheduler.update_state_after_alloc

    @functools.wraps(update)
    def observed_update(self, request, blocks, external, *args, **kwargs):
        full = blocks.get_block_ids()
        result = update(self, request, blocks, external, *args, **kwargs)
        pending = self._reqs_need_recv.get(request.request_id)
        if pending is not None:
            params = self._req_kv_params.get(request.request_id, {})
            spec = self.kv_cache_config.transfer_groups[
                self._attn_group_ids[0]
            ].kv_cache_spec
            page_tokens = spec.block_size
            if spec.dcp_sharded:
                page_tokens *= (
                    self.vllm_config.parallel_config.decode_context_parallel_size
                )
            emit(
                "plan",
                request_id=request.request_id,
                transfer_id=params.get("transfer_id"),
                external_tokens=external,
                prompt_tokens=request.num_prompt_tokens,
                page_tokens=page_tokens,
                full_block_ids=full,
                local=pending[1],
                attention_full=self.split_block_groups(full)[0],
                remote=params.get("remote_block_ids"),
            )
        return result

    scheduler.update_state_after_alloc = observed_update
    post = worker._post_read_with_backoff
    read = wrapper.read_remote_data

    @functools.wraps(read)
    def observed_read(self, sizes, local=0, remote=0, session=None):
        status = read(self, sizes, local, remote, session)
        context = getattr(_context, "read", None)
        if context is not None:
            context["attempts"] += 1
            if status.Failed():
                emit("read_rejected", **context, message=status.Message())
        return status

    wrapper.read_remote_data = observed_read

    @functools.wraps(post)
    def observed_post(self, session, sizes, local, remote, request_id, layer, deadline):
        record = {
            "request_id": request_id,
            "rank": self.tp_rank,
            "layer": layer,
            "kind": "kda" if self._is_mamba_layer(layer) else "attention",
            "operation": next(_sequence),
            "payload_bytes": sum(sizes),
            "attempts": 0,
        }
        _context.read = record
        start = time.perf_counter()
        try:
            status = post(
                self, session, sizes, local, remote, request_id, layer, deadline
            )
        finally:
            _context.read = None
        record["submit_seconds"] = time.perf_counter() - start
        if "mechanism" in request_id:
            emit(
                "read_submit",
                **record,
                failed=bool(status.Failed()),
                local_offsets=local,
                remote_offsets=remote,
                sizes=sizes,
            )
        if not hasattr(self, "_prefix_observed_reads"):
            self._prefix_observed_reads = {}
        self._prefix_observed_reads.setdefault(request_id, []).append(record)
        return status

    worker._post_read_with_backoff = observed_post
    pop = worker._pop_done_transfers

    @functools.wraps(pop)
    def observed_pop(self):
        before = {
            req: value[2]
            for req, value in self._recving_transfers_callback_addr.items()
        }
        done = pop(self)
        records = getattr(self, "_prefix_observed_reads", {})
        for request_id, transfer_id in before.items():
            if request_id not in self._recving_transfers:
                completed = records.pop(request_id, [])
                for kind in ("attention", "kda"):
                    selected = [r for r in completed if r["kind"] == kind]
                    if selected:
                        emit(
                            "read_complete",
                            request_id=request_id,
                            rank=self.tp_rank,
                            kind=kind,
                            transfer_id=transfer_id,
                            operations=[r["operation"] for r in selected],
                            payload_bytes=sum(r["payload_bytes"] for r in selected),
                            attempts=sum(r["attempts"] for r in selected),
                            success=transfer_id in done,
                        )
        waits = getattr(self, "_prefix_observed_waits", [])
        if waits:
            emit("wait_window", rank=self.tp_rank, waits=waits)
            self._prefix_observed_waits = []
        return done

    worker._pop_done_transfers = observed_pop
    for method, kind in (
        ("_await_reads_issued_this_step", "kda"),
        ("wait_for_layer_load", "attention"),
    ):
        original = getattr(worker, method)

        def make_wait(original, kind):
            @functools.wraps(original)
            def observed_wait(self, *args, **kwargs):
                requests = list(self._recving_transfers)
                if not requests:
                    return original(self, *args, **kwargs)
                start = time.perf_counter()
                try:
                    return original(self, *args, **kwargs)
                finally:
                    if not hasattr(self, "_prefix_observed_waits"):
                        self._prefix_observed_waits = []
                    self._prefix_observed_waits.append(
                        {
                            "kind": kind,
                            "request_ids": requests,
                            "layer": args[0] if args else None,
                            "seconds": time.perf_counter() - start,
                        }
                    )

            return observed_wait

        setattr(worker, method, make_wait(original, kind))


def patch_source(root):
    """Install identical observation code on the frozen A and B sources."""
    import ast
    import hashlib
    import shutil

    root = Path(root)
    relative = "vllm/distributed/kv_transfer/kv_connector/v1/moriio"
    module = "vllm.distributed.kv_transfer.kv_connector.v1.moriio.pd_prefix_observer"
    shutil.copyfile(__file__, root / relative / "pd_prefix_observer.py")
    connector = root / relative / "moriio_connector.py"
    text = connector.read_text()
    assert "pd_prefix_observer" not in text
    text += f"\nfrom {module} import install as _install_prefix_observer\n_install_prefix_observer(globals())\n"
    ast.parse(text)
    connector.write_text(text)
    scheduler = root / "vllm/v1/core/sched/scheduler.py"
    text = scheduler.read_text()
    anchor = "                    self.connector.update_state_after_alloc(\n"
    assert text.count(anchor) == 1
    probe = (
        f"                    from {module} import emit as _prefix_emit\n"
        "                    _prefix_emit('admission', request_id=request.request_id,\n"
        "                        local_tokens=num_new_local_computed_tokens,\n"
        "                        external_tokens=num_external_computed_tokens,\n"
        "                        prompt_tokens=request.num_prompt_tokens,\n"
        "                        preemptions=request.num_preemptions)\n"
    )
    text = text.replace(anchor, probe + anchor)
    ast.parse(text)
    scheduler.write_text(text)
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (connector, scheduler, root / relative / "pd_prefix_observer.py")
    }


if __name__ == "__main__":
    import sys

    print(json.dumps(patch_source(sys.argv[1]), indent=2))

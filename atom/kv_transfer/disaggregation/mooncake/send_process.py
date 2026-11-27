# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""P->D send process (``ATOM_PD_SEND_PROCESS``): a Mooncake producer stage's send path,
unchanged, in a process of its own over one HIP IPC mapping of the stage's paged pool.
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from atom.kv_transfer.disaggregation.pd_producer import (
    STAGING_SLOT_BYTES,
    mooncake_producer_connectors,
)
from atom.kv_transfer.disaggregation.staging_config import minimax_page_staging
from atom.utils import envs, mooncake_hca

logger = logging.getLogger("atom")

READY, SAVED, STOP = "ready", "saved", "stop"
BOUND, DONE, FATAL = "bound", "done", "fatal"
# Spawn to BOUND: the child's imports (25 s cold from NFS on a CPU host), engine,
# staging, warm-up and bind. A child that dies is seen at once (EOF).
BOUND_TIMEOUT_S = 300.0
# The child's HIP context, allocator and kernels: 298-386 MiB measured on MI355X;
# every GPU run used this 1 GiB. Its staging is in the staging reserve.
SEND_PROCESS_RESERVE_BYTES = 1 << 30
# What the stage's config and registration set for the send path; the rest is
# `_init_transfer_state`'s, as in the stage.
STATE_KEYS = (
    "tp_rank", "tp_size", "dp_rank", "pp_rank", "pp_size", "num_hidden_layers",
    "block_size", "dcp_size", "_start_layer", "_num_local_layers", "local_ip",
    "protocol", "ib_devices", "_side_channel_port", "num_blocks", "block_len",
    "_per_block_bytes_list", "_block_region_roles", "_block_head_major_runs",
    "_block_unit_specs", "_block_region_consumer_indices", "_fp4_index_layout",
    "_kv_staging_reserve", "_num_send_workers", "_cuda_device",
)  # fmt: skip


def send_process_enabled(config) -> bool:
    """Whether the Mooncake producer of ``config`` sends from a send process; the
    flag on a producer that cannot is refused, not ignored."""
    if not envs.ATOM_PD_SEND_PROCESS:
        return False
    producers = mooncake_producer_connectors(config)
    if not producers:
        return False
    if not minimax_page_staging(config) or any(
        c.get("enable_chunked_transfer", False) for c in producers
    ):
        raise RuntimeError(
            "ATOM_PD_SEND_PROCESS serves a MiniMax page-staging Mooncake producer "
            "(ATOM_PD_MINIMAX_LANDING, TP1) without chunked transfer only"
        )
    return True


@dataclass
class SendProcessSpec:
    state: dict[str, Any]
    pool_handle: bytes
    # (region, byte offset from the pool allocation's base, view shape)
    views: list[tuple[int, int, tuple[int, ...]]]
    label: str


def export_views(views: dict[int, torch.Tensor]) -> tuple[bytes, list]:
    """The IPC handle of the one allocation every view lies in, and each view's
    place in it."""
    from atom.mono.runtime.hip_ipc import get_allocation_base, get_ipc_handle

    ends = [(v.data_ptr(), v.data_ptr() + v.numel() - 1) for v in views.values()]
    bases = {get_allocation_base(p) for end in ends for p in end}
    if len(bases) != 1:
        raise RuntimeError(
            "ATOM_PD_SEND_PROCESS needs every staged KV view inside one "
            f"allocation; they touch {len(bases)}"
        )
    (base,) = bases
    placed = [
        (i, v.data_ptr() - base, tuple(v.shape)) for i, v in sorted(views.items())
    ]
    return get_ipc_handle(base), placed


def start(stage, views: dict[int, torch.Tensor]) -> StageLink:
    """Spawn ``stage``'s send process over ``views`` and wait until it serves."""
    handle, placed = export_views(views)
    state = {key: getattr(stage, key) for key in STATE_KEYS}
    return StageLink(
        SendProcessSpec(state, handle, placed, f"pd-send-pp{stage.pp_rank}")
    )


class StageLink:
    """The stage's end: the child, the relay thread (the only writer to the
    child) and the DONE / FATAL reader (the stage's main thread only)."""

    def __init__(self, spec: SendProcessSpec) -> None:
        from atom.utils import get_mp_context

        ctx = get_mp_context()
        child_in, self._to_child = ctx.Pipe(duplex=False)
        self._from_child, child_out = ctx.Pipe(duplex=False)
        self.label = spec.label
        # Started from the runner's main thread, which PDEATHSIG follows.
        self.proc = ctx.Process(
            target=child_main,
            args=(spec, child_in, child_out),
            name=spec.label,
            daemon=True,
        )
        started = time.monotonic()
        self.proc.start()
        child_in.close()
        child_out.close()
        try:
            self._await_bound(started)
        except BaseException:
            self.proc.kill()
            self.proc.join(5)
            raise
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._relay = threading.Thread(
            target=self._relay_loop, name="pd-send-relay", daemon=True
        )
        self._relay.start()

    def _await_bound(self, started: float) -> None:
        if not self._from_child.poll(BOUND_TIMEOUT_S):
            raise RuntimeError(f"{self.label} did not bind within {BOUND_TIMEOUT_S}s")
        try:
            kind, payload = self._from_child.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"{self.label} exited ({self.proc.exitcode}) before it bound"
            ) from exc
        if kind != BOUND:
            raise RuntimeError(f"{self.label} failed to start: {payload}")
        logger.info(
            "%s serves this stage's P->D sends (engine %s), bound %.1fs after spawn",
            self.label,
            payload,
            time.monotonic() - started,
        )

    def saved(self, entries: dict) -> None:
        self._queue.put((SAVED, entries))

    def ready(self, event, req_ids: list) -> None:
        self._queue.put((READY, (event, list(req_ids))))

    def _relay_loop(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get()
                if kind == READY:
                    event, payload = payload
                    # The forward that wrote these pages has completed.
                    event.synchronize()
                self._to_child.send((kind, payload))
                if kind == STOP:
                    return
        except BaseException as exc:  # noqa: BLE001 - raised by poll_done
            self._error = exc

    def poll_done(self) -> set:
        """Transfers released since the last call; raises once the child or the
        relay is gone. The stage's main thread, once per step."""
        if self._error is not None:
            raise RuntimeError(f"{self.label} relay failed") from self._error
        done: set = set()
        try:
            while self._from_child.poll(0):
                kind, payload = self._from_child.recv()
                if kind != DONE:
                    raise RuntimeError(f"{self.label} failed: {payload}")
                done.update(payload)
        except EOFError as exc:
            raise RuntimeError(f"{self.label} exited ({self.proc.exitcode})") from exc
        if not self.proc.is_alive():
            raise RuntimeError(f"{self.label} exited ({self.proc.exitcode})")
        return done

    def close(self) -> None:
        """STOP behind the queued hand-offs, then the child is gone."""
        self._queue.put((STOP, None))
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5)


class ReadyTable:
    """Transfers whose KV the stage reported complete: the newest ``keep``, as
    the stage remembers its expired ready events."""

    def __init__(self, keep: int) -> None:
        self._cv = threading.Condition()
        self._ready: OrderedDict = OrderedDict()
        self._keep = keep

    def mark(self, transfer_ids) -> None:
        with self._cv:
            for tid in transfer_ids:
                self._ready[tid] = None
                self._ready.move_to_end(tid)
            while len(self._ready) > self._keep:
                self._ready.popitem(last=False)
            self._cv.notify_all()

    def wait(self, transfer_id, timeout_s: float) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: transfer_id in self._ready, timeout_s)

    def drop(self, transfer_id) -> None:
        with self._cv:
            self._ready.pop(transfer_id, None)


def _connector_class():
    """`SendProcessConnector`, built on first use: the connector needs AITER."""
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    class SendProcessConnector(mc.MooncakeConnector):
        """The producer half of `MooncakeConnector`, unchanged, in the send
        process: its own engine and staging, the stage's KV through views."""

        @classmethod
        def create(cls, state: dict, views: dict, done) -> SendProcessConnector:
            self = object.__new__(cls)  # no TP / DP group in this process
            self.is_producer, self.is_consumer = True, False
            self._init_transfer_state()
            self.__dict__.update(state)
            self._done, self._transfer = done, threading.local()
            self._ready = ReadyTable(mc.KV_CACHE_READY_EXPIRED_MAX)
            if sorted(views) != list(range(len(self._per_block_bytes_list))):
                raise RuntimeError("the send process needs one view per block region")
            self._kv_source_views = views
            self.kv_caches_base_addr = [views[i].data_ptr() for i in sorted(views)]
            self._block_regions = list(
                zip(self.kv_caches_base_addr, self._per_block_bytes_list)
            )
            return self

        def serve(self) -> threading.Thread:
            """Engine, staging (its only registered memory), warm gathers, listener."""
            mooncake_hca.configure_transport(self.protocol)
            self._init_transfer_engine(",".join(self.ib_devices))
            device = next(iter(self._kv_source_views.values())).device
            staging = self._alloc_kv_staging(STAGING_SLOT_BYTES, device)
            sizes = self._rdma_chunk_sizes(staging.total_bytes, staging.unit_bytes)
            ptrs = (staging.base_addr + np.cumsum([0, *sizes[:-1]])).tolist()
            if self.transfer_engine.batch_register_memory(ptrs, sizes) != 0:
                raise RuntimeError("the send process could not register its staging")
            self._local_metadata = self._make_local_metadata(num_slots=0)
            self._warm_gathers()
            # Recorded behind the warm-up gathers, so they ran before this serves.
            self._ready_event = _completed_event(device)
            self._send_executor = self._make_send_executor()
            listener = threading.Thread(
                target=self._write_listener, daemon=True, name="mooncake-write-listener"
            )
            listener.start()
            self._await_listener()
            return listener

        def _warm_gathers(self) -> None:
            """Compile the head-cut gathers a TP2/4/8 decode's transfers launch,
            index and output aligned or not, before any transfer arrives."""
            staging = self._kv_staging[0]
            index = torch.zeros(2, dtype=torch.int64, device=staging.device)
            for region, view in self._kv_source_views.items():
                runs = self._block_head_major_runs[region]
                if runs is None:
                    continue
                heads, widths = runs
                src_bytes = self._per_block_bytes_list[region]
                for tp in (t for t in (2, 4, 8) if heads % t == 0 or t % heads == 0):
                    count = max(1, heads // tp)
                    dst_bytes = count * sum(widths)
                    rows = mc.consumer_head_slices(
                        src_bytes, dst_bytes, runs, (count, widths), 0, tp, "warm-up"
                    )
                    plan = mc.GatherLayout.from_slices(src_bytes, dst_bytes, rows)
                    for start, offset in ((0, 0), (1, 0), (0, 4), (1, 4)):
                        mc.gather_units(
                            view,
                            index[start : start + 1],
                            staging[offset : offset + dst_bytes],
                            plan,
                        )

        def cache_saved_prefills(self, entries: dict) -> None:
            with self._completed_prefills_cv:
                self._completed_prefills.update(entries)
                self._completed_prefills_cv.notify_all()

        def _execute_transfer(self, request_data: dict) -> None:
            transfer = self._transfer
            transfer.deadline = request_data.get("_deadline")
            try:
                super()._execute_transfer(request_data)
            finally:
                vars(transfer).clear()

        def _get_kv_cache_ready_event(self, req_id):
            timeout_s = mc.PREFILL_LOOKUP_TIMEOUT
            deadline = self._transfer.deadline
            if deadline is not None:
                timeout_s = min(timeout_s, deadline - time.monotonic())
            if not self._ready.wait(req_id, timeout_s):
                # Raised into `_execute_transfer`, which fails the transfer.
                raise RuntimeError(f"the stage never reported {req_id}'s KV complete")
            return self._ready_event

        def _rdma_write(
            self, target, src_addrs, dst_addrs, sizes, req_id, label, **kwargs
        ):
            low = self._kv_staging.data_ptr()
            high = low + self._kv_staging.numel()
            if src_addrs and (
                min(src_addrs) < low
                or max(s + n for s, n in zip(src_addrs, sizes)) > high
            ):
                # The KV is registered with no NIC here: only staged slots go out.
                raise RuntimeError(f"{label} write for {req_id} is not from staging")
            return super()._rdma_write(
                target, src_addrs, dst_addrs, sizes, req_id, label, **kwargs
            )

        def _drop_completed_prefill(self, transfer_id) -> None:
            super()._drop_completed_prefill(transfer_id)
            self._ready.drop(transfer_id)

        def _mark_done_sending(self, transfer_id) -> None:
            self._done(transfer_id)

    return SendProcessConnector


def _completed_event(device: torch.device):
    """A long-complete event: what the send path waits on once READY came."""
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    event.synchronize()
    return event


def _open_views(spec: SendProcessSpec) -> dict[int, torch.Tensor]:
    """The stage's staged KV views, over one IPC mapping of its paged pool."""
    from atom.mono.runtime.hip_ipc import open_ipc_handle

    base = open_ipc_handle(spec.pool_handle)
    device = torch.device("cuda", spec.state["_cuda_device"])
    return {
        region: _device_view(base + offset, shape, device)
        for region, offset, shape in spec.views
    }


def _device_view(address: int, shape, device: torch.device) -> torch.Tensor:
    from atom.mono.runtime.uncached import DeviceBytes

    nbytes = math.prod(shape)
    return torch.as_tensor(DeviceBytes(address, nbytes), device=device).view(shape)


class _LogPrefix(logging.Filter):
    def __init__(self, prefix: str) -> None:
        super().__init__()
        self._prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"{self._prefix} {record.msg}"
        return True


def child_main(spec: SendProcessSpec, from_stage, to_stage) -> None:
    """The send process: serve the stage's transfers until STOP, EOF or a fault."""
    from atom.utils import enable_orphan_reaping, set_process_title

    enable_orphan_reaping()  # first: a dead stage takes this process with it
    set_process_title(spec.label)
    logger.addFilter(_LogPrefix(f"[pd-send pp{spec.state['pp_rank']}]"))
    lock = threading.Lock()

    def report(kind: str, payload) -> None:
        with lock:
            to_stage.send((kind, payload))

    started = time.monotonic()
    try:
        torch.cuda.set_device(spec.state["_cuda_device"])
        views = _open_views(spec)
        connector = _connector_class().create(
            spec.state, views, lambda tid: report(DONE, [tid])
        )
        listener = connector.serve()
    except BaseException as exc:
        logger.exception("failed to start")
        try:
            report(FATAL, repr(exc))
        finally:
            os._exit(1)
    logger.info("set up in %.1fs after its imports", time.monotonic() - started)
    report(BOUND, connector.engine_id)

    def watch_listener() -> None:
        listener.join()
        report(FATAL, f"write listener exited: {connector._listener_error!r}")

    threading.Thread(target=watch_listener, name="pd-send-watch", daemon=True).start()
    _follow_stage(connector, from_stage)
    os._exit(0)


def _follow_stage(connector, from_stage) -> None:
    """Apply the stage's READY and SAVED until STOP or the stage is gone."""
    while True:
        try:
            kind, payload = from_stage.recv()
        except EOFError:
            return
        if kind == READY:
            connector._ready.mark(payload)
        elif kind == SAVED:
            connector.cache_saved_prefills(payload)
        else:
            return

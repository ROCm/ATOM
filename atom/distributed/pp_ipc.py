# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""PP stage boundary over HIP IPC (``ATOM_PP_IPC``): the sender copies its outputs into
the receiver's ring on a copy stream, ordered by interprocess events (`PPIpcSender.send`).
"""

from __future__ import annotations

import ctypes
import logging
import pickle

import torch

from atom.models.utils import IntermediateTensors
from atom.utils import envs

logger = logging.getLogger("atom")

_KEYS = ("hidden_states", "residual")
_HIP_MEMCPY_DEVICE_TO_DEVICE = 3
_HIP_DEVICE_MALLOC_DEFAULT = 0x0
SLOTS = 3
# Records per interprocess event before it is replaced (fails at 33).
EVENT_USES = 16
# Replaced events awaiting their fence; past this the oldest fence is waited.
_RETIRED_MAX = 64
# Unwaited gloo isends; a peer posts the matching receive only on its next
# transfer, so they are never drained on idle.
_SENDS_MAX = 64
_sends: list = []


def pp_ipc_enabled(config) -> bool:
    """ATOM_PP_IPC: a MiniMax-M3 P/D prefill producer under PP with one rank a
    stage on gfx950 takes the IPC boundary; every other server keeps NCCL."""
    if not envs.ATOM_PP_IPC or config.pipeline_parallel_size <= 1:
        return False
    sizes = (
        config.tensor_parallel_size,
        config.prefill_context_parallel_size,
        config.decode_context_parallel_size,
        config.sequence_parallel_size,
        config.parallel_config.data_parallel_size,
    )
    if any(size != 1 for size in sizes):
        return False
    from atom.config import _is_minimax_m3_config
    from atom.kv_transfer.disaggregation.pd_producer import pd_producer_configured

    if not _is_minimax_m3_config(config.hf_config):
        return False
    if not pd_producer_configured(config):
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


# ---- gloo messages ------------------------------------------------------------


def _send_obj(obj, dst: int, group, blocking: bool = True) -> None:
    raw = torch.frombuffer(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
    size = torch.tensor([raw.numel()], device="cpu")
    if blocking:
        torch.distributed.send(size, dst=dst, group=group)
        torch.distributed.send(raw, dst=dst, group=group)
        return
    # Kept alive until done, never waited for here: waiting would chain this
    # stage's CPU to the peer's.
    for t in (size, raw):
        _sends.append((torch.distributed.isend(t, dst=dst, group=group), t))
    while _sends and _sends[0][0].is_completed():
        _sends.pop(0)[0].wait()
    while len(_sends) > _SENDS_MAX:
        _sends.pop(0)[0].wait()


def _recv_obj(src: int, group):
    size = torch.empty(1, dtype=torch.long, device="cpu")
    torch.distributed.recv(size, src=src, group=group)
    raw = torch.empty(int(size.item()), dtype=torch.uint8, device="cpu")
    torch.distributed.recv(raw, src=src, group=group)
    return pickle.loads(raw.numpy().tobytes())


def _hip():
    from atom.mono.runtime.hip_ipc import load_runtime

    lib = load_runtime()
    lib.hipMemcpyAsync.restype = ctypes.c_int
    lib.hipMemcpyAsync.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    return lib


# ---- interprocess events ------------------------------------------------------


class _Retired:
    """Replaced events, each destroyed once its local fence completed: the
    fence follows every wait on the event that can still be queued."""

    def __init__(self):
        self.items: list = []

    def add(self, event, stream) -> None:
        fence = torch.cuda.Event()
        fence.record(stream)
        self.items.append((event, fence))
        self.reap()

    def reap(self) -> None:
        self.items = [(e, f) for e, f in self.items if not f.query()]
        while len(self.items) > _RETIRED_MAX:
            self.items.pop(0)[1].synchronize()

    def __len__(self) -> int:
        return len(self.items)


class _Rotating:
    """One interprocess event per slot this side records, replaced every
    ``EVENT_USES`` records; the new handle travels with the message it orders."""

    def __init__(self, retired: _Retired):
        self.retired = retired
        self.events = [None] * SLOTS
        self.uses = [0] * SLOTS
        self.created = 0

    def record(self, k: int, stream) -> bytes | None:
        """Record slot k's event on ``stream``; the new event's IPC handle if
        it was replaced (the peer's waits on the old one are queued by then)."""
        handle = None
        if self.events[k] is None or self.uses[k] >= EVENT_USES:
            if self.events[k] is not None:
                self.retired.add(self.events[k], stream)
            self.events[k] = torch.cuda.Event(interprocess=True, enable_timing=False)
            self.uses[k] = 0
            self.created += 1
            handle = bytes(self.events[k].ipc_handle())
        self.events[k].record(stream)
        self.uses[k] += 1
        return handle


class _Opened:
    """The peer's interprocess events this side waits on, one per slot."""

    def __init__(self, device, retired: _Retired):
        self.device = device
        self.retired = retired
        self.events = [None] * SLOTS

    def wait(self, k: int, handle: bytes | None, stream) -> None:
        if handle is not None:
            if self.events[k] is not None:
                # This stream's waits on it are all queued already.
                self.retired.add(self.events[k], stream)
            self.events[k] = torch.cuda.Event.from_ipc_handle(self.device, handle)
        stream.wait_event(self.events[k])


# ---- the two ends -------------------------------------------------------------


class _End:
    def close(self) -> None:
        """Graceful exit: destroy every interprocess event, so HIP removes
        their /dev/shm segments (a killed process leaves its live ones)."""
        torch.cuda.synchronize(self.device)
        for events in (self.ready, self.aux_ready, self.consumed):
            events.events = [None] * SLOTS
        self.retired.items.clear()


class PPIpcReceiver(_End):
    """The receiving side of one stage boundary (this stage's input)."""

    def __init__(self, max_tokens: int, hidden: int, dtype: torch.dtype, aux_cols: int):
        from aiter.dist.parallel_state import get_pp_group

        from atom.mono.runtime.hip_ipc import (
            HipRuntime,
            get_allocation_base,
            get_ipc_handle,
        )
        from atom.mono.runtime.uncached import DeviceBytes

        pp = get_pp_group()
        self.src = pp.ranks[(pp.rank_in_group - 1) % pp.world_size]
        self.group = pp.cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.hidden, self.dtype, self.aux_cols = hidden, dtype, aux_cols
        self.esize = torch.empty(0, dtype=dtype).element_size()
        self.key_bytes = max_tokens * hidden * self.esize
        self.aux_bytes = max_tokens * aux_cols * self.esize
        self.slot_bytes = len(_KEYS) * self.key_bytes + self.aux_bytes
        nbytes = SLOTS * self.slot_bytes
        # Never freed (the ring lives as long as the stage).
        ptr = HipRuntime().allocate_uncached(nbytes, _HIP_DEVICE_MALLOC_DEFAULT)
        self.buf = torch.as_tensor(DeviceBytes(ptr, nbytes), device=self.device)
        base = get_allocation_base(ptr)
        _send_obj(
            {
                "handle": get_ipc_handle(base),
                "offset": ptr - base,
                "slot_bytes": self.slot_bytes,
                "key_bytes": self.key_bytes,
                "aux_bytes": self.aux_bytes,
            },
            self.src,
            self.group,
        )
        self.retired = _Retired()
        self.ready = _Opened(self.device, self.retired)
        self.aux_ready = _Opened(self.device, self.retired)
        self.consumed = _Rotating(self.retired)
        self._aux = None  # (slot, rows, aux tensor, aux_ready handle)
        logger.info(
            f"PP IPC receiver: {SLOTS} slots x {self.slot_bytes / 2**20:.0f} MiB"
            f" (aux {aux_cols} columns)"
        )

    def _view(self, k: int, offset: int, rows: int, cols: int) -> torch.Tensor:
        start = k * self.slot_bytes + offset
        view = self.buf[start : start + rows * cols * self.esize]
        return view.view(self.dtype).view(rows, cols)

    def recv(self) -> tuple[IntermediateTensors, torch.Tensor | None]:
        """This forward's hidden_states/residual, copied out of the ring after
        their transfer, and the aux block's tensor, which `finish_aux` fills."""
        self.retired.reap()
        meta = _recv_obj(self.src, self.group)
        k, rows = meta["slot"], meta["rows"]
        self.ready.wait(k, meta["ready"], torch.cuda.current_stream())
        # Local copies: the model faulted reading the ring slots directly.
        tensors = {
            key: self._view(k, j * self.key_bytes, rows, self.hidden).clone()
            for j, key in enumerate(_KEYS)
        }
        aux = None
        if meta["aux_cols"]:
            aux = torch.empty(rows, self.aux_cols, dtype=self.dtype, device=self.device)
            self._aux = (k, rows, aux, meta["aux_ready"])
        else:
            self._release(k)
        return IntermediateTensors(tensors), aux

    def finish_aux(self) -> None:
        """Fill the aux tensor `recv` returned and free its slot; call before
        the aux is first read."""
        if self._aux is None:
            return
        k, rows, aux, handle = self._aux
        self._aux = None
        self.aux_ready.wait(k, handle, torch.cuda.current_stream())
        aux.copy_(self._view(k, len(_KEYS) * self.key_bytes, rows, self.aux_cols))
        self._release(k)

    def _release(self, k: int) -> None:
        """Every copy out of slot k is enqueued: hand it back to the sender."""
        handle = self.consumed.record(k, torch.cuda.current_stream())
        meta = {"slot": k, "consumed": handle}
        _send_obj(meta, self.src, self.group, blocking=False)


class PPIpcSender(_End):
    """The sending side of one stage boundary (this stage's output)."""

    def __init__(self):
        from aiter.dist.parallel_state import get_pp_group

        from atom.mono.runtime.hip_ipc import open_ipc_handle

        pp = get_pp_group()
        self.dst = pp.ranks[(pp.rank_in_group + 1) % pp.world_size]
        self.group = pp.cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.hip = _hip()
        theirs = _recv_obj(self.dst, self.group)
        self.remote = open_ipc_handle(theirs["handle"]) + theirs["offset"]
        self.slot_bytes = theirs["slot_bytes"]
        self.key_bytes, self.aux_bytes = theirs["key_bytes"], theirs["aux_bytes"]
        self.retired = _Retired()
        self.consumed = _Opened(self.device, self.retired)
        self.ready = _Rotating(self.retired)
        self.aux_ready = _Rotating(self.retired)
        # High priority: its own hardware queue, so the copy's completion
        # barrier never sits in front of the next forward's kernels.
        self.stream = torch.cuda.Stream(self.device, priority=-1)
        self.sent = 0
        logger.info("PP IPC sender ready")

    def _copy(self, dst: int, t: torch.Tensor, limit: int) -> None:
        assert t.is_contiguous(), "PP IPC sends contiguous tensors"
        nbytes = t.numel() * t.element_size()
        assert nbytes <= limit, (nbytes, limit)
        err = self.hip.hipMemcpyAsync(
            ctypes.c_void_p(dst),
            ctypes.c_void_p(t.data_ptr()),
            ctypes.c_size_t(nbytes),
            _HIP_MEMCPY_DEVICE_TO_DEVICE,
            ctypes.c_void_p(self.stream.cuda_stream),
        )
        if err != 0:
            raise RuntimeError(f"hipMemcpyAsync to the next PP stage: {err}")
        # The caching allocator keeps it until the copy is done.
        t.record_stream(self.stream)

    def send(self, it: IntermediateTensors, aux: torch.Tensor | None) -> None:
        """Copy hidden_states/residual (and the relayed aux block) into the
        next stage's ring; they are this forward's own outputs, so no clone."""
        self.retired.reap()
        k = self.sent % SLOTS
        if self.sent >= SLOTS:
            ack = _recv_obj(self.dst, self.group)
            assert ack["slot"] == k, (ack, k)
            self.consumed.wait(k, ack["consumed"], self.stream)
        self.stream.wait_stream(torch.cuda.current_stream())
        slot = self.remote + k * self.slot_bytes
        for j, key in enumerate(_KEYS):
            self._copy(slot + j * self.key_bytes, it.tensors[key], self.key_bytes)
        ready = self.ready.record(k, self.stream)
        aux_ready, aux_cols = None, 0
        if aux is not None:
            aux_cols = aux.shape[1]
            self._copy(slot + len(_KEYS) * self.key_bytes, aux, self.aux_bytes)
            aux_ready = self.aux_ready.record(k, self.stream)
        self.sent += 1
        meta = {
            "slot": k,
            "rows": it.tensors["hidden_states"].shape[0],
            "ready": ready,
            "aux_ready": aux_ready,
            "aux_cols": aux_cols,
        }
        _send_obj(meta, self.dst, self.group, blocking=False)

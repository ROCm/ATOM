# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where the PLE n-gram table's rows live: page-locked host memory, or HBM.

Host-side throughout -- page-locking, the fallback and the shard addresses --
so a CPU-only machine can import and test it. The kernel that reads the rows
is `ops.ple.sharded_table_lookup`.
"""

import ctypes
import functools
import logging
import threading
from collections.abc import Callable

import torch

from atom.model_ops.engram import host_memory

logger = logging.getLogger("atom")


@functools.cache
def _hip_device_pointer():
    # The runtime `host_memory` page-locks with: another one does not know
    # its allocations.
    hip = host_memory._hip_runtime()
    if hip is None:
        return None
    try:
        fn = hip.hipHostGetDevicePointer
    except AttributeError:
        return None
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    return fn


def device_address(host_address: int) -> int:
    """Where the device reaches page-locked host memory at `host_address`.

    Not always the host address: memory registered in place
    (`hipHostRegister`) is mapped at a different device address, and a kernel
    handed the host one faults. Identity where there is no HIP runtime.
    """
    fn = _hip_device_pointer()
    if fn is None:
        return host_address
    out = ctypes.c_void_p()
    if fn(ctypes.byref(out), host_address, 0) != 0 or not out.value:
        raise RuntimeError(f"no device address for host memory at {host_address:#x}")
    return out.value


class PleHostTable:
    """One rank's rows `[vocab_start, vocab_end)` of a table loaded shard by shard.

    Each checkpoint shard is copied into page-locked memory as it arrives
    (`host_memory.pin_rows`), so the table is never whole anywhere and never
    in HBM. `shard_bases` then gives, per shard, the address of its first
    global row for the device to read over UVA.

    Always copied, whatever `ATOM_ENGRAM_UVA_COPY` says: the shards share
    safetensors files with other weights, and registering part of a file
    mapping in place makes ordinary host-to-device copies from the rest of
    that mapping fail (`hipErrorInvalidValue`, or a GPU memory fault). The
    registered pages would also be private copies (the mapping is
    `MAP_PRIVATE`), so registering would not share them across processes.
    And the kernel reads a shard from one base address, while in-place
    registrations of adjacent, unaligned shards share their boundary pages
    (`host_memory._PageRuns`): a shard then spans device ranges that are
    mapped apart.

    One shard that cannot be page-locked moves the table to `device` -- the
    rows already pinned are copied over -- and the addresses point there
    instead. There it is the data of `weight`, the module's parameter, so
    releasing and resuming the weights frees and reallocates it as they would
    an HBM embedding; `shard_bases` then has to be asked again. Loader threads
    may call `load_rows` concurrently.
    """

    def __init__(
        self,
        vocab_start: int,
        vocab_end: int,
        weight: torch.nn.Parameter,
        shard_rows: int,
        device: torch.device,
        prefix: str,
    ):
        self.vocab_start, self.vocab_end = vocab_start, vocab_end
        # `[0, dim]` until the table falls back to HBM.
        self.weight = weight
        self.dim, self.dtype = weight.shape[1], weight.dtype
        self.shard_rows = shard_rows
        self.device = device
        self.prefix = prefix
        # The shards holding this rank's rows.
        self.shards = range(vocab_start // shard_rows, -(-vocab_end // shard_rows))
        # Per shard: (first global row held, page-locked rows, or None once
        # they are in `weight`).
        self._rows: dict[int, tuple[int, torch.Tensor | None]] = {}
        self._in_hbm = False
        self._lock = threading.Lock()

    @property
    def on_host(self) -> bool:
        return not self._in_hbm

    @property
    def pinned_bytes(self) -> int:
        return sum(t.nbytes for _, t in self._rows.values() if t is not None)

    def load_rows(self, shard: int, row_start: int, rows: torch.Tensor) -> None:
        """Take this rank's rows of checkpoint shard `shard`, from global row
        `row_start`."""
        if shard not in self.shards or not (
            self.vocab_start <= row_start
            and row_start + rows.shape[0] <= self.vocab_end
        ):
            raise ValueError(
                f"{self.prefix}: rows [{row_start}, {row_start + rows.shape[0]}) "
                f"of shard {shard} are outside [{self.vocab_start}, "
                f"{self.vocab_end})"
            )
        if not self._in_hbm:
            pinned = host_memory.pin_rows([rows.contiguous()], copy=True)
            with self._lock:
                if pinned is not None and not self._in_hbm:
                    self._rows[shard] = (row_start, pinned[0])
                    return
                self._to_hbm()
        local = row_start - self.vocab_start
        self.weight.data.narrow(0, local, rows.shape[0]).copy_(rows)
        with self._lock:
            self._rows[shard] = (row_start, None)

    def sync_rows(self, first: int, rows: torch.Tensor) -> None:
        """A weight sync's global rows `[first, first + len(rows))`: this
        rank's part is written in place, over the page-locked rows or into
        `weight` in HBM. Nothing is allocated or page-locked again, so
        `shard_bases`, and the graphs that captured it, stay valid.

        The caller makes sure no lookup is still reading the rows.
        """
        if rows.dim() != 2 or rows.shape[1] != self.dim:
            raise ValueError(
                f"{self.prefix}: expected [rows, {self.dim}] n-gram rows, "
                f"got {tuple(rows.shape)}"
            )
        if rows.dtype != self.dtype and self.dtype.itemsize < 2:
            # The FP8 table's rows go with its scale: no conversion is right.
            raise ValueError(
                f"{self.prefix}: expected {self.dtype} n-gram rows, got {rows.dtype}"
            )
        lo = max(first, self.vocab_start)
        hi = min(first + rows.shape[0], self.vocab_end)
        if self._in_hbm:
            if lo < hi:
                self.weight.data.narrow(0, lo - self.vocab_start, hi - lo).copy_(
                    rows.narrow(0, lo - first, hi - lo)
                )
            return
        written = 0
        for row_start, pinned in self._rows.values():
            start = max(lo, row_start)
            end = min(hi, row_start + pinned.shape[0])
            if start < end:
                pinned.narrow(0, start - row_start, end - start).copy_(
                    rows.narrow(0, start - first, end - start)
                )
                written += end - start
        if written != max(hi - lo, 0):
            raise ValueError(
                f"{self.prefix}: n-gram rows [{lo}, {hi}) were never loaded, "
                "so they cannot be synced"
            )

    def _to_hbm(self) -> None:
        """Make `weight` the table on `device`, holding every row pinned so
        far; under `_lock`."""
        if not self._in_hbm:
            logger.warning(
                "%s: page-locking the n-gram table on the host failed; "
                "falling back to HBM",
                self.prefix,
            )
            hbm = torch.empty(
                self.vocab_end - self.vocab_start,
                self.dim,
                dtype=self.dtype,
                device=self.device,
            )
            self.weight.data = hbm
            self._in_hbm = True
            for shard, (row_start, pinned) in list(self._rows.items()):
                if pinned is not None:
                    local = row_start - self.vocab_start
                    hbm.narrow(0, local, pinned.shape[0]).copy_(pinned)
                    self._rows[shard] = (row_start, None)

    def shard_bases(
        self, num_shards: int, initialize: Callable[[], None] | None = None
    ) -> torch.Tensor:
        """`[num_shards]` int64 on the CPU: the address of each local shard's
        first global row (0 for another rank's shards).

        Call once loading is done, and again whenever `weight` has been
        reallocated. A table that received nothing (a dummy load) is
        allocated in `weight`, as an HBM embedding would be, and filled by
        `initialize` -- left uninitialized without one; one missing only some
        shards is an error.
        """
        missing = [shard for shard in self.shards if shard not in self._rows]
        if len(missing) == len(self.shards):
            with self._lock:
                self._to_hbm()
            if initialize is not None:
                initialize()
        elif missing:
            raise ValueError(
                f"{self.prefix}: n-gram checkpoint shards {missing[:8]} were "
                "never loaded"
            )
        row_bytes = self.dim * self.dtype.itemsize
        bases = torch.zeros(num_shards, dtype=torch.int64)
        for shard in self.shards:
            first = shard * self.shard_rows
            if self._in_hbm:
                address = self.weight.data_ptr()
                address += (first - self.vocab_start) * row_bytes
            else:
                row_start, pinned = self._rows[shard]
                address = device_address(pinned.data_ptr())
                address -= (row_start - first) * row_bytes
            bases[shard] = address
        return bases

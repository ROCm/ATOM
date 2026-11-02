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

# Where the table is page-locked on the host by default: the archs the host
# path has been validated on (bitwise lookups, GSM8K, page-locking ~100 GB),
# MI355X and MI308X. Elsewhere it stays in HBM until it has been.
VALIDATED_ARCHS = ("gfx950", "gfx942")


def validated_arch(arch: str) -> bool:
    return arch in VALIDATED_ARCHS


def fp8_rows_fnuz(dtype: torch.dtype) -> bool:
    """Which FP8 E4M3 encoding an FP8 table's rows are stored in, for the
    kernel to read them as: `float8_e4m3fnuz` (True) or `float8_e4m3fn`. The
    two encode the same byte as values a factor of two apart."""
    if dtype not in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        raise ValueError(f"not an FP8 E4M3 n-gram table: {dtype}")
    return dtype == torch.float8_e4m3fnuz


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
    an HBM embedding; `shard_bases` then has to be asked again. Page-locked
    rows are freed and reallocated the same way (`release` / `resume`).

    Loader threads may call `load_rows` concurrently. Each page-locks its rows
    outside `_lock` -- the copy is most of the load, and under the lock the
    threads would take turns -- and records them under it, as everything
    reading or changing `_rows` and `_in_hbm` does.
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
        # Per shard while released: (first global row, number of rows).
        self._released: dict[int, tuple[int, int]] = {}
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
                self._to_hbm(pin_failed=True)
        local = row_start - self.vocab_start
        self.weight.data.narrow(0, local, rows.shape[0]).copy_(rows)
        with self._lock:
            self._rows[shard] = (row_start, None)

    def check_resumed(self) -> None:
        """Raise if the table is released: a weight sync has to wait for the
        resume."""
        with self._lock:
            self._check_resumed()

    def _check_resumed(self, use: str = "sync its rows") -> None:
        # In HBM, `weight` itself is released, with the other parameters.
        if self._released or (self._in_hbm and not self.weight.numel()):
            raise ValueError(
                f"{self.prefix}: the n-gram table is released; {use} after the "
                "weights are resumed"
            )

    def check_rows(self, rows: torch.Tensor) -> None:
        """Raise unless `rows` can be written into the table as they are."""
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

    def sync_rows(self, first: int, rows: torch.Tensor) -> None:
        """A weight sync's global rows `[first, first + len(rows))`: this
        rank's part is written in place, over the page-locked rows or into
        `weight` in HBM. Nothing is allocated or page-locked again, so
        `shard_bases`, and the graphs that captured it, stay valid.

        The caller makes sure no lookup is still reading the rows.
        """
        self.check_rows(rows)
        lo = max(first, self.vocab_start)
        hi = min(first + rows.shape[0], self.vocab_end)
        with self._lock:
            self._check_resumed()
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

    def release(self) -> int:
        """Free the page-locked rows, as releasing the weights frees the
        parameters; the bytes freed. In HBM the table is `weight`, released
        with them, and there is nothing else to free.

        Nothing may read the rows any more: the graphs that captured
        `shard_bases` are released first, and `hipHostFree` waits for the
        device.
        """
        with self._lock:
            if self._in_hbm or self._released:
                return 0
            freed = sum(pinned.nbytes for _, pinned in self._rows.values())
            self._released = {
                shard: (row_start, pinned.shape[0])
                for shard, (row_start, pinned) in self._rows.items()
            }
            self._rows = {}
        return freed

    def resume(self) -> None:
        """Page-lock the rows `release` freed again, at new addresses, so
        `shard_bases` has to be asked again. Their values are whatever the
        memory held, as for the parameters `MemoryManagerMixin` reallocates:
        the weight sync that follows a resume writes them. A shard that cannot
        be page-locked moves the table to HBM, as at load."""
        with self._lock:
            released, self._released = self._released, {}
        for shard, (row_start, num_rows) in released.items():
            rows = torch.empty(num_rows, self.dim, dtype=self.dtype, device="cpu")
            self.load_rows(shard, row_start, rows)

    def _to_hbm(self, pin_failed: bool) -> None:
        """Make `weight` the table on `device`, holding every row pinned so
        far; under `_lock`. Either page-locking a shard failed, or nothing
        was loaded (a dummy load), which is expected and not warned about."""
        if not self._in_hbm:
            if pin_failed:
                logger.warning(
                    "%s: page-locking the n-gram table on the host failed; "
                    "falling back to HBM",
                    self.prefix,
                )
            else:
                logger.info(
                    "%s: no n-gram rows loaded (dummy load); the table is "
                    "allocated in HBM",
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
        reallocated or the rows resumed. A table that received nothing (a
        dummy load) is allocated in `weight`, as an HBM embedding would be,
        and filled by `initialize` -- left uninitialized without one; one
        missing only some shards is an error.
        """
        with self._lock:
            self._check_resumed("take its addresses")
            missing = [shard for shard in self.shards if shard not in self._rows]
            empty = len(missing) == len(self.shards)
            if empty:
                self._to_hbm(pin_failed=False)
            elif missing:
                raise ValueError(
                    f"{self.prefix}: n-gram checkpoint shards {missing[:8]} were "
                    "never loaded"
                )
        if empty and initialize is not None:
            initialize()
        row_bytes = self.dim * self.dtype.itemsize
        bases = torch.zeros(num_shards, dtype=torch.int64, device="cpu")
        with self._lock:
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

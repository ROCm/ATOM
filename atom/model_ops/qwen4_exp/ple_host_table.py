# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Where the PLE n-gram table's rows live: page-locked host memory, or HBM.

Host-side throughout -- page-locking, the fallback and the shard addresses --
so a CPU-only machine can import and test it. The kernel that reads the rows
is `ops.ple.sharded_table_lookup`.
"""

import logging
import threading

import torch

from atom.model_ops.engram import host_memory

logger = logging.getLogger("atom")


def check_fp8_rows(prefix: str, rows: torch.Tensor, dtype: torch.dtype) -> None:
    """An FP8 table's rows go with its scale: no conversion is right."""
    if rows.dtype != dtype and dtype.itemsize < 2:
        raise ValueError(f"{prefix}: expected {dtype} n-gram rows, got {rows.dtype}")


class PleHostTable:
    """One rank's rows `[vocab_start, vocab_end)` of a table loaded shard by shard.

    Each checkpoint shard is copied into page-locked memory as it arrives, so
    the table is never whole anywhere and never in HBM; `shard_bases` gives the
    device the address of each shard's first global row. Always copied: the
    shards share safetensors files with other weights, and registering part of
    a file mapping in place makes host-to-device copies from the rest of it
    fail.

    One shard that cannot be page-locked moves the table to `device`, as the
    data of `weight`, so it is released and resumed with the parameters.

    Loader threads may call `load_rows` concurrently: the copy runs outside
    `_lock` (it is most of the load), and `_rows` / `_in_hbm` change under it.
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
        self.shards = range(
            vocab_start // shard_rows, (vocab_end + shard_rows - 1) // shard_rows
        )
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
        `row_start`; a shard already held (a weight sync) is written in place."""
        with self._lock:
            self._check_resumed("sync its rows")
            held = shard in self._rows
        if held:
            self.sync_rows(row_start, rows)
        else:
            self._take_rows(shard, row_start, rows)

    def _take_rows(self, shard: int, row_start: int, rows: torch.Tensor) -> None:
        if not self._in_hbm:
            pinned = host_memory.pin_rows([rows.contiguous()], copy=True)
            with self._lock:
                if pinned is not None and not self._in_hbm:
                    self._rows[shard] = (row_start, pinned[0])
                    return
                if not self._in_hbm:
                    logger.warning(
                        "%s: page-locking the n-gram table on the host failed; "
                        "falling back to HBM",
                        self.prefix,
                    )
                self._to_hbm()
        local = row_start - self.vocab_start
        self.weight.data.narrow(0, local, rows.shape[0]).copy_(rows)
        with self._lock:
            self._rows[shard] = (row_start, None)

    def _check_resumed(self, use: str) -> None:
        # In HBM, `weight` itself is released, with the other parameters.
        if self._released or (self._in_hbm and not self.weight.numel()):
            raise ValueError(
                f"{self.prefix}: the n-gram table is released; {use} after the "
                "weights are resumed"
            )

    def sync_rows(self, first: int, rows: torch.Tensor) -> None:
        """Write this rank's part of global rows `[first, first + len(rows))`
        in place, so `shard_bases` and the graphs that captured it stay valid.
        The caller makes sure no lookup is still reading the rows."""
        if rows.dim() != 2 or rows.shape[1] != self.dim:
            raise ValueError(
                f"{self.prefix}: expected [rows, {self.dim}] n-gram rows, "
                f"got {tuple(rows.shape)}"
            )
        check_fp8_rows(self.prefix, rows, self.dtype)
        lo = max(first, self.vocab_start)
        hi = min(first + rows.shape[0], self.vocab_end)
        with self._lock:
            self._check_resumed("sync its rows")
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

    def release(self) -> None:
        """Free the page-locked rows (in HBM, `weight` is released with the
        parameters). The graphs that captured `shard_bases` are released
        first, and `hipHostFree` waits for the device."""
        with self._lock:
            if self._in_hbm:
                return
            self._released |= {
                shard: (row_start, pinned.shape[0])
                for shard, (row_start, pinned) in self._rows.items()
            }
            self._rows = {}

    def resume(self) -> None:
        """Page-lock the released rows again, at new addresses and with
        undefined values, as the parameters are reallocated; the weight sync
        after a resume writes them. A shard stays released until it is back,
        so a resume that runs out of memory can be retried."""
        with self._lock:
            released = list(self._released.items())
        for shard, (row_start, num_rows) in released:
            rows = torch.empty(num_rows, self.dim, dtype=self.dtype, device="cpu")
            self._take_rows(shard, row_start, rows)
            with self._lock:
                del self._released[shard]

    def _to_hbm(self) -> None:
        """Make `weight` the table on `device`, with the rows pinned so far;
        under `_lock`."""
        if not self._in_hbm:
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

    def shard_bases(self, num_shards: int) -> torch.Tensor:
        """`[num_shards]` int64 on the CPU: the address of each local shard's
        first global row (0 for other ranks'). Ask again after a resume."""
        row_bytes = self.dim * self.dtype.itemsize
        bases = torch.zeros(num_shards, dtype=torch.int64, device="cpu")
        with self._lock:
            self._check_resumed("take its addresses")
            missing = [shard for shard in self.shards if shard not in self._rows]
            if missing:
                raise ValueError(
                    f"{self.prefix}: n-gram checkpoint shards {missing[:8]} were "
                    "never loaded"
                )
            for shard in self.shards:
                first = shard * self.shard_rows
                if self._in_hbm:
                    address = self.weight.data_ptr()
                    address += (first - self.vocab_start) * row_bytes
                else:
                    row_start, pinned = self._rows[shard]
                    address = pinned.data_ptr() - (row_start - first) * row_bytes
                bases[shard] = address
        return bases


class HostTableHooks:
    """The weight lifecycle hooks of a module whose rows are in a
    `PleHostTable` (`table`, `num_embeddings` rows in all), read through
    `shard_bases`."""

    def load_shard(self, shard_index: int, row_start: int, rows: torch.Tensor) -> None:
        self.table.load_rows(shard_index, row_start, rows)

    def sync_weight(
        self, param_name: str, tensor: torch.Tensor, shard_id: int | None = None
    ) -> bool:
        """`WeightUpdaterMixin`: written in place, a checkpoint shard through
        the shard loader it was loaded with."""
        if param_name != "weight":
            return False
        if shard_id is not None:
            self.weight.weight_loader(self.weight, tensor, shard_id)
        elif tensor.shape[0] != self.num_embeddings:
            raise ValueError(
                f"{self.prefix}: expected {self.num_embeddings} n-gram rows, "
                f"got {tuple(tensor.shape)}"
            )
        else:
            self.table.sync_rows(0, tensor)
        return True

    def release_weight_memory(self) -> None:
        self.table.release()

    def resume_weight_memory(self) -> None:
        self.table.resume()
        self.shard_bases.copy_(self.table.shard_bases(self.shard_bases.numel()))

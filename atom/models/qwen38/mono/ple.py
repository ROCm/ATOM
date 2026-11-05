# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Mono's placement of the PLE n-gram table: its checkpoint shards loaded directly into page-locked host memory.

The Native PLE prelude still owns hashing, gating and recurrent state; only where the BF16 table lives changes (the
rows read are the checkpoint's, bit for bit). Chosen at construction, before checkpoint loading, when Mono serves the
deployment (``config.config_refusal``, TP1); otherwise the model keeps ATOM's own embedding.
"""

import ctypes
import functools
import logging
import threading

import torch
import triton
import triton.language as tl
from torch import nn

from atom.model_ops.engram import host_memory
from atom.model_ops.utils import atom_parameter

logger = logging.getLogger("atom")


def pin_rows(sources):
    """Copy with ATOM's exact-size HIP allocator; never register a file mapping.

    Main's public pin_rows lacks F's explicit copy override. Reuse its allocator,
    leases and chunked copy without changing Engram's allocation policy.
    """
    allocator = host_memory._host_allocator()
    if allocator is None:
        return None
    leases = []
    try:
        out = host_memory._copy(allocator, sources, leases)
    except Exception:
        logger.warning("Mono PLE: host shard copy failed", exc_info=True)
        out = None
    if out is None:
        for lease in leases:
            lease.release()
    return out


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


class MonoPleHostTable:
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
    instead. Loader threads may call `load_rows` concurrently.
    """

    def __init__(
        self,
        vocab_start: int,
        vocab_end: int,
        dim: int,
        dtype: torch.dtype,
        shard_rows: int,
        device: torch.device,
        prefix: str,
    ):
        self.vocab_start, self.vocab_end = vocab_start, vocab_end
        self.dim, self.dtype = dim, dtype
        self.shard_rows = shard_rows
        self.device = device
        self.prefix = prefix
        # The shards holding this rank's rows.
        self.shards = range(vocab_start // shard_rows, -(-vocab_end // shard_rows))
        # Per shard: (first global row held, page-locked rows, or None once
        # they are in `_hbm`).
        self._rows: dict[int, tuple[int, torch.Tensor | None]] = {}
        self._hbm: torch.Tensor | None = None
        self._lock = threading.Lock()

    @property
    def on_host(self) -> bool:
        return self._hbm is None

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
        hbm = self._hbm
        if hbm is None:
            pinned = pin_rows([rows.contiguous()])
            with self._lock:
                if pinned is not None and self._hbm is None:
                    self._rows[shard] = (row_start, pinned[0])
                    return
                hbm = self._to_hbm()
        hbm.narrow(0, row_start - self.vocab_start, rows.shape[0]).copy_(rows)
        with self._lock:
            self._rows[shard] = (row_start, None)

    def _to_hbm(self) -> torch.Tensor:
        """The table on `device`, holding every row pinned so far; under `_lock`."""
        if self._hbm is None:
            logger.warning(
                "%s: page-locking the n-gram table on the host failed; "
                "falling back to HBM",
                self.prefix,
            )
            self._hbm = torch.empty(
                self.vocab_end - self.vocab_start,
                self.dim,
                dtype=self.dtype,
                device=self.device,
            )
            for shard, (row_start, pinned) in list(self._rows.items()):
                if pinned is not None:
                    local = row_start - self.vocab_start
                    self._hbm.narrow(0, local, pinned.shape[0]).copy_(pinned)
                    self._rows[shard] = (row_start, None)
        return self._hbm

    def shard_bases(self, num_shards: int) -> torch.Tensor:
        """`[num_shards]` int64 on the CPU: the address of each local shard's
        first global row (0 for another rank's shards).

        Call once loading is done. A table that received nothing (a dummy
        load) is allocated on `device` uninitialized, as an HBM embedding
        would be; one missing only some shards is an error.
        """
        missing = [shard for shard in self.shards if shard not in self._rows]
        if len(missing) == len(self.shards):
            with self._lock:
                self._to_hbm()
        elif missing:
            raise ValueError(
                f"{self.prefix}: n-gram checkpoint shards {missing[:8]} were "
                "never loaded"
            )
        row_bytes = self.dim * self.dtype.itemsize
        bases = torch.zeros(num_shards, dtype=torch.int64)
        for shard in self.shards:
            first = shard * self.shard_rows
            if self._hbm is not None:
                address = self._hbm.data_ptr()
                address += (first - self.vocab_start) * row_bytes
            else:
                row_start, pinned = self._rows[shard]
                address = device_address(pinned.data_ptr())
                address -= (row_start - first) * row_bytes
            bases[shard] = address
        return bases


class MonoPleEmbedding(nn.Module):
    """The BF16 PLE table (TP1: every row) in page-locked host memory, read over UVA.

    The table is ~100 GB and a token reads `ngram_heads` rows of it, so it
    does not earn its HBM. `MonoPleHostTable` page-locks each checkpoint shard as
    it loads, and `sharded_table_lookup` reads the rows a step names across the
    bus through `shard_bases`. Those addresses are fixed from load to exit, so
    the lookup can be captured in a CUDA graph. If page-locking fails the
    table moves to HBM and the same kernel reads it there; either way the rows
    returned are the checkpoint's, bit for bit.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, shard_rows: int, prefix: str
    ):
        super().__init__()
        self.prefix = prefix
        self.vocab_start_idx, self.vocab_end_idx = 0, num_embeddings
        self.embedding_dim = embedding_dim
        self.shard_rows = shard_rows
        self.output_dtype = dtype = torch.get_default_dtype()
        # The loader's handle on the table: the dtype the shard loader checks
        # and the `weight_loader`, holding no rows.
        self.weight = atom_parameter(
            torch.empty(0, embedding_dim, dtype=dtype, device="cpu")
        )
        self.register_buffer(
            "shard_bases",
            torch.zeros(-(-num_embeddings // shard_rows), dtype=torch.int64),
            persistent=False,
        )
        self.table = MonoPleHostTable(
            self.vocab_start_idx,
            self.vocab_end_idx,
            embedding_dim,
            dtype,
            shard_rows,
            self.shard_bases.device,
            prefix,
        )

    def process_weights_after_loading(self) -> None:
        self.shard_bases.copy_(self.table.shard_bases(self.shard_bases.numel()))
        if self.table.on_host:
            logger.info(
                "%s: n-gram table in page-locked host memory, %.1f GiB read "
                "over UVA, none in HBM",
                self.prefix,
                self.table.pinned_bytes / 1024**3,
            )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return sharded_table_lookup(
            ids,
            self.shard_bases,
            self.vocab_start_idx,
            self.vocab_end_idx,
            self.shard_rows,
            self.embedding_dim,
            self.output_dtype,
        )

    def load_rows(self, shard, row_start, rows):
        self.table.load_rows(shard, row_start, rows)


@triton.jit
def _mono_ple_lookup_kernel(
    Ids,
    Bases,
    Output,
    num_ids,
    START,
    END,
    SHARD_ROWS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Gather rows of a table held as separately allocated shards.

    `Bases[s]` is the address of global row `s * SHARD_ROWS`, wherever shard
    `s` lives -- page-locked host memory read over UVA, or HBM. Rows outside
    `[START, END)` (another rank's, or a placeholder id) store zeros. The grid
    is persistent: over the bus the cost is latency, and fewer programs keep
    more of their address translations warm.
    """
    cols = tl.arange(0, BLOCK_D)
    in_row = cols < DIM
    row_dtype = Output.dtype.element_ty
    for first in tl.range(
        tl.program_id(0) * BLOCK_R, num_ids, tl.num_programs(0) * BLOCK_R
    ):
        rows = first + tl.arange(0, BLOCK_R)
        valid = rows < num_ids
        token = tl.load(Ids + rows, mask=valid, other=-1).to(tl.int64)
        owned = valid & (token >= START) & (token < END)
        token = tl.where(owned, token, 0)
        shard = token // SHARD_ROWS
        base = tl.load(Bases + shard, mask=owned, other=0)
        src = base.to(tl.pointer_type(row_dtype))
        offsets = (token - shard * SHARD_ROWS)[:, None] * DIM + cols[None, :]
        values = tl.load(
            src[:, None] + offsets, mask=owned[:, None] & in_row[None, :], other=0.0
        )
        tl.store(
            Output + rows.to(tl.int64)[:, None] * DIM + cols[None, :],
            values,
            mask=valid[:, None] & in_row[None, :],
        )


@functools.cache
def _compute_units(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def sharded_table_lookup(
    ids: torch.Tensor,
    bases: torch.Tensor,
    vocab_start_idx: int,
    vocab_end_idx: int,
    shard_rows: int,
    dim: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Rows `ids` of a sharded table (`_mono_ple_lookup_kernel`), `[N, dim]`,
    copied in `output_dtype`, which is their dtype."""
    ids = ids.reshape(-1)
    output = torch.empty((ids.numel(), dim), dtype=output_dtype, device=ids.device)
    if ids.numel() == 0:
        return output
    block_rows = 16
    grid = min(
        triton.cdiv(ids.numel(), block_rows), 4 * _compute_units(ids.device.index)
    )
    _mono_ple_lookup_kernel[(grid,)](
        ids,
        bases,
        output,
        ids.numel(),
        vocab_start_idx,
        vocab_end_idx,
        shard_rows,
        dim,
        triton.next_power_of_2(dim),
        block_rows,
    )
    return output


def _mono_serves() -> bool:
    """Whether Mono serves this deployment (``config.config_refusal`` of the current ATOM config)."""
    from atom.config import get_current_atom_config
    from atom.models.qwen38.mono.config import config_refusal

    try:
        atom_config = get_current_atom_config()
    except AssertionError:  # (no current config: not an ATOM engine's model)
        return False
    return config_refusal(atom_config) is None


def make_host_embedding(num_embeddings, embedding_dim, shard_rows, prefix, fp8):
    """Mono's table placement, chosen at construction before checkpoint loading:
    a BF16 table in page-locked host memory, or None (ATOM's own embedding)."""
    if (
        fp8
        or torch.version.hip is None
        or torch.empty(0).device.type != "cuda"
        or not _mono_serves()
    ):
        return None
    return MonoPleEmbedding(num_embeddings, embedding_dim, shard_rows, prefix)

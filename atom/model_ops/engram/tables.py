# SPDX-License-Identifier: MIT
"""Native host table provider from ROCm/ATOM PR #2185; no request scheduling.

Host-side throughout, including the page-locking: the kernel that reads a
registered shard lives in `device.uva`, so that the prefetch runtime can import
this module on a machine with no Triton.
"""

import ctypes
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from atom.utils import envs

logger = logging.getLogger("atom")

# Threads copying a shard into page-locked memory; one thread falls well short
# of what the node can copy out of the checkpoint's page cache.
_COPY_THREADS = 16


class _HipHostAllocator:
    """`hipHostMalloc`/`hipHostFree` through ctypes.

    Not torch's pinned allocator: that one rounds every block up to a power of
    two, which for a large table shard wastes a large share of page-locked
    memory. ctypes also drops the GIL for the call.
    """

    def __init__(self, lib):
        self._malloc, self._free = lib.hipHostMalloc, lib.hipHostFree
        self._malloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        self._free.argtypes = [ctypes.c_void_p]

    def alloc(self, nbytes: int) -> int | None:
        ptr = ctypes.c_void_p()
        if self._malloc(ctypes.byref(ptr), max(nbytes, 1), 0) != 0 or not ptr.value:
            return None
        return ptr.value

    def free(self, ptr: int) -> None:
        self._free(ptr)


def _parallel_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """`dst.copy_(src)` for flat byte tensors, in chunks on `_COPY_THREADS`
    threads (each `copy_` releases the GIL)."""
    nbytes = src.numel()
    step = max(1, -(-nbytes // (_COPY_THREADS * 4)))

    def copy_chunk(start: int) -> None:
        dst[start : start + step].copy_(src[start : start + step])

    with ThreadPoolExecutor(_COPY_THREADS) as pool:
        list(pool.map(copy_chunk, range(0, nbytes, step)))


def _host_allocator():
    """The page-locked allocator `enable_uva` copies a shard into, or None to
    register the checkpoint mapping in place instead (`ATOM_ENGRAM_UVA_COPY=0`,
    or no HIP runtime to call)."""
    if not envs.ATOM_ENGRAM_UVA_COPY or not torch.version.hip:
        return None
    try:
        return _HipHostAllocator(ctypes.CDLL("libamdhip64.so"))
    except OSError:
        return None


class HostEmbeddingTable:
    """One engram layer's table, memory-mapped and gathered row-wise.

    The reference keeps the table as a float32 numpy array, which for this model
    would be 393 GB per layer -- 786 GB of host RAM for the pair, before any
    staging buffer. Rows are kept in their stored dtype and converted only after
    the gather, so the resident cost is the page cache the OS chooses to keep.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        num_rows: int,
        head_dim: int,
        scale: torch.Tensor | None = None,
    ):
        if tensor.shape[0] != num_rows:
            raise ValueError(f"table has {tensor.shape[0]} rows, expected {num_rows}")
        if tensor.shape[1] != head_dim:
            raise ValueError(
                f"table row is {tensor.shape[1]} wide, expected {head_dim}"
            )
        self._tensor = tensor
        self.num_rows = num_rows
        self.head_dim = head_dim
        self._scale = scale
        self.block_size = 0
        # The registered row range, and the page-aligned base of each
        # registration made for it.
        self._uva: tuple[int, int] | None = None
        self._pinned: tuple[int, ...] = ()
        # Set when the range was copied into page-locked memory rather than
        # registered in place: the copies, as the kernel addresses them.
        self._copies: tuple[torch.Tensor, ...] = ()
        self._allocator = None
        if scale is not None:
            # gather() does `scale.to(float32)`; that decodes 2**(code-127) only
            # for a float8 E8M0 dtype. A raw uint8 exponent-code table would be
            # read as plain magnitudes (~127x off), so fail loud instead.
            if not scale.is_floating_point():
                raise ValueError(
                    f"engram block scale must be a float8 (E8M0) dtype, got "
                    f"{scale.dtype}"
                )
            if scale.shape[0] != num_rows:
                raise ValueError(
                    f"scale has {scale.shape[0]} rows, expected {num_rows}"
                )
            if head_dim % scale.shape[1]:
                raise ValueError(
                    f"head_dim {head_dim} is not divisible by {scale.shape[1]} "
                    f"scale blocks"
                )
            self.block_size = head_dim // scale.shape[1]

    @property
    def dtype(self) -> torch.dtype:
        return self._tensor.dtype

    @property
    def quantized(self) -> bool:
        """Whether rows carry a block scale, and so need decoding after a gather."""
        return self._scale is not None

    def gather(
        self, row_indices: np.ndarray, out_dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:
        """Gather rows named by `row_indices` ([...] ints) -> [..., head_dim].

        Out-of-range indices are a bug in the hash layout rather than something
        to clamp away quietly: a clamp turns a wrong table into plausible
        numbers, which is far harder to notice than an exception.
        """
        flat = np.ascontiguousarray(row_indices.reshape(-1))
        if flat.size and (flat.min() < 0 or flat.max() >= self.num_rows):
            raise IndexError(
                f"engram row index out of range: [{flat.min()}, {flat.max()}] "
                f"not within [0, {self.num_rows})"
            )
        index = torch.from_numpy(flat)
        # Gather the selected rows together.
        # PyTorch 2.9 has no CPU advanced-index kernel for float8. Gather the
        # stored bytes first, then reinterpret only the selected rows.
        rows = self._gather_rows(self._tensor, index).to(out_dtype)
        if self._scale is not None:
            # Block-quantized: each scale covers `block_size` consecutive values
            # of a row. Skipping this does not fail, it returns values two orders
            # of magnitude off, so it is not optional.
            scale = self._gather_rows(self._scale, index).to(out_dtype)
            rows = (
                rows.reshape(-1, scale.shape[1], self.block_size) * scale.unsqueeze(-1)
            ).reshape(-1, self.head_dim)
        return rows.reshape(*row_indices.shape, self.head_dim)

    # ---- UVA path ----------------------------------------------------------
    # The serving default (ATOM_ENGRAM_UVA), with the host gather above kept as
    # the reference implementation. Only THIS RANK'S shard of the table is
    # page-locked -- no HBM -- and a device kernel reads the rows it needs
    # across the bus, which also moves the fp8 dequantization off the host.
    #
    # The shard is a whole number of hash heads. Each head owns a disjoint,
    # contiguous, prime-sized row range, so whole heads are a contiguous BYTE
    # range -- which is what lets a rank register a slice instead of the ~98 GB
    # table. Registering the whole table on every rank is what a TP=4 job cannot
    # afford: 4 x 203 GB of unswappable pages.

    _PAGE = 4096

    def enable_uva(self, row_start: int = 0, row_end: int | None = None) -> bool:
        """Page-lock rows `[row_start, row_end)` so a device kernel can read them.

        By default the rows are copied into freshly page-locked memory
        (`_copy_into_pinned`). Registering the checkpoint mapping in place
        instead copies every page on its write fault (the mapping is private)
        on one thread while holding the HIP runtime lock, which also stalls
        the rest of startup. The memory is the same either way; the registered
        pages are anonymous copies too.

        In place, the range is widened to page boundaries, which registration
        requires; the extra bytes are neighbouring rows this rank never
        addresses.
        """
        row_end = self.num_rows if row_end is None else row_end
        if not 0 <= row_start <= row_end <= self.num_rows:
            raise ValueError(
                f"engram shard rows [{row_start}, {row_end}) outside "
                f"[0, {self.num_rows})"
            )
        if self._uva == (row_start, row_end):
            return True
        allocator = _host_allocator()
        if allocator is not None:
            return self._copy_into_pinned(allocator, row_start, row_end)
        rt = torch.cuda.cudart()
        regions = [(self._tensor, self.head_dim)]
        if self._scale is not None:
            regions.append((self._scale, self._scale.shape[1]))
        pinned = []
        for tensor, width in regions:
            item = tensor.element_size()
            base = tensor.data_ptr() + row_start * width * item
            nbytes = (row_end - row_start) * width * item
            lo = base - (base % self._PAGE)
            size = -(-(base + nbytes - lo) // self._PAGE) * self._PAGE
            if int(rt.cudaHostRegister(lo, size, 0)) != 0:
                # This object is about to report that it holds neither region.
                for done in pinned:
                    rt.cudaHostUnregister(done)
                return False
            pinned.append(lo)
        self._pinned = tuple(pinned)
        self._uva = (row_start, row_end)
        return True

    def _copy_into_pinned(self, allocator, row_start: int, row_end: int) -> bool:
        """Copy the shard's rows (and scales) into page-locked memory of exactly
        their size, on `_COPY_THREADS` threads; all-or-nothing.

        False, with everything taken freed, on any failure -- no memory, no
        thread to copy on -- so the TP group can fall back as one.
        """
        regions = [self._tensor]
        if self._scale is not None:
            regions.append(self._scale)
        taken, copies = [], []
        try:
            for tensor in regions:
                src = tensor[row_start:row_end]
                nbytes = src.numel() * src.element_size()
                ptr = allocator.alloc(nbytes)
                if ptr is None:
                    raise MemoryError(f"no page-locked memory for {nbytes} bytes")
                taken.append(ptr)
                dst = torch.frombuffer(
                    (ctypes.c_char * max(nbytes, 1)).from_address(ptr),
                    dtype=torch.uint8,
                )[:nbytes]
                _parallel_copy(dst, src.reshape(-1).view(torch.uint8))
                copies.append(dst.view(tensor.dtype).view(src.shape))
        except Exception:
            logger.warning("engram: copying the UVA shard failed", exc_info=True)
            for ptr in taken:
                allocator.free(ptr)
            return False
        self._copies = tuple(copies)
        self._pinned = tuple(taken)
        self._allocator = allocator
        self._uva = (row_start, row_end)
        return True

    def disable_uva(self) -> None:
        """Release this table's page-locked range, if it holds one.

        What comes back is what went in, rather than the same offsets derived a
        second time -- two copies of that arithmetic would have to agree.
        """
        if self._uva is None:
            return
        if self._copies:
            self._copies = ()
            for ptr in self._pinned:
                self._allocator.free(ptr)
            self._pinned = ()
            self._uva = None
            return
        rt = torch.cuda.cudart()
        for base in self._pinned:
            rt.cudaHostUnregister(base)
        self._pinned = ()
        self._uva = None

    def registered_shard(self):
        """`(weight, scales, row_start, row_end)` as the UVA kernel addresses them.

        The rows are the registered range alone, so the kernel reaches a row by
        `index - row_start`. `scales` is the exponent bytes -- Triton has no
        pointer type for float8_e8m0fnu -- or, for an unquantized table, the
        weight itself, which is the argument the kernel then never reads.
        """
        if self._uva is None:
            raise RuntimeError("engram UVA lookup needs enable_uva() first")
        row_start, row_end = self._uva
        if self._copies:
            weight = self._copies[0]
            scales = (
                weight if self._scale is None else self._copies[1].view(torch.uint8)
            )
            return weight, scales, row_start, row_end
        weight = self._tensor[row_start:row_end]
        scales = (
            weight
            if self._scale is None
            else self._scale[row_start:row_end].view(torch.uint8)
        )
        return weight, scales, row_start, row_end

    @staticmethod
    def _gather_rows(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if tensor.element_size() == 1:
            return tensor.view(torch.uint8)[index].view(tensor.dtype)
        return tensor[index]

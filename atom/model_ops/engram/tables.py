# SPDX-License-Identifier: MIT
"""Native host table provider from ROCm/ATOM PR #2185; no request scheduling.

Host-side throughout, including the page-locking: the kernel that reads a
registered shard lives in `device.uva`, so that the prefetch runtime can import
this module on a machine with no Triton.
"""

import ctypes
import functools

import numpy as np
import torch


@functools.cache
def _hip_runtime():
    """The HIP runtime torch already loaded (e.g. a wheel's libamdhip64.so.7)."""
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                if "libamdhip64" in line:
                    return ctypes.CDLL(line.split()[-1])
    except OSError:
        pass
    return None


def _clear_last_error() -> None:
    """Reset HIP's last error after a refused registration.

    `cudaHostRegister` reports the refusal through its return code but also
    leaves it as the runtime's last error, and torch raises that at its next
    kernel-launch check -- so the host-path fallback would die there ("part or
    all of the requested memory range is already mapped").
    """
    hip = _hip_runtime()
    if hip is not None:
        hip.hipGetLastError()


class _PageRuns:
    """Page-locked runs of this process, reference-counted.

    `cudaHostRegister` refuses any range that overlaps one already registered,
    and a table widens each range to page boundaries. Ranges stored back to
    back want the page they share: a layer's scales sit right before its
    values in the checkpoint shard, and a rank that holds every hash head (one
    DP-attention rank per GPU) registers both whole. A run is registered once
    and released with its last user.
    """

    def __init__(self):
        self._runs: dict[int, list[int]] = {}  # base -> [end, users]

    def acquire(self, rt, lo: int, hi: int) -> list[int] | None:
        """Reference the runs covering `[lo, hi)`, registering the uncovered
        gaps; their bases, or None (holding nothing) if a registration is
        refused."""
        held: list[int] = []
        cursor = lo
        overlapping = sorted(b for b, (e, _) in self._runs.items() if b < hi and e > lo)
        for base in overlapping + [None]:
            gap_end = hi if base is None else base
            if cursor < gap_end:
                if int(rt.cudaHostRegister(cursor, gap_end - cursor, 0)) != 0:
                    _clear_last_error()
                    self.release(rt, held)
                    return None
                self._runs[cursor] = [gap_end, 1]
                held.append(cursor)
            if base is None:
                break
            self._runs[base][1] += 1
            held.append(base)
            cursor = max(cursor, self._runs[base][0])
        return held

    def release(self, rt, bases) -> None:
        for base in bases:
            run = self._runs[base]
            run[1] -= 1
            if run[1] == 0:
                rt.cudaHostUnregister(base)
                del self._runs[base]


_PAGE_RUNS = _PageRuns()


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
        # The registered row range, and the page-locked runs (see `_PageRuns`)
        # this table holds a reference on for it.
        self._uva: tuple[int, int] | None = None
        self._pinned: tuple[int, ...] = ()
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
    # page-locked in place -- no copy, no HBM -- and a device kernel reads the
    # rows it needs across the bus, which also moves the fp8 dequantization off
    # the host.
    #
    # The shard is a whole number of hash heads. Each head owns a disjoint,
    # contiguous, prime-sized row range, so whole heads are a contiguous BYTE
    # range -- which is what lets a rank register a slice instead of the ~98 GB
    # table. Registering the whole table on every rank is what a TP=4 job cannot
    # afford: 4 x 203 GB of unswappable pages.

    _PAGE = 4096

    def enable_uva(self, row_start: int = 0, row_end: int | None = None) -> bool:
        """Page-lock rows `[row_start, row_end)` so a device kernel can read them.

        Registers the existing mapping rather than copying it. Registration
        faults the pages in, so it is slow (~1 GB/s) and happens once, at load.
        The range is widened to page boundaries, which registration requires;
        the extra bytes are neighbouring rows this rank simply never addresses.
        """
        row_end = self.num_rows if row_end is None else row_end
        if not 0 <= row_start <= row_end <= self.num_rows:
            raise ValueError(
                f"engram shard rows [{row_start}, {row_end}) outside "
                f"[0, {self.num_rows})"
            )
        if self._uva == (row_start, row_end):
            return True
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
            hi = -(-(base + nbytes) // self._PAGE) * self._PAGE
            runs = _PAGE_RUNS.acquire(rt, lo, hi)
            if runs is None:
                # This object is about to report that it holds neither region.
                _PAGE_RUNS.release(rt, pinned)
                return False
            pinned.extend(runs)
        self._pinned = tuple(pinned)
        self._uva = (row_start, row_end)
        return True

    def disable_uva(self) -> None:
        """Release this table's page-locked range, if it holds one.

        What comes back is what went in, rather than the same offsets derived a
        second time -- two copies of that arithmetic would have to agree.
        """
        if self._uva is None:
            return
        _PAGE_RUNS.release(torch.cuda.cudart(), self._pinned)
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

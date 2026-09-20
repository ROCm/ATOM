# SPDX-License-Identifier: MIT
"""Native host table provider from ROCm/ATOM PR #2185; no request scheduling."""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _uva_lookup_kernel(
    weight,
    scales,
    ids,
    out,
    num_rows,
    vocab_start,
    vocab_end,
    ids_stride_t,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    """Gather this rank's hash heads from a host table over UVA, dequantize, store.

    `weight`/`scales` address page-locked HOST memory holding only this rank's
    shard, so a row is addressed by `index - vocab_start`. A head this rank does
    not own writes zeros, which is what makes the all-gather that follows a plain
    concatenation. The ue8m0 scale byte IS an fp32 exponent field, so its decode
    is a shift.

    The grid is persistent and sized to the device rather than to the batch: the
    table dwarfs any TLB, so the win is in reusing warmed translations, not in
    one program per row.
    """
    cols = tl.arange(0, DIM)
    for base in tl.range(
        tl.program_id(0) * BLOCK_R, num_rows, tl.num_programs(0) * BLOCK_R
    ):
        rows = base + tl.arange(0, BLOCK_R)
        valid = rows < num_rows
        head = HEAD_START + rows % LOCAL_HEADS
        token = (rows // LOCAL_HEADS).to(tl.int64)
        index = tl.load(
            ids + token * ids_stride_t + head,
            mask=valid & (head < TOTAL_HEADS),
            other=-1,
        ).to(tl.int64)
        owned = valid & (head < TOTAL_HEADS)
        owned &= (index >= vocab_start) & (index < vocab_end)
        local = tl.where(owned, index - vocab_start, 0)
        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        ).to(tl.float32)
        if HAS_SCALE:
            scale = tl.load(
                scales
                + local[:, None] * (DIM // QUANT_BLOCK)
                + (cols // QUANT_BLOCK)[None, :],
                mask=owned[:, None],
                other=0,
            )
            values = values * (scale.to(tl.int32) << 23).to(tl.float32, bitcast=True)
        tl.store(out + rows[:, None] * DIM + cols[None, :], values, mask=valid[:, None])


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
        if getattr(self, "_uva", None) == (row_start, row_end):
            return True
        rt = torch.cuda.cudart()
        for tensor, width in (
            (self._tensor, self.head_dim),
            (self._scale, None if self._scale is None else self._scale.shape[1]),
        ):
            if tensor is None:
                continue
            item = tensor.element_size()
            base = tensor.data_ptr() + row_start * width * item
            nbytes = (row_end - row_start) * width * item
            lo = base - (base % self._PAGE)
            size = -(-(base + nbytes - lo) // self._PAGE) * self._PAGE
            if int(rt.cudaHostRegister(lo, size, 0)) != 0:
                return False
        self._uva = (row_start, row_end)
        return True

    def disable_uva(self) -> None:
        """Release this table's page-locked range, if it holds one.

        Registration is per table, so a set that fails partway has to give back
        what it already took: those pages are unswappable, and the caller is
        about to fall back to the host path that does not want them.
        """
        state = getattr(self, "_uva", None)
        if state is None:
            return
        # `cudaHostUnregister` takes the base pointer alone, so the range's end
        # is not needed to give the pages back.
        row_start, _ = state
        rt = torch.cuda.cudart()
        for tensor, width in (
            (self._tensor, self.head_dim),
            (self._scale, None if self._scale is None else self._scale.shape[1]),
        ):
            if tensor is None:
                continue
            item = tensor.element_size()
            base = tensor.data_ptr() + row_start * width * item
            rt.cudaHostUnregister(base - (base % self._PAGE))
        self._uva = None

    def gather_into(
        self,
        ids: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        local_heads: int,
        total_heads: int,
    ) -> None:
        """Fill `out` ([tokens, local_heads, head_dim], device) from `ids`.

        `ids` is the FULL `[tokens, total_heads]` index matrix -- every rank sees
        every index and skips the ones outside its shard, so no index exchange is
        needed. Heads this rank does not own are left as zeros for the caller's
        all-gather. Requires `enable_uva`.
        """
        if getattr(self, "_uva", None) is None:
            raise RuntimeError("engram UVA lookup needs enable_uva() first")
        row_start, row_end = self._uva
        tokens = ids.shape[0]
        num_rows = tokens * local_heads
        if num_rows == 0:
            return
        if out.shape != (tokens, local_heads, self.head_dim):
            raise ValueError(
                f"engram UVA output is {tuple(out.shape)}, expected "
                f"{(tokens, local_heads, self.head_dim)}"
            )
        # Address the shard, not the table: the kernel subtracts `row_start`.
        weight = self._tensor[row_start:row_end]
        # The kernel reads the ue8m0 scale as a raw exponent byte, and Triton has
        # no pointer type for float8_e8m0fnu, so hand it the bytes.
        scales = (
            weight
            if self._scale is None
            else self._scale[row_start:row_end].view(torch.uint8)
        )
        num_sms = torch.cuda.get_device_properties(out.device).multi_processor_count
        block = 16
        grid = min(-(-num_rows // block), num_sms)
        _uva_lookup_kernel[(grid,)](
            weight,
            scales,
            ids,
            out,
            num_rows,
            row_start,
            row_end,
            ids.stride(0),
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            TOTAL_HEADS=total_heads,
            DIM=self.head_dim,
            QUANT_BLOCK=self.block_size or 1,
            BLOCK_R=block,
            HAS_SCALE=self._scale is not None,
        )

    @staticmethod
    def _gather_rows(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        if tensor.element_size() == 1:
            return tensor.view(torch.uint8)[index].view(tensor.dtype)
        return tensor[index]

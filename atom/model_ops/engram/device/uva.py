# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The UVA gather: this rank's hash heads, read from host memory by the device.

The page-locking, and the host gather this path replaces, stay in `..tables`:
that module is what the prefetch runtime imports, and it has to keep importing
on a machine with no Triton.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.engram.host_memory import shard_device_address


@triton.jit
def _uva_lookup_kernel(
    tables,
    ids,
    out,
    num_rows,
    ids_stride_l,
    ids_stride_t,
    out_stride_l,
    HEAD_START: tl.constexpr,
    LOCAL_HEADS: tl.constexpr,
    TOTAL_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    BLOCK_R: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    WEIGHT: tl.constexpr,
):
    """Gather this rank's hash heads from host tables over UVA, dequantize, store:
    layer ``program_id(1)``'s table, its ids and its rows' plane.

    ``tables`` is a row of int64s a layer (``UvaGather``): the weight's and the
    scales' addresses in page-locked HOST memory holding only this rank's
    shard, so a row is addressed by `index - vocab_start`, then vocab_start and
    vocab_end. A head this rank does not own writes zeros, which is what makes
    the all-gather that follows a plain concatenation. The ue8m0 scale byte IS
    an fp32 exponent field, so its decode is a shift.

    The grid is persistent and sized to the device rather than to the batch: the
    table dwarfs any TLB, so the win is in reusing warmed translations, not in
    one program per row.
    """
    layer = tl.program_id(1)
    row = tables + layer * 4
    weight = tl.load(row).to(tl.pointer_type(WEIGHT))
    scales = tl.load(row + 1).to(tl.pointer_type(tl.uint8))
    vocab_start = tl.load(row + 2)
    vocab_end = tl.load(row + 3)
    ids += layer * ids_stride_l
    out += layer * out_stride_l
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


_TL_DTYPES = {
    torch.float8_e4m3fn: tl.float8e4nv,
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}


class UvaGather:
    """The UVA gather of a run of host tables (one launch for all of them),
    bound once: each table's registered shard (``table.enable_uva()``) as a
    row of the device table the kernel reads. The tables share their row
    width, quantization and weight dtype; the shards are held here, so a
    table closed under a bound gather keeps its memory until the gather goes.
    """

    def __init__(self, tables, device) -> None:
        tables = list(tables)
        shards = [t.registered_shard() for t in tables]
        first = tables[0]
        if any(
            (t.head_dim, t.quantized, t.block_size, s[0].dtype)
            != (first.head_dim, first.quantized, first.block_size, shards[0][0].dtype)
            for t, s in zip(tables, shards)
        ):
            raise ValueError("engram UVA tables of one gather must share a layout")
        self.layers = len(tables)
        self.head_dim = first.head_dim
        self._shards = shards
        self.quant_block = first.block_size or 1
        self.quantized = first.quantized
        dtype = shards[0][0].dtype
        if dtype not in _TL_DTYPES:
            raise ValueError(
                f"engram UVA gather: table dtype {dtype} is not one of "
                f"{sorted(map(str, _TL_DTYPES))}"
            )
        self.weight_dtype = _TL_DTYPES[dtype]
        # the device's addresses of the page-locked shards, which the kernel
        # dereferences from this table
        self.rows = torch.tensor(
            [
                [shard_device_address(w), shard_device_address(s), start, end]
                for w, s, start, end in shards
            ],
            dtype=torch.int64,
            device=device,
        )
        self._sms = torch.cuda.get_device_properties(device).multi_processor_count

    def __call__(
        self,
        ids: torch.Tensor,
        out: torch.Tensor,
        *,
        head_start: int,
        local_heads: int,
        total_heads: int,
    ) -> None:
        """Fill `out` ([layers, tokens, local_heads, head_dim], device) from
        `ids` ([layers, tokens, total_heads]), a layer's from its table.

        `ids` holds the FULL index matrix of each layer -- every rank sees every
        index and skips the ones outside its shard, so no index exchange is
        needed. Heads this rank does not own are left as zeros for the
        caller's all-gather.
        """
        tokens = ids.shape[1]
        num_rows = tokens * local_heads
        if num_rows == 0:
            return
        want = (self.layers, tokens, local_heads, self.head_dim)
        if out.shape != want or ids.shape[0] != self.layers:
            raise ValueError(
                f"engram UVA output is {tuple(out.shape)} from ids "
                f"{tuple(ids.shape)}, expected {want}"
            )
        block = 16
        grid = min(-(-num_rows // block), self._sms)
        _uva_lookup_kernel[(grid, self.layers)](
            self.rows,
            ids,
            out,
            num_rows,
            ids.stride(0),
            ids.stride(1),
            out.stride(0),
            HEAD_START=head_start,
            LOCAL_HEADS=local_heads,
            TOTAL_HEADS=total_heads,
            DIM=self.head_dim,
            QUANT_BLOCK=self.quant_block,
            BLOCK_R=block,
            HAS_SCALE=self.quantized,
            WEIGHT=self.weight_dtype,
        )

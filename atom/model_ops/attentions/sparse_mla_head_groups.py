# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Independent head groups for small BF16 persistent sparse-prefill batches,
and the work-plan granularity shared by the builder, the DCP rebuild and the
sglang GLM-5.2 bridge."""

import torch
import triton
import triton.language as tl
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

MAX_HEAD_GROUP_ROWS = 32
# The one geometry the split serves: 128 BF16 query heads of a 512 + 64 wide
# latent, as two groups of 64, each row's region at most one top-k (2048).
HEADS = 128
GROUPS = 2
GROUP_HEADS = HEADS // GROUPS
KV_LORA_RANK = 512
Q_DIM = KV_LORA_RANK + 64
REGION = 2048
# A persistent attention kernel's work plan, one set a forward.
PERSISTENT_WORK_BUFFERS = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


def sparse_prefill_rows_whole(dtype_q, dtype_kv):
    """Whether a sparse-prefill work plan keeps each row in one work item. Such
    a plan has no split rows, so `mla_decode_fwd` is called with
    has_partials=False and allocates no split buffers: a split row would then
    write out of bounds. Every planner of these rows keeps them whole: the
    builder's dcp>1 "q" plans use the next power of two of the row width, head
    groups use REGION, and the rest (dcp=1 builder, DCP rebuild, the sglang
    GLM-5.2 bridge) take `sparse_prefill_kv_granularity`."""
    return dtype_q == dtype_kv == torch.bfloat16


def sparse_prefill_kv_granularity(dtype_q, dtype_kv, row_width, page_size=1):
    """kv_granularity of a sparse-prefill work plan. A BF16 row is planned
    whole: splitting its candidates by batch occupancy changes the reduction
    order, so a row's output would depend on what it is batched with.
    Runs per step, hence the bit trick over `triton.next_power_of_2`."""
    if sparse_prefill_rows_whole(dtype_q, dtype_kv):
        return max(1 << max(row_width - 1, 0).bit_length(), page_size, 16)
    return max(page_size, 16)


@triton.jit
def _repeat_sparse_csr_rows(
    indices,
    indptr,
    out_indices,
    out_indptr,
    qptr,
    last,
    rows,
    GROUPS: tl.constexpr,
    REGION: tl.constexpr,
):
    """Give each of the row's GROUPS head groups its own copy of the row's KV
    region, in order: out row `row * GROUPS + g` reads region copy g."""
    row = tl.program_id(0)
    begin = tl.load(indptr + row)
    end = tl.load(indptr + row + 1)
    length = end - begin
    offsets = tl.arange(0, REGION)
    values = tl.load(indices + begin + offsets, mask=offsets < length, other=0)
    for group in tl.static_range(GROUPS):
        tl.store(
            out_indices + begin * GROUPS + group * length + offsets,
            values,
            mask=offsets < length,
        )
        tl.store(out_indptr + row * GROUPS + group, begin * GROUPS + group * length)
        tl.store(qptr + row * GROUPS + group, row * GROUPS + group)
        tl.store(last + row * GROUPS + group, 1)
    if row == rows - 1:
        tl.store(out_indptr + rows * GROUPS, end * GROUPS)
        tl.store(qptr + rows * GROUPS, rows * GROUPS)


class SparsePrefillHeadGroups:
    """Reuse integer buffers while splitting 128 heads into two groups of 64.

    The caller supplies the DCP packer's nonempty CSR regions, including one
    dummy slot for a row with no owned KV. Each head group visits the complete
    region in its original order. The true owned counts stay at query-row
    granularity for the existing DCP merge to mask dummy rows.
    """

    def __init__(self, device):
        group_rows = MAX_HEAD_GROUP_ROWS * GROUPS
        self.indices = torch.empty(
            group_rows * REGION, dtype=torch.int32, device=device
        )
        self.indptr = torch.empty(group_rows + 1, dtype=torch.int32, device=device)
        self.qptr = torch.empty_like(self.indptr)
        self.last = torch.empty(group_rows, dtype=torch.int32, device=device)
        sizes = get_mla_metadata_info_v1(
            group_rows,
            1,
            GROUP_HEADS,
            torch.bfloat16,
            torch.bfloat16,
            is_sparse=True,
            fast_mode=True,
            max_split_per_batch=-1,
        )
        self.work = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in zip(PERSISTENT_WORK_BUFFERS, sizes)
        }

    @staticmethod
    def applies(
        q: torch.Tensor,
        output: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_granularity: int,
    ) -> bool:
        """Whether this step's sparse-prefill call has the geometry above.
        `kv_granularity` (the layer's row width) bounds each row's region."""
        return (
            0 < q.shape[0] <= MAX_HEAD_GROUP_ROWS
            and q.dtype == kv_cache.dtype == torch.bfloat16
            and q.shape[1:] == (HEADS, Q_DIM)
            and output.shape[1:] == (HEADS, KV_LORA_RANK)
            and q.is_contiguous()
            and kv_granularity == REGION
        )

    def prepare(self, q, output, indptr, indices):
        """The grouped `(q, output, qptr, kvptr, kv_indices, last_page_lens)`
        for `mla_decode_fwd`, with `self.work` rebuilt for them."""
        rows = q.shape[0]
        _repeat_sparse_csr_rows[(rows,)](
            indices,
            indptr,
            self.indices,
            self.indptr,
            self.qptr,
            self.last,
            rows,
            GROUPS=GROUPS,
            REGION=REGION,
            num_warps=4,
        )
        group_rows = rows * GROUPS
        qptr = self.qptr[: group_rows + 1]
        kvptr = self.indptr[: group_rows + 1]
        last = self.last[:group_rows]
        w = self.work
        get_mla_metadata_v1(
            qptr,
            kvptr,
            last,
            GROUP_HEADS,
            1,
            True,
            w["work_meta_data"],
            w["work_info_set"],
            w["work_indptr"],
            w["reduce_indptr"],
            w["reduce_final_map"],
            w["reduce_partial_map"],
            page_size=1,
            dtype_q=torch.bfloat16,
            dtype_kv=torch.bfloat16,
            kv_granularity=REGION,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            fast_mode=True,
            max_split_per_batch=-1,
        )
        return (
            q.view(group_rows, GROUP_HEADS, Q_DIM),
            output.view(group_rows, GROUP_HEADS, KV_LORA_RANK),
            qptr,
            kvptr,
            self.indices,
            last,
        )

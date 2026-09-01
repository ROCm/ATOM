"""GLM/DeepSeek MTP sparse-index sharing helpers.

The MTP model is decorated with ``support_torch_compile``. Its custom
dispatcher reuses the first compiled bytecode and therefore cannot use a
mutable Python ``skip_topk`` attribute to select a different graph later.
Keep the fresh-index branch on that compiled dispatcher, and run only the
reuse branch through the original eager forward. The existing DraftGraph stays
the full-indexer fallback for short contexts.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_packed_sparse_rows_kernel(
    source,
    source_indptr,
    slot_ids,
    temporary,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    source_row = tl.load(slot_ids + row)
    source_start = tl.load(source_indptr + source_row)
    source_end = tl.load(source_indptr + source_row + 1)
    valid = columns < tl.minimum(source_end - source_start, TOPK)
    values = tl.load(source + source_start + columns, mask=valid, other=0)
    tl.store(temporary + row * TOPK + columns, values, mask=valid)


@triton.jit
def _store_packed_sparse_rows_kernel(
    temporary,
    destination,
    destination_indptr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    destination_start = tl.load(destination_indptr + row)
    destination_end = tl.load(destination_indptr + row + 1)
    valid = columns < tl.minimum(destination_end - destination_start, TOPK)
    values = tl.load(temporary + row * TOPK + columns, mask=valid)
    tl.store(destination + destination_start + columns, values, mask=valid)


def _eager_model(model):
    """Return the model behind a level-1 ``torch.compile`` wrapper, if any."""
    return getattr(model, "_orig_mod", model)


def _resolve_sparse_attention(self_attn):
    frontend = getattr(self_attn, "mla_attn", None)
    for candidate in (frontend, getattr(frontend, "impl", None)):
        if candidate is not None and hasattr(candidate, "sparse_kv_indices_buffer"):
            return candidate
    return None


def _iter_sparse_mtp_attentions(mtp_predictor) -> Iterator[tuple[Any, Any]]:
    layers = getattr(mtp_predictor, "layers", {})
    for layer in layers.values():
        mtp_block = getattr(layer, "mtp_block", None)
        self_attn = getattr(mtp_block, "self_attn", None)
        if self_attn is None or getattr(self_attn, "indexer", None) is None:
            continue
        sparse_attn = _resolve_sparse_attention(self_attn)
        if hasattr(self_attn, "skip_topk") and sparse_attn is not None:
            yield self_attn, sparse_attn


def supports_mtp_index_share(mtp_predictor) -> bool:
    """Whether this predictor exposes an index-owning sparse MTP attention."""
    return next(_iter_sparse_mtp_attentions(mtp_predictor), None) is not None


def can_reuse_mtp_indices(context_lens, num_rows: int, topk: int) -> bool:
    """Whether sparse row lengths stay fixed while MTP advances the context."""
    return (
        num_rows > 0
        and len(context_lens) >= num_rows
        and all(int(context_len) >= topk for context_len in context_lens[:num_rows])
    )


def set_mtp_index_reuse(mtp_predictor, reuse: bool) -> None:
    """Set all index-owning MTP attentions to compute or reuse sparse indices."""
    for self_attn, _ in _iter_sparse_mtp_attentions(mtp_predictor):
        self_attn.skip_topk = reuse


def forward_with_fresh_mtp_indices(model, **kwargs):
    """Run the compiled MTP path that computes top-k, then restore reuse mode.

    This must be the first branch seen by ``support_torch_compile`` so its
    immutable bytecode contains the full indexer. High-concurrency chunked
    prefill depends on the compiled path; only reuse forwards bypass it.
    """
    eager_model = _eager_model(model)
    mtp_predictor = eager_model.model
    set_mtp_index_reuse(mtp_predictor, False)
    try:
        return model(**kwargs)
    finally:
        set_mtp_index_reuse(mtp_predictor, True)


def forward_with_reused_mtp_indices(model, **kwargs):
    """Run the explicit eager reuse branch outside the full-indexer DraftGraph."""
    eager_model = _eager_model(model)
    set_mtp_index_reuse(eager_model.model, True)
    return eager_model.forward(**kwargs)


def _compact_packed_sparse_rows(
    sparse_buffer: torch.Tensor,
    sparse_indptr: torch.Tensor,
    slot_ids: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Compact variable-length packed rows and return their new indptr."""
    num_rows = slot_ids.numel()
    if sparse_buffer.device.type == "cpu":
        segments = [
            sparse_buffer[int(sparse_indptr[row]) : int(sparse_indptr[row + 1])].clone()
            for row in slot_ids.tolist()
        ]
        destination_indptr = torch.zeros(
            num_rows + 1, dtype=torch.int32, device=sparse_buffer.device
        )
        for row, segment in enumerate(segments):
            destination_indptr[row + 1] = destination_indptr[row] + segment.numel()
            start = int(destination_indptr[row])
            sparse_buffer[start : start + segment.numel()].copy_(segment)
        return destination_indptr

    slot_ids = slot_ids.to(device=sparse_buffer.device, dtype=torch.long)
    starts = torch.index_select(sparse_indptr, 0, slot_ids)
    ends = torch.index_select(sparse_indptr, 0, slot_ids + 1)
    lengths = ends - starts
    destination_indptr = torch.empty(
        num_rows + 1, dtype=torch.int32, device=sparse_buffer.device
    )
    destination_indptr[0] = 0
    torch.cumsum(lengths, dim=0, out=destination_indptr[1:])

    temporary = torch.empty(
        (num_rows, topk), dtype=sparse_buffer.dtype, device=sparse_buffer.device
    )
    block_size = 256
    grid = (num_rows, triton.cdiv(topk, block_size))
    _gather_packed_sparse_rows_kernel[grid](
        sparse_buffer,
        sparse_indptr,
        slot_ids,
        temporary,
        TOPK=topk,
        BLOCK_SIZE=block_size,
    )
    _store_packed_sparse_rows_kernel[grid](
        temporary,
        sparse_buffer,
        destination_indptr,
        TOPK=topk,
        BLOCK_SIZE=block_size,
    )
    return destination_indptr


def compact_mtp_sparse_indices(
    model,
    slot_ids: torch.Tensor,
    sparse_kv_indptr: torch.Tensor,
    running_rows: int,
) -> None:
    """Move selected packed token rows to the front of each MTP sparse buffer."""
    if running_rows > slot_ids.numel():
        if slot_ids.numel() == 0:
            raise RuntimeError("Cannot pad empty MTP sparse-index rows")
        # DraftGraph pads every model input by repeating the last real row.
        # Sparse physical indices must use the identical row mapping; otherwise
        # padded MLA rows read uninitialized slots even though outputs are dropped.
        slot_ids = torch.cat(
            [
                slot_ids,
                slot_ids[-1:].expand(running_rows - slot_ids.numel()),
            ]
        )
    elif running_rows < slot_ids.numel():
        raise RuntimeError(
            f"MTP sparse-index rows exceed running batch: "
            f"selected={slot_ids.numel()}, running={running_rows}"
        )

    eager_model = _eager_model(model)
    seen_buffers: set[int] = set()
    for self_attn, sparse_attn in _iter_sparse_mtp_attentions(eager_model.model):
        sparse_buffer = sparse_attn.sparse_kv_indices_buffer
        if sparse_buffer is None or sparse_buffer.numel() == 0:
            continue

        topk = int(
            getattr(
                sparse_attn,
                "topk_tokens",
                getattr(self_attn.indexer, "topk_tokens", 0),
            )
        )
        if topk <= 0:
            raise RuntimeError(
                "Invalid MTP sparse-index buffer layout: "
                f"shape={tuple(sparse_buffer.shape)}, topk={topk}"
            )

        # IndexShare layers can reference the exact same global scratch tensor.
        # Compact a shared allocation once; applying slot_ids twice corrupts it.
        buffer_id = id(sparse_buffer)
        if buffer_id in seen_buffers:
            continue
        seen_buffers.add(buffer_id)

        source_indptr = sparse_kv_indptr
        dcp_indptr = getattr(sparse_attn, "dcp_sparse_kv_indptr_buffer", None)
        if (
            int(getattr(sparse_attn, "dcp_world_size", 1)) > 1
            and torch.is_tensor(dcp_indptr)
            and dcp_indptr.numel() > 0
        ):
            source_indptr = dcp_indptr

        destination_indptr = _compact_packed_sparse_rows(
            sparse_buffer,
            source_indptr,
            slot_ids,
            topk,
        )
        if source_indptr is dcp_indptr:
            dcp_indptr[: destination_indptr.numel()].copy_(destination_indptr)

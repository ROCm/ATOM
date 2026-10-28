# SPDX-License-Identifier: MIT
"""TP row ownership for large mHC seams, retaining native mHC arithmetic.

TP expert GEMMs consume replicated BF16 input. Full-head attention consumes
owner rows; exact index scoring exchanges selected ids. The TP expert partial
output is reduce-scattered, while residual and gates stay on owners.
"""

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group
from aiter.jit.utils.torch_guard import torch_compile_guard

MIN_ROWS = 4096


def _local_rows_fake(value, row_dim):
    shape = list(value.shape)
    tp = get_tp_group().world_size
    shape[row_dim] = (shape[row_dim] + tp - 1) // tp
    return value.new_empty(shape)


@torch_compile_guard(mutates_args=[], gen_fake=_local_rows_fake)
def v41_mhc_local_rows(value: torch.Tensor, row_dim: int) -> torch.Tensor:
    """Publish a fixed-capacity owner tensor, including the last rank's padding.

    The rank-dependent valid-row slice stays behind this boundary. An aligned
    compile example must not erase zero padding needed by a later ragged step.
    """
    group = get_tp_group()
    rows = value.shape[row_dim]
    count = (rows + group.world_size - 1) // group.world_size
    start = group.rank_in_group * count
    valid = min(count, max(0, rows - start))
    part = value.narrow(row_dim, min(start, rows), valid)
    shape = list(value.shape)
    shape[row_dim] = count - valid
    return torch.cat((part, value.new_zeros(shape)), dim=row_dim)


def eligible(rows, tp_size):
    return tp_size == 4 and rows >= MIN_ROWS


def _scatter_fake(value):
    return value.new_empty(
        (
            (value.shape[0] + get_tp_group().world_size - 1)
            // get_tp_group().world_size,
            *value.shape[1:],
        )
    )


@torch_compile_guard(mutates_args=[], gen_fake=_scatter_fake)
def v41_mhc_reduce_scatter(value: torch.Tensor) -> torch.Tensor:
    group = get_tp_group()
    output = _scatter_fake(value)
    pad = output.shape[0] * group.world_size - value.shape[0]
    if pad:
        value = torch.cat((value, value.new_zeros((pad, *value.shape[1:]))), dim=0)
    dist.reduce_scatter_tensor(output, value.contiguous(), group=group.device_group)
    return output


def _gather_fake(value):
    return value.new_empty(
        (value.shape[0] * get_tp_group().world_size, *value.shape[1:])
    )


@torch_compile_guard(mutates_args=[], gen_fake=_gather_fake)
def v41_mhc_gather_rows(value: torch.Tensor) -> torch.Tensor:
    group = get_tp_group()
    value = value.contiguous()
    # Norm rows are pure copies. Keep residuals, index tensors and unsupported
    # communicators on the original transport; custom AG itself has no gate.
    if value.ndim == 2 and value.shape[1] == 5120 and value.dtype == torch.bfloat16:
        communicator = getattr(group.device_communicator, "ca_comm", None)
        if communicator is not None and communicator.should_custom_ag(value):
            return communicator.custom_all_gather(value, dim=0)
    output = _gather_fake(value)
    dist.all_gather_into_tensor(output, value, group=group.device_group)
    return output


def localize(state):
    """Replicated state to equally padded owners; only the last owner has invalid rows."""
    from atom.model_ops.deepseek_v41.mhc import SinglePassHCState

    if state.row_sharded:
        return state
    rows = state.residual.shape[-3]
    values = []
    for value, dim in zip(state.fields(), (-3, -2, -2, -2, -3)):
        if value is None:
            values.append(None)
            continue
        values.append(v41_mhc_local_rows(value, dim))
    return SinglePassHCState(*values, row_sharded=True, original_rows=rows)


def replicate(state):
    """Settle on owners, then gather only residual and incoming pre-mix.

    This is charged at compiled-stage exits. No pending post/gates
    are lost: native settle applies them before their storage is released.
    """
    from atom.model_ops.deepseek_v41.mhc import SinglePassHCState

    if not state.row_sharded:
        return state
    settled = state.settle()
    *outer, rows, hc, hidden = settled.residual.shape
    residual = v41_mhc_gather_rows(settled.residual.reshape(rows, hc, hidden))
    pre = v41_mhc_gather_rows(settled.pre_mix.reshape(rows, hc))
    full = rows * get_tp_group().world_size
    original = settled.original_rows
    return SinglePassHCState(
        residual.view(*outer, full, hc, hidden).narrow(-3, 0, original),
        pre.view(*outer, full, hc).narrow(-2, 0, original),
    )


def gather_norm(normed, original_rows=None):
    *outer, rows, hidden = normed.shape
    value = v41_mhc_gather_rows(normed.reshape(rows, hidden)).view(
        *outer, rows * get_tp_group().world_size, hidden
    )
    return value if original_rows is None else value.narrow(-2, 0, original_rows)


def score_owned_rows(indexer, query, weights, cache, step):
    """Native exact scoring on owned rows, retaining packed global offsets."""
    from atom.model_ops.attentions.deepseek_v41.score_planner import score_layout
    from atom.model_ops.deepseek_v41.candidate_table import CandidateBlocks
    from atom.model_ops.deepseek_v41.paged_scoring import score_topk_quantized
    from atom.model_ops.deepseek_v41.score_workspace import PackedRows

    group = get_tp_group()
    width = weights.shape[0]
    start = group.rank_in_group * width
    count = min(width, max(0, step.width - start))
    query = tuple(x[:count] for x in query)
    weights = weights[:count]
    span = slice(start, start + count)
    spec = indexer.spec
    bounds = indexer._bounds(cache, step)
    candidates = bounds["candidates"]
    if candidates is not None:
        bounds["candidates"] = CandidateBlocks(
            candidates.ids[span],
            candidates.table[span],
            candidates.context[span],
            candidates.rows_per_block,
        )
    ragged = indexer._ragged(cache, step)
    packed = None
    if ragged is not None:
        ragged = ragged.band(start, count, step.width)
        full = score_layout(step, spec.ratio)
        if full is not None:
            bands = tuple(
                (max(first, start) - start, min(end, start + count) - start)
                for first, end in full.bands
                if first < start + count and end > start
            )
            packed = PackedRows(
                full.visible[span], full.offsets[span], full.block_offsets[span], bands
            )
    tiles = None if ragged is not None else cache.unit_tiles(step, spec.ratio)[span]
    selected, chosen = score_topk_quantized(
        query,
        weights,
        cache.index_units[spec.kv_owner],
        tiles,
        step.visible[spec.ratio][span],
        topk=indexer.topk,
        workspace=cache.workspace,
        weight_scale=indexer.weights_scale,
        ragged=ragged,
        packed=packed,
        **bounds,
    )

    def publish(value):
        if value is None:
            return None
        if count < width:
            value = torch.cat(
                (value, value.new_full((width - count, *value.shape[1:]), -1)), dim=0
            )
        return v41_mhc_gather_rows(value)[: step.width]

    indexer._publish(publish(selected), publish(chosen), step)

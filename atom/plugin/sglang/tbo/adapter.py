"""Adapt SGLang-owned TBO splits to ATOM ``UBatchSlice`` objects.

SGLang remains responsible for TBO eligibility and child-batch splitting.
This module validates the generated children, maps their real token ranges to
ATOM ubatch slices, and adapts SGLang padding/graph shapes for ATOM execution.
It never chooses a split point or executes the model.

Padding contract:
``ForwardBatch.prepare_mlp_sync_batch()`` chooses SGLang's DP ``MAX_LEN`` or
``SUM_LEN`` layout and applies attention TP/CP alignment before
``TboForwardBatchPreparer`` creates the children. A child range can therefore
contain parent-tail padding, while ``child.tbo_padded_len`` records the physical
execution shape required by SGLang metadata. This policy is independent of the
MoE A2A backend; MORI does not disable DP ``MAX_LEN`` padding.

ATOM must preserve both views of the child: trim slices to real tokens for TBO
eligibility, output concatenation, and child-local metadata, while using
SGLang's already-padded child ``input_ids`` and ``positions`` directly for model
execution. Child outputs are trimmed before concatenation; padded KV writes keep
SGLang's reserved dummy-slot semantics. Padding is physical shape only and must
never regain real-token output semantics.
"""

import copy
from dataclasses import dataclass

import torch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from atom.utils.tbo.ubatch_splitting import UBatchSlice


@dataclass(frozen=True)
class AdaptedSGLangUBatch:
    input_ids: torch.Tensor
    positions: torch.Tensor
    num_reqs: int
    num_tokens: int
    running_bs: int
    running_tokens: int | None


def normalize_child_forward_batches(
    child_forward_batches: list[ForwardBatch],
    ubatch_slices: list[UBatchSlice],
) -> list[ForwardBatch]:
    """Create child-local execution views without mutating SGLang batches.

    Native SGLang treats each child as a parent-subrange descriptor and its
    operation scheduler keeps the parent coordinate system while slicing and
    merging hidden states. ATOM instead calls the complete model once per child,
    so execution metadata must describe a standalone batch whose local token
    coordinates start at zero. Keep the parent range only for output ordering.

    This runs during CUDA Graph capture, so it must not read GPU scalars with
    ``item()`` or otherwise synchronize the device.
    """

    normalized_children = []
    # Request-boundary children retain parent-global extend offsets. Two-chunk
    # children overlap in request space and SGLang has already rebuilt child1's
    # offsets locally.
    request_slices_overlap = (
        ubatch_slices[0].request_slice.stop > ubatch_slices[1].request_slice.start
    )
    for child, ub_slice in zip(child_forward_batches, ubatch_slices):
        child_batch = copy.copy(child)
        token_start = ub_slice.token_slice.start
        token_stop = ub_slice.token_slice.stop
        token_count = token_stop - token_start

        extend_start_loc = child_batch.extend_start_loc
        if (
            token_start > 0
            and not request_slices_overlap
            and extend_start_loc is not None
            and extend_start_loc.numel() > 0
        ):
            child_batch.extend_start_loc = extend_start_loc - token_start

        # Preserve the logical parent mapping for output order, while exposing
        # child-local real-token counts to ATOM's standalone model call.
        child_batch.tbo_parent_token_range = (token_start, token_stop)
        child_batch.num_token_non_padded_cpu = token_count
        child_batch.global_num_tokens_cpu = [token_count]
        normalized_children.append(child_batch)

    return normalized_children


def prepare_sglang_ubatch(
    ub_slice: UBatchSlice,
    child_forward_batch: ForwardBatch,
    *,
    is_prefill: bool,
    full_running_bs: int,
    ubatch_idx: int,
    num_ubatches: int,
    ub_max_tokens_across_dp: tuple[int, int] | None,
) -> AdaptedSGLangUBatch:
    """Apply SGLang-specific adaptations on top of native ATOM TBO.

    ``ub_slice`` describes the logical real-token range. SGLang has already
    padded the child tensors to the physical shape required by DP MAX_LEN,
    attention TP/CP alignment, or a CUDA Graph bucket, so use those tensors
    directly instead of slicing the parent and recreating the same padding.
    The merged output still uses the logical token count. Prefill also uses the
    cross-DP maximum token count as running_tokens so every MORI rank allocates
    matching communication buffers.
    """

    ub_num_reqs = ub_slice.request_slice.stop - ub_slice.request_slice.start
    ub_num_tokens = ub_slice.token_slice.stop - ub_slice.token_slice.start
    child_padded_len = child_forward_batch.tbo_padded_len
    if child_padded_len is None:
        raise RuntimeError("SGLang TBO child is missing tbo_padded_len")

    ub_input_ids = child_forward_batch.input_ids
    ub_positions = child_forward_batch.positions
    input_len = ub_input_ids.shape[0]
    position_len = ub_positions.shape[0]
    if input_len != position_len or input_len != child_padded_len:
        raise RuntimeError(
            "SGLang TBO child tensor shape does not match its execution length: "
            f"input_ids={input_len}, positions={position_len}, "
            f"tbo_padded_len={child_padded_len}"
        )
    if input_len < ub_num_tokens:
        raise RuntimeError(
            "SGLang TBO child execution length is smaller than its real tokens: "
            f"execution={input_len}, real={ub_num_tokens}"
        )
    if ub_num_tokens <= 0:
        raise RuntimeError(
            f"SGLang ATOM TBO ubatch {ubatch_idx} has empty token slice "
            f"{ub_slice.token_slice}"
        )

    if is_prefill:
        running_bs = ub_num_reqs
    elif ubatch_idx < num_ubatches - 1:
        running_bs = full_running_bs // num_ubatches
    else:
        running_bs = full_running_bs - (full_running_bs // num_ubatches) * (
            num_ubatches - 1
        )

    ub_running_tokens = child_padded_len
    if (
        is_prefill
        and ub_max_tokens_across_dp is not None
        and len(ub_max_tokens_across_dp) == num_ubatches
    ):
        ub_running_tokens = max(
            ub_running_tokens, int(ub_max_tokens_across_dp[ubatch_idx])
        )

    return AdaptedSGLangUBatch(
        input_ids=ub_input_ids,
        positions=ub_positions,
        num_reqs=ub_num_reqs,
        num_tokens=ub_num_tokens,
        running_bs=running_bs,
        running_tokens=ub_running_tokens if is_prefill else None,
    )


def _compute_child_real_token_range(
    token_range: tuple[int, int],
    parent_num_real_tokens: int,
) -> tuple[int, int]:
    """Map a padded SGLang child range to its logical real-token range.

    Treating parent-tail padding as real would keep fake hidden states in the
    merged output and make child-local token metadata include synthetic tokens.
    The computed range also prevents an all-padding child from entering the
    paired TBO executor.
    """

    start, stop = token_range
    # This computes the logical length later used to trim each model output;
    # SGLang's child tensors remain padded and reach the model unchanged.
    real_stop = min(max(parent_num_real_tokens, start), stop)
    return start, real_stop


def _can_execute_tbo_on_local_rank(
    child_ranges: list[tuple[int, int]],
    parent_num_tokens: int,
) -> bool:
    """Return whether this rank has two token slices ATOM can execute.

    A local rank cannot enter the paired executor when either child has no
    real tokens (for example an idle rank, decode batch size 1, or a split in
    tail padding), when padding trim leaves a gap/overlap between the children,
    or when the resulting ranges exceed the parent input.
    """

    (c0_start, c0_stop), (c1_start, c1_stop) = child_ranges
    # The executor concatenates child outputs directly, so a gap would drop
    # tokens and an overlap would duplicate them.
    if c0_stop != c1_start:
        return False
    # Idle ranks, decode batch size 1, and splits in tail padding can leave
    # one child without real tokens; the paired executor requires both.
    if c0_stop <= c0_start or c1_stop <= c1_start:
        return False
    # A stale real-token count must not create an out-of-bounds parent slice.
    return c1_stop <= parent_num_tokens


def adapt_sglang_tbo_ubatch_slices(
    forward_batch: ForwardBatch,
) -> list[UBatchSlice] | None:
    """Adapt SGLang TBO children to ATOM UBatchSlice.

    Request-boundary TBO has non-overlapping request slices. SGLang two-chunk
    TBO splits one long extend request across both children, so the split
    request appears in both child request ranges while token ranges stay
    contiguous. Return ``None`` when the children cannot be represented safely.
    """

    children = forward_batch.tbo_children
    if children is None or len(children) != 2:
        return None

    raw_child_ranges = [child.tbo_parent_token_range for child in children]
    parent_num_real_tokens = forward_batch.num_token_non_padded_cpu
    if parent_num_real_tokens is None:
        raise RuntimeError(
            "SGLang TBO parent is missing the authoritative "
            "num_token_non_padded_cpu value"
        )
    child_ranges = [
        _compute_child_real_token_range(token_range, parent_num_real_tokens)
        for token_range in raw_child_ranges
    ]
    if not _can_execute_tbo_on_local_rank(
        child_ranges, parent_num_tokens=len(forward_batch.input_ids)
    ):
        return None

    (c0_start, c0_stop), (c1_start, c1_stop) = child_ranges
    parent_batch_size = forward_batch.batch_size
    child0_req_stop = children[0].batch_size
    child1_req_start = parent_batch_size - children[1].batch_size

    return [
        UBatchSlice(
            request_slice=slice(0, child0_req_stop),
            token_slice=slice(c0_start, c0_stop),
        ),
        UBatchSlice(
            request_slice=slice(child1_req_start, parent_batch_size),
            token_slice=slice(c1_start, c1_stop),
        ),
    ]

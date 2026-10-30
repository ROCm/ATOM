"""Pure admission and explicit raw-launch argument ownership for C1 q4."""

from typing import NamedTuple

ROWS, HIDDEN, INTERMEDIATE, EXPERTS, TOPK, BM = 4, 2560, 640, 513, 11, 16
CAPACITY = ROWS * TOPK * BM


def step_supported(context, rows):
    """Choose the full-row path before the layer loop mutates any state."""
    if rows != ROWS or context.context is None or context.context.is_prefill:
        return False
    if context.ubatch_slices is not None:
        return False
    metadata = getattr(context, "attn_metadata", None)
    gdn = getattr(metadata, "gdn_metadata", None)
    return (
        gdn is not None
        and gdn.num_prefills == 0
        and gdn.num_decodes == 0
        and gdn.num_spec_decodes == 1
        and gdn.num_actual_tokens == ROWS
        and bool(getattr(gdn, "replayssm", False))
        and getattr(gdn, "replayssm_max_query_len", 0) == ROWS
    )


class G1Static(NamedTuple):
    a_quant: int
    a_scale: int
    weight: int
    weight_scale: int
    expert_ids: int
    valid: int
    m_indices: int
    rows: int
    grid: int
    intermediate: int
    intermediate_scale: int

    def pack(self, mixed_pointer, stream):
        # Native launch_gemm1: static prefix, BF16 hidden, no bias, stream.
        return (*self, mixed_pointer, 0, stream)


class G2Static(NamedTuple):
    intermediate: int
    intermediate_scale: int
    weight: int
    weight_scale: int
    expert_ids: int
    valid: int
    sorted_ids: int
    sorted_weights: int
    unused_bias: int
    rows: int
    max_m_blocks: int
    grid_blocks: int
    intermediate_width: int
    hidden_width: int
    output: int
    unused_output_scale: int

    def pack(self, stream):
        return (*self, stream)

"""RTP-only Qwen3.8 PLE snapshots at cacheable prefix boundaries."""

import torch
import triton
import triton.language as tl

from atom.model_ops.qwen4_exp.ple_layer import Qwen4ExpPLELayer


def bind_ple_checkpoint_hooks(model: torch.nn.Module) -> None:
    """Bind checkpoint writes only on RTP model instances."""
    for module in model.modules():
        if isinstance(module, Qwen4ExpPLELayer):
            module._state_checkpoint_writer = write_ple_checkpoints


@triton.jit
def _write_ple_checkpoints_kernel(
    ids,
    normalized,
    ngram,
    conv,
    starts,
    conv_rows,
    conv_slots,
    conv_offsets,
    ngram_rows,
    ngram_slots,
    ngram_offsets,
    ids_stride,
    norm_stride_token,
    norm_stride_channel,
    ngram_stride_slot,
    ngram_stride_pos,
    conv_stride_slot,
    conv_stride_channel,
    conv_stride_pos,
    CHANNELS: tl.constexpr,
    NGRAM: tl.constexpr,
    HISTORY: tl.constexpr,
    N_CONV: tl.constexpr,
    N_NGRAM: tl.constexpr,
    BLOCK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    target = tl.program_id(0)
    channel_block = tl.program_id(1)
    if channel_block == 0 and target < N_NGRAM:
        ngram_row = tl.load(ngram_rows + target)
        ngram_slot = tl.load(ngram_slots + target)
        ngram_end = tl.load(starts + ngram_row) + tl.load(ngram_offsets + target)
        positions = tl.arange(0, WINDOW)
        tokens = tl.load(
            ids + (ngram_end - NGRAM + positions) * ids_stride, positions < NGRAM, 0
        )
        tl.store(
            ngram + ngram_slot * ngram_stride_slot + positions * ngram_stride_pos,
            tokens,
            positions < NGRAM,
        )
    if target < N_CONV:
        conv_row = tl.load(conv_rows + target)
        conv_slot = tl.load(conv_slots + target)
        conv_end = tl.load(starts + conv_row) + tl.load(conv_offsets + target)
        channels = channel_block * BLOCK + tl.arange(0, BLOCK)
        history = tl.arange(0, WINDOW)
        values = tl.load(
            normalized
            + (conv_end - HISTORY + history[None, :]) * norm_stride_token
            + channels[:, None] * norm_stride_channel,
            (channels[:, None] < CHANNELS) & (history[None, :] < HISTORY),
            0,
        )
        tl.store(
            conv
            + conv_slot * conv_stride_slot
            + channels[:, None] * conv_stride_channel
            + history[None, :] * conv_stride_pos,
            values,
            (channels[:, None] < CHANNELS) & (history[None, :] < HISTORY),
        )


def write_ple_checkpoints(
    input_ids: torch.Tensor,
    normalized: torch.Tensor,
    metadata,
    history_width: int,
    conv_state_len: int,
) -> None:
    """Persist PLE n-gram and convolution windows for every GDN checkpoint."""
    targets = getattr(metadata, "rtp_checkpoint_targets", None)
    if targets is None:
        return
    conv_targets = targets["conv"]
    ngram_targets = targets["ngram"]
    conv_count = 0 if conv_targets is None else conv_targets["rows"].numel()
    ngram_count = 0 if ngram_targets is None else ngram_targets["rows"].numel()
    if not max(conv_count, ngram_count):
        return
    if conv_targets is None:
        conv_targets = ngram_targets
    if ngram_targets is None:
        ngram_targets = conv_targets
    ngram = metadata.ngram_state
    conv = metadata.conv_state
    width = int(history_width)
    history = int(conv_state_len)
    # MTP capacity has extra candidate slots; prefix cache is prefill-only.
    if metadata.num_accepted_tokens is not None:
        raise ValueError("PLE prefix checkpoints require non-speculative prefill")
    if normalized.ndim != 2 or normalized.shape[1] != conv.shape[1]:
        raise ValueError("PLE checkpoint convolution width mismatch")
    if width <= 0 or history <= 0 or max(width, history) > 64:
        raise ValueError("PLE checkpoint history must fit one prefix block")
    if ngram.shape[1] < width or conv.shape[2] < history:
        raise ValueError("PLE checkpoint pool is narrower than model history")
    _write_ple_checkpoints_kernel[
        (max(conv_count, ngram_count), triton.cdiv(normalized.shape[1], 32))
    ](
        input_ids.reshape(-1),
        normalized,
        ngram,
        conv,
        metadata.query_start_loc,
        conv_targets["rows"],
        conv_targets["slots"],
        conv_targets["offs"],
        ngram_targets["rows"],
        ngram_targets["slots"],
        ngram_targets["offs"],
        input_ids.stride(0),
        normalized.stride(0),
        normalized.stride(1),
        ngram.stride(0),
        ngram.stride(1),
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        normalized.shape[1],
        width,
        history,
        conv_count,
        ngram_count,
        32,
        triton.next_power_of_2(max(width, history)),
    )

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
    rows,
    slots,
    offsets,
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
    BLOCK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    target = tl.program_id(0)
    channel_block = tl.program_id(1)
    row = tl.load(rows + target)
    slot = tl.load(slots + target)
    end = tl.load(starts + row) + tl.load(offsets + target)
    if channel_block == 0:
        positions = tl.arange(0, WINDOW)
        tokens = tl.load(
            ids + (end - NGRAM + positions) * ids_stride, positions < NGRAM, 0
        )
        tl.store(
            ngram + slot * ngram_stride_slot + positions * ngram_stride_pos,
            tokens,
            positions < NGRAM,
        )
    channels = channel_block * BLOCK + tl.arange(0, BLOCK)
    history = tl.arange(0, WINDOW)
    values = tl.load(
        normalized
        + (end - HISTORY + history[None, :]) * norm_stride_token
        + channels[:, None] * norm_stride_channel,
        (channels[:, None] < CHANNELS) & (history[None, :] < HISTORY),
        0,
    )
    tl.store(
        conv
        + slot * conv_stride_slot
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
        (targets["rows"].numel(), triton.cdiv(normalized.shape[1], 32))
    ](
        input_ids.reshape(-1),
        normalized,
        ngram,
        conv,
        metadata.query_start_loc,
        targets["rows"],
        targets["slots"],
        targets["offs"],
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
        32,
        triton.next_power_of_2(max(width, history)),
    )

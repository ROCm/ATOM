"""RTP-only GDN prefix checkpoints and native FlyDSL snapshot handling."""

from __future__ import annotations

from contextvars import ContextVar

import torch
import triton
import triton.language as tl

_NATIVE_VK_SNAPSHOT: ContextVar[bool] = ContextVar(
    "rtp_native_vk_snapshot", default=False
)


def install_rtp_native_prefill_snapshot() -> None:
    """Keep AITER's native VK intermediate only for RTP prefix checkpointing."""
    from atom.model_ops.fla_ops import gdn_flydsl
    from atom.plugin.prepare import is_rtpllm

    original = gdn_flydsl.prefill
    if getattr(original, "_rtp_native_snapshot", False):
        return

    def prefill(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        metadata,
        keep_intermediate_states=False,
        *,
        state_indices=None,
        has_initial_state=None,
    ):
        if not is_rtpllm() or not keep_intermediate_states:
            _NATIVE_VK_SNAPSHOT.set(False)
            return original(
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                cu_seqlens,
                metadata,
                keep_intermediate_states,
                state_indices=state_indices,
                has_initial_state=has_initial_state,
            )
        if not gdn_flydsl.prefill_supported(q, k, v, g, beta, metadata):
            raise ValueError("Unsupported AITER FlyDSL prefill shape/metadata")
        gdn_flydsl._log_dispatch("prefill K1-K5 (ATOM K6)")
        schedule, chunks = metadata
        q, k = (
            gdn_flydsl.l2norm_fwd(q.contiguous()),
            gdn_flydsl.l2norm_fwd(k.contiguous()),
        )
        w, u, gc = gdn_flydsl.ops()[1](
            k=k,
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            cu_seqlens=cu_seqlens,
            BT=64,
            use_exp2=True,
            prefill_metadata=schedule,
        )
        h0 = gdn_flydsl.prepare_prefill_state(
            initial_state, state_indices, has_initial_state
        )
        h, vn, ht = gdn_flydsl.ops()[2](
            k=k,
            w=w,
            u=u,
            g=gc,
            initial_state=h0,
            output_final_state=True,
            chunk_size=64,
            cu_seqlens=cu_seqlens,
            state_dtype=torch.float32,
            snapshot_dtype=torch.bfloat16,
            use_exp2=True,
            g_head_major=True,
            bf16_convert_trunc=False,
            prefill_metadata=schedule,
        )
        output = gdn_flydsl.chunk_fwd_o(
            q,
            k,
            vn,
            h,
            gc,
            cu_seqlens=cu_seqlens,
            head_major_vk=True,
            chunk_indices=chunks,
        )
        _NATIVE_VK_SNAPSHOT.set(True)
        return output, ht.transpose(-1, -2), h

    prefill._rtp_native_snapshot = True
    gdn_flydsl.prefill = prefill


@triton.jit
def _copy_rtp_checkpoints(
    h,
    state,
    x,
    conv,
    rows,
    slots,
    offsets,
    is_end,
    runtime_slots,
    chunk_offsets,
    cu_seqlens,
    stride_h,
    stride_state,
    stride_x,
    stride_conv_slot,
    stride_conv_channel,
    stride_conv_position,
    STATE_ELEMENTS: tl.constexpr,
    D: tl.constexpr,
    STATE_LEN: tl.constexpr,
    CHUNK: tl.constexpr,
    NBLK_STATE: tl.constexpr,
    BLOCK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    STATE_VK: tl.constexpr,
    H_VK: tl.constexpr,
):
    target = tl.program_id(0)
    block = tl.program_id(1)
    row = tl.load(rows + target).to(tl.int64)
    slot = tl.load(slots + target).to(tl.int64)
    offset = tl.load(offsets + target).to(tl.int64)

    if block < NBLK_STATE:
        idx = block * BLOCK + tl.arange(0, BLOCK)
        mask = idx < STATE_ELEMENTS
        if STATE_VK == H_VK:
            src_idx = idx
            dst_idx = idx
        else:
            vk_idx = idx // (K * V) * (K * V) + (idx % V) * K + (idx % (K * V)) // V
            src_idx = vk_idx if H_VK else idx
            dst_idx = vk_idx if STATE_VK else idx
        if tl.load(is_end + target) != 0:
            source_slot = tl.load(runtime_slots + target).to(tl.int64)
            state_value = tl.load(
                state + source_slot * stride_state + dst_idx, mask=mask, other=0
            )
        else:
            source_chunk = tl.load(chunk_offsets + row).to(tl.int64) + offset // CHUNK
            state_value = tl.load(
                h + source_chunk * stride_h + src_idx, mask=mask, other=0
            ).to(state.dtype.element_ty)
        tl.store(state + slot * stride_state + dst_idx, state_value, mask=mask)
    else:
        end = tl.load(cu_seqlens + row).to(tl.int64) + offset
        channel = (block - NBLK_STATE) * BLOCK + tl.arange(0, BLOCK)
        mask = channel < D
        for pos in tl.static_range(STATE_LEN):
            conv_value = tl.load(
                x + (end - STATE_LEN + pos) * stride_x + channel,
                mask=mask,
                other=0,
            )
            tl.store(
                conv
                + slot * stride_conv_slot
                + channel * stride_conv_channel
                + pos * stride_conv_position,
                conv_value.to(conv.dtype.element_ty),
                mask=mask,
            )


def write_rtp_gdn_checkpoints(
    h: torch.Tensor,
    state: torch.Tensor,
    x: torch.Tensor,
    conv: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    offsets: torch.Tensor,
    is_end: torch.Tensor,
    runtime_slots: torch.Tensor,
    chunk_offsets: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    conv_state_len: int,
    h_vk: bool = False,
) -> None:
    """Write both state and conv checkpoints without assuming pool slot stride."""
    if not rows.numel():
        return
    state_elements = state.shape[1] * state.shape[2] * state.shape[3]
    state_vk = state.stride(-2) == 1
    expected = (state.shape[-1], state.shape[-2]) if h_vk else state.shape[-2:]
    if h.ndim != 5 or h.shape[2] != state.shape[1] or h.shape[-2:] != expected:
        raise ValueError("GDN checkpoint snapshot and state shapes differ")
    if h.stride(-1) != 1 or h.stride(-2) != h.shape[-1]:
        raise ValueError("GDN checkpoint snapshot must be physically contiguous")
    if h.stride(2) != h.shape[-2] * h.shape[-1]:
        raise ValueError("GDN checkpoint snapshot heads must be contiguous")
    if state.stride(0) < state_elements:
        raise ValueError("GDN state slot is smaller than its logical state")
    if state_vk and state.stride(1) != state.shape[-2] * state.shape[-1]:
        raise ValueError("GDN state heads must be physically contiguous")
    if not 0 < conv_state_len <= conv.shape[2]:
        raise ValueError("GDN convolution window is wider than the pool")
    if x.stride(-1) != 1:
        raise ValueError("GDN convolution input feature dimension must be contiguous")
    block = 4096 if state_vk == h_vk else 1024
    state_blocks = triton.cdiv(state_elements, block)
    _copy_rtp_checkpoints[
        (rows.numel(), state_blocks + triton.cdiv(conv.shape[1], block))
    ](
        h,
        state,
        x,
        conv,
        rows,
        slots,
        offsets,
        is_end,
        runtime_slots,
        chunk_offsets,
        cu_seqlens,
        h.stride(1),
        state.stride(0),
        x.stride(0),
        conv.stride(0),
        conv.stride(1),
        conv.stride(2),
        STATE_ELEMENTS=state_elements,
        D=conv.shape[1],
        STATE_LEN=conv_state_len,
        CHUNK=chunk_size,
        NBLK_STATE=state_blocks,
        BLOCK=block,
        K=state.shape[-2],
        V=state.shape[-1],
        STATE_VK=state_vk,
        H_VK=h_vk,
    )


def install_rtp_gdn_checkpoint_writer() -> None:
    """Install once in an RTP worker; other frameworks keep the shared writer."""
    from atom.model_ops.fla_ops import state_checkpoint
    from atom.plugin.prepare import is_rtpllm

    original = state_checkpoint.write_state_checkpoints
    if getattr(original, "_rtp_gdn_checkpoint_writer", False):
        return

    def write_state_checkpoints(
        h,
        state,
        x,
        conv,
        rows,
        slots,
        offsets,
        is_end,
        runtime_slots,
        chunk_offsets,
        cu_seqlens,
        chunk_size,
        conv_state_len,
        h_vk=False,
    ):
        native_vk = _NATIVE_VK_SNAPSHOT.get()
        _NATIVE_VK_SNAPSHOT.set(False)
        if not is_rtpllm():
            return original(
                h,
                state,
                x,
                conv,
                rows,
                slots,
                offsets,
                is_end,
                runtime_slots,
                chunk_offsets,
                cu_seqlens,
                chunk_size,
                conv_state_len,
            )
        return write_rtp_gdn_checkpoints(
            h,
            state,
            x,
            conv,
            rows,
            slots,
            offsets,
            is_end,
            runtime_slots,
            chunk_offsets,
            cu_seqlens,
            chunk_size,
            conv_state_len,
            h_vk=native_vk or h_vk,
        )

    write_state_checkpoints._rtp_gdn_checkpoint_writer = True
    state_checkpoint.write_state_checkpoints = write_state_checkpoints

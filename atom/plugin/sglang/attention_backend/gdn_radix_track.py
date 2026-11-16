# SPDX-License-Identifier: MIT
"""SGLang mamba radix cache extend snapshots for one GDN layer, in one launch.

SGLang's own GDN backend fills the track slots after an extend with eager ops:
the conv window gather plus its ``index_put``, then ``h[src]``, the cast to the
pool dtype and its ``index_put`` for unaligned rows, then a slot-to-slot copy
for aligned rows (``HybridLinearAttnBackend._track_mamba_state_extend``). That
is five to seven small launches per GDN layer per prefill.

This kernel performs the same copies from the same SGLang track metadata in
one launch. It addresses every tensor by its real strides, so the pool views
the plugin hands over (a transposed conv pool, the VK view of the SSM pool, a
column slice of the fused projection) are written byte for byte where the
eager ``index_put`` would write them. The SSM value is read from ``h`` in its
dtype and cast to the pool dtype, exactly as ``h[src].to(pool.dtype)`` does.

Grid axis 0 is the target list ``[conv rows | h rows | final rows]``; axis 1
covers the feature dim (conv) or the flattened ``H*K*V`` state (SSM).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _radix_track_snapshot_kernel(
    x,  # [T, D] conv input (pre-conv mixed_qkv), any row stride
    conv_idx,  # [N_CONV, STATE_LEN] token index of each window position
    conv_slots,  # [N_CONV] destination slot of each conv window
    conv_dst,  # conv pool, logical [slot, D, >= STATE_LEN]
    h,  # [NT, H, K, V] per-chunk states (logical layout)
    h_src,  # [N_H] chunk index of each unaligned target
    h_slots,  # [N_H] destination slot of each unaligned target
    final_src,  # [N_F] source slot of each aligned target
    final_slots,  # [N_F] destination slot of each aligned target
    ssm,  # SSM pool, logical [slot, H, K, V]
    stride_x_t,
    stride_cd_s,
    stride_cd_d,
    stride_cd_l,
    stride_h_t,
    stride_h_h,
    stride_h_k,
    stride_h_v,
    stride_s_s,
    stride_s_h,
    stride_s_k,
    stride_s_v,
    N_CONV,
    N_H,
    D: tl.constexpr,
    STATE_LEN: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    HKV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_blk = tl.program_id(1)
    offs = i_blk * BLOCK + tl.arange(0, BLOCK)
    if i_t < N_CONV:
        dmask = offs < D
        slot = tl.load(conv_slots + i_t).to(tl.int64)
        for j in tl.static_range(STATE_LEN):
            tok = tl.load(conv_idx + i_t * STATE_LEN + j).to(tl.int64)
            cv = tl.load(x + tok * stride_x_t + offs, mask=dmask, other=0.0)
            tl.store(
                conv_dst + slot * stride_cd_s + offs * stride_cd_d + j * stride_cd_l,
                cv.to(conv_dst.dtype.element_ty),
                mask=dmask,
            )
    else:
        smask = offs < HKV
        i_h = offs // (K * V)
        i_k = (offs // V) % K
        i_v = offs % V
        dst_off = i_h * stride_s_h + i_k * stride_s_k + i_v * stride_s_v
        t = i_t - N_CONV
        if t < N_H:
            src = tl.load(h_src + t).to(tl.int64)
            slot = tl.load(h_slots + t).to(tl.int64)
            sv = tl.load(
                h
                + src * stride_h_t
                + i_h * stride_h_h
                + i_k * stride_h_k
                + i_v * stride_h_v,
                mask=smask,
                other=0.0,
            ).to(ssm.dtype.element_ty)
        else:
            src_slot = tl.load(final_src + (t - N_H)).to(tl.int64)
            slot = tl.load(final_slots + (t - N_H)).to(tl.int64)
            sv = tl.load(ssm + src_slot * stride_s_s + dst_off, mask=smask, other=0.0)
        tl.store(ssm + slot * stride_s_s + dst_off, sv, mask=smask)


def fused_track_metadata_supported(metadata) -> bool:
    """The metadata half of ``fused_track_supported``: no recompute rows and
    all four SSM index tensors present. The same for every layer of a forward."""
    recompute = getattr(metadata, "track_ssm_recompute_dst", None)
    if recompute is not None and recompute.numel() > 0:
        return False
    for name in (
        "track_ssm_h_src",
        "track_ssm_h_dst",
        "track_ssm_final_src",
        "track_ssm_final_dst",
    ):
        if getattr(metadata, name, None) is None:
            return False
    return True


def fused_track_supported(metadata, h: torch.Tensor | None) -> bool:
    """Whether ``write_radix_track_snapshots`` covers this metadata exactly.

    Recompute rows (Mamba2 only) are not a GDN path; they, metadata without the
    SSM index tensors, and an ``h`` that is not a 4-D chunk state after the
    batch squeeze stay on SGLang's eager helper.
    """
    if not fused_track_metadata_supported(metadata):
        return False
    if metadata.track_ssm_h_src.numel() == 0:
        return True
    return h is not None and h.squeeze(0).dim() == 4


class RadixTrackPlan:
    """The layer-invariant half of ``write_radix_track_snapshots``.

    The track rows, their slots and the grid's row count come from SGLang's
    forward metadata and are the same for every GDN layer of a forward, so the
    plugin builds this once per forward. ``write`` takes the layer's conv
    input, pools and chunk states.
    """

    __slots__ = (
        "conv_idx",
        "conv_slots",
        "final_slots",
        "final_src",
        "h_slots",
        "h_src",
        "n_conv",
        "n_f",
        "n_h",
        "state_len",
    )

    def __init__(
        self,
        conv_idx: torch.Tensor | None,
        conv_slots: torch.Tensor | None,
        h_src: torch.Tensor,
        h_slots: torch.Tensor,
        final_src: torch.Tensor,
        final_slots: torch.Tensor,
    ) -> None:
        n_conv = 0 if conv_idx is None else conv_idx.shape[0]
        if n_conv:
            state_len = conv_idx.shape[1]
            conv_idx = conv_idx.contiguous()
        else:
            state_len = 1
            conv_idx = conv_slots = h_src
        self.conv_idx = conv_idx
        self.conv_slots = conv_slots
        self.h_src = h_src
        self.h_slots = h_slots
        self.final_src = final_src
        self.final_slots = final_slots
        self.n_conv = n_conv
        self.n_h = h_src.numel()
        self.n_f = final_src.numel()
        self.state_len = state_len

    def write(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        h: torch.Tensor | None,
        ssm_state: torch.Tensor,
    ) -> None:
        n_conv, n_h, n_f = self.n_conv, self.n_h, self.n_f
        rows = n_conv + n_h + n_f
        if rows == 0:
            return
        d = conv_state.shape[1]
        if n_conv:
            assert (
                conv_state.shape[2] == self.state_len
            ), "conv window must fill the pool"
            assert x.stride(-1) == 1, "conv input must be contiguous along features"
        _, n_heads, k, v = ssm_state.shape
        hkv = n_heads * k * v
        if n_h:
            h = h.squeeze(0)
            assert tuple(h.shape[1:]) == (
                n_heads,
                k,
                v,
            ), "h and pool state shapes differ"
        else:
            h = ssm_state
        nblk = max(
            triton.cdiv(d, _BLOCK) if n_conv else 0,
            triton.cdiv(hkv, _BLOCK) if n_h + n_f else 0,
        )
        cs = conv_state.stride()
        _radix_track_snapshot_kernel[(rows, nblk)](
            x,
            self.conv_idx,
            self.conv_slots,
            conv_state,
            h,
            self.h_src,
            self.h_slots,
            self.final_src,
            self.final_slots,
            ssm_state,
            x.stride(0),
            cs[0],
            cs[1],
            cs[2],
            *h.stride(),
            *ssm_state.stride(),
            n_conv,
            n_h,
            D=d,
            STATE_LEN=self.state_len,
            K=k,
            V=v,
            HKV=hkv,
            BLOCK=_BLOCK,
        )


def write_radix_track_snapshots(
    x: torch.Tensor,
    conv_idx: torch.Tensor | None,
    conv_slots: torch.Tensor | None,
    conv_state: torch.Tensor,
    h: torch.Tensor | None,
    h_src: torch.Tensor,
    h_slots: torch.Tensor,
    final_src: torch.Tensor,
    final_slots: torch.Tensor,
    ssm_state: torch.Tensor,
) -> None:
    """Equivalent of, in one launch::

        conv_state[conv_slots] = x[conv_idx].transpose(1, 2)
        ssm_state[h_slots] = h.squeeze(0)[h_src].to(ssm_state.dtype)
        ssm_state[final_slots] = ssm_state[final_src]

    ``conv_idx`` is SGLang's ``track_conv_indices`` (already clamped to the
    batch), so the window positions are taken as given. ``conv_state`` must be
    exactly ``STATE_LEN`` wide, as the eager assignment requires.
    """
    RadixTrackPlan(conv_idx, conv_slots, h_src, h_slots, final_src, final_slots).write(
        x, conv_state, h, ssm_state
    )

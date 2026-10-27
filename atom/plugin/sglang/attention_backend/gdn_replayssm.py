# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang TARGET_VERIFY using Native ReplaySSM.

Native MTP verify does not snapshot a full SSM state per draft token. It
appends (k, u, g) records and advances ``write_pos`` by the accepted count
after the step. SGLang's later ``fused_mamba_state_scatter_with_mask`` would
overwrite that checkpoint with a per-step snapshot, so once this path is
engaged the commit hook scatters only the conv window.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from atom.model_ops.fla_ops.replayssm import (
    flush_threshold_ok,
    replayssm_commit,
)
from atom.utils import envs

logger = logging.getLogger(__name__)

_LOGGED = False


class ReplaySSMRuntime:
    """One record ring per GDN layer, one cursor shared by every layer."""

    def __init__(self) -> None:
        self.write_pos: torch.Tensor | None = None
        self.bufs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        self.cache_len = 0
        self.max_query_len = 0
        self.engaged = False

    def reset(self) -> None:
        self.write_pos = None
        self.bufs.clear()
        self.cache_len = 0
        self.max_query_len = 0
        self.engaged = False


_RUNTIME = ReplaySSMRuntime()


def replay_runtime() -> ReplaySSMRuntime:
    return _RUNTIME


def reset_replay_runtime_for_tests() -> None:
    global _LOGGED
    _RUNTIME.reset()
    _LOGGED = False


def replayssm_enabled() -> bool:
    """Match Native: on for speculative serving unless the env forces it off.

    ``ATOM_ENABLE_REPLAYSSM=0`` keeps the per-step snapshot path.
    ``ATOM_ENABLE_REPLAYSSM=1`` forces the ring even without server args
    (tests). Unset follows whether this process is an MTP server.
    """
    override = envs.ATOM_ENABLE_REPLAYSSM
    if override is not None:
        return bool(override)
    return _server_is_speculative()


def _server_is_speculative() -> bool:
    try:
        from sglang.srt.runtime_context import get_server_args

        args = get_server_args()
    except Exception:  # noqa: BLE001 - unit tests have no server
        return False
    if args is None:
        return False
    if int(getattr(args, "speculative_num_steps", 0) or 0) > 0:
        return True
    algo = getattr(args, "speculative_algorithm", None)
    return algo is not None and str(algo).upper() not in ("", "NONE")


def _cache_len(max_query_len: int) -> int:
    requested = int(envs.ATOM_REPLAYSSM_CACHE_LEN)
    needed = 2 * max_query_len
    cache_len = max(requested, needed)
    if not flush_threshold_ok(cache_len, max_query_len):
        raise RuntimeError(
            f"ReplaySSM cache_len={cache_len} is below 2*verify_window=" f"{needed}."
        )
    return cache_len


def prepare_layer(
    layer_num: int,
    temporal: torch.Tensor,
    *,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    max_query_len: int,
    activation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate this layer's ring on the first verify. Addresses stay put."""
    global _LOGGED
    install_sglang_replayssm_commit()
    if torch.cuda.is_current_stream_capturing() and _RUNTIME.write_pos is None:
        raise RuntimeError(
            "ReplaySSM record buffers must be allocated on the eager "
            "TARGET_VERIFY warmup, before CUDA graph capture."
        )
    cache_len = _cache_len(max_query_len)
    if _RUNTIME.write_pos is None:
        slots = int(temporal.shape[0])
        _RUNTIME.write_pos = torch.zeros(
            slots, dtype=torch.int32, device=temporal.device
        )
        _RUNTIME.cache_len = cache_len
        _RUNTIME.max_query_len = max_query_len
        _RUNTIME.engaged = True
        if not _LOGGED:
            _LOGGED = True
            logger.info(
                "ReplaySSM enabled for SGLang GDN verify: cache_len=%d, "
                "route=%s, verify window=%d (1 state slot per request, "
                "no per-draft SSM snapshot).",
                cache_len,
                envs.ATOM_REPLAYSSM_ROUTE,
                max_query_len,
            )
    elif _RUNTIME.max_query_len != max_query_len or _RUNTIME.cache_len != cache_len:
        raise RuntimeError(
            "ReplaySSM verify window changed from "
            f"{_RUNTIME.max_query_len} to {max_query_len} after the ring "
            "was allocated."
        )
    bufs = _RUNTIME.bufs.get(layer_num)
    if bufs is None:
        slots = int(_RUNTIME.write_pos.shape[0])
        if int(temporal.shape[0]) != slots:
            raise RuntimeError(
                "ReplaySSM temporal slot count "
                f"{temporal.shape[0]} does not match the cursor length {slots}."
            )
        hv = num_v_heads
        device = temporal.device
        dtype = activation.dtype
        bufs = (
            torch.zeros(slots, hv, cache_len, head_k_dim, dtype=dtype, device=device),
            torch.zeros(slots, hv, cache_len, head_v_dim, dtype=dtype, device=device),
            torch.zeros(slots, hv, cache_len, dtype=torch.float32, device=device),
        )
        _RUNTIME.bufs[layer_num] = bufs
    return (*bufs, _RUNTIME.write_pos)


def note_full_state_write(slot_idx: torch.Tensor | None) -> None:
    """A prefill just rewrote the checkpoint. Drop uncommitted records.

    Commit runs after verify, not before the next kernel, so the Native
    prefill sentinel (-1) would discard the first verify's accepts. Parking
    the cursor at 0 is the equivalent: the next verify folds nothing.
    """
    write_pos = _RUNTIME.write_pos
    if write_pos is None or slot_idx is None or slot_idx.numel() == 0:
        return
    live = slot_idx[slot_idx >= 0].to(torch.int64)
    if live.numel() == 0:
        return
    write_pos.index_fill_(0, live, 0)


def install_sglang_replayssm_commit() -> None:
    """Advance the ring by the accepted count and skip the SSM snapshot scatter."""
    try:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            HybridLinearAttnBackend,
        )
    except Exception:
        logger.exception("ReplaySSM commit patch was not installed")
        return
    if getattr(HybridLinearAttnBackend, "_atom_replayssm_commit", False):
        return
    original = HybridLinearAttnBackend.update_mamba_state_after_mtp_verify

    def update_mamba_state_after_mtp_verify(
        self,
        last_correct_step_indices: torch.Tensor,
        mamba_track_indices: torch.Tensor | None,
        mamba_steps_to_track: torch.Tensor | None,
        model: Any,
        req_pool_indices: torch.Tensor | None = None,
    ) -> None:
        if not _RUNTIME.engaged or _RUNTIME.write_pos is None:
            return original(
                self,
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
                model,
                req_pool_indices,
            )
        _commit_accepted(
            self,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )

    HybridLinearAttnBackend.update_mamba_state_after_mtp_verify = (
        update_mamba_state_after_mtp_verify
    )
    HybridLinearAttnBackend._atom_replayssm_commit = True


def _commit_accepted(
    backend: Any,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None,
    mamba_steps_to_track: torch.Tensor | None,
) -> None:
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_conv_window_scatter_with_mask,
    )

    request_number = int(last_correct_step_indices.shape[0])
    if request_number == 0:
        return
    linear = backend.linear_attn_backend
    metadata = linear.forward_metadata
    slot_idx = metadata.mamba_cache_indices[:request_number]
    if slot_idx.dtype != torch.int32:
        slot_idx = slot_idx.to(torch.int32)
    # Chain verify: step index of the last accepted token, plus one, is the
    # accept count including the bonus. A full reject is -1 and commits 0.
    num_accepted = (last_correct_step_indices + 1).to(torch.int32)
    replayssm_commit(
        _RUNTIME.write_pos,
        slot_idx,
        num_accepted,
        _RUNTIME.max_query_len,
        _RUNTIME.cache_len,
    )
    mamba_caches = linear.req_to_token_pool.get_speculative_mamba2_params_all_layers()
    for conv_states, window in zip(
        mamba_caches.conv, mamba_caches.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states,
            window,
            slot_idx,
            last_correct_step_indices,
        )
    if mamba_track_indices is None:
        return
    for conv_states, window in zip(
        mamba_caches.conv, mamba_caches.intermediate_conv_window
    ):
        fused_conv_window_scatter_with_mask(
            conv_states,
            window,
            mamba_track_indices,
            mamba_steps_to_track,
        )

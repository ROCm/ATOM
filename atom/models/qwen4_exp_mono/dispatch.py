# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Install and route the supported Qwen3.8-Flash-Next decode steps."""

import logging

import torch
from torch import nn

from atom.models.qwen4_exp_mono.config import (
    config_refusal,
    enabled_widths,
    uses_mtp_target,
    uses_native_mxfp4_moe,
)
from atom.models.qwen4_exp_mono.shapes import MTP_TARGET_ROWS
from atom.mono.runtime.consensus import MonoUnsupported
from atom.mono.runtime.route import LazyRunner
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


def step_supported(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors,
    inputs_embeds,
    tp=1,
    *,
    mtp_target=False,
) -> bool:
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() != 1:
        return False
    # Native Atom publishes token IDs through an int32 CpuGpuBuffer. Keep the
    # route contract identical to that graph-stable input ABI.
    widths = MTP_TARGET_ROWS if mtp_target and tp == 1 else enabled_widths(tp)
    if input_ids.dtype != torch.int32 or input_ids.numel() not in widths:
        return False
    if intermediate_tensors is not None or inputs_embeds is not None:
        return False
    # Text-only Qwen can still carry mRoPE positions as [3, S] during FULL
    # graph capture; kernels consume the same S rows either way.
    rows = input_ids.numel()
    if not isinstance(positions, torch.Tensor) or positions.dtype != torch.int64:
        return False
    if tuple(positions.shape) not in ((rows,), (3, rows)):
        return False
    fwd = get_forward_context()
    if fwd.context is None or fwd.context.is_prefill or fwd.ubatch_slices is not None:
        return False
    metadata = getattr(fwd, "attn_metadata", None)
    if metadata is None or getattr(metadata, "image_mask", None) is not None:
        return False
    gdn = getattr(metadata, "gdn_metadata", None)
    if mtp_target:
        # Attention receives the original request boundaries and accepted
        # counts. q4 index uses stream-ordered publish/compress; Native retains
        # ReplaySSM and the conv device function's history update.
        # Graph padding is consumed by Native MoE's device valid-row count.
        return (
            tp == 1
            and gdn is not None
            and gdn.num_prefills == 0
            and gdn.num_decodes == 0
            and 0 < gdn.num_spec_decodes <= rows // 4
            and gdn.num_actual_tokens == 4 * gdn.num_spec_decodes
            and bool(getattr(gdn, "replayssm", False))
            and getattr(gdn, "replayssm_max_query_len", 0) == 4
        )
    return gdn is not None and (
        gdn.num_prefills,
        gdn.num_decodes,
        gdn.num_actual_tokens,
        gdn.num_spec_decodes,
        bool(getattr(gdn, "replayssm", False)),
    ) == (0, rows, rows, 0, False)


class MonoDecodeModel(nn.Module):
    """Run accepted widths on mono and preserve the original fallback."""

    def __init__(
        self,
        model: nn.Module,
        tp=1,
        max_model_len=8192,
        native_mxfp4_moe=False,
        mtp_target=False,
    ) -> None:
        super().__init__()
        self.model = model
        self.tp = tp
        self.mtp_target = mtp_target

        def create_runner():
            try:
                if native_mxfp4_moe:
                    from atom.models.qwen4_exp_mono.quark import QuarkMonoRunner

                    return QuarkMonoRunner(model, max_model_len)
                from atom.models.qwen4_exp_mono.runner import QwenMonoRunner
            except ImportError as error:
                raise MonoUnsupported(f"mono kernels unavailable: {error}") from error
            return QwenMonoRunner(model, max_model_len)

        self._mono = LazyRunner(create_runner, "Qwen mono")

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        if step_supported(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            self.tp,
            mtp_target=self.mtp_target,
        ):
            tokens = input_ids.numel()
            if self._mono.ready(tokens):
                return self._mono.runner.forward(input_ids, positions)
            if not self._mono.enabled and self._mono.runner is not None:
                self._mono.runner.close()
                self._mono.runner = None
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_mono_decode(model, atom_config, drafter, *, metadata_builder=None):
    """Wrap the exact supported native-Atom deployment, otherwise return model."""

    why = config_refusal(atom_config)
    if why is None and drafter is not None and not uses_mtp_target(atom_config):
        why = "speculative drafter"
    if why is not None:
        logger.info("Qwen mono off: %s", why)
        return model
    return MonoDecodeModel(
        model,
        atom_config.tensor_parallel_size,
        atom_config.max_model_len,
        native_mxfp4_moe=uses_native_mxfp4_moe(atom_config),
        mtp_target=uses_mtp_target(atom_config),
    )

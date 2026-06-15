# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 attention adaptations for SGLang plugin mode."""

from __future__ import annotations

import types
import os

import torch
from torch import nn


def patch_deepseek_v4_attention_for_sglang(attn: nn.Module) -> None:
    """Patch ATOM V4 attention for SGLang's padded prefill execution.

    SGLang can present padded prefill tensors (e.g. bucket width 256) while the
    ATOM V4 metadata built by the proxy bridge describes only real tokens.  Run
    native ATOM attention on the real token prefix, then pad the output back so
    the surrounding dense graph still sees the original tensor shape.
    """
    if hasattr(attn, "_sglang_v4_forward_impl"):
        return

    original_forward_impl = attn.forward_impl
    attn._sglang_v4_forward_impl = original_forward_impl

    def _forward_impl(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        from atom.utils.forward_context import AttnState, get_forward_context

        fc = get_forward_context()
        if fc.context.is_dummy_run:
            return self._sglang_v4_forward_impl(x, positions)

        attn_md = fc.attn_metadata
        if attn_md is not None and attn_md.state is not AttnState.DECODE:
            batch_id_per_token = getattr(attn_md, "batch_id_per_token", None)
            num_real = (
                int(batch_id_per_token.shape[0])
                if torch.is_tensor(batch_id_per_token)
                else x.shape[0]
            )
            if 0 <= num_real < x.shape[0]:
                if os.environ.get("ATOM_SGLANG_V4_DEBUG") == "1":
                    import logging

                    logging.getLogger("atom.plugin.sglang.deepseek_v4_attention").info(
                        "Slice padded V4 prefill attention: layer=%s real=%s padded=%s",
                        getattr(self, "layer_id", None),
                        num_real,
                        x.shape[0],
                    )
                out = self._sglang_v4_forward_impl(x[:num_real], positions[:num_real])
                return torch.nn.functional.pad(out, (0, 0, 0, x.shape[0] - num_real))
        return self._sglang_v4_forward_impl(x, positions)

    attn.forward_impl = types.MethodType(_forward_impl, attn)

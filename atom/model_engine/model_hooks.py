# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The decode round a model's install may give DeepSeek-V4.1's attention
metadata builder (``DeepseekV41MetadataBuilder.use_decode_round``): the part of
a decode step the model takes over, as the builder lays the step out.

The builder asks it at four points of a step, every one of them its own:
whether the step consumes the previous verdict on the device
(``consumes_device_verdict``, the builder's answer to the runner), the step's
one routing decision before its layout (``service``), the step laid out
(``laid_out``) and the speculative commit (``committed``). ``ModelRunner``
calls none of it. DeepSeek-V4.1's hi path (``atom/models/deepseek_v41/mono/``)
is the one model with a decode round.
"""

from __future__ import annotations

from atom.utils.forward_context import RoundService


class DecodeRound:
    """A model's decode round at its builder's points of a step. Every method
    defaults to "nothing taken over"."""

    # On a served step, the model looks its Engram rows up itself (the staging
    # only advances the cursor: ``EngramStep.served_by_round``); a model that
    # reads the staged rows says False.
    looks_up_engram_rows = True

    def consumes_device_verdict(self, batch, max_seqlen_q) -> bool:
        """Before the step's input ids, every forward: whether the step
        advances from the previous verdict on the device
        (``RoundService.DEVICE``), so the runner defers its status wait
        (``AttentionMetadataBuilder.consumes_device_verdict``)."""
        return False

    def service(
        self,
        batch,
        spans,
        running_bs,
        running_tokens,
        *,
        max_q_len,
        decode,
        tentative,
        is_prefill,
        resumes,
        image_rows,
    ) -> RoundService:
        """Before a step's layout: how much of it the round takes over
        (``AttentionMetaData.served_by_round``)."""
        return RoundService.NONE

    def laid_out(self, metadata) -> None:
        """After the step's layout and publication."""

    def committed(self, metadata, anchors) -> bool:
        """Before the speculative commit of a served step at the verdict's
        ``anchors``: True when the forward committed it already."""
        return False

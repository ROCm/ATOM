# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The target's final norm and LM head of a mono step in one launch: the
shared mono LM head (``atom.mono.lm_head``) over V4.1's ``model.norm`` and
``model.head``, its rows the step's collapsed hidden state."""

from atom.mono.lm_head.head import MonoLmHead


def bind_lm_head(model, tp: int, max_rows: int, verifiers, input_ids):
    """The launch over ``model``'s final norm and head (``MonoLmHead``), None
    where the norm is not the plain RMSNorm the launch applies, the head not
    this TP group's vocab shard, or the launch does not serve the head
    (``MonoLmHead.refusal``)."""
    head, norm = model.head, model.norm
    plain_norm = not (
        norm.x_pad_to_multiple or norm.use_fused_quant or norm.fused_allreduce
    )
    if not plain_norm or head.tp_size != tp:
        return None
    return MonoLmHead.bind(
        head.weight, head.bias, (norm.weight, norm.eps), tp, max_rows,
        verifiers, input_ids,
    )  # fmt: skip

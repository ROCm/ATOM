# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The hand-offs of a sequence-parallel decoder layer on four ranks, hidden 6144, 2048 attention-output columns per
rank (``handoffs.py``): the attention output's MXFP8 pack + unpacking all-to-all + o_proj_full + post-attention norm
(+ MoE input preparation) + gather, the FFN output's reduce-scatter + next input norm / quant + gather, owned with
their communicators, pool slots and warm-up; ``handoffs_sdma.py`` the same phases on SDMA pushes. The torch contract is ``reference.py``."""

from .handoffs import (
    H,
    MAX_TOKENS,
    MIN_TOKENS,
    Q,
    ROW_ALIGN,
    WORLD,
    SeqParallelHandoffs,
    pack,
    shard,
    shard_rows,
)
from .handoffs_sdma import SdmaSeqParallelHandoffs
from .mxfp8_pack_q2048 import mxfp8_pack_q2048

__all__ = ["H", "MAX_TOKENS", "MIN_TOKENS", "Q", "ROW_ALIGN", "WORLD", "SeqParallelHandoffs", "SdmaSeqParallelHandoffs", "pack", "shard", "shard_rows",
           "mxfp8_pack_q2048"]

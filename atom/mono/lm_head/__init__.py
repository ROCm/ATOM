# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""A model's LM head picking each row's id in one mono launch, for any model
whose head is a vocab-parallel bf16 projection: ``dot`` the vocab shard's
logits, ``argmax`` the pick over every TP rank, ``kernel`` the launch (an
optional final RMSNorm first, a temperature row's draw), ``top_p`` a top-p
row's draw in it (``top_p_candidates`` over the largest logits, else
``top_p_scan``; ``top_p_logits`` the slices both read), ``head`` its binding
as a verify head's picks."""

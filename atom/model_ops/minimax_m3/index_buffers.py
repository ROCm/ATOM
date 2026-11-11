# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MXFP4 index scorer's persistent state as one record, in a module that imports
no aiter or triton: the kernels' modules (`sparse_attn`, `index_topk`) import aiter, and
the non-GPU test suite has to be able to build the record."""

from dataclasses import dataclass

import torch

from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace


@dataclass(frozen=True)
class IndexScoreBuffers:
    """The MXFP4 index scorer's persistent state, one immutable record so that none of
    its parts can be present without the others.

    ``scores`` is the `LogitsWorkspace` every layer's scorer writes its fp32 block
    scores into (the same plane the other indexers use), viewed contiguous
    ``[heads, rows, width]`` -- the layout aiter's selector needs. Never filled: the
    selectors read a row only up to its valid blocks, which the scorer writes every
    step. ``work_map`` is the decode scorer's dispatch map and ``max_block`` the block
    count it was built at, which the scorer checks for an exact match and which must
    not move between graph replays: the builder holds the capacity buffer, the step's
    metadata the map built into it. A prefill step's record has neither.
    """

    scores: LogitsWorkspace
    work_map: torch.Tensor | None = None
    max_block: int = 0

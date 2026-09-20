# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Row chunking for the dense ``fp8_mqa_logits`` indexer logits buffer.

Every sparse-MLA indexer prefill path (native DeepSeek V3.2/V4, GLM-5.x, and the
vLLM plugin) scores queries against the committed KV through a dense
``[rows, row_width]`` fp32 logits matrix. ``row_width`` is the sum of all
co-scheduled prefill contexts, which ``max_num_batched_tokens`` does not bound,
so a burst of long-context requests can push one allocation to tens of GiB
(issue #1376). Those paths therefore split the Q rows into chunks; this module
owns the one rule they all need to agree on.
"""

# A buffer resource descriptor addresses at most 2^31 bytes, so aiter's
# fp8_mqa_logits drops to plain global load/store once the logits tensor reaches
# 2 GiB. On gfx950 that USE_BUFFER_STORE=False specialization of the gluon kernel
# does not survive codegen -- the AMDGCN backend trips
# `llvm/ADT/Sequence.h: Assertion 'Begin <= End'` and abort()s the process, with
# no Python traceback and every TP rank dying at once. Landing exactly on the cap
# is easy to do by accident: 16384 rows x 32768 committed tokens x 4 B is 2 GiB
# to the byte. Keep the buffer STRICTLY under the cap so the buffer-store
# specialization is the only one ever compiled.
_BUFFER_DESCRIPTOR_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

_LOGIT_BYTES = 4  # fp32
# The kernels tile rows by 128; keep chunk sizes on that grid.
_ROW_TILE = 128


def sparse_indexer_row_chunk(total_rows: int, row_width: int, budget_mb: int) -> int:
    """Rows to score per ``fp8_mqa_logits`` call for a ``[rows, row_width]`` buffer.

    ``budget_mb`` is the caller's soft byte budget (``0`` = no soft budget); the
    2 GiB buffer-descriptor cap applies either way, because exceeding it is a
    hard crash rather than a memory-pressure trade-off.

    Returns ``total_rows`` when one shot already fits. Otherwise the budget-derived
    row count rounded DOWN: to a multiple of 128 in the normal regime, avoiding
    coarse power-of-2 doubling; below 128 rows (extreme ``row_width``) to a
    power-of-2 floor, so it degrades 64/32/.../1 instead of collapsing to 1.
    """
    if total_rows <= 0 or row_width <= 0:
        return max(total_rows, 1)

    budget_bytes = budget_mb * 1024 * 1024
    cap_bytes = _BUFFER_DESCRIPTOR_LIMIT_BYTES
    if budget_bytes > 0:
        cap_bytes = min(budget_bytes, cap_bytes)

    # Exclusive bound: the kernel needs bytes < cap, not <=.
    max_rows = (cap_bytes - 1) // (row_width * _LOGIT_BYTES)
    if max_rows >= total_rows:
        return total_rows
    if max_rows >= _ROW_TILE:
        return (max_rows // _ROW_TILE) * _ROW_TILE
    return 1 << (max(1, max_rows).bit_length() - 1)

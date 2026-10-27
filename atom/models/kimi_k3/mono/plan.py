# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which steps Kimi-K3's mono kernels serve, and how a served step is keyed.

Every decision here is arithmetic over a step's metadata: no kernels, no TP
group, no GPU. It sits apart from ``runner`` for that reason -- ``runner``
binds the TP group and the FlyDSL launchers at import, so a test that imported
it to reach ``step_shape`` could only run on a machine with AITER and FlyDSL,
and ATOM's only automated test gate (``.github/scripts/run_unit_tests.sh``) is
CPU-only. DeepSeek-V4.1 splits its per-step decisions out the same way
(``deepseek_v41/mono/*_plan.py``), each called by both the kernel and its CPU
test.
"""

import torch

from atom.models.kimi_k3.mono.config import MAX_QLEN, MAX_ROWS


def kda_metadata(forward_context):
    """This step's KDA (or GDN) metadata, or None."""
    md = forward_context.attn_metadata
    for name in ("kda_metadata", "gdn_metadata"):
        found = getattr(md, name, None)
        if found is not None:
            return found
    return None


def step_shape(metadata, rows: int) -> tuple[int | None, str]:
    """``(the uniform spec length of a step the kernels serve, why not)``.

    K1 places one task per (request, head, v-chunk) and reads a per-token
    snapshot slot ``st_idx[request][token]``, so the step must be one shape: no
    prefill, no plain decode mixed in, every request the same query length, and
    the rows within the width the buffers and the build were made for.

    Every refusal is named rather than silent. Mono that serves nothing looks
    exactly like mono that serves everything from the outside -- the answers are
    the original path's either way -- so a step turned away has to say so.
    """
    if metadata is None:
        return None, "no KDA metadata"
    if getattr(metadata, "spec_sequence_masks", None) is None:
        return None, "not a speculative step (no spec_sequence_masks)"
    if metadata.num_prefills:
        return None, f"{metadata.num_prefills} prefills in the batch"
    if metadata.num_decodes:
        return None, f"{metadata.num_decodes} plain decodes in the batch"
    if metadata.num_spec_decodes < 1:
        return None, "no spec decodes"
    if getattr(metadata, "replayssm", False):
        # ReplaySSM keeps the slot table's shape but leaves only column 0 live
        return None, "ReplaySSM (only column 0 of the slot table is live)"
    index = metadata.spec_state_indices_tensor
    if index is None:
        return None, "no spec_state_indices_tensor"
    if index.dtype != torch.int32 or index.stride(-1) != 1:
        return None, f"slot table {index.dtype} stride {index.stride(-1)}"
    qlen = index.size(-1)
    if not 0 < qlen <= MAX_QLEN:
        return None, f"query length {qlen} > {MAX_QLEN}"
    if metadata.num_spec_decodes * qlen != rows:
        return None, (
            f"{metadata.num_spec_decodes} requests x {qlen} tokens != {rows} rows"
        )
    if rows != metadata.num_actual_tokens:
        return None, f"{rows} rows != num_actual_tokens {metadata.num_actual_tokens}"
    if not 0 < rows <= MAX_ROWS:
        return None, f"{rows} rows > the {MAX_ROWS}-row buffers"
    accepted = metadata.num_accepted_tokens
    if accepted is None or accepted.dtype != torch.int32:
        return None, "no int32 num_accepted_tokens"
    return qlen, ""


def step_rows(metadata, rows: int) -> int | None:
    """``step_shape``'s query length alone."""
    return step_shape(metadata, rows)[0]


def width_key(rows: int, qlen: int) -> int:
    """``WidthBuilds``' key for a step: both its rows and its query length pick
    the kernels, and the two are packed into the one int it keys by."""
    return rows * (MAX_QLEN + 1) + qlen


def _width(key: int) -> tuple[int, int]:
    return divmod(key, MAX_QLEN + 1)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""References for the presence penalty (`atom/model_ops/presence_penalty.py`).

`expected_logits` is vLLM's rule computed from each request's generated tokens
directly, with no table. `penalize_rows_twin` and `mark_emitted_twin` follow
the contract of the kernels in `triton_presence_penalty.py` on the bit table,
in torch, so the CPU tests can drive the token-constraint boundary; the GPU
test holds the kernels to `expected_logits`.
"""

import numpy as np
import torch


def expected_logits(logits, generated, row_request, row_drafts, penalties):
    """`logits` [rows, V]; `generated[r]` request r's output ids so far;
    `row_request[t]` row t's request; `row_drafts[t]` the drafts before row t;
    `penalties[r]` request r's penalty (0 for none)."""
    out = logits.float().clone()
    for t, r in enumerate(row_request):
        if penalties[r] == 0:
            continue
        ids = sorted(set(generated[r]) | set(row_drafts[t]))
        out[t, ids] -= penalties[r]
    return out.to(logits.dtype)


def bit_rows(seen):
    """The table as a bool [slots, 32 * words] array."""
    words = seen.cpu().numpy().view(np.uint32)
    return (
        np.unpackbits(words.view(np.uint8), bitorder="little")
        .reshape(seen.shape[0], -1)
        .astype(bool)
    )


def penalize_rows_twin(seen, rows, logits, input_ids, max_drafts):
    for slot in rows.fresh.tolist():
        seen[slot] = 0
    vocab = min(logits.shape[-1], seen.shape[1] * 32)
    _mark_twin(seen, rows.resumed_slots, rows.resumed_ids.view(-1, 1), vocab)
    bits = bit_rows(seen)
    for t, slot in enumerate(rows.slots.tolist()):
        if slot < 0:
            continue
        hit = torch.from_numpy(bits[slot, :vocab].copy())
        if input_ids is not None and max_drafts:
            d = int(rows.drafts_before[t])
            for k in range(1, d + 1):
                hit[int(input_ids[t - d + k])] = True
        logits[t, :vocab][hit] -= rows.penalties[t]


def mark_emitted_twin(seen, rows, tokens, vocab):
    bs = rows.emit_slots.numel()
    _mark_twin(seen, rows.emit_slots, tokens.reshape(tokens.shape[0], -1)[:bs], vocab)


def _mark_twin(seen, slots, tokens, vocab):
    one = torch.tensor(1, dtype=torch.int32)
    for r, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        for tok in tokens[r].tolist():
            if 0 <= tok < vocab:
                seen[slot, tok // 32] |= one << (tok % 32)

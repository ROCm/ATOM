# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``BatchShardedSampling`` (``host.py``), on the whole batch in one process.

The block only sequences the per-kernel entries, so their references are its contract:

- ``target_logits``: rank r's rows are the owner-sorted rows ``hidden[sorted_rows]`` of owner r, each the full
  vocabulary row ``bf16(fp32 hidden_row @ W^T)`` (``bf16_gemm_n50016_k6144``'s reference) with shard s at columns
  [s * 50016, (s + 1) * 50016).
- ``gather_sampler_output``: batch request i is local row i // 4 of rank i % 4 (``owner_tables``); tokens, counts
  and NaN counts are copied (exact).
- ``markov_draft``: for request i (owner i % 4) and step j, ``base = logits(head_hidden[sample_indices[i * steps +
  j]])``, ``bias = markov_w2 @ markov_w1[prev]`` (fp32 accumulate, bf16), the cache row ``bf16(base + bias)`` and
  the Gumbel draw of ``gumbel_partials_bias``'s reference at position ``sample_pos - 1``; ``prev`` = the drawn id
  (step 0: ``input_ids[anchor_idx[i]]``). ``markov_draft_reference`` below is the greedy (temperature 0) case: the
  draw is then exact (no noise), the first maximum.

Tolerance: the cache rows cosine ~1 against this reference (the GEMMs' fp32 K order differs from torch's), the ids
equal where the rows have no near-tie; the tests compare the block with the per-kernel entries bit for bit.
"""

import torch

V, H, RANK, WORLD = 200064, 6144, 256, 4


def owner_order_reference(num_reqs: int) -> list[list[int]]:
    """The batch positions each rank owns, in its local row order."""
    return [list(range(r, num_reqs, WORLD)) for r in range(WORLD)]


def markov_draft_reference(rows: torch.Tensor, w: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
                           prev: torch.Tensor, steps: int):
    """Greedy Markov draft of n requests: ``rows`` bf16 [n, steps, 6144] (each step's head row), ``w`` the row-major
    lm_head [200064, 6144], ``prev`` int64 [n] (the anchor tokens) -> (ids int64 [n, steps], cache rows bf16
    [n, steps, 200064])."""
    n = rows.shape[0]
    ids = torch.empty((n, steps), dtype=torch.int64, device=rows.device)
    cache = torch.empty((n, steps, V), dtype=torch.bfloat16, device=rows.device)
    prev = prev.clone()
    for j in range(steps):
        base = (rows[:, j].float() @ w.float().T).to(torch.bfloat16)
        bias = (w1[prev].float() @ w2.float().T).to(torch.bfloat16)
        y = (base.float() + bias.float()).to(torch.bfloat16)
        cache[:, j] = y
        prev = y.float().argmax(dim=1)
        ids[:, j] = prev
    return ids, cache

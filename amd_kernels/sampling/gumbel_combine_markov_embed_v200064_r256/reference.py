# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``gumbel_combine_markov_embed_packed``: the tail of one DSpark Markov draft step on the owner rank.

Inputs: the Gumbel kernel's per-block partials of the owned draft rows, ``local_max`` fp32 [n, 196] (the noised
maximum of each 1024-column block of the 200064-column row) and ``local_argmax`` int64 [n, 196] (its vocab id);
``local_rows`` int64 [n] (each owned request's row in the padded batch); ``markov_w1`` bf16 [200064, 256].

For each owned row r: the winning block is the first block whose value is NaN, otherwise the first block that holds
the row maximum (vLLM's ``_gumbel_combine_kernel``, i.e. ``torch.argmax`` semantics); ``tok = local_argmax[r, win]``.
Then (vLLM's ``DSparkSpeculator._sample_sequential_local``): ``draft_ids[r, step] = tok``, ``prev[local_rows[r]] =
tok`` and ``embed[local_rows[r]] = markov_w1[tok]`` (the next step's ``markov_embed`` row, a copy). All results are
exact: the tests require equality. Rows of ``prev`` / ``embed`` that no owned row names are left as they are.
"""

import torch

V, NB, RANK = 200064, 196, 256


def combine_reference(local_max: torch.Tensor, local_argmax: torch.Tensor) -> torch.Tensor:
    """int64 [n]: the token of the first NaN block, otherwise of the first maximal block."""
    n, nb = local_max.shape
    blk = torch.arange(nb, device=local_max.device).expand(n, nb)
    big = torch.full_like(blk, nb)
    isn = torch.isnan(local_max)
    first_nan = torch.where(isn, blk, big).min(dim=1).values
    v = torch.where(isn, torch.full_like(local_max, float("-inf")), local_max)
    mx = v.max(dim=1, keepdim=True).values
    first_max = torch.where(v == mx, blk, big).min(dim=1).values
    win = torch.where(first_nan < nb, first_nan, first_max)
    return local_argmax.gather(1, win[:, None])[:, 0]


def gumbel_combine_markov_embed_reference(local_max, local_argmax, local_rows, markov_w1, draft_ids, step, prev,
                                          embed):
    tok = combine_reference(local_max, local_argmax)
    draft_ids[: tok.shape[0], step] = tok
    prev.index_copy_(0, local_rows, tok)
    embed.index_copy_(0, local_rows, markov_w1[tok])
    return tok

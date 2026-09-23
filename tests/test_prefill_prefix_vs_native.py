# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Splitting a token's keys into prefix + extend must not change its output.

`sparse_attn_v4_paged_prefill` takes each query token's keys in two pieces: the
`prefix` rows it reads out of the paged pool, and the `extend` rows carried in
this forward's own KV tensor. Which keys land in which piece is decided by
where the prompt was cut, not by anything the attention itself cares about --
an unchunked prompt puts every key in `extend`, and the same prompt cut in two
moves the head of that list into `prefix`.

So the split is a scheduling artifact, and the kernel owes the caller the same
answer either way. If it does not, chunked prefill is lossy by construction:
every resumed chunk would compute attention differently from the one-shot run,
permanently, for reasons no amount of state carry-over can fix.

The arms below feed BIT-IDENTICAL key values under both splits -- the pool rows
are overwritten with the very tensors the extend arm passes inline -- so the
only difference is which argument they arrive through.
"""

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "exercises the V4 prefill attention kernel; needs a real GPU",
        allow_module_level=True,
    )

from atom.model_ops.v4_kernels.paged_prefill import sparse_attn_v4_paged_prefill

DEV = "cuda"
H = 64
D = 512  # the bf16 OPUS kernel is compiled for D=512 only
PAGES = 1024
SCALE = 1.0 / (D**0.5)


def _indptr(counts):
    v = np.zeros(len(counts) + 1, np.int32)
    v[1:] = np.cumsum(counts)
    return torch.tensor(v, dtype=torch.int32, device=DEV)


def _fixture(n_tokens, keys_per_token, seed=0):
    """Query rows plus one shared key table every arm draws the same keys from."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(n_tokens, H, D, generator=g, dtype=torch.float32, device=DEV).to(
        torch.bfloat16
    )
    keys = torch.randn(
        n_tokens * keys_per_token, D, generator=g, dtype=torch.float32, device=DEV
    ).to(torch.bfloat16)
    return q, keys


def _run_split(q, keys, keys_per_token, n_prefix):
    """Token t attends to its own `keys_per_token` keys, `n_prefix` via the pool.

    The pool is built so pool[row] == keys[row] for the rows routed through
    `prefix`, which is what makes the two arms comparable at all.
    """
    n_tokens = q.shape[0]
    pool = torch.zeros(PAGES, D, dtype=torch.bfloat16, device=DEV)

    prefix_rows, extend_rows = [], []
    for t in range(n_tokens):
        base = t * keys_per_token
        mine = list(range(base, base + keys_per_token))
        head, tail = mine[:n_prefix], mine[n_prefix:]
        # Route `head` through the pool: park each key at a distinct pool slot
        # holding exactly that key's value.
        for r in head:
            pool[r % PAGES] = keys[r]
        prefix_rows.append([r % PAGES for r in head])
        extend_rows.append(tail)

    p_idx = torch.tensor(
        [c for row in prefix_rows for c in row] or [0], dtype=torch.int32, device=DEV
    )
    e_idx = torch.tensor(
        [c for row in extend_rows for c in row] or [0], dtype=torch.int32, device=DEV
    )
    out = sparse_attn_v4_paged_prefill(
        q,
        pool,
        p_idx,
        _indptr([len(r) for r in prefix_rows]),
        keys,
        e_idx,
        _indptr([len(r) for r in extend_rows]),
        torch.zeros(H, dtype=torch.float32, device=DEV),
        SCALE,
    )
    torch.cuda.synchronize()
    return out


ULP_BF16 = 2.0**-8  # relative step of bf16's 8-bit significand


@pytest.mark.parametrize("n_prefix", [1, 4, 7])
def test_the_split_only_reassociates_it_does_not_bias(n_prefix):
    """Moving keys between the two arguments may reround, but must not skew.

    The split DOES change the output bitwise -- it reassociates the sum, and
    ~10% of slots land on a different bf16 value. That by itself costs nothing:
    what would cost accuracy is a systematic pull in one direction. So the
    assertion is on the mean of the difference against its own spread, not on
    how many slots moved. Counting slots is the mistake that once produced a
    wrong root cause here ("the kernel's split dependency"); reassociation
    error is unbiased and unbiased error does not accumulate into a loss.
    """
    keys_per_token = 8
    q, keys = _fixture(n_tokens=8, keys_per_token=keys_per_token)

    native = _run_split(q, keys, keys_per_token, n_prefix=0)
    assert native.abs().sum().item() > 0, "native arm produced nothing; test is vacuous"
    split = _run_split(q, keys, keys_per_token, n_prefix=n_prefix)

    d = (native.float() - split.float()).flatten()
    nz = d[d != 0]
    assert nz.numel() > 0, "arms were bit-identical; this case proves nothing"

    # Magnitude: within a couple of ulp of the values being summed.
    scale = native.float().abs().max().item()
    assert nz.abs().max().item() <= 4 * ULP_BF16 * scale, (
        f"n_prefix={n_prefix}: difference is too large to be rounding "
        f"(max|diff|={nz.abs().max().item():.3g}, 1 ulp ~ {ULP_BF16 * scale:.3g})"
    )
    # Direction: the mean must sit inside the noise of its own sample.
    t_stat = nz.mean().item() / (nz.std().item() / nz.numel() ** 0.5)
    assert abs(t_stat) < 4.0, (
        f"n_prefix={n_prefix}: the split is BIASED, not just rerounded "
        f"(mean/SEM={t_stat:+.2f} over {nz.numel()} differing slots) -- "
        f"a directional error here would make chunked prefill lossy"
    )


def test_all_keys_in_prefix_equals_all_keys_in_extend():
    """The extreme split: nothing carried inline vs everything carried inline.

    Unlike the mixed splits above this one IS exact -- with every key on one
    side there is nothing to reassociate -- which is what shows the differences
    above come from mixing the two accumulators, not from the pool path itself
    reading different values.
    """
    keys_per_token = 8
    q, keys = _fixture(n_tokens=4, keys_per_token=keys_per_token, seed=3)
    all_extend = _run_split(q, keys, keys_per_token, n_prefix=0)
    all_prefix = _run_split(q, keys, keys_per_token, n_prefix=keys_per_token)
    diff = (all_extend.float() - all_prefix.float()).abs()
    assert torch.equal(all_extend, all_prefix), (
        f"all-prefix differs from all-extend in "
        f"{int((diff > 0).sum().item())} slots, max|diff|={diff.max().item():.6g}"
    )

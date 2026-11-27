# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU checks for the DP greedy argmax head, with external kernels stubbed.

The real head exchange runs without AITER or a GPU. The
exchange's pack boundary uses a CPU reference; test_lm_head_argmax.py covers
the actual Triton kernel separately. Startup and request admission are in
test_dp_main_argmax_init.py, the sampler bypass in
test_token_constraint_wiring.py.
"""

import types

import pytest
import torch
from aiter_stub import atom_without_aiter

eh = None  # bound against the CPU stand-ins by `_modules`


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global eh
    with atom_without_aiter() as load:
        eh = load("atom.model_ops.embed_head")
        yield


DP, VSHARD = 4, 129280 // 4


# ---------------------------------------------------------------- bitwise ---
def _rows_with_ties(n_rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = DP * VSHARD
    rows = (torch.randint(-4, 4, (n_rows, v), generator=g).float() * 0.5).to(
        torch.bfloat16
    )
    special = [
        torch.full((v,), 1.25),  # all equal -> id 0
        torch.full((v,), -float("inf")),  # all -inf -> id 0
    ]
    r = torch.full((v,), -3.0)
    r[[VSHARD + 7, 3 * VSHARD + 1, 2 * VSHARD - 1]] = 9.0  # tie across shards
    special.append(r)
    r = torch.full((v,), -1.0)
    r[[5, VSHARD + 2]] = -0.0
    r[[3, 2 * VSHARD]] = 0.0  # signed-zero tie -> id 3
    special.append(r)
    r = torch.full((v,), -2.0)
    r[v - 1] = 4.0  # last id
    special.append(r)
    r = torch.full((v,), -float("inf"))
    r[2 * VSHARD + 11] = -7.0
    special.append(r)
    for i, s in enumerate(special):
        if i < n_rows:
            rows[i] = s.to(torch.bfloat16)
    return rows


def _lowest_max_id(rows):
    f = rows.float()
    hit = f == f.max(dim=1, keepdim=True).values
    return torch.where(hit, torch.arange(f.shape[1]), f.shape[1]).min(dim=1).values


def _cpu_pack(logits, vocab_start):
    """Reference for the external pack boundary, with first-index tie breaking."""
    values, indices = logits.float().max(dim=1)
    return torch.stack((values, (indices + vocab_start).float()), dim=1)


@pytest.mark.parametrize("local_rows", [3, 2])
def test_dp_argmax_matches_lowest_id_of_full_row(monkeypatch, local_rows):
    """The real exchange chooses the lowest global max id across vocab shards,
    including padding, cross-shard ties, signed zero and all-negative-infinity
    rows. Only the external GEMM and pack kernels are replaced."""
    max_rows = 3
    full = _rows_with_ties(DP * max_rows)  # the Σrows x V logits every path sees
    ref = _lowest_max_id(full)
    shards = [full[:, r * VSHARD : (r + 1) * VSHARD].contiguous() for r in range(DP)]
    packs = [_cpu_pack(shards[r], r * VSHARD) for r in range(DP)]

    monkeypatch.setattr(
        eh, "envs", types.SimpleNamespace(ATOM_USE_CUSTOM_ALL_GATHER=False)
    )
    monkeypatch.setattr(eh, "lm_head_argmax_pack", _cpu_pack)
    for rank in range(DP):
        monkeypatch.setattr(
            eh, "tgemm", types.SimpleNamespace(mm=lambda *a, _r=rank, **k: shards[_r])
        )

        def gather(x, dim=0, use_custom=False, _r=rank):
            if x.shape[-1] == 2 and x.dtype == torch.float32:
                assert torch.equal(x, packs[_r])
                return torch.cat(packs, dim=0)
            return torch.cat([x] * DP, dim=0)

        group = types.SimpleNamespace(
            world_size=DP, rank_in_group=rank, all_gather=gather
        )
        monkeypatch.setattr(eh, "get_dp_group", lambda _g=group: _g)
        ctx = types.SimpleNamespace(running_tokens=max_rows)
        monkeypatch.setattr(
            eh, "get_forward_context", lambda _c=ctx: types.SimpleNamespace(context=_c)
        )
        head = eh.ParallelLMHead.__new__(eh.ParallelLMHead)
        torch.nn.Module.__init__(head)
        head.tp_size, head.num_embeddings = 1, DP * VSHARD
        head.weight = torch.nn.Parameter(torch.empty(DP * VSHARD, 4), False)
        head.bias = None
        got = head._dp_sharded_logits(torch.zeros(local_rows, 4), "argmax")
        assert got.dtype == torch.float32 and got.shape == (local_rows,)
        start = rank * max_rows
        assert got.to(torch.int32).tolist() == ref[start : start + local_rows].tolist()


# ----------------------------------------------------------- head choice ---
@pytest.mark.parametrize(
    "mode, ctx, dp_meta, expect",
    [
        ("argmax", {}, True, "argmax"),
        ("all2all", {}, True, "all2all"),
        ("allgather", {}, True, "allgather"),
        ("default", {}, True, None),
        ("argmax", {"is_draft": True}, True, None),
        ("argmax", {"is_prefill": True}, True, None),
        ("argmax", {"running_tokens_are_unified": False}, True, None),
        ("argmax", {}, False, None),
    ],
)
def test_forward_takes_the_modes_exchange(monkeypatch, mode, ctx, dp_meta, expect):
    """A sharded decode exchanges what `ATOM_DP_LM_HEAD_MODE` names (greedy
    ids under `argmax`); any other step runs the replicated GEMM."""
    c = types.SimpleNamespace(
        is_prefill=False,
        is_draft=False,
        running_tokens_are_unified=True,
        scheduled_bs=2,
    )
    for k, v in ctx.items():
        setattr(c, k, v)
    monkeypatch.setattr(eh, "is_plugin_mode", lambda: False)
    monkeypatch.setattr(eh, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(eh, "sp_gather_rows", lambda x, idx: x)
    monkeypatch.setattr(eh, "envs", types.SimpleNamespace(ATOM_DP_LM_HEAD_MODE=mode))
    monkeypatch.setattr(
        eh, "get_dp_group", lambda: types.SimpleNamespace(world_size=DP)
    )
    meta = types.SimpleNamespace(cu_seqlens_q=torch.tensor([0, 1, 2]))
    monkeypatch.setattr(
        eh,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            context=c, attn_metadata=meta, dp_metadata=object() if dp_meta else None
        ),
    )
    replicated = torch.zeros(2, 8)
    monkeypatch.setattr(
        eh, "tgemm", types.SimpleNamespace(mm=lambda *a, **k: replicated)
    )
    head = eh.ParallelLMHead.__new__(eh.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.tp_size, head.num_embeddings = 1, DP * VSHARD
    head.weight, head.bias = None, None
    head.dp_shards_vocab = eh.dp_shards_vocab(head.tp_size, head.num_embeddings)
    calls = []
    ids = torch.tensor([5.0, 7.0])

    def sharded(x, m):
        calls.append(m)
        return ids if m == "argmax" else torch.ones(2, 8)

    head._dp_sharded_logits = sharded
    out = head(torch.zeros(2, 4))
    assert calls == ([expect] if expect else [])
    if expect == "argmax":
        assert out is ids
    elif expect is None:
        assert out is replicated

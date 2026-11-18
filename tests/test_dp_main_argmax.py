"""CPU checks for the opt-in DP main argmax path, with external kernels stubbed.

The real head exchange and runner methods run without AITER or a GPU. The
exchange's pack boundary uses a CPU reference; test_lm_head_argmax.py covers
the actual Triton kernel separately. Constraint/fallback and sampler-bypass
coverage lives in test_token_constraint_wiring.py.
"""

import types

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

mr = eh = None  # bound against the CPU stand-ins by `_modules`


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global mr, eh
    with atom_without_aiter() as load:
        mr = load("atom.model_engine.model_runner")
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


# ------------------------------------------------------------ head gating ---
class _Ctx(types.SimpleNamespace):
    pass


@pytest.mark.parametrize(
    "plugin, mode, ctx, dp_meta, expect",
    [
        (False, "allgather", {}, True, True),
        (False, "all2all", {}, True, True),
        (True, "allgather", {}, True, False),
        (False, "default", {}, True, False),
        (False, "allgather", {"is_prefill": True}, True, False),
        (False, "allgather", {"is_draft": True}, True, False),
        (False, "allgather", {"running_tokens_are_unified": False}, True, False),
        (False, "allgather", {}, False, False),
    ],
)
def test_dp_argmax_token_gate(monkeypatch, plugin, mode, ctx, dp_meta, expect):
    c = _Ctx(is_prefill=False, is_draft=False, running_tokens_are_unified=True)
    for k, v in ctx.items():
        setattr(c, k, v)
    monkeypatch.setattr(eh, "is_plugin_mode", lambda: plugin)
    monkeypatch.setattr(eh, "envs", types.SimpleNamespace(ATOM_DP_LM_HEAD_MODE=mode))
    monkeypatch.setattr(
        eh, "get_dp_group", lambda: types.SimpleNamespace(world_size=DP)
    )
    monkeypatch.setattr(
        eh,
        "get_forward_context",
        lambda: types.SimpleNamespace(
            context=c, dp_metadata=object() if dp_meta else None
        ),
    )
    head = eh.ParallelLMHead.__new__(eh.ParallelLMHead)
    torch.nn.Module.__init__(head)
    head.tp_size, head.num_embeddings = 1, DP * VSHARD
    calls = []
    head._dp_sharded_logits = lambda x, m: calls.append(m) or torch.zeros(2)
    out = head.dp_argmax_token(torch.zeros(2, 4))
    assert (out is not None) is expect
    assert calls == (["argmax"] if expect else [])


# --------------------------------------------------------- runner wiring ---
class _Head:
    def __init__(self, ids):
        self.ids, self.local_rows = ids, []

    def dp_argmax_token(self, x):
        return self.ids

    def local_logits(self, x):
        self.local_rows.append(x.shape[0])
        return torch.ones(x.shape[0], 8, dtype=torch.bfloat16)


def _runner(head=None, compute_head=None):
    runner = object.__new__(mr.ModelRunner)
    runner.compute_head = compute_head
    runner.dp_argmax_head = head
    runner.compute_logits_calls = 0

    def compute_logits(x):
        runner.compute_logits_calls += 1
        return torch.zeros(x.shape[0], 8, dtype=torch.bfloat16)

    runner.model = types.SimpleNamespace(compute_logits=compute_logits)
    return runner


def _batch(n=3):
    return types.SimpleNamespace(
        temperatures=np.zeros(n, np.float32),
        any_logprobs=False,
        structured_requests={},
        required_channel=None,
    )


def test_head_output_takes_ids_only_with_logits():
    ids = torch.tensor([5.0, 7.0, 1.0])
    r = _runner(_Head(ids))
    assert r._head_output(torch.zeros(3, 4)) is ids
    assert r.compute_logits_calls == 0
    assert r._head_output(torch.zeros(3, 4), logits=False) is None
    r = _runner(_Head(None))
    assert r._head_output(torch.zeros(3, 4)).shape == (3, 8)
    assert r.compute_logits_calls == 1
    r = _runner(None)
    assert r._head_output(torch.zeros(3, 4)).shape == (3, 8)


def test_step_head_without_batch_uses_local_logits():
    ids = torch.tensor([5.0, 7.0, 1.0])
    head = _Head(ids)
    r = _runner(head)
    out = r._step_head(ids, torch.zeros(4, 4), None)
    assert out.dim() == 2 and out.shape == (3, 8)
    assert head.local_rows == [3]
    assert r.compute_logits_calls == 0


def test_step_head_logits_untouched():
    logits = torch.zeros(3, 8)
    r = _runner(_Head(None))
    assert r._step_head(logits, torch.zeros(3, 4), _batch(3)) is logits
    r._step_head(None, torch.zeros(3, 4), _batch(3))
    assert r.compute_logits_calls == 1
    ids = torch.tensor([5.0, 7.0, 1.0])
    off = _runner(None)  # flag off: a 1-D head is not taken for ids
    assert off._step_head(ids, torch.zeros(3, 4), _batch(3)) is ids

"""CPU checks for ATOM_DP_MAIN_ARGMAX (needs aiter; the pack kernel runs in
the Triton interpreter):

  TRITON_INTERPRET=1 python -m pytest -q tests/test_dp_main_argmax.py
"""

import os
import types

import numpy as np
import pytest
import torch

_arch = pytest.importorskip(
    "aiter.ops.triton.utils._triton.arch_info",
    reason="model_runner and embed_head import the AITER runtime",
)
if _arch._CACHED_ARCH is None and not torch.cuda.is_available():
    _arch._CACHED_ARCH = "gfx1250"  # GPU-less: aiter probes the arch at import

import atom.model_engine.model_runner as mr
import atom.model_ops.embed_head as eh

interpret_only = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1", reason="needs TRITON_INTERPRET=1"
)

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


@interpret_only
@pytest.mark.parametrize("local_rows", [3, 2])
def test_dp_argmax_matches_lowest_id_of_full_row(monkeypatch, local_rows):
    """Real pack kernel (interpreted) + the real exchange in `_dp_sharded_logits`
    pick, per row, the lowest global id of the full-vocab max: what the
    allgather head + `topk_select(k=1, tie="low")` greedy sampler picks."""
    max_rows = 3
    full = _rows_with_ties(DP * max_rows)  # the Σrows x V logits every path sees
    ref = _lowest_max_id(full)
    shards = [full[:, r * VSHARD : (r + 1) * VSHARD].contiguous() for r in range(DP)]
    packs = [eh.lm_head_argmax_pack(shards[r], r * VSHARD) for r in range(DP)]

    monkeypatch.setattr(eh.envs, "ATOM_USE_CUSTOM_ALL_GATHER", False)
    for rank in range(DP):
        monkeypatch.setattr(eh.tgemm, "mm", lambda *a, _r=rank, **k: shards[_r])

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
    monkeypatch.setattr(eh.envs, "ATOM_DP_LM_HEAD_MODE", mode)
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


def _batch(n=3, temps=None, logprobs=False, structured=None, channel=None):
    temps = np.zeros(n, np.float32) if temps is None else np.asarray(temps, np.float32)
    return types.SimpleNamespace(
        temperatures=temps,
        any_logprobs=logprobs,
        structured_requests=structured or {},
        required_channel=channel,
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


def test_step_head_greedy_serves_int32_ids():
    ids = torch.tensor([5.0, 129279.0, 0.0])
    r = _runner(_Head(ids))
    out = r._step_head(ids, torch.zeros(4, 4), _batch(3))
    assert out.dtype == torch.int32 and out.tolist() == [5, 129279, 0]
    assert out.data_ptr() != ids.data_ptr()
    assert r.compute_logits_calls == 0


@pytest.mark.parametrize(
    "kw",
    [
        {"temps": [0.0, 0.7, 0.0]},
        {"logprobs": True},
        {"structured": {0: object()}},
        {"channel": object()},
        None,
    ],
)
def test_step_head_fallback_is_local_logits(kw):
    ids = torch.tensor([5.0, 7.0, 1.0])
    head = _Head(ids)
    r = _runner(head)
    batch = None if kw is None else _batch(3, **kw)
    out = r._step_head(ids, torch.zeros(4, 4), batch)
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


# ----------------------------------------------------------- postprocess ---
def test_postprocess_takes_ids_without_sampler(monkeypatch):
    monkeypatch.setattr(mr, "get_pcp_world_size", lambda: 1)
    monkeypatch.setattr(mr, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mr, "sp_is_enabled", lambda: False)
    monkeypatch.setattr(
        mr,
        "get_forward_context",
        lambda: types.SimpleNamespace(spec_decode_metadata=None),
    )
    monkeypatch.setattr(mr, "get_tp_group", lambda: types.SimpleNamespace(world_size=1))
    seen = {}
    runner = object.__new__(mr.ModelRunner)
    runner.rank = 0
    runner._served = None
    runner.dp_argmax_head = object()

    def sampler(*a, **k):
        raise AssertionError("sampler must not run on ids")

    runner.sampler = sampler

    def prepare(batch, sampled, *a, **k):
        seen["sampled"] = sampled
        return {}, {}

    runner.forward_done_event = types.SimpleNamespace(record=lambda stream=None: None)
    runner.tokenID_processor = types.SimpleNamespace(
        is_deferred_out=False,
        prev_batch=None,
        zero_spec_counts=torch.zeros(128, dtype=torch.int32),
        prepare_sampled_ids=prepare,
    )
    batch = types.SimpleNamespace(
        total_seqs_num=3,
        total_tokens_num=3,
        req_ids=[0, 1, 2],
        return_logprobs=[False] * 3,
        any_logprobs=False,
        logprob_rows=None,
        required_channel=None,
        num_spec_step=0,
    )
    ids = torch.tensor([4, 9, 2, 6], dtype=torch.int32)
    runner.postprocess(
        batch=batch,
        logits=ids,
        temperatures=torch.full((3,), 1e-10),
        top_ks=None,
        top_ps=None,
        all_greedy=True,
        hidden_states=None,
    )
    assert seen["sampled"].dtype == torch.int32
    assert seen["sampled"].tolist() == [4, 9, 2]

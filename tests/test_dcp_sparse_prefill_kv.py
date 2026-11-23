# SPDX-License-Identifier: MIT
"""DCP sparse prefill, "kv" transport: gather the step's KV rows, index them.

Covers ``dcp_sparse_prefill_kv_plan``, ``dcp_gather_sparse_prefill_kv``'s local
half and ``triton_dcp_sparse_prefill_kv_index`` in ``atom/model_ops/dcp_ops.py``,
and the wiring around them: the builder's per-step plan and the attention's
kernel arguments for a "kv" step.

The end-to-end check simulates the DCP group on one device: each rank's cache
is filled where the WRITE side puts its tokens (``dcp_prefill_slot_mapping``,
checked against decode's ``_dcp_round_robin_slot``), the gather concatenates
every rank's plan rows rank-major as the all-gather does, and every candidate
the index kernel emits must read back the KV of the position the top-k named.
A wrong owner, local index or shard base reads another token's row.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

# The CPU job runs everything but the Triton kernels and the memory profile.
gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels and memory stats need a GPU"
)

dcp_ops = aiter_mla = attention_mla = indexer_forward = None
mla_dcp_sparse_prefill_kv_splits = PERSISTENT_WORK_BUFFERS = None

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TOPK = 256
D = 576


def _fake_mla_metadata_info_v1(tokens, max_seqlen_qo, heads, q_dtype, kv_dtype, **kw):
    """The shape of aiter's sizes on a CPU runner, where aiter is a stub: 48
    heads take more tiles a token than 16, 64 or 128 (3x, as on gfx950)."""
    rows = tokens // 8 * (3 if heads == 48 else 1)
    return [
        (2, torch.int32),
        (tokens + 1, torch.int32),
        ((rows, 8), torch.int32),
        (rows + 1, torch.int32),
        ((rows, 2), torch.int32),
        (rows, torch.int32),
    ]


@pytest.fixture(scope="module", autouse=True)
def _modules():
    global dcp_ops, aiter_mla, attention_mla, indexer_forward
    global mla_dcp_sparse_prefill_kv_splits, PERSISTENT_WORK_BUFFERS
    with atom_without_aiter() as load:
        dcp_ops = load("atom.model_ops.dcp_ops")
        aiter_mla = load("atom.model_ops.attentions.aiter_mla")
        attention_mla = load("atom.model_ops.attention_mla")
        indexer_forward = load("atom.model_ops.indexer.indexer_forward")
        mla_dcp_sparse_prefill_kv_splits = load(
            "atom.distributed.dcp_utils"
        ).mla_dcp_sparse_prefill_kv_splits
        PERSISTENT_WORK_BUFFERS = load(
            "atom.model_ops.attentions.sparse_mla_head_groups"
        ).PERSISTENT_WORK_BUFFERS
        if isinstance(aiter_mla.get_mla_metadata_info_v1, MagicMock):
            aiter_mla.get_mla_metadata_info_v1 = _fake_mla_metadata_info_v1
        yield


def decode_writer_slot(block_table_row, pos, rank, world, page, interleave):
    """Where decode writes global position ``pos`` on ``rank``; -1 if another
    rank owns it."""
    stub = SimpleNamespace(
        dcp_world_size=world,
        dcp_rank=rank,
        cp_kv_cache_interleave_size=interleave,
        model_runner=SimpleNamespace(block_size=page),
    )
    return int(
        aiter_mla.AiterMLAMetadataBuilder._dcp_round_robin_slot(
            stub, block_table_row, pos
        )
    )


def _case(context_lens, q_lens, world, page, interleave, seed=0):
    rng = np.random.default_rng(seed)
    vbs = page * world
    tables, next_block = [], 0
    for n in context_lens:
        blocks = -(-n // vbs) + 1  # one spare local block, as a real table may have
        tables.append(list(range(next_block, next_block + blocks)))
        next_block += blocks
    num_slots = next_block * page
    kv = [torch.randn(n, D, device=DEV).to(torch.float8_e4m3fn) for n in context_lens]
    flat_kv = torch.cat(kv)
    caches = []
    for r in range(world):
        slots = dcp_ops.dcp_prefill_slot_mapping(
            tables, [0] * len(context_lens), context_lens, page, world, r, interleave
        )
        # Prefill and decode write a position to the same slot.
        p = 0
        for i, n in enumerate(context_lens):
            for pos in range(n):
                want = decode_writer_slot(tables[i], pos, r, world, page, interleave)
                assert slots[p] == want, (r, i, pos)
                p += 1
        slots = torch.tensor(slots, device=DEV)
        mine = slots >= 0
        cache = torch.zeros(num_slots, 1, D, device=DEV, dtype=torch.float8_e4m3fn)
        cache[slots[mine], 0] = flat_kv[mine]
        caches.append(cache)
    # Query tokens: the last q_lens[i] positions of each request, each with a
    # causal top-k of its visible positions (-1 past them).
    batch_id, topk, counts = [], [], []
    for i, (n, q) in enumerate(zip(context_lens, q_lens)):
        for pos in range(n - q, n):
            visible = rng.permutation(pos + 1)[:TOPK]
            row = np.full(TOPK, -1, dtype=np.int32)
            row[: len(visible)] = visible
            batch_id.append(i)
            topk.append(row)
            counts.append(len(visible))
    indptr = np.zeros(len(counts) + 1, dtype=np.int32)
    np.cumsum(counts, out=indptr[1:])
    return SimpleNamespace(
        tables=tables,
        kv=kv,
        caches=caches,
        batch_id=torch.tensor(batch_id, dtype=torch.int32, device=DEV),
        topk=torch.from_numpy(np.stack(topk)).to(DEV),
        indptr=torch.from_numpy(indptr).to(DEV),
        counts=counts,
    )


def _plan(context_lens, tables, world, page, **kw):
    kw.setdefault("max_tokens", 1 << 20)
    return dcp_ops.dcp_sparse_prefill_kv_plan(
        np.array(context_lens),
        tables,
        world,
        page,
        device=torch.device(DEV),
        **kw,
    )


def _index(c, plan, world, interleave, topk=None):
    out = torch.full((int(c.indptr[-1]),), -7, dtype=torch.int32, device=DEV)
    dcp_ops.triton_dcp_sparse_prefill_kv_index(
        c.indptr,
        c.batch_id,
        c.topk if topk is None else topk,
        plan,
        world,
        out=out,
        NUM_TOPK_TOKENS=TOPK,
        cp_kv_cache_interleave_size=interleave,
    )
    return out


def _compiled(kernel):
    return sum(len(c[0]) for c in kernel.device_caches.values())


@gpu
def test_the_warmup_compiles_what_a_live_kv_step_runs():
    """A "kv" step launches the index kernel that the builder warmed: its
    sizes vary per step but key no new compile."""
    kernel = dcp_ops._dcp_sparse_prefill_kv_index_kernel
    dcp_ops.warm_dcp_sparse_prefill_kv_index(4, 1, TOPK, torch.device(DEV))
    warmed = _compiled(kernel)
    for tokens, reqs, rows_pad, out_rows in [(1, 1, 1, 7), (17, 3, 17, 333)]:
        topk = torch.randint(
            0, 1 << 16, (tokens + 8, TOPK), dtype=torch.int32, device=DEV
        )
        indptr = torch.zeros(tokens + 1, dtype=torch.int32, device=DEV)
        indptr[1:] = torch.randint(0, TOPK, (tokens,), device=DEV).cumsum(0)
        base = torch.arange(reqs + 1, dtype=torch.int32, device=DEV) * rows_pad
        dcp_ops.triton_dcp_sparse_prefill_kv_index(
            indptr,
            torch.randint(0, reqs, (tokens,), dtype=torch.int32, device=DEV),
            topk[:tokens],
            dcp_ops.DcpSparsePrefillKV(base[:0], base, rows_pad),
            4,
            torch.empty(out_rows + 13, dtype=torch.int32, device=DEV)[:out_rows],
            NUM_TOPK_TOKENS=TOPK,
        )
    assert _compiled(kernel) == warmed


@gpu
@pytest.mark.parametrize(
    "world,page,interleave",
    [(4, 16, 1), (8, 16, 1), (4, 16, 4), (2, 1, 1)],
)
def test_every_candidate_reads_the_kv_of_its_position(world, page, interleave):
    context_lens = [300, 37, 129]
    q_lens = [40, 37, 9]
    c = _case(context_lens, q_lens, world, page, interleave)
    plan = _plan(context_lens, c.tables, world, page)
    assert plan is not None
    # The all-gather: every rank's plan rows, rank-major.
    workspace = torch.cat(
        [dcp_ops.dcp_gather_compressed_kv(cache, plan.slot_ids) for cache in c.caches]
    )
    assert workspace.shape[0] == world * plan.rows_pad
    rows = _index(c, plan, world, interleave).long()
    got = workspace[rows].view(torch.uint8)
    starts = c.indptr.tolist()
    for t, n in enumerate(c.counts):
        req = int(c.batch_id[t])
        pos = c.topk[t, :n].long()
        want = c.kv[req][pos].view(torch.uint8)
        assert torch.equal(got[starts[t] : starts[t] + n], want), f"token {t}"


@gpu
def test_a_position_past_the_request_reads_row_zero():
    """A position past the request's padded rows would land in the next
    request's shard; it is written as the dcp=1 kernel writes an invalid one."""
    context_lens, q_lens = [37, 64], [1, 1]
    c = _case(context_lens, q_lens, 4, 16, 1)
    plan = _plan(context_lens, c.tables, 4, 16)
    topk = c.topk.clone()
    topk[0, 0] = 64  # request 0 has 16 rows a rank: local index 16 is past them
    rows = _index(c, plan, 4, 1, topk=topk)
    assert int(rows[0]) == 0


def test_plan_rows_and_bases():
    """Each request pads its local rows to whole local blocks, the same count
    on every rank, and the bases are their running sum."""
    tables = [list(range(3)), list(range(3, 5))]
    plan = _plan([130, 64], tables, 4, 16)
    # ceil(130 / 64) = 3 local blocks, ceil(64 / 64) = 1.
    assert plan.rows_pad == 3 * 16 + 1 * 16
    assert plan.shard_base.tolist() == [0, 48, 64]
    assert plan.slot_ids.dtype == torch.int32
    assert plan.slot_ids.tolist() == list(range(64))


@pytest.mark.parametrize(
    "context_lens,max_tokens,taken",
    [
        ([1024], 1 << 20, True),
        ([65536], 1 << 15, False),  # over the token cap
    ],
)
def test_plan_takes_kv_within_its_cap(context_lens, max_tokens, taken):
    tables = [list(range(-(-n // 64) + 1)) for n in context_lens]
    plan = _plan(context_lens, tables, 4, 16, max_tokens=max_tokens)
    assert (plan is not None) == taken


def test_plan_refuses_a_block_table_too_short():
    with pytest.raises(ValueError):
        _plan([200], [[0]], 4, 16)  # needs ceil(200 / 64) = 4 blocks


# ------------------------------------------------------------------ wiring
def _builder(monkeypatch, plan, nonps):
    """The builder fields `_dcp_sparse_prefill_step` reads; the plan and the
    aiter metadata call are stubbed, recording the step's total KV the plan
    is given and the heads the work plan is built for."""
    seen = []
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b.dcp_world_size = 4
    b.dcp_sparse_prefill_nonps = nonps
    b.padded_num_attention_heads = 16
    b.sparse_prefill_num_heads = 64
    b.block_size = 1
    b.dtype_q = b.dtype_kv = torch.float8_e4m3fn
    b.model_runner = SimpleNamespace(
        forward_vars={f"sparse_prefill_{n}": n for n in PERSISTENT_WORK_BUFFERS}
    )
    b._sparse_prefill_work_views = lambda prefix, heads: {
        n: n for n in PERSISTENT_WORK_BUFFERS
    }
    b._dcp_sparse_prefill_kv_plan = lambda batch, total_kv, num_tokens: (
        seen.append(("plan", total_kv, num_tokens)) or plan
    )
    monkeypatch.setattr(
        aiter_mla, "get_mla_metadata_v1", lambda *a, **k: seen.append(a[3])
    )
    md = SimpleNamespace(
        sparse_cu_seqlens_q=None,
        sparse_kv_indptr=None,
        sparse_kv_last_page_lens=None,
        total_kv=7,
    )
    return b, md, seen


@pytest.mark.parametrize("nonps", [True, False])
def test_a_kv_step_plans_this_ranks_heads(monkeypatch, nonps):
    """A "kv" step runs the dcp=1 kernel: its work plan is for the per-rank
    padded heads, built even when the q path's non-persistent kernel reads none."""
    plan = object()
    b, md, seen = _builder(monkeypatch, plan, nonps)
    b._dcp_sparse_prefill_step(None, md, "", 3)
    assert md.dcp_sparse_prefill_kv is plan
    assert seen == [("plan", 7, 3), 16]
    assert md.sparse_prefill_work_meta_data == "work_meta_data"


@pytest.mark.parametrize("nonps,planned", [(True, []), (False, [64])])
def test_a_q_step_keeps_its_plan(monkeypatch, nonps, planned):
    b, md, seen = _builder(monkeypatch, None, nonps)
    b._dcp_sparse_prefill_step(None, md, "", 3)
    assert md.dcp_sparse_prefill_kv is None
    assert seen == [("plan", 7, 3)] + planned


def test_a_kv_step_plans_into_its_prefix_buffers(monkeypatch):
    """A TBO ubatch's "kv" step plans into that ubatch's buffers, over the
    step's own last-page lengths."""
    b, md, _ = _builder(monkeypatch, object(), True)
    seen = []
    b._sparse_prefill_work_views = lambda prefix, heads: seen.append(prefix) or {
        n: n for n in PERSISTENT_WORK_BUFFERS
    }
    monkeypatch.setattr(
        aiter_mla, "get_mla_metadata_v1", lambda *a, **k: seen.append(a[2])
    )
    md.sparse_kv_last_page_lens = object()
    b._dcp_sparse_prefill_step(None, md, "ub0_", 3)
    assert seen == ["ub0_", md.sparse_kv_last_page_lens]


def _finisher(monkeypatch, mode, prefix=""):
    """`_finish_prefill_batch` at DCP=1 with no cached prefix; records which
    planner it calls and with what heads."""
    seen = []
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b._dcp_sparse_prefill_mode = mode
    b.dcp_world_size = 1
    b.sparse_prefill_num_heads = 64
    b.attn_prefill_chunk_size = 0
    b.kv_shuffled = False
    b.model_runner = SimpleNamespace(forward_vars={})
    staged = SimpleNamespace(index_rows=SimpleNamespace(passes=None), pcp_split=False)
    staged.total_kv, staged.sparse_counts = None, np.ones(3)
    b._sparse_prefill_staged = {prefix: staged}
    b._sparse_prefill_work_plan = lambda md, prefix, heads, *a, **k: seen.append(
        ("work_plan", heads, prefix, a[0])
    )
    b._dcp_sparse_prefill_step = lambda batch, md, prefix, n: seen.append(
        ("kv_step", prefix, n)
    )
    b._finish_prefill_index_rows = lambda *a: None
    b._prefill_reads_token_csr = lambda md: False
    monkeypatch.setattr(
        aiter_mla.CommonAttentionBuilder, "_finish_prefill_batch", lambda *a: None
    )
    md = SimpleNamespace(sparse_kv_last_page_lens=object(), has_cached=False)
    return b, md, seen


@pytest.mark.parametrize(
    "mode,want",
    [("q", [("work_plan", 64)]), ("auto", ["kv_step"]), ("kv", ["kv_step"])],
)
@pytest.mark.parametrize("prefix", ["", "ub0_"])
def test_a_q_builder_skips_the_kv_step(monkeypatch, mode, want, prefix):
    """The default transport (and DCP=1) plans as before the "kv" transport,
    without its per-step host work; a TBO ubatch's step keeps its prefix."""
    b, md, seen = _finisher(monkeypatch, mode, prefix)
    b._finish_prefill_batch(SimpleNamespace(scheduled_bs=1), md, prefix)
    last_page = md.sparse_kv_last_page_lens
    assert seen == [
        ("kv_step", prefix, 3) if w == "kv_step" else (*w, prefix, last_page)
        for w in want
    ]


def _work_sizer(mode, local, gathered):
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b._dcp_sparse_prefill_mode = mode
    b.padded_num_attention_heads = local
    b.sparse_prefill_num_heads = gathered
    b.dtype_q = b.dtype_kv = torch.float8_e4m3fn
    return b


def _numel(size):
    return int(np.prod(size))


@pytest.mark.parametrize("mode", ["auto", "kv"])
def test_the_work_buffers_fit_a_kv_steps_plan(mode):
    """Kimi-K3 TP2/DCP2: 48 heads a rank, 128 gathered. A "kv" step plans the
    48, which can take more tiles a token than the 128 the buffers were sized
    for; they must fit both widths."""
    tokens = 4096
    got = _work_sizer(mode, 48, 128)._sparse_prefill_work_info(tokens)
    for heads in (48, 128):
        want = aiter_mla.get_mla_metadata_info_v1(
            tokens,
            1,
            heads,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            is_sparse=True,
            fast_mode=True,
        )
        for (g, _), (w, _) in zip(got, want):
            assert _numel(g) >= _numel(w)


def _viewer(prefix_tokens, local=48, gathered=128):
    b = _work_sizer("auto", local, gathered)
    b._sparse_prefill_work_view_cache = {}
    b.max_num_batched_tokens = prefix_tokens[""]
    b._ubatch_tokens = lambda: prefix_tokens["ub0_"]
    var = {}
    for prefix, tokens in prefix_tokens.items():
        for name, (size, dtype) in zip(
            aiter_mla.PERSISTENT_WORK_BUFFERS, b._sparse_prefill_work_info(tokens)
        ):
            var[f"{prefix}sparse_prefill_{name}"] = torch.empty(size, dtype=dtype)
    b.model_runner = SimpleNamespace(forward_vars=var)
    return b


@pytest.mark.parametrize("prefix", ["", "ub0_"])
@pytest.mark.parametrize("heads", [128, 48])
def test_a_step_plans_into_its_own_widths_buffers(prefix, heads):
    """aiter sizes its reduce grid and fp32 partials by the buffers a plan is
    given, so each step sees the buffers its width alone would have, not the
    max over both."""
    tokens = {"": 4096, "ub0_": 2048}
    b = _viewer(tokens)
    work = b._sparse_prefill_work_views(prefix, heads)
    want = aiter_mla.get_mla_metadata_info_v1(
        tokens[prefix],
        1,
        heads,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        is_sparse=True,
        fast_mode=True,
    )
    for name, (size, _) in zip(aiter_mla.PERSISTENT_WORK_BUFFERS, want):
        assert tuple(work[name].shape) == tuple(np.atleast_1d(size))


def _sized_for(buffers, tokens, heads):
    want = aiter_mla.get_mla_metadata_info_v1(
        tokens,
        1,
        heads,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        is_sparse=True,
        fast_mode=True,
    )
    return all(
        tuple(buffers(name).shape) == tuple(np.atleast_1d(size))
        for name, (size, _) in zip(aiter_mla.PERSISTENT_WORK_BUFFERS, want)
    )


@pytest.mark.parametrize("local,gathered", [(48, 128), (16, 48)])
@pytest.mark.parametrize("kv", [False, True])
def test_a_step_publishes_its_widths_buffers(monkeypatch, kv, local, gathered):
    """The attention reads the work buffers off the metadata, and sizes its
    fp32 partials and reduce grid by them. 48 heads take more tiles than 128
    or 16, so each width pair leaves one of its steps on a slice."""
    b = _viewer({"": 4096, "ub0_": 2048}, local, gathered)
    b.dcp_world_size = 2
    b.dcp_sparse_prefill_nonps = False
    b.block_size = 1
    b._dcp_sparse_prefill_kv_plan = lambda batch, total_kv, n: (
        object() if kv else None
    )
    monkeypatch.setattr(aiter_mla, "get_mla_metadata_v1", lambda *a, **k: None)
    md = SimpleNamespace(
        sparse_cu_seqlens_q=None,
        sparse_kv_indptr=None,
        sparse_kv_last_page_lens=None,
        total_kv=0,
    )
    b._dcp_sparse_prefill_step(None, md, "", 3)
    assert _sized_for(
        lambda name: getattr(md, f"sparse_prefill_{name}"),
        4096,
        local if kv else gathered,
    )


def test_steps_alternating_widths_each_see_their_own():
    tokens = {"": 4096, "ub0_": 2048}
    b = _viewer(tokens)
    for prefix, heads in [("", 48), ("", 128), ("ub0_", 48), ("", 48), ("ub0_", 128)]:
        work = b._sparse_prefill_work_views(prefix, heads)
        assert _sized_for(work.__getitem__, tokens[prefix], heads)


def test_shared_views_are_read_only():
    """Every step gets the same cached views, so a consumer must not edit them."""
    work = _viewer({"": 4096, "ub0_": 2048})._sparse_prefill_work_views("", 48)
    with pytest.raises(TypeError):
        work["work_indptr"] = None


def test_widths_of_one_size_see_the_whole_buffers():
    """GLM-5.2's 16 heads a rank take the tiles its 64 gathered do."""
    b = _viewer({"": 4096, "ub0_": 2048}, local=16, gathered=64)
    var = b.model_runner.forward_vars
    for heads in (16, 64):
        work = b._sparse_prefill_work_views("", heads)
        for name in aiter_mla.PERSISTENT_WORK_BUFFERS:
            base = var[f"sparse_prefill_{name}"]
            assert work[name].shape == base.shape
            assert work[name].data_ptr() == base.data_ptr()


def test_a_q_builder_sizes_the_work_buffers_as_before():
    tokens = 4096
    got = _work_sizer("q", 48, 128)._sparse_prefill_work_info(tokens)
    want = aiter_mla.get_mla_metadata_info_v1(
        tokens,
        1,
        128,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        is_sparse=True,
        fast_mode=True,
    )
    assert got == want


def _resolver(
    monkeypatch, env, *, sparse=True, dcp=4, page=1, plugin=False, pcp=False, heads=16
):
    monkeypatch.setenv("ATOM_DCP_SPARSE_PREFILL", env)
    monkeypatch.setattr("atom.plugin.is_plugin_mode", lambda: plugin)
    monkeypatch.setattr("atom.distributed.pcp_utils.pcp_is_enabled", lambda: pcp)
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b.is_sparse, b.dcp_world_size, b.block_size = sparse, dcp, page
    b.num_attention_heads = heads
    return b


@pytest.mark.parametrize(
    "env,kw,dtype,want",
    [
        ("auto", {}, "fp8", "auto"),
        ("kv", {}, "fp8_e4m3", "kv"),
        ("q", {}, "fp8", "q"),
        ("auto", {}, "bf16", "q"),
        ("kv", {}, "bf16", "q"),
        ("auto", {"page": 64}, "fp8", "q"),
        ("auto", {"plugin": True}, "fp8", "q"),
        ("kv", {"pcp": True}, "fp8", "q"),
        ("kv", {"dcp": 1}, "fp8", "q"),
        ("kv", {"sparse": False}, "fp8", "q"),
    ],
)
def test_the_transport_resolves_to_what_the_builder_can_run(
    monkeypatch, env, kw, dtype, want
):
    b = _resolver(monkeypatch, env, **kw)
    assert b._resolve_dcp_sparse_prefill_mode(dtype) == want


def test_an_unknown_transport_is_refused(monkeypatch):
    b = _resolver(monkeypatch, "KV")
    with pytest.raises(ValueError):
        b._resolve_dcp_sparse_prefill_mode("fp8")


@pytest.mark.parametrize("interleave", [1, 4])
@pytest.mark.parametrize("mode,warmed", [("auto", True), ("kv", True), ("q", False)])
def test_a_builder_that_can_take_kv_warms_the_index_kernel(
    monkeypatch, mode, warmed, interleave
):
    calls = []
    monkeypatch.setattr(
        dcp_ops, "warm_dcp_sparse_prefill_kv_index", lambda *a: calls.append(a)
    )
    b = _resolver(monkeypatch, mode)
    b._dcp_sparse_prefill_mode = mode
    b.cp_kv_cache_interleave_size, b.index_topk, b.device = interleave, TOPK, "dev"
    b._warm_dcp_sparse_prefill_kv()
    assert calls == ([(4, interleave, TOPK, "dev")] if warmed else [])


@pytest.mark.parametrize("kw", [{"sparse": False}, {"dcp": 1}])
def test_a_builder_without_a_dcp_sparse_prefill_ignores_the_value(monkeypatch, kw):
    b = _resolver(monkeypatch, "KV", **kw)
    assert b._resolve_dcp_sparse_prefill_mode("fp8") == "q"


def _planner(monkeypatch, mode, max_tokens=1 << 20, dcp=4, heads=16):
    """The builder fields `_dcp_sparse_prefill_kv_plan` reads, set by the real
    resolver, with the real plan; records its warnings."""
    monkeypatch.setenv("ATOM_DCP_SPARSE_PREFILL_KV_MAX_TOKENS", str(max_tokens))
    b = _resolver(monkeypatch, mode, dcp=dcp, heads=heads)
    assert b._resolve_dcp_sparse_prefill_mode("fp8") == mode
    b._dcp_sparse_prefill_mode = mode
    warned = []
    monkeypatch.setattr(aiter_mla.logger, "warning", lambda *a: warned.append(a))
    b._dcp_sparse_prefill_kv_fallback_warned = False
    b.model_runner = SimpleNamespace(block_size=16)
    b.device = torch.device(DEV)
    return b, warned


def _batch(context_lens, q_lens, tables=None, dummy=False):
    if tables is None:
        tables = [list(range(-(-n // 64) + 1)) for n in context_lens]
    return SimpleNamespace(
        context_lens=np.array(context_lens),
        seqlens_q=np.array(q_lens),
        block_tables=tables,
        is_dummy_run=dummy,
    )


def _take(b, batch):
    """The builder's plan for ``batch``, given its staged total KV and tokens."""
    return b._dcp_sparse_prefill_kv_plan(
        batch, int(batch.context_lens.sum()), int(batch.seqlens_q.sum())
    )


@pytest.mark.parametrize(
    "mode,batch,taken",
    [
        ("auto", _batch([4096], [4096]), True),  # cold
        ("auto", _batch([131072], [64]), True),  # 128K hit + 64: "kv" measured faster
        ("auto", _batch([524288], [64]), False),  # 512K hit + 64: "q" faster
        ("auto", _batch([524288], [16384]), True),  # 512K hit + 16K: "kv" faster
        ("kv", _batch([524288], [64]), True),  # forced
        ("auto", _batch([4096], [4096], dummy=True), False),
        ("kv", _batch([4096], [4096], tables=[]), False),  # padding batch
    ],
)
def test_a_step_takes_kv_by_its_mode_and_batch(monkeypatch, mode, batch, taken):
    b, warned = _planner(monkeypatch, mode)
    assert (_take(b, batch) is not None) == taken
    assert warned == []


@pytest.mark.parametrize("over,taken", [(0, True), (1, False)])
def test_auto_takes_kv_up_to_its_context_bound(monkeypatch, over, taken):
    """Two requests sharing the bound exactly, then one row past it."""
    b, _ = _planner(monkeypatch, "auto")
    bound = dcp_ops.dcp_sparse_prefill_kv_context_bound(16, 128)
    assert (
        _take(b, _batch([bound - 1024, 1024 + over], [64, 64])) is not None
    ) == taken


@pytest.mark.parametrize(
    "dcp,heads,taken", [(4, 16, True), (8, 8, False), (4, 8, False)]
)
def test_the_context_bound_is_tighter_with_fewer_heads(monkeypatch, dcp, heads, taken):
    """512K + 7K new tokens: under 200K + 52 per token at 16 heads a rank,
    over 40 at 8 (TP8), whatever the DCP size."""
    b, _ = _planner(monkeypatch, "auto", dcp=dcp, heads=heads)
    plan = _take(b, _batch([524288], [7168]))
    assert (plan is not None) == taken


def test_a_256k_hit_with_1k_new_tokens_stays_on_q(monkeypatch):
    """Measured on GLM-5.2-FP8 tp4/dcp4: q 0.428 s, kv 0.449 s."""
    b, _ = _planner(monkeypatch, "auto")
    assert _take(b, _batch([262144], [1024])) is None


def test_a_plan_at_the_cap_uploads_under_the_pageable_cliff(monkeypatch):
    """The step's slots are expanded on the GPU from its blocks, so even a
    plan at the default cap uploads less than a pageable copy's cliff."""
    from atom.utils import _PAGEABLE_H2D_LIMIT

    sizes = []
    real = dcp_ops.upload_numpy
    monkeypatch.setattr(
        dcp_ops, "upload_numpy", lambda a, d: sizes.append(a.nbytes) or real(a, d)
    )
    n = 1 << 20
    plan = _plan([n], [list(range(-(-n // 64) + 1))], 4, 16)
    assert plan.rows_pad * 4 == n
    assert plan.slot_ids.tolist()[:20] == list(range(20))
    assert max(sizes) <= _PAGEABLE_H2D_LIMIT


@pytest.mark.parametrize("mode,warns", [("kv", 1), ("auto", 0)])
def test_only_a_forced_kv_warns_over_the_cap(monkeypatch, mode, warns):
    b, warned = _planner(monkeypatch, mode, max_tokens=1 << 12)
    for _ in range(2):
        assert _take(b, _batch([65536], [65536])) is None
    assert len(warned) == warns


def _kv_attention():
    """An `MLAAttention` with only the fields a "kv" `_forward_prefill_mla`
    reads, set the way a dcp4 fp8 sparse layer with the non-persistent q-path
    kernel has them."""
    a = attention_mla.MLAAttention.__new__(attention_mla.MLAAttention)
    a.is_sparse_mla = True
    a.dcp_world_size = 4
    a.use_seg_mla = False
    a.head_pad = 0
    a.num_heads = 16
    a.kv_lora_rank, a.qk_rope_head_dim = 512, 64
    a.dtype = torch.bfloat16
    a.kv_cache_dtype = "fp8"
    a.scale = 1.0
    a._q_scale = a._k_scale = torch.ones(1)
    a.dcp_sparse_prefill_persistent = False
    a.sparse_dcp_metadata_rebuild = True
    a.dcp_persistent_supported = True
    a.dcp_sparse_prefill_nonps = True
    a.sparse_prefill_head_groups = None
    a.owns_sparse_indexer = True
    a._v_up_proj_and_o_proj = lambda o: o
    return a


def test_a_kv_step_runs_the_dcp1_kernel_arguments(monkeypatch):
    """Global indptr, the dcp=1 split count and the step's work plan -- not
    the per-rank compacted indptr, the DCP split count or no plan."""
    seen = {}

    def fake_mla_decode_fwd(q, kv, o, cu, indptr, indices, lpl, max_q, **kw):
        seen.update(q=q, indptr=indptr, **kw)
        return None, None

    monkeypatch.setattr(attention_mla, "mla_decode_fwd", fake_mla_decode_fwd)
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(context=SimpleNamespace(scheduled_bs=1)),
    )
    n = 3
    md = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, n], dtype=torch.int32),
        kv_indptr=None,
        kv_indices=None,
        kv_last_page_lens=None,
        max_seqlen_q=n,
        sparse_cu_seqlens_q=torch.arange(n + 1, dtype=torch.int32),
        sparse_kv_indptr=torch.arange(n + 1, dtype=torch.int32),
        dcp_sparse_kv_indptr=torch.zeros(n + 1, dtype=torch.int32),
        sparse_kv_indices=torch.zeros(n, dtype=torch.int32),
        sparse_kv_last_page_lens=torch.ones(n, dtype=torch.int32),
        **{f"sparse_prefill_{w}": w for w in PERSISTENT_WORK_BUFFERS},
    )
    q = torch.zeros(n, 16, 576, dtype=torch.float8_e4m3fn)
    kv = torch.zeros(8, 1, 576, dtype=torch.float8_e4m3fn)
    attention_mla.MLAAttention._forward_prefill_mla(
        _kv_attention(), q, kv, md, dcp_kv=True
    )
    assert seen["indptr"] is md.sparse_kv_indptr
    assert seen["q"].shape[1] == 16
    assert seen["num_kv_splits"] == mla_dcp_sparse_prefill_kv_splits(False, 1)
    assert {w: seen[w] for w in PERSISTENT_WORK_BUFFERS} == {
        w: w for w in PERSISTENT_WORK_BUFFERS
    }


class _Stop(Exception):
    pass


@pytest.mark.parametrize("kv", [True, False])
def test_a_sparse_prefill_takes_the_transport_its_plan_names(monkeypatch, kv):
    """A step with a "kv" plan gathers the KV and attends as dcp=1; one
    without gathers the query heads and keeps the LSE for the merge."""
    calls = []
    monkeypatch.setattr(
        attention_mla,
        "dcp_all_gather_prefill_query_heads",
        lambda group, q, heads: calls.append("gather q") or q,
    )
    gathered = torch.empty(1)
    monkeypatch.setattr(
        attention_mla,
        "dcp_gather_sparse_prefill_kv",
        lambda *args: calls.append(("gather kv", args)) or gathered,
    )

    def forward_prefill_mla(self, q, kv_cache, md, **kw):
        calls.append((q, kv_cache, kw))
        raise _Stop

    monkeypatch.setattr(
        attention_mla.MLAAttention, "_forward_prefill_mla", forward_prefill_mla
    )
    a = _kv_attention()
    a.dcp_group = object()
    a.dcp_sparse_prefill_num_heads = 64
    md = SimpleNamespace(dcp_sparse_prefill_kv=object() if kv else None)
    q, cache = torch.zeros(2, 16, D), torch.empty(1)
    with pytest.raises(_Stop):
        attention_mla.MLAAttention._dcp_sparse_prefill(a, q, cache, md)
    if kv:
        # The attention reads the gathered rows, which the indexer numbered.
        assert calls[0] == ("gather kv", (a.dcp_group, cache, md.dcp_sparse_prefill_kv))
        assert calls[1][0] is q and calls[1][1] is gathered
        assert calls[1][2] == {"dcp_kv": True}
    else:
        assert calls[0] == "gather q"
        assert calls[1][1] is cache
        assert calls[1][2].get("return_lse") and "dcp_kv" not in calls[1][2]


@pytest.mark.parametrize(
    "kv,env,group", [(False, "1", True), (True, "1", False), (False, "0", False)]
)
def test_a_kv_prefill_projects_only_this_ranks_heads(monkeypatch, kv, env, group):
    """ATOM_DCP_PREFILL_QREP projects the group's heads for a "q" step only."""
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(
                run_sparse=True, dcp_sparse_prefill_kv=object() if kv else None
            ),
            context=SimpleNamespace(is_dummy_run=False, is_prefill=True),
            kv_cache_data={"layer_0": SimpleNamespace(k_cache=None)},
        ),
    )
    seen = {}

    def q_proj_and_k_up_proj(self, x, x_scale=None, group=False):
        seen["group"] = group
        raise _Stop

    monkeypatch.setattr(
        attention_mla.MLAAttention, "_q_proj_and_k_up_proj", q_proj_and_k_up_proj
    )
    a = _kv_attention()
    a.layer_num = 0
    a.qrep_enabled = True
    a.dcp_prefill_qrep = env == "1"
    with pytest.raises(_Stop):
        attention_mla.MLAAttention.forward_impl(a, torch.zeros(2, 8), None, None)
    assert seen == {"group": group}


@pytest.mark.parametrize("kv", [True, False])
def test_a_kv_prefill_indexes_rows_of_the_gathered_kv(monkeypatch, kv):
    """A "kv" step's top-k goes to rows of the gathered KV, not through the
    owned-slot filter."""
    f = indexer_forward
    called = []
    for name in (
        "triton_dcp_sparse_prefill_kv_index",
        "triton_filter_and_convert_dcp_index_prefill",
        "triton_convert_req_index_to_global_index_dsa_prefill",
    ):
        monkeypatch.setattr(
            f, name, lambda *a, _n=name, **kw: called.append((_n, a, kw))
        )
    monkeypatch.setattr(f, "get_dcp_world_size", lambda: 4)
    monkeypatch.setattr(f, "get_dcp_rank", lambda: 0)
    monkeypatch.setattr(f, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(f, "indexer_k_quant_and_cache", lambda *a, **kw: None)
    monkeypatch.setattr(f, "_index_cache", lambda *a: None)
    topk = object()
    monkeypatch.setattr(f, "score_topk", lambda *a, out, **kw: topk)
    indexer = SimpleNamespace(
        _indexer_fp4=False, use_nope_cache_fusion=False, _index_cache_layout=None
    )
    monkeypatch.setattr(
        f,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            kv_cache_block_size=16,
            dcp_config=SimpleNamespace(interleave_size=2),
            compilation_config=SimpleNamespace(static_forward_context={"idx": indexer}),
        ),
    )
    md = SimpleNamespace(
        slot_mapping=None,
        run_sparse=True,
        index_rows=SimpleNamespace(out_rows=2),
        dcp_sparse_prefill_kv=object() if kv else None,
        **{
            n: object()
            for n in (
                "sparse_kv_indptr",
                "batch_id_per_q_token",
                "sparse_kv_indices",
                "block_tables",
                "dcp_sparse_kv_indptr",
                "dcp_owned_counts",
            )
        },
    )
    monkeypatch.setattr(
        f,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=md,
            context=SimpleNamespace(is_dummy_run=False, is_prefill=True),
        ),
    )
    args = dict.fromkeys(inspect.signature(f.sparse_attn_indexer).parameters)
    args.update(
        hidden_states=torch.zeros(2, 8),
        k_cache_prefix="idx.k_cache",
        kv_cache=torch.zeros(1, 16, 4),
        topk_tokens=TOPK,
        use_qk_rope_cache_fusion=False,
    )
    f.sparse_attn_indexer(**args)
    (name, a, kw), *rest = called
    assert rest == []
    if kv:
        assert name == "triton_dcp_sparse_prefill_kv_index"
        assert a == (
            md.sparse_kv_indptr,
            md.batch_id_per_q_token,
            topk,
            md.dcp_sparse_prefill_kv,
            4,
        )
        assert kw == {
            "out": md.sparse_kv_indices,
            "NUM_TOPK_TOKENS": TOPK,
            "cp_kv_cache_interleave_size": 2,
        }
    else:
        assert name == "triton_filter_and_convert_dcp_index_prefill"
        assert kw["out"] is md.sparse_kv_indices


@pytest.mark.parametrize(
    "env,kw,cap,ok",
    [
        ("auto", {}, "-1", False),
        ("kv", {}, "-1", False),
        ("kv", {}, "0", True),
        ("auto", {}, "4096", True),
        # Nothing reads the cap: a "q" builder starts.
        ("q", {}, "-1", True),
        ("auto", {"dcp": 1}, "-1", True),
        ("auto", {"sparse": False}, "-1", True),
        ("auto", {"page": 64}, "-1", True),
    ],
)
def test_a_negative_kv_cap_is_refused_where_it_is_read(monkeypatch, env, kw, cap, ok):
    monkeypatch.setenv("ATOM_DCP_SPARSE_PREFILL_KV_MAX_TOKENS", cap)
    b = _resolver(monkeypatch, env, **kw)
    if ok:
        if b._resolve_dcp_sparse_prefill_mode("fp8") != "q":
            assert b._dcp_sparse_prefill_kv_max_tokens == int(cap)
    else:
        with pytest.raises(ValueError, match=">= 0"):
            b._resolve_dcp_sparse_prefill_mode("fp8")


# ------------------------------------------------------------- memory profile
def test_the_kv_scratch_grows_with_the_cap_and_the_tokens():
    args = {"local_heads": 16, "padded_heads": 16, "kv_lora_rank": 512, "rope_dim": 64}
    f = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes
    base = f(1024, kv_itemsize=1, dcp_world_size=4, kv_max_rows=1 << 18, **args)
    assert f(1024, kv_itemsize=1, dcp_world_size=4, kv_max_rows=1 << 19, **args) > base
    assert f(4096, kv_itemsize=1, dcp_world_size=4, kv_max_rows=1 << 18, **args) > base
    # at least the gathered KV itself, 576 B a row
    assert base >= (1 << 18) * 576


def test_the_kv_scratch_counts_the_padded_query_copy():
    """TP8 GLM-5.2: 8 heads a rank padded to 16 copies the query."""
    f = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes
    args = {
        "kv_lora_rank": 512,
        "rope_dim": 64,
        "kv_itemsize": 1,
        "dcp_world_size": 8,
        "kv_max_rows": 1 << 18,
    }
    unpadded = f(4096, local_heads=16, padded_heads=16, **args)
    padded = f(4096, local_heads=8, padded_heads=16, **args)
    assert padded - unpadded == 4096 * 16 * 576


def test_a_forced_kv_profiles_the_cap():
    cap = 1 << 20
    assert (
        dcp_ops.dcp_sparse_prefill_kv_profile_rows("kv", 64, 16, 256, 16, 4, cap) == cap
    )


@pytest.mark.parametrize("tokens", [1, 64, 2048, 16384])
@pytest.mark.parametrize("heads", [8, 16])
@pytest.mark.parametrize("many", [False, True])
def test_auto_profiles_every_row_it_can_gather(tokens, heads, many):
    """The rows "auto" admits for a step of ``tokens`` new tokens fit what the
    profile holds: the context bound filled to the last row, by as many
    requests as the step can have, each a row past whole local blocks."""
    cap, dcp, block, max_num_seqs = 1 << 20, 4, 16, 256
    n = min(tokens, max_num_seqs) if many else 1
    bound = (
        dcp_ops.DCP_SPARSE_PREFILL_KV_CONTEXT_BASE
        + dcp_ops.dcp_sparse_prefill_kv_context_per_new_token(heads) * tokens
    )
    stride = block * dcp
    context = np.full(n, (bound // n - 1) // stride * stride + 1, dtype=np.int64)
    context[-1] += bound - context.sum()
    tables = [np.arange(cap // block, dtype=np.int32)] * n
    plan = dcp_ops.dcp_sparse_prefill_kv_plan(context, tables, dcp, block, cap, DEV)
    held = dcp_ops.dcp_sparse_prefill_kv_profile_rows(
        "auto", tokens, heads, max_num_seqs, block, dcp, cap
    )
    rows = dcp_ops.dcp_sparse_prefill_kv_rows(context, dcp, block) * dcp
    assert (plan is not None) == (rows <= cap)
    assert min(rows, cap) <= held <= cap


def _dummy_attention(
    monkeypatch, env, kv_cache_dtype, is_prefill, cap=1 << 14, batch_tokens=5
):
    """A dcp4 sparse `MLAAttention` reaching `forward_impl`'s dummy branch."""
    monkeypatch.setenv("ATOM_DCP_SPARSE_PREFILL", env)
    monkeypatch.setenv("ATOM_DCP_SPARSE_PREFILL_KV_MAX_TOKENS", str(cap))
    monkeypatch.setattr("atom.plugin.is_plugin_mode", lambda: False)
    monkeypatch.setattr("atom.distributed.pcp_utils.pcp_is_enabled", lambda: False)
    monkeypatch.setattr(
        attention_mla,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(run_sparse=True),
            context=SimpleNamespace(is_dummy_run=True, is_prefill=is_prefill),
        ),
    )
    monkeypatch.setattr(
        attention_mla,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            hf_config=SimpleNamespace(hidden_size=8),
            torch_dtype=torch.bfloat16,
            max_num_seqs=256,
            max_num_batched_tokens=batch_tokens,
            kv_cache_block_size=16,
        ),
    )
    a = _kv_attention()
    a.kv_cache_dtype = kv_cache_dtype
    a.dcp_sparse_prefill_num_heads = 64
    a.padded_num_heads = 16
    return a


def _profiled_peak(a, num_tokens):
    """Peak allocation a dummy forward adds, as the memory profile sees it."""
    q = torch.zeros(num_tokens, 16, 576, device=DEV)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    attention_mla.MLAAttention.forward_impl(a, q, None, None)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - before


@gpu
@pytest.mark.parametrize(
    "env,dtype,held",
    [
        ("auto", "fp8", True),
        ("kv", "fp8", True),
        ("q", "fp8", False),
        ("auto", "bf16", False),
    ],
)
def test_a_dummy_prefill_profiles_the_kv_scratch(monkeypatch, env, dtype, held):
    """The memory profile's peak includes the gathered KV at the cap exactly
    when a step can take "kv"."""
    cap = 1 << 16
    a = _dummy_attention(monkeypatch, env, dtype, is_prefill=True, cap=cap)
    kv = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(5, 16, 16, 512, 64, 1, 4, cap)
    assert (_profiled_peak(a, 5) >= kv) == held


@gpu
def test_a_dummy_prefill_under_auto_holds_less_than_the_cap(monkeypatch):
    """A 5-token profile under "auto" can gather ~200K rows, not 2^20."""
    cap = 1 << 20
    a = _dummy_attention(monkeypatch, "auto", "fp8", is_prefill=True, cap=cap)
    rows = dcp_ops.dcp_sparse_prefill_kv_profile_rows("auto", 5, 16, 256, 16, 4, cap)
    held = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(5, 16, 16, 512, 64, 1, 4, rows)
    at_cap = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(5, 16, 16, 512, 64, 1, 4, cap)
    assert held <= _profiled_peak(a, 5) < at_cap


@gpu
def test_a_short_dummy_prefill_holds_the_rows_of_a_full_batch(monkeypatch):
    """A dummy prefill shorter than the batch budget still holds what a step
    at the budget adds: its gathered rows and its per-token scratch."""
    cap, budget = 1 << 20, 8192
    a = _dummy_attention(
        monkeypatch, "auto", "fp8", is_prefill=True, cap=cap, batch_tokens=budget
    )
    rows = dcp_ops.dcp_sparse_prefill_kv_profile_rows(
        "auto", budget, 16, 256, 16, 4, cap
    )
    held = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(
        budget, 16, 16, 512, 64, 1, 4, rows
    )
    assert _profiled_peak(a, 5) >= held


@gpu
def test_a_dummy_decode_profiles_no_kv_scratch(monkeypatch):
    cap = 1 << 16
    a = _dummy_attention(monkeypatch, "auto", "fp8", is_prefill=False, cap=cap)
    kv = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(5, 16, 16, 512, 64, 1, 4, cap)
    assert _profiled_peak(a, 5) < kv


def _held_bytes(monkeypatch, a, num_tokens):
    """The scratch a dummy forward holds, recorded instead of allocated."""
    held, real = [], torch.empty

    def empty(*size, dtype=None, **kw):
        if dtype is torch.uint8:
            held.append(size[0])
            size = (0,)
        return real(*size, dtype=dtype, **kw)

    monkeypatch.setattr(torch, "empty", empty)
    attention_mla.MLAAttention.forward_impl(
        a, torch.zeros(num_tokens, 16, 576), None, None
    )
    return held


@pytest.mark.parametrize("heads", [16, 12])
@pytest.mark.parametrize(
    "env,dtype,page,prefill,held",
    [
        ("auto", "fp8", "1", True, True),
        ("kv", "fp8", "1", True, True),
        ("q", "fp8", "1", True, False),
        ("auto", "bf16", "1", True, False),
        ("auto", "fp8", "64", True, False),
        ("auto", "fp8", "1", False, False),
    ],
)
def test_a_dummy_prefill_holds_the_kv_scratch_of_a_full_batch(
    monkeypatch, env, dtype, page, prefill, held, heads
):
    """A dummy prefill holds, once, what a "kv" step at the batch budget adds
    for this layer's heads and padded heads; nothing where no step can take
    "kv". The GPU tests above check the same against the allocator."""
    cap, budget = 1 << 20, 8192
    monkeypatch.setenv("ATOM_MLA_PAGE_SIZE", page)
    a = _dummy_attention(
        monkeypatch, env, dtype, is_prefill=prefill, cap=cap, batch_tokens=budget
    )
    a.num_heads, a.padded_num_heads = heads, 32
    rows = dcp_ops.dcp_sparse_prefill_kv_profile_rows(
        env, budget, heads, 256, 16, 4, cap
    )
    want = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(
        budget, heads, 32, 512, 64, 1, 4, rows
    )
    assert _held_bytes(monkeypatch, a, 5) == ([want] if held else [])

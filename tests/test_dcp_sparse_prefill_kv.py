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

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import gpu_stack_import

# Before the imports: gpu_stack_import skips per test, not per module, so on the
# CPU-only runner it would fail collection here.
if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)

dcp_ops = gpu_stack_import("atom.model_ops.dcp_ops")
aiter_mla = gpu_stack_import("atom.model_ops.attentions.aiter_mla")
attention_mla = gpu_stack_import("atom.model_ops.attention_mla")

from atom.distributed.dcp_utils import mla_dcp_sparse_prefill_kv_splits
from atom.model_ops.attentions.sparse_mla_head_groups import (
    PERSISTENT_WORK_BUFFERS,
)

DEV = "cuda"
TOPK = 256
D = 576


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
    kw.setdefault("max_context", float("inf"))
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
    "context_lens,max_tokens,max_context,taken",
    [
        ([1024], 1 << 20, float("inf"), True),
        ([65536], 1 << 15, float("inf"), False),  # over the token cap
        ([131072], 1 << 20, 131072, True),  # at the context bound
        ([131072, 1], 1 << 20, 131072, False),  # past it
    ],
)
def test_plan_takes_kv_within_its_bounds(context_lens, max_tokens, max_context, taken):
    tables = [list(range(-(-n // 64) + 1)) for n in context_lens]
    plan = _plan(
        context_lens, tables, 4, 16, max_tokens=max_tokens, max_context=max_context
    )
    assert (plan is not None) == taken


def test_plan_refuses_a_block_table_too_short():
    with pytest.raises(AssertionError):
        _plan([200], [[0]], 4, 16)  # needs ceil(200 / 64) = 4 blocks


# ------------------------------------------------------------------ wiring
def _builder(monkeypatch, plan, nonps):
    """The builder fields `_dcp_sparse_prefill_step` reads; the plan and the
    aiter metadata call are stubbed, recording the heads the plan is built for."""
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
    b._dcp_sparse_prefill_kv_plan = lambda batch: seen.append("plan") or plan
    monkeypatch.setattr(
        aiter_mla, "get_mla_metadata_v1", lambda *a, **k: seen.append(a[3])
    )
    md = SimpleNamespace(
        sparse_cu_seqlens_q=None, sparse_kv_indptr=None, sparse_kv_last_page_lens=None
    )
    return b, md, seen


@pytest.mark.parametrize("nonps", [True, False])
def test_a_kv_step_plans_this_ranks_heads(monkeypatch, nonps):
    """A "kv" step runs the dcp=1 kernel: its work plan is for the per-rank
    padded heads, built even when the q path's non-persistent kernel reads none."""
    plan = object()
    b, md, seen = _builder(monkeypatch, plan, nonps)
    b._dcp_sparse_prefill_step(None, md, "")
    assert md.dcp_sparse_prefill_kv is plan
    assert seen == ["plan", 16]
    assert md.sparse_prefill_work_meta_data == "work_meta_data"


@pytest.mark.parametrize("nonps,planned", [(True, []), (False, [64])])
def test_a_q_step_keeps_its_plan(monkeypatch, nonps, planned):
    b, md, seen = _builder(monkeypatch, None, nonps)
    b._dcp_sparse_prefill_step(None, md, "")
    assert md.dcp_sparse_prefill_kv is None
    assert seen == ["plan"] + planned


def _resolver(monkeypatch, env, *, sparse=True, dcp=4, page=1, plugin=False, pcp=False):
    monkeypatch.setenv("ATOM_DCP_SPARSE_PREFILL", env)
    monkeypatch.setattr("atom.plugin.is_plugin_mode", lambda: plugin)
    monkeypatch.setattr("atom.distributed.pcp_utils.pcp_is_enabled", lambda: pcp)
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b.is_sparse, b.dcp_world_size, b.block_size = sparse, dcp, page
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


def _planner(monkeypatch, mode, max_tokens=1 << 20, dcp=4):
    """The builder fields `_dcp_sparse_prefill_kv_plan` reads, with the real
    plan; records its warnings."""
    warned = []
    monkeypatch.setattr(aiter_mla.logger, "warning", lambda *a: warned.append(a))
    b = object.__new__(aiter_mla.AiterMLAMetadataBuilder)
    b._dcp_sparse_prefill_mode = mode
    b._dcp_sparse_prefill_kv_max_tokens = max_tokens
    b._dcp_sparse_prefill_kv_fallback_warned = False
    b.dcp_world_size = dcp
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


@pytest.mark.parametrize(
    "mode,batch,taken",
    [
        ("auto", _batch([4096], [4096]), True),  # cold
        ("auto", _batch([131072], [64]), True),  # 128K hit + 64: "kv" measured faster
        ("auto", _batch([524288], [64]), False),  # 512K hit + 64: "q" faster
        ("auto", _batch([524288], [16384]), True),  # 512K hit + 16K: "kv" faster
        ("kv", _batch([524288], [64]), True),  # forced
        ("q", _batch([4096], [4096]), False),
        ("auto", _batch([4096], [4096], dummy=True), False),
        ("kv", _batch([4096], [4096], tables=[]), False),  # padding batch
    ],
)
def test_a_step_takes_kv_by_its_mode_and_batch(monkeypatch, mode, batch, taken):
    b, warned = _planner(monkeypatch, mode)
    assert (b._dcp_sparse_prefill_kv_plan(batch) is not None) == taken
    assert warned == []


@pytest.mark.parametrize("dcp,taken", [(4, True), (8, False)])
def test_the_context_bound_is_tighter_at_dcp8(monkeypatch, dcp, taken):
    """512K + 7K new tokens: under dcp4's 200K + 52 per token, over dcp8's 40."""
    b, _ = _planner(monkeypatch, "auto", dcp=dcp)
    plan = b._dcp_sparse_prefill_kv_plan(_batch([524288], [7168]))
    assert (plan is not None) == taken


def test_a_256k_hit_with_1k_new_tokens_stays_on_q(monkeypatch):
    """Measured on GLM-5.2-FP8 tp4/dcp4: q 0.428 s, kv 0.449 s."""
    b, _ = _planner(monkeypatch, "auto")
    assert b._dcp_sparse_prefill_kv_plan(_batch([262144], [1024])) is None


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
        assert b._dcp_sparse_prefill_kv_plan(_batch([65536], [65536])) is None
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


# ------------------------------------------------------------- memory profile
def test_the_kv_scratch_grows_with_the_cap_and_the_tokens():
    args = {"local_heads": 16, "kv_lora_rank": 512, "rope_dim": 64}
    f = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes
    base = f(1024, kv_itemsize=1, dcp_world_size=4, kv_max_rows=1 << 18, **args)
    assert f(1024, kv_itemsize=1, dcp_world_size=4, kv_max_rows=1 << 19, **args) > base
    assert f(4096, kv_itemsize=1, dcp_world_size=4, kv_max_rows=1 << 18, **args) > base
    # at least the gathered KV itself, 576 B a row
    assert base >= (1 << 18) * 576


def _dummy_attention(monkeypatch, env, kv_cache_dtype, is_prefill, cap=1 << 14):
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
            hf_config=SimpleNamespace(hidden_size=8), torch_dtype=torch.bfloat16
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
    kv = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(5, 16, 512, 64, 1, 4, cap)
    assert (_profiled_peak(a, 5) >= kv) == held


def test_a_dummy_decode_profiles_no_kv_scratch(monkeypatch):
    cap = 1 << 16
    a = _dummy_attention(monkeypatch, "auto", "fp8", is_prefill=False, cap=cap)
    kv = dcp_ops.dcp_sparse_prefill_kv_scratch_bytes(5, 16, 512, 64, 1, 4, cap)
    assert _profiled_peak(a, 5) < kv

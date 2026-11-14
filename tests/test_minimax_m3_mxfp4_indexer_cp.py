# SPDX-License-Identifier: MIT
"""Indexer-only CP over MiniMax-M3's MXFP4 index cache.

Each CP rank scores every index head over its 1/P of the blocks with aiter's MXFP4
decode scorer (`indexer_context_scores_mxfp4`), keeps its shard's top-k, and the
rank owning head h merges every shard's head-h candidates. The CPU tests pin the
wiring: the shard's map, bound and workspace, what the scorer is handed, which
scorer the CP decode runs with what page geometry its merge emits, and that the
installed aiter's signatures take those calls. The gfx950 test runs the four ranks
in one process, the exchange done by indexing, and pins the selection itself: the
merged top-k, sparse block table and context equal the native TP selector's for the
same head, bit for bit.
"""

import inspect
import types

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES
from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

aiter = pytest.importorskip("aiter")
gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)

WORLD, BOUND, MAX_BS, MAX_QLEN, PREFILL_ROWS = 4, 63, 16, 8, 40
TOPK, DIM = 16, 128
SM_SCALE = DIM**-0.5


def _config(max_model_len=BOUND * 128, max_num_batched_tokens=PREFILL_ROWS):
    return types.SimpleNamespace(
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        index_cache_dtype="fp4",
        enable_tbo=False,
        enable_tbo_decode=False,
    )


# ─────────────────────────────────────────────────────────────── wiring ──


@pytest.mark.parametrize("cp", [(1, 0), (WORLD, 0), (WORLD, WORLD - 1)])
def test_a_shard_sizes_its_map_bound_and_workspace(monkeypatch, cp):
    """Under CP the decode scorer runs every index head over ceil(blocks / P) local
    blocks: the map's capacity is the shard's, the record's bound the local count,
    and the workspace tall enough for that as well as for a prefill chunk's rows over
    every block (a prefill stays TP). Without CP nothing changes: no shard is named."""
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    calls = []

    def capacity(*args, **shard):
        calls.append((args, shard))
        return 100

    monkeypatch.setattr(
        "aiter.ops.flydsl.minimax_m3_index_score_work_map_capacity",
        capacity,
        raising=False,
    )
    world, rank = cp
    record = allocate_index_scorers(
        _config(), "cpu", max_bs=MAX_BS, max_qlen=MAX_QLEN, num_idx_heads=1, cp=cp
    )[""]
    local = -(-BOUND // world)
    shard = {} if world == 1 else {"cp_world": world, "cp_rank": rank}
    assert calls == [((MAX_BS, local, MAX_QLEN, world), shard)]
    assert (record.max_block, record.shard) == (local, shard)
    assert (record.cp_world, record.cp_rank) == cp
    assert record.work_map.shape == (100, 2)
    prefill = max(MAX_BS * MAX_QLEN, PREFILL_ROWS) * BOUND
    decode = world * MAX_BS * MAX_QLEN * local
    assert record.scores.capacity == max(prefill, decode)
    if world > 1:
        # 4 heads x 128 rows x 16 blocks outgrow 1 head x 128 rows x 63 blocks
        assert decode > prefill


def test_the_decode_metadata_builds_the_shards_map(monkeypatch):
    """The step's map covers every index head (this rank's times the world) over the
    record's shard, at its local bound; the step record keeps the shard."""
    from atom.model_ops.minimax_m3 import sparse_attn
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    built = []

    def work_map(seq_lens, max_block, q, heads, out=None, **shard):
        built.append((max_block, q, heads, shard))
        return out[:3]

    monkeypatch.setattr(
        sparse_attn,
        "flydsl_ops",
        lambda: types.SimpleNamespace(minimax_m3_index_score_work_map=work_map),
    )
    monkeypatch.setattr(
        sparse_attn, "build_n_valid_column_per_row", lambda *a, **k: None
    )
    record = IndexScoreBuffers(
        LogitsWorkspace(4, "cpu"), torch.zeros(8, 2), 16, cp_world=WORLD, cp_rank=2
    )
    md = sparse_attn.make_sparse_decode_metadata(
        seq_lens=torch.ones(2, dtype=torch.int32),
        block_table=None,
        slot_mapping=None,
        max_seq_len=5,
        num_idx_heads=1,
        max_query_len=4,
        index_scorer=record,
    )
    assert built == [(16, 4, WORLD, {"cp_world": WORLD, "cp_rank": 2})]
    assert md.index_scorer.shard == record.shard
    assert md.index_scorer.work_map.shape == (3, 2)


def _cp_scorer(monkeypatch):
    """`indexer_context_scores_mxfp4` with aiter's scorer doubled; returns it and
    what the double was handed."""
    from atom.model_ops.minimax_m3 import indexer_context_parallel as icp

    seen = {}

    def scorer(*args, **kwargs):
        seen.update(args=args, **kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        icp,
        "flydsl_ops",
        lambda: types.SimpleNamespace(minimax_m3_index_score_flydsl=scorer),
    )
    monkeypatch.setattr(icp, "page_views", lambda cache: ("data", "scale"))
    return icp.indexer_context_scores_mxfp4, seen


def test_the_cp_scorer_writes_the_shard_into_the_workspace(monkeypatch):
    """aiter's decode scorer with the shard: every index head of the step's rows over
    the local blocks, into a contiguous view of the shared workspace's head, on the
    step's map and the cache's two page views."""
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    score, seen = _cp_scorer(monkeypatch)
    batch, S, local, rank = 3, 2, 5, 1
    rows = batch * S
    ws = LogitsWorkspace(WORLD * rows * local + 7, "cpu")  # sized for more
    wmap = torch.zeros(4, 2, dtype=torch.int32)
    q = torch.zeros(rows, WORLD, DIM)
    table = torch.zeros(batch, WORLD * local, dtype=torch.int32)
    seq = torch.ones(batch, dtype=torch.int32)
    record = IndexScoreBuffers(ws, wmap, local, cp_world=WORLD, cp_rank=rank)

    out = score(q, "cache", table, seq, WORLD * local * 128, S, 0.5, record)
    assert out.shape == (WORLD, rows, local) and out.is_contiguous()
    assert out.data_ptr() == ws.logits(1, 1).data_ptr()
    args = seen["args"]
    assert args[0] is q and args[2] is table and args[3] is seq
    assert args[1] == "data" and args[4:] == (S, WORLD, 0.5, local)
    assert seen["work_map"] is wmap and seen["k_scale"] == "scale"
    assert (seen["cp_world"], seen["cp_rank"]) == (WORLD, rank)


def test_the_cp_scorer_refuses_what_it_cannot_score(monkeypatch):
    """No quiet allocation: a record without the step's map (a prefill's), a context
    wider than the shard's bound and a workspace too small for the step are refused."""
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    score, _ = _cp_scorer(monkeypatch)
    batch, S, local = 2, 1, 3
    q = torch.zeros(batch * S, WORLD, DIM)
    table = torch.zeros(batch, WORLD * local, dtype=torch.int32)
    seq = torch.ones(batch, dtype=torch.int32)
    wmap = torch.zeros(1, 2)
    fits = WORLD * local * 128  # 12 blocks, 3 a shard

    def call(ws, work_map=wmap, max_seq_len=fits):
        record = IndexScoreBuffers(ws, work_map, local, cp_world=WORLD, cp_rank=0)
        return score(q, None, table, seq, max_seq_len, S, 1.0, record)

    ws = LogitsWorkspace(WORLD * batch * S * local, "cpu")
    assert call(ws).shape == (WORLD, batch * S, local)
    with pytest.raises(ValueError, match="needs the step's index_scorer"):
        call(ws, work_map=None)
    with pytest.raises(ValueError, match="wider than the 3 blocks a shard"):
        call(ws, max_seq_len=fits + 1)
    with pytest.raises(ValueError, match="exceeds its workspace"):
        call(LogitsWorkspace(WORLD * batch * S * local - 1, "cpu"))


@pytest.mark.parametrize(
    "index_dtype,kv_dtype", [("fp4", "fp4"), ("fp4", "fp8"), ("fp8", "fp8")]
)
def test_the_cp_decode_runs_the_index_caches_scorer_and_the_kvs_geometry(
    monkeypatch, index_dtype, kv_dtype
):
    """The MXFP4 index scores with the step's record, any other index cache with the
    Triton scorer; the merge emits the KV's page geometry, as the TP selector does
    (one page-128 page a block for the MXFP4 KV, the page-16 defaults otherwise)."""
    from atom.distributed import indexer_cp
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3 import indexer_candidate_exchange as exchange
    from atom.model_ops.minimax_m3 import indexer_context_parallel as icp

    calls = []

    def double(name, result):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result

        return fn

    monkeypatch.setattr(icp, "indexer_context_scores", double("triton", "scores"))
    monkeypatch.setattr(icp, "indexer_context_scores_mxfp4", double("mxfp4", "scores"))
    monkeypatch.setattr(exchange, "local_candidate_keys", double("local", "keys"))
    monkeypatch.setattr(indexer_cp, "exchange_candidates", double("exchange", "all"))
    monkeypatch.setattr(exchange, "merge_candidate_keys", double("merge", (1, 2, 3)))

    impl = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
    impl.index_cache_dtype, impl.kv_cache_dtype = index_dtype, kv_dtype
    impl.index_cache, impl.scale = "cache", 0.5
    impl.indexer_cp_rank, impl.indexer_cp_world = 1, WORLD
    impl.topk, impl.init_blocks, impl.local_blocks = TOPK, 0, 1
    page_kw = (
        {"pages_per_block": 1, "block_page_stride": 1} if kv_dtype == "fp4" else {}
    )
    impl._select_page_kwargs = page_kw
    record = object()
    md = types.SimpleNamespace(index_scorer=record)
    table = torch.zeros(2, 9, dtype=torch.int32)

    got = impl._cp_index_topk_decode(
        "q", table, "seq", 1000, 2, index_scorer=impl._index_scorer(md)
    )
    assert got == (1, 2, 3)
    scorer = "mxfp4" if index_dtype == "fp4" else "triton"
    assert [c[0] for c in calls] == [scorer, "local", "exchange", "merge"]
    args = calls[0][1]
    if scorer == "mxfp4":
        assert args[:2] == ("q", "cache") and args[2] is table
        assert args[3:] == ("seq", 1000, 2, 0.5, record)
    else:
        assert args[3:] == ("seq", 1000, 1, WORLD, 2, 0.5)
    # the merge reads the exchanged keys with the KV's page geometry; the local
    # top-k knows the global block count, off the table
    assert calls[1][1] == ("scores", "seq", TOPK, 1, WORLD, 2, 9, 0, 1)
    assert calls[3][1][0] == "all" and calls[3][2] == page_kw


@pytest.mark.parametrize("enabled", [False, True])
def test_the_builder_sizes_the_scorer_for_this_ranks_shard(monkeypatch, enabled):
    """The builder hands `allocate_index_scorers` this rank's indexer-CP shard,
    ``(1, 0)`` without CP."""
    from atom.distributed import indexer_cp
    from atom.model_ops.attentions import aiter_attention

    monkeypatch.setattr(indexer_cp, "indexer_cp_enabled", lambda: enabled)
    monkeypatch.setattr(indexer_cp, "get_indexer_cp_world_size", lambda: WORLD)
    monkeypatch.setattr(indexer_cp, "get_indexer_cp_rank", lambda: 2)
    shard = aiter_attention.AiterAttentionMetadataBuilder._index_cp_shard()
    assert shard == ((WORLD, 2) if enabled else (1, 0))


def test_the_installed_aiter_takes_the_shards_calls(monkeypatch):
    """Each FlyDSL call a shard's decode makes binds to the installed aiter's
    signature, so an aiter that does not take one fails here, with no GPU, rather
    than at the first MXFP4 CP decode. The scorer's `k_scale` needs aiter#7060, the
    PR that added the MXFP4 scorer itself; its `cp_world` / `cp_rank` are older."""
    flydsl = pytest.importorskip("aiter.ops.flydsl")
    from atom.model_ops.minimax_m3 import indexer_context_parallel as icp
    from atom.model_ops.minimax_m3 import sparse_attn
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    bound = {}

    def binding(name, result):
        signature = inspect.signature(getattr(flydsl, name))

        def call(*args, **kwargs):
            bound[name] = signature.bind(*args, **kwargs).arguments
            return result(kwargs)

        return call

    capacity = "minimax_m3_index_score_work_map_capacity"
    monkeypatch.setattr(flydsl, capacity, binding(capacity, lambda kw: 4))
    ops = types.SimpleNamespace(
        minimax_m3_index_score_work_map=binding(
            "minimax_m3_index_score_work_map", lambda kw: kw["out"][:3]
        ),
        minimax_m3_index_score_flydsl=binding(
            "minimax_m3_index_score_flydsl", lambda kw: kw["out"]
        ),
    )
    monkeypatch.setattr(sparse_attn, "flydsl_ops", lambda: ops)
    monkeypatch.setattr(
        sparse_attn, "build_n_valid_column_per_row", lambda *a, **k: None
    )
    monkeypatch.setattr(icp, "flydsl_ops", lambda: ops)
    monkeypatch.setattr(icp, "page_views", lambda cache: ("data", "scale"))

    seq = torch.ones(2, dtype=torch.int32)
    record = allocate_index_scorers(
        _config(), "cpu", max_bs=2, max_qlen=1, num_idx_heads=1, cp=(WORLD, 1)
    )[""]
    step = sparse_attn.make_sparse_decode_metadata(
        seq_lens=seq,
        block_table=None,
        slot_mapping=None,
        max_seq_len=128,
        num_idx_heads=1,
        index_scorer=record,
    ).index_scorer
    table = torch.zeros(2, BOUND, dtype=torch.int32)
    q = torch.zeros(2, WORLD, DIM)
    icp.indexer_context_scores_mxfp4(q, "cache", table, seq, 128, 1, SM_SCALE, step)
    assert bound.keys() == {
        capacity,
        "minimax_m3_index_score_work_map",
        "minimax_m3_index_score_flydsl",
    }
    scorer = bound["minimax_m3_index_score_flydsl"]
    assert (scorer["k_scale"], scorer["cp_world"], scorer["cp_rank"]) == (
        "scale",
        WORLD,
        1,
    )


# ─────────────────────────────────────────────────────────── selection ──


def make_case(batch, S, seed, max_len, fp8_q):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = torch.randint(S + 1, max_len, (batch,), device="cuda", generator=g)
    seq[0] = max_len  # one full-length request
    blocks = -(-max_len // 128)
    pages = batch * blocks + 3
    cache = torch.randint(
        0, 256, (pages, PAGE_BYTES), device="cuda", generator=g, dtype=torch.uint8
    )
    # e8m0 bytes within 2^-2..2^2 of unity, as a quantized activation's are
    d = cache.view(pages, 1, PAGE_BYTES)[:, :, 8192:]
    d.copy_(
        torch.randint(125, 130, d.shape, device="cuda", generator=g).to(torch.uint8)
    )
    table = torch.stack(
        [
            torch.randperm(pages, device="cuda", generator=g)[:blocks]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    q = torch.randn(batch * S, WORLD, DIM, device="cuda", generator=g) * 0.7
    q = q.to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    return cache, table, seq.to(torch.int32), q


def _step_record(table, seq, S, cp):
    """This rank's builder record and the step's map built from it, as the builder
    and `make_sparse_decode_metadata` make them, the workspace filled with values no
    selection may depend on."""
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    batch, blocks = table.shape
    record = allocate_index_scorers(
        _config(blocks * 128, 1),
        "cuda",
        max_bs=batch,
        max_qlen=S,
        num_idx_heads=1,
        cp=cp,
    )[""]
    flat = record.scores.logits(1, record.scores.capacity)
    junk = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 3e38], device="cuda"
    )
    flat.copy_(junk[torch.randint(0, 4, flat.shape, device="cuda")])
    return make_sparse_decode_metadata(
        seq_lens=seq,
        block_table=table,
        slot_mapping=torch.zeros(batch * S, dtype=torch.int64, device="cuda"),
        max_seq_len=blocks * 128,
        num_idx_heads=1,
        max_query_len=S,
        index_scorer=record,
    ).index_scorer


def native_head(cache, table, seq, q, S, head, forced, page_kw):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    return minimax_m3_index_topk_decode(
        q[:, head : head + 1].contiguous(), cache, table, seq, int(seq.max()), TOPK,
        *forced, 1, SM_SCALE, emit_sparse_block_table=True, max_query_len=S,
        index_scorer=_step_record(table, seq, S, (1, 0)), **page_kw,
    )  # fmt: skip


def cp_shard_keys(cache, table, seq, q, S, rank, forced):
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        local_candidate_keys,
    )
    from atom.model_ops.minimax_m3.indexer_context_parallel import (
        indexer_context_scores_mxfp4,
    )

    record = _step_record(table, seq, S, (WORLD, rank))
    scores = indexer_context_scores_mxfp4(
        q, cache, table, seq, int(seq.max()), S, SM_SCALE, record
    )
    return local_candidate_keys(
        scores, seq, TOPK, rank, WORLD, S, table.shape[1], *forced
    )


@gfx950
# 16-pages a block: 1 for the MXFP4 KV, 8 for the page-16 SHUFFLE KV an index-only
# MXFP4 run keeps
@pytest.mark.parametrize("ppb", [1, 8])
@pytest.mark.parametrize("forced", [(0, 1), (1, 1)])
@pytest.mark.parametrize(
    "batch,S,max_len,fp8_q",
    [
        (1, 8, 128 * 70, True),
        (5, 8, 128 * 70, True),
        (17, 8, 128 * 300 + 77, True),  # 301 blocks: shards of 76, 75, 75, 75
        (6, 1, 128 * 41, False),
    ],
)
def test_cp_selection_equals_native(batch, S, max_len, fp8_q, forced, ppb):
    from atom.model_ops.minimax_m3.indexer_candidate_exchange import (
        merge_candidate_keys,
    )

    page_kw = {"pages_per_block": ppb, "block_page_stride": ppb}
    cache, table, seq, q = make_case(batch, S, batch * 100 + S, max_len, fp8_q)
    shard_keys = [
        cp_shard_keys(cache, table, seq, q, S, rank, forced) for rank in range(WORLD)
    ]
    for head in range(WORLD):
        want = native_head(cache, table, seq, q, S, head, forced, page_kw)
        got = merge_candidate_keys(
            torch.stack([keys[head] for keys in shard_keys]), table, seq, TOPK,
            *forced, S, **page_kw,
        )  # fmt: skip
        for name, g, w in zip(("top-k", "sparse block table", "ctx"), got, want):
            assert torch.equal(g, w), f"head {head} {name}"

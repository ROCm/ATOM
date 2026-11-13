# SPDX-License-Identifier: MIT
"""Indexer-only CP on the FP4 (MXFP4 / NVFP4) index caches, CPU part: the scorer
record's CP geometry, the arm each decode step takes and its batch threshold. The CP
chain's exactness against the TP selector is the GPU parity
(``tests/model_ops/run_indexer_cp_fp4_tp4.py``)."""

import types

import pytest
import torch

pytest.importorskip("aiter")

WORLD, RANK, HEADS = 4, 2, 1
MAX_BLOCK = 66  # -> a shard's local bound of 17


def _config(index="nvfp4", tbo=False):
    return types.SimpleNamespace(
        max_model_len=MAX_BLOCK * 128,
        max_num_batched_tokens=40,
        index_cache_dtype=index,
        enable_tbo=tbo,
        enable_tbo_decode=tbo,
    )


def _record(monkeypatch, min_batch=16, **kw):
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    calls = []

    def capacity(*args, **cp):
        calls.append((args, cp))
        return 100 if not cp else 150

    monkeypatch.setattr(
        "aiter.ops.flydsl.minimax_m3_index_score_work_map_capacity",
        capacity,
        raising=False,
    )
    rec = allocate_index_scorers(
        _config(**kw),
        "cpu",
        max_bs=32,
        max_qlen=4,
        num_idx_heads=HEADS,
        cp_world=WORLD,
        cp_rank=RANK,
        cp_min_batch=min_batch,
    )[""]
    return rec, calls


def test_the_cp_record_is_sized_for_both_arms(monkeypatch):
    rec, calls = _record(monkeypatch)
    # the TP arm's map geometry, then the CP arm's: every index head over a shard
    assert calls[0] == ((32, MAX_BLOCK, 4, HEADS), {})
    assert calls[1] == (
        (32, 17, 4, WORLD),
        {"cp_world": WORLD, "cp_rank": RANK},
    )
    assert rec.work_map.shape[0] == 150  # the larger of the two grids
    # one workspace: a prefill chunk's rows or the CP arm's heads x rows x local
    assert rec.scores.capacity == max(HEADS * 128 * MAX_BLOCK, WORLD * 32 * 4 * 17)
    assert (rec.cp_world, rec.cp_rank, rec.cp_max_block) == (WORLD, RANK, 17)
    assert rec.nvfp4


def test_a_record_without_cp_is_the_tp_record(monkeypatch):
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    monkeypatch.setattr(
        "aiter.ops.flydsl.minimax_m3_index_score_work_map_capacity",
        lambda *a, **k: 100,
        raising=False,
    )
    rec = allocate_index_scorers(
        _config("fp4"), "cpu", max_bs=32, max_qlen=4, num_idx_heads=HEADS
    )[""]
    assert not rec.cp_decode(10**6) and not rec.nvfp4
    assert rec.work_map.shape[0] == 100


@pytest.mark.parametrize("batch", [8, 15, 16, 32])
def test_a_decode_step_builds_the_map_of_the_arm_it_runs(monkeypatch, batch):
    from atom.model_ops.minimax_m3 import sparse_attn

    rec, _ = _record(monkeypatch)
    built = []
    maps = types.SimpleNamespace(
        minimax_m3_index_score_work_map=lambda *a, **kw: built.append((a, kw))
        or kw["out"]
    )
    monkeypatch.setattr(sparse_attn, "flydsl_ops", lambda: maps)
    monkeypatch.setattr(
        sparse_attn, "build_n_valid_column_per_row", lambda *a, **k: None
    )
    seq = torch.full((batch,), 900, dtype=torch.int32)
    md = sparse_attn.make_sparse_decode_metadata(
        seq_lens=seq,
        block_table=torch.zeros(batch, MAX_BLOCK, dtype=torch.int32),
        slot_mapping=torch.zeros(batch * 4, dtype=torch.int64),
        max_seq_len=900,
        num_idx_heads=HEADS,
        max_query_len=4,
        index_scorer=rec,
    )
    args, kw = built[0]
    step = md.index_scorer
    if batch >= 16:  # the CP arm: all heads, a shard's blocks, this rank's geometry
        assert args[1:] == (17, 4, WORLD)
        assert (kw["cp_world"], kw["cp_rank"]) == (WORLD, RANK)
        assert (step.cp_world, step.max_block) == (WORLD, 17)
    else:  # the TP arm: the record's own head over every block, no shard
        assert args[1:] == (MAX_BLOCK, 4, HEADS)
        assert "cp_world" not in kw
        assert (step.cp_world, step.max_block) == (1, MAX_BLOCK)
    assert step.work_map is rec.work_map


@pytest.mark.parametrize(
    "index, env, want",
    [
        ("nvfp4", None, 16),
        ("fp4", None, 16),
        ("fp8", None, 1),
        ("bf16", None, 1),
        ("nvfp4", "1", 1),
        ("fp8", "8", 8),
    ],
)
def test_the_min_batch_default_follows_the_index_cache(monkeypatch, index, env, want):
    from atom.distributed import indexer_cp

    monkeypatch.setattr(
        indexer_cp,
        "get_current_atom_config",
        lambda: types.SimpleNamespace(index_cache_dtype=index),
    )
    monkeypatch.delenv("ATOM_M3_INDEXER_CP_MIN_BATCH", raising=False)
    if env is not None:
        monkeypatch.setenv("ATOM_M3_INDEXER_CP_MIN_BATCH", env)
    assert indexer_cp.indexer_cp_min_batch() == want


@pytest.mark.parametrize("batch, cp_arm", [(15, False), (16, True)])
def test_the_sparse_decode_takes_the_cp_arm_from_the_min_batch(
    monkeypatch, batch, cp_arm
):
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn

    seen = {}

    def cp_chain(index_q, bt, sl, max_seq_len, mq, scorer):
        seen["cp"] = (tuple(index_q.shape), scorer)
        n = index_q.shape[0]
        return torch.zeros(1, n, 2, dtype=torch.int32), None, None

    def tp_chain(index_q, *a, **kw):
        seen["tp"] = (tuple(index_q.shape), tuple(index_q.stride()))
        return torch.zeros(1, index_q.shape[0], 2, dtype=torch.int32)

    monkeypatch.setattr(index_topk, "minimax_m3_index_topk_decode", tp_chain)
    monkeypatch.setattr(
        sparse_attn, "minimax_m3_sparse_attn_decode_asm", lambda *a, **k: None
    )
    layer = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(layer)
    layer._cp_index_topk_decode = cp_chain
    layer.kv_cache_dtype = layer.index_cache_dtype = "nvfp4"
    layer._select_page_kwargs = {}
    layer.skip_index_topk, layer.index_topk_cache_state = False, {}
    layer.indexer_cp_group, layer.indexer_cp_rank = object(), RANK
    layer.indexer_cp_min_batch = 16
    layer.topk, layer.init_blocks, layer.local_blocks = 2, 1, 0
    layer.num_kv_heads, layer.scale = 1, 1.0
    layer.index_cache, layer.layer_num, layer.sparse_layer_ordinal = None, 0, 0
    layer.nvfp4_k_scale = layer.nvfp4_v_scale = None
    layer._index_q = torch.zeros(batch, WORLD, 8)  # every index head, one row a request
    layer._index_q_cache_key_info = None
    layer._index_scorer = lambda md: md.index_scorer
    attend = types.SimpleNamespace(called=False)
    from atom.model_ops.minimax_m3 import sparse_attn_nvfp4

    monkeypatch.setattr(
        sparse_attn_nvfp4,
        "sparse_pa_nvfp4_decode",
        lambda *a, **k: setattr(attend, "called", True),
    )
    scorer = object()
    md = types.SimpleNamespace(
        max_seq_len=100,
        n_valid_column_per_row=None,
        index_scorer=scorer,
        decode=types.SimpleNamespace(
            seq_lens=torch.zeros(batch, dtype=torch.int32),
            block_table=torch.zeros(batch, 4, dtype=torch.int32),
            max_query_len=1,
        ),
    )
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(sparse_attention_metadata=md)
    )
    q = v = torch.zeros(batch, 4, 8)
    layer._sparse_decode(q, None, v, None, None, None, None, ctx)
    assert attend.called
    if cp_arm:  # every head's query and the step's record go to the CP chain
        assert seen["cp"] == ((batch, WORLD, 8), scorer) and "tp" not in seen
    else:  # this rank's own head of the wide projection, as prefill takes it
        assert seen["tp"][0] == (batch, 1, 8) and "cp" not in seen
        assert seen["tp"][1] == (WORLD * 8, 8, 1)

# SPDX-License-Identifier: MIT
"""Dense MiniMax-M3 attention under ``--kv-cache-dtype fp4`` (CPU, no kernels).

Pinned here: the layer picks the MXFP4 impl only for fp4; the writer is called in
its no-index MXFP4 form; a q_len > 1 decode is folded into length-1 rows whose
contexts are exactly the causal mask; prefill reads every token back from the
pages; and every mode the impl does not support, and every base runner that cannot
read MXFP4 pages, raises instead of misreading. The kernels themselves (writer,
``pa_decode``, gather) have their own GPU tests and are replaced by spies.
"""

import sys
import types

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")

from atom.model_ops import attention_mha as am
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES

H, HKV, D = 4, 2, 128


@pytest.fixture(autouse=True)
def no_context_parallel(monkeypatch):
    """``get_dcp_world_size`` reads the engine config, which a unit test has none of."""
    import atom.distributed.dcp_utils as dcp

    monkeypatch.setattr(dcp, "get_dcp_world_size", lambda: 1)


def gemma_norm():
    norm = GemmaRMSNorm.__new__(GemmaRMSNorm)
    nn.Module.__init__(norm)
    norm.weight = nn.Parameter(torch.ones(D))
    norm.variance_epsilon = 1e-6
    return norm


def make_impl(cls=am.MiniMaxM3DenseMxfp4PagedAttentionImpl, **over):
    """An impl without ``__init__`` (which wants a GPU); fields as it would set."""
    impl = object.__new__(cls)
    nn.Module.__init__(impl)
    fields = {
        "num_heads": H,
        "num_kv_heads": HKV,
        "head_dim": D,
        "scale": D**-0.5,
        "layer_num": 0,
        "kv_cache_dtype": "fp4",
        "sliding_window": -1,
        "sinks": None,
        "alibi_slopes": None,
        "use_flash_layout": False,
        "q_norm": gemma_norm(),
        "k_norm": gemma_norm(),
        "rotary_emb": types.SimpleNamespace(rotary_dim=64),
    }
    fields.update(over)
    for name, value in fields.items():
        setattr(impl, name, value)
    return impl


def pages(n=3):
    return torch.zeros(n, HKV, PAGE_BYTES, dtype=torch.uint8)


def fwd_ctx(is_prefill, **md):
    return types.SimpleNamespace(
        context=types.SimpleNamespace(is_prefill=is_prefill),
        attn_metadata=types.SimpleNamespace(**md),
        kv_cache_data={
            "layer_0": types.SimpleNamespace(k_cache=pages(), v_cache=pages())
        },
    )


# --- selection -------------------------------------------------------------


def test_only_fp4_takes_the_mxfp4_impl():
    pytest.importorskip("atom.models.minimax_m3.model")
    from atom.models.minimax_m3.model import _dense_attention_kwargs

    assert _dense_attention_kwargs("fp4") == {
        "kv_cache_dtype": "fp4",
        "impl_cls": am.MiniMaxM3DenseMxfp4PagedAttentionImpl,
    }
    for dtype in ("bf16", "fp8"):
        assert _dense_attention_kwargs(dtype) == {"kv_cache_dtype": dtype}


# --- unsupported modes -----------------------------------------------------


def test_a_supported_layer_passes_its_checks():
    make_impl()._check_supported()


@pytest.mark.parametrize(
    "over, why",
    [
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype fp4"),
        ({"head_dim": 64}, "head_dim 128"),
        ({"sliding_window": 128}, "sliding window"),
        ({"sinks": nn.Parameter(torch.ones(H))}, "sliding window or sinks"),
        ({"alibi_slopes": [0.5]}, "alibi"),
        ({"q_norm": None, "k_norm": None}, "Gemma q/k norm"),  # EAGLE3: no qk norm
        ({"rotary_emb": None}, "Gemma q/k norm"),
        ({"num_heads": 34, "num_kv_heads": 2}, "1..16 query heads"),
        ({"use_flash_layout": True}, "flash layout"),
    ],
)
def test_unsupported_modes_raise(over, why):
    with pytest.raises(NotImplementedError, match=why):
        make_impl(**over)._check_supported()


def test_context_parallel_decode_raises(monkeypatch):
    import atom.distributed.dcp_utils as dcp

    monkeypatch.setattr(dcp, "get_dcp_world_size", lambda: 2)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        make_impl()._check_supported()


def test_unified_attention_env_raises(monkeypatch):
    monkeypatch.setattr(am.envs, "ATOM_USE_UNIFIED_ATTN", True, raising=False)
    with pytest.raises(NotImplementedError, match="UNIFIED_ATTN"):
        make_impl()._check_supported()


@pytest.mark.parametrize(
    "call",
    [
        lambda i: i.rope_cache(None, None, None, None, None, fwd_ctx(False)),
        lambda i: i.dispatch_backend(fwd_ctx(False), None, None, None),
        lambda i: i.paged_attention_unified(*[None] * 7, fwd_ctx(False)),
        lambda i: i.prefill_attention_triton(*[None] * 7, fwd_ctx(True)),
    ],
)
def test_base_runners_refuse_mxfp4_pages_and_leave_fp8_alone(call):
    base = make_impl(cls=am.PagedAttentionImpl)
    with pytest.raises(NotImplementedError, match="MXFP4 KV pages"):
        call(base)
    base.kv_cache_dtype = "fp8"
    base._refuse_mxfp4("x")  # a no-op for every other dtype


def test_the_sparse_impl_still_overrides_the_base_guard():
    assert (
        am.SparseMHAPagedAttentionImpl.dispatch_backend
        is not am.PagedAttentionImpl.dispatch_backend
    )
    assert (
        am.SparseMHAPagedAttentionImpl.rope_cache
        is not am.PagedAttentionImpl.rope_cache
    )


# --- writer ----------------------------------------------------------------


@pytest.fixture
def writer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        am.aiter, "fused_qknorm_idxrqknorm", lambda *a, **k: calls.append((a, k))
    )
    # the writer wrapper imports the model's cos/sin helper; the model module is
    # heavy (moe, distributed) and not under test, so a stand-in is installed
    monkeypatch.setitem(
        sys.modules,
        "atom.models.minimax_m3.model",
        types.SimpleNamespace(_minimax_m3_cos_sin_cache=lambda *_: torch.zeros(1, 64)),
    )
    return calls


@pytest.mark.parametrize("is_prefill", [False, True])
def test_writer_is_the_no_index_mxfp4_form(writer, is_prefill):
    impl = make_impl()
    tokens = 3
    qkv = torch.zeros(tokens, (H + 2 * HKV) * D, dtype=torch.bfloat16)
    slots = torch.arange(tokens)
    ctx = fwd_ctx(is_prefill, slot_mapping=slots)
    out = impl.rope_cache(None, "k", "v", qkv, torch.arange(tokens), ctx)
    q, k, v, k_pages, v_pages, k_scale, v_scale = out
    ((args, kwargs),) = writer
    # positional: qkv, q_w, k_w, cos_sin, pos, H, Hkv, rotary_dim, eps, then no index
    assert args[5:8] == (H, HKV, 64)
    assert args[9:12] == (None, None, 0)  # index norms, num_index_heads
    assert args[12] is slots and args[15] is None  # slot_mapping, index_cache
    assert args[16] == 128  # one page of 128 tokens
    assert args[13] is k_pages and args[14] is v_pages
    assert kwargs == {"kv_cache_dtype": "mxfp4", "asm_layout": False}
    q_out = args[17]
    # decode attends with the e4m3 q pa_decode takes; prefill with bf16
    assert q_out.dtype == (torch.bfloat16 if is_prefill else torch.float8_e4m3fn)
    assert q.shape == (tokens, H, D) and q.data_ptr() == q_out.data_ptr()
    assert (k, v, k_scale, v_scale) == ("k", "v", None, None)


def test_writer_refuses_a_non_mxfp4_pool_and_non_bf16_qkv(writer):
    impl = make_impl()
    qkv = torch.zeros(2, (H + 2 * HKV) * D, dtype=torch.bfloat16)
    ctx = fwd_ctx(False, slot_mapping=torch.arange(2))
    ctx.kv_cache_data["layer_0"].k_cache = torch.zeros(3, HKV, 128, 128).to(
        torch.float8_e4m3fn
    )
    with pytest.raises(RuntimeError, match="MXFP4 pages"):
        impl.rope_cache(None, None, None, qkv, torch.arange(2), ctx)
    with pytest.raises(NotImplementedError, match="bf16 qkv"):
        impl.rope_cache(None, None, None, qkv.half(), torch.arange(2), fwd_ctx(False))
    assert not writer  # nothing was written


# --- decode ----------------------------------------------------------------


def paged_tokens(ctx, bt, pool):
    """The first ``ctx`` tokens a block-table row addresses."""
    return pool[bt].reshape(-1, pool.shape[-1])[:ctx]


@pytest.mark.parametrize("qlen", [1, 2, 4])
def test_row_fold_is_the_causal_mask(qlen):
    """Each folded row attends what token j of its request may attend."""
    gen = torch.Generator().manual_seed(qlen)
    block, per_seq, dim = 4, 5, 8
    pool_k = torch.randn(16, block, dim, generator=gen)
    pool_v = torch.randn(16, block, dim, generator=gen)
    ctx = torch.tensor([qlen + 3, 17, qlen], dtype=torch.int32)
    bt = torch.stack([torch.randperm(16, generator=gen)[:per_seq] for _ in ctx]).int()
    q = torch.randn(len(ctx) * qlen, dim, generator=gen)

    def attend(qv, k, v):
        return torch.softmax(qv @ k.T, -1) @ v

    want = []
    for b in range(len(ctx)):
        k = paged_tokens(int(ctx[b]), bt[b].long(), pool_k)
        v = paged_tokens(int(ctx[b]), bt[b].long(), pool_v)
        for j in range(qlen):
            keys = int(ctx[b]) - qlen + j + 1  # position + 1: j-th of the last qlen
            want.append(attend(q[b * qlen + j], k[:keys], v[:keys]))
    rows_ctx, rows_bt = am.dense_decode_rows(ctx, bt, qlen)
    got = [
        attend(
            q[r],
            paged_tokens(int(rows_ctx[r]), rows_bt[r].long(), pool_k),
            paged_tokens(int(rows_ctx[r]), rows_bt[r].long(), pool_v),
        )
        for r in range(len(rows_ctx))
    ]
    torch.testing.assert_close(torch.stack(got), torch.stack(want))


def test_row_fold_values_and_padding():
    ctx = torch.tensor([7, 130, 0], dtype=torch.int32)
    bt = torch.arange(6, dtype=torch.int32).view(3, 2)
    rows_ctx, rows_bt = am.dense_decode_rows(ctx, bt, 3)
    assert rows_ctx.tolist() == [5, 6, 7, 128, 129, 130, 0, 0, 0]  # pad stays 0
    assert rows_ctx.dtype == torch.int32
    assert rows_bt.tolist() == [[0, 1]] * 3 + [[2, 3]] * 3 + [[4, 5]] * 3
    same = am.dense_decode_rows(ctx, bt, 1)
    assert same[0] is ctx and same[1] is bt  # plain decode: no copy


def test_decode_hands_pa_decode_the_folded_rows(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        am, "dense_pa_mxfp4", lambda *a, **k: seen.update(args=a, kw=k) or "out"
    )
    impl = make_impl()
    md = {
        "max_seqlen_q": 2,
        "context_lens": torch.tensor(
            [7, 130, 0], dtype=torch.int32
        ),  # a padded request
        "block_tables": torch.arange(12, dtype=torch.int32).view(3, 4),
    }
    q = torch.zeros(4, H, D, dtype=torch.float8_e4m3fn)
    ctx = fwd_ctx(False, **md)
    assert impl.dispatch_backend(ctx, q, None, None) == impl._dense_decode
    out = impl._dense_decode(q, None, None, "kp", "vp", None, None, ctx)
    assert out == "out"
    _, kp, vp, rows_ctx, rows_bt, scale = seen["args"]
    assert (kp, vp, scale) == ("kp", "vp", impl.scale)
    assert rows_ctx.tolist() == [6, 7, 129, 130]
    assert rows_bt.tolist() == [[0, 1, 2, 3]] * 2 + [[4, 5, 6, 7]] * 2
    # per-layer refresh, never the step metadata: see the collision test below
    assert seen["kw"]["plan_step_owner"] is None
    with pytest.raises(NotImplementedError, match="q_len 2"):  # ragged
        impl._dense_decode(q[:3], None, None, "kp", "vp", None, None, ctx)
    with pytest.raises(NotImplementedError, match="q_len 2"):  # more rows than requests
        impl._dense_decode(
            torch.zeros(8, H, D, dtype=torch.float8_e4m3fn),
            None, None, "kp", "vp", None, None, ctx,
        )  # fmt: skip


@pytest.fixture
def decode_spies(monkeypatch):
    """Spies for the planner and ``pa_decode`` that ``dense_pa_mxfp4`` calls."""
    import aiter.ops.flydsl.pa_decode as flydsl

    import atom.model_ops.base_attention as base

    seen = {"plans": [], "calls": []}

    def plan(**kw):
        seen["plans"].append(kw)
        return "plan"

    monkeypatch.setattr(base, "_flydsl_prepare_explicit_plan", plan)
    monkeypatch.setattr(base, "_flydsl_plan_scratch", lambda *a: (None, None, None))
    monkeypatch.setattr(base, "_FLYDSL_PLAN_MAX_BATCH", 2)
    monkeypatch.setattr(am, "dense_decode_splits", lambda *a: 4)
    monkeypatch.setattr(
        flydsl, "pa_decode", lambda *a, **k: seen["calls"].append((a, k)), raising=False
    )
    return seen


def decode_args(q_dtype=torch.bfloat16, rows=5):
    q = torch.zeros(rows, H, D, dtype=torch.float32).to(q_dtype)
    ctx = torch.arange(1, rows + 1, dtype=torch.int32)
    return q, pages(), pages(), ctx, torch.zeros(rows, 3, dtype=torch.int32), 0.1


@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_dense_decode_takes_a_bf16_or_e4m3_query(decode_spies, q_dtype):
    out = am.dense_pa_mxfp4(*decode_args(q_dtype))
    assert out.dtype == torch.bfloat16 and out.shape == (5, H, D)
    # 5 rows in plans of 2: three segments, each handed its own slice of the rows
    assert [len(a[0][1]) for a in decode_spies["calls"]] == [2, 2, 1]
    assert all(a[0][1].dtype == q_dtype for a in decode_spies["calls"])
    assert [p["plan_segment"] for p in decode_spies["plans"]] == [0, 1, 2]


@pytest.mark.parametrize("q_dtype", [torch.float16, torch.float32])
def test_dense_decode_refuses_other_query_dtypes(decode_spies, q_dtype):
    with pytest.raises(TypeError, match="e4m3 or bf16"):
        am.dense_pa_mxfp4(*decode_args(q_dtype))
    assert not decode_spies["calls"]


def test_dense_decode_refuses_too_many_query_heads_per_kv_head(decode_spies):
    _q, k, v, ctx, bt, scale = decode_args()
    with pytest.raises(ValueError, match="1..16"):
        am.dense_pa_mxfp4(
            torch.zeros(5, 34, D, dtype=torch.bfloat16), k, v, ctx, bt, scale
        )


def test_plan_step_owner_is_forwarded_as_given(decode_spies):
    """None (a draft's mutable metadata) refreshes the plan every call; an owner
    refreshes once per step. The planner keys the two apart, so both are safe."""
    am.dense_pa_mxfp4(*decode_args())
    assert all(p["plan_step_owner"] is None for p in decode_spies["plans"])
    owner = object()
    am.dense_pa_mxfp4(*decode_args(), plan_step_owner=owner)
    assert decode_spies["plans"][-1]["plan_step_owner"] is owner


def test_dense_decode_plan_never_clobbers_a_sparse_decode_plan(monkeypatch):
    """Sparse (Hkv/rank=1, step-owned) and dense decode build the same plan key at
    batch >= 128. A step-owned dense plan would refresh the sparse plan with dense
    lengths, and the next sparse layer (already marked refreshed) would reuse them.
    ``_dense_decode`` passes None, which keys its plan apart and refreshes per call."""
    import aiter.ops.flydsl.pa_decode as flydsl

    import atom.model_ops.base_attention as base

    def fake_plan(lengths, num_kv_heads, **kw):
        plan = kw.get("plan") or types.SimpleNamespace(max_partitions=1, capacity=128)
        plan.lengths = lengths
        return plan

    monkeypatch.setattr(flydsl, "plan_pa_decode", fake_plan, raising=False)
    monkeypatch.setattr(base, "_FLYDSL_RUNTIME_PLANS", {})
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: types.SimpleNamespace(cuda_stream=7),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    seen = {}
    monkeypatch.setattr(
        am, "dense_pa_mxfp4", lambda *a, **k: seen.update(kw=k) or "out"
    )
    impl = make_impl()
    md = {
        "max_seqlen_q": 1,
        "context_lens": torch.ones(128, dtype=torch.int32),
        "block_tables": torch.zeros(128, 4, dtype=torch.int32),
    }
    q = torch.zeros(128, H, D, dtype=torch.float8_e4m3fn)
    impl._dense_decode(q, None, None, "kp", "vp", None, None, fwd_ctx(False, **md))
    dense_owner = seen["kw"]["plan_step_owner"]

    sparse_md = types.SimpleNamespace()
    sparse_ctx = torch.full((128,), 1, dtype=torch.int32)
    dense_ctx = torch.full((128,), 2, dtype=torch.int32)

    def plan_for(owner, ctx):  # same shape on both sides: Hkv=1, 128 rows, one split
        return base._flydsl_prepare_explicit_plan(
            q=torch.empty((128, 16, D), device="meta"),
            k_cache=torch.empty((8, 1, 8, 16, 16), device="meta"),
            context_lens=ctx,
            num_seqs=128,
            query_length=1,
            plan_partition_cap=1,
            context_bound=512,
            existing_plan=None,
            plan_step_owner=owner,
        )

    sparse = plan_for(sparse_md, sparse_ctx)
    dense = plan_for(dense_owner, dense_ctx)
    again = plan_for(sparse_md, sparse_ctx)  # the next sparse layer of the step
    assert again is sparse and dense is not sparse
    assert sparse.lengths.tolist() == sparse_ctx.tolist()  # not the dense lengths


# --- prefill ---------------------------------------------------------------


def prefill_md(**over):
    md = {
        "block_tables": torch.arange(8, dtype=torch.int32).view(2, 4),
        "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 3, 8], dtype=torch.int32),
        "context_lens": torch.tensor([3, 5], dtype=torch.int32),
        "total_kv": 8,
        "max_seqlen_q": 5,
        "max_seqlen_k": 5,
        "min_seqlen_q": 0,
        "dropout_p": 0.0,
        "has_cached": False,
        "batch_id_per_k_token": None,
        "seq_starts": None,
    }
    md.update(over)
    return md


@pytest.fixture
def prefill_spies(monkeypatch):
    import atom.model_ops.minimax_m3.gather_mxfp4 as gather

    seen = {}

    def fake_gather(*a, **k):
        seen["gather"] = a
        return "k_full", "v_full"

    def fake_varlen(*a, **k):
        seen["varlen"] = (a, k)
        return "o"

    monkeypatch.setattr(gather, "gather_mxfp4_prefix", fake_gather)
    monkeypatch.setattr(am.aiter, "flash_attn_varlen_func", fake_varlen)
    return seen


def test_fresh_prefill_still_reads_every_token_from_the_pages(prefill_spies):
    """No prefix hit: the builder uploads no key-side map, so it is derived."""
    impl = make_impl()
    ctx = fwd_ctx(True, **prefill_md())
    q = torch.zeros(8, H, D, dtype=torch.bfloat16)
    k_pages, v_pages = pages(), pages()
    assert (
        impl.prefill_attention(q, None, None, k_pages, v_pages, None, None, ctx) == "o"
    )
    kc, vc, bt, _cu_k, ids, starts, total = prefill_spies["gather"]
    assert (kc is k_pages, vc is v_pages, total) == (True, True, 8)
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert starts.tolist() == [0, 0] and bt is ctx.attn_metadata.block_tables
    args, kwargs = prefill_spies["varlen"]
    assert args == (q, "k_full", "v_full") and kwargs["causal"] is True
    assert kwargs["cu_seqlens_k"] is ctx.attn_metadata.cu_seqlens_k
    assert kwargs["window_size"] == (-1, -1, 0)
    assert not hasattr(ctx.attn_metadata, "_dense_mxfp4_gather_inputs")  # not cached


def test_prefix_prefill_uses_the_builders_map(prefill_spies):
    ids = torch.tensor([0, 1], dtype=torch.int32)
    starts = torch.ones(2, dtype=torch.int32)
    md = prefill_md(has_cached=True, batch_id_per_k_token=ids, seq_starts=starts)
    make_impl().prefill_attention(
        torch.zeros(8, H, D),
        None,
        None,
        pages(),
        pages(),
        None,
        None,
        fwd_ctx(True, **md),
    )
    assert prefill_spies["gather"][4] is ids and prefill_spies["gather"][5] is starts


def _tbo_prefill(prefill_spies, **over):
    make_impl().prefill_attention(
        torch.zeros(8, H, D),
        None, None, pages(), pages(), None, None,
        fwd_ctx(True, **prefill_md(**over)),
    )  # fmt: skip
    return prefill_spies["gather"]


def test_tbo_no_cache_ubatch_takes_total_from_its_batch_ids(prefill_spies):
    """split_attn_metadata gives a no-cache ubatch total_kv=None but a sliced map."""
    ids = torch.tensor([0] * 5 + [1] * 3, dtype=torch.int32)
    g = _tbo_prefill(
        prefill_spies,
        total_kv=None,
        batch_id_per_k_token=ids,
        context_lens=torch.tensor([5, 3], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 5, 8], dtype=torch.int32),
    )
    assert g[6] == 8 and g[4] is ids and g[5].tolist() == [0, 0]


def test_tbo_no_cache_ubatch_without_a_map_derives_total_from_q(prefill_spies):
    # no cached prefix, so the 8 query rows _tbo_prefill feeds are the 8 keys
    cu_k = torch.tensor([0, 5, 8], dtype=torch.int32)
    g = _tbo_prefill(
        prefill_spies,
        has_cached=False,
        total_kv=None,
        context_lens=torch.tensor([5, 3], dtype=torch.int32),
        cu_seqlens_k=cu_k,
    )
    assert cu_k[-1].item() == 8 and g[6] == 8
    assert g[4].tolist() == [0] * 5 + [1] * 3


def test_tbo_prefix_hit_keeps_the_builders_total(prefill_spies):
    ids = torch.zeros(4, dtype=torch.int32)  # numel differs from total_kv on purpose
    g = _tbo_prefill(
        prefill_spies,
        has_cached=True,
        total_kv=12,
        batch_id_per_k_token=ids,
        seq_starts=torch.ones(2, dtype=torch.int32),
    )
    assert g[6] == 12 and g[4] is ids


def test_prefill_without_block_tables_raises(prefill_spies):
    with pytest.raises(RuntimeError, match="block_tables"):
        make_impl().prefill_attention(
            None, None, None, pages(), pages(), None, None,
            fwd_ctx(True, **prefill_md(block_tables=None)),
        )  # fmt: skip
    assert "gather" not in prefill_spies

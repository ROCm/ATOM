# SPDX-License-Identifier: MIT
"""EAGLE3's draft attention under ``--kv-cache-dtype fp4`` (CPU, no kernels).

Pinned here: which dense-impl methods the EAGLE3 impl overrides and which it
inherits; the norm-free write order (RoPE in place, then the K/V store at the
metadata's slots); the decode fold with ``plan_step_owner=None`` and a bf16 q; the
slot guard that stands between a skipped (slot < 0) token and the full-KV prefill
gather, on every prefill and never silently under capture; and every mode the impl
does not support. The kernels (``store_kv_mxfp4``, ``pa_decode``, the gather) have
their own GPU tests and are replaced by spies. Needs aiter and triton like the
dense impl's tests, so the CPU CI job skips it.
"""

import types

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops import attention_mha as am
from atom.model_ops.minimax_m3 import eagle3_attn_mxfp4 as mod
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, PAGE_TOKENS
from atom.model_ops.minimax_m3.store_mxfp4 import require_prefill_slots_written

H, HKV, D = 4, 2, 128
Impl = mod.Eagle3Mxfp4PagedAttentionImpl


@pytest.fixture(autouse=True)
def no_context_parallel(monkeypatch):
    """The CP world sizes read the engine config, which a unit test has none of."""
    import atom.distributed.dcp_utils as dcp
    import atom.distributed.pcp_utils as pcp

    monkeypatch.setattr(dcp, "get_dcp_world_size", lambda: 1)
    monkeypatch.setattr(pcp, "get_pcp_world_size", lambda: 1)


def make_impl(**over):
    """An impl without ``__init__`` (which wants a GPU); fields as it would set."""
    impl = object.__new__(Impl)
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
        "q_norm": None,  # EAGLE3: no q/k norm
        "k_norm": None,
        "rotary_emb": lambda position, q, k: (q, k),
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


# --- what is overridden, what is the dense impl's ----------------------------


def test_the_eagle3_impl_overrides_exactly_the_norm_free_parts():
    dense = am.MiniMaxM3DenseMxfp4PagedAttentionImpl
    for name in (
        "_check_supported",
        "rope_cache",
        "prefill_attention",
        "_dense_decode",
    ):
        assert getattr(Impl, name) is not getattr(dense, name), name
    for name in ("dispatch_backend", "_pages"):  # the reader and page checks are shared
        assert getattr(Impl, name) is getattr(dense, name), name


# --- unsupported modes -------------------------------------------------------


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
        ({"rotary_emb": None}, "rotary_emb"),
        ({"q_norm": nn.Identity(), "k_norm": nn.Identity()}, "norm-free"),
        ({"num_heads": 34, "num_kv_heads": 2}, "1..16 query heads"),
        ({"use_flash_layout": True}, "flash layout"),
    ],
)
def test_unsupported_modes_raise(over, why):
    with pytest.raises(NotImplementedError, match=why):
        make_impl(**over)._check_supported()


@pytest.mark.parametrize("which", ["dcp", "pcp"])
def test_context_parallel_raises(monkeypatch, which):
    import atom.distributed.dcp_utils as dcp
    import atom.distributed.pcp_utils as pcp

    owner, name = (
        (dcp, "get_dcp_world_size")
        if which == "dcp"
        else (
            pcp,
            "get_pcp_world_size",
        )
    )
    monkeypatch.setattr(owner, name, lambda: 2)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        make_impl()._check_supported()


def test_unified_attention_env_raises(monkeypatch):
    monkeypatch.setattr(mod.envs, "ATOM_USE_UNIFIED_ATTN", True, raising=False)
    with pytest.raises(NotImplementedError, match="ATOM_USE_UNIFIED_ATTN"):
        make_impl()._check_supported()


# --- writer ------------------------------------------------------------------


def split_qkv(tokens):
    """q, k, v as ``forward_impl`` hands them over: views of one qkv tensor."""
    qkv = torch.randn(tokens, (H + 2 * HKV) * D).to(torch.bfloat16)
    q, k, v = torch.split(qkv, [H * D, HKV * D, HKV * D], dim=-1)
    return (
        qkv,
        q.view(-1, H, D),
        k.view(-1, HKV, D),
        v.view(-1, HKV, D),
    )


@pytest.mark.parametrize("is_prefill", [False, True])
def test_rope_runs_in_place_then_kv_is_stored_at_the_metadata_slots(
    monkeypatch, is_prefill
):
    order = []

    def rotary(position, q, k):
        order.append("rope")
        q.add_(2)  # in place, on the views of the projection output
        k.add_(1)
        return q, k

    def store(k, v, k_pages, v_pages, slots):
        order.append("store")
        seen["k"], seen["v"] = k.clone(), v.clone()
        seen["pages"], seen["slots"], seen["strides"] = (
            (k_pages, v_pages),
            slots,
            k.stride(),
        )

    seen = {}
    monkeypatch.setattr(mod, "store_kv_mxfp4", store)
    tokens = 3
    qkv, q, k, v = split_qkv(tokens)
    q0, k0, v0 = q.clone(), k.clone(), v.clone()
    slots = torch.tensor([5, -1, 130])
    ctx = fwd_ctx(is_prefill, slot_mapping=slots)
    out = make_impl(rotary_emb=rotary).rope_cache(
        q, k, v, qkv, torch.arange(tokens), ctx
    )
    rq, rk, rv, kp, vp, k_scale, v_scale = out
    assert order == ["rope", "store"]
    # the store sees the ROTATED k and the raw v, K's own strides, the real pages
    torch.testing.assert_close(seen["k"], k0 + 1)
    torch.testing.assert_close(seen["v"], v0)
    assert seen["strides"][0] == (H + 2 * HKV) * D  # a view of qkv, not a copy
    assert seen["slots"] is slots
    assert seen["pages"][0] is kp is ctx.kv_cache_data["layer_0"].k_cache
    assert vp is ctx.kv_cache_data["layer_0"].v_cache
    # q stays bf16 (not e4m3) and is dense for the readers; k/v come back as given
    assert rq.dtype == torch.bfloat16 and rq.is_contiguous()
    assert rq.shape == (tokens, H, D)
    torch.testing.assert_close(rq, q0 + 2)
    assert (rk.data_ptr(), rv.data_ptr()) == (k.data_ptr(), v.data_ptr())
    assert (k_scale, v_scale) == (None, None)


def test_the_writer_refuses_a_non_mxfp4_pool_and_writes_nothing(monkeypatch):
    stored = []
    monkeypatch.setattr(mod, "store_kv_mxfp4", lambda *a: stored.append(a))
    qkv, q, k, v = split_qkv(2)
    ctx = fwd_ctx(False, slot_mapping=torch.arange(2))
    ctx.kv_cache_data["layer_0"].k_cache = torch.zeros(3, HKV, 128, 128).to(
        torch.float8_e4m3fn
    )
    with pytest.raises(RuntimeError, match="MXFP4 pages"):
        make_impl().rope_cache(q, k, v, qkv, torch.arange(2), ctx)
    assert not stored


def test_the_writer_needs_positions_and_a_bf16_qkv(monkeypatch):
    stored = []
    monkeypatch.setattr(mod, "store_kv_mxfp4", lambda *a: stored.append(a))
    qkv, q, k, v = split_qkv(2)
    ctx = fwd_ctx(False, slot_mapping=torch.arange(2))
    with pytest.raises(ValueError, match="positions"):
        make_impl().rope_cache(q, k, v, qkv, None, ctx)
    with pytest.raises(NotImplementedError, match="bf16"):
        make_impl().rope_cache(q.half(), k.half(), v.half(), qkv, torch.arange(2), ctx)
    assert not stored


# --- decode ------------------------------------------------------------------


@pytest.fixture
def decode_spy(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        am, "dense_pa_mxfp4", lambda *a, **k: seen.update(args=a, kw=k) or "out"
    )
    return seen


def test_decode_folds_q_len_rows_and_never_takes_a_plan_step_owner(decode_spy):
    impl = make_impl()
    md = {
        "max_seqlen_q": 2,
        "context_lens": torch.tensor([7, 130, 0], dtype=torch.int32),  # a padded row
        "block_tables": torch.arange(12, dtype=torch.int32).view(3, 4),
    }
    ctx = fwd_ctx(False, **md)
    q = torch.zeros(4, H, D, dtype=torch.bfloat16)  # 2 requests x q_len 2
    assert impl.dispatch_backend(ctx, q, None, None) == impl._dense_decode
    assert impl._dense_decode(q, None, None, "kp", "vp", None, None, ctx) == "out"
    q_seen, kp, vp, rows_ctx, rows_bt, scale = decode_spy["args"]
    assert q_seen is q and q_seen.dtype == torch.bfloat16  # bf16 q, not cast
    assert (kp, vp, scale) == ("kp", "vp", impl.scale)
    assert rows_ctx.tolist() == [6, 7, 129, 130]
    assert rows_bt.tolist() == [[0, 1, 2, 3]] * 2 + [[4, 5, 6, 7]] * 2
    # the draft's context_lens change under one shared metadata: no once-per-owner plan
    assert decode_spy["kw"] == {"plan_step_owner": None}


def test_a_plain_decode_step_passes_its_rows_through(decode_spy):
    ctx_lens = torch.tensor([9, 4], dtype=torch.int32)
    bt = torch.arange(8, dtype=torch.int32).view(2, 4)
    ctx = fwd_ctx(False, max_seqlen_q=1, context_lens=ctx_lens, block_tables=bt)
    make_impl()._dense_decode(
        torch.zeros(2, H, D, dtype=torch.bfloat16),
        None,
        None,
        "k",
        "v",
        None,
        None,
        ctx,
    )
    # one row a request, q_len 1: the same rows (sliced to the requests this step runs)
    assert torch.equal(decode_spy["args"][3], ctx_lens)
    assert torch.equal(decode_spy["args"][4], bt)


def test_a_ragged_decode_raises(decode_spy):
    ctx = fwd_ctx(
        False,
        max_seqlen_q=4,
        context_lens=torch.ones(2, dtype=torch.int32),
        block_tables=torch.zeros(2, 4, dtype=torch.int32),
    )
    with pytest.raises(NotImplementedError, match="q_len 4"):
        make_impl()._dense_decode(
            torch.zeros(6, H, D, dtype=torch.bfloat16),
            None,
            None,
            "k",
            "v",
            None,
            None,
            ctx,
        )
    assert not decode_spy


# --- prefill: the slot guard in front of the full-KV gather ------------------


def prefill_md(slots, **over):
    md = {
        "slot_mapping": torch.tensor(slots, dtype=torch.int64),
        "block_tables": torch.arange(8, dtype=torch.int32).view(2, 4),
        "cu_seqlens_q": torch.tensor([0, 3, 8], dtype=torch.int32),
        "cu_seqlens_k": torch.tensor([0, 3, 8], dtype=torch.int32),
        "context_lens": torch.tensor([3, 5], dtype=torch.int32),
        "total_kv": 8,
        "max_seqlen_q": 5,
        "max_seqlen_k": 5,
        "min_seqlen_q": 0,
        "dropout_p": 0.0,
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


def run_prefill(slots, k_pages=None, **over):
    ctx = fwd_ctx(True, **prefill_md(slots, **over))
    q = torch.zeros(8, H, D, dtype=torch.bfloat16)
    return make_impl().prefill_attention(
        q,
        None,
        None,
        k_pages if k_pages is not None else pages(),
        pages(),
        None,
        None,
        ctx,
    )


def test_a_fully_written_prefill_gathers_through_the_dense_path(prefill_spies):
    assert run_prefill(list(range(8))) == "o"
    *_, ids, starts, total = prefill_spies["gather"]
    assert total == 8 and ids.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert starts.tolist() == [0, 0]


@pytest.mark.parametrize("bad_slot", [-1, 3 * PAGE_TOKENS])
def test_a_skipped_token_stops_the_prefill_before_the_gather(prefill_spies, bad_slot):
    slots = list(range(8))
    slots[4] = bad_slot  # a real token of request 1 (a pool of 3 pages holds 384 slots)
    with pytest.raises(RuntimeError, match=f"token 4 .* slot {bad_slot}"):
        run_prefill(slots)
    assert "gather" not in prefill_spies and "varlen" not in prefill_spies


def test_pad_tokens_past_the_last_request_may_be_skipped(prefill_spies):
    # a TBO ubatch's pad tokens: beyond cu_seqlens_q[-1], so no request's K covers them
    slots = list(range(8)) + [-1, -1]
    ctx = fwd_ctx(True, **prefill_md(slots))
    out = make_impl().prefill_attention(
        torch.zeros(10, H, D, dtype=torch.bfloat16),
        None, None, pages(), pages(), None, None, ctx,
    )  # fmt: skip
    assert out == "o" and "gather" in prefill_spies


def test_a_prefill_in_a_captured_graph_is_refused_not_unchecked(
    monkeypatch, prefill_spies
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: True, raising=False
    )
    with pytest.raises(RuntimeError, match="captured graph"):
        run_prefill(list(range(8)))
    assert "gather" not in prefill_spies


def test_every_prefill_dispatch_reaches_the_guard(monkeypatch, prefill_spies):
    """dispatch_backend names ``prefill_attention`` for any prefill, and that is the
    only method that gathers: there is no other way to the full-KV read."""
    impl = make_impl()
    ctx = fwd_ctx(True, **prefill_md([-1] * 8))
    assert impl.dispatch_backend(ctx, None, None, None) == impl.prefill_attention
    fn = impl.dispatch_backend(ctx, None, None, None)
    with pytest.raises(RuntimeError, match="unwritten page"):
        fn(torch.zeros(8, H, D, dtype=torch.bfloat16), None, None, pages(), pages(), None, None, ctx)  # fmt: skip
    assert "gather" not in prefill_spies


# --- the guard itself --------------------------------------------------------


def test_the_guard_checks_only_request_tokens_and_the_pool_bound():
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    ok = torch.tensor([0, 1, 2, 3, -1, -1])  # tokens 4, 5 are a ubatch's pad
    require_prefill_slots_written(ok, cu, 6, num_pages=1)
    with pytest.raises(RuntimeError, match="token 2 .* slot -1"):
        require_prefill_slots_written(torch.tensor([0, 1, -1, 3, 0, 0]), cu, 6, 1)
    with pytest.raises(RuntimeError, match="slot 128"):
        require_prefill_slots_written(torch.tensor([0, 1, 2, 128]), cu, 4, 1)
    with pytest.raises(RuntimeError, match="covers 3 tokens"):
        require_prefill_slots_written(torch.tensor([0, 1, 2]), cu, 4, 1)

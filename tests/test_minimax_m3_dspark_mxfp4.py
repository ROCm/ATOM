# SPDX-License-Identifier: MIT
"""MiniMax-M3.1 DSpark draft under ``--kv-cache-dtype fp4`` (CPU, aiter faked).

The draft's K/V are MXFP4 pages like the target's: written by the aiter fused
writer, read per row by ``pa_decode`` over the page-128 table with an exact
window start. These pin what the draft hands those two calls; the kernels'
numerics are aiter's tests'.
"""

import sys
import types

import pytest
import torch

import atom.models.minimax_m3.dspark as m3
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES

NH, NKV, D, ROT = 16, 1, 128, 64
EPS = 1e-6


def _rotary():
    rope = torch.nn.Module()
    ang = torch.arange(64, dtype=torch.float32)[:, None] * torch.ones(ROT // 2)
    rope.register_buffer("cos_cache", ang.cos().to(torch.bfloat16)[:, None, None, :])
    rope.register_buffer("sin_cache", ang.sin().to(torch.bfloat16)[:, None, None, :])
    rope.rotary_dim = ROT
    return rope


def _attn(mxfp4=True, pages=3):
    a = object.__new__(m3.M3DSparkAttention)
    torch.nn.Module.__init__(a)
    a.mxfp4 = mxfp4
    a.num_heads, a.num_kv_heads, a.head_dim = NH, NKV, D
    a.q_size, a.kv_size, a.scaling = NH * D, NKV * D, D**-0.5
    a.kv_cache_dtype = "fp8"
    qw, kw = torch.rand(D).bfloat16(), torch.rand(D).bfloat16()
    a.q_norm = types.SimpleNamespace(weight=qw, variance_epsilon=EPS)
    a.k_norm = types.SimpleNamespace(weight=kw, variance_epsilon=EPS)
    a.rotary_emb = _rotary()
    a.qkv_proj = lambda x: torch.ones(x.shape[0], (NH + 2 * NKV) * D).bfloat16()
    a.o_proj = lambda x: x
    a.kv = m3.KVCacheBinding(0)
    if mxfp4:
        a.kv.k_cache = torch.zeros(pages, NKV, PAGE_BYTES, dtype=torch.uint8)
        a.kv.v_cache = torch.zeros(pages, NKV, PAGE_BYTES, dtype=torch.uint8)
    return a


def _no_triton(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fp8 Triton writer reached under fp4")

    monkeypatch.setattr(m3, "triton_fused_norm_rope_cache", boom)


def _record_writer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        m3.aiter,
        "fused_qknorm_idxrqknorm",
        lambda *a, **k: calls.append((a, k)),
        raising=False,
    )
    return calls


def test_context_write_hands_the_fused_writer_the_whole_qkv_and_mxfp4_pages(
    monkeypatch,
):
    _no_triton(monkeypatch)
    calls = _record_writer(monkeypatch)
    a = _attn()
    hidden = torch.zeros(5, 8, dtype=torch.bfloat16)
    pos = torch.arange(5)
    slots = torch.arange(5)
    fc = types.SimpleNamespace(attn_metadata=types.SimpleNamespace(slot_mapping=slots))
    monkeypatch.setattr(m3, "get_forward_context", lambda: fc)
    layer = object.__new__(m3.M3DSparkDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.self_attn = a
    layer.write_context_kv(hidden, pos)

    ((args, kw),) = calls
    qkv, qw, kw_, cos_sin, positions, nq, nkv, rot, eps = args[:9]
    assert qkv.shape == (5, (NH + 2 * NKV) * D)  # q is computed, not skipped
    assert qw is a.q_norm.weight and kw_ is a.k_norm.weight  # raw: kernel does 1 + w
    assert cos_sin.shape == (64, ROT) and positions is pos
    assert (nq, nkv, rot, eps) == (NH, NKV, ROT, EPS)
    # no index branch: heads 0, no index weights, no index cache or slots
    assert args[9:12] == (None, None, 0) and args[12] is slots
    assert args[13] is a.kv.k_cache and args[14] is a.kv.v_cache
    assert args[15] is None and args[16] == 128 and args[18:] == (None, None)
    q_out = args[17]
    assert q_out.dtype == torch.float8_e4m3fn and q_out.shape == (5, NH * D)
    assert kw == {
        "kv_cache_dtype": "mxfp4",
        "index_cache_dtype": "auto",
        "asm_layout": False,
    }  # no k_scale / v_scale, no skip_index_branch


def test_block_pass_writes_then_reads_the_exact_window_over_the_page_128_table(
    monkeypatch,
):
    _no_triton(monkeypatch)
    calls = _record_writer(monkeypatch)
    reads = []
    fake = types.SimpleNamespace(pa_decode=lambda *a, **k: reads.append((a, k)))
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.pa_decode", fake)
    a = _attn()
    rows = 6
    md = types.SimpleNamespace(
        slot_mapping=torch.arange(rows),
        dspark_bt_rows=torch.zeros(rows, 4, dtype=torch.int32),
        dspark_ctx_rows=torch.full((rows,), 300, dtype=torch.int32),
        dspark_start_rows=torch.arange(rows, dtype=torch.int32),
        dspark_max_span=300,
    )
    fc = types.SimpleNamespace(attn_metadata=md)
    monkeypatch.setattr(m3, "get_forward_context", lambda: fc)
    out = a.forward(torch.arange(rows), torch.zeros(rows, 8, dtype=torch.bfloat16))

    ((wargs, _),) = calls
    assert wargs[12] is md.slot_mapping  # the block's own slots
    ((rargs, rkw),) = reads
    out_buf, q, k_data, v_data, ctx, bt, _scale, qlen = rargs
    assert q.dtype == torch.float8_e4m3fn and q.shape == (rows, NH, D)
    assert q.data_ptr() == wargs[17].data_ptr()  # the writer's q, not a copy
    assert k_data.shape == (3, NKV, 128, 64) and v_data.shape == k_data.shape
    assert ctx is md.dspark_ctx_rows and bt is md.dspark_bt_rows and qlen == 1
    assert rkw["context_starts"] is md.dspark_start_rows
    assert rkw["max_context_length"] == 300 and rkw["work_plan"] is None
    assert rkw["key_scale"].shape == (3, NKV, 128, 4)
    assert out_buf.dtype == torch.bfloat16 and out.shape == (rows, NH * D)


@pytest.mark.parametrize(
    "k, v, scales, why",
    [
        (torch.zeros(2, 1, PAGE_BYTES), None, False, "dtype"),
        (torch.zeros(2, 2, PAGE_BYTES, dtype=torch.uint8), None, False, "heads"),
        (torch.zeros(2, 1, 4096, dtype=torch.uint8), None, False, "page bytes"),
        (torch.zeros(2, 1, PAGE_BYTES, dtype=torch.uint8), None, True, "scales"),
    ],
)
def test_a_cache_that_is_not_this_layers_mxfp4_pages_is_refused(k, v, scales, why):
    a = _attn()
    a.kv.k_cache = k
    a.kv.v_cache = k if v is None else v
    if scales:
        a.kv.k_scale = torch.zeros(1)
    with pytest.raises(ValueError):
        a._bound_cache()


def test_the_fp8_draft_still_writes_through_triton_and_never_the_fused_writer(
    monkeypatch,
):
    triton = []
    monkeypatch.setattr(
        m3, "triton_fused_norm_rope_cache", lambda *a, **k: triton.append((a, k))
    )
    calls = _record_writer(monkeypatch)
    a = _attn(mxfp4=False)
    seen = {}
    a._bound_cache = lambda: seen.setdefault("c", ("k", "v", "ks", "vs"))
    a.write_context_kv(
        torch.zeros(3, 8, dtype=torch.bfloat16), torch.arange(3), torch.arange(3)
    )
    assert not calls and len(triton) == 1
    assert triton[0][1]["num_heads"] == 0 and triton[0][1]["kv_cache_dtype"] == "fp8"


def test_mxfp4_refuses_a_head_dim_the_pages_do_not_hold(monkeypatch):
    monkeypatch.setattr(m3, "get_tensor_model_parallel_world_size", lambda: 4)
    config = types.SimpleNamespace(
        num_attention_heads=64, num_key_value_heads=4, head_dim=64
    )
    atom_config = types.SimpleNamespace(kv_cache_dtype="fp4")
    with pytest.raises(ValueError, match="head_dim 128"):
        m3.M3DSparkAttention(atom_config, config, None, layer_num=0)


def _rows(window, positions, ctx, real_rows, T, tables):
    rb = ctx.shape[0]
    n, W = rb * T, tables.shape[1]
    row_req = torch.arange(rb).repeat_interleave(T)
    tb = torch.zeros(n, W, dtype=torch.int32)
    ln = torch.zeros(n, dtype=torch.int32)
    st = torch.full((n,), -7, dtype=torch.int32)
    m3._mxfp4_rows(
        tables,
        ctx,
        positions.view(-1),
        row_req,
        window,
        real_rows,
        torch.zeros(n, dtype=torch.int64),
        tb,
        ln,
        st,
    )
    return tb, ln, st, row_req


@pytest.mark.parametrize("window", [0, 5, 128, 129, 300])
def test_each_row_reads_exactly_the_keys_of_its_window(window):
    """Token-exact: no page rounding, across 128-page boundaries."""
    T = 4
    anchors = torch.tensor([126, 127, 128, 255, 300])  # around page edges
    ctx = (anchors + 1 + T).to(torch.int32)
    positions = anchors[:, None] + torch.arange(T)
    tables = torch.arange(5 * 4, dtype=torch.int32).view(5, 4)
    tb, ln, st, row_req = _rows(window, positions, ctx, 5 * T, T, tables)
    for r in range(5 * T):
        p, req = int(positions.view(-1)[r]), int(row_req[r])
        want = range(max(p - window + 1, 0) if window else 0, int(ctx[req]))
        assert list(range(int(st[r]), int(ln[r]))) == list(want), (r, p, window)
        assert torch.equal(tb[r], tables[req])  # from logical token 0, unshifted


def test_pad_rows_are_empty_and_a_row_past_its_length_keeps_a_key():
    T = 2
    tables = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.int32)
    # requests 1 and 2 are padding (running_bs 3, one scheduled request)
    ctx = torch.tensor([200, 128, 128], dtype=torch.int32)
    positions = torch.tensor([[198, 199], [5000, 5001], [9, 9]])
    _, ln, st, _ = _rows(64, positions, ctx, 1 * T, T, tables)
    assert st[:2].tolist() == [135, 136] and ln[:2].tolist() == [200, 200]
    assert ln[2:].tolist() == [0] * 4 and st[2:].tolist() == [0] * 4
    # a real row whose position is past its allocated length still reads a key
    _, ln2, st2, _ = _rows(64, positions, ctx, 3 * T, T, tables)
    assert (st2 < ln2).all() and st2[2:4].tolist() == [127, 127]


def test_a_zero_length_request_is_an_empty_range_starting_at_zero():
    T = 2
    tables = torch.zeros(1, 2, dtype=torch.int32)
    _, ln, st, _ = _rows(
        64,
        torch.tensor([[300, 301]]),
        torch.tensor([0], dtype=torch.int32),
        T,
        T,
        tables,
    )
    assert ln.tolist() == [0, 0] and st.tolist() == [0, 0]


@pytest.mark.parametrize("window", [1, 5, 128, 129, 300])
def test_no_row_spans_more_than_the_bound_it_is_given(window):
    """The bound the reader schedules by covers the real widest row."""
    T = 4
    anchors = torch.tensor([0, 126, 127, 128, 255, 300, 1000])
    ctx = (anchors + 1 + T).to(torch.int32)
    positions = anchors[:, None] + torch.arange(T)
    tables = torch.zeros(len(anchors), 8, dtype=torch.int32)
    _, ln, st, _ = _rows(window, positions, ctx, len(anchors) * T, T, tables)
    m = object.__new__(m3.MiniMaxM3DSpark)
    m.sliding_window = window
    assert int((ln - st).max()) <= m._mxfp4_max_span(8, T)


@pytest.mark.parametrize(
    "window, T, cols, want",
    [(4096, 5, 512, 4101), (4096, 5, 16, 2048), (0, 5, 16, 2048), (0, 5, 512, 65536)],
)
def test_the_reader_is_given_the_span_not_the_context(window, T, cols, want):
    """A windowed row spans at most window + T; a whole-context row the table."""
    m = object.__new__(m3.MiniMaxM3DSpark)
    m.sliding_window = window
    assert m._mxfp4_max_span(cols, T) == want


@pytest.mark.parametrize(
    "cols, want",
    [
        (4093, 4093 * 128),
        (4094, 524032),
        (4095, 524033),
        (4096, 524033),
        (1 << 20, 524033),
    ],
)
def test_a_whole_context_row_hint_stays_within_what_aiter_stages(cols, want):
    """4094 table pages is the widest hint aiter stages whole (4096 page ids);
    from 4095 the unclamped hint (4095 * 128) would need 4098 and fail to compile.
    Only the hint is clamped, so a table that wide is never allocated here."""
    m = object.__new__(m3.MiniMaxM3DSpark)
    m.sliding_window = 0
    span = m._mxfp4_max_span(cols, 5)
    assert span == want
    assert 2 * -(-(span + 255) // 256) <= 4096


def test_the_clamped_hint_leaves_a_windowed_row_and_the_rows_untouched():
    """The window's own hint is already small; the rows' start/context, which
    decide the keys read, do not depend on the hint at all."""
    T, window, cols = 5, 4096, 4095
    m = object.__new__(m3.MiniMaxM3DSpark)
    m.sliding_window = window
    assert m._mxfp4_max_span(cols, T) == window + T
    anchors = torch.tensor([524000, 524200])
    ctx = (anchors + 1 + T).to(torch.int32)
    positions = anchors[:, None] + torch.arange(T)
    tables = torch.zeros(len(anchors), 1, dtype=torch.int32)
    _, ln, st, _ = _rows(window, positions, ctx, len(anchors) * T, T, tables)
    assert ln.tolist() == [524006] * 5 + [524206] * 5
    assert st.tolist() == (positions.flatten() - window + 1).tolist()

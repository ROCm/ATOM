# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-Flash-Next mono's binding of ATOM's caches (``state``) and the
per-step words of its tables (``step_meta``)."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.model_ops.attentions.gdn_attn import LINEAR_STATE_ROWS
from atom.model_ops.attentions.qwen4_exp_attn import QSA_ROWS
from atom.models.qwen4_exp.mono import state as S
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C
from atom.mono.runtime.consensus import MonoUnsupported

BF16, F32 = torch.bfloat16, torch.float32
KINDS = ["delta", "delta_ple", "delta", "qsa"]
SLOTS, PAGES = 2, 2
# each DeltaNet layer's pool row: bound in another order than the layers'
GDN_ROWS = {0: 2, 1: 0, 2: 1}


class _Module:
    """A layer's attention module: hashable by identity, as ATOM's are."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def _qsa_caches(pages=PAGES, device="cpu"):
    def cache(*page):
        return torch.zeros(pages, *page, dtype=BF16, device=device)

    return {
        "k_cache": cache(16, C.kv_heads, C.head_dim),
        "v_cache": cache(16, C.kv_heads, C.head_dim),
        "raw_key_cache": cache(16, 1, C.index_dim),
        "compressed_key_cache": cache(4, 1, C.index_dim),
    }


def _deployment(vk=True, spec_tokens=0):
    """A metadata builder and the decoder layers of a 4-layer model (layer 1
    the PLE layer, layer 3 QSA), its caches as ATOM allocates them (with
    ``spec_tokens`` MTP draft tokens)."""
    gdn, k = len(GDN_ROWS), spec_tokens
    runner = SimpleNamespace(
        mamba_k_cache=torch.zeros(gdn, SLOTS, 3 + k, C.delta_qkv, dtype=BF16),
        mamba_v_cache=torch.zeros(
            gdn, SLOTS, C.delta_v_heads, C.delta_v_dim, C.delta_k_dim, dtype=F32
        ),
        ple_conv_state=torch.zeros(SLOTS, C.hc_width, 9 + k, dtype=BF16),
    )
    layers, gdn_rows, qsa_rows = [], {}, {}
    for i, kind in enumerate(KINDS):
        if kind == "qsa":
            m = _Module(is_qsa_attention=True, rope_position_cache=None)
            m.__dict__.update(_qsa_caches())
            qsa_rows[m] = 0
            layers.append(SimpleNamespace(self_attn=m))
        else:
            policy = SimpleNamespace(decode=False, prefill=False)
            m = _Module(impl=SimpleNamespace(gdn_flydsl_policy=policy))
            gdn_rows[m] = GDN_ROWS[i]
            layers.append(SimpleNamespace(linear_attn=SimpleNamespace(attn=m)))
    builder = SimpleNamespace(
        model_runner=runner,
        pool_rows={LINEAR_STATE_ROWS: gdn_rows, QSA_ROWS: qsa_rows},
        gdn_state_vk=vk,
        replayssm=False,
    )
    return builder, layers


def _gdn(layers, i):
    return layers[i].linear_attn.attn


def test_the_caches_are_bound():
    builder, layers = _deployment()
    st = S.bind_state(builder, layers, KINDS)
    assert st.gdn_rows == GDN_ROWS
    assert st.slots == SLOTS
    assert st.conv is builder.model_runner.mamba_k_cache
    assert st.rec is builder.model_runner.mamba_v_cache
    assert st.ple_conv is builder.model_runner.ple_conv_state
    assert set(st.qsa) == {3}
    m = layers[3].self_attn
    names = {name: attr for name, (attr, _) in S.QSA_CACHES.items()}
    for name, attr in names.items():
        assert st.qsa[3][name] is getattr(m, attr)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_mtp_windows_are_bound(k):
    builder, layers = _deployment(spec_tokens=k)
    st = S.bind_state(builder, layers, KINDS, spec_tokens=k)
    assert st.conv is builder.model_runner.mamba_k_cache
    assert st.ple_conv is builder.model_runner.ple_conv_state


@pytest.mark.parametrize(
    "pools, bound, reason",
    [
        (1, 0, "GDN conv state"),
        (0, 1, "GDN conv state"),
        (1, 2, "GDN conv state"),
    ],
)
def test_windows_for_other_draft_counts_are_refused(pools, bound, reason):
    builder, layers = _deployment(spec_tokens=pools)
    with pytest.raises(MonoUnsupported, match=reason):
        S.bind_state(builder, layers, KINDS, spec_tokens=bound)


def test_a_kv_recurrent_state_is_refused():
    # ATOM's own FlyDSL decode off (gfx950) and VK storage not asked for
    builder, layers = _deployment(vk=False)
    with pytest.raises(
        MonoUnsupported, match="layer 0 GDN recurrent state not stored VK"
    ):
        S.bind_state(builder, layers, KINDS)


def test_atom_decode_stores_it_vk_too():
    builder, layers = _deployment(vk=False)
    for i in GDN_ROWS:
        _gdn(layers, i).impl.gdn_flydsl_policy.decode = True
    S.bind_state(builder, layers, KINDS)


def _unbound(builder, layers):
    del builder.pool_rows[QSA_ROWS][layers[3].self_attn]


def _shared(builder, layers):
    builder.pool_rows[LINEAR_STATE_ROWS][_gdn(layers, 2)] = 2


def _no_row(builder, layers):
    del builder.pool_rows[LINEAR_STATE_ROWS][_gdn(layers, 1)]


def _qsa(**fields):
    def change(builder, layers):
        layers[3].self_attn.__dict__.update(fields)

    return change


def _runner(**fields):
    def change(builder, layers):
        builder.model_runner.__dict__.update(fields)

    return change


def _spec_conv():
    # a speculative deployment keeps its draft rows in the conv window
    return torch.zeros(3, SLOTS, 3 + 1, C.delta_qkv, dtype=BF16)


@pytest.mark.parametrize(
    "change, reason",
    [
        (lambda b, layers: setattr(b, "replayssm", True), "ReplaySSM on"),
        (_runner(mamba_k_cache=None), "GDN state not allocated"),
        (_runner(mamba_k_cache=_spec_conv()), "GDN conv state"),
        (
            _runner(
                mamba_v_cache=torch.zeros(
                    3, SLOTS, C.delta_v_heads, C.delta_v_dim, C.delta_k_dim, dtype=BF16
                )
            ),
            "GDN recurrent state",
        ),
        (
            _runner(mamba_k_cache=torch.zeros(3, SLOTS, C.delta_qkv, 3, dtype=BF16).mT),
            "GDN conv state strides",
        ),
        (_runner(ple_conv_state=None), "PLE conv state not allocated"),
        (
            _runner(ple_conv_state=torch.zeros(SLOTS, C.hc_width, 12, dtype=BF16)),
            "PLE conv state",
        ),
        (_qsa(is_qsa_attention=False), "layer 3 not QSA"),
        (_unbound, "layer 3 QSA caches unbound"),
        (_qsa(rope_position_cache=torch.zeros(1)), "layer 3 caches mRoPE positions"),
        (
            _qsa(k_cache=torch.zeros(PAGES, 16, 2, 256, dtype=torch.float8_e4m3fn)),
            "layer 3 k_cache",
        ),
        (_qsa(v_cache=None), "layer 3 v_cache not allocated"),
        (
            _qsa(compressed_key_cache=_qsa_caches(PAGES + 1)["compressed_key_cache"]),
            "page counts differ",
        ),
        # a cache past a buffer resource's 32-bit reach (shape only)
        (
            _qsa(**_qsa_caches(S.MAX_FIELD_BYTES // (16 * 2 * 256 * 2), "meta")),
            "layer 3 k_cache: .* bytes per field",
        ),
        (_no_row, "layer 1 GDN row None"),
        (_shared, "GDN rows shared"),
    ],
)
def test_a_cache_not_as_the_kernels_address_it_is_refused(change, reason):
    builder, layers = _deployment()
    change(builder, layers)
    with pytest.raises(MonoUnsupported, match=reason):
        S.bind_state(builder, layers, KINDS)


# ------------------------------------------------------------------ step_meta
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="step_meta is a Triton kernel"
)

FILL = 7  # every word the step does not write
NULL_SLOTS = {"conv": 11 << 20, "rec": 13 << 20, "ple": 17 << 20, "pages": 19 << 20}
NULL_CACHES = [23 << 20, 29 << 20, 31 << 20, 37 << 20]


def _nulls():
    caches = torch.tensor(NULL_CACHES, dtype=torch.int64, device="cuda")
    return NULL_SLOTS | {"caches": caches}


def _tables(rows, n_delta=3, n_qsa=2):
    from atom.models.qwen4_exp.mono import step_meta as M
    from atom.models.qwen4_exp.mono.runner import Tables

    def table(*shape):
        return torch.full(shape, FILL, dtype=torch.int64, device="cuda")

    return Tables(
        pos=torch.zeros(rows, dtype=torch.int32, device="cuda"),
        delta=table(n_delta, rows, M.DELTA_WORDS),
        ple=table(rows, M.PLE_WORDS),
        qsa=table(n_qsa, rows, M.QSA_WORDS),
        gdn_conv=torch.tensor(
            [(1 + k) << 32 for k in range(n_delta)], dtype=torch.int64, device="cuda"
        ),
        gdn_rec=torch.tensor(
            [(9 + k) << 32 for k in range(n_delta)], dtype=torch.int64, device="cuda"
        ),
        qsa_caches=torch.tensor(
            [[(40 + 4 * q + c) << 32 for c in range(4)] for q in range(n_qsa)],
            dtype=torch.int64,
            device="cuda",
        ),
        conv_slot=3 * C.delta_qkv * 2,
        rec_slot=C.delta_v_heads * C.delta_v_dim * C.delta_k_dim * 4,
        ple_base=5 << 36,
        ple_slot=C.hc_width * 9 * 2,
    )


def _want(t, positions, s_in, s_out, block_tables):
    """The tables ``write_step_meta`` leaves, on the host."""
    from atom.models.qwen4_exp.mono import step_meta as M

    rows = t.pos.shape[0]
    delta, ple, qsa = (x.cpu().clone() for x in (t.delta, t.ple, t.qsa))
    conv, rec = t.gdn_conv.tolist(), t.gdn_rec.tolist()
    pos = positions[:rows].to(torch.int32).cpu()
    nulls = NULL_SLOTS

    def addr(slot, base, stride, null):
        return base + slot * stride if slot >= 0 else null

    for r in range(rows):
        i, o = int(s_in[r]), int(s_out[r])
        for k in range(delta.shape[0]):
            delta[k, r] = torch.tensor(
                [
                    addr(i, conv[k], t.conv_slot, nulls["conv"]),
                    addr(o, conv[k], t.conv_slot, nulls["conv"]),
                    addr(i, rec[k], t.rec_slot, nulls["rec"]),
                    addr(o, rec[k], t.rec_slot, nulls["rec"]),
                ]
            )
        ple[r, M.PLE_CONV_IN] = addr(i, t.ple_base, t.ple_slot, nulls["ple"])
        ple[r, M.PLE_CONV_IN + 1] = addr(o, t.ple_base, t.ple_slot, nulls["ple"])
        caches = slice(M.QSA_KEY, M.QSA_KEY + len(M.QSA_CACHES))
        if o >= 0:
            page_row = block_tables.stride(0) * block_tables.element_size()
            qsa[:, r, M.QSA_PAGES] = block_tables.data_ptr() + r * page_row
            qsa[:, r, caches] = t.qsa_caches.cpu()
        else:
            # a padded row: the null block table and caches, at PAD_POS
            pos[r] = M.PAD_POS
            qsa[:, r, M.QSA_PAGES] = nulls["pages"]
            qsa[:, r, caches] = torch.tensor(NULL_CACHES)
    return pos, delta, ple, qsa


@requires_gpu
@pytest.mark.parametrize(
    "s_in, s_out",
    [
        ([1], [1]),  # one request, its slot read and written in place
        ([0], [2]),  # a forked prefix: read one slot, write another
        ([1, -1], [1, -1]),  # a padded row
        ([3, 0, -1, 2], [3, 1, -1, 2]),
    ],
)
def test_step_meta_writes_the_step_words(s_in, s_out):
    from atom.models.qwen4_exp.mono.step_meta import write_step_meta

    rows = len(s_in)
    t = _tables(rows)
    counter = torch.tensor([41], dtype=torch.int32, device="cuda")
    # ATOM's persistent buffers: wider than the step
    positions = torch.arange(100, 108, dtype=torch.int64, device="cuda")
    slots_in = torch.full((8,), 5, dtype=torch.int32, device="cuda")
    slots_out = torch.full((8,), 5, dtype=torch.int32, device="cuda")
    slots_in[:rows] = torch.tensor(s_in, dtype=torch.int32)
    slots_out[:rows] = torch.tensor(s_out, dtype=torch.int32)
    block_tables = torch.zeros(8, 10, dtype=torch.int32, device="cuda")

    want = _want(t, positions, s_in, s_out, block_tables)
    write_step_meta(counter, positions, slots_in, slots_out, block_tables, t, _nulls())
    torch.cuda.synchronize()
    assert int(counter) == 42
    for got, w in zip((t.pos, t.delta, t.ple, t.qsa), want):
        assert torch.equal(got.cpu(), w)


# ------------------------------------------------------------------ verify
PLE_CHANNELS = 3 * 1024 + 5  # not a multiple of the gather's block


def _ple_pool(k, slots=3):
    g = torch.Generator().manual_seed(k)
    pool = torch.randn(slots, PLE_CHANNELS, 9 + k, generator=g).to(BF16)
    return pool.cuda()


@requires_gpu
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_verify_meta_writes_the_verify_words(k, accepted):
    import dataclasses

    from atom.models.qwen4_exp.mono import step_meta as M

    rows = k + 1
    if accepted > rows:
        pytest.skip("a request accepts at most k + 1 tokens")
    # the conv window holds 3 + k rows of QKV bf16
    row = C.delta_qkv * 2
    t = dataclasses.replace(_tables(rows), conv_slot=(rows + 2) * row)
    counter = torch.tensor([41], dtype=torch.int32, device="cuda")
    positions = torch.arange(100, 108, dtype=torch.int64, device="cuda")
    # ATOM's spec slots: the request's k + 1, the request's slot 0 first
    spec = torch.tensor([[2, 0, 1, 3][:rows], [5] * rows], dtype=torch.int32)
    spec = spec.cuda()
    n_acc = torch.tensor([accepted, 1], dtype=torch.int32, device="cuda")
    block_tables = torch.zeros(8, 10, dtype=torch.int32, device="cuda")
    pool = _ple_pool(k)
    before = pool.clone()
    window = torch.zeros(PLE_CHANNELS, M.PLE_COLS, dtype=BF16, device="cuda")

    delta, ple, qsa = (x.cpu().clone() for x in (t.delta, t.ple, t.qsa))
    s, a = spec[0].tolist(), accepted
    for d, (conv, rec) in enumerate(zip(t.gdn_conv.tolist(), t.gdn_rec.tolist())):
        base = conv + s[0] * t.conv_slot
        for r in range(rows):
            delta[d, r] = torch.tensor(
                [
                    base + (a - 1) * row,
                    base + r * row,
                    rec + s[a - 1] * t.rec_slot,
                    rec + s[r] * t.rec_slot,
                ]
            )
    qsa[:, 0, M.QSA_PAGES] = block_tables.data_ptr()

    M.write_verify_meta(
        counter, positions, spec[0], n_acc, block_tables, t, pool, window
    )
    torch.cuda.synchronize()
    assert int(counter) == 42
    assert torch.equal(t.pos.cpu(), positions[:rows].to(torch.int32).cpu())
    assert torch.equal(t.delta.cpu(), delta)
    assert torch.equal(t.ple.cpu(), ple)
    assert torch.equal(t.qsa.cpu(), qsa)
    assert torch.equal(window, pool[s[0], :, a - 1 : a + 8])
    assert torch.equal(pool, before)


@requires_gpu
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_a_verify_leaves_atoms_ple_window(k, accepted):
    """Gather, the seq build's snapshots (window after each row), commit:
    ATOM's update, the 8 columns from ``accepted`` on, then the k + 1
    inputs."""
    from atom.models.qwen4_exp.mono import step_meta as M

    rows = k + 1
    if accepted > rows:
        pytest.skip("a request accepts at most k + 1 tokens")
    t = _tables(rows)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    positions = torch.zeros(rows, dtype=torch.int64, device="cuda")
    spec = torch.tensor([1, 0, 2, 3][:rows], dtype=torch.int32, device="cuda")
    n_acc = torch.tensor([accepted], dtype=torch.int32, device="cuda")
    block_tables = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    pool = _ple_pool(k)
    old = pool.clone()
    window = torch.zeros(PLE_CHANNELS, M.PLE_COLS, dtype=BF16, device="cuda")
    M.write_verify_meta(counter, positions, spec, n_acc, block_tables, t, pool, window)

    x = torch.randn(PLE_CHANNELS, rows, device="cuda").to(BF16)
    snaps = torch.stack(
        [torch.cat([window[:, r + 1 :], x[:, : r + 1]], 1) for r in range(rows)]
    )
    M.commit_ple(pool, snaps.contiguous(), spec)
    torch.cuda.synchronize()

    s0 = int(spec[0])
    want = old.clone()
    want[s0] = torch.cat([old[s0, :, accepted : accepted + 8], x], 1)
    assert torch.equal(pool, want)


@requires_gpu
@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_draft_meta_writes_the_draft_words(rows):
    """A draft step: the draft's counter, each row's cache row, and every
    row's QSA table pointed at the request's block-table row; nothing else."""
    from atom.models.qwen4_exp.mono import step_meta as M

    counter = torch.tensor([5], dtype=torch.int32, device="cuda")
    # ATOM's qsa logical positions: the rows' cache rows, then padding
    logical = torch.tensor([300, 301, 302, 303, -1], dtype=torch.int64, device="cuda")
    block_tables = torch.zeros(2, 10, dtype=torch.int32, device="cuda")
    pos = torch.full((rows,), FILL, dtype=torch.int32, device="cuda")
    qsa = torch.full((rows, M.QSA_WORDS), FILL, dtype=torch.int64, device="cuda")
    want = qsa.cpu().clone()
    want[:, M.QSA_PAGES] = block_tables.data_ptr()

    M.write_draft_meta(counter, logical, block_tables, pos, qsa)
    torch.cuda.synchronize()
    assert int(counter) == 6
    assert pos.tolist() == [300, 301, 302, 303][:rows]
    assert torch.equal(qsa.cpu(), want)


@requires_gpu
def test_a_shared_expert_slot_is_laid_out_as_atoms_fused_one():
    """``weights.expert_slot`` of a dense shared expert equals the slot ATOM's
    MXFP4 MoE gives it fused: experts padded to 768 (gate rows from 0, up
    rows from 768, zero scales), then shuffled per expert."""
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    from atom.models.qwen4_exp.mono import weights as W

    H, P, inter, E = C.hidden, W.MOE_INTER_PADDED, C.shared_inter, 3
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    g = torch.Generator(device="cuda").manual_seed(0)

    def rand(*shape):
        return torch.randint(0, 256, shape, generator=g, **u8)

    w13, s13 = torch.zeros(E, 2 * P, H // 2, **u8), torch.zeros(E, 2 * P, H // 32, **u8)
    w2, s2 = torch.zeros(E, H, P // 2, **u8), torch.zeros(E, H, P // 32, **u8)
    for e in range(E):
        for half in range(2):
            rows = slice(half * P, half * P + inter)
            w13[e, rows], s13[e, rows] = rand(inter, H // 2), rand(inter, H // 32)
        w2[e, :, : inter // 2], s2[e, :, : inter // 32] = (
            rand(H, inter // 2),
            rand(H, inter // 32),
        )
    last = E - 1
    dense = (
        torch.cat([w13[last, :inter], w13[last, P : P + inter]]),
        torch.cat([s13[last, :inter], s13[last, P : P + inter]]),
        w2[last, :, : inter // 2].contiguous(),
        s2[last, :, : inter // 32].contiguous(),
    )
    fused = (
        moe_shuffle_weight(w13, E, is_guinterleave=True, gate_up=True)[last],
        moe_shuffle_scale(s13.view(E * 2 * P, -1), E, True, True).view(E, -1)[last],
        moe_shuffle_weight(w2, E, is_guinterleave=True, gate_up=False)[last],
        moe_shuffle_scale(s2.view(E * H, -1), E, True, False).view(E, -1)[last],
    )
    for got, want in zip(W.expert_slot(*dense), fused):
        assert torch.equal(got, want)

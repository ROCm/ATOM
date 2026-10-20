# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Qwen3.8-Flash-Next mono's MTP steps of several requests: which
forwards are routed (``groups``), how the wrappers take them, and the per-step words of each group (``group_meta``)
against the single-request ones (``step_meta``)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.qwen4_exp.mono import dispatch, groups
from atom.models.qwen4_exp.mono.kernels.config import DEFAULT as C

BF16 = torch.bfloat16


def _context(
    monkeypatch,
    *,
    s=2,
    rows_per_group=None,
    requests=2,
    prefill=False,
    draft=False,
    context=True,
    ubatch=None,
    qsa=True,
    ple=True,
    gdn=True,
    spec=True,
    accepted=True,
    max_seqlen_q=None,
    decodes=0,
):
    """A verify (or with ``draft`` a draft) step's forward context:
    ``requests`` spec decodes of ``s`` (k + 1) tokens each."""
    gdn_md = None
    if gdn:
        gdn_md = SimpleNamespace(
            spec_state_indices_tensor=torch.zeros(4, s) if spec else None,
            num_spec_decodes=requests,
            num_decodes=decodes,
            num_accepted_tokens=torch.ones(4) if accepted else None,
        )
    fwd = SimpleNamespace(
        context=(
            SimpleNamespace(is_prefill=prefill, is_draft=draft) if context else None
        ),
        ubatch_slices=ubatch,
        attn_metadata=SimpleNamespace(
            qsa_metadata=object() if qsa else None,
            ple_metadata=object() if ple else None,
            gdn_metadata=gdn_md,
            max_seqlen_q=s if max_seqlen_q is None else max_seqlen_q,
        ),
    )
    monkeypatch.setattr(groups, "get_forward_context", lambda: fwd)
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: fwd)
    return fwd


def _ids(n):
    return torch.zeros(n, dtype=torch.int64)


# ------------------------------------------------------------------ verify
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize(
    "graph, requests", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3), (4, 1)]
)
def test_a_verify_of_several_requests_is_routed(monkeypatch, k, graph, requests):
    """G groups of k + 1 rows, the trailing ones padding under a graph."""
    s = k + 1
    _context(monkeypatch, s=s, requests=requests)
    assert groups.group_step_supported(_ids(graph * s), None, s, 4)


@pytest.mark.parametrize(
    "rows, max_groups, kwargs",
    [
        # one request: the single-request path's
        (2, 4, {"requests": 1}),
        # more groups than allowed, or than rows
        (6, 2, {"requests": 3}),
        (10, 4, {"requests": 5}),
        (4, 4, {"requests": 3}),
        (4, 1, {}),
        # rows not whole groups, query length not k + 1
        (5, 4, {}),
        (4, 4, {"max_seqlen_q": 1}),
        (4, 4, {"max_seqlen_q": 4}),
        # no request, plain decodes, missing verify inputs
        (4, 4, {"requests": 0}),
        (4, 4, {"decodes": 1}),
        (4, 4, {"spec": False}),
        (4, 4, {"accepted": False}),
        # not a decode step, or warmup and profiling forwards
        (4, 4, {"prefill": True}),
        (4, 4, {"context": False}),
        (4, 4, {"ubatch": object()}),
        (4, 4, {"qsa": False}),
        (4, 4, {"ple": False}),
        (4, 4, {"gdn": False}),
    ],
)
def test_other_verify_steps_are_not_grouped(monkeypatch, rows, max_groups, kwargs):
    _context(monkeypatch, s=2, **kwargs)
    assert not groups.group_step_supported(_ids(rows), None, 2, max_groups)


def test_grouped_embeddings_in_are_not_routed(monkeypatch):
    _context(monkeypatch)
    assert not groups.group_step_supported(_ids(4), torch.zeros(4, 8), 2, 4)


def test_without_mtp_nothing_is_grouped(monkeypatch):
    _context(monkeypatch, s=1, max_seqlen_q=1)
    assert not groups.group_step_supported(_ids(2), None, 0, 4)
    assert not groups.group_step_supported(_ids(2), None, 1, 4)


# ------------------------------------------------------------------ draft
@pytest.mark.parametrize(
    "k, requests, step, rows",
    [
        (1, 2, 0, 4),
        (1, 4, 0, 8),
        (2, 3, 0, 9),
        (2, 3, 1, 3),
        (3, 2, 0, 8),
        (3, 4, 0, 16),
        (3, 2, 1, 2),
        (3, 4, 2, 4),
    ],
)
def test_draft_steps_of_several_requests_are_routed(
    monkeypatch, k, requests, step, rows
):
    s = k + 1
    _context(
        monkeypatch,
        s=s,
        requests=requests,
        draft=True,
        max_seqlen_q=s if step == 0 else 1,
    )
    assert groups.group_draft_step_supported(_ids(rows), None, s, 4)


@pytest.mark.parametrize(
    "k, rows, max_groups, kwargs",
    [
        # one request: the single-request path's
        (2, 3, 4, {"requests": 1}),
        (2, 1, 4, {"requests": 1, "max_seqlen_q": 1}),
        # too many requests
        (1, 6, 2, {"requests": 3}),
        # the rows are not the requests' (padded, or a step that is neither)
        (1, 8, 4, {"requests": 3}),
        (2, 4, 4, {"requests": 3, "max_seqlen_q": 1}),
        (2, 6, 4, {"requests": 2, "max_seqlen_q": 2}),
        # k = 1 has no later steps
        (1, 2, 4, {"requests": 2, "max_seqlen_q": 1}),
        # the target's verify, the drafter's prefill, no context
        (1, 4, 4, {"draft": False}),
        (1, 4, 4, {"prefill": True}),
        (1, 4, 4, {"context": False}),
        (1, 4, 4, {"ubatch": object()}),
        (1, 4, 4, {"qsa": False}),
        (1, 4, 4, {"gdn": False}),
        (1, 4, 4, {"decodes": 1}),
    ],
)
def test_other_draft_steps_are_not_grouped(monkeypatch, k, rows, max_groups, kwargs):
    s = k + 1
    _context(monkeypatch, **({"s": s, "draft": True} | kwargs))
    assert not groups.group_draft_step_supported(_ids(rows), None, s, max_groups)


# ------------------------------------------------------------------ wrappers
class _Model(nn.Module):
    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        return "native"


class _Draft(nn.Module):
    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None):
        return "atom"


class _Lazy:
    def __init__(self, out):
        self.runner = SimpleNamespace(forward=lambda ids, x: out)
        self.asked = []

    def ready(self, rows):
        self.asked.append(rows)
        return True


def _config(k):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=k)
    )


def test_a_batched_verify_is_routed(monkeypatch):
    mono = dispatch.MonoDecodeModel(_Model(), _config(3), SimpleNamespace())
    mono._mono = lazy = _Lazy("mono")
    _context(monkeypatch, s=4, requests=2)
    assert mono(_ids(8), _ids(8)) == "mono"
    assert lazy.asked == [8]


def test_a_batched_draft_step_is_routed(monkeypatch):
    wrapper = dispatch.MonoDraftModel(_Draft(), _config(3), SimpleNamespace())
    wrapper._mono = _Lazy("mono")
    _context(monkeypatch, s=4, requests=2, draft=True, max_seqlen_q=1)
    assert wrapper(_ids(2), _ids(2), torch.zeros(2, 8)) == "mono"


def test_the_wrappers_group_up_to_four_requests():
    mono = dispatch.MonoDecodeModel(_Model(), _config(2), SimpleNamespace())
    assert mono.groups == 4
    wrapper = dispatch.MonoDraftModel(_Draft(), _config(2), SimpleNamespace())
    assert wrapper.groups == 4


def test_without_mtp_the_decode_wrapper_does_not_group():
    config = SimpleNamespace(speculative_config=None)
    assert dispatch.MonoDecodeModel(_Model(), config, SimpleNamespace()).groups == 1


def test_the_wrappers_create_their_runners(monkeypatch):
    for name in [
        "Qwen4MonoDecodeRunner",
        "Qwen4MonoGroupRunner",
        "Qwen4MonoDraftRunner",
    ]:
        monkeypatch.setattr(dispatch, name, lambda *a, name=name: name)
    decode = dispatch.MonoDecodeModel(
        _Model(), SimpleNamespace(speculative_config=None), SimpleNamespace()
    )
    mono = dispatch.MonoDecodeModel(_Model(), _config(1), SimpleNamespace())
    wrapper = dispatch.MonoDraftModel(_Draft(), _config(1), SimpleNamespace())
    assert decode._mono._create() == "Qwen4MonoDecodeRunner"
    assert mono._mono._create() == "Qwen4MonoGroupRunner"
    assert wrapper._mono._create() == "Qwen4MonoDraftRunner"


# ------------------------------------------------------------------ group_meta
requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="group_meta is a Triton kernel"
)

FILL = 7
NULL_SLOTS = {"conv": 11 << 20, "rec": 13 << 20, "ple": 17 << 20, "pages": 19 << 20}
NULL_CACHES = [23 << 20, 29 << 20, 31 << 20, 37 << 20]
PLE_CHANNELS = 3 * 1024 + 5  # not a multiple of the gather's block


def _nulls():
    caches = torch.tensor(NULL_CACHES, dtype=torch.int64, device="cuda")
    return NULL_SLOTS | {"caches": caches}


def _tables(rows, s, n_delta=3, n_qsa=2):
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
        # the conv window holds 3 + k rows of QKV bf16
        conv_slot=(s + 2) * C.delta_qkv * 2,
        rec_slot=C.delta_v_heads * C.delta_v_dim * C.delta_k_dim * 4,
        ple_base=5 << 36,
        ple_slot=C.hc_width * 9 * 2,
    )


def _ple_pool(k, slots=17):
    g = torch.Generator().manual_seed(k)
    pool = torch.randn(slots, PLE_CHANNELS, 9 + k, generator=g).to(BF16)
    return pool.cuda()


def _step(k, graph, requests):
    """ATOM's persistent verify inputs for ``requests`` requests in a
    ``graph``-request step: each request its own spec slots and accepted
    count, the rows past them padding (slots -1, accepted 1)."""
    s = k + 1
    spec = torch.full((8, s), -1, dtype=torch.int32)
    for g in range(requests):
        spec[g] = torch.tensor([(4 * g + 3 * r + 1) % 16 for r in range(s)])
    n_acc = torch.ones(8, dtype=torch.int32)
    n_acc[:requests] = torch.tensor([1 + (g * 2 + k) % s for g in range(requests)])
    positions = torch.arange(100, 100 + 8 * s, dtype=torch.int64)
    for g in range(graph):
        positions[g * s : (g + 1) * s] += 1000 * g
    block_tables = torch.zeros(8, 10, dtype=torch.int32)
    return spec.cuda(), n_acc.cuda(), positions.cuda(), block_tables.cuda()


@requires_gpu
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("graph, requests", [(2, 2), (2, 1), (3, 3), (4, 3), (4, 4)])
def test_each_group_gets_its_requests_verify_words(k, graph, requests):
    """Each live group's rows: ``write_verify_meta``'s words for its request
    alone; each padded group's the null state."""
    from atom.models.qwen4_exp.mono import step_meta as M
    from atom.models.qwen4_exp.mono.group_meta import write_group_verify_meta

    s = k + 1
    rows = graph * s
    t = _tables(rows, s)
    counter = torch.tensor([41], dtype=torch.int32, device="cuda")
    spec, n_acc, positions, block_tables = _step(k, graph, requests)
    pool = _ple_pool(k)
    before = pool.clone()
    windows = torch.full((graph, PLE_CHANNELS, M.PLE_COLS), 3, dtype=BF16).cuda()
    write_group_verify_meta(
        counter, positions, spec, n_acc, block_tables, t, pool, windows, _nulls()
    )
    torch.cuda.synchronize()
    assert int(counter) == 42
    assert torch.equal(pool, before)
    # the PLE table is the runner's, fixed when the width is prepared
    assert torch.equal(t.ple, torch.full_like(t.ple, FILL))

    page_row = block_tables.stride(0) * block_tables.element_size()
    caches = slice(M.QSA_KEY, M.QSA_KEY + len(M.QSA_CACHES))
    row = C.delta_qkv * 2
    for g in range(graph):
        mine = slice(g * s, (g + 1) * s)
        if g < requests:
            one = _tables(s, s)
            n = torch.tensor([41], dtype=torch.int32, device="cuda")
            window = torch.zeros(PLE_CHANNELS, M.PLE_COLS, dtype=BF16, device="cuda")
            M.write_verify_meta(
                n,
                positions[g * s :],
                spec[g],
                n_acc[g:],
                block_tables[g:],
                one,
                pool,
                window,
            )
            torch.cuda.synchronize()
            assert torch.equal(t.pos[mine], one.pos)
            assert torch.equal(t.delta[:, mine], one.delta)
            assert torch.equal(windows[g], window)
            pages = block_tables.data_ptr() + g * page_row
            assert (t.qsa[:, mine, M.QSA_PAGES] == pages).all()
            want = t.qsa_caches[:, None, :].expand(-1, s, -1)
            assert torch.equal(t.qsa[:, mine, caches], want)
        else:
            pad = torch.arange(s, dtype=torch.int32, device="cuda") + M.PAD_POS
            assert torch.equal(t.pos[mine], pad)
            null_conv, null_rec = NULL_SLOTS["conv"], NULL_SLOTS["rec"]
            for r in range(s):
                want = [null_conv, null_conv + r * row, null_rec, null_rec]
                assert t.delta[:, g * s + r].tolist() == [want] * t.delta.shape[0]
            assert (windows[g] == 0).all()
            assert (t.qsa[:, mine, M.QSA_PAGES] == NULL_SLOTS["pages"]).all()
            want = torch.tensor(NULL_CACHES, device="cuda")
            assert (t.qsa[:, mine, caches] == want).all()
        # every other QSA word is the runner's
        rest = torch.ones(M.QSA_WORDS, dtype=torch.bool)
        rest[M.QSA_PAGES] = False
        rest[caches] = False
        assert (t.qsa[:, mine][..., rest.cuda()] == FILL).all()


@requires_gpu
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("graph, requests", [(2, 2), (3, 2), (4, 4)])
def test_each_live_group_leaves_atoms_ple_window(k, graph, requests):
    """Gather, the seq build's snapshots, commit: each request's window as
    ``commit_ple`` leaves it for that request alone; no other slot moves."""
    from atom.models.qwen4_exp.mono import step_meta as M
    from atom.models.qwen4_exp.mono.group_meta import (
        commit_ple_groups,
        write_group_verify_meta,
    )

    s = k + 1
    rows = graph * s
    t = _tables(rows, s)
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    spec, n_acc, positions, block_tables = _step(k, graph, requests)
    pool = _ple_pool(k)
    old = pool.clone()
    windows = torch.zeros(graph, PLE_CHANNELS, M.PLE_COLS, dtype=BF16, device="cuda")
    write_group_verify_meta(
        counter, positions, spec, n_acc, block_tables, t, pool, windows, _nulls()
    )
    x = torch.randn(graph, PLE_CHANNELS, s, device="cuda").to(BF16)
    snaps = torch.stack(
        [
            torch.cat([windows[g][:, r + 1 :], x[g][:, : r + 1]], 1)
            for g in range(graph)
            for r in range(s)
        ]
    )
    commit_ple_groups(pool, snaps.contiguous(), spec)
    torch.cuda.synchronize()

    want = old.clone()
    for g in range(requests):
        s0, a = int(spec[g, 0]), int(n_acc[g])
        want[s0] = torch.cat([old[s0, :, a : a + 8], x[g]], 1)
    assert torch.equal(pool, want)


@requires_gpu
@pytest.mark.parametrize(
    "rows, group", [(4, 2), (8, 4), (12, 4), (9, 3), (3, 1), (4, 1)]
)
def test_draft_rows_take_their_requests_pages(rows, group):
    from atom.models.qwen4_exp.mono import step_meta as M
    from atom.models.qwen4_exp.mono.group_meta import write_group_draft_meta

    counter = torch.tensor([5], dtype=torch.int32, device="cuda")
    logical = torch.arange(300, 320, dtype=torch.int64, device="cuda")
    block_tables = torch.zeros(4, 10, dtype=torch.int32, device="cuda")
    pos = torch.full((rows,), FILL, dtype=torch.int32, device="cuda")
    qsa = torch.full((rows, M.QSA_WORDS), FILL, dtype=torch.int64, device="cuda")
    want = qsa.cpu().clone()
    page_row = block_tables.stride(0) * block_tables.element_size()
    for r in range(rows):
        want[r, M.QSA_PAGES] = block_tables.data_ptr() + (r // group) * page_row

    write_group_draft_meta(counter, logical, block_tables, pos, qsa, group)
    torch.cuda.synchronize()
    assert int(counter) == 6
    assert pos.tolist() == list(range(300, 300 + rows))
    assert torch.equal(qsa.cpu(), want)

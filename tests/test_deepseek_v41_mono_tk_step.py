# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 TileRT-path step behind ATOM's step override (``mono/tk_step``), on the CPU: no GPU, aiter or FlyDSL.

- ``kv``: one request's KV in ATOM's V4.1 layout (BF16 main pages through a shuffled block table, FP8 index tiles,
  133-slot window rings, 7-slot compressor rings, the Engram cursor) -> TileRT's decode planes, each part against
  what TileRT's own quantizers make of the request's rows in position order;
- ``runner``: admission (sampling mode, TileRT's counter, admit, import) once a request, the verify outputs and
  verdict gathered from the logs a fake TkRank appends to, the next drafts;
- ``dispatch``: which steps are served (decided once, at the step's layout: ``TkRound``), that a request TkRank
  served cannot fall back, and a served step's commit of ATOM's state cursor (as ATOM's own commit would).
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.models.deepseek_v41.mono.hi.pd import convert as C
from atom.models.deepseek_v41.mono.tk_step import dispatch as D
from atom.models.deepseek_v41.mono.tk_step import kv as KV
from atom.models.deepseek_v41.mono.tk_step import runner as RN

E4M3 = torch.float8_e4m3fn


def _fp8_index_pages(keys: torch.Tensor, pages: int, per: int, blocks) -> torch.Tensor:
    """ATOM's FP8 index plane holding ``keys`` (float [count, 128]) in ``blocks``' pages: E4M3 with one fp32 power-of-two
    scale a row, 8-row tiles (byte ``(d // 16) * 128 + row * 16 + d % 16``, then the 8 scales).
    """
    count = keys.shape[0]
    rows = torch.zeros(len(blocks) * per, 128)
    rows[:count] = keys
    scale = C._pow2_ceil(rows.abs().amax(-1).clamp_min(1e-4) / 448)
    q = (rows / scale[:, None]).to(E4M3).view(torch.uint8)
    tiles = torch.cat([q.view(-1, 8, 8, 16).permute(0, 2, 1, 3).reshape(-1, 1024), scale.view(-1, 8).view(torch.uint8)], 1)  # fmt: skip
    plane = torch.zeros(pages, per * 132, dtype=torch.uint8)
    plane[torch.as_tensor(blocks)] = tiles.view(len(blocks), per * 132)
    return plane, (q.view(E4M3).float() * scale[:, None])[:count]


def _request(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    blocks = [4, 0, 3]  # the request's 3 blocks (128 tokens each), out of 5 pages
    main, index, logical = {}, {}, {}
    for s, ratio in KV.KV_SOURCES.items():
        per = 128 // ratio
        pages = torch.randn(5, per, 512, generator=g).to(torch.bfloat16)
        main[s] = pages
        logical[s] = pages[blocks].reshape(-1, 512)[: n // ratio]
        index[s], logical[f"ix{s}"] = _fp8_index_pages(torch.randn(n // ratio, 128, generator=g), 5, per, blocks)  # fmt: skip
    window = torch.zeros(43, 133, 512, dtype=torch.bfloat16)
    pos = torch.arange(n - 133, n)
    window[:, pos % 133] = (pos.float()[None, :, None] / 64 + torch.arange(43).float()[:, None, None] / 8).to(torch.bfloat16)  # fmt: skip
    comp = {s: (torch.randn(7, 512, generator=g), torch.randn(7, 512, generator=g)) for s in (2, 8, 14)}  # fmt: skip
    cursor = torch.tensor([n, 11, 12, 13])
    return (
        KV.RequestKV(main, index, window, comp, torch.tensor(blocks), cursor),
        logical,
    )


@pytest.mark.parametrize("n", [301, 300])
def test_planes_from_atom_layout(n):
    kv, logical = _request(n)
    planes = KV.request_planes(kv, n)
    for s, ratio in KV.KV_SOURCES.items():
        want = C.to_main(logical[s].float())
        assert all(torch.equal(a, b) for a, b in zip(planes["main_kv"][s], want))
        assert planes["main_kv"][s][0].shape == (n // ratio, 256)
        vals, scs = planes["index_k"][s]
        assert (vals.shape, scs.shape) == ((n // ratio, 64), (n // ratio, 4))
        assert all(
            torch.equal(a, b) for a, b in zip((vals, scs), KV.mxfp4(logical[f"ix{s}"]))
        )
    lo = n - 128
    for i, (wkv, wsc) in enumerate(planes["win"]):
        pos = torch.arange(lo, n)
        q, s_ = C.to_window(kv.window[i, pos % 133].float())
        assert torch.equal(wkv.view(torch.uint8)[pos % 256], q) and torch.equal(
            wsc[pos % 256], s_
        )
        rest = torch.ones(256, dtype=torch.bool)
        rest[pos % 256] = False
        assert not wkv.view(torch.uint8)[rest].any() and not wsc[rest].any()
    for s in (2, 8, 14):
        st = planes["state"][s]
        if n % 2:
            k, sc = kv.compress[s]
            assert torch.equal(
                st[(n - 1) % 16], torch.stack([k[(n - 1) % 7], sc[(n - 1) % 7]])
            )
            st = torch.cat([st[: (n - 1) % 16], st[(n - 1) % 16 + 1 :]])
        assert not st[:, 0].any() and torch.isinf(st[:, 1]).all()
    h = planes["hist"]
    assert [int(h[(n - 3 + j) % 16]) for j in range(3)] == [11, 12, 13] and int(
        h.abs().sum()
    ) == 36
    assert set(planes) == {"main_kv", "index_k", "win", "state", "hist"}


def test_mxfp4_and_hist_paths():
    x = torch.randn(5, 128)
    v, s = KV.mxfp4(x)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 2) * torch.tensor(
        [1.0] * 8 + [-1.0] * 8
    )
    deq = torch.stack([lut[(v & 15).long()], lut[(v >> 4).long()]], -1).reshape(
        5, 4, 32
    )
    step = torch.exp2(s.float() - 127)[..., None]
    assert (
        (deq * step - x.view(5, 4, 32)).abs() <= step
    ).all()  # within one E2M1 step of the group scale
    tm = torch.arange(1000) * 3
    assert torch.equal(KV.hist(20, tail_ids=list(range(4, 20)), token_map=tm), C.hist_plane(list(range(4, 20)), 20, tm))  # fmt: skip
    with pytest.raises(ValueError):
        KV.hist(20, cursor=torch.tensor([19, 1, 2, 3]))


class FakeTk:
    """TkRank's logs as admit_kernel and the step write them; scripted accepts."""

    def __init__(self, accepts=(3, 6, 1)):
        self.accepts, self.calls, self.k = accepts, [], 0
        self.acc, self.num = torch.zeros(64, dtype=torch.int32), torch.zeros(
            64, dtype=torch.int32
        )
        self.vtok = torch.arange(8, dtype=torch.int32) + 100

    def set_sampling(self, params):
        self.calls.append(("set_sampling", list(params)))

    def admit(self, first, pos, params):
        self.calls.append(("admit", first, pos, params))
        self.acc.zero_(), self.num.zero_()
        self.acc[:2] = torch.tensor([1, first])
        self.num[:2] = 1

    def import_planes(self, planes, n):
        self.calls.append(("import", n))

    def step(self):
        a = self.accepts[self.k % len(self.accepts)]
        c = int(self.acc[0])
        self.acc[c + 1 : c + 1 + a] = torch.arange(a) + 1000 * (self.k + 1)
        self.acc[0] += a
        self.num[int(self.num[0]) + 1] = a
        self.num[0] += 1
        self.k += 1
        self.vtok += 1

    def logs(self):
        return self.acc, self.num

    def drafts(self):
        return self.vtok


def test_runner_admits_once_and_reads_the_logs():
    tk = FakeTk()
    r = RN.TkStepRunner(tk)
    r.output_stream = None  # (a CPU run)
    ids = torch.tensor([7, 1, 1, 1, 1, 1])
    for k, a in enumerate((3, 6, 1)):
        r.run_step(41, ids, 500, 1.0, lambda: {"planes": True})
        out, verdict = r.verify_outputs(1)
        assert (
            out.dtype == torch.int32 and out.shape == (1, 6) and verdict.shape == (3, 1)
        )
        assert (
            out[0, :a].tolist() == [1000 * (k + 1) + j for j in range(a)]
            and (out[0, a:] == -1).all()
        )
        assert verdict[:, 0].tolist() == [a - 1, 6 - a, a - 1]
        assert r.draft_ids(1).tolist() == [
            [102 + k + j for j in range(5)]
        ]  # vtok[1:6] after the step
    assert r.pos == 500  # ATOM's anchor position, no device read
    (_, p1), (_, f, n, none), _ = tk.calls[:3]
    assert (
        tk.calls[0][0] == "set_sampling"
        and p1[3] == 1
        and (f, n, none) == (7, 500, None)
    )
    r.run_step(
        42, ids, 600, 1.0, dict
    )  # the next request, same sampling mode: the parameters ride along
    assert (
        tk.calls[-2][:3] == ("admit", 7, 600)
        and tk.calls[-2][3][3] == 2
        and tk.calls[-1] == ("import", 600)
    )
    assert (
        tk.calls[-2][3][1] == p1[1] and tk.calls[-2][3][0] != p1[0]
    )  # a seed of its own


def test_sampling_refusals():
    assert RN.sampling_refusal(0.0, 1.0, -1).startswith("greedy")
    assert RN.sampling_refusal(1.0, 0.9, -1) and RN.sampling_refusal(1.0, 1.0, 20)
    assert RN.sampling_refusal(0.7, 1.0, -1) is None


def _served_model(monkeypatch, defers=False):
    """A ``TkStepModel`` over a fake TkRank, its ``TkRound`` on a builder whose ``step_needs_eager`` reads the
    batch's ``eager``; ``defers``: the builder's Engram on the device (held steps skip the verdict wait).
    """
    fc = SimpleNamespace(
        spec_decode_metadata=object(), attn_metadata=SimpleNamespace(cache=None)
    )
    monkeypatch.setattr(D, "get_forward_context", lambda: fc)
    monkeypatch.setattr(D.envs, "ATOM_MONO_CHECK", False, raising=False)
    runner = RN.TkStepRunner(FakeTk())
    runner.output_stream = None
    drafter = SimpleNamespace(
        compute_draft_kv=lambda *a: "prod-kv", propose=lambda *a: "prod"
    )
    m = D.TkStepModel(torch.nn.Identity(), drafter, runner)
    monkeypatch.setattr(D.TkStepModel, "_anchor_position", staticmethod(lambda b: 500))
    monkeypatch.setattr(RN.KV, "request_planes", lambda kv, n, **k: {})
    monkeypatch.setattr(
        "atom.models.deepseek_v41.mono.tk_step.kv.from_cache", lambda *a: "kv"
    )
    builder = SimpleNamespace(step_needs_eager=lambda b: b.eager)
    if defers:
        builder.engram = SimpleNamespace(host=SimpleNamespace(overlap=object()))
    return m, D.TkRound(m, builder), runner, drafter


def _batch(rid, t=1.0, logprobs=False, dummy=False, eager=False):
    return SimpleNamespace(req_ids=[rid], total_seqs_num=1, temperatures=[t], top_ps=[1.0], top_ks=[-1],
                           any_logprobs=logprobs, structured_requests={}, required_channel=None, is_dummy_run=dummy,
                           context_lens=[506], state_slots_committed=[0], block_tables=[[0]], eager=eager)  # fmt: skip


def _layout(rnd, batch, bs=1, rows=6, decode=True):
    """The builder's routing decision on ``batch`` (``_prepare_step``)."""
    return rnd.service(batch, [], bs, bs * (rows or 6), max_q_len=rows, decode=decode, tentative=decode,
                       is_prefill=not decode, resumes=None, image_rows=False)  # fmt: skip


IDS = torch.tensor([7, 1, 1, 1, 1, 1])


def test_dispatch_serves_and_holds(monkeypatch):
    m, rnd, runner, drafter = _served_model(monkeypatch)
    tk, LEAN, NONE = runner.dev, D.RoundService.LEAN, D.RoundService.NONE

    def step(batch, **kw):
        return _layout(rnd, batch, **kw), m.step_override(IDS, 1, batch)

    assert (
        step(_batch(9, dummy=True)) == (NONE, None) and not tk.calls
    )  # a warmup's dummy decode
    assert step(_batch(4, eager=True)) == (
        NONE,
        None,
    )  # an eager step gets no step override
    assert step(_batch(4), decode=False, rows=None) == (NONE, None)  # a prefill
    assert step(_batch(5)) == (LEAN, runner)
    runner.verify_outputs(1)
    assert drafter.propose().shape == (1, 5) and drafter.compute_draft_kv() is None
    for kw in (
        {"bs": 2},
        {"rows": 12},
        {"decode": False},
    ):  # the held request in a step it cannot serve
        with pytest.raises(RuntimeError, match="cannot fall back"):
            _layout(rnd, _batch(5), **kw)
    with pytest.raises(RuntimeError, match="cannot fall back"):
        _layout(rnd, _batch(5, eager=True))
    assert step(_batch(6, t=0.0)) == (
        NONE,
        None,
    )  # another request, greedy: the production path
    assert drafter.propose() == "prod" and drafter.compute_draft_kv() == "prod-kv"
    assert step(_batch(7)) == (LEAN, runner) and runner.holds(
        7
    )  # served: admitted afresh
    assert [c[0] for c in tk.calls] == [
        "set_sampling",
        "admit",
        "import",
        "admit",
        "import",
    ]


def test_dispatch_serves_exactly_the_steps_laid_out_for_it(monkeypatch):
    m, rnd, _, _ = _served_model(monkeypatch)
    assert (
        m.step_override(IDS, 1, _batch(5)) is None
    )  # no decision (a builder without the round): production
    assert _layout(rnd, _batch(5)) == D.RoundService.LEAN
    with pytest.raises(RuntimeError, match="laid out for it runs another batch"):
        m.step_override(IDS, 1, _batch(5))  # not the batch laid out
    assert _layout(rnd, _batch(5)) == D.RoundService.LEAN
    with pytest.raises(RuntimeError, match="ran ATOM's forward"):
        m(IDS, None)  # laid out LEAN, then run eagerly
    b = _batch(5)
    assert _layout(rnd, b) == D.RoundService.LEAN
    meta = SimpleNamespace(served_by_round=D.RoundService.LEAN, cache=None, step=None)
    with pytest.raises(RuntimeError, match="was not served"):
        rnd.committed(meta, None)  # laid out, never offered
    assert not rnd.committed(SimpleNamespace(served_by_round=D.RoundService.NONE), None)


def _fake_step(positions, slots, length=6):
    spans = [SimpleNamespace(request_id=i, position=p, length=length, end=p + length, slot=s)
             for i, (p, s) in enumerate(zip(positions, slots))]  # fmt: skip
    n = len(spans)
    rows = [torch.arange(p, p + length) for p in positions] + [torch.zeros(5)]
    return SimpleNamespace(
        requests=spans,
        positions=torch.cat(rows).to(torch.int32),
        scheduled_bs=n,
        cu_seqlens_q=torch.arange(n + 1, dtype=torch.int32) * length,
        slots=torch.tensor(list(slots) + [0, 0], dtype=torch.int32),
        request_positions=np.asarray(positions, dtype=np.int64),
    )


def _tentative_state_cls(monkeypatch):
    """ATOM's ``TentativeState`` (its CPU commit), loaded alone; a stand-in Triton where none is installed (its
    kernel is the GPU commit, not run here)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        tl = types.ModuleType("triton.language")
        tl.constexpr = object
        tr = types.ModuleType("triton")
        tr.jit, tr.language = (lambda f: f), tl
        monkeypatch.setitem(sys.modules, "triton", tr)
        monkeypatch.setitem(sys.modules, "triton.language", tl)
    import atom

    path = (
        Path(atom.__file__).parent / "model_ops/attentions/deepseek_v41/speculative.py"
    )
    spec = importlib.util.spec_from_file_location("_v41_speculative", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TentativeState


def test_commit_served_matches_atoms_commit(monkeypatch):
    """A served step's cursor (``commit_served``) where ATOM's own commit (``TentativeState.commit``) puts it, for
    every anchor; its tentative state dropped."""
    TentativeState = _tentative_state_cls(monkeypatch)
    positions, slots = (4782, 200, 37), (3, 1, 6)
    step = _fake_step(positions, slots)
    for anchors in ([0, 6, 12], [1, 11, 17], [5, 8, 14]):
        anchors = torch.tensor(anchors, dtype=torch.int64)
        width = 4  # [end, 3 history ids], as ATOM's V4.1 cursor
        staged = np.zeros((len(slots), 6, width), dtype=np.int64)
        staging = SimpleNamespace(np=staged, copy_to_gpu=lambda n, s=staged: torch.from_numpy(s[:n]))  # fmt: skip
        theirs = SimpleNamespace(cursor=torch.full((8, width), -7, dtype=torch.int64), tentative_staging=staging)  # fmt: skip
        state = TentativeState(theirs, step, np.full((len(slots), width - 1), -1, dtype=np.int64))  # fmt: skip
        for span in step.requests:
            state.stage_history(span, list(range(span.length)))
        state.commit(anchors)
        ours = SimpleNamespace(cursor=torch.full((8, width), -7, dtype=torch.int64), pending=state)  # fmt: skip
        D.commit_served(ours, step, anchors)
        assert ours.pending is None
        assert torch.equal(ours.cursor[:, 0], theirs.cursor[:, 0])
        assert (ours.cursor[list(slots), 0] - torch.tensor(positions)).tolist() == (
            anchors - step.cu_seqlens_q[:3] + 1
        ).tolist()  # the next step starts past its accepted rows


def test_dispatch_commits_a_served_step(monkeypatch):
    """The served step's verdict, through ``TkRound.committed`` (``commit_speculative_state``), is where the next
    step's state probe (``prepare_state``: cursor[slot, 0] == the step's start) looks.
    """
    m, rnd, runner, _ = _served_model(monkeypatch)
    pos = 4782
    for k, a in enumerate(runner.dev.accepts):
        b = _batch(5)
        assert (
            _layout(rnd, b) == D.RoundService.LEAN
            and m.step_override(IDS, 1, b) is runner
        )
        _, verdict = runner.verify_outputs(1)
        cache = SimpleNamespace(
            cursor=torch.zeros(4, 4, dtype=torch.int64), pending=object()
        )
        meta = SimpleNamespace(
            served_by_round=D.RoundService.LEAN,
            cache=cache,
            step=_fake_step([pos], [2]),
        )
        assert rnd.committed(meta, verdict[2]) and cache.pending is None
        pos += a
        assert (
            int(cache.cursor[2, 0]) == pos
        )  # the next start: the step's accepted tokens past this one's


def test_tk_step_refuses_prefix_caching_and_eager(monkeypatch):
    from atom.models.deepseek_v41.mono import tk_step
    from atom.mono.runtime import deployment

    monkeypatch.setattr(deployment, "common_refusals", lambda *a, **k: [])

    def config(**kw):
        base = {"speculative_config": SimpleNamespace(method="dspark", num_speculative_tokens=5),
                "dspark": SimpleNamespace(confidence_schedule=False, ragged=False), "max_num_seqs": 1,
                "max_model_len": 65536, "enable_prefix_caching": False, "enforce_eager": False,
                "kv_cache_dtype": "bf16", "index_cache_dtype": "fp8", "kv_cache_block_size": 128}  # fmt: skip
        return SimpleNamespace(**{**base, **kw})

    assert tk_step.refusal(config()) is None
    assert "prefix caching" in tk_step.refusal(config(enable_prefix_caching=True))
    assert "enforce_eager" in tk_step.refusal(config(enforce_eager=True))
    # the caches its KV conversion reads (the hi path's FP4 index is not one)
    assert "--index-cache-dtype fp4" in tk_step.refusal(config(index_cache_dtype="fp4"))
    assert "--kv_cache_dtype fp8" in tk_step.refusal(config(kv_cache_dtype="fp8"))
    assert "--block-size 16" in tk_step.refusal(config(kv_cache_block_size=16))
    # a field missing: a deployment it is not written for, not an error
    assert "max_num_seqs None" in tk_step.refusal(SimpleNamespace(**{k: v for k, v in vars(config()).items() if k != "max_num_seqs"}))  # fmt: skip


def test_install_needs_the_v41_builder(monkeypatch):
    monkeypatch.setattr(D, "refusal", lambda c: None)
    runner = RN.TkStepRunner(FakeTk())
    monkeypatch.setattr(
        RN.TkStepRunner, "create", classmethod(lambda cls, *a, **k: runner)
    )
    model, cfg = torch.nn.Identity(), SimpleNamespace(model="m")
    # nothing to keep ATOM's state bookkeeping: refused (mono.dispatch then takes the hi / mono paths)
    assert "metadata builder" in D.install_refusal(cfg, None)
    builder = SimpleNamespace(use_decode_round=lambda r: setattr(builder, "round", r))
    assert D.install_refusal(cfg, builder) is None
    wrapped = D.install_tk_step(model, cfg, None, builder)
    assert isinstance(wrapped, D.TkStepModel) and builder.round.model is wrapped


def test_held_steps_skip_the_verdict_wait(monkeypatch):
    """The admission step waits for its verdict (the builder's probe checks it); every later step of the held request
    is built without the wait (``consumes_device_verdict``), laid out ``DEVICE`` (no probe), and commits by moving the
    cursor on by the verdict's accepted count, whatever the positions it was built with: the cursor stays at the
    request's next start. Any other forward in between, or another request, waits again.
    """
    m, rnd, runner, _ = _served_model(monkeypatch, defers=True)
    LEAN, DEVICE = D.RoundService.LEAN, D.RoundService.DEVICE
    cache = SimpleNamespace(cursor=torch.zeros(4, 4, dtype=torch.int64), pending=None)

    def step(kind, positions_start):
        b = _batch(5)
        on = rnd.consumes_device_verdict(b, 6)
        assert (on, _layout(rnd, b)) == (kind is DEVICE, kind)
        assert m.step_override(IDS, 1, b) is runner
        _, verdict = runner.verify_outputs(1)
        cache.pending = object()
        meta = SimpleNamespace(served_by_round=kind, cache=cache, step=_fake_step([positions_start], [2]))  # fmt: skip
        assert rnd.committed(meta, verdict[2]) and cache.pending is None

    pos = 4782
    step(LEAN, pos)  # admission
    pos += runner.dev.accepts[0]
    assert int(cache.cursor[2, 0]) == pos
    for k in range(1, 6):
        step(
            DEVICE, pos + 1000
        )  # built without the verdict: its positions are not the request's
        pos += runner.dev.accepts[k % len(runner.dev.accepts)]
        assert int(cache.cursor[2, 0]) == pos
    # a forward between them (another request's prefill), then the held request again: it waits
    assert not rnd.consumes_device_verdict(SimpleNamespace(**{**vars(_batch(7)), "total_seqs_num_prefill": 1}), 6)  # fmt: skip
    rnd.service(_batch(7), [], 1, 100, max_q_len=None, decode=False, tentative=False, is_prefill=True, resumes=None, image_rows=False)  # fmt: skip
    assert not rnd.consumes_device_verdict(_batch(5), 6)
    # never for a dummy run, another width, or without the on-device Engram path
    rnd._ready = 5
    assert not rnd.consumes_device_verdict(SimpleNamespace(**{**vars(_batch(5)), "is_dummy_run": True}), 6)  # fmt: skip
    rnd._ready = 5
    assert not rnd.consumes_device_verdict(_batch(5), 4)
    _, rnd2, _, _ = _served_model(monkeypatch)
    rnd2._ready = 5
    assert not rnd2.defers and not rnd2.consumes_device_verdict(_batch(5), 6)


def test_tk_step_obeys_the_mono_master_switch(monkeypatch):
    """ATOM_MONO_ENABLE=0 turns every mono path off, the TileRT-path step too; on, it is gfx950's, with no switch of
    its own (``mono.dispatch.install_mono_decode`` then asks its install refusal)."""
    from atom.models.deepseek_v41.mono import tk_step

    monkeypatch.setattr(tk_step, "_arch", lambda: "gfx950")
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    assert not tk_step.enabled()
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    assert tk_step.enabled()
    for arch in ("gfx942", "gfx1250", None):
        monkeypatch.setattr(tk_step, "_arch", lambda a=arch: a)
        assert not tk_step.enabled()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 TileRT-path step behind ATOM's step override (``mono/tk_step``), on the CPU: no GPU, aiter or FlyDSL.

- ``kv``: one request's KV in ATOM's V4.1 layout (BF16 main pages through a shuffled block table, FP8 index tiles,
  133-slot window rings, 7-slot compressor rings, the Engram cursor) -> TileRT's decode planes, each part against
  what TileRT's own quantizers make of the request's rows in position order;
- ``runner``: admission (sampling mode, TileRT's counter, admit, import) once a request, the verify outputs and
  verdict gathered from the logs a fake TkRank appends to, the next drafts;
- ``dispatch``: which steps are served, and that a request TkRank served cannot fall back.
"""

from types import SimpleNamespace

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
    assert r.pos == 500 + 3 + 6
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


def test_dispatch_serves_and_holds(monkeypatch):
    fc = SimpleNamespace(
        spec_decode_metadata=object(), attn_metadata=SimpleNamespace(cache=None)
    )
    monkeypatch.setattr(D, "get_forward_context", lambda: fc)
    monkeypatch.setattr(D.envs, "ATOM_MONO_CHECK", False, raising=False)
    tk = FakeTk()
    runner = RN.TkStepRunner(tk)
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

    def batch(rid, t=1.0, logprobs=False):
        return SimpleNamespace(req_ids=[rid], total_seqs_num=1, temperatures=[t], top_ps=[1.0], top_ks=[-1],
                               any_logprobs=logprobs, structured_requests={}, required_channel=None,
                               context_lens=[506], state_slots_committed=[0], block_tables=[[0]])  # fmt: skip

    ids = torch.tensor([7, 1, 1, 1, 1, 1])
    assert m.step_override(ids, 1, batch(5)) is runner
    runner.verify_outputs(1)
    assert drafter.propose().shape == (1, 5) and drafter.compute_draft_kv() is None
    with pytest.raises(
        RuntimeError, match="cannot fall back"
    ):  # the held request in a step it cannot serve
        m.step_override(torch.zeros(12), 2, batch(5))
    assert (
        m.step_override(ids, 1, batch(6, t=0.0)) is None
    )  # another request, greedy: the production path
    assert drafter.propose() == "prod" and drafter.compute_draft_kv() == "prod-kv"
    assert m.step_override(ids, 1, batch(7)) is runner and runner.holds(
        7
    )  # served: admitted afresh
    assert [c[0] for c in tk.calls] == [
        "set_sampling",
        "admit",
        "import",
        "admit",
        "import",
    ]

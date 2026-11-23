# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TileRT-structured decode step (``mono/gfx950/tk``) and its PD decode server (``mono/gfx950/pd``) on the CPU: no
GPU, no aiter, no FlyDSL.

Host and build (``tk``):
- ``flybuild``: one recipe per kernel the plan launches plus ``admit_kernel``, each named by its TileRT symbol; the
  cache key follows the sources a recipe imports and its build switches; ``code_table`` builds what the cache lacks,
  once, and maps every TileRT symbol to its build's function and layout;
- ``host.admit_reference`` / ``admit_args``: ``admit_kernel``'s effect, for tkserve's request shapes and a reset; the
  blob of a recorded admission;
- the token tables: ``tkworker.load_toktab``'s cache and GPU build; the device build's (``toktab_device``) kernargs and
  buffers.

Step override (``tk.gate``, ``kv``, ``runner``, ``dispatch``):
- ``kv``: one request's KV in ATOM's V4.1 layout (BF16 main pages through a shuffled block table, FP8 index tiles,
  133-slot window rings, 7-slot compressor rings, the Engram cursor) -> TileRT's decode planes, each part against what
  TileRT's own quantizers make of the request's rows in position order;
- ``runner``: admission (sampling mode, TileRT's counter, admit, import) once a request, the verify outputs and
  verdict gathered from the logs a fake TkRank appends to, the next drafts;
- ``dispatch``: which steps are served (decided once, at the step's layout: ``TkRound``), that a request TkRank served
  cannot fall back, and a served step's commit of ATOM's state cursor (as ATOM's own commit would); ``gate``: the
  deployment it serves.

PD server (``pd``: ``receive``, ``session``, ``server``, ``fanout``), end to end over real sockets on
``session.FakeEngine`` and ``fanout.FakeRank`` workers: a prefill rank on the control plane (hello, admission, the
landed header standing in for the RDMA write, done) and the router on HTTP (``/pd/decode`` streamed as NDJSON, read as
TileRT's ``DecodeReader`` reads it: the ``t`` batches, then a ``done`` whose ``n`` must equal the tokens streamed):
- a streamed request: the first token alone, the accepted tokens in order, EOS ends it unstreamed; the sampling mode
  set then the admission (TileRT's slow path), the KV loaded, the slot freed; the next request with the same mode
  admits with the parameters (the fast path), ``ignore_eos`` streams EOS, ``max_tokens`` ends it (``length``);
- the KV landing before or after the decode call; a non-streamed request;
- refusals (busy, sampling, penalties, grammar, logprobs, malformed body, KV timeout), each releasing the request;
- ``/pd/cancel`` and a client gone mid-stream; the control plane's busy / tombstone / oversize rejects;
- the loop's chunking against the budget; greedy refused; the rank group's failures; the modules import without the
  GPU stack.
"""

import http.client
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.models.deepseek_v41.mono.gfx950.pd import convert as C
from atom.models.deepseek_v41.mono.gfx950.pd import fanout as F
from atom.models.deepseek_v41.mono.gfx950.pd import receive as R
from atom.models.deepseek_v41.mono.gfx950.pd import server as SV
from atom.models.deepseek_v41.mono.gfx950.pd import session as S
from atom.models.deepseek_v41.mono.gfx950.pd import tkworker as TW
from atom.models.deepseek_v41.mono.gfx950.pd import wire as W
from atom.models.deepseek_v41.mono.gfx950.tk import codeobj, flybuild, gate
from atom.models.deepseek_v41.mono.gfx950.tk import decode_launches as DL
from atom.models.deepseek_v41.mono.gfx950.tk import dispatch as D
from atom.models.deepseek_v41.mono.gfx950.tk import host as H
from atom.models.deepseek_v41.mono.gfx950.tk import kv as KV
from atom.models.deepseek_v41.mono.gfx950.tk import runner as RN
from atom.models.deepseek_v41.mono.gfx950.tk import toktab_device as TD

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "atom/models/deepseek_v41/mono/gfx950/tk/kernels"
E4M3 = torch.float8_e4m3fn

# =====================================================================================================================
# host and build: flybuild


def test_one_recipe_per_plan_kernel_and_admit():
    plan = DL.load()
    syms = [r.symbol for r in flybuild.RECIPES]
    assert len(set(syms)) == len(syms) == len({r.key for r in flybuild.RECIPES})
    assert {r.symbol for r in flybuild.STEP_RECIPES} == set(plan["kernels"])
    assert flybuild.BY_KEY["admit"].symbol == H.ADMIT_SYMBOL
    for r in flybuild.RECIPES:
        assert f"def {r.builder}(" in (KERNELS / f"{r.module}.py").read_text(), r
        assert r.preload in (0, 7)
        assert all(k.startswith("TK_") for k, _ in r.env)


def test_cache_key_follows_sources_and_switches(monkeypatch, tmp_path):
    names = {f.name for f in flybuild.source_files("qkv")}
    assert {"qkv.py", "common.py", "ops.py"} <= names
    r = flybuild.BY_KEY["qkv_kg6_idx0"]
    k0 = flybuild.recipe_key(r, "v")
    assert k0 == flybuild.recipe_key(r, "v") and k0 != flybuild.recipe_key(r, "w")
    r2 = flybuild.Recipe(
        r.key, r.symbol, r.module, r.builder, (("TK_QKV_IDX", "16"),), r.preload
    )
    assert flybuild.recipe_key(r2, "v") != k0
    extra = tmp_path / "x.py"
    extra.write_text("a = 1\n")
    k1 = flybuild.recipe_key(r, "v", [extra])
    extra.write_text("a = 2\n")
    assert flybuild.recipe_key(r, "v", [extra]) != k1
    env = flybuild.build_env(r)
    assert (env["TK_QKV_IDX"], env["ARCH"], env["FLYDSL_RUNTIME_ENABLE_CACHE"]) == ("0", "gfx950", "0")  # fmt: skip
    monkeypatch.setenv("TK_VARIANT", "afm6")
    # the caller's switches never leak into a build
    assert "TK_VARIANT" not in flybuild.build_env(r)


def test_code_table_builds_once_and_maps_tilert_symbols(monkeypatch, tmp_path):
    built = []

    def fake_build(r, stem, timeout):
        built.append(r.key)
        lay = ",".join(f"{8 * i}:8" for i in range(3))
        stem.with_suffix(".json").write_text(json.dumps({"function": f"fn_{r.key}", "layout": lay}))  # fmt: skip
        stem.with_suffix(".hsaco").write_bytes(r.key.encode())

    monkeypatch.setattr(flybuild, "_build_into", fake_build)
    monkeypatch.setattr(codeobj, "kernels", lambda elf: {f"fn_{elf.decode()}": [(0, 8), (8, 8), (16, 8)]})  # fmt: skip
    rs = tuple(flybuild.RECIPES[:3]) + (flybuild.BY_KEY["admit"],)
    t = flybuild.code_table(tmp_path, rs, jobs=2)
    assert sorted(built) == sorted(r.key for r in rs)
    assert set(t.entries) == {r.symbol for r in rs}
    e = t.entries[H.ADMIT_SYMBOL]
    assert e.function == "fn_admit" and e.layout == ((0, 8), (8, 8), (16, 8))
    flybuild.code_table(tmp_path, rs)
    assert len(built) == len(rs)  # cached
    tsv = tmp_path / "t.tsv"
    flybuild.write_tsv(t, tsv)
    res = flybuild.compare_tsv(t, tsv)
    assert res["lines"] == res["symbol"] == res["function"] == res["layout"] == len(rs)
    # a layout that is not the module's LAYOUT is refused
    monkeypatch.setattr(codeobj, "kernels", lambda elf: {f"fn_{elf.decode()}": [(0, 8)]})  # fmt: skip
    with pytest.raises(RuntimeError, match="layout"):
        flybuild.code_table(tmp_path, rs)


# =====================================================================================================================
# host and build: admit_kernel's model and its arguments


def _admit_mem(n=40, nplain=50):
    p = 0x5A5A5A5A
    return {"plain_acc": [p] * nplain, "ar_acc": [p] * n, "ar_num": [p] * n, "ctl": [p] * 16,
            "blk_tok": [p] * 8, "blk_pos": [p] * 8, "vtok": [p] * 8, "vpos": [p] * 8, "samp_params": [7] * 4,
            "X": bytearray(b"\xa5" * 8 * H.X_ROW)}  # fmt: skip


def _admit_blob(first, pos, mtp, params, cap, n=40, nplain=50):
    addr = {k: 0x1000 * (i + 1) for i, k in enumerate(
        ("temp:plain_acc", "temp:ar_acc", "temp:ar_num", "temp:ctl", "temp:blk_tok", "temp:blk_pos", "temp:vtok",
         "temp:vpos", "param:embed", "temp:X", "temp:samp_params"))}  # fmt: skip
    b = bytearray(H.admit_args(addr, first, pos, mtp, params))
    for off, cnt in (
        (0x08, nplain),
        (0x18, n),
        (0x28, n),
    ):  # small buffers: the counts cut down
        struct.pack_into("<i", b, off, cnt)
    struct.pack_into("<i", b, 0x80, cap)
    return bytes(b)


def _row(tok):
    return bytes([tok % 251]) * H.ROW_BYTES


def test_admit_reference_request_and_reset():
    first, pos, cap = 2581, 30, 33
    params = [591402, 0x3F8000003F800000, 0, 2]
    mem = _admit_mem()
    H.admit_reference(_admit_blob(first, pos, True, params, cap), mem, _row)
    assert mem["plain_acc"] == [0] * 50
    assert mem["ar_acc"][:3] == [1, first, 0] and set(mem["ar_acc"][2:]) == {0}
    assert mem["ar_num"][:3] == [1, 1, 0] and set(mem["ar_num"][2:]) == {0}
    assert mem["ctl"] == [pos - 1, first, 0, cap, 0, 1] + [0] * 10
    assert mem["blk_tok"] == [first] + [H.PAD_TOKEN] * 7
    assert mem["blk_pos"] == [30, 31, 32, -1, -1, -1, -1, -1]  # pos + t < cap, t < 5
    assert mem["vtok"][:6] == [first] + [H.PAD_TOKEN] * 5 and mem["vtok"][6:] == [0x5A5A5A5A] * 2  # fmt: skip
    assert mem["vpos"][:6] == [30, 31, 32, -1, -1, -1]
    assert mem["samp_params"] == params
    X = mem["X"]
    assert X[: H.ROW_BYTES] == _row(first) and X[3 * H.ROW_BYTES : H.X_ROW] == _row(first)  # fmt: skip
    assert X[H.X_ROW : H.X_ROW + H.ROW_BYTES] == _row(H.PAD_TOKEN)
    # a reset (_reset_streams): the logs and ctl only, cap kept; samp_params untouched without params
    mem = _admit_mem()
    H.admit_reference(_admit_blob(0, 0, False, None, cap), mem, _row)
    assert mem["ar_acc"] == [0] * 40 and mem["ar_num"] == [0] * 40
    assert mem["ctl"] == [0, 0, 0, cap] + [0] * 12 and mem["samp_params"] == [7] * 4
    assert mem["blk_tok"] == [0x5A5A5A5A] * 8 and mem["X"] == bytearray(b"\xa5" * 8 * H.X_ROW)  # fmt: skip
    # a first token past the vocabulary: a zero row
    mem = _admit_mem()
    H.admit_reference(_admit_blob(129285, 3, True, None, cap), mem, _row)
    assert mem["X"][: H.X_ROW] == bytes(H.X_ROW)


def test_admit_kernel_abi():
    src = (KERNELS / "admit.py").read_text()
    assert "KARG_WORDS = 22" in src and 22 * 8 == H.ADMIT_BYTES
    assert "GRID, THREADS = 64, 256" in src
    assert (H.ADMIT_GRID, H.ADMIT_BLOCK) == ((64, 1, 1), (256, 1, 1))


# tkserve 11-16, rank 0, the 2nd dsv41_e2e_begin (m7/rec_admit.jsonl seq 52488) and its buffers (the same process's map)
REC_ADMIT = (
    "0000ae2fde7e0000400012000000000000003ca2df7e0000400010000000000000800c14557e000040001000000000000078"
    "7f18707f0000007a7f18707f0000007c7f18707f00000800000000000000007e7f18707f000000807f18707f00000600000000"
    "0000000000a04d057f000000006018707f0000ad1a0600150a0000000010000100000000867f18707f00002a06090000000000"
    "0000803f0000803f00000000000000000300000000000000"
)
REC_ADDR = {
    "temp:plain_acc": 0x7EDE2FAE0000,
    "temp:ar_acc": 0x7EDFA23C0000,
    "temp:ar_num": 0x7E55140C8000,
    "temp:ctl": 0x7F70187F7800,
    "temp:blk_tok": 0x7F70187F7A00,
    "temp:blk_pos": 0x7F70187F7C00,
    "temp:vtok": 0x7F70187F7E00,
    "temp:vpos": 0x7F70187F8000,
    "param:embed": 0x7F054DA00000,
    "temp:X": 0x7F7018600000,
    "temp:samp_params": 0x7F70187F8600,
}


def test_admit_args_rebuild_recorded_blob():
    # dsv41_e2e_begin(first 2581, pos 400045, pack_params(temperature 1, seed 591402)): the .so's sampling counter is
    # the 4th word (3: set_sampling made it 1, this is the 2nd begin)
    params = S.pack_params(S.SamplingParams(temperature=1.0, seed=591402))
    assert params[3] == 0
    blob = H.admit_args(REC_ADDR, 2581, 400045, True, params[:3] + [3])
    assert blob == bytes.fromhex(REC_ADMIT)
    reset = H.admit_args(
        REC_ADDR, 0, 0, False, None
    )  # _reset_streams: no parameters, flag 0
    assert reset[:0x78] == blob[:0x78]
    assert reset[0x78:] == bytes(8) + (1 << 20).to_bytes(4, "little") + bytes(44)


# =====================================================================================================================
# host and build: toktab


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _rn32(q: Fraction) -> float:
    """An exact rational rounded to float32, nearest even (normal range)."""
    f = struct.unpack("<f", struct.pack("<f", float(q)))[0]
    cands = [f] + [_f32((_bits(f) + d) & 0xFFFFFFFF) for d in (-1, 1)]
    return min(cands, key=lambda v: (abs(Fraction(v) - q), _bits(v) & 1))


def test_load_toktab_cache_and_builds(monkeypatch, tmp_path):
    """``tkworker.load_toktab``: TileRT's list format read as it is; the GPU build (default) into the toktab buffers,
    then saved; a cache of another build is rebuilt."""
    got = {}
    bufs = {f"toktab:{sc}.{n}": torch.zeros(2) for sc in ("layers.0", "mtp.0") for n in ("q", "kv", "hc")}  # fmt: skip
    params = {f"param:{sc}.{s}": torch.zeros(1) for sc in TD.SCOPES for s in TD.SLOTS}
    tk = SimpleNamespace(
        device="cpu",
        params={"param:embed": torch.zeros(2), **params},
        buffers=SimpleNamespace(
            tensors=bufs,
            toktab=lambda scope, q, kv, hc: got.__setitem__(scope, (q, kv, hc)),
        ),
    )
    tabs = [torch.full((2,), float(i)) for i in range(6)]
    torch.save(tabs, tmp_path / "rank0.pt")
    TW.load_toktab(tk, 0, str(tmp_path))
    assert got["layers.0"][0][0] == 0 and got["mtp.0"][2][0] == 5
    built = []

    class FakeBuilder:
        def __init__(self, table, device):
            built.append(("init", table))

        def build(self, p, embed, out=None, log=None):
            assert set(p) == set(TD.SLOTS)
            built.append("scope")
            for k, t in enumerate(out):
                t.fill_(10 + k)
            return out

    monkeypatch.setattr(TD, "Builder", FakeBuilder)
    d = tmp_path / "gpu"
    TW.load_toktab(tk, 1, str(d), table="T")
    assert built == [("init", "T"), "scope", "scope"]
    assert float(bufs["toktab:mtp.0.hc"][0]) == 12
    saved = torch.load(d / "rank1.pt")
    assert saved["build"] == "gpu" and float(saved["layers.0"][1][0]) == 11
    built.clear()
    TW.load_toktab(tk, 1, str(d), table="T")  # the cache: no build
    assert built == [] and float(got["layers.0"][1][0]) == 11
    torch.save(
        {"layers.0": tabs[0:3], "mtp.0": tabs[3:6], "build": "cpu"}, d / "rank1.pt"
    )
    TW.load_toktab(tk, 1, str(d), table="T")  # a cache another build wrote: rebuilt
    assert built == [("init", "T"), "scope", "scope"]


def test_blobs_acceptance_modes(monkeypatch):
    """head_seed_fold's simulated acceptance word (``H.SIM_ACC_OFF``): as recorded, 0.0 for real acceptance, or the
    mean acceptance length ATOM forces (``--spec-decode-acceptance-length``); no other launch is touched.
    """
    rec = bytearray(H.SIM_ACC_OFF + 8)
    struct.pack_into("<f", rec, H.SIM_ACC_OFF, 3.51)
    launches = [
        {"kernel": "k_" + H.ACCEPT_KERNEL + "ILi6E", "words": []},
        {"kernel": "other", "words": []},
    ]
    monkeypatch.setattr(H.DL, "rank_launches", lambda plan, rank: launches)
    monkeypatch.setattr(
        H.DL, "kernargs", lambda launch, rank, parity, addr, mode: bytes(rec)
    )

    def acc(accept):
        out = H.blobs({"mode": 5}, 0, 0, [], accept=accept)
        return [round(struct.unpack_from("<f", b, H.SIM_ACC_OFF)[0], 4) for b in out]

    assert acc("recorded") == [3.51, 3.51]
    assert acc("real") == [0.0, 3.51]
    assert acc(3.2) == [3.2, 3.51]
    for bad in ("sim", 0.5, 7.0):
        with pytest.raises(ValueError):
            H.TkRank(0, accept=bad)


def test_device_build_kernargs_name_the_plans_buffers():
    """The device build's mhc / qkv kernargs put its buffers where the decode plan's launches of the same kernels put
    theirs (``decode_launches``: the step's ``temp:`` / ``param:`` names)."""
    b = object.__new__(TD.Builder)
    names = (
        "X",
        "P",
        "hcA",
        "xn",
        "xq",
        "xs",
        "ws_m",
        "ws_q",
        "rope",
        "pos",
        "qr",
        "q",
        "wk",
        "wsc",
        "kv",
    )
    for k, n in enumerate(names):
        setattr(b, n, SimpleNamespace(data_ptr=lambda k=k: (k + 1) << 20, stride=lambda d: 4))  # fmt: skip
    blobs = {}

    class FakeLauncher:
        def arg_bytes(self, sym):
            return 0

        def params(self, sym, buf):
            blobs[sym] = bytes(buf)

    b.launcher = FakeLauncher()
    p = {s: SimpleNamespace(data_ptr=lambda k=k: (k + 1) << 32, dtype=torch.bfloat16)
         for k, s in enumerate(TD.SLOTS)}  # fmt: skip
    b._launches(p)
    addr = {f"param:{{S+0}}.{s}": (k + 1) << 32 for k, s in enumerate(TD.SLOTS)}
    addr.update({"temp:X2": b.X.data_ptr(), "temp:hcF": b.P.data_ptr(), "temp:hcA": b.hcA.data_ptr(),
                 "temp:Xn": b.xn.data_ptr(), "temp:Xq": b.xq.data_ptr(), "temp:Xs": b.xs.data_ptr(),
                 "temp:ws_mhc": b.ws_m.data_ptr(), "temp:ws_qkv": b.ws_q.data_ptr(), "param:rope0": b.rope.data_ptr(),
                 "temp:vpos": b.pos.data_ptr(), "temp:Qr": b.qr.data_ptr(), "temp:Q": b.q.data_ptr(),
                 "cache:win.{N+0}.kv": b.wk.data_ptr(), "cache:win.{N+0}.s": b.wsc.data_ptr()})  # fmt: skip
    plan = DL.load()
    for sym in (TD.MHC, TD.QKV):
        fam = next(f for f in plan["families"] if plan["kernels"][f["kernel"]] == sym)
        words = struct.unpack(f"<{len(blobs[sym]) // 8}Q", blobs[sym])
        assert len(words) * 8 == fam["nbytes"]
        for k, spec in enumerate(fam["words"]):
            if isinstance(spec, list) and spec[0] == "p":
                assert words[k] == addr[spec[1]] + spec[2], (sym[:40], k, spec)
    kvx = struct.unpack("<3Q", blobs[TD.KVX])
    assert kvx == (b.ws_q.data_ptr(), addr["param:{S+0}.window_kv.kv_norm"], b.kv.data_ptr())  # fmt: skip


def test_device_build_buffers_ignore_the_process_defaults(monkeypatch):
    """A ModelRunner process defaults to bf16 (and its GPU): the build's buffers keep their own dtypes (a bf16 rope
    read as f32 made every q row wrong there), and ``load_toktab`` restores the caller's defaults.
    """
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        b = object.__new__(TD.Builder)
        b.device = torch.device("cpu")
        b._alloc()
        assert b.rope.dtype == torch.float32 and b.rope.shape == (16, 64)
        assert bool((b.rope[:, :32] == 1).all() and (b.rope[:, 32:] == 0).all())
        assert (b.P.dtype, b.hcA.dtype, b.pos.dtype) == (torch.float32, torch.float32, torch.int32)  # fmt: skip
        assert (
            b.P[:, 0].tolist() == [1.0] * TD.ROWS and float(b.P[:, 1:].abs().sum()) == 0
        )
        seen = []
        monkeypatch.setattr(TW, "_load_toktab", lambda *a: seen.append(torch.get_default_dtype()))  # fmt: skip
        TW.load_toktab(None, 0)
        assert seen == [torch.float32] and torch.get_default_dtype() == torch.bfloat16
    finally:
        torch.set_default_dtype(before)


# =====================================================================================================================
# step override: the request's KV (``kv``)


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
        assert all(torch.equal(a, b) for a, b in zip((vals, scs), KV.mxfp4(logical[f"ix{s}"])))  # fmt: skip
    lo = n - 128
    for i, (wkv, wsc) in enumerate(planes["win"]):
        pos = torch.arange(lo, n)
        q, s_ = C.to_window(kv.window[i, pos % 133].float())
        assert torch.equal(wkv.view(torch.uint8)[pos % 256], q) and torch.equal(wsc[pos % 256], s_)  # fmt: skip
        rest = torch.ones(256, dtype=torch.bool)
        rest[pos % 256] = False
        assert not wkv.view(torch.uint8)[rest].any() and not wsc[rest].any()
    for s in (2, 8, 14):
        st = planes["state"][s]
        if n % 2:
            k, sc = kv.compress[s]
            assert torch.equal(st[(n - 1) % 16], torch.stack([k[(n - 1) % 7], sc[(n - 1) % 7]]))  # fmt: skip
            st = torch.cat([st[: (n - 1) % 16], st[(n - 1) % 16 + 1 :]])
        assert not st[:, 0].any() and torch.isinf(st[:, 1]).all()
    h = planes["hist"]
    assert [int(h[(n - 3 + j) % 16]) for j in range(3)] == [11, 12, 13]
    assert int(h.abs().sum()) == 36
    assert set(planes) == {"main_kv", "index_k", "win", "state", "hist"}


def test_mxfp4_and_hist_paths():
    x = torch.randn(5, 128)
    v, s = KV.mxfp4(x)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6] * 2) * torch.tensor([1.0] * 8 + [-1.0] * 8)  # fmt: skip
    deq = torch.stack([lut[(v & 15).long()], lut[(v >> 4).long()]], -1).reshape(5, 4, 32)  # fmt: skip
    step = torch.exp2(s.float() - 127)[..., None]
    # within one E2M1 step of the group scale
    assert ((deq * step - x.view(5, 4, 32)).abs() <= step).all()
    tm = torch.arange(1000) * 3
    assert torch.equal(KV.hist(20, tail_ids=list(range(4, 20)), token_map=tm), C.hist_plane(list(range(4, 20)), 20, tm))  # fmt: skip
    with pytest.raises(ValueError):
        KV.hist(20, cursor=torch.tensor([19, 1, 2, 3]))


# =====================================================================================================================
# step override: the runner on TkRank


class FakeTk:
    """TkRank's logs as admit_kernel and the step write them; scripted accepts."""

    def __init__(self, accepts=(3, 6, 1)):
        self.accepts, self.calls, self.k = accepts, [], 0
        self.acc, self.num = torch.zeros(64, dtype=torch.int32), torch.zeros(64, dtype=torch.int32)  # fmt: skip
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
        assert out.dtype == torch.int32 and out.shape == (1, 6) and verdict.shape == (3, 1)  # fmt: skip
        assert out[0, :a].tolist() == [1000 * (k + 1) + j for j in range(a)] and (out[0, a:] == -1).all()  # fmt: skip
        assert verdict[:, 0].tolist() == [a - 1, 6 - a, a - 1]
        # vtok[1:6] after the step
        assert r.draft_ids(1).tolist() == [[102 + k + j for j in range(5)]]
    assert r.pos == 500  # ATOM's anchor position, no device read
    (_, p1), (_, f, n, none), _ = tk.calls[:3]
    assert tk.calls[0][0] == "set_sampling" and p1[3] == 1 and (f, n, none) == (7, 500, None)  # fmt: skip
    # the next request, same sampling mode: the parameters ride along
    r.run_step(42, ids, 600, 1.0, dict)
    assert tk.calls[-2][:3] == ("admit", 7, 600) and tk.calls[-2][3][3] == 2
    assert tk.calls[-1] == ("import", 600)
    assert (
        tk.calls[-2][3][1] == p1[1] and tk.calls[-2][3][0] != p1[0]
    )  # a seed of its own


def test_sampling_refusals():
    assert RN.sampling_refusal(0.0, 1.0, -1).startswith("greedy")
    assert RN.sampling_refusal(1.0, 0.9, -1) and RN.sampling_refusal(1.0, 1.0, 20)
    assert RN.sampling_refusal(0.7, 1.0, -1) is None


# =====================================================================================================================
# step override: the routing wrapper and its decode round (``dispatch``), the gate


def _served_model(monkeypatch, defers=False):
    """A ``TkStepModel`` over a fake TkRank, its ``TkRound`` on a builder whose ``step_needs_eager`` reads the
    batch's ``eager``; ``defers``: the builder's Engram on the device (held steps skip the verdict wait).
    """
    fc = SimpleNamespace(spec_decode_metadata=object(), attn_metadata=SimpleNamespace(cache=None))  # fmt: skip
    monkeypatch.setattr(D, "get_forward_context", lambda: fc)
    monkeypatch.setattr(D.envs, "ATOM_MONO_CHECK", False, raising=False)
    runner = RN.TkStepRunner(FakeTk())
    runner.output_stream = None
    drafter = SimpleNamespace(compute_draft_kv=lambda *a: "prod-kv", propose=lambda *a: "prod")  # fmt: skip
    m = D.TkStepModel(torch.nn.Identity(), drafter, runner)
    monkeypatch.setattr(D.TkStepModel, "_anchor_position", staticmethod(lambda b: 500))
    monkeypatch.setattr(RN.KV, "request_planes", lambda kv, n, **k: {})
    monkeypatch.setattr(KV, "from_cache", lambda *a: "kv")
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
    # the held request in a step it cannot serve
    for kw in ({"bs": 2}, {"rows": 12}, {"decode": False}):
        with pytest.raises(RuntimeError, match="cannot fall back"):
            _layout(rnd, _batch(5), **kw)
    with pytest.raises(RuntimeError, match="cannot fall back"):
        _layout(rnd, _batch(5, eager=True))
    # another request, greedy: the production path
    assert step(_batch(6, t=0.0)) == (NONE, None)
    assert drafter.propose() == "prod" and drafter.compute_draft_kv() == "prod-kv"
    assert step(_batch(7)) == (LEAN, runner) and runner.holds(
        7
    )  # served: admitted afresh
    assert [c[0] for c in tk.calls] == ["set_sampling", "admit", "import", "admit", "import"]  # fmt: skip


def test_dispatch_serves_exactly_the_steps_laid_out_for_it(monkeypatch):
    m, rnd, _, _ = _served_model(monkeypatch)
    # no decision (a builder without the round): production
    assert m.step_override(IDS, 1, _batch(5)) is None
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
        # the next step starts past its accepted rows
        starts = ours.cursor[list(slots), 0] - torch.tensor(positions)
        assert starts.tolist() == (anchors - step.cu_seqlens_q[:3] + 1).tolist()


def test_dispatch_commits_a_served_step(monkeypatch):
    """The served step's verdict, through ``TkRound.committed`` (``commit_speculative_state``), is where the next
    step's state probe (``prepare_state``: cursor[slot, 0] == the step's start) looks.
    """
    m, rnd, runner, _ = _served_model(monkeypatch)
    pos = 4782
    for k, a in enumerate(runner.dev.accepts):
        b = _batch(5)
        assert _layout(rnd, b) == D.RoundService.LEAN and m.step_override(IDS, 1, b) is runner  # fmt: skip
        _, verdict = runner.verify_outputs(1)
        cache = SimpleNamespace(cursor=torch.zeros(4, 4, dtype=torch.int64), pending=object())  # fmt: skip
        meta = SimpleNamespace(served_by_round=D.RoundService.LEAN, cache=cache, step=_fake_step([pos], [2]))  # fmt: skip
        assert rnd.committed(meta, verdict[2]) and cache.pending is None
        pos += a
        # the next start: the step's accepted tokens past this one's
        assert int(cache.cursor[2, 0]) == pos


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


def test_install_needs_the_v41_builder(monkeypatch):
    monkeypatch.setattr(D, "refusal", lambda c: None)
    runner = RN.TkStepRunner(FakeTk())
    monkeypatch.setattr(RN.TkStepRunner, "create", classmethod(lambda cls, *a, **k: runner))  # fmt: skip
    model, cfg = torch.nn.Identity(), SimpleNamespace(model="m")
    # nothing to keep ATOM's state bookkeeping: refused (the selection then takes the hi / mono paths)
    assert "metadata builder" in D.install_refusal(cfg, None)
    builder = SimpleNamespace(use_decode_round=lambda r: setattr(builder, "round", r))
    assert D.install_refusal(cfg, builder) is None
    wrapped = D.install_tk_step(model, cfg, None, builder)
    assert isinstance(wrapped, D.TkStepModel) and builder.round.model is wrapped


def test_the_gate_refuses_prefix_caching_and_eager(monkeypatch):
    from atom.mono.runtime import deployment

    monkeypatch.setattr(deployment, "common_refusals", lambda *a, **k: [])

    def config(**kw):
        base = {"speculative_config": SimpleNamespace(method="dspark", num_speculative_tokens=5),
                "dspark": SimpleNamespace(confidence_schedule=False, ragged=False), "max_num_seqs": 1,
                "max_model_len": 65536, "enable_prefix_caching": False, "enforce_eager": False,
                "kv_cache_dtype": "bf16", "index_cache_dtype": "fp8", "kv_cache_block_size": 128}  # fmt: skip
        return SimpleNamespace(**{**base, **kw})

    assert gate.refusal(config()) is None
    assert "prefix caching" in gate.refusal(config(enable_prefix_caching=True))
    assert "enforce_eager" in gate.refusal(config(enforce_eager=True))
    # the caches its KV conversion reads (the hi path's FP4 index is not one)
    assert "--index-cache-dtype fp4" in gate.refusal(config(index_cache_dtype="fp4"))
    assert "--kv_cache_dtype fp8" in gate.refusal(config(kv_cache_dtype="fp8"))
    assert "--block-size 16" in gate.refusal(config(kv_cache_block_size=16))
    # a field missing: a deployment it is not written for, not an error
    assert "max_num_seqs None" in gate.refusal(SimpleNamespace(**{k: v for k, v in vars(config()).items() if k != "max_num_seqs"}))  # fmt: skip


def test_the_gate_obeys_the_mono_master_switch(monkeypatch):
    """ATOM_MONO_ENABLE=0 turns every mono path off, the TileRT-path step too; on, it is gfx950's, with no switch of
    its own (``mono.selection`` then asks its install refusal)."""
    monkeypatch.setattr(gate, "_arch", lambda: "gfx950")
    monkeypatch.setenv("ATOM_MONO_ENABLE", "0")
    assert not gate.enabled()
    monkeypatch.setenv("ATOM_MONO_ENABLE", "1")
    assert gate.enabled()
    for arch in ("gfx942", "gfx1250", None):
        monkeypatch.setattr(gate, "_arch", lambda a=arch: a)
        assert not gate.enabled()


# =====================================================================================================================
# PD server: the stack, the prefill rank, the router

MAX_SEQ = 1 << 20
FAKE_RANK = "atom.models.deepseek_v41.mono.gfx950.pd.fanout:FakeRank"


class FakeTransport:
    name = "mooncake"

    def local_meta(self) -> dict:
        return {"session_id": "127.0.0.1:1"}


@contextmanager
def stack(**engine_kw):
    """A receive server, a decode server over ``FakeEngine(**engine_kw)``, and the landed buffer (its header)."""
    engine = S.FakeEngine(max_seq_len=MAX_SEQ, **engine_kw)
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    recv = R.ReceiveServer(MAX_SEQ, 0x7F0000000000, FakeTransport(), port=0, host="127.0.0.1", buffer=buf)  # fmt: skip
    app = SV.DecodeServer(recv, S.Decoder(engine), poll_s=0.002, abandon_drain_s=0.3)
    http_port = app.start()
    try:
        yield engine, recv, app, http_port, buf
    finally:
        app.shutdown()
        recv.close()


@contextmanager
def rank_stack(nranks=8, timeouts=None, **rank_kw):
    """The decode server over ``fanout.RankEngine`` and ``nranks`` spawned ``FakeRank`` workers."""
    group = F.RankGroup(nranks, FAKE_RANK, rank_kw, timeouts=timeouts)
    try:
        eng = F.RankEngine(group)
        info = group.reports[0]
        transport = R.PublishedTransport(info["transport"], {"session_id": info["session_id"]})  # fmt: skip
        recv = R.ReceiveServer(eng.max_seq_len, info["base_ptr"], transport, port=0, host="127.0.0.1")  # fmt: skip
        app = SV.DecodeServer(recv, S.Decoder(eng), poll_s=0.002, abandon_drain_s=0.3)
        port = app.start()
        try:
            yield group, eng, recv, port
        finally:
            app.shutdown()
            recv.close()
    finally:
        group.close()


def land(buf, seq_len: int) -> None:
    """What P's RDMA write leaves in the header (the rest of the image is not read by ``FakeEngine``)."""
    h = buf[: W.HDR_BYTES].view(torch.int32)
    h.zero_()
    h[W.H_MAGIC], h[W.H_VER], h[W.H_N] = W.WIRE_MAGIC, W.WIRE_VERSION, seq_len


def send_kv(ctrl_port: int, rid: str, seq_len: int, buf, done: bool = True, sampling=None):  # fmt: skip
    """P rank 0: hello, admission, the header written (the RDMA write's stand-in), done. ``(hello, reply, conn)``;
    ``conn`` stays open when ``done`` is False."""
    c = socket.create_connection(("127.0.0.1", ctrl_port), timeout=10)
    hello = W.recv_msg(c)
    msg = {"rid": rid, "rank": 0, "seq_len": seq_len, "last_prompt_token": 7, "sampling": sampling or {},
           "admission_window_s": 0.6}  # fmt: skip
    W.send_msg(c, msg)
    reply = W.recv_msg(c)
    if reply.get("accepted") and done:
        land(buf, seq_len)
        W.send_msg(c, W.done_msg(rid, 0, reply["generation"]))
    if done or not reply.get("accepted"):
        c.close()
        c = None
    return hello, reply, c


def post(port: int, path: str, body) -> tuple[int, dict]:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json"})  # fmt: skip
    r = c.getresponse()
    out = r.status, json.loads(r.read() or b"null")
    c.close()
    return out


def get(port: int, path: str) -> dict:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("GET", path)
    r = c.getresponse()
    assert r.status == 200, r.status
    out = json.loads(r.read())
    c.close()
    return out


def stream(port: int, body: dict, on_line=None):
    """``(status, lines)`` of a streamed ``/pd/decode``; ``on_line(line, conn)`` after each line (may close ``conn``
    and return True to stop reading)."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("POST", "/pd/decode", body=json.dumps({**body, "stream": True}), headers={"Content-Type": "application/json"})  # fmt: skip
    r = c.getresponse()
    if r.status != 200:
        out = r.status, json.loads(r.read())
        c.close()
        return out
    assert r.getheader("Content-Type") == "application/x-ndjson"
    lines = []
    while True:
        raw = r.readline()
        if not raw:
            break
        lines.append(json.loads(raw))
        if on_line is not None and on_line(lines[-1], c):
            break
    c.close()
    return 200, lines


def leave(line, conn):
    """``stream``'s ``on_line``: the client goes away after the first line."""
    conn.sock.shutdown(socket.SHUT_RDWR)
    return True


def cancel_after(port: int, rid: str, first: int):
    """``stream``'s ``on_line``: ``/pd/cancel`` once the first token is in."""

    def on_line(line, conn):
        if line == {"t": [first]}:
            assert post(port, "/pd/cancel", {"rid": rid}) == (200, {"cancelled": rid})
        return False

    return on_line


def read_reply(lines: list[dict]) -> tuple[list[int], dict]:
    """TileRT's ``DecodeReader``: the tokens of the ``t`` lines and the terminal line, which must be the last and
    declare every token streamed."""
    toks = [t for ln in lines if "t" in ln for t in ln["t"]]
    term = lines[-1]
    assert "done" in term, term
    assert term["n"] == len(toks), (term["n"], len(toks))
    assert all("t" in ln for ln in lines[:-1]), lines
    return toks, term


def _reply(lines):
    """``(tokens, terminal line)`` of a finished reply, or ``(None, the error line)``."""
    if lines and "error" in lines[-1]:
        return None, lines[-1]
    return read_reply(lines)


def expected(first: int, n: int, eos_at=None) -> list[int]:
    return [first] + [S.EOS_ID if j == eos_at else S.FakeEngine.token(first, j) for j in range(1, n)]  # fmt: skip


def wait_idle(port: int, timeout: float = 10.0) -> dict:
    t = time.time() + timeout
    while time.time() < t:
        st = get(port, "/decode_status")
        if st["status"] == "idle":
            return st
        time.sleep(0.01)
    raise AssertionError(f"still busy: {st}")


# =====================================================================================================================
# PD server: one engine (``session.FakeEngine``)


def test_stream_eos_then_fast_path_and_length():
    with stack(eos_at=10) as (eng, recv, _app, port, buf):
        assert get(port, "/health") == {"status": "ok"}
        caps = get(port, "/capabilities")
        assert caps["profile"] == "dsv41"
        assert caps["capabilities"] == {"penalties": False, "ignore_eos": True, "logprobs": False}  # fmt: skip

        # the decode call first, the KV landing while it waits
        def late_p():
            time.sleep(0.2)
            late_p.hello, late_p.reply, _ = send_kv(recv.port, "r1", 1000, buf)

        th = threading.Thread(target=late_p)
        th.start()
        body = {"rid": "r1", "first_token_id": 12345, "max_tokens": 256, "sampling": {"temperature": 1.0}}  # fmt: skip
        status, lines = stream(port, body)
        th.join()
        assert status == 200
        h = late_p.hello
        assert (h["magic"], h["protocol_version"], h["layout_version"]) == ("tilert-pd", 2, 2041)  # fmt: skip
        assert (h["transport"], h["session_id"], h["max_seq_len"], h["busy"]) == ("mooncake", "127.0.0.1:1", MAX_SEQ, False)  # fmt: skip
        assert (h["senders"], h["dsv41_base"], h["dsv41_wire"]) == (1, 0x7F0000000000, 1)  # fmt: skip
        assert late_p.reply == {"accepted": True, "rid": "r1", "rank": 0, "generation": 1}  # fmt: skip
        assert lines[0] == {"t": [12345]}  # the first token alone
        toks, term = read_reply(lines)
        assert toks == expected(12345, 10)  # EOS (index 10) ends the reply, unstreamed
        assert (term["finish_reason"], term["seq_len"]) == ("stop", 1000)
        assert term["timing_ms"]["mtp_verify_calls"] >= 3
        assert eng.calls[:3] == [("set_sampling", True), ("admit", 12345, 1000, False), ("load_kv", "r1", 1000)]  # fmt: skip
        assert eng.params[1] == S.pack_params(S.SamplingParams())[1] == 0x3F8000003F800000  # fmt: skip
        assert wait_idle(port)["current_rid"] is None
        assert recv.state_snapshot()["rid"] is None

        # the KV first, then the decode call; same sampling mode: the admission carries the parameters
        send_kv(recv.port, "r2", 2000, buf)
        eng.calls.clear()
        body = {"rid": "r2", "first_token_id": 777, "max_tokens": 23, "sampling": {"temperature": 1.0, "ignore_eos": True}}  # fmt: skip
        status, lines = stream(port, body)
        toks, term = read_reply(lines)
        assert toks == expected(
            777, 23, eos_at=10
        )  # ignore_eos: EOS streamed like any token
        assert term["finish_reason"] == "length"
        assert eng.calls[:2] == [("admit", 777, 2000, True), ("load_kv", "r2", 2000)]
        steps = [c[1] for c in eng.calls if c[0] == "step"]
        assert all(1 <= k <= S.AR_STEPS for k in steps)
        assert sum(steps) * 6 >= 22  # enough steps were asked for the budget
        wait_idle(port)


def test_non_stream_and_greedy_mode_switch():
    with stack() as (eng, recv, _app, port, buf):
        send_kv(recv.port, "r3", 64, buf)
        body = {"rid": "r3", "first_token_id": 5, "max_tokens": 9, "sampling": {"temperature": 0}}  # fmt: skip
        st, out = post(port, "/pd/decode", body)
        assert st == 200, out
        assert out["rid"] == "r3" and out["seq_len"] == 64
        assert out["token_ids"] == expected(5, 9)
        assert out["timing_ms"]["finish_reason"] == "length"
        assert eng.calls[0] == ("set_sampling", False)  # greedy
        assert wait_idle(port)["status"] == "idle"


def test_refusals_release_the_request():
    with stack(step_s=0.01) as (_eng, recv, _app, port, buf):
        st, out = post(port, "/pd/decode", {"first_token_id": 1})
        assert st == 422 and "rid" in out["detail"]
        cases = [
            ({"sampling": {"temperature": 1.0, "top_p": 0.9}}, 501, "capability_unavailable"),
            ({"sampling": {"repetition_penalty": 1.2}}, 501, "capability_unavailable"),
            ({"sampling": {"temperature": -1}}, 400, "invalid_parameter"),
            ({"grammar_spec": {"type": "json_object"}}, 501, "grammar_unsupported"),
            ({"top_logprobs": 2}, 501, "logprobs_unavailable"),
        ]  # fmt: skip
        for i, (extra, status, etype) in enumerate(cases):
            st, out = post(port, "/pd/decode", {"rid": f"x{i}", "first_token_id": 3, **extra})  # fmt: skip
            assert (st, out.get("error_type")) == (status, etype), (extra, st, out)
            assert wait_idle(port)["current_rid"] is None
        # a refused request's late KV is refused too (tombstone), so it cannot land on the next request
        _, reply, _ = send_kv(recv.port, "x0", 10, buf)
        assert reply == {"accepted": False, "error": "cancelling", "rid": "x0", "detail": "no_consumer"}  # fmt: skip
        st, out = post(port, "/pd/decode", {"rid": "late", "first_token_id": 3, "timeout_s": 0.3})  # fmt: skip
        assert (st, out) == (504, {"error": "kv_transfer_timeout", "rid": "late"})

        # busy while a request decodes
        send_kv(recv.port, "long", 100, buf)
        got, res = threading.Event(), {}
        body = {"rid": "long", "first_token_id": 9, "max_tokens": 300}
        th = threading.Thread(target=lambda: res.update(r=stream(port, body, lambda line, conn: got.set())))  # fmt: skip
        th.start()
        assert got.wait(10)
        st, out = post(port, "/pd/decode", {"rid": "other", "first_token_id": 3})
        assert (st, out) == (429, {"error": "busy", "current_rid": "long"})
        th.join()
        toks, term = read_reply(res["r"][1])
        assert len(toks) == 300 and term["finish_reason"] == "length"
        wait_idle(port)


def test_cancel_and_client_gone():
    with stack(step_s=0.005) as (eng, recv, _app, port, buf):
        send_kv(recv.port, "c1", 100, buf)
        body = {"rid": "c1", "first_token_id": 42, "max_tokens": 100000}
        st, lines = stream(port, body, cancel_after(port, "c1", 42))
        assert st == 200
        toks, term = read_reply(lines)
        assert term["finish_reason"] == "cancelled"
        assert toks == expected(42, len(toks)) and len(toks) < 2000
        assert post(port, "/pd/cancel", {"rid": "c1"})[0] == 404
        wait_idle(port)
        # the client leaves mid-stream: the decode is cancelled and the request released
        send_kv(recv.port, "c2", 100, buf)
        stream(port, {"rid": "c2", "first_token_id": 43, "max_tokens": 100000}, leave)
        assert wait_idle(port)["current_rid"] is None
        assert eng.next_index < 20000  # stopped well before the budget


def test_stream_release_waits_for_the_decode_worker():
    """A streamed decode whose worker outlives its stream (here a slow chunk when the client leaves) keeps the
    request until the worker stops: a decode call meanwhile is answered busy, not started beside it.
    """
    with stack(step_s=0.05) as (_eng, recv, app, port, buf):
        app.join_s = 0.01
        send_kv(recv.port, "w1", 100, buf)
        stream(port, {"rid": "w1", "first_token_id": 3, "max_tokens": 100000}, leave)
        time.sleep(0.1)
        assert get(port, "/decode_status")["status"] == "busy"
        assert post(port, "/pd/decode", {"rid": "w2", "first_token_id": 4})[0] == 429
        assert wait_idle(port)["current_rid"] is None


def test_control_plane_rejects():
    with stack() as (_eng, recv, _app, port, buf):
        _, reply, _ = send_kv(recv.port, "big", MAX_SEQ + 1, buf)
        assert reply["accepted"] is False and reply["error"] == "seq_len exceeds max_seq_len"  # fmt: skip
        _, r1, held = send_kv(
            recv.port, "a", 10, buf, done=False
        )  # a writer still transferring
        assert r1["accepted"]
        hello, r2, _ = send_kv(recv.port, "b", 10, buf)
        assert hello["busy"] is True
        assert r2 == {"accepted": False, "error": "busy", "busy_rid": "a", "busy_state": "transferring"}  # fmt: skip
        land(buf, 10)
        W.send_msg(held, W.done_msg("a", 0, r1["generation"]))
        held.close()
        st, out = post(port, "/pd/decode", {"rid": "a", "first_token_id": 2, "max_tokens": 4})  # fmt: skip
        assert st == 200 and out["token_ids"] == expected(2, 4)
        _, again, _ = send_kv(
            recv.port, "a", 10, buf
        )  # a late duplicate of a finished request
        assert again["accepted"] is False and again["error"] == "cancelling"
        _, r3, _ = send_kv(recv.port, "b", 10, buf)
        assert r3["accepted"] and r3["generation"] == r1["generation"] + 1


def test_receive_slot_held_while_its_kv_is_read():
    """While the decode call reads a request's KV out of the buffer (``claim`` .. ``read_done``), a P of another
    request (one left behind by an abandoned request, landing late) is refused busy; after it, TileRT's rule.
    """
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    recv = R.ReceiveServer(MAX_SEQ, 0, FakeTransport(), port=0, host="127.0.0.1", buffer=buf)  # fmt: skip
    try:
        send_kv(recv.port, "a", 100, buf)
        rec = recv.completed.get(timeout=5)
        assert recv.claim(rec)
        _, reply, _ = send_kv(recv.port, "late", 200, buf)
        assert reply["error"] == "busy" and reply["busy_rid"] == "a"
        recv.read_done("a")
        _, reply, _ = send_kv(recv.port, "late", 200, buf)
        assert reply["accepted"]
        assert not recv.claim(rec)  # the slot moved on: a's buffer may hold late's KV
        recv.release("a")
        assert recv.state_snapshot()["rid"] == "late"
    finally:
        recv.close()


def _received(rid, seq_len=100):
    return W.ReceivedRequest(rid=rid, seq_len=seq_len, last_prompt_token=0, first_token_id=None, sampling=None)  # fmt: skip


def test_decoder_chunks_and_unbegun_path():
    eng = S.FakeEngine(max_seq_len=4096, accepts=(6,))
    dec = S.Decoder(eng)
    dec.inject(
        _received("u", 4000), None
    )  # no begin: decode sets the mode and admits after the first token
    got = []
    toks = dec.decode(11, 500, {"temperature": 1.0}, on_token=got.append)
    # budget min(500, 4096 - 4000 - 1) = 95; but a chunk only goes where the cache has room for all its rows with 8
    # positions of slack: (4096 - 4001 - 8) // 6 = 14 steps, so 8 + 6 steps, 84 tokens after the first
    assert toks == got == expected(11, len(toks))
    assert dec.last_stats["finish_reason"] == "length" and len(toks) == 85
    assert [c[1] for c in eng.calls if c[0] == "step"] == [8, 6]
    assert eng.calls[0] == ("load_kv", "u", 4000) and eng.calls[1][0] == "set_sampling"
    assert eng.calls[2] == ("admit", 11, 4000, False)
    assert S.pack_params(S.SamplingParams(temperature=0.5, seed=-1))[0] == -1


def test_decoder_discards_what_a_reply_leaves():
    """``Decoder`` over ``FakeEngine``: a stop with a chunk in flight discards it; a budget end has none; a failure
    mid-reply discards and forgets the sampling mode, so the next request sets it again.
    """
    eng = S.FakeEngine(max_seq_len=1 << 16, eos_at=10)
    dec = S.Decoder(eng)

    def run(first, max_tokens, sampling, on_token=None):
        dec.begin(first, sampling)
        dec.inject(_received(str(first)), None)
        eng.calls.clear()
        return dec.decode(first, max_tokens, sampling, on_token=on_token)

    assert run(5, 256, {"temperature": 1.0}) == expected(5, 10)
    assert eng.calls[-1] == ("discard", 1) and dec.last_stats["discarded"] == 1
    run(6, 20, {"temperature": 1.0, "ignore_eos": True})
    assert dec.last_stats["finish_reason"] == "length"
    assert not any(c[0] == "discard" for c in eng.calls)

    def boom(tok):
        if tok != 7:
            raise ConnectionError("client gone")

    with pytest.raises(ConnectionError):
        run(7, 256, {"temperature": 1.0, "ignore_eos": True}, on_token=boom)
    assert eng.calls[-1][0] == "discard" and dec._samp_mode is None
    eng.calls.clear()
    dec.begin(8, {"temperature": 1.0})
    dec.inject(_received("8"), None)
    # not the parameters riding with the admission
    assert eng.calls[0] == ("set_sampling", True)


# =====================================================================================================================
# PD server: the rank group (``fanout``) and TkRank's worker (``tkworker``)


def test_fanout_eight_ranks_end_to_end():
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    with rank_stack(eos_at=10) as (group, eng, recv, port):
        assert [r["rank"] for r in group.reports] == list(range(8))
        assert len({r["pid"] for r in group.reports}) == 8  # one process a rank
        send_kv(recv.port, "f1", 1000, buf)
        body = {"rid": "f1", "first_token_id": 12345, "max_tokens": 256, "sampling": {"temperature": 1.0}}  # fmt: skip
        status, lines = stream(port, body)
        toks, term = read_reply(lines)
        assert status == 200 and toks == expected(12345, 10) and term["finish_reason"] == "stop"  # fmt: skip
        send_kv(recv.port, "f2", 2000, buf)
        body = {"rid": "f2", "first_token_id": 777, "max_tokens": 23, "sampling": {"temperature": 1.0, "ignore_eos": True}}  # fmt: skip
        status, lines = stream(port, body)
        toks, term = read_reply(lines)
        assert (
            toks == expected(777, 23, eos_at=10) and term["finish_reason"] == "length"
        )
        states = [a.json() for a in group.call("state")]
        logs = [st["log"] for st in states]
        assert all(lg == logs[0] for lg in logs)  # every rank got the same commands
        # f1 stopped with a chunk in flight: every rank dropped it
        assert [e[0] for e in logs[0]] == ["set_sampling", "admit", "load_kv", "discard", "admit", "load_kv"]  # fmt: skip
        assert all(st["in_flight"] == [] for st in states)
        (_, p1, on), (_, f, pos, mtp, p_none) = logs[0][0], logs[0][1]
        assert on and p1[3] == 1 and (f, pos, mtp, p_none) == (12345, 1000, True, None)
        # same sampling mode: the parameters ride with the admission; [0]: a fresh seed
        _, f, pos, mtp, p2 = logs[0][4]
        assert (f, pos, mtp) == (777, 2000, True) and p2[1:3] == p1[1:3] and p2[3] == 2
        assert eng.counter == 2
        for st in states:  # each rank's AdmitArgs over its own buffers
            blob, addr = bytes.fromhex(st["admit_blob"]), st["addr"]
            assert blob == H.admit_args(addr, 777, 2000, True, p2)
            assert int.from_bytes(blob[0x88:0x90], "little") == addr["temp:samp_params"]
        assert wait_idle(port)["current_rid"] is None and recv.state_snapshot()["rid"] is None  # fmt: skip


def test_fanout_failures():
    # an op raising on one rank: RankError with that rank's traceback; the group stays usable
    g = F.RankGroup(2, FAKE_RANK, {"fail": {"op": "admit", "rank": 1}})
    try:
        eng = F.RankEngine(g)
        with pytest.raises(F.RankError) as e:
            eng.admit(5, 10, None)
        assert e.value.op == "admit" and sorted(e.value.errors) == [1]
        assert "injected failure" in e.value.errors[1]
        assert len(g.call("state")) == 2 and g.broken is None
    finally:
        g.close()
    # an op not answered in time: RankTimeout, then the group is broken
    g = F.RankGroup(2, FAKE_RANK, {"hang": {"op": "step", "rank": 1, "s": 5}}, timeouts={"step": 0.3})  # fmt: skip
    try:
        t = time.monotonic()
        with pytest.raises(F.RankTimeout, match=r"\[1\]"):
            g.call("step", [1, 1])
        assert time.monotonic() - t < 3
        with pytest.raises(F.RankGroupBroken):
            g.call("state")
    finally:
        g.close(graceful=False)
    # a worker gone: RankDied at once, not at the timeout
    g = F.RankGroup(2, FAKE_RANK, {"hang": {"op": "sync", "rank": 0, "s": 30}})
    try:
        g.procs[1].kill()
        g.procs[1].join(5)
        t = time.monotonic()
        with pytest.raises(F.RankDied, match=r"\[1\]"):
            g.call("state")
        assert time.monotonic() - t < 3
    finally:
        g.close(graceful=False)
    # a backend that cannot start: its traceback at startup
    with pytest.raises(F.RankError) as e:
        F.RankGroup(2, FAKE_RANK, {"no_such_arg": 1})
    assert e.value.op == "hello" and "no_such_arg" in e.value.errors[0]


def test_fanout_stop_with_chunks_in_flight_then_many_requests():
    """A reply that ends at a stop id leaves the chunk enqueued after the one holding it (two in flight): every rank
    drops it at the reply's end (``discard``), so any number of such replies run back to back. Before: nothing popped
    it, so the fifth one left behind made the next request's first ``step`` fail on every rank ("more than 4 chunks in
    flight"). A reply ended by its budget leaves none."""
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    replies, states = [], []
    with rank_stack(nranks=4, eos_at=10) as (group, _eng, recv, port):
        for i in range(9):
            rid, length = f"s{i}", i % 3 == 2
            send_kv(recv.port, rid, 1000 + i, buf)
            body = {"rid": rid, "first_token_id": 100 + i, "max_tokens": 40 if length else 256,
                    "sampling": {"temperature": 1.0, "ignore_eos": length}}  # fmt: skip
            replies.append((length, i, stream(port, body)[1]))
            wait_idle(port)
            states.append([a.json() for a in group.call("state")])
        for length, i, lines in replies:  # every reply finished
            toks, term = _reply(lines)
            assert toks is not None, (i, term)
            if length:
                assert term["finish_reason"] == "length" and len(toks) == 40
                assert "discarded" not in term["timing_ms"]
            else:
                assert term["finish_reason"] == "stop" and toks == expected(100 + i, 10)
                assert term["timing_ms"]["discarded"] == 1
        # nothing carried over, on any rank
        assert all(st["in_flight"] == [] for sts in states for st in sts)
        logs = [st["log"] for st in states[-1]]
        assert all(lg == logs[0] for lg in logs)  # every rank got the same commands
        assert sum(e[0] == "discard" for e in logs[0]) == 6
        assert group.broken is None


def test_fanout_rank_failure_mid_reply_leaves_no_chunks():
    """An op failing on one rank mid-reply (here rank 0's ``wait_read``, before it pops the chunk, as a KeyError there
    would): the ranks' in-flight chunks differ, and the failed reply's ``discard`` empties every rank's; the error
    names the rank's last commands."""
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    with rank_stack(nranks=2, fail={"op": "wait_read", "rank": 0}) as (group, _eng, recv, port):  # fmt: skip
        send_kv(recv.port, "x1", 500, buf)
        _, lines = stream(port, {"rid": "x1", "first_token_id": 9, "max_tokens": 256})
        toks, err = _reply(lines)
        assert toks is None and "wait_read failed on rank(s) [0]" in err["error"]
        assert "recent commands" in err["error"] and "step[" in err["error"]
        wait_idle(port)
        states = [a.json() for a in group.call("state")]
        assert [s["in_flight"] for s in states] == [[], []]
        (d0,), (d1,) = ([e for e in s["log"] if e[0] == "discard"] for s in states)
        # rank 0 still held the chunk it failed on, rank 1 had popped it; both dropped what they held
        assert d0[1] == d1[1] == d1[2] and len(d0[2]) == len(d1[2]) + 1
        assert group.broken is None


def test_fanout_cancel_reconnect_and_abandoned_request():
    """The sequence around a router restart, every request on every rank: a cancel after the first token (the chunks
    in flight still stream: nothing to drop), a client gone mid-stream, a request abandoned before its KV came (504;
    its late P refused), a P of a request this node never heard of (the old router's: admitted, landed, never asked
    for), then a request that stops early (its stale predecessor in the completed queue dropped, its own leftover chunk
    discarded). No rank keeps a chunk, every rank got the same commands, the group stays whole.
    """
    buf = torch.zeros(W.HDR_BYTES, dtype=torch.uint8)
    no_eos = {"temperature": 1.0, "ignore_eos": True}
    with rank_stack(nranks=4, step_s=0.002, eos_at=30) as (group, _eng, recv, port):

        def in_flight():
            wait_idle(port)
            return [a.json()["in_flight"] for a in group.call("state")]

        send_kv(recv.port, "c1", 100, buf)
        body = {"rid": "c1", "first_token_id": 42, "max_tokens": 100000, "sampling": no_eos}  # fmt: skip
        toks, term = _reply(stream(port, body, cancel_after(port, "c1", 42))[1])
        assert term["finish_reason"] == "cancelled" and "discarded" not in term["timing_ms"]  # fmt: skip
        assert in_flight() == [[]] * 4

        send_kv(recv.port, "c2", 100, buf)
        stream(port, {"rid": "c2", "first_token_id": 43, "max_tokens": 100000, "sampling": no_eos}, leave)  # fmt: skip
        assert in_flight() == [[]] * 4

        # abandoned: the decode call gives up on its KV; its P, late, is refused
        st, out = post(port, "/pd/decode", {"rid": "gone", "first_token_id": 1, "timeout_s": 0.3})  # fmt: skip
        assert st == 504 and out["error"] == "kv_transfer_timeout"
        _, reply, _ = send_kv(recv.port, "gone", 100, buf)
        assert reply["error"] == "cancelling" and reply["detail"] == "no_consumer"
        # the old router's request, never asked for here: its P lands
        _, reply, _ = send_kv(recv.port, "orphan", 300, buf)
        assert reply["accepted"]
        # the next request: the orphan is dropped from the completed queue, the reply stops with a chunk in flight
        send_kv(recv.port, "n1", 4000, buf)
        _, lines = stream(port, {"rid": "n1", "first_token_id": 77, "max_tokens": 256})
        toks, term = _reply(lines)
        assert toks == expected(77, 30) and term["finish_reason"] == "stop" and term["seq_len"] == 4000  # fmt: skip
        assert term["timing_ms"]["discarded"] == 1
        assert in_flight() == [[]] * 4
        logs = [a.json()["log"] for a in group.call("state")]
        assert all(lg == logs[0] for lg in logs)
        assert [e[1] for e in logs[0] if e[0] == "load_kv"] == [100, 100, 4000]
        assert group.broken is None and recv.state_snapshot()["rid"] is None


def test_rank_group_one_caller_at_a_time():
    """A second thread's call while one is in flight is refused (``RankGroupBusy``) instead of overwriting the
    mailbox under ranks that have not read it yet; the call in flight completes and the group stays whole.
    """
    g = F.RankGroup(2, FAKE_RANK, {"hang": {"op": "set_sampling", "rank": 0, "s": 1.0}})
    try:
        out = {}

        def first():
            out["first"] = g.call("set_sampling", [1, 1, 2, 3, 4])

        th = threading.Thread(target=first, name="first")
        th.start()
        time.sleep(0.2)
        with pytest.raises(F.RankGroupBusy, match=r"\('set_sampling', 'first'\)"):
            g.call("discard", [7])
        th.join(10)
        assert len(out["first"]) == 2 and g.broken is None
        assert [a.res for a in g.call("discard", [])] == [[], []]
        logs = [a.json()["log"] for a in g.call("state")]
        # the refused call reached no rank
        assert logs[0] == logs[1] == [["set_sampling", [1, 2, 3, 4], True], ["discard", [], []]]  # fmt: skip
    finally:
        g.close()


def _watch_cmd(shm_name, n, out):
    """Another process polling slot 0's command number, as a worker does: every value it reads, counted."""
    from multiprocessing import shared_memory

    shm = shared_memory.SharedMemory(name=shm_name)
    slot = F._Slot(shm.buf, 0)
    zeros, seen = 0, 0
    try:
        while seen < n:
            v = slot.cmd()
            if v == 0 and seen:
                zeros += 1
            seen = max(seen, v)
    finally:
        slot.release()
        shm.close()
    out.put(zeros)


def test_mailbox_sequence_number_never_reads_zero():
    """A posted command number is one aligned store: a reader in another process never sees the 0 that
    ``struct.pack_into`` wrote first (it zeroes its target before the fields), which ran a payload twice.
    """
    import multiprocessing as mp
    from multiprocessing import shared_memory

    n = 200_000
    shm = shared_memory.SharedMemory(create=True, size=F.SLOT)
    shm.buf[: F.SLOT] = bytes(F.SLOT)
    slot = F._Slot(shm.buf, 0)
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    p = ctx.Process(target=_watch_cmd, args=(shm.name, n, out))
    try:
        p.start()
        time.sleep(0.5)
        for seq in range(1, n + 1):
            slot.post(seq, F.OPS["step"], [8, seq])
        assert out.get(timeout=60) == 0
        p.join(timeout=10)
    finally:
        if p.is_alive():
            p.terminate()
        slot.release()
        shm.close()
        shm.unlink()


def test_tk_worker_discard_and_missing_chunk():
    w = TW.TkRankWorker.__new__(TW.TkRankWorker)
    w.rank, w.events = 3, {5: object(), 6: object()}
    with pytest.raises(RuntimeError, match=r"rank 3: chunk 9 is not in flight \(in flight: \[5, 6\]\)"):  # fmt: skip
        w.wait_read(9, 1, 8, 1)
    assert w.discard([6]) == [5, 6] and w.events == {}


def test_tk_worker_greedy():
    eng = F.RankEngine.__new__(F.RankEngine)
    eng._S, eng.greedy = S, False
    with pytest.raises(S.CapabilityUnavailable, match="mode-2"):
        eng.check_sampling(None)
    eng.check_sampling(S.SamplingParams())


def test_load_planes_file(tmp_path):
    planes = {"main_kv": {2: (torch.zeros(3, 256, dtype=torch.uint8),)}, "index_k": {}, "win": [], "state": {}}  # fmt: skip
    torch.save({"n": 6, "first": 9, "planes": {**planes, "hist": torch.zeros(16, dtype=torch.int32)}}, tmp_path / "ok.pt")  # fmt: skip
    snap = TW.load_planes_file(str(tmp_path / "ok.pt"), 6)
    assert (snap["n"], snap["first"]) == (6, 9)
    assert snap["planes"]["main_kv"][2][0].shape == (3, 256)
    for bad, args in (({"n": 6, "first": 9, "planes": planes}, (6,)), ({"n": 6}, (None,))):  # fmt: skip
        torch.save(bad, tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            TW.load_planes_file(str(tmp_path / "bad.pt"), *args)
    with pytest.raises(ValueError, match="6 positions"):
        TW.load_planes_file(str(tmp_path / "ok.pt"), 7)


def _share(rank, port, q):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)  # fmt: skip
    planes = None
    if rank == 0:
        win = (torch.full((2, 4), 0.5).to(torch.float8_e4m3fn), torch.ones(2, dtype=torch.uint8))  # fmt: skip
        planes = {"main_kv": {2: (torch.arange(6, dtype=torch.uint8).view(2, 3),)}, "win": [win]}  # fmt: skip
    got = TW.share_planes(planes)
    w = got["win"][0]
    q.put((rank, got["main_kv"][2][0].tolist(), w[0].dtype == torch.float8_e4m3fn, w[0].float().tolist(), w[1].tolist()))  # fmt: skip
    dist.destroy_process_group()


def test_share_planes_over_gloo():
    """``kv_share="gloo"``: rank 0's converted planes reach every rank, dtypes kept."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = torch.multiprocessing.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_share, args=(r, port, q)) for r in range(2)]
    for p_ in ps:
        p_.start()
    got = sorted(q.get(timeout=120) for _ in ps)
    for p_ in ps:
        p_.join(30)
    assert [g[0] for g in got] == [0, 1]
    assert all(g[1:] == ([[0, 1, 2], [3, 4, 5]], True, [[0.5] * 4] * 2, [1, 1]) for g in got)  # fmt: skip


def test_server_flags_for_a_real_run():
    a = SV.build_parser().parse_args(["--engine", "tk", "--with-mtp", "--num-mtp", "5", "--accept", "recorded", "--kv-share", "ipc"])  # fmt: skip
    assert (a.accept, a.kv_share, a.max_seq_len, a.ctrl_port, a.http_port) == ("recorded", "ipc", 1 << 20, 5556, 5557)  # fmt: skip
    d = SV.build_parser().parse_args([])
    assert (d.accept, d.kv_share) == ("real", "ipc")


def test_imports_without_gpu_stack():
    pd = "atom.models.deepseek_v41.mono.gfx950.pd"
    code = (
        f"import sys; import {pd}.server, {pd}.tkworker; "
        "print(sorted({m.split('.')[0] for m in sys.modules} & {'aiter', 'flydsl', 'triton'}))"
    )
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join([str(ROOT), *sys.path])}  # fmt: skip
    out = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=120, check=False)  # fmt: skip
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout

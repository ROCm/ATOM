# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TileRT-derived decode host's own artifacts on the CPU (``mono/hi/tk``): no GPU, no aiter, no FlyDSL.

- ``flybuild``: one recipe per kernel the plan launches plus ``admit_kernel``, each named by its TileRT symbol; the
  cache key follows the sources a recipe imports and its build switches; ``code_table`` builds what the cache lacks,
  once, and maps every TileRT symbol to its build's function and layout; a swap table round trip;
- ``host.admit_reference``: ``admit_kernel``'s effect, for tkserve's request shapes and a reset;
- ``toktab``: the float32 primitives (``fma32`` against exact rationals, bf16 RNE against torch's), the GEMM's
  chains against a scalar loop, the E4M3 blocks, a table row's shapes; ``tkworker.load_toktab``'s cache.
"""

import json
import random
import struct
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from atom.models.deepseek_v41.mono.hi.plan import decode_launches as DL
from atom.models.deepseek_v41.mono.hi.tk import (
    codeobj,
    flybuild,
    host,
    toktab,
)

# ---------------------------------------------------------------------------------------------------------------
# flybuild


def test_one_recipe_per_plan_kernel_and_admit():
    plan = DL.load()
    syms = [r.symbol for r in flybuild.RECIPES]
    assert len(set(syms)) == len(syms) and len(
        {r.key for r in flybuild.RECIPES}
    ) == len(syms)
    assert {r.symbol for r in flybuild.STEP_RECIPES} == set(plan["kernels"])
    assert flybuild.BY_KEY["admit"].symbol == host.ADMIT_SYMBOL
    kdir = ROOT / "atom/models/deepseek_v41/mono/hi/kernels/gfx950"
    for r in flybuild.RECIPES:
        src = (kdir / f"{r.module}.py").read_text()
        assert f"def {r.builder}(" in src, r
        assert r.preload in (0, 7)
        assert all(k.startswith("TK_") for k, _ in r.env)


def test_cache_key_follows_sources_and_switches(monkeypatch, tmp_path):
    files = flybuild.source_files("qkv")
    names = {f.name for f in files}
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
    assert (
        env["TK_QKV_IDX"] == "0"
        and env["ARCH"] == "gfx950"
        and env["FLYDSL_RUNTIME_ENABLE_CACHE"] == "0"
    )
    monkeypatch.setenv("TK_VARIANT", "afm6")
    assert "TK_VARIANT" not in flybuild.build_env(
        r
    )  # the caller's switches never leak into a build


def test_code_table_builds_once_and_maps_tilert_symbols(monkeypatch, tmp_path):
    built = []

    def fake_build(r, stem, timeout):
        built.append(r.key)
        lay = ",".join(f"{8 * i}:8" for i in range(3))
        stem.with_suffix(".json").write_text(
            json.dumps({"function": f"fn_{r.key}", "layout": lay})
        )
        stem.with_suffix(".hsaco").write_bytes(r.key.encode())

    monkeypatch.setattr(flybuild, "_build_into", fake_build)
    monkeypatch.setattr(
        codeobj,
        "kernels",
        lambda elf: {f"fn_{elf.decode()}": [(0, 8), (8, 8), (16, 8)]},
    )
    rs = tuple(flybuild.RECIPES[:3]) + (flybuild.BY_KEY["admit"],)
    t = flybuild.code_table(tmp_path, rs, jobs=2)
    assert sorted(built) == sorted(r.key for r in rs)
    assert set(t.entries) == {r.symbol for r in rs}
    e = t.entries[host.ADMIT_SYMBOL]
    assert e.function == "fn_admit" and e.layout == ((0, 8), (8, 8), (16, 8))
    flybuild.code_table(tmp_path, rs)
    assert len(built) == len(rs)  # cached
    tsv = tmp_path / "t.tsv"
    flybuild.write_tsv(t, tsv)
    assert codeobj.CodeTable().swap(tsv).entries == t.entries
    res = flybuild.compare_tsv(t, tsv)
    assert res["lines"] == res["symbol"] == res["function"] == res["layout"] == len(rs)
    # a layout that is not the module's LAYOUT is refused
    monkeypatch.setattr(
        codeobj, "kernels", lambda elf: {f"fn_{elf.decode()}": [(0, 8)]}
    )
    with pytest.raises(RuntimeError, match="layout"):
        flybuild.code_table(tmp_path, rs)


# ---------------------------------------------------------------------------------------------------------------
# admit_kernel's model


def _admit_mem(n=40, nplain=50):
    p = 0x5A5A5A5A
    return {"plain_acc": [p] * nplain, "ar_acc": [p] * n, "ar_num": [p] * n, "ctl": [p] * 16,
            "blk_tok": [p] * 8, "blk_pos": [p] * 8, "vtok": [p] * 8, "vpos": [p] * 8, "samp_params": [7] * 4,
            "X": bytearray(b"\xa5" * 8 * host.X_ROW)}  # fmt: skip


def _admit_blob(first, pos, mtp, params, cap, n=40, nplain=50):
    addr = {k: 0x1000 * (i + 1) for i, k in enumerate(
        ("temp:plain_acc", "temp:ar_acc", "temp:ar_num", "temp:ctl", "temp:blk_tok", "temp:blk_pos", "temp:vtok",
         "temp:vpos", "param:embed", "temp:X", "temp:samp_params"))}  # fmt: skip
    b = bytearray(host.admit_args(addr, first, pos, mtp, params))
    for off, cnt in (
        (0x08, nplain),
        (0x18, n),
        (0x28, n),
    ):  # small buffers: the counts cut down
        struct.pack_into("<i", b, off, cnt)
    struct.pack_into("<i", b, 0x80, cap)
    return bytes(b)


def _row(tok):
    return bytes([tok % 251]) * host.ROW_BYTES


def test_admit_reference_request_and_reset():
    first, pos, cap = 2581, 30, 33
    params = [591402, 0x3F8000003F800000, 0, 2]
    mem = _admit_mem()
    host.admit_reference(_admit_blob(first, pos, True, params, cap), mem, _row)
    assert mem["plain_acc"] == [0] * 50
    assert mem["ar_acc"][:3] == [1, first, 0] and set(mem["ar_acc"][2:]) == {0}
    assert mem["ar_num"][:3] == [1, 1, 0] and set(mem["ar_num"][2:]) == {0}
    assert mem["ctl"] == [pos - 1, first, 0, cap, 0, 1] + [0] * 10
    assert mem["blk_tok"] == [first] + [host.PAD_TOKEN] * 7
    assert mem["blk_pos"] == [30, 31, 32, -1, -1, -1, -1, -1]  # pos + t < cap, t < 5
    assert (
        mem["vtok"][:6] == [first] + [host.PAD_TOKEN] * 5
        and mem["vtok"][6:] == [0x5A5A5A5A] * 2
    )
    assert mem["vpos"][:6] == [30, 31, 32, -1, -1, -1]
    assert mem["samp_params"] == params
    X = mem["X"]
    assert X[: host.ROW_BYTES] == _row(first) and X[
        3 * host.ROW_BYTES : host.X_ROW
    ] == _row(first)
    assert X[host.X_ROW : host.X_ROW + host.ROW_BYTES] == _row(host.PAD_TOKEN)
    # a reset (_reset_streams): the logs and ctl only, cap kept; samp_params untouched without params
    mem = _admit_mem()
    host.admit_reference(_admit_blob(0, 0, False, None, cap), mem, _row)
    assert mem["ar_acc"] == [0] * 40 and mem["ar_num"] == [0] * 40
    assert mem["ctl"] == [0, 0, 0, cap] + [0] * 12 and mem["samp_params"] == [7] * 4
    assert mem["blk_tok"] == [0x5A5A5A5A] * 8 and mem["X"] == bytearray(
        b"\xa5" * 8 * host.X_ROW
    )
    # a first token past the vocabulary: a zero row
    mem = _admit_mem()
    host.admit_reference(_admit_blob(129285, 3, True, None, cap), mem, _row)
    assert mem["X"][: host.X_ROW] == bytes(host.X_ROW)


def test_admit_kernel_abi():
    src = (
        ROOT / "atom/models/deepseek_v41/mono/hi/kernels/gfx950/admit.py"
    ).read_text()
    assert "KARG_WORDS = 22" in src and 22 * 8 == host.ADMIT_BYTES
    assert (
        "GRID, THREADS = 64, 256" in src
        and host.ADMIT_GRID == (64, 1, 1)
        and host.ADMIT_BLOCK == (256, 1, 1)
    )


# ---------------------------------------------------------------------------------------------------------------
# toktab


def _f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _bits(f: float) -> int:
    return struct.unpack("<I", struct.pack("<f", f))[0]


def _rn32(q: Fraction) -> float:
    """An exact rational rounded to float32, nearest even (normal range)."""
    f = struct.unpack("<f", struct.pack("<f", float(q)))[0]
    cands = [f] + [
        struct.unpack("<f", struct.pack("<I", (_bits(f) + d) & 0xFFFFFFFF))[0]
        for d in (-1, 1)
    ]
    return min(cands, key=lambda v: (abs(Fraction(v) - q), _bits(v) & 1))


def test_fma32_is_the_fused_result():
    rnd = random.Random(1)
    a, b, c = [], [], []
    for _ in range(3000):
        a.append(_f32(rnd.getrandbits(32) & 0x3FFFFFFF | 0x30000000))
        b.append(
            _f32(rnd.getrandbits(32) & 0x3FFFFFFF | 0x30000000) * rnd.choice((1, -1))
        )
        c.append(-a[-1] * b[-1] * (1 + rnd.choice((0, 2**-24, -(2**-25), 2**-30))) if rnd.random() < 0.5
                 else _f32(rnd.getrandbits(32) & 0x3FFFFFFF | 0x30000000))  # fmt: skip
    ta, tb, tc = (torch.tensor(v, dtype=torch.float32) for v in (a, b, c))
    got = toktab.fma32(ta, tb, tc).tolist()
    for x, y, z, g in zip(ta.tolist(), tb.tolist(), tc.tolist(), got):
        assert g == _rn32(Fraction(x) * Fraction(y) + Fraction(z)), (x, y, z, g)


def test_bf16_rounding_is_rne():
    x = torch.randn(100000) * 10.0 ** torch.randint(-20, 20, (100000,)).float()
    x = torch.cat([x, torch.tensor([1.0 + 2**-8, 1.0 + 3 * 2**-8, -(1.0 + 2**-8)])])
    assert torch.equal(toktab.bf16_round(x), x.to(torch.bfloat16).float())
    assert torch.equal(toktab.as_bf16(toktab.bf16_bits(x)), x.to(torch.bfloat16))


def test_tree_and_sequential_sums():
    v = torch.randn(3, 64)
    exp = v.clone()
    while exp.shape[-1] > 1:
        exp = exp[:, 0::2] + exp[:, 1::2]
    assert torch.equal(toktab.tree64(v), exp[:, 0])
    s = v[:, 0]
    for k in range(1, 64):
        s = s + v[:, k]
    assert torch.equal(toktab.seq_sum(v), s)


def _e4m3_values(n, gen):
    return (
        torch.randint(0, 256, n, generator=gen, dtype=torch.uint8)
        .view(torch.float8_e4m3fn)
        .float()
        .nan_to_num(0.0)
    )


def test_gemm_is_the_waves_fma_chains():
    gen = torch.Generator().manual_seed(0)
    t, n, k, waves = 3, 64, 256, 2
    x, w = _e4m3_values((t, k), gen), _e4m3_values((n, k), gen)
    xs = torch.randint(110, 130, (t, k // 32), generator=gen)
    ws = torch.randint(110, 130, (n // 32, k // 32), generator=gen, dtype=torch.uint8)
    got = toktab.gemm(x, xs, w, ws, waves)
    per = k // 32 // waves
    for i in range(t):
        for j in range(n):
            tot = None
            for wv in range(waves):
                acc = 0.0
                for b in range(wv * per, (wv + 1) * per):
                    exact = sum(
                        Fraction(x[i, b * 32 + q].item())
                        * Fraction(w[j, b * 32 + q].item())
                        for q in range(32)
                    )
                    m = (
                        _rn32(exact) if exact else 0.0
                    )  # the MFMA's block sum, rounded once
                    c = Fraction(2) ** (int(xs[i, b]) - 127) * Fraction(2) ** (
                        int(ws[j // 32, b]) - 127
                    )
                    v = Fraction(m) * c + Fraction(acc)
                    acc = _rn32(v) if v else 0.0
                tot = (
                    acc
                    if tot is None
                    else (_rn32(Fraction(tot) + Fraction(acc)) if tot + acc else 0.0)
                )
            assert got[i, j].item() == tot, (i, j)


def test_mhc_pre_blocks_and_rows():
    gen = torch.Generator().manual_seed(2)
    x = (torch.randn(4, toktab.D, generator=gen) * 0.05).to(torch.bfloat16).float()
    g = (1 + 0.1 * torch.randn(toktab.D, generator=gen)).to(torch.bfloat16).float()
    q, e = toktab.mhc_pre(x, g)
    assert q.shape == x.shape and e.shape == (4, 160)
    blocks = q.view(4, 160, 32).abs().amax(-1)
    assert bool((blocks <= 448).all()) and bool(
        (blocks >= 224).all()
    )  # amax / 2^(e - 127) in (224, 448]
    r = torch.rsqrt((x.double() ** 2).mean(-1) + 1e-20)
    y = (g * x * r[:, None].float()).view(4, 160, 32)
    deq = q.view(4, 160, 32) * torch.exp2(e.double() - 127)[..., None].float()
    assert float(((deq - y).abs() / (y.abs().amax(-1, keepdim=True))).max()) < 2**-3


def test_rows_shapes():
    gen = torch.Generator().manual_seed(3)

    def fp8(*s):
        return _e4m3_values(s, gen) * 0.01

    w = {"fn": torch.randn(24, 4, toktab.D, generator=gen) * 1e-3, "base": torch.randn(24, generator=gen),
         "scale": torch.rand(3, generator=gen), "attn_gamma": torch.ones(toktab.D), "qa_w": fp8(1280, toktab.D),
         "qa_s": torch.full((40, 160), 120, dtype=torch.uint8), "q_norm": torch.ones(1280),
         "kv_w": fp8(512, toktab.D), "kv_s": torch.full((16, 160), 120, dtype=torch.uint8),
         "kv_norm": torch.ones(512), "qb_w": fp8(4096, 1280), "qb_s": torch.full((128, 40), 120, dtype=torch.uint8)}  # fmt: skip
    x = (torch.randn(2, toktab.D, generator=gen) * 0.02).to(torch.bfloat16).float()
    sh = toktab.shared_rows(w, x)
    assert {k: (tuple(v.shape), v.dtype) for k, v in sh.items()} == {
        "latent": ((2, 1280), torch.uint8), "latent_e": ((2, 40), torch.uint8),
        "kv": ((2, 512), torch.bfloat16), "hc": ((2, 24), torch.float32)}  # fmt: skip
    hc = sh["hc"]
    assert bool(
        (hc[:, :4] > 0).all() and (hc[:, :4] < 1.01).all() and (hc[:, 4:8] < 2).all()
    )
    comb = hc[:, 8:].view(2, 4, 4)
    assert torch.allclose(
        comb.sum(-2), torch.ones(2, 4), atol=1e-3
    )  # Sinkhorn: columns sum to 1
    q = toktab.q_rows(w, sh["latent"], sh["latent_e"])
    assert (
        q.shape == (2, 4096)
        and q.dtype == torch.bfloat16
        and bool(torch.isfinite(q.float()).all())
    )


# ---------------------------------------------------------------------------------------------------------------
# tkworker.load_toktab: the cache


def test_load_toktab_cache_and_builds(monkeypatch, tmp_path):
    from atom.models.deepseek_v41.mono.hi.pd import tkworker as TW
    from atom.models.deepseek_v41.mono.hi.tk import toktab_device

    got = {}
    bufs = {
        f"toktab:{sc}.{n}": torch.zeros(2)
        for sc in ("layers.0", "mtp.0")
        for n in ("q", "kv", "hc")
    }
    tk = SimpleNamespace(
        device="cpu",
        params={
            "param:embed": torch.zeros(2),
            **{
                f"param:{sc}.{s}": torch.zeros(1)
                for sc in toktab_device.SCOPES
                for s in toktab_device.SLOTS
            },
        },
        buffers=SimpleNamespace(
            tensors=bufs,
            toktab=lambda scope, q, kv, hc: got.__setitem__(scope, (q, kv, hc)),
        ),
    )
    with pytest.raises(ValueError, match="build"):
        TW.load_toktab(tk, 0, build="tpu")
    # TileRT's list format is read as it is
    tabs = [torch.full((2,), float(i)) for i in range(6)]
    torch.save(tabs, tmp_path / "rank0.pt")
    TW.load_toktab(tk, 0, str(tmp_path))
    assert got["layers.0"][0][0] == 0 and got["mtp.0"][2][0] == 5
    # the GPU build (default): into the toktab buffers, then saved; a cache of another build is rebuilt
    built = []

    class FakeBuilder:
        def __init__(self, table, device):
            built.append(("init", table))

        def build(self, p, embed, out=None, log=None):
            assert set(p) == set(toktab_device.SLOTS)
            built.append("scope")
            for k, t in enumerate(out):
                t.fill_(10 + k)
            return out

    monkeypatch.setattr(toktab_device, "Builder", FakeBuilder)
    d = tmp_path / "gpu"
    TW.load_toktab(tk, 1, str(d), table="T")
    assert (
        built == [("init", "T"), "scope", "scope"]
        and float(bufs["toktab:mtp.0.hc"][0]) == 12
    )
    saved = torch.load(d / "rank1.pt")
    assert saved["build"] == "gpu" and float(saved["layers.0"][1][0]) == 11
    built.clear()
    TW.load_toktab(tk, 1, str(d), table="T")  # the cache: no build
    assert built == [] and float(got["layers.0"][1][0]) == 11
    calls = []
    monkeypatch.setattr(
        toktab,
        "build_rank_tables",
        lambda ckpt, rank, group=None, log=None, hw=None: calls.append((ckpt, hw.name))
        or tabs,
    )
    with pytest.raises(ValueError, match="checkpoint"):
        TW.load_toktab(tk, 1, str(d), build="cpu")  # a gpu cache is not a cpu one
    TW.load_toktab(tk, 1, str(d), ckpt="/ckpt", build="cpu")
    assert calls == [("/ckpt", "cpu")] and torch.load(d / "rank1.pt")["build"] == "cpu"
    TW.load_toktab(tk, 1, str(d), table="T")  # a cpu cache under the gpu build: rebuilt
    assert built == [("init", "T"), "scope", "scope"]


def test_device_build_kernargs_name_the_plans_buffers():
    """The device build's mhc / qkv kernargs put its buffers where the decode plan's launches of the same kernels put
    theirs (``plan.decode_launches``: the step's ``temp:`` / ``param:`` names)."""
    from atom.models.deepseek_v41.mono.hi.tk import toktab_device as TD

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
        setattr(
            b,
            n,
            SimpleNamespace(data_ptr=lambda k=k: (k + 1) << 20, stride=lambda d: 4),
        )
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
    assert kvx == (
        b.ws_q.data_ptr(),
        addr["param:{S+0}.window_kv.kv_norm"],
        b.kv.data_ptr(),
    )

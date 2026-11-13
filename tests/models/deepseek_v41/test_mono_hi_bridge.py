# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono hi hook's cache bridge and round kernels (``atom/models/deepseek_v41/mono``) in the Triton
interpreter, no GPU.

- [2] / [3] / [4]: each bridge body as a kernel of its own (``mono_hi_bridge_ref``) against a torch model of ATOM's
  pool: the rings' import / export, the request words of the in-place programs, the mHC hand-off, the DSpark tap;
- [2k] / [2kr]: ATOM's fp8 pool (``KV8``): its MXFP8 rows decoded and encoded as ATOM's reader / writer do
  (``dequantize_rows`` / ``fp8_groups``, torch models of them here: the interpreter's fp8 conversion rounds ties away
  and drops subnormals, which is why the bridge encodes in integer arithmetic), and the rings' import / export over it;
- [5]: the fused begin-step / end-step launches == those separate launches (and the Engram segments; on the fp8
  pool);
- [8]: the incremental ring import == the full import over a scripted run;
- [10]: the bridge at B requests a step and on the batch bands' slots, the block draft's begin kernel;
- [ef] / [vf] / [pv] / [gf] / [2ka]: the Engram fold, the verdict reference, the in-graph post-verify (the forward's
  head output), the glue folds and the fp8 rows' export / import against ATOM's own launches (these need AITER and a
  GPU: they skip without them).

The interpreter's sections need Triton's interpreter from the start (minutes):

    TRITON_INTERPRET=1 python -m pytest -q tests/models/deepseek_v41/test_mono_hi_bridge.py

and skip otherwise; the AITER sections run on a GPU (without ``TRITON_INTERPRET``).
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

triton = pytest.importorskip("triton")
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
interpreted = pytest.mark.skipif(
    not INTERPRET,
    reason="TRITON_INTERPRET=1 from the start (the Triton interpreter, minutes)",
)

ROOT = Path(__file__).resolve().parents[3]
MONO = ROOT / "atom/models/deepseek_v41/mono"
HELPERS = (
    "_ring_in",
    "_ring_out",
    "_kv8_load",
    "_kv8_store",
    "_req_words",
    "_dead_words",
    "_eng_fold",
    "_eng_hist",
    "_eng_row",
    "_expand_row",
    "_collapse_chunk",
    "_planar",
    "_tap",
)


def _load(path: Path, name: str, namespace=None, interpret=True):
    """``path`` loaded as module ``name``, its Triton kernels in the interpreter (``interpret``) or compiled
    (``namespace``: names set in the module before it runs)."""
    old = os.environ.get("TRITON_INTERPRET")
    os.environ["TRITON_INTERPRET"] = "1" if interpret else "0"
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update(namespace or {})
        sys.modules[name] = module  # (a dataclass resolves its module by name)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            os.environ.pop("TRITON_INTERPRET")
        else:
            os.environ["TRITON_INTERPRET"] = old
    return module


_ML: dict = {}
_MODE = {"interpret": True}


def _layout(interpret: bool):
    """``hi_bridge`` with the reference launches beside it, as one namespace (interpreted, or compiled for a
    GPU)."""
    if interpret not in _ML:
        tag = "interp" if interpret else "gpu"
        layout = _load(MONO / "hi_bridge.py", f"_hi_bridge_{tag}", interpret=interpret)
        ref = _load(
            Path(__file__).with_name("mono_hi_bridge_ref.py"),
            f"_hi_bridge_ref_{tag}",
            {k: getattr(layout, k) for k in HELPERS},
            interpret=interpret,
        )
        kernels = {k: v for k, v in vars(ref).items() if k.endswith("_kernel")}
        _ML[interpret] = SimpleNamespace(**{**vars(layout), **kernels})
    return _ML[interpret]


class _Ml:
    """``ml`` of the checks below: the layout in the current mode, loaded on first use."""

    def __getattr__(self, name):
        return getattr(_layout(_MODE["interpret"]), name)


ml = _Ml()
_FAILS: list = []


def _interpreter_bool_masks():
    """Triton's interpreter (3.8) builds the mask of a scalar condition and a vector one (``ok & (d < DIM)``) as
    int32, which its load / store refuse ("mask has unsupported dtype"); a compiled kernel takes it as is. Its loads
    and stores here take any mask as bool, so the bridge's bodies run unchanged."""
    from triton.runtime import interpreter

    native = interpreter._interpreter
    if getattr(native, "_bool_masks", False):
        return
    load, store = native.load, native.store

    class Native:
        _bool_masks = True

        def __getattr__(self, name):
            return getattr(native, name)

        @staticmethod
        def load(ptrs, mask, other, dtype):
            return load(ptrs, mask.astype(bool, copy=False), other, dtype)

        @staticmethod
        def store(ptrs, value, mask):
            return store(ptrs, value, mask.astype(bool, copy=False))

    interpreter._interpreter = Native()


if INTERPRET:
    _interpreter_bool_masks()


# ---------------------------------------------------------------- ATOM's fp8 pool rows (torch models)
#
# A row: 512 E4M3 bytes, then 16 E8M0 group-32 codes (``V41PoolGeometry.row_bytes``: 528). ATOM's reader
# (``fp8_attention.dequantize_rows``) when it imports here, else the same torch body; its writer (``fp8_groups``,
# Triton) as torch: f32 arithmetic and torch's fp8 conversion, which rounds to nearest even.


def _atom_dequantize_rows():
    try:
        from atom.model_ops.attentions.deepseek_v41.fp8_attention import (
            dequantize_rows,
        )

        return dequantize_rows
    except Exception:  # noqa: BLE001 - its module needs AITER's attention (and what it imports)

        def dequantize_rows(values, scales):
            groups = values.float().unflatten(-1, (-1, 32))
            steps = torch.exp2(scales.view(torch.uint8).float() - 127)
            return (groups * steps[..., None]).flatten(-2)

        return dequantize_rows


def kv8_decode(rows: torch.Tensor) -> torch.Tensor:
    """ATOM's fp8 pool rows (uint8 [n, 528]) as ATOM's reader takes them: f32 [n, 512]."""
    values = rows[:, :512].contiguous().view(torch.float8_e4m3fn)
    scales = rows[:, 512:].contiguous().view(torch.float8_e8m0fnu)
    return _atom_dequantize_rows()(values, scales)


def kv8_encode(x: torch.Tensor) -> torch.Tensor:
    """``x`` [n, 512] as ATOM's writer stores it (``fp8_groups``): uint8 [n, 528]. A group's amax floored at 1e-10,
    code = the smallest power of two >= amax x fp32(1 / 448) (``ceil_pow2_code``), the values times the scale's exact
    reciprocal (``pow2_reciprocal``), clamped to +-448, E4M3 rounded to nearest even."""
    n = x.shape[0]
    g = x.float().reshape(n, 16, 32)
    amax = torch.maximum(g.abs().amax(-1), torch.tensor(1e-10, dtype=torch.float32, device=x.device))
    t = (amax * torch.tensor(1 / 448, dtype=torch.float32, device=x.device)).view(torch.int32)
    code = (t >> 23) + ((t & 0x7FFFFF) != 0).int()
    rcp = ((254 - code).clamp_min(0) << 23).view(torch.float32)
    q = (g * rcp[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return torch.cat([q.view(torch.uint8).reshape(n, 512), code.to(torch.uint8)], -1)


def kv8_rows(n: int, seed: int = 0) -> torch.Tensor:
    """``_kv8_rows`` built on the host whatever the default device (a GPU run sets cuda)."""
    with torch.device("cpu"):
        return _kv8_rows(n, seed)


def _kv8_rows(n: int, seed: int = 0) -> torch.Tensor:
    """n bf16 rows [n, 512] that reach every case of the encoding: groups from 2^-40 to 2^20 in scale, an all-zero
    group (the amax floor), +-0, exact powers of two at the clamp, values at the E4M3 midpoints (normal and
    subnormal: the rounding's ties), values below the smallest subnormal."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 16, 32, generator=g)
    x *= torch.exp2(torch.randint(-40, 21, (n, 16, 1), generator=g).float())
    x[0, 0] = 0.0
    x[0, 1, ::2] = -0.0
    # amax 448 x 2^k (code 127 + k exactly), then the midpoints between E4M3 neighbours (scaled by 2^k): normal
    # (2.125: 2 | 2.25, 1.0625 x 2^5, 3.5 x 2^-7 ..) and subnormal (k units of 2^-9 plus half a unit), and 2^-11
    ties = torch.tensor([2.125, 2.375, 1.1875, 1.0625 * 32, 3.5 * 2**-7, 0.5 * 2**-9, 1.5 * 2**-9, 2.5 * 2**-9,
                         7.5 * 2**-9, 2**-11, 15.5 * 2**-7, 240.0, 232.0, 0.0])  # fmt: skip
    for r in range(min(n, 8)):
        k = r * 3 - 9
        row = x[r, 2 + r % 14]
        row[:] = 0.0
        row[0] = 448.0 * 2.0**k
        row[1 : 1 + len(ties)] = ties * 2.0**k
        row[1 + len(ties) : 1 + 2 * len(ties)] = -ties * 2.0**k
    return x.reshape(n, 512).bfloat16()


def report(name, ok, extra=""):
    if not ok:
        _FAILS.append(f"{name} {extra}")


def _run(check, *args):
    """``check``, its failures collected."""
    _FAILS.clear()
    check(*args)
    assert not _FAILS, _FAILS


def ring_world(seed=0, kv8=False):
    torch.manual_seed(seed)
    w = SimpleNamespace(
        dim=512,
        ring_slots=134,
        slot_rows=40_000,
        ring_start0=11,
        num_slots=3,
        layers=40,
    )
    w.pool = torch.randn(
        w.ring_start0 + w.num_slots * w.slot_rows + 8, w.dim
    ).bfloat16()
    if kv8:  # ATOM's fp8 pool: its writer's rows of the same keys
        w.pool = kv8_encode(w.pool)
    return w


def check_rings(kv8=False):
    """[2] (``kv8`` [2kr]: on ATOM's fp8 pool, its rows decoded on import and the step's rows encoded on export)."""
    kv = {"KV8": 1} if kv8 else {}
    tag = "[2kr] fp8 rings" if kv8 else "[2] rings"
    mf = [0, 2, 3, 20, 39]
    ids = torch.tensor(mf, dtype=torch.int32)
    for ctx, rows, slot, pad in ((5, 6, 1, False), (127, 6, 2, False), (1000, 6, 2, False), (1131, 6, 1, False),
                                 (300, 6, 1, True)):  # fmt: skip
        w = ring_world(ctx, kv8)
        positions = torch.arange(ctx, ctx + rows, dtype=torch.int64)
        batch = torch.full((rows,), -1 if pad else 0, dtype=torch.int32)
        slots = torch.tensor([slot, 0, 0], dtype=torch.int32)
        rings = torch.full((len(mf), 1, 128, w.dim), 7.0).bfloat16()
        pos, bounds = torch.zeros(1, dtype=torch.int32), torch.zeros(
            3, 8, dtype=torch.int32
        )
        geo = (w.ring_start0, w.slot_rows, w.ring_slots, w.num_slots, 1 << 20)
        before = w.pool.clone()
        ml._import_rings_kernel[(len(mf), 128)](w.pool, positions, batch, slots, rings, ids, pos, bounds, *geo, rows,
                                                WIN=128, DIM=w.dim, RB=8, **kv)  # fmt: skip
        ok = int(pos) == ctx
        for m, layer in enumerate(mf):
            want = torch.zeros(128, w.dim).bfloat16()
            if not pad:
                for s in range(128):
                    q = ctx - 128 + (s - ctx % 128 + 128) % 128
                    if q >= 0:
                        i = (
                            w.ring_start0
                            + layer * w.ring_slots
                            + slot * w.slot_rows
                            + q % w.ring_slots
                        )
                        want[s] = kv8_decode(before[i : i + 1])[0] if kv8 else before[i]
            ok &= torch.equal(rings[m, 0], want)
        j = torch.arange(rows)
        ok &= torch.equal(bounds[0, :rows], ((ctx + j + 1) // 2).int()) and torch.equal(
            bounds[1, :rows], (ctx + j + 1).int()
        )
        ok &= torch.equal(bounds[2, :rows], ((ctx + j + 1 + 7) // 8).int())
        new = torch.randn(len(mf), rows, w.dim).bfloat16()
        for m in range(len(mf)):
            for r in range(rows):
                rings[m, 0, (ctx + r) % 128] = new[m, r]
        ml._export_rings_kernel[(len(mf), rows)](
            w.pool, positions, batch, slots, rings, ids, *geo, WIN=128, DIM=w.dim, **kv
        )
        want_pool = before.clone()
        if not pad:
            for m, layer in enumerate(mf):
                for r in range(rows):
                    want_pool[
                        w.ring_start0
                        + layer * w.ring_slots
                        + slot * w.slot_rows
                        + (ctx + r) % w.ring_slots
                    ] = (kv8_encode(new[m, r][None])[0] if kv8 else new[m, r])
        ok &= torch.equal(w.pool, want_pool)
        report(f"{tag} ctx {ctx} rows {rows} slot {slot} pad {pad}", ok)


def check_kv8_codec():
    """[2k] ATOM's fp8 pool rows (``_kv8_store`` / ``_kv8_load``, the bridge's export and import of a row): the bytes
    ATOM's writer stores for a bf16 row (``kv8_encode``: every case ``kv8_rows`` reaches), decoded as ATOM's reader
    does, exactly in bf16; and ATOM's rows of any codes it writes decoded alike."""
    n, dim = 64, 512
    x = kv8_rows(n)
    got = torch.zeros(n, ml.KV8_ROW_BYTES, dtype=torch.uint8)
    ml._kv8_store_kernel[(n,)](x, got, DIM=dim)
    want = kv8_encode(x)
    bad = (got != want).any(-1).nonzero().flatten().tolist()
    report("[2k] export: ATOM's writer's bytes", not bad,
           f"(rows {bad[:8]}; values {int((got[:, :dim] != want[:, :dim]).sum())} differ, codes "
           f"{int((got[:, dim:] != want[:, dim:]).sum())})")  # fmt: skip
    back = torch.zeros(n, dim).bfloat16()
    ml._kv8_load_kernel[(n,)](want, back, DIM=dim)
    ref = kv8_decode(want)
    report("[2k] import: ATOM's reader's values, exact in bf16",
           torch.equal(back.float(), ref) and torch.equal(ref.bfloat16().float(), ref))  # fmt: skip
    # ATOM's rows of random value bytes (no NaN: 0x7f / 0xff are its clamp's) and codes 85 .. 200 (the amax floor's
    # code is 85)
    g = torch.Generator().manual_seed(1)
    rows = torch.randint(0, 256, (n, ml.KV8_ROW_BYTES), generator=g, dtype=torch.int32).to(torch.uint8)
    rows[:, :dim][(rows[:, :dim] & 0x7F) == 0x7F] = 0x7E
    rows[:, dim:] = torch.randint(85, 201, (n, dim // 32), generator=g, dtype=torch.int32).to(torch.uint8)
    ml._kv8_load_kernel[(n,)](rows, back, DIM=dim)
    report("[2k] import: random ATOM rows", torch.equal(back.float(), kv8_decode(rows)))
    # a row the hi path re-exports after importing it decodes to the same values (its bytes may differ: a group whose
    # largest value rounded down to 224 x scale re-encodes at half the scale, every value doubled)
    again = torch.zeros_like(got)
    ml._kv8_store_kernel[(n,)](back, again, DIM=dim)
    report("[2k] export after import: the same values", torch.equal(kv8_decode(again), kv8_decode(rows)))


def check_req_words():
    """[3] The request words of the programs that use ATOM's storage in place (``_req_words`` / ``_dead_words``): a
    live request's batch id, slot and liveness and its rows' (its padding rows dead), a pad request (batch id -1) and
    a request whose slot is out of range dead with all their rows, the batch slots past the step's requests and the
    rows past its rows dead (none of their stores may reach ATOM's storage), their first positions 0; the page
    geometry's words untouched."""
    num_slots, bm, rq, nb = 4, 6, 6, 4
    rows = nb * rq
    batch = torch.tensor(
        [0] * 6 + [1] * 4 + [-1] * 2 + [-1] * 6 + [3] * 6, dtype=torch.int32
    )
    slot_map = torch.tensor([2, 0, 1, 9], dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int64) + 100
    req = torch.full((ml.REQ_WORDS,), 77, dtype=torch.int32)
    pos = torch.full((bm,), 77, dtype=torch.int32)
    ml._req_words_kernel[(nb,)](positions, batch, slot_map, num_slots, 1 << 20, req, pos, rows, NB=nb, BM=bm,
                                RQP=8)  # fmt: skip
    want = [77] * 4
    want += [
        0,
        1,
        -1,
        3,
        -1,
        -1,
        77,
        77,
    ]  # batch ids (slots 6, 7: past the step's batch slots)
    want += [2, 0, 0, 0, 0, 0, 77, 77]  # slots
    want += [1, 1, 0, 0, 0, 0, 77, 77]  # liveness
    want += [1] * 10 + [0] * (ml.MAX_ROWS - 10)  # rows
    report(
        "[3] request words: live / padding rows / pad request / bad slot / dead slots and rows",
        req.tolist() == want and pos.tolist() == [77] * nb + [0] * (bm - nb),
        f"({req.tolist()}, {pos.tolist()})",
    )


def check_hc():
    torch.manual_seed(3)
    rows, hidden = 6, 2048
    gate = torch.rand(rows, 24)
    pre, post, comb = torch.zeros(rows, 4), torch.zeros(rows, 4), torch.zeros(rows, 16)
    ml._gates_planar_kernel[(rows,)](gate, pre, post, comb, NMIX=24, NHC=4)
    ok = (
        torch.equal(pre, gate[:, :4])
        and torch.equal(post, gate[:, 4:8])
        and torch.equal(comb, gate[:, 8:])
    )
    res, pend = (
        torch.randn(rows, 4, hidden).bfloat16(),
        torch.randn(rows, hidden).bfloat16(),
    )
    h_in, pend_out, gin = (
        torch.zeros_like(res),
        torch.zeros_like(pend),
        torch.zeros(rows, 24),
    )
    ml._atom_to_mf_kernel[(rows, 5)](res, pend, pre, post, comb, h_in, pend_out, gin, HIDDEN=hidden, BLOCK=1024,
                                     NMIX=24, NHC=4)  # fmt: skip
    ok &= (
        torch.equal(h_in, res)
        and torch.equal(pend_out, pend)
        and torch.equal(gin, gate)
    )
    report("[4] gates planar <-> packed, ATOM state -> hi inputs", ok)
    # ATOM's settle (expand_residual's torch body) of the hi runtime's outputs == the hi runtime's fold, comb[i_in][o_out]
    try:
        from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
    except Exception as e:  # noqa: BLE001 - an aiter-free image
        print(
            f"  [4] settle orientation: skipped here ({type(e).__name__}); checked in ATOM's image"
        )
        SinglePassHCState = None
    if SinglePassHCState is not None:
        h = 64
        r_f, y = torch.randn(rows, 4, h).bfloat16(), torch.randn(rows, h).bfloat16()
        st = SinglePassHCState(r_f.view(1, rows, 4, h), pre.view(1, rows, 4), y.view(1, rows, h), post.view(1, rows, 4),
                               comb.view(1, rows, 4, 4))  # fmt: skip
        got = st.settle().residual.view(rows, 4, h)
        want = torch.stack(
            [
                (
                    post[:, o, None] * y.float()
                    + sum(
                        gate[:, 8 + 4 * i + o, None] * r_f[:, i].float()
                        for i in range(4)
                    )
                ).bfloat16()
                for o in range(4)
            ],
            1,
        )
        report(
            "[4] ATOM settle == hi fold (comb[i_in][o_out])",
            torch.equal(got, want),
        )
    r_a = torch.randn(rows, 4, hidden).bfloat16()
    aux = torch.zeros(rows, hidden).bfloat16()
    ml._aux_tap_kernel[(rows,)](r_a, aux, HIDDEN=hidden, BLOCK=1024, NHC=4)
    acc = r_a[:, 0].float()
    for i in range(1, 4):
        acc = acc + r_a[:, i].float()
    rne = torch.equal(aux, (acc * 0.25).bfloat16())
    # Triton's interpreter truncates f32 -> bf16 whatever the rounding mode; the GPU build rounds to nearest even
    trunc = torch.equal(
        aux.view(torch.int16), ((acc * 0.25).view(torch.int32) >> 16).to(torch.int16)
    )
    report(
        "[4] DSpark tap (mean over the bf16 streams, in order, / 4)",
        rne or trunc,
        (
            "(RNE)"
            if rne
            else "(the interpreter's truncating bf16 cast: the sum and scale match; GPU rounds RNE)"
        ),
    )


def check_fused(engram: bool = False, kv8: bool = False):
    """[5] The fused bridge (``_begin_step_kernel`` / ``_end_step_kernel``, the hook's default) writes the same bytes
    as the separate kernels it replaces ([2]-[4] check those against the reference model): one world, the rings, the
    request words, three taps, the gates; random state everywhere so an unwritten or misrouted byte shows; the hi
    side's writes between import and export are the same random values for both. ``engram`` ([5e]): the begin-step's
    Engram segments too, against ``_engram_rows_kernel`` / ``_engram_hist_kernel``: ATOM's staged rows of two layers
    (a strided view, as a wider staging buffer gives) into the kv launch's ``eng<L>.rows``, and (the table lookup)
    the history from ATOM's cursor. ``kv8`` ([5k]): on ATOM's fp8 pool (``KV8``)."""
    torch.manual_seed(5)
    dim, rows, win, hid = 512, 6, 128, 2048
    mf = [0, 2, 3, 20, 39]
    ring_start0, ring_slots, slot_rows, num_slots = 10000, 134, 6000, 3
    base = {
        "pool": torch.randn(ring_start0 + num_slots * slot_rows + 8, dim).bfloat16(),
        "rings": torch.randn(len(mf), 1, win, dim).bfloat16(),
        "pos": torch.zeros(1, dtype=torch.int32), "bounds": torch.zeros(3, 8, dtype=torch.int32),
        "hidden": torch.randn(rows, hid).bfloat16(), "h_in": torch.randn(8, 4, hid).bfloat16(),
        "step": torch.tensor([5], dtype=torch.int32), "gate": torch.rand(rows, 24),
        "pre": torch.zeros(rows, 4), "post": torch.zeros(rows, 4), "comb": torch.zeros(rows, 16),
        "req": torch.randint(-9, 9, (ml.REQ_WORDS,), dtype=torch.int32),
    }  # fmt: skip
    kv = {"KV8": 1} if kv8 else {}
    if kv8:  # ATOM's writer's rows of keys over many scales
        base["pool"] = kv8_encode(base["pool"].float() * torch.exp2(torch.randint(-30, 20, (len(base["pool"]), 1))))
    if (
        engram
    ):  # ATOM's staged rows (views of a wider buffer), the kv launch's row buffers, the cursor, hist
        kin = 6144
        base["staged"] = torch.randn(2, 8, kin + 64).bfloat16()
        base["eng_rows0"], base["eng_rows1"] = (
            torch.randn(8, kin).bfloat16(),
            torch.randn(8, kin).bfloat16(),
        )
        base["cursor"] = torch.randint(-1, 99000, (3, 4), dtype=torch.int64)
        base["hist"] = torch.full((1, 2048), -7, dtype=torch.int32)
    for t in range(3):
        base[f"r_a{t}"] = torch.randn(rows, 4, hid).bfloat16()
        base[f"aux{t}"] = torch.zeros(rows, hid).bfloat16()
    ids = torch.tensor(mf, dtype=torch.int32)
    slot_map = torch.tensor([2, 1, 0], dtype=torch.int32)
    geo = (ring_start0, slot_rows, ring_slots, num_slots, 1 << 20)
    for ctx, b, pad in (
        (37, 1, False),
        (1131, 0, False),
        (300, 0, True),
        (5, 0, False),
        (2, 1, False),
    ):
        positions = torch.arange(ctx, ctx + rows, dtype=torch.int64)
        batch = torch.full((rows,), -1 if pad else b, dtype=torch.int32)
        worlds = [{k: v.clone() for k, v in base.items()} for _ in range(2)]
        g = torch.Generator().manual_seed(ctx)
        writes = {"rings": torch.randn(base["rings"].shape, generator=g)}

        def hi_writes(
            w, writes=writes
        ):  # the layers' writes between the import and the export
            for k, v in writes.items():
                w[k].copy_(v)

        # A: the separate kernels
        w = worlds[0]
        w["step"].add_(1)
        w["h_in"][:rows].copy_(w["hidden"].view(rows, 1, -1).expand(rows, 4, hid))
        ml._import_rings_kernel[(len(mf), win)](w["pool"], positions, batch, slot_map, w["rings"], ids, w["pos"],
                                                w["bounds"], *geo, rows, WIN=win, DIM=dim, RB=8, **kv)  # fmt: skip
        ml._req_words_kernel[(1,)](positions, batch, slot_map, num_slots, 1 << 20, w["req"], w["pos"], rows, NB=1, BM=1,
                                   RQP=8)  # fmt: skip
        hi_writes(w)
        ml._gates_planar_kernel[(rows,)](
            w["gate"], w["pre"], w["post"], w["comb"], NMIX=24, NHC=4
        )
        ml._export_rings_kernel[(len(mf), rows)](w["pool"], positions, batch, slot_map, w["rings"], ids, *geo, WIN=win,
                                                 DIM=dim, **kv)  # fmt: skip
        for t in range(3):
            ml._aux_tap_kernel[(rows,)](
                w[f"r_a{t}"], w[f"aux{t}"], HIDDEN=hid, BLOCK=1024, NHC=4
            )
        if engram:
            for e in range(2):
                src = w["staged"][e, :, :kin]
                ml._engram_rows_kernel[(rows,)](
                    src, src.stride(0), w[f"eng_rows{e}"], KIN=kin, BLOCK=1024
                )
            ml._engram_hist_kernel[(1,)](
                w["cursor"],
                w["cursor"].stride(0),
                slot_map,
                positions,
                w["hist"],
                NGRAM1=3,
            )
        # B: the two fused launches
        w = worlds[1]
        if (
            engram
        ):  # both Engram segments in one launch (the hook uses one or the other)
            s0, s1 = w["staged"][0, :, :kin], w["staged"][1, :, :kin]
            eflat = [s0, s0.stride(0), w["eng_rows0"], s1, s1.stride(0), w["eng_rows1"], w["cursor"],
                     w["cursor"].stride(0), w["hist"]]  # fmt: skip
            ek = {"NE0": rows, "NE1": rows, "KIN": kin, "EHIST": 1}
        else:
            d = w["pos"]
            eflat, ek = [d, 0, d, d, 0, d, d, 0, d], {"NE0": 0, "NE1": 0, "EHIST": 0}
        grid = len(mf) * win + rows + ek["NE0"] + ek["NE1"] + ek["EHIST"]
        ml._begin_step_kernel[(grid,)](w["pool"], positions, batch, slot_map, w["rings"], ids, w["pos"], w["bounds"], *geo,
                                       rows, w["hidden"], w["hidden"].stride(0), w["h_in"], w["step"],
                                       w["req"], *eflat, w["pos"], NMF=len(mf), WIN=win, DIM=dim, RB=8,
                                       HIDDEN=hid, BLOCK=1024, NHC=4, **ek, RQP=8, **kv)  # fmt: skip
        hi_writes(w)
        taps = [w[f"{k}{t}"] for t in range(3) for k in ("r_a", "aux")]
        ml._end_step_kernel[((len(mf) + 3 + 1) * rows,)](
            w["pool"], positions, batch, slot_map, w["rings"], ids, *geo, rows, *taps, w["gate"], w["pre"], w["post"],
            w["comb"], w["pos"], NMF=len(mf), WIN=win, DIM=dim, HIDDEN=hid, BLOCK=1024, NHC=4, NMIX=24, NTAPS=3,
            PLANAR=True, **kv,
        )  # fmt: skip
        a, bw = worlds
        diff = [k for k in base if not torch.equal(a[k], bw[k])]
        changed = [k for k in base if not torch.equal(a[k], base[k])]
        if engram:  # and the copies are the staged rows themselves
            ok_rows = all(
                torch.equal(a[f"eng_rows{e}"][:rows], base["staged"][e, :rows, :kin])
                for e in range(2)
            )
            diff += [] if ok_rows else ["eng_rows (vs the staged rows)"]
        tag = "[5e] fused bridge + Engram segments," if engram else "[5] fused bridge,"
        tag = tag.replace("bridge", "bridge (fp8 pool)").replace("[5", "[5k" if kv8 else "[5")
        report(f"{tag} == separate kernels, ctx {ctx} request {b} pad {pad}", not diff,
               f"({len(base) - len(diff)} / {len(base)} tensors equal; {len(changed)} written"
               + (f"; DIFF {diff}" if diff else "") + ")")  # fmt: skip


def check_engram_hist():
    """[5e] The table lookup's history: ATOM's cursor [slot, 1:] (oldest first, -1 none) -> the hi runtime's hist at the 3
    positions before the step (none before position 0)."""
    cursor = torch.tensor(
        [[0, 7, 8, 9], [0, 17, 18, 19], [0, -1, -1, 5]], dtype=torch.int64
    )
    ok = True
    for slot, p0, want in (
        (1, 100, {97: 17, 98: 18, 99: 19}),
        (0, 2, {0: 8, 1: 9}),
        (2, 3, {0: -1, 1: -1, 2: 5}),
    ):
        hist = torch.full((1, 128), -7, dtype=torch.int32)
        ml._engram_hist_kernel[(1,)](cursor, cursor.stride(0), torch.tensor([slot], dtype=torch.int32),
                                     torch.tensor([p0], dtype=torch.int32), hist, NGRAM1=3)  # fmt: skip
        h = hist[0].cpu()
        ok &= all(int(h[q]) == v for q, v in want.items()) and int(
            (h != -7).sum()
        ) == len(want)
    report("[5e] Engram history from ATOM's cursor -> hist", ok)


def check_incremental(configs=((16, 7), (128, 64), (5, 300)), control: bool = True):
    """[8] The incremental ring import (module docstring, 8): world A runs the fused bridge with the full import, world
    B with the incremental one (``INCR`` / ``TRACK``, ``imp``), on the same scripted run. Between a step's two launches
    both get the same hi writes (every row's ring slot, padding rows included); ATOM's events (a prefill rewrites the
    pool's rings, then the hook's ``invalidate``) hit both. After every begin-step launch the hi-side tensors, and
    after every end-step launch ATOM's, are compared byte for byte."""
    torch.manual_seed(70)
    dim, R, win, hid = 512, 6, 128, 2048
    mf = [0, 2, 3, 20, 39]
    ring_start0, ring_slots, slot_rows, num_slots = 12000, 134, 6000, 3
    ids = torch.tensor(mf, dtype=torch.int32)
    geo = (ring_start0, slot_rows, ring_slots, num_slots, 1 << 20)
    dev = torch.empty(0).device

    def rand(shape, g):
        return torch.randn(shape, generator=g, device="cpu").to(dev)

    def atom_writes(w, g):  # ATOM writes the caches itself (a prefill)
        w["pool"].copy_(rand(w["pool"].shape, g).bfloat16())

    base = {
        "pool": torch.zeros(ring_start0 + num_slots * slot_rows + 8, dim).bfloat16(),
        "rings": torch.zeros(len(mf), 1, win, dim).bfloat16(),
        "pos": torch.zeros(1, dtype=torch.int32), "bounds": torch.zeros(3, 8, dtype=torch.int32),
        "hidden": torch.randn(R, hid).bfloat16(), "h_in": torch.zeros(8, 4, hid).bfloat16(),
        "step": torch.tensor([5], dtype=torch.int32), "gate": torch.rand(R, 24),
        "pre": torch.zeros(R, 4), "post": torch.zeros(R, 4), "comb": torch.zeros(R, 16),
        "imp": torch.full((ml.IMP_WORDS,), -1, dtype=torch.int32),
        "req": torch.zeros(ml.REQ_WORDS, dtype=torch.int32),
    }  # fmt: skip
    for t in range(3):
        base[f"r_a{t}"], base[f"aux{t}"] = (
            torch.randn(R, 4, hid).bfloat16(),
            torch.zeros(R, hid).bfloat16(),
        )
    g = torch.Generator().manual_seed(7)
    atom_writes(base, g)
    mono = [
        k
        for k in base
        if k.startswith(("rings", "pos", "bounds", "h_in", "step", "req"))
    ]
    atom = [k for k in base if k.startswith(("pool", "aux"))]

    def differ(names, worlds):
        return [k for k in names if not torch.equal(worlds[0][k].contiguous().view(torch.uint8),
                                                    worlds[1][k].contiguous().view(torch.uint8))]  # fmt: skip

    def served(worlds, ctx, live, req, slot, rp):
        """One served step at ctx (``live`` real rows; 0: a pad step) in both worlds -> (hi-side names that differ
        after the begin-step launch, ATOM-side names that differ after the end-step launch, B's import was incremental).
        """
        positions = torch.arange(ctx, ctx + R, dtype=torch.int64)
        batch = torch.tensor(
            [0 if j < live else -1 for j in range(R)], dtype=torch.int32
        )
        slot_map = torch.tensor([slot], dtype=torch.int32)
        imp = worlds[1]["imp"].tolist()
        incr = live > 0 and imp[0] == slot and imp[1] <= ctx <= imp[1] + imp[2]
        writes = [("rings", (m, 0, (ctx + j) % win), rand((dim,), g).bfloat16()) for j in range(R)
                  for m in range(len(mf))]  # fmt: skip
        for inc, w in enumerate(worlds):
            d = w["pos"]
            rpl = rp if inc else win
            ml._begin_step_kernel[(len(mf) * rpl + R,)](
                w["pool"], positions, batch, slot_map, w["rings"], ids, w["pos"], w["bounds"], *geo, R,
                w["hidden"], w["hidden"].stride(0), w["h_in"], w["step"], w["req"],
                d, 0, d, d, 0, d, d, 0, d, w["imp"], NMF=len(mf), WIN=win, DIM=dim, RB=8, HIDDEN=hid, BLOCK=1024,
                NHC=4, NE0=0, NE1=0, EHIST=0, INCR=inc, RP=rpl, RQP=8,
            )  # fmt: skip
        d_begin = differ(mono, worlds)
        for w in worlds:
            for k, at, v in writes:
                w[k][at] = v
            taps = [w[f"{k}{t}"] for t in range(3) for k in ("r_a", "aux")]
            ml._end_step_kernel[((len(mf) + 3 + 1) * R,)](
                w["pool"], positions, batch, slot_map, w["rings"], ids, *geo, R, *taps, w["gate"], w["pre"], w["post"],
                w["comb"], w["imp"], NMF=len(mf), WIN=win, DIM=dim, HIDDEN=hid, BLOCK=1024, NHC=4, NMIX=24, NTAPS=3,
                PLANAR=True, TRACK=int(w is worlds[1]),
            )  # fmt: skip
        return d_begin, differ(atom, worlds), incr

    def prefill(worlds, invalidate=True):
        seed = int(torch.randint(0, 2**30, (1,), generator=g, device="cpu"))
        for w in worlds:
            atom_writes(w, torch.Generator().manual_seed(seed))
        if invalidate:
            worlds[1]["imp"].fill_(-1)

    # the script: (event, ...). "steps" = served steps with accepts drawn from 1..live
    script = [
        ("prefill",),
        (
            "steps",
            "A",
            1,
            37,
            30,
            6,
        ),  # request A, slot 1: ~30 steps from ctx 37 (wraps the ring)
        ("steps", "A", 1, None, 4, 3),  # padding rows: 3 live rows of 6
        ("pad",),
        ("steps", "A", 1, None, 3, 6),  # a pad step, then A again (in full)
        ("prefill",),
        (
            "steps",
            "B",
            1,
            "+2",
            6,
            6,
        ),  # slot reuse: B's prompt ends 2 past A's last step, invalidated
        (
            "steps",
            "C",
            2,
            500,
            5,
            6,
        ),  # a slot change, no invalidate: the step rule imports in full
        (
            "steps",
            "C",
            2,
            "-40",
            3,
            6,
        ),  # a rewind (recompute) on the same slot, no invalidate
        ("steps", "C", 2, "+20", 3, 6),  # a jump past a step's rows, no invalidate
        ("prefill",),
        ("steps", "D", 0, 1100, 6, 6),  # a long context
    ]
    for rp, _ in configs:
        worlds = [{k: v.clone() for k, v in base.items()} for _ in range(2)]
        ctx, last, nsteps, nincr, fails = 0, None, 0, 0, []
        for ev in script:
            if ev[0] == "prefill":
                prefill(worlds)
                continue
            if ev[0] == "pad":
                db, da, _ = served(worlds, ctx, 0, last[0], last[1], rp)
                nsteps += 1
                fails += [("pad", db, da)] if db or da else []
                continue
            _, req, slot, start, count, live = ev
            if start is not None:
                ctx = ctx + int(start) if isinstance(start, str) else start
            for _ in range(count):
                db, da, inc = served(worlds, ctx, live, req, slot, rp)
                nsteps, nincr = nsteps + 1, nincr + inc
                if db or da:
                    fails.append((req, ctx, inc, db, da))
                ctx += int(torch.randint(1, live + 1, (1,), generator=g, device="cpu"))
            last = (req, slot)
        report(f"[8] incremental ring import == full import, {nsteps} steps ({nincr} incremental), RP {rp}",
               not fails and nincr > nsteps // 2, f"(fails {fails[:3]})" if fails else "")  # fmt: skip
    if not control:
        return
    # the negative control: a prefill the hook is not told of (no invalidate) on a continuing step is caught
    worlds = [{k: v.clone() for k, v in base.items()} for _ in range(2)]
    served(worlds, 300, 6, "A", 1, 16)
    prefill(worlds, invalidate=False)
    db, _, inc = served(worlds, 303, 6, "A", 1, 16)
    report(
        "[8] negative control: an un-invalidated prefill differs",
        inc and bool(db),
        f"({len(db)} tensors differ)",
    )


class _BatchWorld:
    """ATOM's side and the hi-side buffers of a batch-capable hook (``BM`` request slots), with requests "A".."F"
    each its own slot: the pool (its rings); the hi-side rings [NMF, BM, WIN, DIM], ``pos`` [BM], bounds [3, RB] a
    row, the request words ``req``, ``imp`` [BM x 4]; the end step's taps / gates a row. (``pos`` has 8 spare words:
    a one-request launch on a later request's views clears the dead slots' first positions past its own.)
    """

    dim, win, hid, RB = 512, 128, 2048, 32
    mf = (0, 2, 3, 20, 39)
    ring_start0, ring_slots, slot_rows, num_slots = 37000, 134, 6000, 6

    def __init__(self, bm: int, rows: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        dev = torch.get_default_device()

        def rnd(*shape):
            return torch.randn(*shape, generator=g, device="cpu").to(dev)

        self.rnd, self.g, self.bm, self.R = rnd, g, bm, rows
        d, w = self.dim, {}
        w["pool"] = rnd(
            self.ring_start0 + self.num_slots * self.slot_rows + 8, d
        ).bfloat16()
        w["rings"] = rnd(len(self.mf), bm, self.win, d).bfloat16()
        w["pos"] = torch.zeros(bm + 8, dtype=torch.int32)
        w["bounds"] = torch.zeros(3, self.RB, dtype=torch.int32)
        w["req"] = torch.full((ml.REQ_WORDS,), 9, dtype=torch.int32)
        w["hidden"] = rnd(rows, self.hid).bfloat16()
        w["h_in"] = rnd(rows + 2, 4, self.hid).bfloat16()
        w["step"] = torch.tensor([5], dtype=torch.int32)
        w["gate"], w["pre"], w["post"], w["comb"] = rnd(rows, 24), torch.zeros(rows, 4), torch.zeros(rows, 4), \
            torch.zeros(rows, 16)  # fmt: skip
        w["imp"] = torch.full((bm * 4,), -1, dtype=torch.int32)
        for t in range(3):
            w[f"r_a{t}"], w[f"aux{t}"] = (
                rnd(rows, 4, self.hid).bfloat16(),
                torch.zeros(rows, self.hid).bfloat16(),
            )
        self.w = w

    def clone(self):
        c = object.__new__(type(self))
        c.__dict__.update(self.__dict__)
        c.w = {k: v.clone() for k, v in self.w.items()}
        return c

    def atom_writes(self, seed):
        """ATOM writes the caches itself (a prefill): the pool."""
        g = torch.Generator().manual_seed(seed)
        w = self.w
        w["pool"].copy_(
            torch.randn(w["pool"].shape, generator=g, device="cpu")
            .to(w["pool"].device)
            .bfloat16()
        )

    def atom_writes_slot(self, seed, slot):
        """ATOM writes one request's caches only (its prefill): its ring rows (every layer's, its slot)."""
        g = torch.Generator().manual_seed(seed)
        w = self.w
        r0 = self.ring_start0 + slot * self.slot_rows
        w["pool"][r0 : r0 + self.slot_rows] = torch.randn(self.slot_rows, self.dim, generator=g, device="cpu").to(
            w["pool"].device).bfloat16()  # fmt: skip

    def geo(self):
        return (
            self.ring_start0,
            self.slot_rows,
            self.ring_slots,
            self.num_slots,
            1 << 20,
        )

    def begin(
        self, positions, batch, slot_map, rows, nb, incr=False, rp=16, r0=0, b0=0
    ):
        """The fused begin-step launch for ``nb`` requests of ``rows`` rows (``r0`` / ``b0``: one request's views; its
        request words into a scratch of its own: the hook launches one request at batch slot 0 only).
        """
        w = self.w
        rpl = rp if incr else self.win
        d = w["pos"]
        req = torch.zeros(ml.REQ_WORDS, dtype=torch.int32) if b0 or r0 else w["req"]
        ml._begin_step_kernel[(nb * len(self.mf) * rpl + rows,)](
            w["pool"], positions, batch, slot_map, w["rings"][:, b0:], torch.tensor(self.mf, dtype=torch.int32),
            w["pos"][b0:], w["bounds"].view(-1)[r0:], *self.geo(), rows, w["hidden"][r0:],
            w["hidden"].stride(0), w["h_in"][r0:], w["step"], req, d, 0, d, d, 0, d, d, 0, d,
            w["imp"][4 * b0 :], NMF=len(self.mf), WIN=self.win, DIM=self.dim, RB=self.RB, HIDDEN=self.hid, BLOCK=1024,
            NHC=4, NE0=0, NE1=0, EHIST=0, INCR=int(incr), RP=rpl, NB=nb, BM=self.bm,
            RQP=triton.next_power_of_2(rows // nb),
        )  # fmt: skip

    def end(self, positions, batch, slot_map, rows, nb, r0=0, b0=0, track=True):
        w = self.w
        taps = [w[f"{k}{t}"][r0:] for t in range(3) for k in ("r_a", "aux")]
        ml._end_step_kernel[((len(self.mf) + 3 + 1) * rows,)](
            w["pool"], positions, batch, slot_map, w["rings"][:, b0:], torch.tensor(self.mf, dtype=torch.int32),
            *self.geo(), rows, *taps, w["gate"][r0:], w["pre"][r0:], w["post"][r0:], w["comb"][r0:], w["imp"][4 * b0 :],
            NMF=len(self.mf), WIN=self.win, DIM=self.dim, HIDDEN=self.hid, BLOCK=1024, NHC=4, NMIX=24, NTAPS=3,
            PLANAR=True, TRACK=int(track), NB=nb, BM=self.bm,
        )  # fmt: skip

    def separate(self, positions, batch, slot_map, rows, nb):
        """The separate kernels for ``nb`` requests."""
        w, rq = self.w, rows // nb
        ids = torch.tensor(self.mf, dtype=torch.int32)
        w["step"].add_(1)
        w["h_in"][:rows].copy_(w["hidden"].view(rows, 1, -1).expand(rows, 4, self.hid))
        ml._import_rings_kernel[(len(self.mf), self.win, nb)](
            w["pool"], positions, batch, slot_map, w["rings"], ids, w["pos"], w["bounds"], *self.geo(), rows,
            WIN=self.win, DIM=self.dim, RB=self.RB, NB=nb, BM=self.bm)  # fmt: skip
        ml._req_words_kernel[(nb,)](positions, batch, slot_map, self.num_slots, 1 << 20, w["req"], w["pos"], rows,
                                    NB=nb, BM=self.bm, RQP=triton.next_power_of_2(rq))  # fmt: skip
        return rq

    def separate_end(self, positions, batch, slot_map, rows, nb):
        w, rq = self.w, rows // nb
        ids = torch.tensor(self.mf, dtype=torch.int32)
        ml._gates_planar_kernel[(rows,)](
            w["gate"], w["pre"], w["post"], w["comb"], NMIX=24, NHC=4
        )
        ml._export_rings_kernel[(len(self.mf), rows)](w["pool"], positions, batch, slot_map, w["rings"], ids,
                                                      *self.geo(), WIN=self.win, DIM=self.dim, RQ=rq, BM=self.bm)  # fmt: skip
        for t in range(3):
            ml._aux_tap_kernel[(rows,)](
                w[f"r_a{t}"], w[f"aux{t}"], HIDDEN=self.hid, BLOCK=1024, NHC=4
            )

    def mono_writes(self, positions_host, batch_host, rq, g):
        """The layers' writes between the launches (the same for every world): each row's ring slot in its request's
        batch slot (padding rows and pad requests included, as the programs write)."""
        dev = torch.get_default_device()
        return [("rings", (m, j // rq, max(int(p), 0) % self.win), torch.randn(self.dim, generator=g, device="cpu").to(dev))
                for j, p in enumerate(positions_host) for m in range(len(self.mf))]  # fmt: skip

    def apply(self, writes):
        for k, at, v in writes:
            self.w[k][at] = v.to(self.w[k].dtype)

    MONO = ("rings", "pos", "bounds", "h_in", "step", "req")
    ATOM = ("pool", "aux", "pre", "post", "comb", "imp")


def _batch_step(reqs, rq, slot_of, rows=None):
    """A step's routing for requests ``reqs`` [(name, ctx, live rows) or None (a pad request)] of ``rq`` rows each:
    (positions int64 [rows], batch ids [rows] (ATOM's: request i's rows carry i, pad rows -1), slot map [nb]).
    """
    nb = len(reqs)
    rows = rows or nb * rq
    pos, bat, slots = [0] * rows, [-1] * rows, []
    for b, r in enumerate(reqs):
        if r is None:
            slots.append(0)
            continue
        name, ctx, live = r
        slots.append(slot_of[name])
        for j in range(rq):
            pos[b * rq + j] = ctx + j
            bat[b * rq + j] = b if j < live else -1
    return (
        torch.tensor(pos, dtype=torch.int64),
        torch.tensor(bat, dtype=torch.int32),
        torch.tensor(slots, dtype=torch.int32),
    )


def _differ(names, a, b) -> list:
    return [
        k
        for k in a.w
        if k.startswith(names)
        and not torch.equal(
            a.w[k].contiguous().view(torch.uint8), b.w[k].contiguous().view(torch.uint8)
        )
    ]


def check_batch_bridge():
    """[10a] The bridge at B requests a step (``Scenario(B, 6)``, the hook's batch widths): the fused begin / end
    launches of B requests == B one-request launches on each request's batch slot (views: its rings, caches, pos,
    its rows' bounds / embedding / taps / gates, its ``imp``) == the separate kernels at B requests; B = 2 / 3 / 4,
    requests with other slots / contexts (below the window, wrapping, long), a pad request, padding rows at the end,
    random state everywhere so a misrouted byte shows. Import full (``INCR`` 0) and incremental from a cleared
    ``imp`` (the full grid's loops)."""
    rq = 6
    cases = {
        2: [("A", 37, 6), ("B", 1131, 6)],
        3: [("C", 5, 6), None, ("A", 300, 3)],
        4: [("B", 700, 6), ("A", 127, 6), ("D", 1000, 6), ("C", 2, 4)],
    }
    slot_of = {"A": 2, "B": 0, "C": 5, "D": 4}
    for nb, reqs in cases.items():
        for incr in (False, True):
            base = _BatchWorld(4, nb * rq, 100 + nb)
            positions, batch, slot_map = _batch_step(reqs, rq, slot_of)
            g = torch.Generator().manual_seed(nb)
            writes = base.mono_writes(positions.tolist(), batch.tolist(), rq, g)
            a, b, c = base.clone(), base.clone(), base.clone()
            a.begin(positions, batch, slot_map, nb * rq, nb, incr=incr)
            for r in range(
                nb
            ):  # B: one request at a time, on its own views (positions / ids: its rows)
                sl = slice(r * rq, (r + 1) * rq)
                bb = batch[sl].clone()
                b.begin(positions[sl], bb, slot_map, rq, 1, incr=incr, r0=r * rq, b0=r)
            c.separate(positions, batch, slot_map, nb * rq, nb)
            begin_ab = _differ(_BatchWorld.MONO, a, b)
            # one bump a launch: B bumped nb times
            begin_ab = [k for k in begin_ab if k not in ("step", "req")] + (
                ["step"] if int(b.w["step"]) != 5 + nb else []
            )
            begin_ac = _differ(_BatchWorld.MONO, a, c)
            for w in (a, b, c):
                w.apply(writes)
            a.end(positions, batch, slot_map, nb * rq, nb)
            for r in range(nb):
                sl = slice(r * rq, (r + 1) * rq)
                b.end(
                    positions[sl],
                    batch[sl].clone(),
                    slot_map,
                    rq,
                    1,
                    r0=r * rq,
                    b0=r,
                )
            c.separate_end(positions, batch, slot_map, nb * rq, nb)
            end_ab = _differ(_BatchWorld.ATOM, a, b)
            end_ac = [
                k for k in _differ(_BatchWorld.ATOM, a, c) if k != "imp"
            ]  # (no TRACK in the separate form)
            ok = not (begin_ab or begin_ac or end_ab or end_ac)
            written = sum(not torch.equal(a.w[k], base.w[k]) for k in base.w)
            report(f"[10a] bridge at {nb} requests ({'incr' if incr else 'full'} import) == {nb} one-request launches "
                   f"on their slots == the separate kernels", ok,
                   f"({written} / {len(base.w)} tensors written" + ("" if ok else f"; DIFF begin {begin_ab} / "
                   f"{begin_ac}, end {end_ab} / {end_ac}") + ")")  # fmt: skip


def check_batch_incremental(configs=((16, 7), (128, 64), (5, 300))):
    """[10b] The incremental import with B requests a step: world A imports in full every step, world B
    incrementally (``imp`` a request slot), world C runs each request alone through the one-request bridge on its
    batch slot's views (the c1 form): over a scripted run where requests join and leave (widths 1 -> 2 -> 3 -> 4 ->
    2 -> 1 requests), reorder (a request moves to another batch slot), differ in context, accept 1-6 rows (EOS /
    max_tokens mid-block: a request leaving after any accept), are aborted (gone without a prefill), sit beside a pad
    request or end in padding rows, with prefills (the hook's invalidate) between. After every begin-step launch the
    hi-side tensors of A, B and C are equal byte for byte, and after every end-step launch ATOM's.
    """
    rq, bm = 6, 4
    slot_of = {"A": 1, "B": 3, "C": 0, "D": 5, "E": 2, "F": 4}
    base = _BatchWorld(bm, bm * rq, 77)
    g = torch.Generator().manual_seed(78)
    ctx = {"A": 37, "B": 900, "C": 126, "D": 5, "E": 1100, "F": 300}
    # (event, ...): ("prefill",) ATOM writes + invalidate; ("run", [names or None], steps, live of the last request)
    script = [
        ("prefill",),
        ("run", ["A"], 4, 6),
        ("prefill_one", "F"),
        (
            "run",
            ["A"],
            2,
            6,
        ),  # another request's prefill: A's next step stays incremental
        ("prefill",),
        ("run", ["A", "B"], 6, 6),  # B joins (its prefill invalidated)
        ("prefill",),
        ("run", ["A", "B", "C"], 5, 6),  # a third, short context
        ("prefill",),
        ("run", ["A", "B", "C", "D"], 4, 6),  # four
        (
            "run",
            ["A", "C", "D"],
            3,
            6,
        ),  # B leaves (EOS mid-block / abort: no prefill): C, D move up a slot
        ("run", ["D", "A", "C"], 3, 6),  # a reorder (the scheduler's), no event
        ("run", ["D", None, "C"], 2, 6),  # a pad request between
        ("run", ["D", "C"], 3, 3),  # A gone; C ends in padding rows (3 live of 6)
        ("prefill_one", "A"),
        ("run", ["D", "C"], 2, 3),  # a prefill on A's old slot: D, C stay incremental
        ("prefill",),
        ("run", ["D", "C", "E"], 4, 6),  # E joins near the buffers' end
        ("run", ["E"], 3, 6),  # back to one request (the c1 width), E moved to slot 0
        ("prefill",),
        ("run", ["F", "E"], 3, 6),
    ]
    for rp, _ in configs:
        a, b, c = base.clone(), base.clone(), base.clone()
        nsteps, nincr, fails, keep = 0, 0, [], set()
        for ev in script:
            if (
                ev[0] == "prefill_one"
            ):  # one request's prefill (its pages, ring and compress slot): per-slot invalidate
                seed = int(torch.randint(0, 2**30, (1,), generator=g, device="cpu"))
                for w in (a, b, c):
                    w.atom_writes_slot(seed, slot_of[ev[1]])
                st = torch.tensor([slot_of[ev[1]]], dtype=torch.int32)
                for w in (b, c):
                    ml._invalidate_kernel[(bm,)](w.w["imp"], st, 1, BLOCK=2)
                keep = {x for x in "ABCDEF" if x != ev[1]}
                continue
            if ev[0] == "prefill":
                seed = int(torch.randint(0, 2**30, (1,), generator=g, device="cpu"))
                for w in (a, b, c):
                    w.atom_writes(seed)
                b.w["imp"].fill_(-1)
                c.w["imp"].fill_(-1)
                continue
            _, names, count, last_live = ev
            for _ in range(count):
                nb = len(names)
                reqs = [
                    None if x is None else (x, ctx[x], last_live if i == nb - 1 else rq)
                    for i, x in enumerate(names)
                ]
                positions, batch, slot_map = _batch_step(reqs, rq, slot_of)
                imp = b.w["imp"].view(-1, 4).tolist()
                incr = [r is not None and imp[i][0] == slot_of[r[0]] and imp[i][1] <= r[1] <= imp[i][1] + imp[i][2]
                        for i, r in enumerate(reqs)]  # fmt: skip
                if (
                    keep
                ):  # the first step after another request's prefill: its requests import incrementally
                    lost = [
                        r[0]
                        for i, r in enumerate(reqs)
                        if r is not None and r[0] in keep and not incr[i]
                    ]
                    fails += (
                        [
                            (
                                "kept requests imported in full after another's prefill",
                                lost,
                            )
                        ]
                        if lost
                        else []
                    )
                    keep = set()
                writes = base.mono_writes(positions.tolist(), batch.tolist(), rq, g)
                a.begin(positions, batch, slot_map, nb * rq, nb, incr=False)
                b.begin(positions, batch, slot_map, nb * rq, nb, incr=True, rp=rp)
                for r in range(nb):
                    sl = slice(r * rq, (r + 1) * rq)
                    c.begin(positions[sl], batch[sl].clone(), slot_map, rq, 1, incr=True, rp=rp, r0=r * rq,
                            b0=r)  # fmt: skip
                db = [
                    k
                    for k in _differ(_BatchWorld.MONO, a, b)
                    + [k for k in _differ(_BatchWorld.MONO, a, c) if k != "req"]
                    if k != "step"
                ]
                for w in (a, b, c):
                    w.apply(writes)
                a.end(positions, batch, slot_map, nb * rq, nb, track=False)
                b.end(positions, batch, slot_map, nb * rq, nb)
                for r in range(nb):
                    sl = slice(r * rq, (r + 1) * rq)
                    c.end(
                        positions[sl],
                        batch[sl].clone(),
                        slot_map,
                        rq,
                        1,
                        r0=r * rq,
                        b0=r,
                    )
                da = [
                    k
                    for k in _differ(_BatchWorld.ATOM, a, b)
                    + _differ(_BatchWorld.ATOM, b, c)
                    if k != "imp"
                ]
                da += [] if torch.equal(b.w["imp"], c.w["imp"]) else ["imp (B vs C)"]
                nsteps, nincr = nsteps + 1, nincr + sum(incr)
                if db or da:
                    fails.append((names, [r[1] for r in reqs if r], incr, db, da))
                for i, x in enumerate(
                    names
                ):  # each live request advances by its own accept (1..its live rows)
                    if x is not None:
                        live = last_live if i == nb - 1 else rq
                        ctx[x] += int(
                            torch.randint(1, live + 1, (1,), generator=g, device="cpu")
                        )
        report(f"[10b] batch incremental import == full import == one request at a time, {nsteps} steps "
               f"({nincr} request imports incremental), RP {rp}", not fails and nincr > nsteps,
               f"(fails {fails[:2]})" if fails else "")  # fmt: skip


class _BandWorld(_BatchWorld):
    """``_BatchWorld`` with the batch bands' slots: 8 requests "A".."H" (a slot each) and the bounds of 48 rows."""

    num_slots, RB = 8, 64


def check_bands():
    """[10g] The batch bands (``hi_compose.BANDS``): ``_band_step``: the live words after the step counter and the
    dead slots' import records cleared (the live ones' and the counter untouched); the bridge with the bands' 8 request
    slots at 5 / 8 requests (a band's live requests: the fused launches == one-request launches on their slots == the
    separate kernels, full and incremental import)."""
    import ast

    hook_src = (MONO / "hi_runner.py").read_text()
    from atom.models.deepseek_v41.mono.hi_bands import LiveWords, plan_widths

    tree = ast.parse(hook_src)
    ns = {"ml": ml, "torch": torch, "np": np, "os": os, "MAX_BATCH": 4}
    code = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_band_step"
    ]
    hook = compile(ast.Module(body=code, type_ignores=[]), "hi_runner", "exec")
    exec(hook, ns)  # noqa: S102 - the hook's own source
    ok, notes = True, []
    # _band_step on a stand-in: live words staged, the dead slots' records cleared
    step = torch.tensor([9, 0, 0, 0], dtype=torch.int32)
    h = SimpleNamespace(step=step, live=LiveWords(step, plan_widths([30, 48], 6, ns["MAX_BATCH"], (6, 8))),
                        imp=torch.arange(8 * ml.IMP_WORDS, dtype=torch.int32))  # fmt: skip
    ns["_band_step"](h, 30)
    imp = h.imp.view(8, ml.IMP_WORDS)
    good = h.step.tolist() == [9, 30, 6, 5] and bool((imp[5:6] == -1).all())
    good &= torch.equal(
        imp[:5].reshape(-1), torch.arange(5 * ml.IMP_WORDS, dtype=torch.int32)
    )
    good &= torch.equal(
        imp[6:].reshape(-1),
        torch.arange(6 * ml.IMP_WORDS, 8 * ml.IMP_WORDS, dtype=torch.int32),
    )
    ns["_band_step"](h, 48)
    good &= h.step.tolist() == [9, 48, 6, 8]
    ok &= good
    notes += [] if good else ["_band_step"]
    # the bridge with 8 slots at a band's live requests
    rq = 6
    cases = {5: [("H", 41, 6), ("B", 1131, 6), None, ("G", 300, 3), ("A", 2, 6)],
             8: [("A", 37, 6), ("B", 700, 6), ("C", 5, 6), ("D", 1000, 6), ("E", 127, 6), ("F", 900, 6), ("G", 64, 6),
                 ("H", 333, 4)]}  # fmt: skip
    slot_of = {k: i for i, k in enumerate("CHAEGBDF")}
    for nb, reqs in cases.items():
        for incr in (False, True):
            base = _BandWorld(8, nb * rq, 200 + nb)
            positions, batch, slot_map = _batch_step(reqs, rq, slot_of)
            g = torch.Generator().manual_seed(nb)
            writes = base.mono_writes(positions.tolist(), batch.tolist(), rq, g)
            a, b, c = base.clone(), base.clone(), base.clone()
            a.begin(positions, batch, slot_map, nb * rq, nb, incr=incr)
            for r in range(nb):
                sl = slice(r * rq, (r + 1) * rq)
                b.begin(
                    positions[sl],
                    batch[sl].clone(),
                    slot_map,
                    rq,
                    1,
                    incr=incr,
                    r0=r * rq,
                    b0=r,
                )
            c.separate(positions, batch, slot_map, nb * rq, nb)
            begin_ab = [
                k for k in _differ(_BatchWorld.MONO, a, b) if k not in ("step", "req")
            ]
            begin_ac = _differ(_BatchWorld.MONO, a, c)
            for w in (a, b, c):
                w.apply(writes)
            a.end(positions, batch, slot_map, nb * rq, nb)
            for r in range(nb):
                sl = slice(r * rq, (r + 1) * rq)
                b.end(
                    positions[sl],
                    batch[sl].clone(),
                    slot_map,
                    rq,
                    1,
                    r0=r * rq,
                    b0=r,
                )
            c.separate_end(positions, batch, slot_map, nb * rq, nb)
            end_ab = _differ(_BatchWorld.ATOM, a, b)
            end_ac = [k for k in _differ(_BatchWorld.ATOM, a, c) if k != "imp"]
            if begin_ab or begin_ac or end_ab or end_ac:
                ok = False
                notes.append(f"bridge {nb} ({'incr' if incr else 'full'}): begin {begin_ab} / {begin_ac}, end {end_ab} / "
                             f"{end_ac}")  # fmt: skip
    report(
        "[10g] batch bands: split past MAX_BATCH, _band_step, the bridge at 5 / 8 of 8 request slots",
        ok,
        f"({notes})" if notes else "",
    )


def check_draft_begin():
    """[10e] The block draft's begin kernel (``_draft_begin_kernel``): the block streams (row 0 the anchor's embedding, the
    rest the noise row, in every stream), the launch-tag bump, and the block's first position anchor + 1 clamped to
    [0, max_pos] (the target's rule): a real anchor unchanged, an anchor at / past the bucket's end or below 0 (a
    capture's dummy, a stale anchor) in range, so the draft's RoPE reads (first position .. + 4) stay inside its table.
    """
    dev = torch.get_default_device()
    hid, W, max_ctx = 96, 5, 2048
    max_pos = max_ctx - 16
    rows_rope = max_ctx + 64  # the hook's rope table (``rope_table(max_ctx + 64)``)
    ok, bad = True, []
    for anchor, want in ((0, 1), (700, 701), (max_pos - 1, max_pos), (max_pos, max_pos), (max_ctx - 1, max_pos),
                         (max_ctx + 500, max_pos), (2**31 - 2, max_pos), (-1, 0), (-7, 0)):  # fmt: skip
        g = torch.Generator().manual_seed(anchor & 0xFFFF)
        emb = torch.randn(hid, generator=g, device="cpu").bfloat16().to(dev)
        noise = torch.randn(hid, generator=g, device="cpu").bfloat16().to(dev)
        h_in = torch.zeros(W, 4, hid, dtype=torch.bfloat16, device=dev)
        pos = torch.full((1,), -9, dtype=torch.int32, device=dev)
        step = torch.tensor([3], dtype=torch.int32, device=dev)
        apos = torch.tensor([anchor], dtype=torch.int64, device=dev)
        ml._draft_begin_kernel[(W, 2)](
            emb, noise, h_in, apos, pos, step, hid, max_pos, NHC=4, BLOCK=64
        )
        got = int(pos.cpu())
        rows_ok = torch.equal(h_in[0].cpu(), emb.cpu().expand(4, hid)) and all(
            torch.equal(h_in[r].cpu(), noise.cpu().expand(4, hid)) for r in range(1, W))  # fmt: skip
        good = (
            got == want
            and 0 <= got
            and got + W - 1 < rows_rope
            and int(step.cpu()) == 4
            and rows_ok
        )
        ok &= good
        bad += [] if good else [(anchor, got, want)]
    # a batched draft's later requests: their streams and position at their rows / word, no second bump
    h_in = torch.zeros(2 * W, 4, hid, dtype=torch.bfloat16, device=dev)
    pos = torch.full((2,), -9, dtype=torch.int32, device=dev)
    step = torch.tensor([3], dtype=torch.int32, device=dev)
    emb = torch.randn(2, hid).bfloat16().to(dev)
    noise = torch.randn(hid).bfloat16().to(dev)
    for b, a in enumerate((40, 900)):
        ml._draft_begin_kernel[(W, 2)](emb[b], noise, h_in[b * W :], torch.tensor([a], device=dev), pos[b:], step, hid,
                                       max_pos, NHC=4, BLOCK=64, BUMP=int(b == 0))  # fmt: skip
    ok_b = (
        pos.cpu().tolist() == [41, 901]
        and int(step.cpu()) == 4
        and torch.equal(h_in[W].cpu(), emb[1].cpu().expand(4, hid))
    )
    ok &= ok_b
    report(
        "[10e] draft begin: first position = anchor + 1 clamped to [0, max_ctx - 16] (real anchors unchanged), "
        "streams and the tag bump (once a batched draft)",
        ok,
        f"(wrong {bad}, batched {ok_b})" if not ok else "",
    )


def _small_mapping():
    """A small ATOM ``NgramHashMapping`` (real ``EngramConfig`` arithmetic, a stand-in tokenizer of 1,037 ids onto 300
    compressed ones): (mapping, the tokenizer class, layer ids, compressed vocab)."""
    mp = importlib.import_module("atom.model_ops.engram.mapping")
    layers, ngram, heads, vocab, vc = (1, 14), 4, 8, 1000, 300
    seen, rows = set(), []
    for (
        _
    ) in (
        layers
    ):  # the table rows ATOM's mapping checks its primes against (its own search order)
        n = 0
        for _ in range(ngram - 1):
            start = vocab - 1
            for _ in range(heads):
                start = mp._next_prime(start, seen)
                seen.add(start)
                n += start
        rows.append(n)
    rng = np.random.default_rng(7)

    class Tok:
        lookup_table = rng.integers(0, vc, size=vocab + 37, dtype=np.int64)

        def __len__(self):
            return vc

        def __call__(self, ids):
            arr = np.asarray(ids, dtype=np.int64)
            return np.where(arr >= 0, self.lookup_table[np.maximum(arr, 0)], arr)

    cfg = mp.EngramConfig(layer_ids=layers, num_embeddings=tuple(rows), max_ngram_size=ngram, vocab_size=vocab,
                          n_heads=heads, head_dim=32, pad_token_id=5, compressed_vocab_size=vc, seed=3)  # fmt: skip
    return mp.NgramHashMapping(cfg, Tok()), Tok, layers, vc


def check_engram_fold():
    """[ef] The begin-step's Engram fold (``_eng_fold``, through its program-a-request kernel)
    writes the cursor candidates ATOM's staging snapshot writes (``engram_compress`` + ``engram_snapshot``,
    ``STAGE_CURSOR``) and copies the step's ids, on scripted steps of 1-4 requests (each its own slot and first
    position, 1-6 rows each): live tokens below the width (the rest untouched), dead rows, requests at positions 0-2
    (no / a partial history), the token map the derived one (``engram_consts``)."""
    import triton

    mh = importlib.import_module("atom.models.deepseek_v41.mono.hi_runner")
    hs = importlib.import_module("atom.model_ops.engram.device.hashing")
    dev = torch.get_default_device()
    mapping, Tok, _, vc = _small_mapping()
    tables = hs.EngramHashTables.from_mapping(mapping, dev)
    tmap = mh.engram_consts(mapping)["token_map"].to(dev)
    g = torch.Generator(device="cpu").manual_seed(5)
    nslot = 8
    cursor = torch.randint(-1, vc, (nslot, 4), generator=g, device="cpu").to(dev)
    i32 = {"dtype": torch.int32, "device": dev}
    ok, cases = True, 0
    # (requests' first positions, rows a request, live tokens, dead fraction)
    for p0s, rq, live, deadp in (([900], 6, 6, 0.0), ([77], 6, 4, 0.0), ([0], 6, 6, 0.0), ([2], 6, 6, 0.0),
                                 ([300], 4, 4, 0.3), ([41], 1, 1, 0.0), ([50], 6, 0, 0.0), ([1], 6, 6, 0.5),
                                 ([9990, 32700], 6, 12, 0.0), ([1, 32700, 5000, 2], 6, 24, 0.1),
                                 ([10000, 0, 33000], 6, 15, 0.0)):  # fmt: skip
        nb = len(p0s)
        rows = nb * rq
        slots = torch.randperm(nslot, generator=g, device="cpu")[:nb].to(**i32)
        cur = cursor.clone()
        ids = torch.randint(
            0, len(Tok.lookup_table), (rows,), generator=g, device="cpu"
        ).to(**i32)
        dead = (torch.rand(rows, generator=g, device="cpu") < deadp).to(**i32)
        positions = torch.cat(
            [p + torch.arange(rq, dtype=torch.int32) for p in p0s]
        ).to(dev)
        live_t = torch.tensor([live], **i32)
        want = torch.full((nb, rq, 4), -9, dtype=torch.int64, device=dev)
        comp = torch.zeros(rows, **i32)
        hs.engram_compress(tables, ids, dead_mask=dead, out=comp, live=live_t)
        batch = hs.EngramBatch(compressed=comp, batch_ids=torch.arange(nb, **i32).repeat_interleave(rq),
                               cu_seqlens=torch.arange(0, rows + 1, rq, **i32), history=cur[:, 1:],
                               history_index=slots)  # fmt: skip
        snap = torch.empty(rows, 4, dtype=torch.int64, device=dev)
        hs.engram_snapshot(
            tables,
            batch,
            snap,
            cursor_positions=positions,
            cursor_out=want,
            live=live_t,
        )
        got = torch.full_like(want, -9)
        dst = torch.full((32,), -5, **i32)
        ml._engram_fold_kernel[(nb,)](rq, cur, cur.stride(0), slots, positions, ids, dst, tmap, dead, got, got.stride(1),
                                      got.stride(0), live_t, NGRAM1=3, RPOW=triton.next_power_of_2(rq))  # fmt: skip
        ok &= (
            torch.equal(got, want)
            and torch.equal(dst[:rows], ids)
            and bool((dst[rows:] == -5).all())
        )
        cases += 1
    report(
        f"[ef] Engram fold: ATOM's snapshot cursor candidates + the ids copy in the begin-step's history programs "
        f"({cases} steps of 1-4 requests: live < rows, dead rows, short history, far-apart positions)",
        ok,
    )


def _verdict_launch(rs, spec, head_ids, sampler, uniform=None, seed=None):
    """The verdict reference (``mono_hi_bridge_ref._head_verdict_kernel``) on a step: ``uniform``, the step's uniforms
    (ATOM's), else drawn in the kernel from the sampler's step (``seed``: another seed, the post-verify's).
    """
    cu = spec.cu_num_draft_tokens
    nreq, steps, dev = len(cu), spec.num_spec_steps, head_ids.device
    out = torch.empty((nreq, steps + 1), dtype=torch.int32, device=dev)
    verdict = torch.empty((3, nreq), dtype=torch.int32, device=dev)
    rates = sampler.synthetic_acceptance_rates
    synthetic = rates is not None
    cond = rs._get_synthetic_cond_rates(rates, dev) if synthetic else out
    ml._head_verdict_kernel[(nreq,)](
        out, verdict, cu, spec.input_ids, spec.target_logits_indices, spec.bonus_logits_indices, head_ids,
        uniform if uniform is not None else out, cond,
        rs._SYNTHETIC_RNG_BASE_SEED + int(sampler._synthetic_step) if seed is None else seed, steps,
        SYNTHETIC=synthetic, DRAW=uniform is None, num_warps=1,
    )  # fmt: skip
    if synthetic:
        sampler._synthetic_step += 1
    return out, verdict


def check_verdict_reference():
    """[vf] The verdict reference the in-graph post-verify is checked against (``_verdict_launch``) against ATOM's path
    (the two ``index_select``, ``RejectionSampler.forward``) on scripted steps of 1-4 requests with 0-5 drafts each
    (head ids agreeing with the drafts up to a random row): real acceptance bit for bit; forced acceptance on ATOM's own
    uniforms bit for bit, the step counter advanced alike; with the uniforms drawn in the kernel, the same on a re-run
    (the same seed: TP-uniform), the schedule's certain positions as ATOM's and the fractional one accepted at its
    rate."""
    rs = importlib.import_module("atom.model_ops.rejection_sampler")
    dev = torch.get_default_device()
    g = torch.Generator(device="cpu").manual_seed(11)

    def scenario(drafts, spec=5):
        """A step: per request its draft count; rows = drafts + 1 a request (the bonus row last)."""
        rows, tidx, bidx, cu, ids, head = 0, [], [], [], [], []
        for n in drafts:
            seg = torch.randint(0, 1000, (n + 1,), generator=g, device="cpu")
            agree = int(
                torch.randint(0, n + 2, (1,), generator=g, device="cpu")
            )  # rows whose head id is the next draft
            h = torch.randint(0, 1000, (n + 1,), generator=g, device="cpu")
            for i in range(min(agree, n)):
                h[i] = seg[i + 1]
            tidx += [rows + i for i in range(n)]
            bidx.append(rows + n)
            ids.append(seg)
            head.append(h)
            rows += n + 1
            cu.append(len(tidx))
        i32 = {"dtype": torch.int32, "device": dev}
        meta = SimpleNamespace(input_ids=torch.cat(ids + [torch.zeros(1, dtype=torch.long, device="cpu")]).to(**i32),
                               target_logits_indices=torch.tensor(tidx, dtype=torch.long, device=dev),
                               bonus_logits_indices=torch.tensor(bidx, dtype=torch.long, device=dev),
                               cu_num_draft_tokens=torch.tensor(cu, **i32), num_spec_steps=spec)  # fmt: skip
        return meta, torch.cat(head).to(**i32)

    def atom(meta, head, sampler):
        t = torch.index_select(head, 0, meta.target_logits_indices)
        b = torch.index_select(head, 0, meta.bonus_logits_indices)
        # the verify rows' logits: their argmax the head's ids (the sampler reads them alongside the ids)
        logits = torch.nn.functional.one_hot(t.long(), 1000).float()
        return sampler.forward(meta, logits, b, target_token_ids=t)

    def same(a, b):
        return all(
            torch.equal(x.to(torch.int32), y.to(torch.int32)) for x, y in zip(a, b)
        )

    steps = [[5], [5], [3], [0], [5, 5], [2, 5, 0], [5, 4, 3, 1], [1, 1, 1, 1]] * 4
    got = {
        "real": True,
        "forced, ATOM's uniforms": True,
        "drawn: re-run equal": True,
        "drawn: certain rows": True,
    }
    real = rs.RejectionSampler(None)
    for d in steps:
        meta, head = scenario(d)
        got["real"] &= same(
            atom(meta, head, real), _verdict_launch(rs, meta, head, real)
        )
    rates = rs.acceptance_length_to_rates(3.51, 5)
    fa, fb = rs.RejectionSampler(rates), rs.RejectionSampler(rates)
    frac = [0, 0]
    for d in steps * 4:
        meta, head = scenario(d)
        u = torch.rand(meta.target_logits_indices.shape[0], dtype=torch.float32, device=dev,
                       generator=rs._synthetic_generator(dev).manual_seed(rs._SYNTHETIC_RNG_BASE_SEED + fa._synthetic_step))  # fmt: skip
        f = _verdict_launch(rs, meta, head, fb, u)
        got["forced, ATOM's uniforms"] &= (
            same(atom(meta, head, fa), f) and fa._synthetic_step == fb._synthetic_step
        )
        fb._synthetic_step -= 1
        dr = _verdict_launch(rs, meta, head, fb)
        fb._synthetic_step -= 1
        got["drawn: re-run equal"] &= same(dr, _verdict_launch(rs, meta, head, fb))
        acc = dr[1][0].cpu()
        for r, n in enumerate(d):
            # positions 0, 1 always accepted (rate 1), 3+ never; position 2 at 0.51 (of those reaching it)
            got["drawn: certain rows"] &= (
                int(acc[r]) == n if n <= 2 else 2 <= int(acc[r]) <= 3
            )
            if n >= 3:
                frac[0] += 1
                frac[1] += int(acc[r]) == 3
    rate = frac[1] / max(frac[0], 1)
    got[f"drawn: position 2 accepted {rate:.2f} of {frac[0]} (0.51)"] = (
        abs(rate - 0.51) < 0.12
    )
    report(
        "[vf] verdict reference: == ATOM's selects + rejection (real, forced on ATOM's uniforms); "
        "in-kernel draws TP-uniform at the schedule's rates",
        all(got.values()),
        f"({got})",
    )


def check_post_verify():
    """[pv] The in-graph post-verify kernel (``_post_verify_kernel`` through ``PostVerify.launch`` on stand-ins of the
    hook, cache and step) against the eager path it replaces: the verdict (ATOM's rejection sampler with real
    acceptance; the verdict reference on the same seed with forced acceptance) and Engram's ``_commit_cursors``, at 1-4
    requests of 6 rows (head ids agreeing with the drafts up to a random row); the seed from the hook's step counter;
    the outputs are the forward's head output (``output``); the commit skips its launch at the outputs' own anchors
    and runs ATOM's at any other.
    """
    pvm = importlib.import_module("atom.models.deepseek_v41.mono.round.post_verify")
    rs = importlib.import_module("atom.model_ops.rejection_sampler")
    spec_mod = importlib.import_module(
        "atom.model_ops.attentions.deepseek_v41.speculative"
    )
    dev = torch.get_default_device()
    g = torch.Generator(device="cpu").manual_seed(23)
    i32 = {"dtype": torch.int32, "device": dev}
    rq, nslot, W, vocab = 6, 8, 4, 1000

    def world(nb, rates):
        rows = nb * rq
        seg = torch.randint(0, vocab, (rows + 1,), generator=g, device="cpu")
        head = torch.randint(0, vocab, (rows,), generator=g, device="cpu")
        for b in range(nb):  # the head agrees with the drafts up to a random row
            for j in range(int(torch.randint(0, rq, (1,), generator=g, device="cpu"))):
                head[b * rq + j] = seg[b * rq + j + 1]
        slots = torch.randperm(nslot, generator=g, device="cpu")[:nb]
        req = torch.zeros(ml.REQ_WORDS, **i32)
        req[ml.REQ_LIVE : ml.REQ_LIVE + nb] = (
            1  # (every request live; a padding request: ``check_post_verify_pad``)
        )
        hook = SimpleNamespace(head_ids=head.to(**i32), step_ids=seg.to(**i32), req=req,
                               step=torch.randint(0, 1000, (1,), generator=g, device="cpu").to(**i32),
                               runner=SimpleNamespace(rejection_sampler=rs.RejectionSampler(rates)))  # fmt: skip
        cache = SimpleNamespace(cursor=torch.randint(-1, 300, (nslot, W), generator=g, device="cpu").to(dev),
                                tentative_staging=SimpleNamespace(gpu=torch.randint(
                                    -1, 300, (nslot, rq, W), generator=g, device="cpu").to(dev)))  # fmt: skip
        step = SimpleNamespace(slots=slots.to(**i32), cu_seqlens_q=torch.arange(0, rows + 1, rq).to(**i32),
                               max_q_len=rq)  # fmt: skip
        meta = SimpleNamespace(input_ids=hook.step_ids, cu_num_draft_tokens=torch.arange(1, nb + 1, **i32) * (rq - 1),
                               target_logits_indices=torch.tensor([b * rq + j for b in range(nb) for j in range(rq - 1)],
                                                                  dtype=torch.long, device=dev),
                               bonus_logits_indices=torch.tensor([b * rq + rq - 1 for b in range(nb)], dtype=torch.long,
                                                                 device=dev),
                               num_spec_steps=rq - 1, num_draft_tokens_np=np.full(nb, rq - 1))  # fmt: skip
        return hook, cache, step, meta

    got = {"graph == eager": True, "seed from the step counter": True}
    for rates in (None, rs.acceptance_length_to_rates(3.51, 5)):
        for nb in (1, 2, 3, 4, 1, 4):
            widths = [nb * rq]
            split = {nb * rq: (nb, rq)}
            hook, cache, step, meta = world(nb, rates)
            cur0 = cache.cursor.clone()
            pv = pvm.PostVerify(widths, split, dev)
            seed0 = rs._SYNTHETIC_RNG_BASE_SEED + int(hook.step)
            pv.launch(hook, cache, step, nb * rq)
            got["seed from the step counter"] &= int(pv.used_seed) == seed0
            b = pv.bufs[nb * rq]
            # the eager path: ATOM's verdict (forced: the reference on the same seed), ATOM's commit
            sampler = rs.RejectionSampler(rates)
            if rates is None:
                t = torch.index_select(hook.head_ids, 0, meta.target_logits_indices)
                bonus = torch.index_select(hook.head_ids, 0, meta.bonus_logits_indices)
                logits = torch.nn.functional.one_hot(t.long(), vocab).float()
                out, vd = sampler.forward(meta, logits, bonus, target_token_ids=t)
            else:
                out, vd = _verdict_launch(rs, meta, hook.head_ids, sampler, seed=seed0)
            cur = cur0.clone()
            st = cache.tentative_staging.gpu
            spec_mod._commit_cursors[(nb,)](cur, st, vd[2], step.cu_seqlens_q[:nb], step.slots, cur.stride(0),
                                            st.stride(0), st.stride(1), WIDTH=W, BLOCK=4)  # fmt: skip
            got["graph == eager"] &= (torch.equal(b["out"], out) and torch.equal(b["verdict"], vd)
                                      and torch.equal(cache.cursor, cur))  # fmt: skip
    # the head output and the commit
    hook, cache, step, meta = world(2, None)
    pv = pvm.PostVerify([12, 6], {12: (2, 6), 6: (1, 6)}, dev)
    pv.launch(hook, cache, step, 12)
    out, vd = pv.output(12)
    got["the width's outputs are the head output"] = (
        out is pv.bufs[12]["out"] and vd is pv.bufs[12]["verdict"]
    )
    # the commit: at the outputs' anchors (the runner took them) none is launched (the tentative state released as
    # ``commit_tentative`` releases it); at any other anchors: ATOM's
    st = SimpleNamespace()
    pending = SimpleNamespace(step=st, staged_on_device=True)
    md = SimpleNamespace(cache=SimpleNamespace(pending=pending), step=st)
    got["taken step: no commit, pending released"] = (
        pv.committed(md, vd[2]) and md.cache.pending is None
    )
    md.cache.pending = pending
    got["other anchors: ATOM's commit"] = (
        not pv.committed(md, vd[2].clone()) and md.cache.pending is pending
    )
    report(
        "[pv] post-verify in the graph: == ATOM's verdict + commit (1-4 requests, real; forced on the reference's "
        "draws); the head output; the commit's skip",
        all(got.values()),
        f"({got})",
    )


def check_post_verify_pad(pad_slot: int):
    """The in-graph post-verify (``_post_verify_kernel``) on a bs-4 graph whose last request is padding (the step
    right after a request left: 3 live requests in slots 1, 3, 4; the padding request's slot entry 0, a freed slot,
    or 1, a live request's): the live requests' Engram cursors get their accepted prefix's candidate row, and the
    padding request writes no cursor (neither the freed slot's nor the live one it names). The GPU's pad scenario
    caught the padding request's commit into freed slot 0."""
    pv = _load(
        MONO / "round" / "post_verify.py",
        "_hi_post_verify_interp",
        interpret=_MODE["interpret"],
    )
    hb = ml

    nb, rq, W, nslot = 4, 6, 4, 6
    rows, s = nb * rq, rq - 1
    g = torch.Generator().manual_seed(5)
    i32 = {"dtype": torch.int32}
    seg = torch.randint(0, 900, (rows + 1,), generator=g, **i32)
    head = torch.randint(0, 900, (rows,), generator=g, **i32)
    head[0], head[1] = seg[1], seg[2]  # request 0 accepts two drafts
    slots = torch.tensor([1, 3, 4, pad_slot], **i32)
    live = torch.zeros(hb.REQ_WORDS, **i32)
    live[hb.REQ_LIVE : hb.REQ_LIVE + 3] = 1
    cursor = torch.randint(-1, 300, (nslot, W), generator=g, dtype=torch.int64)
    staging = torch.randint(-1, 300, (nb, rq, W), generator=g, dtype=torch.int64)
    starts = torch.tensor(
        [0, 6, 12, 18, 18], **i32
    )  # ATOM's cu_seqlens_q: the padding request has no rows
    out, verdict = torch.zeros(nb, rq, **i32), torch.zeros(3, nb, **i32)
    cu = torch.arange(1, nb + 1, **i32) * s
    target = torch.tensor([b * rq + j for b in range(nb) for j in range(s)])
    bonus = torch.tensor([b * rq + s for b in range(nb)])
    before = cursor.clone()
    pv._post_verify_kernel[(nb,)](
        out, verdict, cu, seg, target, bonus, head, out, torch.zeros(1, **i32), torch.zeros(1, dtype=torch.int64),
        cursor, cursor.stride(0), staging, staging.stride(0), staging.stride(1), starts, slots, live[hb.REQ_LIVE :],
        s, 0, WIDTH=W, BLOCK=W, SP=8, SYNTHETIC=False, SHADOW=False,
    )  # fmt: skip
    want = before.clone()
    for b in range(3):
        anchor = int(verdict[2, b])
        want[int(slots[b])] = staging[b, anchor - int(starts[b])]
    report(f"[pv] post-verify on a bs-4 graph with a padding request (its slot entry {pad_slot}): the live requests'"
           " cursors committed, no padding commit", torch.equal(cursor, want) and int(verdict[0, 0]) == 2,
           f"(rows {(cursor != want).any(1).nonzero().flatten().tolist()} differ; accepted {verdict[0].tolist()})")  # fmt: skip


def check_glue_folds():
    """[gf] The begin-step's expand row (``_expand_row``) of ATOM's lookup (``_masked_embedding_launcher``, ids in
    range, at the edges and outside it: -1, V, V + 3, zero rows) == ATOM's ``from_embeddings`` (every stream the row),
    bit for bit. The end-step's collapse blocks (``_collapse_chunk``, fp fusion off as the end-step launches it) ==
    ATOM's ``collapse_streams`` on the same streams and pre-mix, bit for bit (and whether fused multiply-adds would
    differ).
    """
    eh = importlib.import_module("atom.model_ops.embed_head")
    mhc = importlib.import_module("atom.model_ops.deepseek_v41.mhc")
    dev = torch.get_default_device()
    g = torch.Generator(device="cpu").manual_seed(31)
    V, H, NHC, rows = 500, 2048, 4, 24
    table = torch.randn(V, H, generator=g, device="cpu").to(torch.bfloat16).to(dev)
    ids = torch.tensor([0, V - 1, -1, V, V + 3] + torch.randint(0, V, (rows - 5,), generator=g, device="cpu").tolist(),
                       dtype=torch.int32, device=dev)  # fmt: skip
    want = eh._masked_embedding_launcher(ids, table, 0, V)
    h_in = torch.full((rows, NHC, H), 7.0, dtype=torch.bfloat16, device=dev)
    ml._expand_rows_kernel[(rows,)](
        want, want.stride(0), h_in, HIDDEN=H, BLOCK=1024, NHC=NHC
    )
    got = {"embedding expand == ATOM's lookup, every stream": torch.equal(h_in, want.unsqueeze(1).expand_as(h_in)),
           "ids outside the table: zero rows": not h_in[2:5].any()}  # fmt: skip
    gate = torch.rand(rows, 24, generator=g, device="cpu").to(dev)
    res = (
        torch.randn(rows, NHC, H, generator=g, device="cpu").to(torch.bfloat16).to(dev)
    )
    ref = mhc.collapse_streams(res, gate[:, :NHC].contiguous())
    out = torch.empty(rows, H, dtype=torch.bfloat16, device=dev)
    ml._collapse_rows_kernel[(rows, H // 1024)](gate, res, out, HIDDEN=H, BLOCK=1024, NHC=NHC, NMIX=24,
                                                enable_fp_fusion=False)  # fmt: skip
    got["collapse == ATOM's collapse_streams"] = torch.equal(out, ref)
    fused = torch.empty_like(out)
    ml._collapse_rows_kernel[(rows, H // 1024)](
        gate, res, fused, HIDDEN=H, BLOCK=1024, NHC=NHC, NMIX=24
    )
    report("[gf] glue folds: the begin-step's embedding expand == ATOM's; the end-step's collapse == ATOM's",
           all(got.values()), f"({got}; with fp fusion on the collapse would differ on "
                              f"{int((fused != ref).any(-1).sum())} of {rows} rows)")  # fmt: skip


def check_kv8_atom():
    """[2ka] The bridge's fp8 row export / import, compiled, against ATOM's own launches on the same bf16 rows:
    ``quantize_fp8`` (``fp8_groups``: the bytes ATOM's writers store) and ``dequantize_rows`` (its reader)."""
    from atom.model_ops.blockscale import quantize_fp8

    n, dim = 64, 512
    x = kv8_rows(n).to(torch.get_default_device())
    got = torch.zeros(n, ml.KV8_ROW_BYTES, dtype=torch.uint8)
    ml._kv8_store_kernel[(n,)](x, got, DIM=dim)
    values, scales = quantize_fp8(x)
    want = torch.cat([values.view(torch.uint8), scales.view(torch.uint8)], -1)
    report("[2ka] export == ATOM's quantize_fp8 bytes", torch.equal(got, want),
           f"({int((got != want).sum())} bytes differ)")  # fmt: skip
    report("[2ka] the torch model == ATOM's quantize_fp8 bytes", torch.equal(kv8_encode(x.cpu()), want.cpu()))
    back = torch.zeros(n, dim, dtype=torch.bfloat16)
    ml._kv8_load_kernel[(n,)](want, back, DIM=dim)
    report("[2ka] import == ATOM's dequantize_rows", torch.equal(back.float(), kv8_decode(want)))


# ---------------------------------------------------------------- the tests


@interpreted
@pytest.mark.parametrize("kv8", [False, True])
def test_rings(kv8):
    _run(check_rings, kv8)


@interpreted
def test_kv8_codec():
    _run(check_kv8_codec)


@interpreted
def test_req_words():
    _run(check_req_words)


@interpreted
def test_hc():
    _run(check_hc)


@interpreted
def test_engram_hist():
    _run(check_engram_hist)


@interpreted
@pytest.mark.parametrize("engram", [False, True])
def test_fused(engram):
    _run(check_fused, engram)


@interpreted
def test_fused_kv8():
    _run(check_fused, False, True)


@interpreted
@pytest.mark.parametrize("rp", [16, 128, 5])
def test_incremental(rp):
    _run(check_incremental, ((rp, 0),), rp == 16)


@interpreted
def test_batch_bridge():
    _run(check_batch_bridge)


@interpreted
@pytest.mark.parametrize("rp", [16, 128, 5])
def test_batch_incremental(rp):
    _run(check_batch_incremental, ((rp, 0),))


@interpreted
def test_bands():
    _run(check_bands)


@interpreted
@pytest.mark.parametrize("pad_slot", [0, 1])
def test_post_verify_pad(pad_slot):
    _run(check_post_verify_pad, pad_slot)


@interpreted
def test_draft_begin():
    _run(check_draft_begin)


@pytest.fixture
def on_gpu():
    """ATOM's own launches (AITER) on a GPU, the layout compiled."""
    pytest.importorskip("aiter")
    if INTERPRET or not torch.cuda.is_available():
        pytest.skip("ATOM's launches run on a GPU, compiled")
    old = torch.get_default_device()
    _MODE["interpret"] = False
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(old)
        _MODE["interpret"] = True


def test_engram_fold(on_gpu):
    _run(check_engram_fold)


def test_verdict_reference(on_gpu):
    _run(check_verdict_reference)


def test_post_verify(on_gpu):
    _run(check_post_verify)


def test_glue_folds(on_gpu):
    _run(check_glue_folds)


def test_kv8_atom(on_gpu):
    _run(check_kv8_atom)

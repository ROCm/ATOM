# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The hi programs' MXFP8 rows of ATOM's fp8 pool (``--kv_cache_dtype fp8``) on an MI355X, against ATOM's own
quantizer and reader: each device path of the ``kv8`` builds in a one-stage program of its own.

- the Compressor's entry store (``compressor._kv8_entry``: through ``paged.row_addr`` and the block table) and the
  DSpark context write's ring store (``draft._ring_store8``): the bytes ATOM's ``quantize_fp8`` stores for the same
  bf16 rows (``fp8_groups``: the amax floor, the ceil code, the exact reciprocal, the clamp, E4M3 RNE), and the
  Compressor's in-launch value (its ``out`` channel's) the one ATOM's ``dequantize_rows`` reads back;
- the sparse attention's gather (``mla_sparse._kv8_issue`` / ``_kv8_land``: a wave-uniform row, the chunk tasks')
  and its merge's loads (a lane's own row), and the draft's ring read (``draft_attn._ring_issue``'s words,
  ``paged.kv8_words``): the bf16 words ``dequantize_rows`` gives.

Rows reach every case of the encoding (scales 2^-40 .. 2^20, an all-zero group, +-0, the clamp, the rounding's ties:
``test_mono_hi_bridge.kv8_rows``). Skips without an MI355X (run: ``python -m pytest -q`` this file on one).
"""

import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("flydsl")
pytest.importorskip("aiter")

import flydsl.expr as fx
import torch
from flydsl.expr.typing import T

from atom.models.deepseek_v41.mono.hi.blocks.gfx950 import compressor
from atom.models.deepseek_v41.mono.hi.blocks.gfx950.attention import (
    mla_sparse,
)
from atom.models.deepseek_v41.mono.hi.device.gfx950 import (
    buffer_ops as bo,
)
from atom.models.deepseek_v41.mono.hi.device.gfx950 import paged
from atom.models.deepseek_v41.mono.hi.device.gfx950.ops import (
    ld_i32,
    rsrc,
    traced,
)
from atom.models.deepseek_v41.mono.hi.plan.program import (
    Program,
    Scenario,
    Stage,
)
from atom.models.deepseek_v41.mono.hi.v41 import draft

ROWS, HD, ROW8 = 64, 512, 528


_PAGED = SimpleNamespace(cache="ckv", ratio=1)  # (compressor._kv8_entry's plan fields)
_RING = SimpleNamespace(ring="ring")  # (draft._ring_store8's)


# ---------------------------------------------------------------- the one-stage programs (a task a row)


@traced
def _entry_task(k, t):
    """Thread c: channel c of row t (``x`` f32) -> ``compressor._kv8_entry`` (entry t of the request), its value."""
    c = k.tid
    v = fx.Float32(
        bo.buffer_load(rsrc(k.args["x"]), t * HD + c, vec_width=1, dtype=T.f32)
    )
    got = compressor._kv8_entry(k, _PAGED, 0, t, c, v, c >= 0)
    bo.buffer_store(got, rsrc(k.args["out"]), t * HD + c)


@traced
def _ring_task(k, t):
    """Thread tid < 256: the pair tid of row t (``x`` bf16 pairs) -> ``draft._ring_store8`` (ring row t)."""
    if k.tid < HD // 2:
        w = ld_i32(k.args["x"], t * (HD // 2) + k.tid)
        draft._ring_store8(k, _RING, t, w, k.tid >= 0)


def _store_words(k, t, words):
    bo.buffer_store(
        fx.Vector.from_elements(words, fx.Int32),
        rsrc(k.args["out"]),
        t * (HD // 2) + k.lane * 4,
    )


@traced
def _load_task(k, t, uni):
    """Wave 0, lane l: dims 8 l .. + 8 of row t by ``mla_sparse._kv8_issue`` / ``_kv8_land`` (``uni``: the chunk
    tasks' wave-uniform row; else the merge's lane-own row)."""
    if k.wave == 0:
        addr = fx.Int64(k.args["pool"]) + fx.Int64(t) * ROW8
        _store_words(
            k,
            t,
            mla_sparse._kv8_land(
                mla_sparse._kv8_issue(addr, k.lane * 8, uni), k.lane * 8
            ),
        )


@traced
def _ring_read_task(k, t):
    """Wave 0, lane l: ``draft_attn._ring_issue``'s conversion of row t (8 B and its group's code word)."""
    if k.wave == 0:
        r_ring = rsrc(k.args["pool"])
        at = t * (ROW8 // 4)
        v = fx.Vector(bo.buffer_load(r_ring, at + k.lane * 2, vec_width=2, dtype=T.i32))
        cw = bo.buffer_load(
            r_ring, at + HD // 4 + k.lane // 16, vec_width=1, dtype=T.i32
        )
        _store_words(
            k, t, paged.kv8_words(v[0], v[1], paged.kv8_code_of(cw, k.lane // 4))
        )


def build(name, emit, args):
    """The one-stage program of ``emit`` over ``ROWS`` tasks (compiled; a GPU only to launch it)."""
    from atom.models.deepseek_v41.mono.hi.runtime.kernel import MonoKernel

    return MonoKernel(
        Program(name, Scenario(1, ROWS), [Stage(name, ROWS, emit, args=args, base=0)])
    )


PROGRAMS = {
    "entry": (_entry_task, ("x", "out", "ckv", *paged.REQ)),
    "ring": (_ring_task, ("x", "ring")),
    "gather": (partial(_load_task, uni=True), ("pool", "out")),
    "merge": (partial(_load_task, uni=False), ("pool", "out")),
    "ring_read": (_ring_read_task, ("pool", "out")),
}


def _launch(name, args):
    emit, names = PROGRAMS[name]
    kernel = build(f"kv8_{name}", emit, names)
    kernel.launch(kernel.table(args), kernel.buffers())
    torch.cuda.synchronize()


# ---------------------------------------------------------------- the references


def _rows():
    sys.path.insert(0, str(Path(__file__).parent / "models" / "deepseek_v41"))
    from test_mono_hi_bridge import kv8_rows

    return kv8_rows(ROWS).cuda()


def _atom(x):
    """ATOM's fp8 pool rows of bf16 ``x`` [rows, 512] (``quantize_fp8``) and the values its reader takes."""
    from atom.model_ops.attentions.deepseek_v41.fp8_attention import dequantize_rows
    from atom.model_ops.blockscale import quantize_fp8

    values, scales = quantize_fp8(x)
    rows = torch.cat([values.view(torch.uint8), scales.view(torch.uint8)], -1)
    return rows, dequantize_rows(values, scales)


def _paged_args(pool):
    """The request words and a block table of one request whose entry e is the pool's row e (one page, ratio 1:
    ``paged.row_addr``)."""
    req = torch.zeros(paged.REQ_WORDS, dtype=torch.int32, device="cuda")
    req[paged.GEO], req[paged.GEO + 1] = 1, paged.PAGE  # table stride, page rows
    req[paged.BID] = 0
    return {
        "ckv": pool,
        "req": req,
        "ptab": torch.zeros(1, dtype=torch.int32, device="cuda"),
    }


@pytest.fixture(scope="module")
def mi355x():
    from atom.models.deepseek_v41.mono.hi.target import MI355X, device_arch

    if device_arch() != MI355X.arch:
        pytest.skip("an MI355X program")


# ---------------------------------------------------------------- the tests


def test_compressor_entry_is_atoms_row(mi355x):
    x = _rows()
    want, values = _atom(x)
    pool = torch.full((ROWS, ROW8), 0xAA, dtype=torch.uint8, device="cuda")
    out = torch.zeros(ROWS, HD, dtype=torch.float32, device="cuda")
    _launch("entry", {"x": x.float(), "out": out, **_paged_args(pool)})
    assert torch.equal(pool, want), f"{int((pool != want).sum())} bytes differ"
    assert torch.equal(out, values)


def test_ring_store_is_atoms_row(mi355x):
    x = _rows()
    want, _ = _atom(x)
    ring = torch.full((ROWS, ROW8), 0xAA, dtype=torch.uint8, device="cuda")
    _launch("ring", {"x": x.contiguous().view(torch.int32), "ring": ring})
    assert torch.equal(ring, want), f"{int((ring != want).sum())} bytes differ"


@pytest.mark.parametrize("name", ["gather", "merge", "ring_read"])
def test_loads_are_atoms_values(mi355x, name):
    """The sparse attention's gather (a wave-uniform row) and merge (a lane's own row), the draft's ring read: the
    bf16 words ATOM's reader's values are."""
    x = _rows()
    pool, values = _atom(x)
    out = torch.zeros(ROWS, HD // 2, dtype=torch.int32, device="cuda")
    _launch(name, {"pool": pool, "out": out})
    assert torch.equal(out.view(torch.bfloat16).float(), values)

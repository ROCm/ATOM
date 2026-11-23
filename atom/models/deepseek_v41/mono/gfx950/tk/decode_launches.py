# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Describes the decode-step launches of TileRT 0.1.7 (MIT, Copyright (c) TileRT-team), DeepSeek-V4.1-Flash on gfx950:
# the kernel launches its compiled host (libtilert_dsv41_rocm.so) captures into the two steady-state DSpark decode
# graphs, rebuilt from recordings of them.
"""The kernel launches of one DSpark decode step at TP8 (TileRT's mode 5: sampling MTP, target verify of 6 rows + 3
draft layers), as rules: per rank, 118 launches in order, each a kernel, grid, block, dynamic LDS and its kernarg bytes
as a list of 8-byte words (the last may be 4 B), every word a *spec* (``decode_launches.json``).

Word specs (``word``), JSON:

- ``v`` (int) / ``["r", a, b]``: a constant; ``a + b * rank``;
- ``["p", buf, off]``: this rank's buffer ``buf`` + ``off`` (``off`` may be ``["r", a, b]``);
- ``["q", owner, buf, off]``: rank ``owner``'s buffer (a peer pointer, P2P);
- ``["e", ch, n]``: the step epoch of channel ``ch``, call ``n``: ``0x10000 * (1 + parity) + (mode << 13) + (ch << 10)
  + n`` -- the only field that differs between the two graphs (parity 0 / 1, replayed alternately);
- ``["s", lo, hi]``: two 32-bit halves, each a spec (e.g. ``lo`` the rank, ``hi`` npes 8);
- ``["g", layer, "lo" | "hi"]``: Engram layer ``layer``'s table row range bound of this rank (``weights.ckpt``);
- ``["j", [h...]]`` / ``["J", [[h0, h1]...]]``: bytes the host left uninitialized (a struct's padding high half): one
  value per rank of the variant / per rank and parity, kept only so a check can compare whole kernargs.

A family (``families``) is the launches of one kernel and rank group sharing a word layout; its specs are templates
over the launch's layer ``L`` (``layers.N`` is layer N, ``mtp.M`` layer 40 + M): a name's ``{S+k}`` is layer L+k's
parameter scope (``layers.N`` / ``mtp.M``), ``{N+k}`` its index (the window cache), a number may be ``["L", a, b]`` =
``a + b * L`` or ``["M", [x...]]``, one per member. ``launches[i]`` lists launch i's variants: ``[family, member]``,
one per rank group (ranks 6 and 7 run other indexer instances than 0..5).

Buffers: ``param:`` / ``temp:`` / ``cache:`` / ``sym:`` as TileRT's ``dsv41_e2e_prepare_rank`` names them, plus what
the .so and the token tables allocate (``internal:`` / ``toktab:``, ``buffers`` gives sizes); a host passes each rank's
``{name: device address}``. The plan was derived from recordings of TileRT's two graphs and every recorded kernarg
rebuilt from it while porting (those tools stay on the development branch).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PLAN_FILE = Path(__file__).with_name("decode_launches.json")
NPES = 8
WORD = 8

# the epoch TileRT bakes into a captured step (host fn_871680): channel k's n-th call in the step
EPOCH_BASE = 0x10000


def epoch(parity: int, mode: int, ch: int, n: int) -> int:
    return EPOCH_BASE + (parity << 16) + (mode << 13) + (ch << 10) + n


# ---------------------------------------------------------------------------------------------------------------
# evaluation


_SCOPE_T = re.compile(r"\{S([+-]\d+)\}")
_INDEX_T = re.compile(r"\{N([+-]\d+)\}")


def _scope(n: int) -> str:
    return f"layers.{n}" if n < 40 else f"mtp.{n - 40}"


def _name(s: str, layer: int | None) -> str:
    if layer is None:
        return s
    s = _SCOPE_T.sub(lambda m: _scope(layer + int(m.group(1))), s)
    return _INDEX_T.sub(lambda m: str(layer + int(m.group(1))), s)


def instantiate(spec, layer: int | None, member: int):
    """A family template's spec for one member (layer ``layer``)."""
    if isinstance(spec, str):
        return _name(spec, layer)
    if not isinstance(spec, list):
        return spec
    if spec and spec[0] == "L":
        return spec[1] + spec[2] * layer
    if spec and spec[0] == "M":
        return spec[1][member]
    if spec and spec[0] in ("j", "J"):
        return spec
    return [instantiate(x, layer, member) for x in spec]


def _rows(layer: int, which: str, rank: int) -> int:
    from .weights.ckpt import engram_hash_consts, engram_row_range

    global _ROWS
    if _ROWS is None:
        _ROWS = engram_hash_consts()
    lo, hi = engram_row_range(_ROWS, layer, rank)
    return lo if which == "lo" else hi


_ROWS = None


def word(
    spec, rank: int, parity: int, addr: list[dict], mode: int, ranks: list[int]
) -> int:
    """A word's value on ``rank`` in graph ``parity``; ``addr[r]``: rank r's ``{buffer: address}``."""
    if isinstance(spec, int):
        return spec
    tag = spec[0]
    if tag == "r":
        return spec[1] + spec[2] * rank
    if tag == "p":
        return addr[rank][spec[1]] + word(spec[2], rank, parity, addr, mode, ranks)
    if tag == "q":
        return addr[spec[1]][spec[2]] + word(spec[3], rank, parity, addr, mode, ranks)
    if tag == "e":
        return epoch(parity, mode, spec[1], spec[2])
    if tag == "s":
        lo = word(spec[1], rank, parity, addr, mode, ranks)
        hi = word(spec[2], rank, parity, addr, mode, ranks)
        return (lo & 0xFFFFFFFF) | (hi & 0xFFFFFFFF) << 32
    if tag == "g":
        return _rows(spec[1], spec[2], rank)
    if tag == "j":
        return spec[1][ranks.index(rank)]
    if tag == "J":
        return spec[1][ranks.index(rank)][parity]
    raise ValueError(f"unknown word spec {spec!r}")


def load(path: str | Path | None = None) -> dict:
    with open(path or PLAN_FILE) as f:
        return json.load(f)


def rank_launches(plan: dict, rank: int) -> list[dict]:
    """Rank ``rank``'s launches in order: ``kernel``, ``grid``, ``block``, ``shm``, ``nbytes``, ``words`` (specs),
    ``ranks`` (the variant's rank group)."""
    out = []
    for variants in plan["launches"]:
        for fi, member in variants:
            fam = plan["families"][fi]
            if rank in fam["ranks"]:
                layer = fam["L"][member]
                out.append(
                    {
                        "kernel": plan["kernels"][fam["kernel"]],
                        "grid": fam["grid"],
                        "block": fam["block"],
                        "shm": fam["shm"],
                        "nbytes": fam["nbytes"],
                        "ranks": fam["ranks"],
                        "layer": layer,
                        "words": [instantiate(w, layer, member) for w in fam["words"]],
                    }
                )
                break
        else:
            raise KeyError(f"no variant for rank {rank}")
    return out


def kernargs(
    launch: dict, rank: int, parity: int, addr: list[dict], mode: int
) -> bytes:
    """A launch's explicit kernarg bytes (``nbytes``) on ``rank`` in graph ``parity``."""
    out = bytearray()
    for k, spec in enumerate(launch["words"]):
        n = min(WORD, launch["nbytes"] - k * WORD)
        v = word(spec, rank, parity, addr, mode, launch["ranks"]) & ((1 << (8 * n)) - 1)
        out += v.to_bytes(n, "little")
    return bytes(out)


def buffers_used(plan: dict) -> dict[str, set]:
    """``{buffer: {"own" | "peer"}}`` over every spec of the plan."""
    used: dict[str, set] = {}

    def walk(s, layer, member):
        if isinstance(s, list) and s:
            if s[0] == "p" and isinstance(s[1], str):
                used.setdefault(_name(s[1], layer), set()).add("own")
            elif s[0] == "q" and isinstance(s[2], str):
                used.setdefault(_name(s[2], layer), set()).add("peer")
            if s[0] in ("j", "J"):
                return
            for x in s:
                walk(x, layer, member)

    for fam in plan["families"]:
        for m, layer in enumerate(fam["L"]):
            for w in fam["words"]:
                walk(instantiate(w, layer, m), layer, m)
    return used

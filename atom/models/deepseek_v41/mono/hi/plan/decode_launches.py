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
``{name: device address}``. ``derive`` builds the plan from a recording, ``check`` rebuilds every recorded kernarg.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
from collections import Counter
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
    from ..weights.ckpt import engram_hash_consts, engram_row_range

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


# ---------------------------------------------------------------------------------------------------------------
# recordings (tkshim launch records + TileRT's name -> address maps)


def read_recording(launches_gz: str | Path, maps_gz: str | Path):
    """``(steps, maps)``: ``steps[(rank, capture)]`` the capture's launch records in order (capture 0 / 1: a rank's
    first / second recorded graph), ``maps[rank]`` its ``[name, ptr, nbytes, dtype, shape]`` entries.
    """
    with gzip.open(launches_gz, "rt") as f:
        recs = [json.loads(line) for line in f]
    caps: dict[int, list] = {}
    for r in recs:
        if r.get("ev") == "begin":
            caps.setdefault(r["dev"], []).append(r["cap"])
    steps: dict = {}
    for r in recs:
        if "k" in r:
            steps.setdefault((r["dev"], caps[r["dev"]].index(r["cap"])), []).append(r)
    for v in steps.values():
        v.sort(key=lambda r: r["seq"])
    with gzip.open(maps_gz, "rt") as f:
        m = json.load(f)
    maps = {d: m[f"addr_rank{d}.json"] for d in range(NPES)}
    return steps, maps


def recorded_addr(plan: dict, steps: dict, maps: dict) -> list[dict]:
    """Each rank's ``{buffer: address}``: the maps' names, and the internal buffers read at their ``anchor`` words (the
    maps do not list what the .so allocates)."""
    addr = [{name: p for name, p, n, *_ in maps[d] if p and n} for d in range(NPES)]
    for name, b in plan["buffers"].items():
        if "anchor" in b:
            li, wo = b["anchor"]
            for d in range(NPES):
                a = bytes.fromhex(steps[(d, 0)][li]["a"])
                addr[d][name] = int.from_bytes(a[wo : wo + 8], "little")
    return addr


# ---------------------------------------------------------------------------------------------------------------
# derivation from a recording

# the .so's own allocations and the token tables: (bytes, anchor (launch, kernarg offset), what)
INTERNAL = {
    "internal:tail_workspace": (
        0x334300,
        (26, 0x1C8),
        "mega_afm_tail TailArgs+0x00, per-device tensor",
    ),
    "internal:next_mhc": (0x23000, (26, 0x208), "TailArgs+0x40, dsv41_mega_tail_alloc"),
    "internal:mhcf": (
        0x69000,
        (17, 0x128),
        "mega_afm MhcF workspace, e2e prepare tensor",
    ),
    "internal:tail_4096": (
        0x1000,
        (26, 0x238),
        "TailArgs+0x70, 4 KiB uncached mailbox",
    ),
    "toktab:mtp.0.q": (
        129280 * 4096 * 2,
        (0, 0xC0),
        "toktab gather<8,0> table A (bf16 [V, 8 x 512])",
    ),
    "toktab:mtp.0.kv": (129280 * 512 * 2, (0, 0xC8), "table B (bf16 [V, 512])"),
    "toktab:mtp.0.hc": (129280 * 24 * 4, (0, 0xD0), "table C (f32 [V, 24])"),
    "toktab:layers.0.q": (129280 * 4096 * 2, (16, 0xC0), "toktab gather<8,6> table A"),
    "toktab:layers.0.kv": (129280 * 512 * 2, (16, 0xC8), "table B"),
    "toktab:layers.0.hc": (129280 * 24 * 4, (16, 0xD0), "table C"),
}
# initial contents (TileRT e2e_graph._temps, end2end._alloc_caches, prepare); a buffer not listed starts zeroed
INIT = {
    "temp:latent_idx": "-1",
    "temp:topk": "-1",
    "temp:aux1": "-1",
    "temp:blk_pos": "-1",
    "temp:vpos": "-1",
    "temp:dsp_pos": "-1",
    "temp:P": "[:, 0] = 1.0",
    "temp:samp_params": "pack_params(SamplingParams()) = [0, 0x3f8000003f800000, 0, 0]",
    "cache:state.2": "[:, 1] = -inf",
    "cache:state.8": "[:, 1] = -inf",
    "cache:state.14": "[:, 1] = -inf",
    "cache:index_k.2.1": "127",
    "cache:index_k.8.1": "127",
    "cache:index_k.14.1": "127",
    "cache:index_k.20.1": "127",
    "internal:tail_4096": "hipMemset 0, hipDeviceMallocUncached",
    "internal:tail_workspace": "the .so's (not recorded)",
    "internal:next_mhc": "the .so's (not recorded)",
    "internal:mhcf": "the .so's (not recorded)",
    "toktab:mtp.0.q": "toktab.build_table (per-token q / kv / mHC rows of the scope)",
    "toktab:layers.0.q": "toktab.build_table",
}
_PTR_LO, _PTR_HI = 0x7000_0000_0000, 0x8000_0000_0000
_LAYER_REF = re.compile(r"param:(layers|mtp)\.(\d+)\.|cache:win\.(\d+)\.")


class _Ranges:
    def __init__(self, spans: dict[str, tuple[int, int]]):
        self.e = sorted((p, p + n, name) for name, (p, n) in spans.items())
        self.starts = [x[0] for x in self.e]
        self.maxlen = max(t - s for s, t, _ in self.e)

    def find(self, v: int):
        """The smallest range holding ``v``: (name, offset) or None."""
        best = None
        j = bisect.bisect_right(self.starts, v) - 1
        while j >= 0 and self.e[j][0] > v - self.maxlen - 1:
            s, t, name = self.e[j]
            if s <= v < t and (best is None or t - s < best[1] - best[0]):
                best = (s, t, name)
            j -= 1
        return None if best is None else (best[2], v - best[0])


def _lin(vs: list[int], rs: list[int]):
    if len(set(vs)) == 1:
        return vs[0]
    b = (vs[1] - vs[0]) // (rs[1] - rs[0])
    a = vs[0] - b * rs[0]
    return ["r", a, b] if all(v == a + b * r for v, r in zip(vs, rs)) else None


class _Deriver:
    def __init__(self, steps: dict, maps: dict, mode: int):
        self.steps, self.mode = steps, mode
        spans = [
            {name: (p, n) for name, p, n, *_ in maps[d] if p and n} for d in range(NPES)
        ]
        for name, (nb, (li, wo), _) in INTERNAL.items():
            for d in range(NPES):
                a = bytes.fromhex(steps[(d, 0)][li]["a"])
                spans[d][name] = (int.from_bytes(a[wo : wo + 8], "little"), nb)
        self.rng = [_Ranges(s) for s in spans]
        self.rows = {
            (layer, w, d): _rows(layer, w, d)
            for layer in (1, 14)
            for w in ("lo", "hi")
            for d in range(NPES)
        }

    def vals(self, i, o, n, rs):
        out = {}
        for d in rs:
            for c in (0, 1):
                a = bytes.fromhex(self.steps[(d, c)][i]["a"])
                out[(d, c)] = int.from_bytes(a[o : o + n], "little")
        return out

    def ptr(self, d, v):
        if not _PTR_LO <= v < _PTR_HI:
            return None
        for e in [d] + [x for x in range(NPES) if x != d]:
            h = self.rng[e].find(v)
            if h:
                return (e, *h)
        return None

    def val(self, V, rs):
        if len(set(V.values())) == 1:
            return next(iter(V.values()))
        if all(V[(d, 0)] == V[(d, 1)] for d in rs):
            vs = [V[(d, 0)] for d in rs]
            lin = _lin(vs, rs)
            if lin is not None:
                return lin
            for layer in (1, 14):
                for w in ("lo", "hi"):
                    if all(self.rows[(layer, w, d)] == v for d, v in zip(rs, vs)):
                        return ["g", layer, w]
            return None
        eps = set()
        for (d, c), v in V.items():
            x = v - epoch(c, self.mode, 0, 0)
            if not 0 <= x < 8 << 10:
                return None
            eps.add((x >> 10, x & 1023))
        return ["e", *eps.pop()] if len(eps) == 1 else None

    def spec(self, i, o, n, rs):
        V = self.vals(i, o, n, rs)
        capinv = all(V[(d, 0)] == V[(d, 1)] for d in rs)
        if n == WORD and capinv:
            P = [self.ptr(d, V[(d, 0)]) for d in rs]
            if all(P) and len({p[1] for p in P}) == 1:
                off = _lin([p[2] for p in P], rs)
                owners = [p[0] for p in P]
                if off is not None and owners == rs:
                    return ["p", P[0][1], off]
                if off is not None and len(set(owners)) == 1:
                    return ["q", owners[0], P[0][1], off]
        s = self.val(V, rs)
        lo = (
            self.val({k: v & 0xFFFFFFFF for k, v in V.items()}, rs)
            if n == WORD
            else None
        )
        hi = self.val({k: v >> 32 for k, v in V.items()}, rs) if n == WORD else None
        if s is not None and (
            isinstance(s, int) or lo is None or hi is None or hi == 0 or lo == 0
        ):
            return s  # a constant, or one field: no packing
        if lo is not None and hi is not None:
            return ["s", lo, hi]  # two 32-bit fields
        if lo is not None:  # a constant low half under stale high bytes
            if capinv:
                return ["s", lo, ["j", [V[(d, 0)] >> 32 for d in rs]]]
            return ["s", lo, ["J", [[V[(d, 0)] >> 32, V[(d, 1)] >> 32] for d in rs]]]
        return ["J", [[V[(d, 0)], V[(d, 1)]] for d in rs]]


def _refs(spec, out: set):
    if isinstance(spec, str):
        for m in _LAYER_REF.finditer(spec):
            if m.group(3) is not None:
                out.add(int(m.group(3)))
            else:
                out.add(int(m.group(2)) + (40 if m.group(1) == "mtp" else 0))
    elif isinstance(spec, list) and not (spec and spec[0] in ("j", "J")):
        for x in spec:
            _refs(x, out)


def _relative(spec, layer):
    if isinstance(spec, str):
        if layer is None:
            return spec

        def scope(m):
            if m.group(3) is not None:
                return f"cache:win.{{N{int(m.group(3)) - layer:+d}}}."
            n = int(m.group(2)) + (40 if m.group(1) == "mtp" else 0)
            return f"param:{{S{n - layer:+d}}}."

        return _LAYER_REF.sub(scope, spec)
    if isinstance(spec, list) and not (spec and spec[0] in ("j", "J")):
        return [_relative(x, layer) for x in spec]
    return spec


def _skeleton(spec):
    if isinstance(spec, int):
        return "#"
    if isinstance(spec, list):
        return (
            spec[0] if spec and spec[0] in ("j", "J") else [_skeleton(x) for x in spec]
        )
    return spec


def _merge(items: list, layers: list):
    first = items[0]
    if all(x == first for x in items):
        return first
    if (
        all(isinstance(x, int) for x in items)
        and None not in layers
        and len(set(layers)) > 1
    ):
        l1 = next(x for x in layers if x != layers[0])
        v1 = items[layers.index(l1)]
        if (v1 - first) % (l1 - layers[0]) == 0:
            b = (v1 - first) // (l1 - layers[0])
            a = first - b * layers[0]
            if all(v == a + b * x for v, x in zip(items, layers)):
                return ["L", a, b]
    if (
        all(
            isinstance(x, list) and x and len(x) == len(first) and x[0] == first[0]
            for x in items
        )
        and isinstance(first[0], str)
        and first[0] not in ("j", "J")
    ):
        return [first[0]] + [
            _merge([x[k] for x in items], layers) for k in range(1, len(first))
        ]
    return ["M", items]


def derive(steps: dict, maps: dict, mode: int = 5) -> dict:
    """The plan of a recording (all 8 ranks, both graphs, same launch sequence)."""
    dv = _Deriver(steps, maps, mode)
    n_launch = len(steps[(0, 0)])
    fams: dict = {}
    order = []
    for i in range(n_launch):
        groups: dict = {}
        for d in range(NPES):
            r0, r1 = steps[(d, 0)][i], steps[(d, 1)][i]
            key = (
                r0["k"],
                tuple(r0["g"]),
                tuple(r0["b"]),
                r0["shm"],
                len(r0["a"]) // 2,
            )
            assert key == (
                r1["k"],
                tuple(r1["g"]),
                tuple(r1["b"]),
                r1["shm"],
                len(r1["a"]) // 2,
            ), (i, d)
            groups.setdefault(key, []).append(d)
        row = []
        for (k, g, b, shm, nb), rs in groups.items():
            words = [dv.spec(i, o, min(WORD, nb - o), rs) for o in range(0, nb, WORD)]
            refs: set = set()
            _refs(words, refs)
            layer = min(refs) if refs else None
            words = _relative(words, layer)
            key = json.dumps([k, g, b, shm, nb, rs, _skeleton(words)])
            f = fams.setdefault(
                key, {"head": (k, g, b, shm, nb, rs), "members": [], "L": []}
            )
            f["members"].append(words)
            f["L"].append(layer)
            row.append([list(fams).index(key), len(f["members"]) - 1])
        order.append(row)
    kernels = sorted({f["head"][0] for f in fams.values()})
    families = []
    for f in fams.values():
        k, g, b, shm, nb, rs = f["head"]
        words = [
            _merge([m[w] for m in f["members"]], f["L"])
            for w in range(len(f["members"][0]))
        ]
        families.append(
            {
                "kernel": kernels.index(k),
                "grid": list(g),
                "block": list(b),
                "shm": shm,
                "nbytes": nb,
                "ranks": rs,
                "L": f["L"],
                "words": words,
            }
        )
    plan = {
        "mode": mode,
        "npes": NPES,
        "kernels": kernels,
        "buffers": {},
        "families": families,
        "launches": order,
    }
    ref = {name: (n, dt, shape) for name, p, n, dt, shape in maps[0]}
    for name in sorted(buffers_used(plan)):
        if name in INTERNAL:
            nb, anchor, what = INTERNAL[name]
            plan["buffers"][name] = {"nbytes": nb, "anchor": list(anchor), "what": what}
        elif not name.startswith("param:"):
            n, dt, shape = ref[name]
            plan["buffers"][name] = {
                "nbytes": n,
                "dtype": dt.replace("torch.", ""),
                "shape": shape,
            }
        else:
            continue
        if name in INIT:
            plan["buffers"][name]["init"] = INIT[name]
    return plan


def dump(plan: dict, path: str | Path) -> None:
    """``plan`` as JSON, one family / launch per line."""
    sep = (",", ":")
    lines = ["{"]
    for key in ("mode", "npes"):
        lines.append(f'"{key}":{json.dumps(plan[key])},')
    for key in ("kernels", "families", "launches"):
        items = plan[key]
        lines.append(f'"{key}":[')
        lines += [
            json.dumps(x, separators=sep) + ("," if k + 1 < len(items) else "")
            for k, x in enumerate(items)
        ]
        lines.append("],")
    lines.append('"buffers":{')
    names = list(plan["buffers"])
    lines += [
        f"{json.dumps(n)}:{json.dumps(plan['buffers'][n], separators=sep)}"
        + ("," if k + 1 < len(names) else "")
        for k, n in enumerate(names)
    ]
    lines.append("}}")
    Path(path).write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# check


def _kind(spec) -> str:
    if isinstance(spec, int) or spec[0] == "r":
        return "const"
    return {
        "p": "ptr",
        "q": "peer_ptr",
        "e": "epoch",
        "s": "split",
        "g": "rank_rows",
        "j": "junk",
        "J": "junk",
    }[spec[0]]


def check(plan: dict, steps: dict, maps: dict) -> dict:
    """Rebuild every recorded launch (kernel, grid, block, shm, kernarg bytes) of all ranks and both graphs."""
    addr = recorded_addr(plan, steps, maps)
    res = {
        "launches": 0,
        "launches_exact": 0,
        "words": 0,
        "words_exact": 0,
        "diffs": [],
        "kinds": Counter(),
    }
    for (d, c), recs in sorted(steps.items()):
        plan_l = rank_launches(plan, d)
        assert len(plan_l) == len(recs), (d, c, len(plan_l), len(recs))
        for i, (pl, r) in enumerate(zip(plan_l, recs)):
            res["launches"] += 1
            head = (pl["kernel"], pl["grid"], pl["block"], pl["shm"]) == (
                r["k"],
                r["g"],
                r["b"],
                r["shm"],
            )
            got = kernargs(pl, d, c, addr, plan["mode"])
            want = bytes.fromhex(r["a"])
            ok = head and got == want
            res["launches_exact"] += ok
            for k, spec in enumerate(pl["words"]):
                res["words"] += 1
                same = got[k * WORD : (k + 1) * WORD] == want[k * WORD : (k + 1) * WORD]
                res["words_exact"] += same
                if c == 0:
                    sk = _kind(spec)
                    if sk == "split":
                        sk = "split:" + "/".join(_kind(x) for x in spec[1:])
                    res["kinds"][sk] += 1
                if not same:
                    res["diffs"].append((d, c, i, k * WORD))
            if not head:
                res["diffs"].append((d, c, i, "head"))
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--launches", required=True, help="launches_cap12_13.jsonl.gz")
    ap.add_argument("--maps", required=True, help="addr_maps.json.gz")
    ap.add_argument("--derive", help="write the plan derived from the recording here")
    ap.add_argument("--plan", help="plan to check (default the packaged one)")
    a = ap.parse_args(argv)
    steps, maps = read_recording(a.launches, a.maps)
    if a.derive:
        dump(derive(steps, maps), a.derive)
    plan = load(a.plan or a.derive)
    res = check(plan, steps, maps)
    print(
        f"launches {res['launches_exact']}/{res['launches']} exact, words {res['words_exact']}/{res['words']} exact"
    )
    print(
        "word kinds (graph 0, summed over the 8 ranks):",
        dict(sorted(res["kinds"].items())),
    )
    for x in res["diffs"][:50]:
        print("DIFF", x)


if __name__ == "__main__":
    main()

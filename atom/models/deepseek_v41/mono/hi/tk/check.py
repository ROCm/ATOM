# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks of the TileRT-derived decode host (``tk``) against a TileRT recording (``plan.decode_launches``' inputs),
TileRT's code objects and our swap table; no GPU.

- ``blobs``: ``host.blobs`` over the recorded address map (``decode_launches.recorded_addr``) rebuilds every recorded
  launch of all ranks and both graphs (kernel, grid, block, LDS, kernarg bytes); ``junk="zero"`` differs only in the
  plan's junk words.
- ``sizes``: ``alloc``'s specs (TileRT's formulas) against the address maps on every rank: every ``temp:`` /
  ``cache:`` / ``sym:`` name, bytes, dtype and shape; every internal / token-table pointer of the plan inside its
  buffer; every parameter the plan names packed by ``weights.params`` (the TileRT fingerprints' names and sizes).
- ``tables``: ``alloc.sym_table`` against TileRT's ``end2end.sym_table`` (when the wheel is given); the arena's layout;
  ``Buffers.address_map`` over fake device addresses: each table entry is rank v's family, each kernarg peer pointer
  rank j's ``sym:indexer``; every plan word naming a ``tab_*`` points at its entry 0 and, where it is a kernel
  parameter of its own, that parameter is declared ``void* const*`` (the demangled symbol, ``c++filt``).
- ``code``: every plan kernel in TileRT's code objects, its explicit kernarg layout ending at the plan's blob size and
  equal to ``kargs.tsv``'s; every swap line's function in its code object, with the line's layout.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from ..plan import decode_launches as DL
from . import alloc, codeobj, host


def check_blobs(plan, steps, maps) -> dict:
    addr = DL.recorded_addr(plan, steps, maps)
    res = Counter()
    for (d, c), recs in sorted(steps.items()):
        launches = DL.rank_launches(plan, d)
        for junk in ("recorded", "zero"):
            got = host.blobs(plan, d, c, addr, junk)
            for pl, blob, r in zip(launches, got, recs):
                same_head = (pl["kernel"], pl["grid"], pl["block"], pl["shm"]) == (
                    r["k"],
                    r["g"],
                    r["b"],
                    r["shm"],
                )
                res[f"{junk}:launches"] += 1
                res[f"{junk}:launches_exact"] += same_head and blob == bytes.fromhex(
                    r["a"]
                )
                want = bytes.fromhex(r["a"])
                for o in range(0, len(want), 8):
                    res[f"{junk}:words"] += 1
                    res[f"{junk}:words_exact"] += blob[o : o + 8] == want[o : o + 8]
    return dict(res)


def check_sizes(plan, maps, paramsum: dict | None) -> dict:
    spec = {
        s.name: s
        for s in alloc.specs()
        if not s.name.startswith(("internal:", "toktab:"))
    }
    res = Counter()
    bad = []
    for d in range(alloc.NPES):
        seen = set()
        for name, ptr, nbytes, dtype, shape in maps[d]:
            if name.startswith("param:"):
                continue
            if name.startswith("sym:"):
                ok = alloc.SYM_BYTES[name[4:]] == nbytes
            else:
                s = spec.get(name)
                ok = s is not None and (
                    s.nbytes,
                    "torch." + s.dtype,
                    list(s.shape),
                ) == (nbytes, dtype, shape)
            seen.add(name)
            res["named"] += 1
            res["named_match"] += ok
            if not ok:
                bad.append((d, name, nbytes, dtype, shape))
        res["spec_names_missing_in_map"] += len(set(spec) - seen)
    # the plan's internal / token-table pointers inside their buffers
    sizes = {s.name: s.nbytes for s in alloc.internal_specs() + alloc.toktab_specs()}
    far: dict = {}

    def walk(s, rank):
        if isinstance(s, list) and s and s[0] in ("p", "q"):
            name, off = (s[1], s[2]) if s[0] == "p" else (s[2], s[3])
            if name in sizes:
                o = off if isinstance(off, int) else off[1] + off[2] * rank
                far[name] = max(far.get(name, 0), o)
        elif isinstance(s, list) and not (s and s[0] in ("j", "J")):
            for x in s:
                walk(x, rank)

    for d in range(alloc.NPES):
        for launch in DL.rank_launches(plan, d):
            for w in launch["words"]:
                walk(w, d)
    res["internal_used"] = len(far)
    res["internal_inside"] = sum(far[n] < sizes[n] for n in far)
    res["plan_internal_bytes_match"] = sum(
        plan["buffers"][n]["nbytes"] == sizes[n] for n in far
    )
    if paramsum is not None:
        used = [n for n in DL.buffers_used(plan) if n.startswith("param:")]
        res["plan_params"] = len(used)
        res["plan_params_packed"] = sum(n in paramsum["0"] for n in used)
        for d in range(alloc.NPES):
            mp = {
                name: nbytes
                for name, ptr, nbytes, *_ in maps[d]
                if name.startswith("param:") and nbytes
            }
            res["map_params"] += len(mp)
            res["map_params_size_match"] += sum(
                paramsum[str(d)].get(n, [None])[0] == b for n, b in mp.items()
            )
    return {**res, "mismatches": bad[:20]}


def _params_of(symbol: str) -> list[str] | None:
    """The parameter types of a kernel, demangled (``c++filt``); None without it."""
    if not shutil.which("c++filt"):
        return None
    s = subprocess.run(
        ["c++filt", symbol], capture_output=True, text=True, check=True
    ).stdout.strip()
    depth, start = (
        0,
        None,
    )  # the parameter list: the "(...)" outside every template argument list
    for i, ch in enumerate(s):
        if ch in "<(":
            if ch == "(" and depth == 0:
                start = i
            depth += 1
        elif ch in ">)":
            depth -= 1
    if start is None:
        return None
    args = s[start + 1 : -1]
    out, depth, cur = [], 0, ""
    for ch in args:
        if ch in "<(":
            depth += 1
        elif ch in ">)":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return out + ([cur.strip()] if cur.strip() else [])


def check_tables(
    plan, maps, wheel: Path | None, table: codeobj.CodeTable | None
) -> dict:
    res = Counter()
    rng = random.Random(0)
    if wheel is not None:
        e2e = _tilert_module(wheel, "tilert.models.ds_v4_1_flash.end2end")
        for fam in alloc.TABLES:
            for n in (8, 4, 2):
                ptrs = [rng.randrange(1, 1 << 47) for _ in range(n)]
                res["sym_table"] += 1
                theirs = e2e.sym_table(fam, ptrs, "cpu").tolist()
                res["sym_table_match"] += theirs == alloc.sym_table(fam, ptrs)
    offs, total = alloc.sym_layout()
    spans = sorted((offs[f], offs[f] + alloc.SYM_BYTES[f]) for f in alloc.TABLES)
    res["arena_aligned_disjoint"] = int(
        all(a % alloc.ALIGN_SYM == 0 for a, _ in spans)
        and all(b <= c for (_, b), (c, _) in itertools.pairwise(spans))
    )
    res["arena_bytes"] = total
    # address_map over fake addresses, for every rank
    for rank in range(alloc.NPES):
        b = object.__new__(alloc.Buffers)
        b.rank, b.sym_offsets = rank, offs
        b.peer_bases = [(0x7F00 + 0x10 * r) << 32 for r in range(alloc.NPES)]
        b.tensors = {
            s.name: _FakeTensor(0x7E00 << 32 | i << 24)
            for i, s in enumerate(alloc.specs())
        }
        addr = b.address_map()
        for f in alloc.TABLES:
            ptrs = [b.peer_bases[v] + offs[f] for v in range(alloc.NPES)]
            res["table_entries_are_peers"] += all(
                e == addr[v][f"sym:{f}"] for v, e in enumerate(alloc.sym_table(f, ptrs))
            )
        for launch in DL.rank_launches(plan, rank):
            for w in launch["words"]:
                if isinstance(w, list) and w[0] == "q":
                    off = w[3] if isinstance(w[3], int) else w[3][1] + w[3][2] * rank
                    res["peer_words"] += 1
                    want = b.peer_bases[w[1]] + offs[w[2][4:]] + off
                    res["peer_words_ok"] += (
                        DL.word(w, rank, 0, addr, plan["mode"], launch["ranks"]) == want
                    )
    ref = {name: (nbytes, dtype, shape) for name, ptr, nbytes, dtype, shape in maps[0]}
    res["tab_shapes_ok"] = sum(
        ref[f"temp:tab_{f}"] == (64, "torch.int64", [8])
        for f in alloc.TABLES[:7] + ("sample",)
    )
    # every word naming a table: entry 0, and its own parameter's type
    types: dict = {}
    for d in range(alloc.NPES):
        for launch in DL.rank_launches(plan, d):
            lay = (
                table.entries[launch["kernel"]].layout
                if table and launch["kernel"] in table.entries
                else None
            )
            for k, w in enumerate(launch["words"]):
                if isinstance(w, list) and w[0] == "p" and w[1].startswith("temp:tab_"):
                    res["tab_words"] += 1
                    res["tab_words_entry0"] += w[2] == 0
                    if lay is None:
                        continue
                    hit = [i for i, (o, n) in enumerate(lay) if o == 8 * k and n == 8]
                    if not hit:
                        res["tab_words_in_struct"] += 1
                        continue
                    if launch["kernel"] not in types:
                        types[launch["kernel"]] = _params_of(launch["kernel"])
                    t = types[launch["kernel"]]
                    if t is None or len(t) != len(lay):
                        res["tab_words_untyped"] += 1
                        continue
                    res["tab_words_param"] += 1
                    res["tab_words_void_ptr_ptr"] += t[hit[0]].replace(" ", "") in (
                        "void*const*",
                        "void**",
                    )
    return dict(res)


class _FakeTensor:
    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr


def _tilert_module(wheel: Path, name: str):
    """A module of the unzipped TileRT wheel ``wheel`` (its ``tilert/``), its parent packages stubbed (TileRT's
    ``__init__`` loads its .so)."""
    import importlib
    import sys
    import types

    parts = name.split(".")
    for k in range(1, len(parts)):
        pkg = ".".join(parts[:k])
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(wheel.joinpath(*parts[1:k]))]
            sys.modules[pkg] = m
    return importlib.import_module(name)


def check_code(
    plan, co_dir: Path, kargs_tsv: Path | None, swap_tsv: Path | None
) -> dict:
    res = Counter()
    table = codeobj.CodeTable.from_dir(co_dir)
    nbytes = {plan["kernels"][f["kernel"]]: f["nbytes"] for f in plan["families"]}
    res["plan_kernels"] = len(nbytes)
    res["in_code_objects"] = len(nbytes) - len(table.missing(nbytes))
    for sym, nb in nbytes.items():
        if sym in table.entries:
            lay = table.entries[sym].layout
            res["layout_end_eq_blob"] += max(o + n for o, n in lay) == nb
    if kargs_tsv is not None:
        for line in kargs_tsv.read_text().splitlines():
            cols = line.split("\t")
            if cols[0] in table.entries:
                res["kargs_rows"] += 1
                res["kargs_layout_match"] += (
                    tuple(codeobj.parse_layout(cols[2]))
                    == table.entries[cols[0]].layout
                )
    if swap_tsv is not None:
        for line in swap_tsv.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.split("\t")
            res["swap_lines"] += 1
            res["swap_symbol_in_plan"] += cols[0] in nbytes
            try:
                ks = codeobj.kernels(Path(cols[1]).read_bytes())
            except OSError:
                res["swap_file_missing"] += 1
                continue
            lay = tuple(codeobj.parse_layout(cols[3]))
            res["swap_function_found"] += cols[2] in ks
            res["swap_layout_match"] += tuple(ks.get(cols[2], ())) == lay
            res["swap_layout_end_eq_blob"] += bool(lay) and max(
                o + n for o, n in lay
            ) == nbytes.get(cols[0])
    return dict(res), table


def check_params_mode(plan, steps, tables: dict) -> dict:
    """``hipModuleLaunchKernel`` in params mode passes only the bytes the loaded function declares: for each table
    (TileRT's code objects, a swap), every recorded launch's nonzero kernarg byte lies inside a declared argument.
    """
    res = Counter()
    for name, table in tables.items():
        for d in range(alloc.NPES):
            for launch, rec in zip(DL.rank_launches(plan, d), steps[(d, 0)]):
                covered = set()
                for o, n in table.entries[launch["kernel"]].layout:
                    covered.update(range(o, o + n))
                blob = bytes.fromhex(rec["a"])
                res[f"{name}:launches"] += 1
                res[f"{name}:launches_all_bytes_passed"] += all(
                    b in covered for b, v in enumerate(blob) if v
                )
    return dict(res)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CPU checks of the tk decode host")
    ap.add_argument(
        "--runs",
        required=True,
        help="runs/20261115-flydsl-tilert (m6/, m5/paramsum.json.gz, shim/)",
    )
    ap.add_argument("--co", help="TileRT's code objects (*.co)")
    ap.add_argument("--swap", help="a swap table (pd/swap_fly20.tsv)")
    ap.add_argument(
        "--wheel", help="the unzipped TileRT wheel's tilert/ (TileRT's sym_table)"
    )
    a = ap.parse_args(argv)
    runs = Path(a.runs)
    plan = DL.load()
    steps, maps = DL.read_recording(
        runs / "m6/launches_cap12_13.jsonl.gz", runs / "m6/addr_maps.json.gz"
    )
    with gzip.open(runs / "m5/paramsum.json.gz", "rt") as f:
        paramsum = json.load(f)
    print("blobs:", check_blobs(plan, steps, maps))
    print("sizes:", check_sizes(plan, maps, paramsum))
    table = None
    if a.co:
        code, table = check_code(
            plan, Path(a.co), runs / "shim/kargs.tsv", Path(a.swap) if a.swap else None
        )
        print("code:", code)
        tables = {"tilert": table, **({"swap": table.swap(a.swap)} if a.swap else {})}
        print("params mode:", check_params_mode(plan, steps, tables))
    print(
        "tables:", check_tables(plan, maps, Path(a.wheel) if a.wheel else None, table)
    )


if __name__ == "__main__":
    main()

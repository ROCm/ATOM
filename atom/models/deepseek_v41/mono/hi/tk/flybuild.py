# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The decode step's code objects built from our FlyDSL sources (``hi/kernels/gfx950``), one per TileRT kernel the
plan launches (``plan.decode_launches``' 28) plus ``admit_kernel``: what ``TkRank.load_kernels`` takes when no TileRT
``.co`` is given (``code_table``).

A recipe (``RECIPES``) names the TileRT symbol it replaces, the kernel module and its builder, the ``TK_*`` build
switches the module reads at import, and the kernargs preloaded into SGPRs (``amdgpu-kernarg-preload-count``: 7 words
for the kernels TileRT preloads 14 dwords of). ``build_one`` compiles a recipe without a GPU (``ARCH``, FlyDSL's
compile-only mode, its disk cache off) in a fresh process (the switches are module constants), and writes the code
object and ``{symbol, layout}`` next to it -- what ``runs/.../m2/build_kernel.py`` did for the swap tables
(``swap_fly28.tsv``: ``<TileRT symbol> <hsaco> <function> <layout>``).

``code_table(cache_dir)``: every recipe's code object from a node-local cache, keyed by a hash of the recipe, the
sources it imports (the module and its ``atom.*`` imports, transitively), this file and the FlyDSL / AITER versions;
the missing ones built in parallel under a lock (8 ranks starting together build once). Each entry's layout is its
code object's explicit kernarg layout (AMDGPU metadata), checked against the module's ``LAYOUT``.

``python -m atom.models.deepseek_v41.mono.hi.tk.flybuild [--cache DIR] [--tsv OUT] [--compare swap_fly28.tsv]``
builds the table on the CPU, optionally writes it as a swap table and compares it with another one.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import codeobj
from .host import ADMIT_SYMBOL

KERNELS = "atom.models.deepseek_v41.mono.hi.kernels.gfx950"
_T = "_ZN6tilert3ops13ds_v4_1_flash"
# the repo root holding the ``atom`` package
_REPO = Path(__file__).resolve().parents[6]
ARCH = "gfx950"


@dataclass(frozen=True)
class Recipe:
    key: str
    symbol: str  # the TileRT kernel it replaces
    module: str  # under KERNELS
    builder: str = "build"
    env: tuple[tuple[str, str], ...] = ()
    preload: int = 0  # kernarg words preloaded (TK_PRELOAD of the build helpers)


_QKV_KG = "3qkv17q_kv_xf_kg_kernelIL{}ELb1ELi8EL{}EEEvPKhS5_S5_S5_S5_S5_mNS1_8mhc_cell15MhcFinalizeArgsENS1_7indexer8PrepArgsEPhPKfS5_S5_SC_SC_PKiP14__hip_bfloat16SG_SA_SA_SA_SA_SA_SA_Pm"
_QKV_FULL = "NS1_8mhc_cell15MhcFinalizeArgsENS1_7indexer8PrepArgsEiNS2_8CompArgsES6_PKfS5_S5_SE_SE_PKiP14__hip_bfloat16SI_S6_S6_S6_S6_S6_S6_S7_"
_MHC = "3mhc13mhc_v3_kernelIL{}EL{}ELb0ELi10ELi32EL{}ELi8ELi0EEEvNS2_6ParamsENS1_6engram10HashParamsE"
_IX = "7indexer12fused_kernelILi1ELi0EL{}EL{}EEEvNS2_9ScoreArgsENS2_10SelectArgsENS2_8PrepArgsE"
_IXE = "7indexer19fused_engram_kernelILi1ELi0EL{}EL{}ELi6EEEvNS2_9ScoreArgsENS2_10SelectArgsENS2_8PrepArgsENS1_6engram10GemvParamsE"
_TOKTAB = "6toktab13gather_kernelILi8EL{}EEEvNS1_6engram10HashParamsEiPKtS7_PKfPKiSB_S9_PtPfPhSE_SE_SE_"


RECIPES: tuple[Recipe, ...] = (
    Recipe("tail", _T + "4mega20mega_afm_tail_kernelILi1ELi384ELi573440ELi8ELi14EEEvNS1_11sparse_attn8AttnArgsENS1_10attn_fused9ProjoArgsENS2_8MhcFArgsENS2_7MoeArgsENS2_8TailArgsE", "layer_tail", env=(("TK_VARIANT", "tail"),), preload=7),
    Recipe("afm6", _T + "4mega15mega_afm_kernelILi6ELi0ELi384ELb0ELi573440ELi8ELb0EEEvNS1_11sparse_attn8AttnArgsENS1_10attn_fused9ProjoArgsENS2_8MhcFArgsENS2_7MoeArgsE", "layer_tail", env=(("TK_VARIANT", "afm6"),), preload=7),
    Recipe("attn_fused", _T + "10attn_fused17attn_fused_kernelILi5ELi2ELb1ELi0ELi3ELi0ELi8EEEvNS1_11sparse_attn8AttnArgsENS2_9ProjoArgsE", "draft_attn", preload=7),
    Recipe("qkv_kg6_idx16", _T + _QKV_KG.format("i6", "i16"), "qkv", "build_finalize_test", (("TK_QKV_IDX", "16"),), 7),
    Recipe("mhc_pre6", _T + _MHC.format("i6", "i2", "i0"), "mhc_pre", preload=7),
    Recipe("mhc_kg6", _T + "3mhc16mhc_v3_kg_kernelILi6ELi0ELb0ELi10ELi32ELi8ELi8EEEvPK14__hip_bfloat16PKfPKhS8_S8_PhjiNS2_6ParamsE", "mhc_post_pre", preload=7),
    Recipe("mhc_full5", _T + _MHC.format("i5", "i0", "i0"), "mhc_full", preload=7),
    Recipe("mhc_mode3_r6", _T + _MHC.format("i6", "i3", "i0"), "mhc_mode3", env=(("TK_MHC3_ROWS", "6"),), preload=7),
    Recipe("mhc_mode3_r5", _T + _MHC.format("i5", "i3", "i0"), "mhc_mode3", env=(("TK_MHC3_ROWS", "5"),), preload=7),
    Recipe("qkv_kg6_idx0", _T + _QKV_KG.format("i6", "i0"), "qkv", "build_finalize_test", (("TK_QKV_IDX", "0"),), 7),
    Recipe("qkv_xf", _T + "3qkv19q_kv_xf_full_kernelILi6ELb1ELi8ELi0ELi3ELi0EEEvPKhS5_S5_S5_S5_S5_PhPm" + _QKV_FULL, "qkv", "build_xf", (("TK_QKV_XF", "1"), ("TK_QKV_IDX", "0")), 7),
    Recipe("qkv_merged", _T + "3qkv18q_kv_merged_kernelILi6ELb1ELi0ELi2ELb1ELi8ELb1ELi64ELb0ELi3EEEvPKhS5_S5_S5_S5_S5_PhPm" + _QKV_FULL, "qkv", "build_xf", (("TK_QKV_XF", "1"), ("TK_QKV_MERGED", "1"), ("TK_QKV_IDX", "0")), 7),
    Recipe("qkv_xf_comp1", _T + "3qkv19q_kv_xf_full_kernelILi6ELb1ELi8ELi1ELi3ELi64EEEvPKhS5_S5_S5_S5_S5_PhPm" + _QKV_FULL, "qkv", "build_xf", (("TK_QKV_XF", "1"), ("TK_QKV_COMP", "1"), ("TK_QKV_IDX", "0")), 7),
    Recipe("moe_rx5", _T + "6moe_rx19dsv41_moe_rx_kernelILi5ELi128ELi8ELb0ELb1EEEvPK14__hip_bfloat16PKhS8_S8_PKfS8_S8_S8_S8_S8_S8_S8_S8_PKiPhPKPviijPS4_PiPfPmNS1_8mhc_cell15MhcFinalizeArgsEi", "draft_moe", preload=7),
    Recipe("dspark_head5", _T + "11dspark_head25dspark_head_sample_kernelILi5ELb0ELi8EEEvPK14__hip_bfloat16PKhS6_S6_PKiiPhPKPviixPiPfS6_PS4_NS1_3mhc6ParamsEPKxSA_", "draft_head", preload=7),
    Recipe("toktab_gather0", _T + _TOKTAB.format("i0"), "toktab", preload=7),
    Recipe("qkv_kg5", _T + _QKV_KG.format("i5", "i0"), "qkv", "build_finalize_test", (("TK_QKV_ROWS", "5"), ("TK_QKV_IDX", "0")), 7),
    Recipe("engram_post6", _T + "11engram_post24engram_post_apply_kernelILi6EEEvNS2_15PostApplyParamsE", "engram_post"),
    Recipe("engram_fused6", _T + "6engram19engram_fused_kernelILi6ELb0ELi1EEEvNS2_10GemvParamsENS2_11ApplyParamsEi", "engram_fused"),
    Recipe("toktab_gather6", _T + _TOKTAB.format("i6"), "engram_gather"),
    Recipe("mhc_post_engram6", _T + _MHC.format("i6", "i1", "i2"), "mhc_post_engram"),
    Recipe("head_seed_fold6", _T + "11sample_fold21head_seed_fold_kernelILi6ELb0ELi8ELb0EEEvPK14__hip_bfloat16PKhPfxNS1_3mhc6ParamsENS1_9main_proj8SeedArgsENS1_7sampler4ArgsENS1_7mtp_ctl10AcceptArgsE", "head_seed_fold"),
    Recipe("indexer_4027", _T + _IX.format("i4027", "i6"), "indexer", env=(("TK_IX", "4027"),)),
    Recipe("indexer_2747", _T + _IX.format("i2747", "i6"), "indexer", env=(("TK_IX", "2747"),)),
    Recipe("indexer_499", _T + _IX.format("i499", "i1"), "indexer", env=(("TK_IX", "499"),)),
    Recipe("indexer_243", _T + _IX.format("i243", "i1"), "indexer", env=(("TK_IX", "243"),)),
    Recipe("indexer_e4027", _T + _IXE.format("i4027", "i6"), "indexer", env=(("TK_IX", "e4027"),)),
    Recipe("indexer_e2747", _T + _IXE.format("i2747", "i6"), "indexer", env=(("TK_IX", "e2747"),)),
    Recipe("admit", ADMIT_SYMBOL, "admit"),
)  # fmt: skip
BY_KEY = {r.key: r for r in RECIPES}
STEP_RECIPES = tuple(r for r in RECIPES if r.key != "admit")


# ---------------------------------------------------------------------------------------------------------------
# cache keys (no import of the kernel modules: their switches are read at import)


def _module_file(name: str) -> Path | None:
    p = _REPO.joinpath(*name.split("."))
    for f in (p.with_suffix(".py"), p / "__init__.py"):
        if f.is_file():
            return f
    return None


def _atom_imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module.startswith("atom."):
                out.add(node.module)
                out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.startswith("atom."))
    return out


def source_files(module: str) -> list[Path]:
    """``module`` (under ``KERNELS``) and every ``atom.*`` file it imports, transitively, sorted."""
    seen: dict[str, Path] = {}
    todo = [f"{KERNELS}.{module}"]
    while todo:
        name = todo.pop()
        f = _module_file(name)
        if f is None or str(f) in seen:
            continue
        seen[str(f)] = f
        todo.extend(_atom_imports(f))
    return sorted(seen.values())


def _versions() -> str:
    from importlib import metadata

    out = []
    for dist in ("flydsl", "amd-aiter", "aiter"):
        try:
            out.append(f"{dist}={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pass
    return ";".join(out)


def _aiter_flydsl_files() -> list[Path]:
    from importlib.util import find_spec

    spec = find_spec("aiter")  # (a top-level spec: aiter is not imported)
    if spec is None or not spec.submodule_search_locations:
        return []
    d = Path(next(iter(spec.submodule_search_locations))) / "ops" / "flydsl" / "kernels"
    return sorted(d.glob("*.py")) if d.is_dir() else []


def recipe_key(
    r: Recipe, versions: str | None = None, extra_files: list[Path] | None = None
) -> str:
    h = hashlib.sha256()
    h.update(
        repr((r.key, r.symbol, r.module, r.builder, r.env, r.preload, ARCH)).encode()
    )
    h.update((versions if versions is not None else _versions()).encode())
    for f in (
        source_files(r.module) + [Path(__file__).resolve()] + list(extra_files or [])
    ):
        h.update(str(f.name).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:20]


def default_cache_dir() -> Path:
    d = os.environ.get("ATOM_DSV41_TK_FLY_CACHE")
    return Path(d) if d else Path.home() / ".cache" / "atom" / "tk_fly"


# ---------------------------------------------------------------------------------------------------------------
# one build (a fresh process: python -m ...flybuild --one KEY --out STEM)


def build_env(r: Recipe) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TK_")}
    env.update(dict(r.env))
    env.update(FLYDSL_RUNTIME_ENABLE_CACHE="0", ARCH=ARCH, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(_REPO), env.get("PYTHONPATH", "")) if p
    )
    return env


def build_one(key: str, stem: str) -> dict:
    """Compile recipe ``key`` (this process's ``TK_*`` environment must be the recipe's) -> ``stem``.hsaco / .json."""
    import contextlib
    import importlib

    from atom.mono.runtime.compile import _code_object, compile_only

    r = BY_KEY[key]
    for k, v in r.env:
        if os.environ.get(k) != v:
            raise RuntimeError(f"{key}: build process without {k}={v}")
    mod = importlib.import_module(f"{KERNELS}.{r.module}")
    name, launch = getattr(mod, r.builder)()
    ctx = contextlib.nullcontext()
    if r.preload:
        from flydsl.compiler.llvm_options import llvm_options

        ctx = llvm_options({"amdgpu-kernarg-preload-count": r.preload})
    with ctx, compile_only():
        launch(*([0] * mod.KARG_WORDS))
    arts = list(launch._mem_cache.values())
    if len(arts) != 1:
        raise RuntimeError(f"{key}: {len(arts)} compiled binaries, want 1")
    elf = _code_object(arts[0]._ir_text)
    meta = {
        "key": key,
        "symbol": r.symbol,
        "function": name,
        "layout": mod.LAYOUT,
        "bytes": len(elf),
    }
    out = Path(stem)
    out.with_suffix(".hsaco").write_bytes(elf)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


# ---------------------------------------------------------------------------------------------------------------
# the table


def _entry(hsaco: Path, meta: dict) -> codeobj.Entry:
    ks = codeobj.kernels(hsaco.read_bytes())
    fn = meta["function"]
    if fn not in ks:
        raise RuntimeError(f"{hsaco}: no kernel {fn} (has {sorted(ks)})")
    lay = tuple(ks[fn])
    if lay != tuple(codeobj.parse_layout(meta["layout"])):
        raise RuntimeError(
            f"{hsaco}: kernarg layout {lay[:3]}... is not the module's LAYOUT"
        )
    return codeobj.Entry(str(hsaco), fn, lay)


def _build_into(r: Recipe, stem: Path, timeout: float) -> None:
    tmp = stem.with_name(stem.name + f".tmp{os.getpid()}")
    cmd = [
        sys.executable,
        "-m",
        f"{__package__}.flybuild",
        "--one",
        r.key,
        "--out",
        str(tmp),
    ]
    p = subprocess.run(
        cmd,
        env=build_env(r),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"build of {r.key} failed ({p.returncode}): {p.stderr[-2000:]}"
        )
    # the code object last: its presence marks the entry complete
    os.replace(tmp.with_suffix(".json"), stem.with_suffix(".json"))
    os.replace(tmp.with_suffix(".hsaco"), stem.with_suffix(".hsaco"))


def code_table(
    cache_dir: str | Path | None = None,
    recipes=RECIPES,
    jobs: int | None = None,
    timeout: float = 3600.0,
    log=None,
) -> codeobj.CodeTable:
    """``CodeTable`` of ``recipes`` (default all: the step's 28 kernels and ``admit_kernel``), each built from source
    into ``cache_dir`` (default ``default_cache_dir()``, node-local) unless its key is there already.
    """
    cache = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    versions = _versions()
    extra = _aiter_flydsl_files()
    stems = {
        r.key: cache / f"{r.key}-{recipe_key(r, versions, extra)}" for r in recipes
    }
    with open(cache / ".lock", "w") as lock:
        fcntl.flock(
            lock, fcntl.LOCK_EX
        )  # one builder at a time; a waiter then finds the entries
        todo = [r for r in recipes if not stems[r.key].with_suffix(".hsaco").is_file()]
        if todo:
            t0 = time.time()
            if log:
                log(
                    f"tk flybuild: building {len(todo)} of {len(recipes)} kernels into {cache}"
                )
            n = jobs or min(len(todo), max(1, (os.cpu_count() or 8) // 4), 16)
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(lambda r: _build_into(r, stems[r.key], timeout), todo))
            if log:
                log(
                    f"tk flybuild: built {len(todo)} kernels in {time.time() - t0:.0f}s"
                )
    entries = {}
    for r in recipes:
        stem = stems[r.key]
        meta = json.loads(stem.with_suffix(".json").read_text())
        entries[r.symbol] = _entry(stem.with_suffix(".hsaco"), meta)
    return codeobj.CodeTable(entries)


def write_tsv(table: codeobj.CodeTable, path: str | Path) -> None:
    """``table`` as a swap table (``CodeTable.swap``'s format)."""
    lines = [
        f"{s}\t{e.path}\t{e.function}\t{','.join(f'{o}:{n}' for o, n in e.layout)}"
        for s, e in table.entries.items()
    ]
    Path(path).write_text("\n".join(lines) + "\n")


def compare_tsv(table: codeobj.CodeTable, tsv: str | Path) -> dict:
    """Per line of a swap table: is its TileRT symbol in ``table``, with the same function and layout."""
    out = {"lines": 0, "symbol": 0, "function": 0, "layout": 0, "function_differs": []}
    for line in Path(tsv).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        sym, _, fn, lay = line.split("\t")
        out["lines"] += 1
        e = table.entries.get(sym)
        if e is None:
            continue
        out["symbol"] += 1
        out["layout"] += e.layout == tuple(codeobj.parse_layout(lay))
        if e.function == fn:
            out["function"] += 1
        else:
            out["function_differs"].append((fn, e.function))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="build the tk decode kernels from the FlyDSL sources (no GPU)"
    )
    ap.add_argument("--one", help="(internal) build one recipe into --out")
    ap.add_argument("--out")
    ap.add_argument(
        "--cache",
        help="the code object cache (default: $ATOM_DSV41_TK_FLY_CACHE or ~/.cache/atom/tk_fly)",
    )
    ap.add_argument("--only", help="comma-separated recipe keys")
    ap.add_argument("--jobs", type=int)
    ap.add_argument("--tsv", help="write the table as a swap table")
    ap.add_argument(
        "--compare", help="a swap table to compare symbols / functions / layouts with"
    )
    a = ap.parse_args(argv)
    if a.one:
        print(json.dumps(build_one(a.one, a.out)))
        return
    rs = RECIPES if not a.only else tuple(BY_KEY[k] for k in a.only.split(","))
    table = code_table(a.cache, rs, a.jobs, log=print)
    for r in rs:
        e = table.entries[r.symbol]
        print(
            f"{r.key}\t{e.function}\t{Path(e.path).stat().st_size}\t{len(e.layout)} args"
        )
    if a.tsv:
        write_tsv(table, a.tsv)
    if a.compare:
        print("compare:", json.dumps(compare_tsv(table, a.compare)))


if __name__ == "__main__":
    main()

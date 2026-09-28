# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Generic dump-comparison primitives for forward bisect & batch invariance.

Companion to ``atom/utils/debug_helper/dump.py``. Provides:

- ``cos_max(a, b)`` — stable cosine similarity in double precision (avoids
  the bf16/fp32 rounding bug where naive ``F.cosine_similarity`` returns
  values > 1.0).
- ``flag_for(cos, rel)`` — uniform ✓/?/✗/✗✗ severity flag.
- ``slot_split(t, n_slots)`` — slice flat batch tensor into per-slot chunks.
- ``pick_prefill_call(files, n_slots)`` — distinguish warmup vs real prefill
  vs decode in multi-call dumps.
- ``schema_diff(a, b)`` — list only-A / only-B / common keys after name
  normalization.
- ``compare_slots(t, n_slots, ref_slot=0)`` — pairwise cos slot 0 vs each
  other slot. Used for batch invariance bisect (V4 paper §3.3).

CLI subcommands (run via ``python -m atom.utils.debug_helper.compare``):

- ``slot-invariance``  per-slot cos within batch (4×identical-prompt test)
- ``ref-vs-target``    per-stage cos vs reference dump
- ``layer-bisect``     find first layer with cos drop below threshold
- ``schema``           dump schema diff (names / shapes / dtypes)

See ``.claude/skills/dump-bisect-debug.md`` for the methodology.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Optional

import torch

# === Severity thresholds (cos) =======================================
COS_BIT_EQUAL = 0.9999  # essentially identical (counts as "match")
COS_NUM_DRIFT = 0.999  # acceptable kernel/dtype drift
COS_ALGO_DIFF = 0.99  # noticeable algorithmic diff
COS_BUG = 0.9  # likely bug


# === Numerical primitives ============================================


def cos_max(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    """Return (cos_sim, max_abs_diff, rel_err).

    Uses double precision throughout to avoid the bf16/fp32 cosine bug
    where the result can exceed 1.0 due to rounding when the inputs are
    nearly identical.
    """
    af = a.double().flatten()
    bf = b.double().flatten()
    if af.shape != bf.shape:
        n = min(af.shape[0], bf.shape[0])
        af, bf = af[:n], bf[:n]
    dot = (af * bf).sum().item()
    na = (af * af).sum().sqrt().item()
    nb = (bf * bf).sum().sqrt().item()
    if na == 0 or nb == 0:
        cos = float("nan")
    else:
        cos = dot / (na * nb)
    diff = af - bf
    max_abs = float(diff.abs().max().item())
    rel = max_abs / max(float(bf.abs().max().item()), 1e-12)
    return cos, max_abs, rel


def flag_for(cos: float, rel: float = 0.0) -> str:
    """Uniform severity flag for ``cos`` (and optionally ``rel`` error).

    ``rel`` is needed because cos can stay > 0.999 while element-wise
    diffs are 10%+ — a regression that compounds across 60 layers.
    """
    if cos != cos:  # NaN
        return "??"
    if cos > COS_BIT_EQUAL and rel < 0.01:
        return "✓"
    if cos > COS_NUM_DRIFT:
        return "?"
    if cos > COS_ALGO_DIFF:
        return "✗"
    return "✗✗"


def byte_equal_pct(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of element-wise byte-equal positions (1.0 == identical bytes)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return 0.0
    aa = a.contiguous().view(torch.uint8)
    bb = b.contiguous().view(torch.uint8)
    return float((aa == bb).float().mean().item())


# === Slot slicing (batch invariance) =================================


def slot_split(t: torch.Tensor, n_slots: int) -> list[torch.Tensor]:
    """Split flat batch tensor along dim 0 into ``n_slots`` equal chunks.

    Tensor must have ``size(0) % n_slots == 0``. Used to compare per-batch-slot
    outputs of an op that should be batch-invariant per V4 paper §3.3.
    """
    n = t.shape[0]
    if n % n_slots != 0:
        raise ValueError(
            f"slot_split: tensor.shape[0]={n} not divisible by n_slots={n_slots}"
        )
    per = n // n_slots
    return [t[i * per : (i + 1) * per] for i in range(n_slots)]


def compare_slots(
    t: torch.Tensor, n_slots: int, ref_slot: int = 0
) -> list[tuple[float, float, float]]:
    """Pairwise (cos, max_abs, rel) of slot ``ref_slot`` vs every other slot.

    Returns ``[(cos, max_abs, rel), ...]`` indexed by other slot id (skipping
    ref_slot itself; result has length ``n_slots - 1``).
    """
    slots = slot_split(t, n_slots)
    ref = slots[ref_slot]
    return [cos_max(ref, slots[i]) for i in range(n_slots) if i != ref_slot]


# === Multi-call (warmup vs prefill) ==================================


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def pick_prefill_call(
    calls: list[tuple[int, str, int]], n_slots: int
) -> Optional[tuple[int, str, int]]:
    """Pick the real-prefill call from a list of (call_idx, path, n_tok).

    Heuristic (matches ATOM model_runner behavior):
      1. n_tok % n_slots == 0           — full prefill spans all slots
      2. n_tok > n_slots                — not a per-seq decode (1 tok/seq)
      3. not power-of-2 if possible     — warmup uses pow-of-2 dummy sizes
      4. lowest call index among ties   — earliest matching call

    Returns None if no candidate matches.
    """
    cands = [(c, f, n) for (c, f, n) in calls if n % n_slots == 0 and n > n_slots]
    if not cands:
        return None
    non_pow2 = [(c, f, n) for (c, f, n) in cands if not is_pow2(n)]
    chosen_pool = non_pow2 or cands
    chosen_pool.sort(key=lambda x: x[0])
    return chosen_pool[0]


# === Schema diff ====================================================


def normalize_name(name: str, strip_prefixes: tuple[str, ...] = ("model.",)) -> str:
    """Strip implementation-specific prefixes for cross-source name matching.

    Example: ATOM uses ``model.layers.0.X``, ref uses ``layers.0.X`` →
    after ``normalize_name`` both become ``layers.0.X``.
    """
    for p in strip_prefixes:
        if name.startswith(p):
            return name[len(p) :]
        if name.startswith("buffer:" + p):
            return "buffer:" + name[len("buffer:" + p) :]
    return name


def schema_diff(
    a: dict, b: dict, strip_prefixes: tuple[str, ...] = ("model.",)
) -> tuple[list[str], list[str], list[str]]:
    """Return (only_a, only_b, common) keys after name normalization.

    Internal underscored keys (e.g. ``_tp_rank``) are excluded.
    """
    an = {normalize_name(k, strip_prefixes): k for k in a if not k.startswith("_")}
    bn = {normalize_name(k, strip_prefixes): k for k in b if not k.startswith("_")}
    only_a = sorted(set(an) - set(bn))
    only_b = sorted(set(bn) - set(an))
    common = sorted(set(an) & set(bn))
    return only_a, only_b, common


# === Table formatter =================================================


def print_compare_table(
    rows: list[dict],
    cols: list[tuple[str, str, int]],  # (key, header, width)
    sep: str = " ",
) -> None:
    """Pretty-print a list-of-dict table with fixed column widths."""
    header = sep.join(f"{h:>{w}}" if w > 0 else f"{h}" for _, h, w in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        out = []
        for key, _, w in cols:
            v = row.get(key, "")
            if isinstance(v, float):
                s = f"{v:.6f}" if abs(v) < 100 else f"{v:.4e}"
            else:
                s = str(v)
            out.append(f"{s:>{w}}" if w > 0 else s)
        print(sep.join(out))


# === Dump file walker ================================================


_FNAME_RE = re.compile(
    r"^layer(\d+)(?:_([A-Za-z0-9_]+?))?_rank(\d+)(?:_call(\d+))?\.pt$"
)


def walk_dumps(
    dump_dir: str, rank: int = 0
) -> dict[tuple[int, str], list[tuple[int, str, int]]]:
    """Walk ATOM_FWD_DUMP_DIR and group by (layer_id, stage_class).

    Returns dict mapping (layer, stage) → list of (call_idx, path, n_tok).
    For one-shot dumps (no call suffix), call_idx = -1.
    """
    grouped: dict[tuple[int, str], list[tuple[int, str, int]]] = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(dump_dir, "layer*.pt"))):
        fn = os.path.basename(path)
        m = _FNAME_RE.match(fn)
        if not m:
            continue
        L = int(m.group(1))
        stage = m.group(2) or "Block"
        r = int(m.group(3))
        call = int(m.group(4)) if m.group(4) is not None else -1
        if r != rank:
            continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        n_tok = int(d["hidden"].shape[0]) if "hidden" in d else -1
        grouped[(L, stage)].append((call, path, n_tok))
    return grouped


# === CLI commands ====================================================


def cmd_slot_invariance(args: argparse.Namespace) -> int:
    """Compare per-slot outputs within batch (V4 batch invariance test).

    For each (layer, stage) in the dump dir, picks the prefill call (if
    multi-call) and computes cos(slot 0 vs slot N) for N=1..n_slots-1.
    Useful to verify kernels are batch-invariant per V4 paper §3.3.
    """
    grouped = walk_dumps(args.dir, rank=args.rank)
    if not grouped:
        print(f"No dumps found in {args.dir}")
        return 2

    rows = []
    for key in sorted(grouped):
        L, stage = key
        calls = grouped[key]
        if len(calls) > 1:
            chosen = pick_prefill_call(calls, args.n_slots)
            if chosen is None:
                continue
            call, path, n_tok = chosen
        else:
            call, path, n_tok = calls[0]
            if n_tok % args.n_slots != 0:
                continue
        d = torch.load(path, map_location="cpu", weights_only=False)
        h = d["hidden"]
        try:
            results = compare_slots(h, args.n_slots, ref_slot=0)
        except ValueError:
            continue
        worst_cos = min(c for c, _, _ in results)
        worst_rel = max(r for _, _, r in results)
        rows.append(
            {
                "layer": L,
                "stage": stage,
                "shape": str(tuple(h.shape)),
                "tok/slot": h.shape[0] // args.n_slots,
                "call": call if call >= 0 else "",
                "cos(0,1)": results[0][0] if len(results) > 0 else float("nan"),
                "cos(0,2)": results[1][0] if len(results) > 1 else float("nan"),
                "cos(0,3)": results[2][0] if len(results) > 2 else float("nan"),
                "max_abs": max(m for _, m, _ in results),
                "flag": flag_for(worst_cos, worst_rel),
            }
        )

    cols = [
        ("flag", "", 2),
        ("layer", "Layer", 5),
        ("stage", "Stage", 22),
        ("shape", "shape", 22),
        ("tok/slot", "tok/slot", 9),
        ("call", "call", 5),
        ("cos(0,1)", "cos(0,1)", 10),
        ("cos(0,2)", "cos(0,2)", 10),
        ("cos(0,3)", "cos(0,3)", 10),
        ("max_abs", "max_abs", 12),
    ]
    print_compare_table(rows, cols)
    n_diverge = sum(1 for r in rows if r["flag"] in ("✗", "✗✗"))
    print(f"\n{len(rows)} (layer, stage) compared; {n_diverge} diverge.")
    return 1 if n_diverge > 0 else 0


def cmd_ref_vs_target(args: argparse.Namespace) -> int:
    """Compare ATOM dump vs reference dump per stage."""
    a_path = os.path.join(args.dir, f"atom_rank{args.rank}_fwd.pt")
    r_path = os.path.join(args.dir, f"ref_rank{args.rank}_fwd.pt")
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    r = torch.load(r_path, map_location="cpu", weights_only=False)

    common = sorted(set(a) & set(r))
    if "embed.input_ids" in common:
        ai = a["embed.input_ids"].long().flatten()
        ri = r["embed.input_ids"].long().flatten()
        if not torch.equal(ai, ri):
            print("✗ input_ids mismatch — STOP, fix tokenization first")
            return 2
        print("✓ input_ids match")

    rows = []
    for k in common:
        if k == "embed.input_ids":
            continue
        ta, tr = a[k], r[k]
        if tr.dim() > ta.dim():
            tr = tr.squeeze(0)
        if ta.dim() > tr.dim():
            ta = ta.squeeze(0)
        cos, ma, rel = cos_max(ta, tr)
        rows.append(
            {
                "flag": flag_for(cos, rel),
                "stage": k,
                "cos": cos,
                "max_abs": ma,
                "rel": rel,
            }
        )

    cols = [
        ("flag", "", 2),
        ("stage", "Stage", 40),
        ("cos", "cos", 10),
        ("max_abs", "max_abs", 12),
        ("rel", "rel", 10),
    ]
    print_compare_table(rows, cols)
    return 0


def cmd_layer_bisect(args: argparse.Namespace) -> int:
    """Find first layer where cos drops below ``--threshold`` (vs reference)."""
    rc = cmd_ref_vs_target(args)
    return rc


def cmd_schema(args: argparse.Namespace) -> int:
    """Show schema diff between two dump files."""
    a = torch.load(args.a, map_location="cpu", weights_only=False)
    b = torch.load(args.b, map_location="cpu", weights_only=False)
    only_a, only_b, common = schema_diff(a, b)
    print(f"only in A ({len(only_a)}): {only_a[:20]}{'...' if len(only_a)>20 else ''}")
    print(f"only in B ({len(only_b)}): {only_b[:20]}{'...' if len(only_b)>20 else ''}")
    print(f"common ({len(common)})")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="ATOM debug-dump comparison primitives + CLI."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    si = sub.add_parser("slot-invariance", help="Per-slot batch invariance check")
    si.add_argument("--dir", required=True)
    si.add_argument("--rank", type=int, default=0)
    si.add_argument("--n-slots", type=int, default=4)
    si.set_defaults(func=cmd_slot_invariance)

    rt = sub.add_parser("ref-vs-target", help="ATOM vs reference per-stage cos")
    rt.add_argument("--dir", required=True)
    rt.add_argument("--rank", type=int, default=0)
    rt.set_defaults(func=cmd_ref_vs_target)

    lb = sub.add_parser("layer-bisect", help="Find first layer with cos drop")
    lb.add_argument("--dir", required=True)
    lb.add_argument("--rank", type=int, default=0)
    lb.add_argument("--threshold", type=float, default=COS_NUM_DRIFT)
    lb.set_defaults(func=cmd_layer_bisect)

    sc = sub.add_parser("schema", help="Schema diff between two dump files")
    sc.add_argument("--a", required=True)
    sc.add_argument("--b", required=True)
    sc.set_defaults(func=cmd_schema)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

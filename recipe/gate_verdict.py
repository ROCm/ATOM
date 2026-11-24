"""Verdict of one gate run vs /shared_nfs/ygan/max/acc/baseline.json (see notes/accuracy.md).

usage: gate_verdict.py <outdir> <gtag> <repo> <pypath_pre> <env-string>
GSM8K: mean strict-match of the gate's repeats >= baseline mean - margin[number of repeats].
AL: real DSpark acceptance length during GSM8K >= baseline - al_margin.  LONG: needle hits >= baseline.
"""

import glob
import json
import os
import re
import sys
import time

A = "/shared_nfs/ygan/max/acc"


def load(p, default=None):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return default


def al_of(path):
    try:
        return json.loads(open(path).read().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return {}


def main():
    out, gtag, repo, pre, envs = sys.argv[1:6]
    base = load(os.path.join(A, "baseline.json"), {})
    reps = sorted(glob.glob(os.path.join(out, "gsm8k_r*")))
    gs = [load(os.path.join(d, "summary.json"), {}) for d in reps]
    alg = al_of(os.path.join(out, "al_gsm8k.txt"))
    all_ = al_of(os.path.join(out, "al_long.txt"))
    lng = load(os.path.join(out, "long.json"))
    lines, ok = [], True

    good = [g for g in gs if g.get("n") == 1319 and g.get("strict") is not None]
    s = sum(g["strict"] for g in good) / len(good) if good else None
    fl = sum(g["flexible"] for g in good) / len(good) if good else None
    runs = ", ".join(f"{g['strict']:.4f}" for g in good)
    if not good or len(good) != len(gs):
        ok = False
        lines.append(f"  GSM8K  ERROR: {len(good)}/{len(gs)} complete runs ({runs})")
    if good and base:
        margin = base["margin"].get(str(len(good)), min(base["margin"].values()))
        thr = base["strict_mean"] - margin
        p = s >= thr
        ok &= p
        lines.append(
            f"  GSM8K  {'PASS' if p else 'FAIL'}: strict mean {s:.4f} (runs {runs}), flexible {fl:.4f} | baseline "
            f"{base['strict_mean']:.4f} (runs {', '.join(f'{x:.4f}' for x in base['strict_runs'])}), threshold "
            f"{thr:.4f} (mean - {margin:.4f}) | empty {sum(g.get('empty_responses') or 0 for g in good)} "
            f"no#### {sum(g.get('no_hash_answer') or 0 for g in good)}"
        )
        corr = load(os.path.join(reps[0], "correct.json"), {})
        for ref in base.get("ref_correct", [])[:1]:
            rc = load(ref, {})
            if rc and corr:
                lost = sum(1 for k, v in rc.items() if v[0] == 1 and corr.get(k, [0])[0] == 0)
                gain = sum(1 for k, v in rc.items() if v[0] == 0 and corr.get(k, [0])[0] == 1)
                lines.append(
                    f"         paired run1 vs baseline {os.path.basename(os.path.dirname(ref))}: lost {lost}, gained "
                    f"{gain} (baseline run-to-run: {base.get('flip_note', '')})"
                )
    elif good:
        lines.append(f"  GSM8K  (no baseline yet): strict mean {s:.4f} (runs {runs}) flexible {fl:.4f}")

    al = alg.get("al")
    if al is None:
        ok = False
        lines.append("  AL     ERROR: no [MTP Stats Interval] lines during GSM8K (is AL=real / DSpark on?)")
    elif base.get("al_gsm8k_mean"):
        thr = base["al_gsm8k_mean"] - base["al_margin"]
        p = al >= thr
        ok &= p
        lines.append(
            f"  AL     {'PASS' if p else 'FAIL'}: GSM8K real acceptance length {al:.3f} (accept rate "
            f"{alg['accept_rate']:.4f}, {alg['steps']} steps) | baseline {base['al_gsm8k_mean']:.3f}, threshold {thr:.3f}"
        )
    else:
        lines.append(f"  AL     GSM8K real acceptance length {al:.3f}")

    if lng is not None:
        hits = sum(r["hit"] for r in lng)
        need = base.get("long_hits", len(lng))
        p = hits >= need
        ok &= p
        lal = all_.get("al")
        lines.append(
            f"  LONG   {'PASS' if p else 'FAIL'}: needle hits {hits}/{len(lng)} (baseline {need}) at "
            f"{[r.get('tokens') for r in lng]} prompt tokens"
            + (f"; AL {lal}" if lal else "")
        )

    head = f"GATE {gtag}: {'PASS' if ok else 'FAIL'}  ({time.strftime('%Y-%m-%d %H:%M')} UTC, repo {repo}, pre {pre}, env [{envs}])"
    print(head)
    print("\n".join(lines))
    lng_s = f"{sum(r['hit'] for r in lng)}/{len(lng)}" if lng is not None else "-"
    with open("/shared_nfs/ygan/p1/c17/gates/index.tsv", "a") as f:
        f.write(
            "\t".join(
                map(str, [time.strftime("%Y-%m-%dT%H:%M"), gtag, "PASS" if ok else "FAIL",
                          f"{s:.4f}" if s is not None else None, runs, al, lng_s, repo, pre, re.sub(r"\s+", " ", envs)])
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""score_diag.py RESULT_DIR [--gpus 4] [--json] -- per-point scoring + cutoff/grace diagnostics (CPU only).

milestone4 (aiperf-anomaly NOTES, recipe changes 1-3). Prints, for one AgentX point:
  RS      official InferenceX score: sum(ISL+OSL) of kept records (profiling phase, no error) / record span / GPUs
          (request_metrics.compute_throughput_stats; summarize_ix.py computes the same number)
  PRO900  time-integrated score: each kept request's tokens prorated by its overlap with [phase start, +DURATION] /
          DURATION / GPUs (no boundary tail; report next to RS for A/B claims below ~0.5%)
  AIP     aiperf's own total_token_throughput / GPUs (phase window incl. any grace wait; NEVER a score)
  diagnostics: in-flight at the sending cutoff, last request end (s after phase start), requests ending after the
  cutoff, aiperf grace_period_timeout, "leaked state" / "Abandoned pending join" warnings (aiperf post-cutoff join bug:
  a grace timeout with in_flight=0 is client bookkeeping, NOT a server stall), cancelled / errored records.
Writes RESULT_DIR/score_diag.json. Logic ported from /shared_nfs/ygan/p1/aiperf-anomaly/defs.py (verified there).
"""
import json, os, re, sys


def mval(r, k):
    v = r.get("metrics", {}).get(k)
    return v.get("value") if isinstance(v, dict) else v


def phase_info(d):
    log = os.path.join(d, "logs", "aiperf.log")
    if not os.path.isfile(log):
        log = os.path.join(d, "client.log")
    try:
        t = open(log, errors="replace").read()
    except OSError:
        return {"log": None}
    o = {"log": log}
    m = re.search(r"Phase profiling \(profiling\) started .*?target: ([\d.]+)s duration", t)
    o["duration"] = float(m.group(1)) if m else None
    m = re.search(r"Phase profiling \(profiling\) sending complete \| sent=([\d,]+), completed=([\d,]+), in_flight=(\d+)", t)
    o["inflight_at_cutoff"] = int(m.group(3)) if m else None
    m = re.search(r"Phase profiling \(profiling\) complete \|.*?elapsed=([\d.]+)s(.*)", t)
    if m:
        o["phase_elapsed_s"] = float(m.group(1))
        o["grace_timeout"] = "grace_period_timeout=True" in m.group(2)
    m = re.search(r"Phase profiling timed out, cancelling all credits\. Stats: (.*?)\s*$", t, re.M)
    o["timeout_stats"] = m.group(1) if m else None
    o["leaked_state_warnings"] = len(re.findall(r"BranchOrchestrator leaked state", t))
    o["abandoned_joins"] = len(re.findall(r"Abandoned pending join", t))
    o["leak_lines"] = [l.split(" - ", 3)[-1][:220] for l in t.splitlines()
                       if "leaked state" in l or "Abandoned pending join" in l][:4]
    for k in ("start_ns", "requests_end_ns"):
        m = re.search(r"phase=CreditPhase\.PROFILING.*?\b%s=(\d+)" % k, t)
        o[k] = int(m.group(1)) if m else None
    return o


def diag(d, gpus=4):
    pi = phase_info(d)
    recs, errs, canc = [], 0, 0
    for line in open(os.path.join(d, "profile_export.jsonl"), errors="replace"):
        line = line.strip().strip("\x00")
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        md = r.get("metadata", {})
        if md.get("benchmark_phase") not in (None, "profiling"):
            continue
        if md.get("was_cancelled"):
            canc += 1
        if r.get("error"):
            errs += 1
            continue
        recs.append((int(md["request_start_ns"]), int(md["request_end_ns"]),
                     (mval(r, "input_sequence_length") or 0) + (mval(r, "output_sequence_length") or 0)))
    if not recs:
        return dict(dir=d, n=0, err=errs, cancelled=canc, **pi)
    dur = pi.get("duration") or 900.0
    s0 = min(s for s, e, t in recs)
    e1 = max(e for s, e, t in recs)
    p0 = pi.get("start_ns") or s0
    tok = sum(t for s, e, t in recs)
    pro = 0.0
    for s, e, t in recs:
        a, b = max(s, p0), min(e, p0 + dur * 1e9)
        if b > a:
            pro += t * (b - a) / max(e - s, 1)
    late = [(s, e, t) for s, e, t in recs if (e - p0) / 1e9 > dur]
    agg = {}
    try:
        agg = json.load(open(os.path.join(d, "profile_export_aiperf.json")))
    except Exception:
        pass
    att = (agg.get("total_token_throughput") or {}).get("avg")
    return dict(dir=d, n=len(recs), err=errs, cancelled=canc, gpus=gpus, duration=dur, tokens=tok,
                span_s=(e1 - s0) / 1e9, last_end_s=(e1 - p0) / 1e9, n_end_after_cutoff=len(late),
                tok_end_after_cutoff=sum(t for s, e, t in late),
                RS=tok / ((e1 - s0) / 1e9) / gpus, PRO900=pro / dur / gpus, AIP=att / gpus if att else None,
                **{k: v for k, v in pi.items() if k != "duration"})


def fmt(x, p=0):
    return "-" if x is None else f"{x:.{p}f}"


if __name__ == "__main__":
    a = sys.argv[1:]
    g = int(a[a.index("--gpus") + 1]) if "--gpus" in a else 4
    d = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or a[i - 1] != "--gpus")][0]
    r = diag(d.rstrip("/"), g)
    tag = os.path.basename(d.rstrip("/"))
    try:
        json.dump(r, open(os.path.join(d, "score_diag.json"), "w"), indent=1, default=str)
    except OSError:
        pass
    if r.get("n"):
        pro_name = "PRO%d" % round(r["duration"])
        print(f"### score {tag}: RS(official) {fmt(r['RS'])} tok/s/GPU | {pro_name}(prorated) {fmt(r['PRO900'])} | "
              f"aiperf-window {fmt(r['AIP'])} (not a score) | n={r['n']} err={r['err']} cancelled={r['cancelled']}")
        print(f"### cutoff {tag}: in_flight@cutoff={r.get('inflight_at_cutoff')} ends_after_cutoff={r['n_end_after_cutoff']} "
              f"({r['tok_end_after_cutoff'] / 1e6:.2f}M tok) last_end={fmt(r['last_end_s'], 2)}s span={fmt(r['span_s'], 2)}s "
              f"phase_elapsed={r.get('phase_elapsed_s')}s grace_timeout={r.get('grace_timeout')} "
              f"leaked_state={r.get('leaked_state_warnings')} abandoned_joins={r.get('abandoned_joins')}")
        if r.get("grace_timeout") and not r.get("leaked_state_warnings"):
            print(f"### WARNING {tag}: grace timeout WITHOUT a leaked-join warning: check for a real stall ({r.get('timeout_stats')})")
        elif r.get("grace_timeout"):
            print(f"### note {tag}: grace timeout from aiperf's leaked join (client bookkeeping, not a stall): {r.get('timeout_stats')}")
        for l in r.get("leak_lines") or []:
            print(f"###   aiperf: {l}")
    else:
        print(f"### score {tag}: no kept records (err={r.get('err')}, cancelled={r.get('cancelled')})")
    if "--json" in a:
        print(json.dumps(r, default=str))

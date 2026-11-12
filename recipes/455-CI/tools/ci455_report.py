"""Build the 455-CI report of one image, in English: highlights and a summary
table for the boss, and a detailed report with every number and each failure's
core error log (455-CI-guide.md §5)."""

import re
from dataclasses import dataclass

from ci455_data import LABEL

# Highlight thresholds (455-CI-guide.md §5.1).
PERF_UP = 0.03  # throughput up by this much against the last success: progress
PERF_DOWN = -0.03  # down by this much: regression
PERF_SEVERE = -0.10  # down by this much: severe regression
BEST_DRIFT = -0.05  # this far below the best earlier image: slow regression
ACC_WARN = 0.04  # GSM8K down by more than this (0.01 = one point): regression
ACC_SEVERE = 0.06
LOG_LINES = 60  # core error log lines shown in the detailed report
NONE = "No failures or major changes."
EXAMPLE = [
    (
        "> ⚠️ **EXAMPLE**: this report only illustrates the format. All numbers are made up; "
        "they are not real test results."
    ),
    "",
]


@dataclass
class ModelResult:
    scenario: str
    slug: str
    name: str
    perf: list  # PerfCell per concurrency
    acc: object  # AccCell
    error_log: str = ""  # runs/<YYYYMMDD>/<slug>/error.log


def pct(x):
    return f"{x * 100:+.1f}%".replace("-", "−")


def pts(x):
    v = round(x * 100, 1)
    return "0.0 pts" if v == 0 else f"{v:+.1f} pts".replace("-", "−")


def num(v):
    return f"{v:,.1f}" if v < 100 else f"{v:,.0f}"


def score(v):
    return f"{v * 100:.1f}%"


def perf_flag(delta):
    if delta is None:
        return ""
    d = round(delta, 6)
    if d <= PERF_SEVERE:
        return "🔴"
    if d <= PERF_DOWN:
        return "⚠️"
    if d >= PERF_UP:
        return "🟢"
    return ""


def acc_flag(delta):
    if delta is None:
        return ""
    d = round(delta, 6)
    if d < -ACC_SEVERE:
        return "🔴"
    if d < -ACC_WARN:
        return "⚠️"
    return ""


def short_name(name):
    """ "Kimi-K3 (2.8T)" -> "Kimi-K3": highlights keep to the point."""
    return re.sub(r"\s*\(.*\)\s*$", "", name)


def headline(r):
    """The cell the summary table shows: the highest concurrency (c64)."""
    return max(r.perf, key=lambda c: c.concurrency)


def _failed_parts(r):
    parts = [f"c{c.concurrency}" for c in r.perf if c.status == "fail"]
    if r.acc.status == "fail":
        parts.append("GSM8K")
    return parts


def _short(error, limit=60):
    line = (error or "").strip().splitlines()[0] if (error or "").strip() else ""
    return line if len(line) <= limit else line[: limit - 1] + "…"


RANK = {"": 0, "🟢": 0, "⚠️": 1, "🔴": 2}


def _phrase(cells, top, key, template, flag=perf_flag):
    """`template` (with {subject} and {pct}) for the headline concurrency
    (c64), whose subject is plain "throughput". The low concurrency is
    mentioned only when c64 has nothing to say, or when it is worse."""
    main = [c for c in cells if c is top]
    low = [c for c in cells if c is not top]
    if main and low and RANK[flag(key(low[0]))] <= RANK[flag(key(main[0]))]:
        low = []
    parts = [
        template.format(subject="throughput", pct=f"{abs(key(c)) * 100:.1f}%")
        for c in main
    ]
    parts += [
        template.format(
            subject=f"c{c.concurrency}" if main else f"c{c.concurrency} throughput",
            pct=f"{abs(key(c)) * 100:.1f}%",
        )
        for c in low
    ]
    return ", ".join(parts)


def highlights(results):
    """One short line per model with news, "<flag> <model>: <what happened>",
    most important first: failures, severe and plain regressions (throughput,
    then GSM8K), slow slides below the best, progress, recoveries. Flat models
    and first results are not news; with no news the list says so."""
    failed, severe, warn, drift, gain, recovered = [], [], [], [], [], []
    for r in results:
        who, top = short_name(r.name), headline(r)
        parts = _failed_parts(r)
        if parts:
            everything = all(c.status != "ok" for c in r.perf) and r.acc.status != "ok"
            errors = [c.error for c in r.perf if c.error] + (
                [r.acc.error] if r.acc.error else []
            )
            err = _short(errors[0]) if errors else ""
            what = "all runs failed" if everything else f"{' and '.join(parts)} failed"
            failed.append(f"❌ {who}: {what}" + (f" ({err})" if err else "") + ".")
            continue
        down = [c for c in r.perf if perf_flag(c.delta) in ("⚠️", "🔴")]
        if down:
            icon = "🔴" if any(perf_flag(c.delta) == "🔴" for c in down) else "⚠️"
            (severe if icon == "🔴" else warn).append(
                f"{icon} {who}: {_phrase(down, top, lambda c: c.delta, '{subject} down {pct}')}."
            )
        flag = acc_flag(r.acc.delta)
        if flag:
            (severe if flag == "🔴" else warn).append(
                f"{flag} {who}: GSM8K down {abs(r.acc.delta) * 100:.1f} points to {score(r.acc.value)}."
            )
        slide = [
            c
            for c in r.perf
            if c not in down
            and c.delta_best is not None
            and round(c.delta_best, 6) <= BEST_DRIFT
        ]
        if slide:
            drift.append(
                f"⚠️ {who}: "
                f"{_phrase(slide, top, lambda c: c.delta_best, '{subject} {pct} below its best')}"
                " (gradual decline)."
            )
        up = [c for c in r.perf if perf_flag(c.delta) == "🟢"]
        if up:
            gain.append(
                f"🟢 {who}: {_phrase(up, top, lambda c: c.delta, '{subject} up {pct}')}."
            )
        if any(c.prev_failed for c in r.perf) or r.acc.prev_failed:
            recovered.append(f"✅ {who}: recovered from the previous failure.")
    return failed + severe + warn + drift + gain + recovered or [NONE]


# ------------------------------------------------------------------ tables


def summary_row(r, date):
    """Model, scenario, previous and current throughput (c64 tok/s per GPU,
    each with its image date), change, GSM8K change. N/A: the run failed."""
    c, a = headline(r), r.acc
    prev = f"{num(c.prev)} ({c.prev_date})" if c.prev is not None else "—"
    if c.status == "fail":
        now, change = f"N/A ({date})", "N/A"
    elif c.value is None:
        now, change = "—", "—"
    else:
        now = f"{num(c.value)} ({date})"
        change = (
            "first"
            if c.delta is None
            else f"{perf_flag(c.delta)} {pct(c.delta)}".strip()
        )
    if a.status == "fail":
        acc = "N/A"
    elif a.value is None:
        acc = "—"
    elif a.delta is None:
        acc = f"first (now {score(a.value)})"
    else:
        acc = f"{acc_flag(a.delta)} {pts(a.delta)} (now {score(a.value)})".strip()
    return [r.name, LABEL[r.scenario], prev, now, change, acc]


def _perf_value(c):
    return {"ok": num(c.value) if c.value is not None else "—", "fail": "N/A"}.get(
        c.status, "—"
    )


def _perf_delta(c, delta):
    if c.status == "fail":
        return "N/A"
    if c.value is None:
        return "—"
    if delta is None:
        return "first"
    return f"{perf_flag(delta)} {pct(delta)}".strip()


def _best_delta(c):
    """Against the best earlier image: flagged only below the slow-slide line."""
    if c.status == "fail":
        return "N/A"
    if c.value is None:
        return "—"
    if c.delta_best is None:
        return "first"
    d = round(c.delta_best, 6)
    if d > 0:
        return f"{pct(d)} (new best)"
    return f"{'⚠️ ' if d <= BEST_DRIFT else ''}{pct(d)}"


def _acc_value(a):
    return {"ok": score(a.value) if a.value is not None else "—", "fail": "N/A"}.get(
        a.status, "—"
    )


def _acc_delta(a):
    if a.status == "fail":
        return "N/A"
    if a.value is None:
        return "—"
    if a.delta is None:
        return "first"
    return f"{acc_flag(a.delta)} {pts(a.delta)}".strip()


def _sha(env, repo):
    return ((env.get(repo) or {}).get("commit") or "")[:9] or "—"


def _image_short(image):
    return (image or "—").split(":")[-1]


def summary_md(date, env, results, pending, detail_name, mock=False):
    out = [
        f"# 455-CI report · {date}" + (" · EXAMPLE" if mock else ""),
        "",
        *(EXAMPLE if mock else []),
        (
            f"Image `{_image_short(env.get('image'))}` · ATOM `main@{_sha(env, 'atom')}`"
            f" · aiter `main@{_sha(env, 'aiter')}` · base `{_image_short(env.get('base_image'))}`"
            f" · [Detailed report]({detail_name})"
        ),
        "",
        "## Highlights",
        "",
        *[f"- {h}" for h in highlights(results)],
        "",
        "## Summary",
        "",
        "| Model | Scenario | Previous (tok/s/GPU) | Current (tok/s/GPU) | Change | GSM8K change |",
        "|---|---|---|---|---|---|",
    ]
    out += ["| " + " | ".join(summary_row(r, date)) + " |" for r in results]
    out += [
        f"| {label} models | {label} | pending (midweek) | | | |" for label in pending
    ]
    return "\n".join(out) + "\n"


def _env_rows(env):
    def commit(repo, url):
        c = env.get(repo) or {}
        sha = c.get("commit") or ""
        if not sha:
            return "—"
        tail = " ".join(x for x in (c.get("date"), c.get("title")) if x)
        return f"`main` @ [`{sha[:9]}`](https://github.com/{url}/commit/{sha})" + (
            f", {tail}" if tail else ""
        )

    digest = f" (digest `{env['digest']}`)" if env.get("digest") else ""
    dates = "; ".join(
        f"{LABEL[s]} {d}" for s, d in (env.get("test_dates") or {}).items()
    )
    return [
        "| Item | Value |",
        "|---|---|",
        f"| Test image | `{env.get('image') or '—'}`{digest} |",
        f"| Base image | `{env.get('base_image') or '—'}` |",
        f"| ATOM | {commit('atom', 'ROCm/ATOM')} |",
        f"| aiter | {commit('aiter', 'ROCm/aiter')} |",
        f"| Dependency changes | {', '.join(env.get('dependency_changes') or []) or 'none'} |",
        f"| Test dates | {dates or '—'} (Beijing time) |",
    ]


SECOND_HEADER = {
    "endpoint": "E2E avg interactivity (tok/s/user)",
    "8k1k": "ITL P50 (ms, lower is better)",
}


def detail_md(
    date, env, results, pending, summary_name, charts_dir, runs_dir, mock=False
):
    """`charts_dir` and `runs_dir` are relative to the report's directory."""
    out = [
        f"# 455-CI detailed report · {date}" + (" · EXAMPLE" if mock else ""),
        "",
        *(EXAMPLE if mock else []),
        f"[← Report]({summary_name})",
        "",
        "## Environment",
        "",
        *_env_rows(env),
    ]
    for scenario in ("endpoint", "8k1k"):
        label = LABEL[scenario]
        out += ["", f"## {label}", ""]
        if label in pending:
            out.append("Pending (midweek).")
            continue
        mine = [r for r in results if r.scenario == scenario]
        out += [
            "### Performance",
            "",
            f"| Model | GPUs | Concurrency | tok/s/GPU | vs previous | vs best | {SECOND_HEADER[scenario]} | vs previous |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in mine:
            for c in r.perf:
                sv = (
                    num(c.second)
                    if c.second is not None
                    else ("N/A" if c.status == "fail" else "—")
                )
                sd = (
                    pct(c.delta_second)
                    if c.delta_second is not None
                    else ("first" if c.second is not None else "—")
                )
                out.append(
                    f"| {r.name} | {c.gpus or '—'} | {c.concurrency} | {_perf_value(c)} | "
                    f"{_perf_delta(c, c.delta)} | {_best_delta(c)} | {sv} | {sd} |"
                )
        out += [
            "",
            "### Accuracy (GSM8K, 3-shot)",
            "",
            "| Model | flexible-extract | strict-match | Previous | vs previous |",
            "|---|---|---|---|---|",
        ]
        for r in mine:
            a = r.acc
            strict = score(a.strict) if a.strict is not None else "—"
            prev = f"{score(a.prev)} ({a.prev_date})" if a.prev is not None else "—"
            out.append(
                f"| {r.name} | {_acc_value(a)} | {strict} | {prev} | {_acc_delta(a)} |"
            )
        out += [
            "",
            "### Trends",
            "",
            f"![{label} throughput per GPU]({charts_dir}/perf-{scenario}.png)",
            "",
            f"![{label} GSM8K]({charts_dir}/accuracy-{scenario}.png)",
        ]
    out += ["", "## Failures", ""]
    broken = [r for r in results if _failed_parts(r)]
    if not broken:
        out.append("No failures.")
    for r in broken:
        errors = [c.error for c in r.perf if c.error] + (
            [r.acc.error] if r.acc.error else []
        )
        out += [
            f"### {r.name} ({LABEL[r.scenario]}): {', '.join(_failed_parts(r))}",
            "",
            f"- Error: `{_short(errors[0], 120) if errors else '—'}`",
            f"- Full excerpt: [error.log]({runs_dir}/{r.slug}/error.log)",
            "",
        ]
        log = (r.error_log or "").rstrip("\n").splitlines()
        if log:
            more = (
                [f"… ({len(log)} lines in all; see error.log)"]
                if len(log) > LOG_LINES
                else []
            )
            out += [
                "Core error log:",
                "",
                "```text",
                *log[:LOG_LINES],
                *more,
                "```",
                "",
            ]
        else:
            out += ["No error.log found; add one as in the guide's §3.4.", ""]
    return "\n".join(out).rstrip("\n") + "\n"

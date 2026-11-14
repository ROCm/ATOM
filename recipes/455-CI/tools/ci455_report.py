"""Build the 455-CI result of one session, in English and in one file: a short
summary for the boss (highlights and a table), the session's findings, then the
details (environment, every number, trend charts, follow-ups and each failure's
core error log) (455-CI-guide.md §5)."""

import re
from dataclasses import dataclass

from ci455_data import LABEL

# Highlight thresholds (455-CI-guide.md §5.1).
PERF_UP = 0.03  # throughput up by this much against the last success: progress
PERF_DOWN = -0.03  # down by this much: regression
PERF_SEVERE = -0.10  # down by this much: severe regression
ACC_WARN = 0.04  # GSM8K down by more than this (0.01 = one point): regression
ACC_SEVERE = 0.06
LOG_LINES = 60  # core error log lines shown in the details
NONE = "No failures or major changes."
EXAMPLE = [
    (
        "> ⚠️ **EXAMPLE**: this file only illustrates the format. All numbers are made up; "
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


def _news(results):
    """(category, slug, line) per piece of news, most important first:
    failures, severe and plain regressions (throughput, then GSM8K),
    progress, recoveries. Everything is against the previous success."""
    items = []
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
            items.append(
                (
                    0,
                    r.slug,
                    f"❌ {who}: {what}" + (f" ({err})" if err else "") + ".",
                )
            )
            continue
        down = [c for c in r.perf if perf_flag(c.delta) in ("⚠️", "🔴")]
        if down:
            icon = "🔴" if any(perf_flag(c.delta) == "🔴" for c in down) else "⚠️"
            items.append(
                (
                    1 if icon == "🔴" else 2,
                    r.slug,
                    f"{icon} {who}: {_phrase(down, top, lambda c: c.delta, '{subject} down {pct}')}.",
                )
            )
        flag = acc_flag(r.acc.delta)
        if flag:
            items.append(
                (
                    1 if flag == "🔴" else 2,
                    r.slug,
                    f"{flag} {who}: GSM8K down {abs(r.acc.delta) * 100:.1f} points to {score(r.acc.value)}.",
                )
            )
        up = [c for c in r.perf if perf_flag(c.delta) == "🟢"]
        if up:
            items.append(
                (
                    4,
                    r.slug,
                    f"🟢 {who}: {_phrase(up, top, lambda c: c.delta, '{subject} up {pct}')}.",
                )
            )
        # Only what ran this session can have recovered: a model left untested
        # (say its machine went down, guide §3.6) has no rows and no news.
        ran = [c for c in r.perf if c.status == "ok"]
        ran += [r.acc] if r.acc.status == "ok" else []
        if any(c.prev_failed for c in ran):
            items.append((5, r.slug, f"✅ {who}: recovered from the previous failure."))
    return sorted(
        items, key=lambda item: item[0]
    )  # stable: model order within a category


# Highlight categories of _news that are problems (failures, severe and plain
# regressions). Each needs a follow-up in followups.csv (455-CI-guide.md §3.7).
PROBLEM = (0, 1, 2)
ACTIONS = ("open_pr", "our_pr", "repro")
PR_URL = re.compile(r"https?://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)")


def pr_label(url):
    """ "https://github.com/ROCm/ATOM/pull/123" -> "ROCm/ATOM#123"."""
    m = PR_URL.match(url)
    return f"{m.group(1)}#{m.group(2)}" if m else url


def _prs(links):
    """Space-separated PR links (a fix may take several, e.g. an ATOM PR and
    an aiter PR together), each shown as "ROCm/ATOM#123", joined by "+"."""
    shown = [
        f"[{pr_label(u)}]({u})" if PR_URL.match(u) else f"<{u}>" for u in links.split()
    ]
    return " + ".join(shown) or "(no link)"


def _follow_text(f):
    """A problem's next step, from its followups.csv row."""
    action, link = (f or {}).get("action"), ((f or {}).get("link") or "").strip()
    if action == "open_pr":
        return f"Fix in progress: {_prs(link)}."
    if action == "our_pr":
        return f"Fixed by us in {_prs(link)}, verified."
    if action == "repro":
        return f"No fix yet; [repro]({link})."
    return ""


def highlights(results, followups=None):
    """One short line per piece of news, "<flag> <model>: <what happened>". A
    problem (❌ 🔴 ⚠️) goes on with its next step from followups.csv. Flat
    models and first results are not news; with no news the list says so."""
    lines = []
    for category, slug, line in _news(results):
        step = _follow_text((followups or {}).get(slug)) if category in PROBLEM else ""
        lines.append(" ".join(x for x in (line, step) if x))
    return lines or [NONE]


def missing_followups(results, followups):
    """Models with a problem in the highlights but no usable follow-up: one of
    open_pr, our_pr or repro, with a link (455-CI-guide.md §3.7)."""
    problems = {slug for category, slug, _ in _news(results) if category in PROBLEM}
    missing = []
    for r in results:
        f = (followups or {}).get(r.slug) or {}
        if r.slug in problems and (
            f.get("action") not in ACTIONS or not (f.get("link") or "").strip()
        ):
            missing.append(r.name)
    return missing


# ------------------------------------------------------------------ tables


def previous_column(results):
    """The previous session's date: what the summary's previous column is
    headed with."""
    return max(
        (headline(r).prev_date for r in results if headline(r).prev_date), default=None
    )


def summary_row(r, date, ref):
    """Model, scenario, previous and current c64 tok/s per GPU (dated only
    when the previous one is not from the `ref` session the column is headed
    with), the delta as a percentage, GSM8K delta. N/A: the run failed."""
    c, a = headline(r), r.acc
    prev = (
        "—"
        if c.prev is None
        else num(c.prev) + (f" ({c.prev_date})" if c.prev_date != ref else "")
    )
    if c.status == "fail":
        now, delta = "N/A", "N/A"
    elif c.value is None:
        now, delta = "—", "—"
    else:
        now = num(c.value)
        if c.delta is None:
            delta = "first"
        else:
            delta = f"{perf_flag(c.delta)} {pct(c.delta)}".strip()
    if a.status == "fail":
        acc = "N/A"
    elif a.value is None:
        acc = "—"
    elif a.delta is None:
        acc = f"first (now {score(a.value)})"
    else:
        acc = f"{acc_flag(a.delta)} {pts(a.delta)} (now {score(a.value)})".strip()
        if a.prev_date and a.prev_date != ref:
            acc += f" (vs {a.prev_date})"
    return [r.name, LABEL[r.scenario], prev, now, delta, acc]


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


def _labels(results):
    return [
        LABEL[sc]
        for sc in ("endpoint", "8k1k")
        if any(r.scenario == sc for r in results)
    ]


def _summary(date, results, followups):
    """The short summary, for the boss: the highlights, then each model's c64
    throughput against the previous session and its GSM8K change."""
    ref = previous_column(results)
    return [
        "## Summary",
        "",
        "### Highlights",
        "",
        *[f"- {h}" for h in highlights(results, followups)],
        "",
        "### Results",
        "",
        f"| Model | Scenario | {ref or '—'} (tok/s/GPU) | {date} (tok/s/GPU) | Delta | GSM8K delta |",
        "|---|---|---|---|---|---|",
        *["| " + " | ".join(summary_row(r, date, ref)) + " |" for r in results],
    ]


def _env_rows(env, date):
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
    return [
        "| Item | Value |",
        "|---|---|",
        f"| Test image | `{env.get('image') or '—'}`{digest} |",
        f"| ATOM | {commit('atom', 'ROCm/ATOM')} |",
        f"| aiter | {commit('aiter', 'ROCm/aiter')} |",
        f"| Test date | {date} |",
    ]


SECOND_HEADER = {
    "endpoint": "E2E avg interactivity (tok/s/user)",
    "8k1k": "ITL P50 (ms, lower is better)",
}


def _vs(cell, text, ref, delta):
    """A change cell; named by date when its comparison is not the column's."""
    if delta is not None and cell.prev_date and cell.prev_date != ref:
        return f"{text} (vs {cell.prev_date})"
    return text


def _details(date, env, results, charts_dir, runs_dir, followups=None):
    """The detailed summary: environment, every number, trends and failures.
    Changes are against the previous session, whose date heads the column; a
    model whose last success is older says so in its cell."""
    out = ["## Details", "", "### Environment", "", *_env_rows(env, date)]
    for scenario in ("endpoint", "8k1k"):
        mine = [r for r in results if r.scenario == scenario]
        if not mine:
            continue
        label = LABEL[scenario]
        ref = max(
            (c.prev_date for r in mine for c in r.perf if c.prev_date), default=None
        )
        vs = f"vs {ref}" if ref else "vs —"
        out += [
            "",
            f"### Performance · {label}",
            "",
            f"| Model | Parallelism | Concurrency | tok/s/GPU | {vs} | {SECOND_HEADER[scenario]} | {vs} |",
            "|---|---|---|---|---|---|---|",
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
                    f"| {r.name} | {c.parallel or '—'} | {c.concurrency} | {_perf_value(c)} | "
                    f"{_vs(c, _perf_delta(c, c.delta), ref, c.delta)} | {sv} | "
                    f"{_vs(c, sd, ref, c.delta_second)} |"
                )
        aref = max((r.acc.prev_date for r in mine if r.acc.prev_date), default=None)
        avs = f"vs {aref}" if aref else "vs —"
        out += [
            "",
            f"### Accuracy (GSM8K, 3-shot) · {label}",
            "",
            f"| Model | flexible-extract | strict-match | {aref or '—'} | {avs} |",
            "|---|---|---|---|---|",
        ]
        for r in mine:
            a = r.acc
            strict = score(a.strict) if a.strict is not None else "—"
            prev = (
                "—"
                if a.prev is None
                else score(a.prev)
                + (f" ({a.prev_date})" if a.prev_date != aref else "")
            )
            out.append(
                f"| {r.name} | {_acc_value(a)} | {strict} | {prev} | {_vs(a, _acc_delta(a), aref, a.delta)} |"
            )
        out += [
            "",
            f"### Trends · {label}",
            "",
            f"![{label} throughput per GPU]({charts_dir}/perf-{scenario}.png)",
            "",
            f"![{label} GSM8K]({charts_dir}/accuracy-{scenario}.png)",
        ]
    problems = {slug for category, slug, _ in _news(results) if category in PROBLEM}
    if problems:
        missing = set(missing_followups(results, followups))
        out += [
            "",
            "### Follow-ups",
            "",
            "| Model | Next step | Note |",
            "|---|---|---|",
        ]
        for r in results:
            if r.slug not in problems:
                continue
            f = (followups or {}).get(r.slug) or {}
            step = (
                "Missing: needs a follow-up" if r.name in missing else _follow_text(f)
            )
            note = (f.get("note") or "").strip().replace("|", "\\|") or "—"
            out.append(f"| {r.name} | {step} | {note} |")
    out += ["", "### Failures", ""]
    broken = [r for r in results if _failed_parts(r)]
    if not broken:
        out.append("No failures.")
    for r in broken:
        errors = [c.error for c in r.perf if c.error] + (
            [r.acc.error] if r.acc.error else []
        )
        out += [
            f"#### {r.name} ({LABEL[r.scenario]}): {', '.join(_failed_parts(r))}",
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
    return out


def report_md(
    date,
    env,
    results,
    charts_dir,
    runs_dir,
    mock=False,
    followups=None,
    findings=None,
):
    """One session's result: the short summary (highlights and the table), the
    findings written down during the session, then the details. `charts_dir`
    and `runs_dir` are relative to the result file's directory."""
    out = [
        f"# 455-CI result · {date} · {' + '.join(_labels(results))}"
        + (" · EXAMPLE" if mock else ""),
        "",
        *(EXAMPLE if mock else []),
        (
            f"Image `{_image_short(env.get('image'))}` · ATOM `main@{_sha(env, 'atom')}`"
            f" · aiter `main@{_sha(env, 'aiter')}`"
        ),
        "",
        *_summary(date, results, followups),
        "",
        "## Findings",
        "",
        (findings or "").strip() or "None recorded.",
        "",
        *_details(date, env, results, charts_dir, runs_dir, followups),
    ]
    return "\n".join(out).rstrip("\n") + "\n"

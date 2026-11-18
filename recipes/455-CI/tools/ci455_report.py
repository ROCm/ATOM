"""Build the 455-CI result of one session, in English and in one file, the MI455
workload weekly status report: the lowlights and highlights for the boss (each
left out when the session has none), the results table, the environment, the
session's findings, then the details (every number, trend charts, follow-ups
and each failure's core error log) (455-CI-guide.md §5)."""

import re
from dataclasses import dataclass, field

from ci455_data import LABEL

# Lowlight thresholds (455-CI-guide.md §5.1).
PERF_UP = 0.03  # throughput up by this much against the last success: progress
PERF_DOWN = -0.03  # down by this much: regression
PERF_SEVERE = -0.10  # down by this much: severe regression
ACC_WARN = 0.04  # GSM8K down by more than this (0.01 = one point): regression
ACC_SEVERE = 0.06
LOG_LINES = 60  # core error log lines shown in the details
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
    # (file name, text) of the commands run, runs/<YYYYMMDD>/<slug>/*_cmd*.sh
    commands: list = field(default_factory=list)
    # The model's known-good GSM8K (flexible-extract), accuracy_baseline.csv
    baseline: float | None = None


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
    """ "Kimi-K3 (2.8T)" -> "Kimi-K3": lowlights keep to the point."""
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
    """(category, slug, flag, model, what happened, err) per piece of news,
    most important first: failures, severe and plain regressions (throughput,
    then GSM8K), progress, recoveries. Everything is against the previous
    success. `err` is a failure's short error (its cause when the follow-up
    gives none); other news has `err` None."""
    items = []
    for r in results:
        who, top = short_name(r.name), headline(r)
        parts = _failed_parts(r)
        if parts:
            # "all runs" counts only what ran: a GSM8K-only session (Wednesday)
            # that fails says "GSM8K failed".
            tried = [c for c in r.perf if c.status != "missing"]
            tried += [r.acc] if r.acc.status != "missing" else []
            everything = len(tried) > 1 and all(x.status == "fail" for x in tried)
            errors = [c.error for c in r.perf if c.error] + (
                [r.acc.error] if r.acc.error else []
            )
            err = _short(errors[0]) if errors else ""
            what = "all runs failed" if everything else f"{' and '.join(parts)} failed"
            items.append((0, r.slug, "❌", who, what, err))
            continue
        down = [c for c in r.perf if perf_flag(c.delta) in ("⚠️", "🔴")]
        if down:
            icon = "🔴" if any(perf_flag(c.delta) == "🔴" for c in down) else "⚠️"
            what = _phrase(down, top, lambda c: c.delta, "{subject} down {pct}")
            items.append((1 if icon == "🔴" else 2, r.slug, icon, who, what, None))
        flag = acc_flag(r.acc.delta)
        if flag:
            what = f"GSM8K down {abs(r.acc.delta) * 100:.1f} points to {score(r.acc.value)}"
            items.append((1 if flag == "🔴" else 2, r.slug, flag, who, what, None))
        elif r.baseline is not None and r.acc.value is not None:
            # Nothing against the last session (say, the model's first run):
            # hold GSM8K to its known-good value instead.
            below = r.baseline - r.acc.value
            flag = acc_flag(-below)
            if flag:
                what = (
                    f"GSM8K {score(r.acc.value)}, {below * 100:.1f} points below "
                    f"the known-good {score(r.baseline)}"
                )
                items.append((1 if flag == "🔴" else 2, r.slug, flag, who, what, None))
        up = [c for c in r.perf if perf_flag(c.delta) == "🟢"]
        if up:
            what = _phrase(up, top, lambda c: c.delta, "{subject} up {pct}")
            items.append((4, r.slug, "🟢", who, what, None))
        # Only what ran this session can have recovered: a model left untested
        # (say its machine went down, guide §3.6) has no rows and no news.
        ran = [c for c in r.perf if c.status == "ok"]
        ran += [r.acc] if r.acc.status == "ok" else []
        if any(c.prev_failed for c in ran):
            what = "recovered from the previous failure"
            items.append((5, r.slug, "✅", who, what, None))
    return sorted(
        items, key=lambda item: item[0]
    )  # stable: model order within a category


# Categories of _news that are problems, the lowlights (failures, severe and plain
# regressions). Each needs a follow-up in followups.csv (455-CI-guide.md §3.7).
PROBLEM = (0, 1, 2)
# open_pr: someone's open PR fixes it; our_pr: we fixed and verified it (the
# links may include others' PRs it needs); merged_pr: the fix reached main after
# the image was built; repro: no fix yet, a repro doc (455-CI-guide.md §3.7).
ACTIONS = ("open_pr", "our_pr", "merged_pr", "repro")
PR_URL = re.compile(r"https?://github\.com/([\w.-]+/[\w.-]+)/pull/(\d+)")


def pr_label(url):
    """ "https://github.com/ROCm/ATOM/pull/123" -> "ROCm/ATOM#123"."""
    m = PR_URL.match(url)
    return f"{m.group(1)}#{m.group(2)}" if m else url


def _prs(links):
    """Space-separated PR links (a fix may take several, e.g. an ATOM PR and
    an aiter PR together), joined by "+". The first PR of a repo is shown as
    "ROCm/ATOM#123", the next ones of the same repo as "#124"."""
    shown, seen = [], set()
    for u in links.split():
        m = PR_URL.match(u)
        if not m:
            shown.append(f"<{u}>")
            continue
        label = f"#{m.group(2)}" if m.group(1) in seen else pr_label(u)
        seen.add(m.group(1))
        shown.append(f"[{label}]({u})")
    return " + ".join(shown) or "(no link)"


def _follow_text(f):
    """A problem's fix, from its followups.csv row: short, because the
    lowlights go to the boss as they are."""
    action, link = (f or {}).get("action"), ((f or {}).get("link") or "").strip()
    if action == "open_pr":
        return f"Needs {_prs(link)} (open)"
    if action == "our_pr":
        return f"Passes with {_prs(link)} (verified)"
    if action == "merged_pr":
        return f"Fixed on main by {_prs(link)} (merged after the image was built)"
    if action == "repro":
        return f"No fix yet; [reproducer]({link})"
    return ""


LOWLIGHT_COLUMNS = ("Model", "What happened", "Cause", "Fix", "Owner")
HIGHLIGHT_COLUMNS = ("Model", "What happened")
GOOD_NEWS = (4, 5)  # categories of _news: gains, recoveries


def lowlights(results, followups=None):
    """The rows of the lowlights table, one per problem (❌ 🔴 ⚠️): the flag
    and the short model name, what happened, its cause, its fix and a blank
    owner. The cause is followups.csv's `problem` (a failure without one: its
    short error), the fix comes from the follow-up's action and links. The
    owner is always left blank for the user to fill in."""
    rows = []
    for category, slug, flag, who, what, err in _news(results):
        if category not in PROBLEM:
            continue
        f = (followups or {}).get(slug) or {}
        code = (err or "").replace("`", "'")
        cause = (f.get("problem") or "").strip() or (f"`{code}`" if code else "")
        rows.append([f"{flag} {who}", what, cause or "—", _follow_text(f) or "—", ""])
    return rows


def highlights(results):
    """The rows of the highlights table, one per piece of good news (🟢 a gain,
    ✅ a recovery): the flag and the short model name and what happened. Only
    the lowlights have an owner column."""
    return [
        [f"{flag} {who}", what]
        for category, _, flag, who, what, _ in _news(results)
        if category in GOOD_NEWS
    ]


def _table(columns, rows):
    """A markdown table, or nothing when there are no rows."""
    if not rows:
        return []
    return [_md_row(columns), "|" + "---|" * len(columns), *map(_md_row, rows)]


def lowlights_md(results, followups=None):
    """The lowlights table; empty when nothing went wrong."""
    return _table(LOWLIGHT_COLUMNS, lowlights(results, followups))


def highlights_md(results):
    """The highlights table; empty when there is no good news."""
    return _table(HIGHLIGHT_COLUMNS, highlights(results))


def _md_row(cells):
    """A markdown table row; a "|" inside a cell is escaped."""
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def missing_followups(results, followups):
    """Models with a problem in the lowlights but no usable follow-up: one of
    the ACTIONS, with a link (455-CI-guide.md §3.7)."""
    problems = {slug for category, slug, *_ in _news(results) if category in PROBLEM}
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


def perf_ran(results):
    """Whether the session measured performance at all: Wednesdays only run
    GSM8K, so their results table and details show accuracy only."""
    return any(c.status != "missing" for r in results for c in r.perf)


def accuracy_row(r, ref):
    """Model, previous and current GSM8K (flexible-extract; the previous one
    dated only when it is not from the `ref` session the column is headed
    with), the change in points. N/A: GSM8K failed. No scenario: in a session
    that measured accuracy only, the performance scenario would only confuse."""
    a = r.acc
    prev = (
        "—"
        if a.prev is None
        else score(a.prev) + (f" ({a.prev_date})" if a.prev_date != ref else "")
    )
    if a.status == "fail":
        now, delta = "N/A", "N/A"
    elif a.value is None:
        now, delta = "—", "—"
    else:
        now = score(a.value)
        delta = (
            "first"
            if a.delta is None
            else f"{acc_flag(a.delta)} {pts(a.delta)}".strip()
        )
    return [r.name, prev, now, delta]


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


def _branch_commit(env, repo):
    """`main` @ `<full sha>` (<commit date>): what the result's first lines say."""
    c = env.get(repo) or {}
    sha = c.get("commit") or ""
    if not sha:
        return "—"
    date = f" ({c['date']})" if c.get("date") else ""
    return f"`{c.get('branch') or 'main'}` @ `{sha}`{date}"


def _lights_and_results(date, env, results, followups):
    """The part for the boss: the lowlights and the highlights (each left out
    when the session has none), each model's c64 throughput and GSM8K against
    the previous session, then the image and the commits it was built from."""
    ref = previous_column(results)
    out = []
    for title, table in (
        ("Lowlights", lowlights_md(results, followups)),
        ("Highlights", highlights_md(results)),
    ):
        if table:
            out += [f"## {title}", "", *table, ""]
    if perf_ran(results):
        table = [
            f"| Model | Scenario | {ref or '—'} (tok/s/GPU) | {date} (tok/s/GPU) | Delta | GSM8K delta |",
            "|---|---|---|---|---|---|",
            *["| " + " | ".join(summary_row(r, date, ref)) + " |" for r in results],
        ]
    else:
        aref = max((r.acc.prev_date for r in results if r.acc.prev_date), default=None)
        table = [
            f"| Model | {aref or '—'} (GSM8K) | {date} (GSM8K) | Delta |",
            "|---|---|---|---|",
            *["| " + " | ".join(accuracy_row(r, aref)) + " |" for r in results],
        ]
    return out + [
        "## Results",
        "",
        *table,
        "",
        "## Environment",
        "",
        f"- Image: `{env.get('image') or '—'}`",
        f"- ATOM: {_branch_commit(env, 'atom')}",
        f"- aiter: {_branch_commit(env, 'aiter')}",
    ]


def _env_rows(env, date):
    def commit(repo, url):
        c = env.get(repo) or {}
        sha = c.get("commit") or ""
        if not sha:
            return "—"
        tail = " ".join(x for x in (c.get("date"), c.get("title")) if x)
        branch = c.get("branch") or "main"
        return f"`{branch}` @ [`{sha}`](https://github.com/{url}/commit/{sha})" + (
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
    out = ["## Details", "", "### Environment details", "", *_env_rows(env, date)]
    for scenario in ("endpoint", "8k1k"):
        mine = [r for r in results if r.scenario == scenario]
        if not mine:
            continue
        label = LABEL[scenario]
        ref = max(
            (c.prev_date for r in mine for c in r.perf if c.prev_date), default=None
        )
        vs = f"vs {ref}" if ref else "vs —"
        measured = perf_ran(mine)  # a GSM8K-only session (Wednesday) has none
        if measured:
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
            *(
                [f"![{label} throughput per GPU]({charts_dir}/perf-{scenario}.png)", ""]
                if measured
                else []
            ),
            f"![{label} GSM8K]({charts_dir}/accuracy-{scenario}.png)",
        ]
    problems = {slug for category, slug, *_ in _news(results) if category in PROBLEM}
    if problems:
        missing = set(missing_followups(results, followups))
        out += [
            "",
            "### Follow-ups",
            "",
            "| Model | Problem | Next step | Note |",
            "|---|---|---|---|",
        ]
        for r in results:
            if r.slug not in problems:
                continue
            f = (followups or {}).get(r.slug) or {}
            step = (
                "Missing: needs a follow-up" if r.name in missing else _follow_text(f)
            )
            note = (f.get("note") or "").strip() or "—"
            why = (f.get("problem") or "").strip() or "—"
            out.append(_md_row([r.name, why, step, note]))
    out += ["", "### Failures", ""]
    shared = {}  # scenario -> the MLPerf Endpoint client its bench commands share
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
        out += _repro(r, env, shared)
    for scenario, client in shared.items():
        out += [
            f"#### Shared {LABEL[scenario]} client",
            "",
            "Each bench command above sets its model's variables and then runs this, on the host (455-CI-recipe.md):",
            "",
            "```bash",
            client,
            "```",
            "",
        ]
    return out


def _split_client(text):
    """An MLPerf Endpoint bench_cmd.sh: (its model's own lines, the shared client
    from the recipe that follows them), or (the whole text, "") for any other."""
    lines = text.rstrip("\n").splitlines()
    for i, line in enumerate(lines):
        if line.startswith(("# point_yaml", "point_yaml() {")):
            return "\n".join(lines[:i]).rstrip(), "\n".join(lines[i:])
    return text.rstrip("\n"), ""


def _repro(r, env, shared):
    """How to reproduce a failure: the test container, then the commands as
    run (runs/<YYYYMMDD>/<slug>/*_cmd*.sh, internal addresses replaced). The
    MLPerf Endpoint client they share is shown once, after all failures."""
    if not r.commands:
        return [
            "No commands recorded (runs/<YYYYMMDD>/<slug>/*_cmd*.sh, guide §3.1).",
            "",
        ]
    out = [
        "Reproduce:",
        "",
        "```bash",
        "# the test container (455-CI-recipe.md); the server and GSM8K commands below run inside it",
        f"docker run -d --name ci455-{r.slug} --network host --ipc host --shm-size 128g \\",
        "  --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE \\",
        "  --security-opt seccomp=unconfined -v <weights dir>:/models/<model dir>:ro -v <results dir>:/results \\",
        f"  {env.get('image') or '<image>'}",
        "```",
        "",
    ]
    for name, text in r.commands:
        own, client = (
            _split_client(text) if name.startswith("bench") else (text.rstrip("\n"), "")
        )
        if client:
            shared.setdefault(r.scenario, client)
            own += "\n# ...then the shared client below"
        out += [f"`{name}`:", "", "```bash", own, "```", ""]
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
    """One session's result: the lights, results and environment, the
    findings written down during the session, then the details. `charts_dir`
    and `runs_dir` are relative to the result file's directory."""
    out = [
        f"# {date} MI455 Workload Weekly Status Report"
        + (" · EXAMPLE" if mock else ""),
        "",
        *(EXAMPLE if mock else []),
        *_lights_and_results(date, env, results, followups),
        "",
        "## Findings",
        "",
        (findings or "").strip() or "None recorded.",
        "",
        *_details(date, env, results, charts_dir, runs_dir, followups),
    ]
    return "\n".join(out).rstrip("\n") + "\n"

"""Build the 455-CI report of one image: highlights and a summary table for
the boss, and a detailed report with every number and each failure's core
error log (455-CI-guide.md §5)."""

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
EXAMPLE = [
    "> ⚠️ **EXAMPLE（示例）**：这是演示报告格式的例子，所有数字都是编的，不是真实测试数据。",
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
    return "0.0 个点" if v == 0 else f"{v:+.1f} 个点".replace("-", "−")


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


def _phrase(cells, top, key, verb, flag=perf_flag):
    """ "吞吐<verb> x%" for the headline concurrency (c64). The low concurrency
    is mentioned only when c64 has nothing to say, or when it is worse."""
    main = [c for c in cells if c is top]
    low = [c for c in cells if c is not top]
    if main and low and RANK[flag(key(low[0]))] <= RANK[flag(key(main[0]))]:
        low = []
    parts = [f"吞吐{verb} {abs(key(c)) * 100:.1f}%" for c in main]
    parts += [
        f"{'低并发' if main else '低并发吞吐'}{verb} {abs(key(c)) * 100:.1f}%"
        for c in low
    ]
    return "，".join(parts)


def highlights(results, pending=()):
    """One short line per model with news, "<flag> <model>：<what happened>",
    most important first: failures, severe and plain regressions (throughput,
    then GSM8K), slow slides below the best, progress, recoveries, first
    results; then one line for everything flat and the scenarios not run."""
    failed, severe, warn, drift, gain, recovered, new, quiet = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        0,
    )
    for r in results:
        who, top = short_name(r.name), headline(r)
        parts = _failed_parts(r)
        if parts:
            everything = all(c.status != "ok" for c in r.perf) and r.acc.status != "ok"
            errors = [c.error for c in r.perf if c.error] + (
                [r.acc.error] if r.acc.error else []
            )
            err = _short(errors[0]) if errors else ""
            what = "全部跑挂" if everything else f"{'、'.join(parts)} 跑挂"
            failed.append(f"❌ {who}：{what}" + (f"（{err}）" if err else "") + "。")
            continue
        said = False
        down = [c for c in r.perf if perf_flag(c.delta) in ("⚠️", "🔴")]
        if down:
            icon = "🔴" if any(perf_flag(c.delta) == "🔴" for c in down) else "⚠️"
            (severe if icon == "🔴" else warn).append(
                f"{icon} {who}：{_phrase(down, top, lambda c: c.delta, '回退')}。"
            )
            said = True
        flag = acc_flag(r.acc.delta)
        if flag:
            (severe if flag == "🔴" else warn).append(
                f"{flag} {who}：GSM8K 掉了 {abs(r.acc.delta) * 100:.1f} 个点，现在 {score(r.acc.value)}。"
            )
            said = True
        slide = [
            c
            for c in r.perf
            if c not in down
            and c.delta_best is not None
            and round(c.delta_best, 6) <= BEST_DRIFT
        ]
        if slide:
            drift.append(
                f"⚠️ {who}：{_phrase(slide, top, lambda c: c.delta_best, '比历史最佳低')}（累计小幅下降）。"
            )
            said = True
        up = [c for c in r.perf if perf_flag(c.delta) == "🟢"]
        if up:
            gain.append(f"🟢 {who}：{_phrase(up, top, lambda c: c.delta, '提升')}。")
            said = True
        if any(c.prev_failed for c in r.perf) or r.acc.prev_failed:
            recovered.append(f"✅ {who}：上次跑挂，这次恢复。")
            said = True
        if all(c.prev is None for c in r.perf) and r.acc.prev is None:
            new.append(who)
            said = True
        quiet += not said
    lines = failed + severe + warn + drift + gain + recovered
    if new:
        lines.append(f"🆕 首次有数据：{'、'.join(new)}。")
    if quiet:
        lines.append(
            f"{'全部' if quiet == len(results) else '其余'} {quiet} 个模型持平。"
        )
    lines += [f"{label} 待测（周中）。" for label in pending]
    return lines


# ------------------------------------------------------------------ tables


def summary_row(r):
    """模型、场景、上次性能、本次性能、变化、精度变化. 性能 is the c64 tok/s
    per GPU; N/A means the run failed."""
    c, a = headline(r), r.acc
    prev = num(c.prev) if c.prev is not None else "—"
    if c.status == "fail":
        now, change = "N/A", "N/A"
    elif c.value is None:
        now, change = "—", "—"
    else:
        now = num(c.value)
        change = (
            "首次"
            if c.delta is None
            else f"{perf_flag(c.delta)} {pct(c.delta)}".strip()
        )
    if a.status == "fail":
        acc = "N/A"
    elif a.value is None:
        acc = "—"
    elif a.delta is None:
        acc = f"首次（当前 {score(a.value)}）"
    else:
        acc = f"{acc_flag(a.delta)} {pts(a.delta)}（当前 {score(a.value)}）".strip()
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
        return "首次"
    return f"{perf_flag(delta)} {pct(delta)}".strip()


def _best_delta(c):
    """Against the best earlier image: flagged only below the slow-slide line."""
    if c.status == "fail":
        return "N/A"
    if c.value is None:
        return "—"
    if c.delta_best is None:
        return "首次"
    d = round(c.delta_best, 6)
    if d > 0:
        return f"{pct(d)}（新高）"
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
        return "首次"
    return f"{acc_flag(a.delta)} {pts(a.delta)}".strip()


def _sha(env, repo):
    return ((env.get(repo) or {}).get("commit") or "")[:9] or "—"


def _image_short(image):
    return (image or "—").split(":")[-1]


LEGEND = (
    "性能 = 并发 64 的每卡吞吐（tok/s/GPU），并发 4 和其他指标见详细报告。N/A = 跑挂，没有值；"
    "— = 没有结果；首次 = 第一次有数据。变化和精度变化都和该模型最近一次成功的镜像比，"
    "⚠️/🔴/🟢 的阈值见 455-CI-guide.md §5.1。"
)


def summary_md(date, env, results, pending, detail_name, mock=False):
    out = [
        f"# 455-CI 报告 · {date}" + (" · 示例 EXAMPLE" if mock else ""),
        "",
        *(EXAMPLE if mock else []),
        (
            f"镜像 `{_image_short(env.get('image'))}` · ATOM `main@{_sha(env, 'atom')}`"
            f" · aiter `main@{_sha(env, 'aiter')}` · base `{_image_short(env.get('base_image'))}`"
            f" · [详细报告]({detail_name})"
        ),
        "",
        "## Highlights",
        "",
        *[f"- {h}" for h in highlights(results, pending)],
        "",
        "## 总表",
        "",
        "| 模型 | 场景 | 上次性能 tok/s/GPU | 本次性能 tok/s/GPU | 变化 | 精度变化（GSM8K） |",
        "|---|---|---|---|---|---|",
    ]
    out += ["| " + " | ".join(summary_row(r)) + " |" for r in results]
    out += [f"| 待测（周中） | {label} | | | | |" for label in pending]
    out += ["", LEGEND, ""]
    return "\n".join(out)


def _env_rows(env):
    def commit(repo, url):
        c = env.get(repo) or {}
        sha = c.get("commit") or ""
        if not sha:
            return "—"
        tail = " ".join(x for x in (c.get("date"), c.get("title")) if x)
        return f"`main` @ [`{sha[:9]}`](https://github.com/{url}/commit/{sha})" + (
            f"，{tail}" if tail else ""
        )

    digest = f"（digest `{env['digest']}`）" if env.get("digest") else ""
    dates = "；".join(
        f"{LABEL[s]} {d}" for s, d in (env.get("test_dates") or {}).items()
    )
    return [
        "| 项目 | 内容 |",
        "|---|---|",
        f"| 测试 image | `{env.get('image') or '—'}`{digest} |",
        f"| base image | `{env.get('base_image') or '—'}` |",
        f"| ATOM | {commit('atom', 'ROCm/ATOM')} |",
        f"| aiter | {commit('aiter', 'ROCm/aiter')} |",
        f"| 依赖变化 | {'、'.join(env.get('dependency_changes') or []) or '无'} |",
        f"| 测试日期 | {dates or '—'}（北京时间） |",
    ]


SECOND_HEADER = {
    "endpoint": "E2E avg interactivity tok/s/user",
    "8k1k": "ITL P50 ms（越低越好）",
}


def detail_md(
    date, env, results, pending, summary_name, charts_dir, runs_dir, mock=False
):
    """`charts_dir` and `runs_dir` are relative to the report's directory."""
    out = [
        f"# 455-CI 详细报告 · {date}" + (" · 示例 EXAMPLE" if mock else ""),
        "",
        *(EXAMPLE if mock else []),
        f"[← 报告]({summary_name})",
        "",
        "## 测试环境",
        "",
        *_env_rows(env),
    ]
    for scenario in ("endpoint", "8k1k"):
        label = LABEL[scenario]
        out += ["", f"## {label}", ""]
        if label in pending:
            out.append("待测（周中）。")
            continue
        mine = [r for r in results if r.scenario == scenario]
        out += [
            "### 性能",
            "",
            f"| 模型 | GPU | 并发 | 每卡吞吐 tok/s/GPU | vs 上次 | vs 历史最佳 | {SECOND_HEADER[scenario]} | vs 上次 |",
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
                    else ("首次" if c.second is not None else "—")
                )
                out.append(
                    f"| {r.name} | {c.gpus or '—'} | {c.concurrency} | {_perf_value(c)} | "
                    f"{_perf_delta(c, c.delta)} | {_best_delta(c)} | {sv} | {sd} |"
                )
        out += [
            "",
            "### 精度（GSM8K，3-shot）",
            "",
            "| 模型 | flexible-extract | strict-match | 上次 | vs 上次 |",
            "|---|---|---|---|---|",
        ]
        for r in mine:
            a = r.acc
            strict = score(a.strict) if a.strict is not None else "—"
            prev = score(a.prev) if a.prev is not None else "—"
            out.append(
                f"| {r.name} | {_acc_value(a)} | {strict} | {prev} | {_acc_delta(a)} |"
            )
        out += [
            "",
            "### 趋势",
            "",
            f"![{label} 每卡吞吐趋势]({charts_dir}/perf-{scenario}.png)",
            "",
            f"![{label} GSM8K 趋势]({charts_dir}/accuracy-{scenario}.png)",
        ]
    out += ["", "## 失败", ""]
    broken = [r for r in results if _failed_parts(r)]
    if not broken:
        out.append("本次没有跑挂的 case。")
    for r in broken:
        errors = [c.error for c in r.perf if c.error] + (
            [r.acc.error] if r.acc.error else []
        )
        out += [
            f"### {r.name}（{LABEL[r.scenario]}）：{'、'.join(_failed_parts(r))}",
            "",
            f"- 报错：`{_short(errors[0], 120) if errors else '—'}`",
            f"- 完整摘录：[error.log]({runs_dir}/{r.slug}/error.log)",
            "",
        ]
        log = (r.error_log or "").rstrip("\n").splitlines()
        if log:
            shown = log[:LOG_LINES] + (
                [f"…（共 {len(log)} 行，其余见 error.log）"]
                if len(log) > LOG_LINES
                else []
            )
            out += ["核心 error log：", "", "```text", *shown, "```", ""]
        else:
            out += ["没有找到 error.log，按 guide §3.4 补上。", ""]
    return "\n".join(out).rstrip("\n") + "\n"

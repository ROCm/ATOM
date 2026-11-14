"""Pick and compare the 455-CI results in perf.csv and accuracy.csv.

A session is one day's run of one scenario: Saturday runs MLPerf Endpoint on
the image built that morning, Wednesday runs 8K/1K on the previous Saturday's
image. Sessions are keyed by their test date (ISO, so string order is date
order); the image_date column only records which image a session used.

Rows are dicts as csv.DictReader yields them, so every value is a string. A
rerun appends rows: for a session and concurrency (perf) or a session
(accuracy) the last row in file order wins.
"""

import csv
from dataclasses import dataclass

# Column order of the two history files (455-CI-guide.md §4.2 and §4.3).
PERF_COLUMNS = (
    "image_date",
    "test_date",
    "scenario",
    "model",
    "gpus",
    "parallel",
    "concurrency",
    "status",
    "tok_s_per_gpu",
    "interactivity",
    "itl_p50_ms",
    "tpot_mean_ms",
    "image",
    "base_image",
    "atom_branch",
    "atom_commit",
    "aiter_branch",
    "aiter_commit",
    "error",
    "note",
)
ACCURACY_COLUMNS = (
    "image_date",
    "test_date",
    "scenario",
    "model",
    "status",
    "gsm8k",
    "gsm8k_strict",
    "num_fewshot",
    "image",
    "base_image",
    "atom_branch",
    "atom_commit",
    "aiter_branch",
    "aiter_commit",
    "error",
    "note",
)
# Columns of followups.csv: the next step for each problem in the highlights
# (455-CI-guide.md §3.7).
FOLLOWUP_COLUMNS = ("test_date", "model", "action", "link", "note")
# Columns of pending_fixes.csv: unmerged PRs a model is known to need on main,
# all of `prs` together (455-CI-guide.md §3.7).
PENDING_COLUMNS = ("model", "prs", "note", "since")
CONCURRENCIES = (4, 64)
# The second perf metric of a scenario: endpoint E2E interactivity, 8K/1K ITL P50.
SECOND = {"endpoint": "interactivity", "8k1k": "itl_p50_ms"}


@dataclass
class PerfCell:
    """One concurrency of one model in the session, and its history."""

    concurrency: int
    status: str  # "ok", "fail", or "missing" (no row, or ok without a number)
    value: float | None  # tok/s per GPU
    second: float | None  # endpoint: E2E avg interactivity; 8k1k: ITL P50 (ms)
    prev: (
        float | None
    )  # tok/s per GPU in the newest earlier session where this point succeeded
    prev_second: float | None
    prev_date: str | None
    prev_failed: bool  # the point failed in the newest earlier session that ran it
    error: str
    gpus: int | None
    parallel: str = ""  # e.g. tp8dcp8, from the recipe's server command

    @property
    def delta(self):
        return _ratio(self.value, self.prev)

    @property
    def delta_second(self):
        return _ratio(self.second, self.prev_second)


@dataclass
class AccCell:
    """GSM8K of one model in the session, and the newest earlier success."""

    status: str  # "ok", "fail", or "missing"
    value: float | None  # flexible-extract
    strict: float | None  # strict-match
    prev: float | None
    prev_date: str | None
    prev_failed: bool
    error: str

    @property
    def delta(self):
        """Change in score (0.01 = one point), not a ratio."""
        if self.value is None or self.prev is None:
            return None
        return self.value - self.prev


@dataclass
class AccuracyPoint:
    date: str  # session (test) date
    flexible: float | None  # None when the run failed
    strict: float | None


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _ratio(now, then):
    if now is None or not then:
        return None
    return now / then - 1


def _rows_up_to(rows, scenario, model, date):
    return [
        r
        for r in rows
        if r["scenario"] == scenario and r["model"] == model and r["test_date"] <= date
    ]


def _last_rows(rows, key):
    last = {}
    for r in rows:
        last[key(r)] = r
    return last


def _usable(row, column="tok_s_per_gpu"):
    return row["status"] == "ok" and _number(row.get(column)) is not None


def perf_cells(rows, scenario, model, date, concurrencies=CONCURRENCIES):
    """A PerfCell per concurrency for the session on `date`."""
    last = _last_rows(
        _rows_up_to(rows, scenario, model, date),
        lambda r: (r["test_date"], int(r["concurrency"])),
    )
    second = SECOND[scenario]
    cells = []
    for conc in concurrencies:
        now = last.get((date, conc))
        earlier = sorted(d for d, c in last if c == conc and d < date)
        good = [last[(d, conc)] for d in earlier if _usable(last[(d, conc)])]
        prev = good[-1] if good else None
        if now is None or (now["status"] == "ok" and not _usable(now)):
            status = "missing"
        else:
            status = "ok" if now["status"] == "ok" else "fail"
        recorded = [
            last[(d, conc)].get("parallel")
            for d in earlier
            if last[(d, conc)].get("parallel")
        ]
        parallel = ((now or {}).get("parallel") or "") or (
            recorded[-1] if recorded else ""
        )
        cells.append(
            PerfCell(
                concurrency=conc,
                status=status,
                value=_number(now["tok_s_per_gpu"]) if status == "ok" else None,
                second=_number(now.get(second)) if status == "ok" else None,
                prev=_number(prev["tok_s_per_gpu"]) if prev else None,
                prev_second=_number(prev.get(second)) if prev else None,
                prev_date=prev["test_date"] if prev else None,
                prev_failed=bool(earlier)
                and last[(earlier[-1], conc)]["status"] != "ok",
                error=(now or {}).get("error", "") if status == "fail" else "",
                gpus=(
                    int(now["gpus"])
                    if now and now.get("gpus")
                    else (int(prev["gpus"]) if prev and prev.get("gpus") else None)
                ),
                parallel=parallel,
            )
        )
    return cells


def perf_trend(rows, scenario, model, conc, date, max_sessions=12):
    """(session date, tok/s per GPU or None) for the `max_sessions` newest
    sessions up to `date` in which the model ran at any concurrency."""
    mine = _rows_up_to(rows, scenario, model, date)
    last = _last_rows(mine, lambda r: (r["test_date"], int(r["concurrency"])))
    sessions = sorted({r["test_date"] for r in mine})[-max_sessions:]
    trend = []
    for d in sessions:
        r = last.get((d, conc))
        trend.append((d, _number(r["tok_s_per_gpu"]) if r and _usable(r) else None))
    return trend


def acc_cell(rows, scenario, model, date):
    last = _last_rows(
        _rows_up_to(rows, scenario, model, date), lambda r: r["test_date"]
    )
    now = last.get(date)
    earlier = sorted(d for d in last if d < date)
    good = [last[d] for d in earlier if _usable(last[d], "gsm8k")]
    prev = good[-1] if good else None
    if now is None or (now["status"] == "ok" and not _usable(now, "gsm8k")):
        status = "missing"
    else:
        status = "ok" if now["status"] == "ok" else "fail"
    return AccCell(
        status=status,
        value=_number(now["gsm8k"]) if status == "ok" else None,
        strict=_number(now.get("gsm8k_strict")) if status == "ok" else None,
        prev=_number(prev["gsm8k"]) if prev else None,
        prev_date=prev["test_date"] if prev else None,
        prev_failed=bool(earlier) and last[earlier[-1]]["status"] != "ok",
        error=(now or {}).get("error", "") if status == "fail" else "",
    )


def accuracy_history(rows, scenario, model, date, max_sessions=12):
    """GSM8K per session, for the `max_sessions` newest sessions up to
    `date`; a failed session stays in as a gap (scores None)."""
    last = _last_rows(
        _rows_up_to(rows, scenario, model, date), lambda r: r["test_date"]
    )
    hist = []
    for day in sorted(last)[-max_sessions:]:
        r = last[day]
        ok = r["status"] == "ok"
        hist.append(
            AccuracyPoint(
                day,
                _number(r["gsm8k"]) if ok else None,
                _number(r.get("gsm8k_strict")) if ok else None,
            )
        )
    return hist


# The test matrix of 455-CI-guide.md §1: (slug, display name) per scenario.
MODELS = {
    "endpoint": [
        ("deepseek-r1", "DeepSeek R1"),
        ("deepseek-v4.1-flash", "DeepSeek-V4.1-Flash"),
        ("kimi-k3", "Kimi-K3 (2.8T)"),
        ("qwen-35b-a3b", "Qwen-35B-A3B"),
    ],
    "8k1k": [
        ("deepseek-v4-pro", "DeepSeek-V4-Pro"),
        ("glm-5.2", "GLM-5.2"),
        ("minimax-m3", "MiniMax-M3 (428B)"),
        ("qwen-3.8-flash", "Qwen-3.8 Flash"),
    ],
}
LABEL = {"endpoint": "MLPerf Endpoint", "8k1k": "8K/1K"}

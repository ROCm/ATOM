# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Judge a finished run from its log."""

from __future__ import annotations

import dataclasses
import math
import re

from .manifest import Checks, MetricCheck

DONE_MARK = "=== LAUNCHER_EXIT="
_DONE_RE = re.compile(r"=== LAUNCHER_EXIT=(\d+) ===")
_STEP_RE = re.compile(r"(?:^|\s)step=(\d+)\s")
_BUCKET_RE = re.compile(r"bucket done - updated=\d+, skipped=(\d+)")
# The session summary; -q drops the "=====" around it.
_PYTEST_RE = re.compile(
    r"^(?:=+ )?(.*\b(?:passed|failed|error|errors|no tests ran)\b.*) in [\d.]+s"
)


@dataclasses.dataclass
class MetricResult:
    name: str
    value: str | None
    status: str  # PASS | FAIL | INFO | MISSING
    detail: str = ""
    ref: float | None = None


@dataclasses.dataclass
class Verdict:
    status: str  # PASS | FAIL
    reasons: list[str]
    metrics: list[MetricResult]
    counts: dict[str, int]
    last_step: int | None
    steps: dict[int, dict[str, str]]

    def to_json(self) -> dict:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "metrics": [dataclasses.asdict(m) for m in self.metrics],
            "counts": self.counts,
            "last_step": self.last_step,
        }


def exit_code(text: str) -> int | None:
    found = _DONE_RE.findall(text)
    return int(found[-1]) if found else None


def metric_lines(text: str, marker: str) -> dict[int, dict[str, str]]:
    """Step -> ``key=value`` pairs of the first metrics line logged for it."""
    steps: dict[int, dict[str, str]] = {}
    for line in text.splitlines():
        if marker not in line:
            continue
        m = _STEP_RE.search(line)
        if not m or int(m.group(1)) in steps:
            continue
        pairs = {}
        for tok in line[m.start() :].split():
            if "=" in tok:
                key, _, value = tok.partition("=")
                pairs[key] = value
        steps[int(m.group(1))] = pairs
    return steps


def judge_metric(check: MetricCheck, raw: str | None) -> MetricResult:
    if raw is None:
        if check.info:
            return MetricResult(check.name, None, "INFO", "not in log")
        return MetricResult(check.name, None, "MISSING", "not in log", check.ref)
    try:
        value = float(raw)
    except ValueError:
        return MetricResult(check.name, raw, "FAIL", "not a number", check.ref)
    if check.info:
        return MetricResult(check.name, raw, "INFO", "recorded")
    if not math.isfinite(value):
        return MetricResult(check.name, raw, "FAIL", "not finite", check.ref)
    if check.rtol is not None:
        dev = (value - check.ref) / abs(check.ref) if check.ref else 0.0
        ok = abs(dev) <= check.rtol
        detail = f"{dev * 100:+.1f}% vs {check.ref:g} (tol {check.rtol * 100:.0f}%)"
    elif check.band is not None:
        lo, hi = abs(check.ref) / check.band, abs(check.ref) * check.band
        ok = lo <= abs(value) <= hi
        detail = f"|x| in [{lo:.3g}, {hi:.3g}]"
    else:
        lo = -math.inf if check.min is None else check.min
        hi = math.inf if check.max is None else check.max
        ok = lo <= value <= hi
        detail = f"in [{lo:g}, {hi:g}]"
    return MetricResult(check.name, raw, "PASS" if ok else "FAIL", detail, check.ref)


def _pytest_summary(text: str) -> tuple[str | None, dict[str, int]]:
    line = None
    for ln in text.splitlines():
        m = _PYTEST_RE.match(ln.strip())
        if m:
            line = m.group(1)
    counts: dict[str, int] = {}
    if line:
        for num, word in re.findall(r"(\d+) (\w+)", line):
            counts[word] = int(num)
    return line, counts


def judge(text: str, checks: Checks, steps: int | None) -> Verdict:
    reasons: list[str] = []
    counts: dict[str, int] = {}
    code = exit_code(text)
    if code is None:
        reasons.append("the run did not finish (no exit marker in the log)")
    elif code != 0:
        reasons.append(f"exit code {code}")

    for pat in checks.forbid:
        counts[pat] = sum(1 for ln in text.splitlines() if pat in ln)
        if counts[pat]:
            reasons.append(f"{counts[pat]} line(s) with {pat!r}")

    if checks.weight_sync_buckets:
        bad = sum(1 for n in _BUCKET_RE.findall(text) if int(n) != 0)
        counts["skipped_buckets"] = bad
        if bad:
            reasons.append(f"{bad} weight-sync bucket(s) skipped tensors")

    if checks.pytest:
        line, pcounts = _pytest_summary(text)
        counts.update({f"pytest_{k}": v for k, v in pcounts.items()})
        if line is None:
            reasons.append("no pytest summary line")
        elif pcounts.get("failed") or pcounts.get("error") or pcounts.get("errors"):
            reasons.append(f"pytest: {line}")

    per_step = metric_lines(text, checks.marker) if not checks.pytest else {}
    last = max(per_step) if per_step else None
    results: list[MetricResult] = []
    if checks.metrics:
        line = per_step.get(checks.step)
        if line is None:
            reasons.append(
                f"no step-{checks.step} metrics line (marker {checks.marker!r})"
            )
        else:
            for check in checks.metrics:
                res = judge_metric(check, line.get(check.name))
                results.append(res)
                if res.status in ("FAIL", "MISSING"):
                    reasons.append(f"{res.name}={res.value} {res.detail}")
    if (
        checks.last_step
        and steps
        and not checks.pytest
        and (last is None or last < steps)
    ):
        reasons.append(f"final step {steps} not reached (last metrics step: {last})")

    return Verdict(
        status="FAIL" if reasons else "PASS",
        reasons=reasons,
        metrics=results,
        counts=counts,
        last_step=last,
        steps=per_step,
    )

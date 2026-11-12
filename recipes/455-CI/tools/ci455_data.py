"""Pick the rows the 455-CI charts draw from perf.csv and accuracy.csv.

Rows are dicts as csv.DictReader yields them, so every value is a string. A
rerun appends rows: for an image and concurrency (perf) or an image (accuracy)
the last row in file order wins. Images are keyed by their ISO date, so
string order is date order.
"""

import csv
from dataclasses import dataclass

# MI455 curve colours from LLM-Booster#246, oldest to newest: the newest
# curve always takes the deepest red.
RAMP = ("#F5998A", "#E8452C", "#C42B4B", "#9B1C1C")

# Column order of the two history files (455-CI-guide.md §4.2 and §4.3).
PERF_COLUMNS = (
    "image_date",
    "test_date",
    "scenario",
    "model",
    "gpus",
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


@dataclass
class Point:
    c: int  # concurrency
    x: float  # endpoint: E2E avg interactivity; 8k1k: 1000 / mean TPOT (ms)
    y: float  # tok/s per GPU


@dataclass
class Curve:
    image_date: str
    gpus: int
    points: list  # Point, sorted by x


@dataclass
class AccuracyPoint:
    image_date: str
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


def _rows_up_to(rows, scenario, model, report_date):
    return [
        r
        for r in rows
        if r["scenario"] == scenario
        and r["model"] == model
        and r["image_date"] <= report_date
    ]


def _last_rows(rows, key):
    last = {}
    for r in rows:
        last[key(r)] = r
    return last


def perf_curves(rows, scenario, model, report_date, max_images=4):
    """One curve per image, for the `max_images` newest images up to
    `report_date` that have at least one successful point."""
    last = _last_rows(
        _rows_up_to(rows, scenario, model, report_date),
        lambda r: (r["image_date"], int(r["concurrency"])),
    )
    by_image = {}
    for (image, conc), r in sorted(last.items()):
        if r["status"] != "ok":
            continue
        y = _number(r["tok_s_per_gpu"])
        if scenario == "8k1k":
            tpot = _number(r["tpot_mean_ms"])
            x = 1000 / tpot if tpot else None
        else:
            x = _number(r["interactivity"])
        if x is None or y is None:
            continue
        curve = by_image.setdefault(image, Curve(image, int(r["gpus"]), []))
        curve.points.append(Point(conc, x, y))
    curves = [by_image[d] for d in sorted(by_image)][-max_images:]
    for curve in curves:
        curve.points.sort(key=lambda p: p.x)
    return curves


def failed_concurrencies(rows, scenario, model, report_date):
    """Concurrencies whose last row on the report image is not ok."""
    last = _last_rows(
        [
            r
            for r in _rows_up_to(rows, scenario, model, report_date)
            if r["image_date"] == report_date
        ],
        lambda r: int(r["concurrency"]),
    )
    return sorted(c for c, r in last.items() if r["status"] != "ok")


def accuracy_history(rows, scenario, model, report_date, max_images=12):
    """GSM8K per image, for the `max_images` newest images up to
    `report_date`; a failed image stays in as a gap (scores None)."""
    last = _last_rows(
        _rows_up_to(rows, scenario, model, report_date), lambda r: r["image_date"]
    )
    hist = []
    for image in sorted(last)[-max_images:]:
        r = last[image]
        ok = r["status"] == "ok"
        hist.append(
            AccuracyPoint(
                image,
                _number(r["gsm8k"]) if ok else None,
                _number(r.get("gsm8k_strict")) if ok else None,
            )
        )
    return hist


def curve_styles(n):
    """(colour, line) for `n` curves, oldest first: the deepest `n` shades of
    RAMP, and only the newest curve solid."""
    if n > len(RAMP):
        raise ValueError(f"at most {len(RAMP)} curves, got {n}")
    colours = RAMP[len(RAMP) - n :]
    return [(c, "solid" if i == n - 1 else "dashed") for i, c in enumerate(colours)]

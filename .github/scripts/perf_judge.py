#! /usr/bin/env python3
"""Judge a paired base/head benchmark comparison for a pull request.

Consumes the two result directories produced by the paired A/B step of the PR
perf check (same job, same machine, same container image, same aiter wheel) and
decides whether the head commit regressed.

Usage:
    python perf_judge.py --base-dir <dir> --head-dir <dir> \\
        [--history data.js] [--output-json verdict.json] \\
        [--comment-file comment.md] > "$GITHUB_STEP_SUMMARY"

Design notes
------------
The primary criterion is *family linkage*, not a per-configuration threshold.
A "family" here is one model entry (backend + display model + isl/osl); its
members are the concurrency levels. Independent concurrency levels drifting in
the same direction at the same time is far less likely than any single level
moving, so linkage is the most noise-resistant signal available -- and, unlike a
sigma gate, it needs no estimate of the noise band.

The sigma gate is auxiliary only. Estimating sigma requires several historical
points on the *same* container image, and those barely exist: across 52
deduplicated perf runs on the public dashboard, 30 of 39 images were benchmarked
exactly once and only 3 reached 3 runs. What history can offer is a cross-image
estimate, which is inflated by image-to-image level shifts and therefore
conservative (a loose gate that under-reports rather than over-reports).

Incomplete data never produces a pass. A family that lost too many concurrency
levels is reported as ``insufficient``; if no family survives, the whole run is
``inconclusive``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize import (  # the sys.path shim above must run first
    _backend_name,
    _config_key,
    _display_model,
    _pct_change,
    load_results,
)

# --- Family linkage (primary criterion; independent of sigma) ---------------
JUDGE_MIN_CONC = 64  # below this, a level is reference-only and never judged
FAMILY_MIN_CONFIGS = 3  # judging levels needed before a family is judged
FAMILY_MIN_DOWN = 3  # of those, how many must be down
FAMILY_MEDIAN_PCT = -3.0  # family median throughput delta that trips the gate
DOWN_EPS_PCT = -2.0  # a level counts as "down" past this
TPOT_MIRROR_RATIO = 0.6  # |median TPOT delta / median tput delta| for mirroring
NONMONOTONIC_MARGIN_PCT = 3.0  # interior level this far outside its neighbours
MEASURED_RESIDUAL_BIAS_PCT = 1.35  # A/A residual after the warmup; see below

# --- Baseline sanity (crimson only) ----------------------------------------
BASELINE_SANITY_PCT = -25.0  # base this far under main's recent median
BASELINE_SANITY_SIGMA = 2.5  # ... and clear of that configuration's own noise
BASELINE_SANITY_MIN_LEVELS = 2  # ... on at least this many judged levels
BASELINE_SANITY_MIN_POINTS = 6  # ... judged against at least this much history

# A paired comparison answers "did this change make it worse" and is blind to
# "main was already broken": if base and head are both 20% down, the delta is
# zero and the run reports clean. This check reads the base measurement against
# main's recent history to catch that -- but only at crimson magnitude. Across
# 372 configurations the 8-run spread on the public dashboard is 7.7% at the
# median, 13.7% at P75 and 25.1% at P90, all of it ordinary cross-image
# variation. Anything finer than that is indistinguishable from noise here.
#
# Slow drift on main -- the 5% kind that accumulates over weeks -- needs a time
# series and a trend criterion, which is a nightly-side job, not a PR one.
DRIFT_TRUST_PCT = 3.0  # base repeated this far apart -> the pairing cannot resolve

# The base commit is measured twice, before and after head. The distance
# between those two readings is drift the pairing accumulated -- caches warming,
# clocks settling, a neighbour arriving -- and it bounds what the comparison can
# resolve. A head-vs-base delta smaller than the drift is not a measurement of
# the change; it is a measurement of time passing. Where both readings exist the
# baseline is their mean, which cancels drift to first order, and the residual
# drift is reported so the reader can see how much was cancelled.

# Low concurrency levels are excluded from the verdict but still measured and
# still shown. They pollute a verdict in both directions: they manufacture false
# positives (an entry whose large-concurrency levels sat at their historical
# peak was flagged at -4.2% purely on c4..c32) and they dilute real ones (an
# all-levels median of -7.1% was -12.0% when restricted to large concurrency).
# Dropping them from the *display* as well, however, removes the very evidence a
# reader needs to tell "small-batch path problem" from "small-c noise" -- so
# they are reported alongside, labelled, and excluded from every computation.

# What FAMILY_MEDIAN_PCT can actually resolve, measured rather than assumed.
#
# An A/A run -- every phase on one commit, so every delta is noise -- gave
# +1.35% on MI308 with DeepSeek-V4-Flash at tp=8, after the warmup. A third
# measurement of the same base commit landed at +1.26%, so the residual is
# drift over the run, not anything specific to the second half: later readings
# are faster than earlier ones by about that much, whichever commit they carry.
#
# The bias is systematic and favours head, so it subtracts from any regression
# rather than inventing one. A true -3% reads as roughly -1.65% and does not
# trip; tripping needs about -4.35%. That is the honest sensitivity of this
# check, and it is fine for the regressions worth catching -- 5% and up -- but
# a reader should not take -3.0 as the resolution.
#
# One run on one shared machine, and drift varied visibly within it (the
# benchmark slowed from 1.07 to 1.65 s/it partway through and recovered), so
# treat 1.35% as an order of magnitude rather than a constant. Averaging a
# repeated base reading halves it, at the cost of a third measurement; see the
# workflow for why that trade was declined.

# --- Single-configuration escalation (auxiliary; uses history sigma) --------
SINGLE_DROP_PCT = -8.0  # a lone level this far down is worth surfacing
SIGMA_MULT = 2.5  # ... if it also clears this many historical sigmas
SINGLE_STANDS_ALONE_PCT = -10.0  # a drop this large needs no corroboration
DRAMATIC_PCT = -25.0  # too large to be ordinary variance at any concurrency

# Metric keys in the benchmark result JSON.
TPUT_KEY = "total_token_throughput"
TPOT_KEY = "mean_tpot_ms"
TTFT_KEY = "mean_ttft_ms"

DASHBOARD_DATA_JS = "https://rocm.github.io/ATOM/benchmark-dashboard/data.js"


# --------------------------------------------------------------------------
# Historical noise band
# --------------------------------------------------------------------------
def _parse_data_js(text):
    """Extract the BENCHMARK_DATA object from a data.js payload."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.loads(text[start:].rstrip().rstrip(";"))
    except json.JSONDecodeError:
        return None


def load_history_cv(source=None):
    """Return {(model, isl_osl, conc): {"cv", "median", "n"}} from history.

    ``source`` may be a local path or None (fetch the public dashboard). Any
    failure yields an empty mapping -- the sigma gate is optional by design and
    must never block a verdict.

    Points are deduplicated by Actions run id first: the dashboard republishes a
    single run against several commits (9% of perf points in the sampled
    window), and duplicates have zero spread, so leaving them in biases the
    estimate toward "suspiciously stable".
    """
    try:
        if source:
            text = Path(source).read_text(encoding="utf-8")
        else:
            req = urllib.request.Request(
                DASHBOARD_DATA_JS, headers={"User-Agent": "atom-perf-judge"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                text = resp.read().decode("utf-8", errors="replace")
    except (OSError, urllib.error.URLError):
        return {}

    data = _parse_data_js(text)
    if not data:
        return {}

    runs = []
    for entry in data.get("entries", {}).values():
        runs.extend(entry)

    seen_run_ids = set()
    series = defaultdict(list)
    name_re = re.compile(r"^(ATOM[^:]*)::(.+?)\s+(\d+/\d+)\s+c=(\d+)\s+Total Tput")

    for run in sorted(runs, key=lambda r: r.get("date", 0)):
        benches = run.get("benches") or []
        if not benches:
            continue
        extra = benches[0].get("extra") or ""
        run_id_match = re.search(r"actions/runs/(\d+)", extra)
        run_id = run_id_match.group(1) if run_id_match else None
        if run_id and run_id in seen_run_ids:
            continue
        if run_id:
            seen_run_ids.add(run_id)
        for bench in benches:
            if bench.get("unit") != "tok/s":
                continue
            match = name_re.match(bench.get("name", ""))
            if not match:
                continue
            key = (match.group(2), match.group(3), int(match.group(4)))
            series[key].append(bench.get("value"))

    stats = {}
    for key, values in series.items():
        values = [v for v in values if isinstance(v, (int, float))]
        if len(values) < BASELINE_SANITY_MIN_POINTS:
            continue
        # The last few runs, not the whole window: a level that shifted weeks
        # ago is the current normal, and holding a run against a long-gone one
        # is the "compared against a peak" mistake in slow motion.
        recent = values[-8:]
        median = statistics.median(recent)
        if median:
            stats[key] = {
                "cv": statistics.pstdev(recent) / median,
                "median": median,
                "n": len(recent),
            }
    return stats


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------
def pair_results(base_results, head_results, base2_results=None):
    """Match head measurements to base measurements by configuration key.

    ``base2_results`` is an optional second reading of the base commit taken
    after head. When present the baseline is the mean of the two readings and
    the drift between them is carried through for reporting.
    """
    base_map = {_config_key(d): d for d in base_results}
    base2_map = {_config_key(d): d for d in (base2_results or [])}
    pairs = []
    for head in head_results:
        key = _config_key(head)
        base = base_map.get(key)
        if base is None:
            continue
        base_tput = base.get(TPUT_KEY)
        head_tput = head.get(TPUT_KEY)
        if not base_tput or not head_tput:
            continue
        base2 = base2_map.get(key)
        base2_tput = base2.get(TPUT_KEY) if base2 else None
        if base2_tput:
            baseline = (base_tput + base2_tput) / 2
            drift_pct = _pct_change(base2_tput, base_tput)
        else:
            baseline = base_tput
            drift_pct = None

        pairs.append(
            {
                "backend": _backend_name(head),
                "model": _display_model(head),
                "isl_osl": f"{head.get('random_input_len', 0)}/"
                f"{head.get('random_output_len', 0)}",
                "conc": int(head.get("max_concurrency", 0)),
                "base_tput": base_tput,
                "base2_tput": base2_tput,
                "head_tput": head_tput,
                "baseline_tput": baseline,
                "drift_pct": drift_pct,
                "tput_pct": _pct_change(head_tput, baseline),
                "tpot_pct": _delta_or_none(base, head, TPOT_KEY),
                "ttft_pct": _delta_or_none(base, head, TTFT_KEY),
            }
        )
    return pairs


def _delta_or_none(base, head, key):
    base_value, head_value = base.get(key), head.get(key)
    if not base_value or not head_value:
        return None
    return _pct_change(head_value, base_value)


# --------------------------------------------------------------------------
# Judgment
# --------------------------------------------------------------------------
def judge_family(members, history_cv):
    """Judge one model entry from its concurrency levels.

    Returns a dict with ``status`` in {triggered, clean, insufficient} plus the
    evidence behind that call.
    """
    members = sorted(members, key=lambda m: m["conc"])
    judging = [m for m in members if m["conc"] >= JUDGE_MIN_CONC]
    reference = [m for m in members if m["conc"] < JUDGE_MIN_CONC]

    tputs = [m["tput_pct"] for m in judging]
    tpots = [m["tpot_pct"] for m in judging if m["tpot_pct"] is not None]

    result = {
        "backend": members[0]["backend"],
        "model": members[0]["model"],
        "isl_osl": members[0]["isl_osl"],
        "members": members,
        "judging": judging,
        "reference": reference,
        "median_tput_pct": statistics.median(tputs) if tputs else None,
        "median_tpot_pct": statistics.median(tpots) if tpots else None,
        "n_down": sum(1 for t in tputs if t <= DOWN_EPS_PCT),
        "n_total": len(tputs),
        "median_drift_pct": _median_or_none(
            [m["drift_pct"] for m in judging if m["drift_pct"] is not None]
        ),
        "nonmonotonic": _nonmonotonic(judging),
        "baseline_sanity": _baseline_sanity(judging, history_cv),
        "escalations": _single_escalations(judging, history_cv),
    }

    if len(tputs) < FAMILY_MIN_CONFIGS:
        result["status"] = "insufficient"
        result["reason"] = (
            f"only {len(tputs)} of {FAMILY_MIN_CONFIGS} required judging levels "
            f"(c >= {JUDGE_MIN_CONC}) reported"
        )
        return result

    median_tput = result["median_tput_pct"]
    median_tpot = result["median_tpot_pct"]

    # Mirroring: throughput down while TPOT goes up, by a comparable magnitude.
    # It separates a real slowdown from measurement wobble, which does not make
    # the two metrics move in lockstep.
    mirror = (
        median_tpot is not None
        and median_tput < 0
        and median_tpot > 0
        and abs(median_tpot / median_tput) >= TPOT_MIRROR_RATIO
    )
    result["mirror"] = mirror
    result["mirror_ratio"] = (
        abs(median_tpot / median_tput)
        if (median_tpot is not None and median_tput)
        else None
    )

    triggered = (
        median_tput <= FAMILY_MEDIAN_PCT
        and result["n_down"] >= FAMILY_MIN_DOWN
        and mirror
    )
    result["status"] = "triggered" if triggered else "clean"
    return result


def _median_or_none(values):
    """Median, or None for an empty sample.

    Spelled out rather than inlined with a walrus so the module keeps parsing
    on older interpreters: CI runs 3.12, but this also has to run wherever
    someone points it at a pair of result directories.
    """
    return statistics.median(values) if values else None


def _baseline_sanity(judging, history):
    """Flag a base measurement that sits far under main's recent level.

    Three gates, all required, following the same shape the nightly monitor
    uses for its highest-confidence class:

    - magnitude: at least BASELINE_SANITY_PCT under the recent median
    - variance: and clear of that configuration's own noise band, because a
      fixed percentage condemns a naturally jumpy configuration for behaving
      normally -- the 8-run spread reaches 25% at P90 across the dashboard
    - corroboration: on at least BASELINE_SANITY_MIN_LEVELS judged levels,
      because one level alone is the shape a scheduling blip takes

    Returns the offending levels, or None.
    """
    flagged = []
    for member in judging:
        hist = history.get((member["model"], member["isl_osl"], member["conc"]))
        if not hist or not hist["median"]:
            continue
        median = hist["median"]
        delta = (member["base_tput"] - median) / median * 100
        if delta > BASELINE_SANITY_PCT:
            continue
        # Variance gate. A configuration whose own history swings this much is
        # not saying anything by swinging again.
        sigma = hist["cv"] * median
        if sigma > 0 and (median - member["base_tput"]) < BASELINE_SANITY_SIGMA * sigma:
            continue
        flagged.append(
            {
                "conc": member["conc"],
                "base_tput": member["base_tput"],
                "history_median": median,
                "pct": delta,
                "sigma_pct": hist["cv"] * 100,
                "n": hist["n"],
            }
        )
    # Corroboration gate: one level is a blip, several at once is the machine
    # or the commit.
    if len(flagged) < BASELINE_SANITY_MIN_LEVELS:
        return None
    return flagged


def _nonmonotonic(judging):
    """Flag an interior level that sits well outside its immediate neighbours.

    A median cannot express this shape at all: c=64 -8.6%, c=128 -0.3%,
    c=256 -7.8% medians to -7.8% and reads as a clean two-sided drop, while the
    actual signal is that one level behaves nothing like the ones around it --
    often a kernel-selection fork. Worth surfacing verbatim rather than
    summarising away.
    """
    if len(judging) < 3:
        return None
    outliers = []
    for i in range(1, len(judging) - 1):
        value = judging[i]["tput_pct"]
        lo = min(judging[i - 1]["tput_pct"], judging[i + 1]["tput_pct"])
        hi = max(judging[i - 1]["tput_pct"], judging[i + 1]["tput_pct"])
        if value > hi + NONMONOTONIC_MARGIN_PCT or value < lo - NONMONOTONIC_MARGIN_PCT:
            outliers.append(
                {
                    "conc": judging[i]["conc"],
                    "pct": value,
                    "neighbours": [judging[i - 1]["conc"], judging[i + 1]["conc"]],
                    "neighbour_pcts": [
                        judging[i - 1]["tput_pct"],
                        judging[i + 1]["tput_pct"],
                    ],
                }
            )
    return outliers or None


def _single_escalations(members, history_cv):
    """Auxiliary per-level flags. Never on their own a family verdict."""
    flags = []
    for member in members:
        delta = member["tput_pct"]
        if delta > SINGLE_DROP_PCT:
            continue

        reasons = []
        if delta <= DRAMATIC_PCT:
            reasons.append(f"drop {delta:.1f}% exceeds {DRAMATIC_PCT:.0f}%")
        # Only judging levels reach this function, so there is no low-c caveat
        # left to apply -- every level here is one the verdict already trusts.
        if delta <= SINGLE_STANDS_ALONE_PCT:
            reasons.append(f"c={member['conc']} dropped {delta:.1f}% on its own")

        hist = history_cv.get((member["model"], member["isl_osl"], member["conc"]))
        if hist and hist["cv"]:
            sigma_pct = hist["cv"] * 100.0
            if abs(delta) > SIGMA_MULT * sigma_pct:
                reasons.append(
                    f"{abs(delta) / sigma_pct:.1f}x the historical sigma "
                    f"({sigma_pct:.2f}%)"
                )
            else:
                # Inside the (conservative, cross-image) noise band: record the
                # miss so the summary can say why nothing was escalated.
                reasons.append(
                    f"within {SIGMA_MULT}x historical sigma "
                    f"({sigma_pct:.2f}%) -- not escalated"
                )

        if reasons and not any("not escalated" in r for r in reasons):
            flags.append({"conc": member["conc"], "pct": delta, "why": reasons})
    return flags


def judge(pairs, history_cv, expected_entries=None):
    """Two-layer verdict: per model entry, then across entries."""
    families = defaultdict(list)
    for pair in pairs:
        families[(pair["backend"], pair["model"], pair["isl_osl"])].append(pair)

    results = [judge_family(members, history_cv) for members in families.values()]
    results.sort(
        key=lambda r: (
            {"triggered": 0, "insufficient": 1, "clean": 2}[r["status"]],
            r["median_tput_pct"] if r["median_tput_pct"] is not None else 0,
        )
    )

    triggered = [r for r in results if r["status"] == "triggered"]
    judged = [r for r in results if r["status"] in ("triggered", "clean")]

    # Entries that were expected but produced no usable pair at all (the whole
    # job died) never reach ``results``, so they have to be counted separately.
    n_missing = max(0, (expected_entries or 0) - len(results))
    incomplete = (len(results) - len(judged)) + n_missing

    # A positive finding stands on its own -- missing coverage elsewhere does
    # not weaken it. But the absence of a finding is only as good as the
    # coverage behind it, so anything short of full coverage downgrades to
    # ``partial``: no regression *in what could be measured*, which is a
    # different claim from "no regression".
    # A non-monotonic family is one the criterion cannot describe: the median
    # and the down-count both assume the levels behave alike, and here they do
    # not. Reporting "clean" would be a claim the method does not support, so
    # such a run is handed to the reader instead of summarised away.
    unclear = [f for f in results if f.get("nonmonotonic")]
    bad_baseline = [f for f in results if f.get("baseline_sanity")]

    # Drift larger than the threshold the criterion is asked to resolve means
    # the comparison cannot answer the question, whatever the deltas look like.
    drifted = [
        f
        for f in results
        if f.get("median_drift_pct") is not None
        and abs(f["median_drift_pct"]) > DRIFT_TRUST_PCT
    ]

    # A broken baseline outranks every other reading, including a trip: a
    # regression measured against a bad number is not a regression, and a clean
    # result against one says nothing at all.
    if bad_baseline:
        verdict = "bad_baseline"
    elif triggered and not drifted:
        verdict = "regression"
    elif not judged:
        verdict = "inconclusive"
    elif drifted:
        verdict = "untrustworthy"
    elif unclear:
        verdict = "unclear"
    elif incomplete:
        verdict = "partial"
    else:
        verdict = "clean"

    return {
        "verdict": verdict,
        "n_triggered": len(triggered),
        "n_judged": len(judged),
        "n_insufficient": len(results) - len(judged),
        "n_missing": n_missing,
        "n_unclear": len(unclear),
        "n_drifted": len(drifted),
        "n_bad_baseline": len(bad_baseline),
        "median_drift_pct": _median_or_none(
            [
                f["median_drift_pct"]
                for f in results
                if f.get("median_drift_pct") is not None
            ]
        ),
        "expected_entries": expected_entries,
        "scope": _scope(triggered, judged),
        "families": results,
        "thresholds": {
            "family_median_pct": FAMILY_MEDIAN_PCT,
            "family_min_down": FAMILY_MIN_DOWN,
            "family_min_configs": FAMILY_MIN_CONFIGS,
            "tpot_mirror_ratio": TPOT_MIRROR_RATIO,
            # Measured residual bias after the warmup, from an A/A run. It is
            # systematic and favours head, so the effective trip point is this
            # much further down than family_median_pct.
            "measured_residual_bias_pct": MEASURED_RESIDUAL_BIAS_PCT,
        },
    }


def _scope(triggered, judged):
    """Second layer: how wide is the blast radius."""
    if not triggered:
        return None
    if len(triggered) == len(judged) and len(judged) > 1:
        return "all_entries"
    if len(triggered) == 1:
        return "single_entry"
    return "several_entries"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_HEADLINE = {
    "clean": "### No significant performance change",
    "partial": "### No regression found, but coverage was incomplete",
    "unclear": "### Needs a look -- levels moved, but not in a shape the criterion can judge",
    "untrustworthy": "### Cannot be trusted -- the base commit did not measure the same twice",
    "bad_baseline": "### Do not act on this -- the baseline itself is far below normal",
    "regression": "### Performance regression detected",
    "inconclusive": "### Inconclusive -- not enough data",
}

_SCOPE_NOTE = {
    "all_entries": (
        "Every judged entry moved, which points at a shared path "
        "(kernel / attention / scheduler) rather than one model."
    ),
    "single_entry": "Only one entry moved, which points at a model-specific path.",
    "several_entries": "Several but not all entries moved.",
}


def _fmt(value, suffix="%"):
    return "n/a" if value is None else f"{value:+.1f}{suffix}"


def _pct(value, bold=False):
    if value is None:
        return "-"
    text = f"{value:+.1f}%"
    return f"**{text}**" if bold else text


def _median_cell(family):
    if family["status"] == "insufficient":
        return "insufficient"
    text = _fmt(family["median_tput_pct"])
    return f"**{text}**" if family["status"] == "triggered" else text


def _entry_rows(family):
    """One row per concurrency level, then the family's median.

    Throughput, TTFT and TPOT are the three independent measurements here --
    output throughput is total throughput divided by a constant (the prompt
    length ratio, fixed by --ignore-eos), and ITL and E2EL are recoverable from
    TTFT and TPOT. Listing those too would spend width without adding a fact.

    Judging and reference levels share the table and are labelled, so the shape
    stays visible: judging levels moving with the reference ones is a broad
    change, reference levels moving alone is a small-batch path or noise.
    """
    tripped = family["status"] == "triggered"
    rows = []
    first = True
    for member in family["judging"] + family["reference"]:
        judged = member["conc"] >= JUDGE_MIN_CONC
        # The marker rides with the number rather than occupying its own
        # column, and says what it does rather than naming a category: a reader
        # should not have to reach the legend to learn that a row does not count.
        label = f"{member['conc']}" if judged else f"{member['conc']} (not judged)"
        rows.append(
            "| {entry} | {c} | {tput} | {ttft} | {tpot} | {drift} |".format(
                entry=family["model"] if first else "",
                c=label,
                tput=_pct(member["tput_pct"], bold=tripped and judged),
                ttft=_pct(member["ttft_pct"]),
                tpot=_pct(member["tpot_pct"]),
                drift=_pct(member.get("drift_pct")),
            )
        )
        first = False

    if family["status"] == "insufficient":
        rows.append("| | **median of judged** | insufficient | | | |")
    else:
        rows.append(
            "| | **median of judged** | {tput} | {ttft} | {tpot} | {drift} |".format(
                tput=_pct(family["median_tput_pct"], bold=tripped),
                ttft="-",
                tpot=_pct(family["median_tpot_pct"], bold=tripped),
                drift=_pct(family.get("median_drift_pct")),
            )
        )
    return rows


def render(report, context):
    """Render the PR comment body."""
    lines = [_HEADLINE[report["verdict"]], ""]
    if context:
        lines += [context, ""]

    lines += [
        "| Model | Concurrency | Total Tput | TTFT | TPOT | Drift |",
        "|---|---|---|---|---|---|",
    ]
    for family in report["families"]:
        lines += _entry_rows(family)

    lines += [
        "",
        (
            f"**Concurrency** is how many requests are in flight at once. Only "
            f"levels at or above {JUDGE_MIN_CONC} decide the verdict. The rows "
            f"marked *(not judged)* were measured and are shown, but excluded "
            f"from every calculation: at low concurrency the numbers swing "
            f"enough to both invent regressions and hide real ones. They are "
            f"here because dropping them entirely would leave no way to tell a "
            f"small-batch problem from ordinary low-concurrency noise."
        ),
        "",
    ]

    # --- baseline sanity: loudest thing on the page when it fires ----------
    for family in report["families"]:
        flags = family.get("baseline_sanity") or []
        if not flags:
            continue
        worst = min(flags, key=lambda f: f["pct"])
        levels = ", ".join(
            f"c={f['conc']} {f['base_tput']:.0f} vs {f['history_median']:.0f}"
            for f in flags
        )
        lines += [
            (
                f"**{family['model']}: the base commit measured "
                f"{abs(worst['pct']):.0f}% under main's recent level on "
                f"{len(flags)} of {len(family['judging'])} judged levels** "
                f"({levels} tok/s, median of the last {worst['n']} nightly "
                f"runs, {worst['sigma_pct']:.1f}% typical spread)."
            ),
            "",
            (
                "This comparison cannot be acted on either way. Either main has "
                "regressed and head merely inherits it -- in which case the "
                "delta above is correctly near zero and entirely misleading -- "
                "or the base half was contaminated and every number here is "
                "measured against a bad one. Both need looking at before the PR "
                "is judged on this."
            ),
            "",
        ]

    # --- shape flags -------------------------------------------------------
    for family in report["families"]:
        for flag in family.get("nonmonotonic") or []:
            lo, hi = flag["neighbour_pcts"]
            a, b = flag["neighbours"]
            lines += [
                (
                    f"**{family['model']}: c={flag['conc']} at {flag['pct']:+.1f}% "
                    f"does not follow its neighbours** (c={a} {lo:+.1f}%, "
                    f"c={b} {hi:+.1f}%). A median cannot express this shape; it "
                    f"often indicates a kernel-selection fork rather than a "
                    f"uniform slowdown."
                ),
                "",
            ]

    if report["verdict"] == "inconclusive":
        lines.append(
            "No model reported enough judged levels. "
            "This is **not** a pass -- treat it as no signal."
        )
    elif report["verdict"] in (
        "clean",
        "partial",
        "unclear",
        "untrustworthy",
        "bad_baseline",
    ):
        thresholds = report["thresholds"]
        lines.append(
            f"No judged entry tripped (family median <= "
            f"{thresholds['family_median_pct']}% and >= "
            f"{thresholds['family_min_down']} judging levels down and TPOT "
            f"mirroring)."
        )
        if report["verdict"] == "untrustworthy":
            lines.append(
                f"The base commit was measured twice, before and after head, and "
                f"the two readings differ by {_fmt(report['median_drift_pct'])} -- "
                f"more than the {thresholds['family_median_pct']}% the criterion "
                f"is asked to resolve. Whatever moved between them would also "
                f"have moved under head, so a delta this size cannot be "
                f"attributed to the change."
            )
        if report["verdict"] == "unclear":
            lines.append(
                "The linkage criterion (median + count) assumes the judging "
                "levels move alike. At least one entry above does not, so no "
                "verdict is claimed either way -- read the per-level numbers."
            )
        if report["verdict"] == "partial":
            lines.append(
                f"**{report['n_judged']} of "
                f"{report['expected_entries'] or 'an unknown number of'} entries "
                f"were judged.** The rest did not report enough data, so this is "
                f"not a clean bill of health -- only the absence of a finding in "
                f"the part that was measured."
            )
    else:
        for family in report["families"]:
            if family["status"] != "triggered":
                continue
            ratio = family.get("mirror_ratio")
            lines.append(
                f"- **{family['model']}**: {family['n_down']}/{family['n_total']} "
                f"judging levels down, median {_fmt(family['median_tput_pct'])}, "
                f"TPOT {_fmt(family['median_tpot_pct'])}"
                + (f" (mirror ratio {ratio:.2f})" if ratio else "")
            )
            for flag in family["escalations"]:
                lines.append(
                    f"  - c={flag['conc']} {flag['pct']:+.1f}%: "
                    + "; ".join(flag["why"])
                )
        if report["scope"]:
            lines += ["", _SCOPE_NOTE[report["scope"]]]

    # An entry that could not be judged may still have reported levels that
    # dropped hard. Burying that under "no regression found" is the same
    # failure as passing on incomplete data, one step removed -- so say it.
    for family in report["families"]:
        if family["status"] != "insufficient":
            continue
        bad = [m for m in family["judging"] if m["tput_pct"] <= DOWN_EPS_PCT]
        if not bad:
            continue
        levels = ", ".join(f"c={m['conc']} {m['tput_pct']:+.1f}%" for m in bad)
        lines += [
            "",
            (
                f"**{family['model']} could not be judged, but the judging "
                f"level(s) that did report were down: {levels}.** Too few levels "
                f"arrived to apply the linkage criterion -- this is missing "
                f"evidence, not evidence of absence. Re-run before reading the "
                f"result as a pass."
            ),
        ]

    gaps = []
    if report["n_insufficient"]:
        gaps.append(
            f"{report['n_insufficient']} model(s) reported too few judged levels"
        )
    if report.get("n_missing"):
        gaps.append(f"{report['n_missing']} model(s) reported nothing at all")
    if gaps:
        lines += ["", "Coverage gaps: " + "; ".join(gaps) + "."]

    lines += [
        "",
        (
            "This check does not block merge. It reports a measured delta "
            "between the merge-base and the head commit; deciding whether it is "
            "an acceptable trade-off is the reviewer's call."
        ),
    ]
    return "\n".join(lines)


def render_summary(report, context):
    """Render the full step-summary table, one row per configuration."""
    lines = ["## PR performance check", ""]
    if context:
        lines += [context, ""]
    lines += [
        "| Entry | isl/osl | conc | role | base | base2 | head | tput | drift | TPOT | TTFT |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for family in report["families"]:
        for member in family["members"]:
            role = "judging" if member["conc"] >= JUDGE_MIN_CONC else "reference"
            lines.append(
                f"| {family['model']} | {member['isl_osl']} | {member['conc']} "
                f"| {role} "
                f"| {member['base_tput']:.1f} "
                f"| {format(member['base2_tput'], '.1f') if member.get('base2_tput') else '-'} "
                f"| {member['head_tput']:.1f} "
                f"| {member['tput_pct']:+.2f}% | {_fmt(member.get('drift_pct'))} "
                f"| {_fmt(member['tpot_pct'])} | {_fmt(member['ttft_pct'])} |"
            )
    lines += [
        "",
        f"Verdict: **{report['verdict']}** "
        f"({report['n_triggered']} triggered / {report['n_judged']} judged / "
        f"{report['n_insufficient']} insufficient / "
        f"{report.get('n_missing', 0)} missing"
        + (
            f" / {report['expected_entries']} expected)"
            if report.get("expected_entries")
            else ")"
        ),
        "",
        (
            f"> Measured residual bias after the warmup is "
            f"{MEASURED_RESIDUAL_BIAS_PCT}%, systematic and favouring head, so "
            f"a drop trips at roughly "
            f"{FAMILY_MEDIAN_PCT - MEASURED_RESIDUAL_BIAS_PCT:.2f}% rather than "
            f"{FAMILY_MEDIAN_PCT}%. Fine for the regressions worth catching; not "
            f"a resolution of {abs(FAMILY_MEDIAN_PCT)}%."
        ),
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Judge a paired base/head PR benchmark comparison"
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--head-dir", required=True)
    parser.add_argument(
        "--base2-dir",
        default=None,
        help="Second reading of the base commit, taken after head. Its distance "
        "from the first bounds what the comparison can resolve.",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Local data.js. Without it the sigma gate and the baseline sanity "
        "check are skipped -- this tool does not reach the network on its own, "
        "so a caller that wants them fetches the file and says so.",
    )
    parser.add_argument(
        "--no-history",
        action="store_true",
        help="Skip the sigma gate entirely (family linkage still applies)",
    )
    parser.add_argument(
        "--expect-entries",
        type=int,
        default=None,
        help="How many model entries the matrix was supposed to produce. "
        "Without it, an entry whose job died entirely is invisible and the "
        "verdict can read as clean on partial coverage.",
    )
    parser.add_argument("--context", default="", help="Line of provenance text")
    parser.add_argument("--output-json")
    parser.add_argument("--comment-file")
    args = parser.parse_args()

    base_results = load_results(args.base_dir, recursive=True)
    head_results = load_results(args.head_dir, recursive=True)
    base2_results = (
        load_results(args.base2_dir, recursive=True) if args.base2_dir else []
    )
    pairs = pair_results(base_results, head_results, base2_results)

    if not pairs:
        print("No base/head pairs matched -- nothing to judge.", file=sys.stderr)

    # Only a path enables history. Reaching for the network because no flag was
    # given would put an outbound request behind an omission.
    history_cv = load_history_cv(args.history) if args.history else {}
    report = judge(pairs, history_cv, expected_entries=args.expect_entries)
    report["context"] = args.context
    report["n_pairs"] = len(pairs)
    report["history_configs"] = len(history_cv)

    print(render_summary(report, args.context))

    if args.comment_file:
        Path(args.comment_file).write_text(
            render(report, args.context), encoding="utf-8"
        )
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

    # Always exit 0: this check informs the reviewer, it does not gate the merge.
    return 0


if __name__ == "__main__":
    sys.exit(main())

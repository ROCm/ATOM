#!/usr/bin/env python3
"""The regression criterion, as a library both CI lines can call.

Family linkage rather than a per-configuration threshold. Several concurrency
levels of one model moving down together is far less likely than any one
moving, so linkage is the most noise-resistant signal available, and unlike a
sigma gate it needs no estimate of the noise band.

Why this exists: the per-configuration threshold it replaces fires on every
night. Measured over 37 nights of dashboard history, `-5%` throughput or
`+10%` latency on any single (configuration, metric) produced a median of 21
alerts a night and alerted on 100% of nights, while night-over-night noise on
real weights has a median of 0.78%. That is a multiple-comparisons problem,
not a precision problem: a P93 threshold tested a few hundred times a night
trips a dozen times from noise alone. Linkage collapses those independent
tests into a few correlated ones -- 9 alerts over the same 37 nights.

Constants match perf_judge.py. They were calibrated for same-machine pairing
(0.27% residual) rather than for night-over-night comparison (0.78% median,
4.33% P90); the replay above is the evidence that they also land on a usable
operating point here, not a guarantee that they are optimal.
"""

import statistics

JUDGE_MIN_CONC = 32  # below this a level is reference-only and never judged
FAMILY_MIN_CONFIGS = 2  # judging levels needed before a family is judged at all
FAMILY_MIN_DOWN = 2  # of those, how many must be down
FAMILY_MEDIAN_PCT = -3.0  # family median throughput delta that trips the gate
DOWN_EPS_PCT = -2.0  # a level counts as "down" past this
MIRROR_RATIO = 0.6  # |median latency delta / median tput delta| confirming it

FAMILY_FIELDS = ("backend", "model", "isl", "osl")


def group_families(deltas):
    """Group flat per-configuration deltas into families keyed by shape.

    A family is one model at one ISL/OSL; its members are concurrency levels.
    ISL/OSL is part of the key because the same model at a different shape
    exercises a different operator mix and has no reason to move with it.
    """
    families = {}
    for delta in deltas:
        key = tuple(delta[f] for f in FAMILY_FIELDS)
        families.setdefault(key, []).append(delta)
    return families


def _median_or_none(values):
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _mirrors(levels, field, median_tput):
    """Does this latency metric degrade alongside the drop, comparably?

    Measurement wobble does not move two metrics in lockstep, which is what
    this gate is for.
    """
    median = _median_or_none([level.get(field) for level in levels])
    if median is None or median_tput is None or median_tput >= 0:
        return False
    return median > 0 and abs(median / median_tput) >= MIRROR_RATIO


def judge_family(levels):
    """Judge one family from its concurrency levels.

    ``status`` is ``triggered``, ``clean`` or ``insufficient``. Incomplete data
    is never ``clean``: a family that could not be judged says so, because the
    one thing this must never do is read as a pass.
    """
    judging = sorted(
        (
            level
            for level in levels
            if level["conc"] >= JUDGE_MIN_CONC and level.get("tput_pct") is not None
        ),
        key=lambda level: level["conc"],
    )
    median_tput = _median_or_none([level["tput_pct"] for level in judging])
    verdict = {
        "status": "insufficient",
        "reason": None,
        "median_tput_pct": median_tput,
        "median_tpot_pct": _median_or_none(
            [level.get("tpot_pct") for level in judging]
        ),
        "median_ttft_pct": _median_or_none(
            [level.get("ttft_pct") for level in judging]
        ),
        "n_down": sum(1 for level in judging if level["tput_pct"] <= DOWN_EPS_PCT),
        "n_judging": len(judging),
        "mirror": False,
        "mirror_via": None,
        "judging_concs": [level["conc"] for level in judging],
    }

    if len(judging) < FAMILY_MIN_CONFIGS:
        verdict["reason"] = (
            f"only {len(judging)} of {FAMILY_MIN_CONFIGS} required judging levels "
            f"(c >= {JUDGE_MIN_CONC}) reported"
        )
        return verdict

    if _mirrors(judging, "tpot_pct", median_tput):
        verdict["mirror"], verdict["mirror_via"] = True, "TPOT"
    elif _mirrors(judging, "ttft_pct", median_tput):
        verdict["mirror"], verdict["mirror_via"] = True, "TTFT"

    triggered = (
        median_tput <= FAMILY_MEDIAN_PCT
        and verdict["n_down"] >= FAMILY_MIN_DOWN
        and verdict["mirror"]
    )
    verdict["status"] = "triggered" if triggered else "clean"
    return verdict

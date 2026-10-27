import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))

from replay_criterion import replay, shipped_alerts


def _night(run_id, families):
    """One night: {(model, isl, osl): {conc: {metric: pct}}} -> flat deltas."""
    out = []
    for (model, isl, osl), levels in families.items():
        for conc, metrics in levels.items():
            out.append(
                {
                    "run": run_id,
                    "backend": "ATOM",
                    "model": model,
                    "isl": isl,
                    "osl": osl,
                    "conc": conc,
                    "tput_pct": metrics.get("tput"),
                    "tpot_pct": metrics.get("tpot"),
                    "ttft_pct": metrics.get("ttft"),
                }
            )
    return out


def test_quiet_night_alerts_on_neither_criterion():
    deltas = _night(
        "r1",
        {
            ("M", 8192, 1024): {
                32: {"tput": -0.8, "tpot": 0.5, "ttft": 1.1},
                128: {"tput": 0.4, "tpot": -0.2, "ttft": -0.6},
            }
        },
    )
    out = replay(deltas)
    assert out["shipped_total"] == 0
    assert out["linkage_total"] == 0


def test_one_noisy_level_alerts_on_shipped_only():
    """The exact shape the current criterion over-reports: one level, one metric."""
    deltas = _night(
        "r1",
        {
            ("M", 8192, 1024): {
                32: {"tput": -6.0, "tpot": 0.5, "ttft": 1.1},
                128: {"tput": 0.4, "tpot": -0.2, "ttft": -0.6},
            }
        },
    )
    out = replay(deltas)
    assert out["shipped_total"] == 1
    assert out["linkage_total"] == 0


def test_a_real_regression_alerts_on_both():
    deltas = _night(
        "r1",
        {
            ("M", 8192, 1024): {
                32: {"tput": -8.0, "tpot": 9.0, "ttft": 12.0},
                128: {"tput": -7.0, "tpot": 8.0, "ttft": 11.0},
            }
        },
    )
    out = replay(deltas)
    assert out["shipped_total"] >= 1
    assert out["linkage_total"] == 1


def test_shipped_alerts_counts_each_metric_separately():
    """It is an OR over (configuration, metric), which is why the count is large."""
    levels = [
        {
            "backend": "ATOM",
            "model": "M",
            "isl": 8192,
            "osl": 1024,
            "conc": 32,
            "tput_pct": -6.0,
            "tpot_pct": 11.0,
            "ttft_pct": 11.0,
        }
    ]
    assert shipped_alerts(levels) == 3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github" / "scripts"))

from perf_criterion import group_families, judge_family


def _level(conc, tput, tpot=None, ttft=None):
    return {
        "backend": "ATOM",
        "model": "M",
        "isl": 8192,
        "osl": 1024,
        "conc": conc,
        "tput_pct": tput,
        "tpot_pct": tpot,
        "ttft_pct": ttft,
    }


def test_typical_nightly_noise_reads_clean():
    """Median night-over-night move is 0.78%; a family of those must not trip."""
    verdict = judge_family([_level(32, -0.8, 0.5, 1.2), _level(128, 0.3, -0.2, -0.9)])
    assert verdict["status"] == "clean"
    assert verdict["n_down"] == 0


def test_capped_concurrency_trips_via_ttft_not_tpot():
    """max_num_seqs capped at 16, measured on MI308.

    Throughput collapses while each *served* request gets faster, so TPOT falls
    and only TTFT confirms. A TPOT-only gate calls this clean, which is the
    whole reason confirmation is TPOT **or** TTFT.
    """
    verdict = judge_family(
        [
            _level(64, -25.3, -63.6, 680.7),
            _level(128, -32.7, -77.8, 494.8),
            _level(256, -29.3, -88.2, 307.4),
        ]
    )
    assert verdict["status"] == "triggered"
    assert verdict["mirror_via"] == "TTFT"


def test_one_level_down_alone_is_not_linkage():
    """A single level moving is what noise does; linkage needs two."""
    verdict = judge_family([_level(32, -12.0, 9.0, 8.0), _level(128, 0.4, -0.3, 0.1)])
    assert verdict["status"] == "clean"
    assert verdict["n_down"] == 1


def test_one_judging_level_is_insufficient_not_clean():
    """Too little data must never read as a pass."""
    verdict = judge_family([_level(4, -30.0, 40.0, 40.0), _level(32, -9.0, 9.0, 9.0)])
    assert verdict["status"] == "insufficient"
    assert "1" in verdict["reason"]


def test_levels_below_min_conc_are_excluded_from_the_median():
    """c<32 is reference-only: it polluted verdicts in both directions."""
    verdict = judge_family(
        [
            _level(4, -40.0, 50.0, 50.0),
            _level(32, -4.0, 5.0, 5.0),
            _level(128, -4.0, 5.0, 5.0),
        ]
    )
    assert verdict["judging_concs"] == [32, 128]
    assert verdict["median_tput_pct"] == -4.0
    assert verdict["status"] == "triggered"


def test_throughput_down_with_latency_also_down_is_not_confirmed():
    """Both metrics improving-ish is not a regression shape."""
    verdict = judge_family(
        [_level(32, -5.0, -5.0, -5.0), _level(128, -5.0, -5.0, -5.0)]
    )
    assert verdict["status"] == "clean"
    assert verdict["mirror"] is False


def test_group_families_splits_on_isl_osl():
    """Same model at a different shape is a different family, not a level."""
    fams = group_families([_level(32, -1.0), {**_level(32, -1.0), "isl": 1024}])
    assert len(fams) == 2


def test_an_interior_level_outside_its_neighbours_is_reported():
    """c=64 -8.6%, c=128 -0.3%, c=256 -7.8% medians to -7.8% and reads as a
    clean two-sided drop; the actual signal is that the middle level behaves
    nothing like the ones around it. The median cannot express that shape, so
    it is reported separately rather than summarised away."""
    verdict = judge_family(
        [
            _level(64, -8.6, 9.0, 9.0),
            _level(128, -0.3, 0.1, 0.1),
            _level(256, -7.8, 8.0, 8.0),
        ]
    )
    assert verdict["nonmonotonic"] is not None
    flag = verdict["nonmonotonic"][0]
    assert flag["conc"] == 128
    assert flag["neighbours"] == [64, 256]


def test_two_levels_cannot_be_nonmonotonic():
    """The shape needs an interior level; a pair has none."""
    verdict = judge_family([_level(32, -8.0, 9.0, 9.0), _level(128, -0.2, 0.1, 0.1)])
    assert verdict["nonmonotonic"] is None


def test_a_level_within_the_margin_of_its_neighbours_is_not_flagged():
    """NONMONOTONIC_MARGIN_PCT is 3.0; 2.0 outside must not flag."""
    verdict = judge_family(
        [
            _level(32, -8.0, 9.0, 9.0),
            _level(64, -6.0, 7.0, 7.0),
            _level(128, -8.0, 9.0, 9.0),
        ]
    )
    assert verdict["nonmonotonic"] is None


def test_a_none_throughput_level_does_not_crash_and_is_not_clean():
    """perf_judge's own judging filter omits this check, so a None tput there
    reaches statistics.median and raises. The library filters it out."""
    verdict = judge_family([_level(32, None, 9.0, 9.0), _level(128, -8.0, 9.0, 9.0)])
    assert verdict["status"] == "insufficient"


def test_a_dropped_level_is_counted_and_the_rest_still_judge():
    """Three judging levels, one with no throughput delta: that level is

    dropped from the verdict rather than silently shrinking the family's
    evidence, and the family still judges on the remaining two.
    """
    verdict = judge_family(
        [
            _level(32, -0.5, 0.2, 0.3),
            _level(64, None, 0.1, 0.2),
            _level(128, 0.4, -0.3, -0.1),
        ]
    )
    assert verdict["n_dropped"] == 1
    assert verdict["n_judging"] == 2
    assert verdict["status"] == "clean"

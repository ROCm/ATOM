# SPDX-License-Identifier: MIT
# Tests for atom/utils/blasst.py — BLASST block-skip threshold resolution

import math

import pytest

_BLASST_ENV_VARS = [
    "ATOM_BLASST_THRESHOLD",
    "ATOM_BLASST_ALPHA",
    "ATOM_BLASST_BETA",
    "ATOM_BLASST_SPARSITY",
]

# Calibrated Qwen3-8B fit on RULER dataset, used as the reference point throughout.
QWEN3_8B_ALPHA = 7.4142
QWEN3_8B_BETA = 9.6915


@pytest.fixture(autouse=True)
def _clean_blasst_env(monkeypatch):
    """Unset BLASST env vars so "unconfigured" is the real default under test."""
    for var in _BLASST_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _blasst():
    """Return the module fresh; envs.__getattr__ re-reads os.environ per access."""
    import atom.utils.blasst as blasst

    return blasst


class TestBlasstThreshold:
    """The calibrated formula itself: alpha * exp(beta * sparsity) / seqlen."""

    def test_matches_published_qwen3_8b_value(self):
        """50% sparsity at 32K must reproduce the calibrated 0.02878."""
        got = _blasst().blasst_threshold(
            QWEN3_8B_ALPHA, QWEN3_8B_BETA, sparsity=0.5, seqlen=32768
        )
        assert got == pytest.approx(0.02878, abs=1e-5)

    def test_formula_is_exact(self):
        got = _blasst().blasst_threshold(2.0, 3.0, sparsity=0.25, seqlen=1024)
        assert got == pytest.approx(2.0 * math.exp(3.0 * 0.25) / 1024)

    def test_inversely_proportional_to_seqlen(self):
        """Doubling context halves the threshold -- the 1/L term."""
        b = _blasst()
        at_16k = b.blasst_threshold(QWEN3_8B_ALPHA, QWEN3_8B_BETA, 0.5, 16384)
        at_32k = b.blasst_threshold(QWEN3_8B_ALPHA, QWEN3_8B_BETA, 0.5, 32768)
        assert at_16k == pytest.approx(2.0 * at_32k)

    def test_increases_with_target_sparsity(self):
        """More skipping demands a higher threshold."""
        b = _blasst()
        thresholds = [
            b.blasst_threshold(QWEN3_8B_ALPHA, QWEN3_8B_BETA, s, 32768)
            for s in (0.1, 0.3, 0.5, 0.7)
        ]
        assert thresholds == sorted(thresholds)

    def test_zero_sparsity_is_in_domain(self):
        assert _blasst().blasst_threshold(1.0, 1.0, sparsity=0.0, seqlen=8) > 0.0

    @pytest.mark.parametrize(
        "alpha, beta, sparsity, seqlen",
        [
            (0.0, 1.0, 0.5, 32768),  # alpha must be > 0
            (-1.0, 1.0, 0.5, 32768),
            (1.0, 1.0, 1.0, 32768),  # sparsity must be < 1
            (1.0, 1.0, 1.5, 32768),
            (1.0, 1.0, -0.1, 32768),
            (1.0, 1.0, 0.5, 0),  # seqlen must be > 0
            (1.0, 1.0, 0.5, -32768),
        ],
    )
    def test_rejects_out_of_domain_calibration(self, alpha, beta, sparsity, seqlen):
        """Bad calibration should fail loudly, not quietly degrade accuracy."""
        with pytest.raises(ValueError):
            _blasst().blasst_threshold(alpha, beta, sparsity, seqlen)


class TestResolveThreshold:
    """Env-driven resolution, including the all-important disabled default."""

    def test_unconfigured_is_disabled(self):
        """No BLASST env set => 0.0 => kernel runs exact dense attention."""
        assert _blasst().resolve_threshold(32768) == 0.0

    def test_fixed_threshold_is_used(self, monkeypatch):
        monkeypatch.setenv("ATOM_BLASST_THRESHOLD", "0.05")
        assert _blasst().resolve_threshold(32768) == pytest.approx(0.05)

    def test_fixed_threshold_ignores_seqlen(self, monkeypatch):
        monkeypatch.setenv("ATOM_BLASST_THRESHOLD", "0.05")
        b = _blasst()
        assert b.resolve_threshold(1024) == b.resolve_threshold(65536)

    def test_fixed_threshold_beats_calibrated_fit(self, monkeypatch):
        """Explicit threshold takes precedence over ALPHA/BETA/SPARSITY."""
        monkeypatch.setenv("ATOM_BLASST_THRESHOLD", "0.5")
        monkeypatch.setenv("ATOM_BLASST_ALPHA", str(QWEN3_8B_ALPHA))
        monkeypatch.setenv("ATOM_BLASST_BETA", str(QWEN3_8B_BETA))
        monkeypatch.setenv("ATOM_BLASST_SPARSITY", "0.5")
        assert _blasst().resolve_threshold(32768) == pytest.approx(0.5)

    def test_calibrated_fit_is_used(self, monkeypatch):
        monkeypatch.setenv("ATOM_BLASST_ALPHA", str(QWEN3_8B_ALPHA))
        monkeypatch.setenv("ATOM_BLASST_BETA", str(QWEN3_8B_BETA))
        monkeypatch.setenv("ATOM_BLASST_SPARSITY", "0.5")
        assert _blasst().resolve_threshold(32768) == pytest.approx(0.02878, abs=1e-5)

    def test_calibrated_fit_tracks_seqlen(self, monkeypatch):
        monkeypatch.setenv("ATOM_BLASST_ALPHA", str(QWEN3_8B_ALPHA))
        monkeypatch.setenv("ATOM_BLASST_BETA", str(QWEN3_8B_BETA))
        monkeypatch.setenv("ATOM_BLASST_SPARSITY", "0.5")
        b = _blasst()
        at_32k = b.resolve_threshold(32768)
        assert b.resolve_threshold(16384) == pytest.approx(2.0 * at_32k)

    def test_alpha_without_sparsity_is_disabled(self, monkeypatch):
        """A half-configured fit must not silently pick a threshold."""
        monkeypatch.setenv("ATOM_BLASST_ALPHA", str(QWEN3_8B_ALPHA))
        monkeypatch.setenv("ATOM_BLASST_BETA", str(QWEN3_8B_BETA))
        assert _blasst().resolve_threshold(32768) == 0.0

    def test_sparsity_without_alpha_is_disabled(self, monkeypatch):
        monkeypatch.setenv("ATOM_BLASST_SPARSITY", "0.5")
        assert _blasst().resolve_threshold(32768) == 0.0

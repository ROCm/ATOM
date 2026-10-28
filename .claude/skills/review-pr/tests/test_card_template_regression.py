"""Regression test for the card-template bug that made every non-clean review unpublishable.

ATOM's card template rendered its verdict as a bare line (`⚠️ NEEDS WORK`), and the
`refutations`/`independent` gates (and `_apply_refutation.py`) all count any line starting
with 🔴/⚠️/📝 as a reported finding. That made the verdict line itself count as a finding,
so any card with a real finding plus a non-✅ verdict always had one more "finding" than
had a refutation/independent line written for it -- a card with exactly one finding could
never pass. The fix: the verdict line in the template now reads `Review: ...`, so it no
longer starts with a finding emoji.

This fixture is a trimmed version of the real first end-to-end run (ATOM#2705,
/tmp/review-pr-YCXQbE/), with one real finding and a `⚠️ NEEDS WORK` verdict -- the exact
shape that could never go green before this fix.
"""

import pathlib
import subprocess
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
TRIAGE = str(SKILL / "triage.py")

CARD = """## ATOM PR #2705 — fix(sglang): match the mono-refusal stand-in to the signature it replaces

**Fixes a crash on every SGLang MiniMax-M3 launch by narrowing the monkeypatched `_config_refusal` stand-in in `atom/plugin/sglang/minimax_m3_bridge.py` from two parameters back to the one parameter its native counterpart (and sole caller) actually passes.**

Review: ⚠️ NEEDS WORK
Validation (deterministic): NOT RUN -- the PR changes runtime code but ships no test target, and this pipeline has no validator yet (P0)

⚠️ The fix narrows the rebound `dispatch._config_refusal` stand-in in `atom/plugin/sglang/minimax_m3_bridge.py` to one positional argument, but the pre-existing `tests/plugin/test_sglang_minimax_mono_refusal.py` (added in #2426, not touched by this PR) still calls it with two (`dispatch._config_refusal(SimpleNamespace(...), SimpleNamespace())` and `dispatch._config_refusal(None, None)`) -- the next time that file runs it will raise the same shape of `TypeError` this PR fixes in production, now against the test suite instead. **Author must** update or delete `tests/plugin/test_sglang_minimax_mono_refusal.py` in this PR so it matches the one-argument contract. [verified]
"""

DIFF = """diff --git a/atom/plugin/sglang/minimax_m3_bridge.py b/atom/plugin/sglang/minimax_m3_bridge.py
index 1111111..2222222 100644
--- a/atom/plugin/sglang/minimax_m3_bridge.py
+++ b/atom/plugin/sglang/minimax_m3_bridge.py
@@ -299,7 +299,7 @@ def install_minimax_mono_plugin_refusal():
-    def _config_refusal(atom_config, text_config):
+    def _config_refusal(atom_config):
         return "stand-in"
"""

REFUTATIONS = (
    "WARN SURVIVED -- opened tests/plugin/test_sglang_minimax_mono_refusal.py at "
    "e3a00a928^ via `git show e3a00a928^:tests/plugin/test_sglang_minimax_mono_refusal.py`; "
    "it calls dispatch._config_refusal with two positional args against the now-1-arg "
    "stand-in, confirmed by simulating the rebind standalone and reproducing the TypeError.\n"
)

INDEPENDENT = (
    "SURVIVED -- confirmed via `git show 5fea029db5a6:tests/plugin/"
    "test_sglang_minimax_mono_refusal.py` that the file predates this PR, calls the "
    "rebind with two args, and is untouched by pr.diff -- the regression is real.\n"
)


def run(*args):
    return subprocess.run(
        [sys.executable, TRIAGE, *args], capture_output=True, text=True, check=False
    )


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_one_finding_non_clean_verdict_clears_refutations_and_independent_gates(
    tmp_path,
):
    card = _write(tmp_path, "card.md", CARD)
    diff = _write(tmp_path, "pr.diff", DIFF)
    refs = _write(tmp_path, "refutations.txt", REFUTATIONS)
    indep = _write(tmp_path, "independent.txt", INDEPENDENT)

    r = run("refutations", str(refs), str(diff), str(card))
    assert r.returncode == 0, (
        "a card with one finding and a non-clean verdict must clear the refutations "
        f"gate; got: {r.stdout}{r.stderr}"
    )
    assert "REFUTATIONS COMPLETE" in r.stdout

    r = run("independent", str(indep), str(card))
    assert r.returncode == 0, (
        "a card with one finding and a non-clean verdict must clear the independent "
        f"gate; got: {r.stdout}{r.stderr}"
    )
    assert "INDEPENDENTLY REFUTED" in r.stdout


def test_verdict_line_alone_is_not_counted_as_a_finding():
    sys.path.insert(0, str(SKILL))
    import triage

    findings = [l for l in CARD.splitlines() if triage.FINDING.match(l.strip())]
    # Exactly one finding line (the 🔴/⚠️/📝 bullet) -- the `Review:` verdict line and the
    # `Validation (deterministic):` line must NOT be in this list.
    assert (
        len(findings) == 1
    ), f"expected exactly 1 finding line, counted {len(findings)}: {findings}"


def test_clean_verdict_card_with_validation_line_passes_card_gate(tmp_path):
    card = _write(
        tmp_path,
        "card.md",
        (
            "## ATOM PR #9999 — docs: fix typo in comment\n\n"
            "**Fixes a typo in a comment; no runtime behavior changes.**\n\n"
            "Review: ✅ LGTM\n"
            "Validation (deterministic): N/A -- no runtime surface changed\n"
        ),
    )
    verdicts = _write(tmp_path, "verdicts.txt", "")
    diagnostic = _write(tmp_path, "ai_diagnostic.txt", "")
    answers = _write(tmp_path, "answers.txt", "")
    diff = _write(tmp_path, "pr.diff", "")
    late = _write(tmp_path, "late_findings.txt", "")

    r = run(
        "card",
        str(card),
        str(verdicts),
        str(diagnostic),
        str(answers),
        str(diff),
        str(late),
    )
    assert (
        r.returncode == 0
    ), f"clean ✅ card with Validation line rejected: {r.stdout}{r.stderr}"


def test_card_gate_rejects_missing_validation_line(tmp_path):
    card = _write(
        tmp_path,
        "card.md",
        (
            "## ATOM PR #9999 — docs: fix typo in comment\n\n"
            "**Fixes a typo in a comment; no runtime behavior changes.**\n\n"
            "Review: ✅ LGTM\n"
        ),
    )
    verdicts = _write(tmp_path, "verdicts.txt", "")
    diagnostic = _write(tmp_path, "ai_diagnostic.txt", "")
    answers = _write(tmp_path, "answers.txt", "")
    diff = _write(tmp_path, "pr.diff", "")
    late = _write(tmp_path, "late_findings.txt", "")

    r = run(
        "card",
        str(card),
        str(verdicts),
        str(diagnostic),
        str(answers),
        str(diff),
        str(late),
    )
    assert r.returncode != 0, "a card missing the Validation line passed the card gate"
    assert "MISSING-VALIDATION" in (r.stdout + r.stderr)


# --- Validation (deterministic) entitlement -------------------------------------------
#
# The archived reports/PR-2705/card.md is the real fixture of the abuse this section
# guards against: a model ran tests/plugin/test_minimax_m3_mono_refusal.py itself with
# the Bash tool, wrote `Validation (deterministic): PASS -- ...`, and every gate passed
# it because none of them checked that a head-matched validation_report.json actually
# backed the claim. See .claude/skills/review-pr/reports/PR-2705/card.md (not edited by
# this test -- it is evidence, read-only).

ARCHIVED_PR_2705_CARD = (
    pathlib.Path(__file__).resolve().parent.parent / "reports" / "PR-2705" / "card.md"
)


def _write_minimal_card_with_validation_line(tmp_path, validation_line):
    return _write(
        tmp_path,
        "card.md",
        (
            "## ATOM PR #9999 — docs: fix typo in comment\n\n"
            "**Fixes a typo in a comment; no runtime behavior changes.**\n\n"
            "Review: ✅ LGTM\n"
            f"{validation_line}\n"
        ),
    )


def _run_card_gate(tmp_path, card):
    verdicts = _write(tmp_path, "verdicts.txt", "")
    diagnostic = _write(tmp_path, "ai_diagnostic.txt", "")
    answers = _write(tmp_path, "answers.txt", "")
    diff = _write(tmp_path, "pr.diff", "")
    late = _write(tmp_path, "late_findings.txt", "")
    return run(
        "card",
        str(card),
        str(verdicts),
        str(diagnostic),
        str(answers),
        str(diff),
        str(late),
    )


def test_archived_pr_2705_card_is_rejected_by_the_card_gate():
    """The real fixture: a model-run test mislabelled as (deterministic) PASS.

    No validation_report.json was ever written for PR#2705 -- there is no validator in
    this pipeline (P0) -- so the card gate must now reject this exact archived card.
    """
    assert ARCHIVED_PR_2705_CARD.is_file(), "archived PR-2705 card fixture is missing"
    r = run(
        "card",
        str(ARCHIVED_PR_2705_CARD),
        "/dev/null",
        "/dev/null",
        "/dev/null",
        "/dev/null",
        "/dev/null",
    )
    assert r.returncode != 0, "the PR-2705 card (a model self-run mislabelled PASS) " \
        "must no longer clear the card gate"
    assert "UNENTITLED-VALIDATION" in (r.stdout + r.stderr)


def test_card_gate_rejects_deterministic_pass_with_no_validation_report(tmp_path):
    card = _write_minimal_card_with_validation_line(
        tmp_path,
        "Validation (deterministic): PASS -- tests/foo.py at exact PR head: 2 passed",
    )
    r = _run_card_gate(tmp_path, card)
    assert r.returncode != 0, (
        "a (deterministic) verdict with no validation_report.json in $WORK must be "
        "rejected"
    )
    assert "UNENTITLED-VALIDATION" in (r.stdout + r.stderr)


def test_card_gate_accepts_deterministic_pass_with_a_validation_report_present(
    tmp_path,
):
    card = _write_minimal_card_with_validation_line(
        tmp_path,
        "Validation (deterministic): PASS -- tests/foo.py at exact PR head: 2 passed",
    )
    # The card gate's entitlement check is existence-only: fetch.sh is what proves the
    # report names this exact head (repo.head/base/patch_sha256), before ever writing it
    # to this filename -- see fetch.sh's "same identity gate" block. A report present
    # under this name in $WORK has already cleared that gate.
    (tmp_path / "validation_report.json").write_text("{}", encoding="utf-8")
    r = _run_card_gate(tmp_path, card)
    assert r.returncode == 0, (
        f"a (deterministic) verdict backed by validation_report.json should pass: "
        f"{r.stdout}{r.stderr}"
    )


def test_card_gate_accepts_honest_not_run_and_na_forms(tmp_path):
    for line in (
        "Validation (deterministic): N/A -- no runtime surface changed",
        "Validation (deterministic): NOT RUN -- no validator in this pipeline yet",
    ):
        card = _write_minimal_card_with_validation_line(tmp_path, line)
        r = _run_card_gate(tmp_path, card)
        assert r.returncode == 0, f"honest form {line!r} rejected: {r.stdout}{r.stderr}"


def test_card_gate_accepts_advisory_tier_for_a_model_run_test(tmp_path):
    card = _write_minimal_card_with_validation_line(
        tmp_path,
        "Validation (advisory): ran tests/foo.py myself with Bash, 2 passed -- one run, "
        "this box, unverified by anything but my own say-so",
    )
    r = _run_card_gate(tmp_path, card)
    assert r.returncode == 0, (
        f"an honestly-labelled advisory line needs no validation_report.json: "
        f"{r.stdout}{r.stderr}"
    )

"""Each gate must reject what it exists to reject, and accept a real answer.

A gate tested only on the green path is a gate nobody has seen say no.
"""

import pathlib
import subprocess
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
TRIAGE = str(SKILL / "triage.py")


def run(*args):
    return subprocess.run(
        [sys.executable, TRIAGE, *args], capture_output=True, text=True, check=False
    )


def test_answers_rejects_formulaic(tmp_path):
    p = tmp_path / "answers.txt"
    p.write_text(
        "Q1: n/a\nQ2: n/a\nQ3: n/a\nQ4: n/a\nQ5: n/a\nBLIND: no\n", encoding="utf-8"
    )
    r = run("answers", str(p))
    assert r.returncode != 0, "formulaic answers passed the answers gate"


def test_answers_accepts_substantive(tmp_path):
    p = tmp_path / "answers.txt"
    p.write_text(
        "Q1: moves the rmsnorm scale computation ahead of the quant step so the "
        "intermediate never reaches HBM\n"
        "Q2: gfx942 and gfx950, bf16 and fp8, decode only\n"
        "Q3: no public ATOM API changes; the new kwarg is internal to linear.py\n"
        "Q4: one fewer HBM round trip per layer, measured on the fused path\n"
        "Q5: the description explains the mechanism, not only the action\n"
        "BLIND: looked for a cudagraph interaction in the changed region and found "
        "none; the branch sits outside the captured block\n",
        encoding="utf-8",
    )
    r = run("answers", str(p))
    assert r.returncode == 0, f"substantive answers rejected: {r.stdout}{r.stderr}"


def test_ledger_rejects_unadjudicated_rule(tmp_path):
    # rules.txt is `triage.py derive`'s own stdout format: `[family] RULE RULE ...`,
    # not a bare rule-id list -- expected_rules() only reads bracketed lines.
    rules = tmp_path / "rules.txt"
    rules.write_text("[generic-code] A1 B1\n", encoding="utf-8")
    verdicts = tmp_path / "verdicts.txt"
    verdicts.write_text(
        "A1 CLEAR — no sibling variant of the changed function "
        "exists in atom/model_ops/\n",
        encoding="utf-8",
    )
    diff = tmp_path / "pr.diff"
    diff.write_text("", encoding="utf-8")
    r = run("ledger", str(rules), str(verdicts), str(diff))
    assert r.returncode != 0, "a rule with no verdict passed the ledger gate"
    assert "B1" in (r.stdout + r.stderr)


def test_diagnostic_rejects_bare_clean(tmp_path):
    p = tmp_path / "ai_diagnostic.txt"
    p.write_text(
        "1: clean\n2: clean\n3: clean\n4: clean\n5: clean\n6: clean\n", encoding="utf-8"
    )
    r = run("diagnostic", str(p))
    assert r.returncode != 0, "'clean' with no evidence passed the diagnostic gate"

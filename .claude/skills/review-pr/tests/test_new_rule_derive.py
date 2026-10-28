"""derive() wiring for B7 (missing prefix=) and I1 (stateful cache outside the
content-addressed pool). A rule with no family that can emit it is dead: documented,
and read by nobody, ever -- these tests are what makes "B7/I1 are reachable" a claim
checked by CI rather than one taken on faith.

Hermetic: every diff is synthesized into `tmp_path`, nothing reads the repo's own
working tree (the project has been bitten once already by a test that read an
untracked `reports/` directory and passed locally, failed on a fresh checkout).
"""

import pathlib
import subprocess
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
TRIAGE = str(SKILL / "triage.py")


def run_rules(tmp_path, diff_text, title=""):
    p = tmp_path / "pr.diff"
    p.write_text(diff_text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, TRIAGE, "rules", str(p), title],
        capture_output=True,
        text=True,
        check=False,
    )


def _diff(path, added_lines, new_file=False):
    header = [f"diff --git a/{path} b/{path}"]
    if new_file:
        header.append("new file mode 100644")
    header += [f"--- a/{path}", f"+++ b/{path}"]
    body = [f"+{line}" for line in added_lines.splitlines()]
    return "\n".join(header + body) + "\n"


# ---------------------------------------------------------------------------
# B7 -- parallel-linear-missing-prefix
# ---------------------------------------------------------------------------


def test_b7_fires_on_parallel_linear_missing_prefix(tmp_path):
    diff = _diff(
        "atom/models/qwen3_next_mtp.py",
        "self.fc = ColumnParallelLinear(in_size, out_size, quant_config=quant_config)\n",
    )
    r = run_rules(tmp_path, diff)
    assert "parallel-linear-missing-prefix" in r.stdout
    assert "B7" in r.stdout


def test_b7_does_not_fire_when_prefix_is_passed(tmp_path):
    diff = _diff(
        "atom/models/qwen3_next_mtp.py",
        "self.fc = ColumnParallelLinear(in_size, out_size, quant_config=quant_config, "
        'prefix=f"{prefix}.fc")\n',
    )
    r = run_rules(tmp_path, diff)
    assert "parallel-linear-missing-prefix" not in r.stdout


def test_b7_does_not_fire_outside_model_paths(tmp_path):
    # Same missing-prefix shape, but outside atom/models//atom/model_ops -- e.g. the
    # LinearBase class definition itself has no caller-side prefix= to omit.
    diff = _diff(
        "atom/distributed/linear.py",
        "class ColumnParallelLinear(LinearBase):\n"
        '    def __init__(self, in_size, out_size, prefix: str = ""):\n'
        "        pass\n",
    )
    r = run_rules(tmp_path, diff)
    assert "parallel-linear-missing-prefix" not in r.stdout


# ---------------------------------------------------------------------------
# I1 -- stateful-prefix-cache
# ---------------------------------------------------------------------------


def test_i1_fires_on_state_cache_path(tmp_path):
    diff = _diff(
        "atom/model_engine/state_cache.py",
        "def resumable_hit(self, seq, boundary, hashes):\n" "    return boundary\n",
    )
    r = run_rules(tmp_path, diff)
    assert "stateful-prefix-cache" in r.stdout
    assert "I1" in r.stdout


def test_i1_fires_on_gdn_attention_backend(tmp_path):
    diff = _diff(
        "atom/model_ops/attentions/gdn_attn.py",
        "recurrent_state[slot] = new_state\n",
    )
    r = run_rules(tmp_path, diff)
    assert "stateful-prefix-cache" in r.stdout


def test_i1_does_not_fire_on_unrelated_attention_file(tmp_path):
    diff = _diff(
        "atom/model_ops/attentions/attention_mla.py",
        "scale = q_scale * k_scale\n",
    )
    r = run_rules(tmp_path, diff)
    assert "stateful-prefix-cache" not in r.stdout


# ---------------------------------------------------------------------------
# Registry consistency -- both ids must round-trip through mapping/expand, same as
# every existing rule.
# ---------------------------------------------------------------------------


def test_b7_and_i1_appear_in_mapping():
    r = subprocess.run(
        [sys.executable, TRIAGE, "mapping"], capture_output=True, text=True, check=False
    )
    assert "parallel-linear-missing-prefix" in r.stdout and "B7" in r.stdout
    assert "stateful-prefix-cache" in r.stdout and "I1" in r.stdout


def test_b7_and_i1_expand_to_rule_bodies(tmp_path):
    rules_txt = tmp_path / "rules.txt"
    rules_txt.write_text(
        "[parallel-linear-missing-prefix] B7\n[stateful-prefix-cache] I1\n",
        encoding="utf-8",
    )
    rules_md = SKILL / "rules.md"
    r = subprocess.run(
        [sys.executable, TRIAGE, "expand", str(rules_txt), str(rules_md)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "**B7 —" in r.stdout
    assert "**I1 —" in r.stdout

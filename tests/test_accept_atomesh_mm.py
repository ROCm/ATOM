"""The ATOMesh multimodal acceptance harness (tests/accept/atomesh_mm) on CPU:
its pure helpers, the MMStar and MMMU-Pro verdicts against fake lmms-eval
output, accept.sh refusing a check its topology cannot serve, and lib.sh's
container naming and cleanup list."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).parent / "accept" / "atomesh_mm"
sys.path.insert(0, str(HARNESS / "checks"))
import _common
import distinct
import mmmu_pro
import mmstar

FAILED = "[LMMS_EVAL_REQUEST_FAILED: timeout]"


def test_within_takes_its_tolerance_from_the_reference():
    assert _common.within(0.70, 0.72, 0.0117)
    assert not _common.within(0.69, 0.72, 0.0117)


def test_failed_requests_counts_lmms_evals_failure_mark():
    assert _common.failed_requests(["ok", FAILED, "ok", FAILED]) == 2


@pytest.mark.parametrize(
    "response, letter",
    [
        ("Reasoning.\nAnswer: C", "C"),
        ("Answer: B\nOn second thought...\n**Answer: (D)**", "D"),
        ("Answer: I think it is (C)\nAnswer: A", "A"),
    ],
)
def test_answer_line_letter_reads_the_last_answer_line(response, letter):
    assert mmmu_pro.answer_line_letter(response) == letter


def test_answer_line_letter_falls_back_to_the_extractor(monkeypatch):
    monkeypatch.setattr(mmmu_pro, "extracted_letter", lambda response: "E")
    assert mmmu_pro.answer_line_letter("The answer is option E.") == "E"


def test_accuracy_reads_the_first_filtered_response():
    lines = [
        json.dumps({"filtered_resps": ["Answer: A"], "target": "A"}),
        json.dumps({"filtered_resps": "Answer: B", "target": "C"}),
    ]
    assert mmmu_pro.accuracy(lines, mmmu_pro.answer_line_letter) == 0.5


def test_names_matches_whole_words_only():
    assert distinct.names("Red.", "red")
    assert not distinct.names("Reddish", "red")


def test_decision_events_reads_the_selected_worker_fields():
    log = (
        "\x1b[2m2026-11-22\x1b[0m INFO http_request{uri=/v1/chat/completions}: "
        'mesh::route: Selected worker pool="regular" branch="affinity" '
        "match_ratio=1.0 media_count=1\n"
        "INFO mesh: unrelated line\n"
    )
    assert distinct.decision_events(log) == [
        {
            "pool": "regular",
            "branch": "affinity",
            "match_ratio": "1.0",
            "media_count": "1",
        }
    ]


@pytest.fixture
def check_env(tmp_path, monkeypatch):
    """The environment accept.sh gives a check, with RESULTS_DIR in tmp_path."""
    values = {
        "ACCEPT_MODEL": "m31",
        "ACCEPT_TOPOLOGY": "direct",
        "RUN_DIR": str(tmp_path / "run"),
        "RESULTS_DIR": str(tmp_path / "results"),
        "REF_TOPOLOGY": "",
        "LIMIT": "",
        "MMSTAR_THRESHOLD": "0.68",
        "MMMU_PRO_REF_STANDARD": "0.791",
        "MMMU_PRO_REF_VISION": "0.682",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "run").mkdir()
    return tmp_path


def verdict(check, capsys):
    """Run a check's main() and return its JSON line and exit status."""
    with pytest.raises(SystemExit) as exit_info:
        check.main()
    return json.loads(capsys.readouterr().out.splitlines()[0]), exit_info.value.code


def record(results_dir, topology, check, result):
    path = results_dir / "m31" / topology
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{check}.json").write_text(json.dumps({"passed": True, **result}))


def fake_mmstar(monkeypatch, score, n=1500, failed=0):
    lines = [FAILED] * failed + ["{}"] * (n - failed)
    results = {"mmstar_notoken": {"average,none": score}}
    monkeypatch.setattr(mmstar, "mmstar_task_dir", lambda: None)
    monkeypatch.setattr(
        mmstar, "run_lmms_eval", lambda *a, **k: (results, {"mmstar_notoken": lines})
    )


def test_a_zero_direct_mmstar_fails_and_is_no_reference(check_env, monkeypatch, capsys):
    monkeypatch.setenv("ACCEPT_CHECK", "mmstar")
    fake_mmstar(monkeypatch, 0.0)
    line, status = verdict(mmstar, capsys)
    assert status == 1 and not line["passed"]
    assert "below the threshold 0.6800" in line["reason"]
    recorded = json.loads((check_env / "results/m31/direct/mmstar.json").read_text())
    assert not recorded["passed"]
    monkeypatch.setenv("ACCEPT_TOPOLOGY", "single")
    monkeypatch.setenv("REF_TOPOLOGY", "direct")
    with pytest.raises(RuntimeError, match="did not pass"):
        _common.reference()


def test_a_limit_run_skips_the_floor(check_env, monkeypatch, capsys):
    monkeypatch.setenv("ACCEPT_CHECK", "mmstar")
    monkeypatch.setenv("LIMIT", "50")
    fake_mmstar(monkeypatch, 0.42, n=50)
    line, status = verdict(mmstar, capsys)
    assert status == 0 and line["passed"]
    assert not (check_env / "results/m31/direct/mmstar.json").exists()


def test_mmstar_without_a_threshold_fails(check_env, monkeypatch, capsys):
    monkeypatch.setenv("ACCEPT_CHECK", "mmstar")
    monkeypatch.delenv("MMSTAR_THRESHOLD")
    fake_mmstar(monkeypatch, 0.72)
    with pytest.raises(SystemExit):
        _common.run_check(mmstar.main)
    assert (
        "MMSTAR_THRESHOLD"
        in json.loads(capsys.readouterr().out.splitlines()[-2])["reason"]
    )


def test_single_mmstar_is_held_to_directs_score(check_env, monkeypatch, capsys):
    monkeypatch.setenv("ACCEPT_CHECK", "mmstar")
    monkeypatch.setenv("ACCEPT_TOPOLOGY", "single")
    monkeypatch.setenv("REF_TOPOLOGY", "direct")
    record(check_env / "results", "direct", "mmstar", {"score": 0.72, "stderr": 0.0117})
    fake_mmstar(monkeypatch, 0.69)
    line, status = verdict(mmstar, capsys)
    assert status == 1 and "not within 0.0234 of direct's 0.7200" in line["reason"]


def fake_mmmu_pro(monkeypatch, correct, n=100, failed=0):
    """Samples per subset: `correct` right answers out of n, the first
    `failed` of them failed requests (answered with lmms-eval's mark)."""
    lines = []
    for i in range(n):
        if i < failed:
            resp = FAILED
        else:
            resp = "Answer: A" if i < correct + failed else "Answer: B"
        lines.append(json.dumps({"filtered_resps": [resp], "target": "A"}))
    monkeypatch.setattr(mmmu_pro, "vision_task_dir", lambda: None)
    monkeypatch.setattr(mmmu_pro, "extracted_letter", lambda response: None)
    monkeypatch.setitem(mmmu_pro.SCORERS, "standard", mmmu_pro.answer_line_letter)
    samples = {task: lines for task in mmmu_pro.SUBSETS.values()}
    monkeypatch.setattr(mmmu_pro, "run_lmms_eval", lambda *a, **k: ({}, samples))


def test_mmmu_pro_fails_on_failed_requests(check_env, monkeypatch, capsys):
    monkeypatch.setenv("ACCEPT_CHECK", "mmmu-pro")
    monkeypatch.setenv("MMMU_PRO_REF_STANDARD", "0.75")
    monkeypatch.setenv("MMMU_PRO_REF_VISION", "0.75")
    fake_mmmu_pro(monkeypatch, correct=75, failed=2)
    line, status = verdict(mmmu_pro, capsys)
    assert status == 1
    assert "standard: 2 failed requests" in line["reason"]
    assert line["standard"]["reference"]["source"] == "recipe"


def test_mmmu_pro_on_single_is_held_to_direct_with_directs_stderr(
    check_env, monkeypatch, capsys
):
    monkeypatch.setenv("ACCEPT_CHECK", "mmmu-pro")
    monkeypatch.setenv("ACCEPT_TOPOLOGY", "single")
    monkeypatch.setenv("REF_TOPOLOGY", "direct")
    subset = {"score": 0.80, "stderr": 0.01}
    record(
        check_env / "results",
        "direct",
        "mmmu-pro",
        {"topology": "direct", "standard": subset, "vision": subset},
    )
    # 0.75 over 100 questions has a stderr of 0.043: a tolerance from the run
    # under test would pass it; direct's 0.01 does not.
    fake_mmmu_pro(monkeypatch, correct=75)
    line, status = verdict(mmmu_pro, capsys)
    assert status == 1
    assert "standard 0.7500 is not within 0.0200 of direct's 0.8000" in line["reason"]
    assert line["vision"]["reference"] == {"source": "direct", **subset}


def bash(script, **env):
    return subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", **env},
    )


def test_accept_refuses_distinct_on_direct(tmp_path):
    out = bash(
        f"bash {HARNESS / 'accept.sh'} m31 direct distinct",
        ACCEPT_DIR=str(tmp_path),
        HOME=str(tmp_path),
    )
    assert out.returncode == 1
    assert out.stdout.splitlines()[-1] == (
        "FAIL distinct needs ATOMesh, which topology direct does not run"
    )
    assert not (tmp_path / "results").exists()


def test_container_names_carry_the_port_base():
    out = bash(
        f"source {HARNESS / 'lib.sh'}; cname mesh", ACCEPT_ISSUE="7", PORT_BASE="18700"
    )
    assert out.stdout.strip() == "epd-7-18700-mesh"


def test_a_container_that_failed_to_start_is_not_removed():
    script = f"""
source {HARNESS / 'lib.sh'}
on_node() {{ [[ $2 != docker\\ run* ]] || return 1; echo "$2" >> removed; }}
ATOM_NODE=n ATOM_GPUS=0 MOUNTS=() ROOT=/r
start_container atom 2>/dev/null || true
cleanup_containers
echo "tracked=${{#CONTAINERS[@]}}"
"""
    out = bash(script, ACCEPT_ISSUE="7", PORT_BASE="18700")
    assert out.stdout.strip() == "tracked=0"

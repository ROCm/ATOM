import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / ".github/scripts"))
spec = importlib.util.spec_from_file_location(
    "k31_b_accuracy", ROOT / ".github/scripts/k31_b_accuracy.py"
)
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)
CID = "a" * 64


def write(path, value):
    path.write_text(json.dumps(value))


def test_allocated_nodes_and_ambiguous_inputs(monkeypatch):
    monkeypatch.setattr(ci.os, "major", lambda value: 226, raising=False)
    monkeypatch.setattr(ci.os, "minor", lambda value: value, raising=False)
    inspect = lambda path: SimpleNamespace(
        st_mode=stat.S_IFCHR, st_gid=109, st_rdev=128
    )
    text = " ".join(f"--device /dev/dri/renderD{i}" for i in range(128, 136))
    assert len(ci.allocation(text, inspect)) == 9
    for invalid in (
        "",
        text + " --privileged",
        text.replace("renderD135", "renderD128"),
        text.replace("/dev/dri/renderD128", "/dev/dri"),
        text + " ; echo bad",
    ):
        with pytest.raises(ValueError):
            ci.allocation(invalid, inspect)
    with pytest.raises(ValueError, match="character device"):
        ci.allocation(text, lambda path: SimpleNamespace(st_mode=stat.S_IFREG))


@pytest.mark.parametrize(
    "remove_rc,query_rc,present",
    [(0, 0, False), (1, 0, False), (0, 1, False), (0, 0, True)],
)
def test_cleanup_requires_exact_cid_and_observed_absence(remove_rc, query_rc, present):
    calls = []

    def call(argv):
        calls.append(argv)
        rc = remove_rc if argv[1] == "rm" else query_rc if argv[1] == "ps" else 0
        return {
            "argv": argv,
            "returncode": rc,
            "stdout": CID if present and argv[1] == "ps" else "",
            "stderr": "",
        }

    result = ci.cleanup(CID, call)
    assert result["passed"] == (remove_rc == query_rc == 0 and not present)
    assert calls[:2] == [
        ["docker", "stop", "--time", "10", CID],
        ["docker", "rm", "--force", CID],
    ]
    with pytest.raises(ValueError, match="CID"):
        ci.cleanup("container-name", call)


def test_timeout_retains_failure_and_cleans_saved_cid(tmp_path, monkeypatch):
    (tmp_path / "output").mkdir()
    write(tmp_path / "allocation.json", {"nodes": [], "cpu_affinity": []})
    policy = tmp_path / "policy.json"
    write(
        policy,
        {"image": "pinned", "config": "configuration", "seconds": {"postflight": 1}},
    )
    monkeypatch.setattr(ci, "POLICY", policy)
    monkeypatch.setenv("GITHUB_RUN_ID", "1")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")

    def command(argv, seconds=30):
        if argv[1] == "create":
            Path(argv[argv.index("--cidfile") + 1]).write_text(CID)
            return {"returncode": 0, "stdout": CID, "stderr": ""}
        return {
            "returncode": 0,
            "stdout": json.dumps(
                [
                    {
                        "Image": "configuration",
                        "State": {"Running": True, "OOMKilled": False, "ExitCode": 0},
                    }
                ]
            ),
            "stderr": "",
        }

    monkeypatch.setattr(ci, "command", command)
    monkeypatch.setattr(
        ci, "cleanup", lambda cid: {"cid": cid, "passed": True, "absent": True}
    )
    monkeypatch.setattr(
        ci.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.TimeoutExpired(a[0], 1)),
    )
    result = ci.container("postflight", tmp_path)
    assert result["status"] == "failed" and result["attach_exit"] == 124
    assert result["cleanup"]["cid"] == CID
    assert (
        json.loads((tmp_path / "output/postflight-container.json").read_text())
        == result
    )


def test_typed_admission_refuses_legacy_before_native_checks(tmp_path):
    write(tmp_path / "admission.json", {"kind": "local-g24", "leg": "B-accuracy"})
    with pytest.raises(ValueError, match="admission type"):
        ci.verify_admission(tmp_path)


def test_missing_client_fails_before_stock_install_path(tmp_path, monkeypatch):
    current = importlib.util.spec_from_file_location(
        "k31_phase0_assets", ROOT / ".github/scripts/k31_phase0_assets.py"
    )
    assets = importlib.util.module_from_spec(current)
    current.loader.exec_module(assets)
    monkeypatch.setitem(sys.modules, "k31_phase0_assets", assets)
    monkeypatch.setattr(ci.os, "environ", {"PATH": "empty"})
    monkeypatch.setattr(ci.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="refuse install"):
        ci.entry(tmp_path)


def test_real_checker_patch_preserves_full_evaluation(tmp_path):
    for name in ("check_results.py", "verify_inputs.py"):
        source = subprocess.check_output(
            [
                "git",
                "show",
                f"3d7a07a470555840eb686f0b8f0375efc07d58fa:seed/3577/{name}",
            ],
            cwd=ROOT,
        )
        (tmp_path / name).write_bytes(source)
    old = (tmp_path / "check_results.py").read_text()
    subprocess.run(
        ["git", "apply", str(ROOT / ".github/validation/3577-ci-binding.patch")],
        cwd=tmp_path,
        check=True,
    )
    new = (tmp_path / "check_results.py").read_text()
    assert (
        old[old.index('assert read("verified-inputs.json")') :]
        == new[new.index('assert read("verified-inputs.json")') :]
    )
    evidence = tmp_path / "B-accuracy"
    evidence.mkdir()
    state = {"Running": False, "OOMKilled": False, "ExitCode": 0}
    cleanup = {
        "cid": CID,
        "passed": True,
        "absent": True,
        "remove": {"returncode": 0, "argv": ["docker", "rm", "--force", CID]},
        "inventory": {"returncode": 0, "stdout": ""},
    }
    for name, value in {
        "entry-exit.json": {"entry_exit": 0},
        "lm-eval-exit.json": {"exit_code": 0},
        "verified-inputs.json": {"source": "fixed"},
        "postflight.json": {"passed": True},
        "compiled-trace-request.json": {"requested": 0},
        "compiled-trace-outcome.json": {"status": "not_requested"},
        "container-lifecycle.json": dict(
            kind="ci-container-lifecycle-v1",
            **{
                k: {
                    "kind": k,
                    "status": "passed",
                    "attach_exit": 0,
                    "state": state,
                    "cleanup": cleanup,
                    "final_inspect": {"stdout": json.dumps([{"Id": CID}])},
                }
                for k in ("model", "postflight")
            },
        ),
    }.items():
        write(evidence / name, value)
    results = evidence / "accuracy_test_results"
    results.mkdir()
    data = {
        "n-samples": {"gsm8k": {"effective": 1319}},
        "n-shot": {"gsm8k": 3},
        "config": {"limit": None},
        "results": {"gsm8k": {"exact_match,flexible-extract": 0.96}},
        "atom_ci_metadata": {"mtp_acceptance_rate": 25},
    }
    write(results / "result.json", data)
    samples = results / "samples_gsm8k.jsonl"
    samples.write_text(
        "\n".join(
            json.dumps(
                {
                    "doc_id": i,
                    "doc": {"question": "question"},
                    "arguments": ["question"],
                    "resps": [["answer"]],
                }
            )
            for i in range(1319)
        )
    )
    run = lambda: subprocess.run(
        [sys.executable, str(tmp_path / "check_results.py"), str(evidence)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run().returncode == 0
    legacy = tmp_path / "legacy_checker.py"
    legacy.write_text(old)
    red = subprocess.run(
        [sys.executable, str(legacy), str(evidence)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert red.returncode != 0 and "supervisor-exit.json" in red.stderr
    data["results"]["gsm8k"]["exact_match,flexible-extract"] = 0.95
    write(results / "result.json", data)
    assert run().returncode != 0
    data["results"]["gsm8k"]["exact_match,flexible-extract"] = 0.96
    write(results / "result.json", data)
    samples.write_text(samples.read_text() + "\n" + samples.read_text().splitlines()[0])
    assert run().returncode != 0


def test_workflow_selects_one_case_and_retains_failure_artifacts():
    workflow = yaml.load(
        (ROOT / ".github/workflows/atom-test.yaml").read_text(), Loader=yaml.BaseLoader
    )
    job = workflow["jobs"]["k31-b-accuracy"]
    assert "inputs.k31_b_accuracy" in job["if"]
    assert job["steps"][0]["if"] == "${{ inputs.k31_phase0 }}"
    assert "!inputs.k31_b_accuracy" in workflow["concurrency"]["cancel-in-progress"]
    assert job["steps"][-2]["if"] == job["steps"][-1]["if"] == "always()"
    assert "work" not in job["steps"][-1]["with"].get("exclude", "")

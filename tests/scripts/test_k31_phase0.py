import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
BASE = "cccc96a039c0936a0dfe2516245fea5e3305c346"


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f".github/scripts/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load("k31_phase0_inputs")
assets = load("k31_phase0_assets")


@pytest.fixture(autouse=True)
def posix_ctime(monkeypatch):
    if os.name == "nt":
        # Windows stat uses creation time, fstat uses change time; Linux agrees.
        original = assets.identity
        monkeypatch.setattr(
            assets, "identity", lambda *args: dict(original(*args), ctime_ns=0)
        )


def test_transfer_handoff_and_tampering(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "request.json").write_bytes(b'{"unchanged":true}\r\n')
    spec = {
        "files": {"request.json": inputs.digest((seed / "request.json").read_bytes())},
    }
    policy = tmp_path / "spec.json"
    policy.write_text(json.dumps(spec))
    out = tmp_path / "out"
    digest = inputs.transfer(seed, policy, out, BASE, BASE)
    inputs.verify(out / "seed", policy, out / "input-manifest.json", digest)
    assert (out / "seed/request.json").read_bytes() == (
        seed / "request.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="immutable SHA"):
        inputs.transfer(seed, policy, tmp_path / "bad", "main", BASE)
    (out / "seed/request.json").write_text("changed")
    with pytest.raises(ValueError, match="seed hash"):
        inputs.verify(out / "seed", policy, out / "input-manifest.json", digest)
    (seed / "extra").touch()
    with pytest.raises(ValueError, match="member set"):
        inputs.checked_files(seed, spec)
    with pytest.raises(ValueError, match="unsafe member"):
        inputs.checked_files(seed, {"files": {"../outside": "unused"}})


def model(tmp_path, repo):
    root = tmp_path / repo
    root.mkdir(parents=True)
    (root / ".hf-revision").write_text("component")
    (root / ".atom_download_complete").touch()
    weight = root / "model.safetensors"
    weight.write_bytes(b"fixed weight")
    checksum = inputs.digest(weight.read_bytes())
    metadata = root / ".cache/huggingface/download/model.safetensors.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"origin\n{checksum}\n1.0\n")
    view = {
        "repo": repo,
        "expected_component_revision": "component",
        "original_local_revision": "origin",
        "files": [
            {
                "name": weight.name,
                "source": {"bytes": 12},
                "expected_lfs_sha256": checksum,
            }
        ],
    }
    return view, weight, metadata


def test_target_never_reads_payload_and_draft_hashes(tmp_path, monkeypatch):
    view, target, metadata = model(tmp_path, "moonshotai/target")
    real = Path.open

    def guarded(path, *args, **kwargs):
        assert path != target, "target payload must not be opened"
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = assets.check_model(view, tmp_path)
    assert result["payload_bytes_read"] == 0
    assert result["files"][0]["measured_payload_sha256"] is None
    metadata.write_text("origin\nwrong\n1.0\n")
    with pytest.raises(ValueError, match="LFS identity"):
        assets.check_model(view, tmp_path)
    draft, path, _ = model(tmp_path, "Inferact/draft")
    assert assets.check_model(draft, tmp_path)["payload_bytes_read"] == 12
    path.write_bytes(b"wrong weight")
    with pytest.raises(ValueError, match="draft payload"):
        assets.check_model(draft, tmp_path)


def test_checksums_reject_changed_files_and_deadlines(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="deadline"):
        assets.measured_hash(path, seconds=-1)
    original = assets.identity
    calls = []

    def changed(*args):
        value = original(*args)
        calls.append(value)
        return dict(value, bytes=99) if len(calls) > 1 else value

    monkeypatch.setattr(assets, "identity", changed)
    with pytest.raises(ValueError, match="identity differs"):
        assets.measured_hash(path)


def test_workflow_preserves_defaults_and_excludes_all_old_jobs():
    file = ".github/workflows/atom-test.yaml"
    old = yaml.load(
        subprocess.check_output(["git", "show", f"{BASE}:{file}"], cwd=ROOT),
        Loader=yaml.BaseLoader,
    )
    new = yaml.load((ROOT / file).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    unwrap = lambda text: text.removeprefix("${{").removesuffix("}}").strip()
    for name, before in old["jobs"].items():
        after = dict(new["jobs"][name])
        guard = unwrap(after.pop("if"))
        prior = dict(before)
        condition = prior.pop("if", None)
        assert guard == (
            f"!inputs.k31_phase0 && ({unwrap(condition)})"
            if condition
            else "!inputs.k31_phase0"
        )
        assert after == prior
    assert new["on"]["workflow_dispatch"]["inputs"]["k31_phase0"]["default"] == "false"
    command = str(new["jobs"]["k31-asset-preflight"])
    assert "--device" not in command and "--network none --ipc private" in command
    assert (
        "src=/models,dst=/models,readonly" in command and "pip install" not in command
    )

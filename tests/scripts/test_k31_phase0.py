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


def load(name, revision=None):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f".github/scripts/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if revision:
        source = subprocess.check_output(
            ["git", "show", f"{revision}:.github/scripts/{name}.py"], cwd=ROOT
        )
        exec(  # noqa: S102 -- pinned regression source
            compile(source, revision, "exec"), module.__dict__
        )
    else:
        spec.loader.exec_module(module)
    return module


inputs = load("k31_phase0_inputs")
assets = load("k31_phase0_assets")
before_fix = load("k31_phase0_assets", "fef4ca63de1cdc14f67292cd10625d97696b29dd")


@pytest.fixture(autouse=True)
def posix_ctime(monkeypatch):
    if os.name == "nt":
        # Windows stat uses creation time, fstat uses change time; Linux agrees.
        for module in (assets, before_fix):
            original = module.identity
            monkeypatch.setattr(
                module,
                "identity",
                lambda *args, fn=original: dict(fn(*args), ctime_ns=0),
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


def model(tmp_path, role):
    repo = assets.MODEL_DIRS[role]
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
        "role": role,
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
    view, target, metadata = model(tmp_path, "target")
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
    draft, path, _ = model(tmp_path, "draft")
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


@pytest.mark.parametrize("alias", [False, True])
def test_marker_is_refused_before_read(tmp_path, monkeypatch, alias):
    view, target, _ = model(tmp_path, "target")
    marker = target.parent / ".hf-revision"
    marker.unlink()
    os.link(target, marker) if alias else marker.write_bytes(b"x" * 129)
    opened, real = [], Path.open

    def watch(path, *args, **kwargs):
        opened.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watch)
    with pytest.raises(ValueError):
        before_fix.check_model(view, tmp_path)
    assert marker in opened
    opened.clear()
    with pytest.raises(ValueError, match="alias refused|oversize"):
        assets.check_model(view, tmp_path)
    assert not opened


def test_nonweight_payload_alias_old_reads_new_refuses(tmp_path, monkeypatch):
    view, target, _ = model(tmp_path, "target")
    alias = target.parent / "config.json"
    os.link(target, alias)
    view["files"].insert(
        0,
        {
            "name": alias.name,
            "actual_sha256": inputs.digest(b"fixed weight"),
            "source": {"bytes": 12},
        },
    )
    opened, real = [], Path.open

    def watch(path, *args, **kwargs):
        if path == alias:
            opened.append(path)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", watch)
    before_fix.check_model(view, tmp_path)
    assert opened == [alias]
    opened.clear()
    with pytest.raises(ValueError, match="payload alias refused"):
        assets.check_model(view, tmp_path)
    assert not opened


def test_final_identity_time_is_charged(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"data")
    for module in (before_fix, assets):
        clock, calls, real = [0], [], module.identity

        def delayed(*args, counter=calls, now=clock, original=real):
            counter.append(args)
            if len(counter) == 4:
                now[0] = 91
            return original(*args)

        with monkeypatch.context() as patch:
            patch.setattr(module, "identity", delayed)
            patch.setattr(module.time, "monotonic", lambda now=clock: now[0])
            if module is before_fix:
                assert module.measured_hash(path)["bytes_read"] == 4
                assert clock[0] == 91
            else:
                with pytest.raises(ValueError, match="checksum deadline"):
                    module.measured_hash(path)


def test_preview_origin_uses_actual_ci_draft_path(tmp_path):
    view, _, _ = model(tmp_path, "draft")
    view["repo"] = "Inferact/Kimi-K3.1-Pro-DSpark-preview"
    with pytest.raises(FileNotFoundError):
        before_fix.check_model(view, tmp_path)
    result = assets.check_model(view, tmp_path)
    assert result["repo"] == view["repo"] and result["payload_bytes_read"] == 12


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

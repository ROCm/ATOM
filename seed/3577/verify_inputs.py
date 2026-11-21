"""Admission checks for the already-staged, isolated executor. Never installs/downloads."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path("/app/atom-3577")
source = root / "source"
evidence = Path(os.environ["EVIDENCE"])
arm = "A" if os.environ["LEG"] == "A-pp2" else "B"
manifest = json.loads((root / "input/source-overlay-manifest.json").read_text())[arm]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() == manifest["source"]
expected = {entry["path"] for entry in manifest["files"]}
changed = set(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], text=True).splitlines())
assert changed == expected, (changed, expected)
assert not subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard"], text=True).strip()
for entry in manifest["files"]:
    assert sha(source / entry["path"]) == entry["overlay_sha256"], entry["path"]

runtime_pins = json.loads((root / "input/runtime-pins.json").read_text())
versions = {name: pin["version"] for name, pin in runtime_pins["packages"].items()}
for name, pin in runtime_pins["packages"].items():
    distribution = metadata.distribution(name)
    assert distribution.version == pin["version"], (name, distribution.version, pin["version"])
    for filename, digest in pin["metadata_sha256"].items():
        matches = [f for f in distribution.files if str(f).endswith(".dist-info/" + filename)]
        assert len(matches) == 1, (name, filename)
        assert sha(distribution.locate_file(matches[0])) == digest, (name, filename)
assert sys.version.split()[0] == runtime_pins["python"]["version"]
assert sha(Path(sys.executable).resolve()) == runtime_pins["python"]["binary_sha256"]
for variable in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    assert os.environ[variable] == "1"
admission = json.loads((root / "admission.json").read_text())
assert admission["leg"] == os.environ["LEG"]
assert admission["image_digest"] == runtime_pins["image_digest"]
assert admission["common_core_sha256"] == runtime_pins["common_core_sha256"]
bindings = json.loads((root / "input/preuse-bindings.json").read_text())
for name, binding in bindings.items():
    assert binding is not None, f"{name} awaits the real executor receipt and reviewer binding"
    assert admission[name + "_sha256"] == binding["sha256"]
    assert sha(binding["path"]) == binding["sha256"]
assert admission["private_writable_layer"] and admission["no_shared_cache_bind"]
for namespace in ("pid", "ipc", "mnt"):
    observed = os.readlink(f"/proc/self/ns/{namespace}")
    assert observed == admission["container_namespaces"][namespace]
    assert observed != admission["host_namespaces"][namespace]
assert str(Path.home() / ".cache/atom") == admission["atom_cache_path"]
assert str((Path.home() / ".cache/atom").resolve()).startswith(str(Path.home()) + "/")
assert admission["cache_path_is_private_container_layer"] is True
assert admission["model_mounts_read_only"] is True
assert admission["devices"] == list(range(8))
import atom
assert Path(atom.__file__).resolve().is_relative_to(source)
import aiter
aiter_root = subprocess.check_output(
    ["git", "-C", str(Path(aiter.__file__).resolve().parent), "rev-parse", "--show-toplevel"],
    text=True).strip()
assert subprocess.check_output(["git", "-C", aiter_root, "rev-parse", "HEAD"],
                               text=True).strip() == runtime_pins["aiter_git"]
assert not subprocess.check_output(
    ["git", "-C", aiter_root, "status", "--porcelain", "--untracked-files=no"], text=True).strip()
from aiter.dist import parallel_state
assert sha(parallel_state.__file__) == runtime_pins["aiter_parallel_state_sha256"]
report = dict(qualification="local-g24-ea74-B9", versions=versions, source=manifest["source"],
              atom_import=str(Path(atom.__file__).resolve()), admission_sha256=sha(root / "admission.json"))

if os.environ["LEG"] == "B-accuracy":
    import datasets
    from lm_eval.tasks import TaskManager, get_task_dict
    raw = root / "data/gsm8k/main"
    hashes = {"test-00000-of-00001.parquet": "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59",
              "train-00000-of-00001.parquet": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d"}
    for filename, digest in hashes.items():
        assert sha(raw / filename) == digest, filename
    task = get_task_dict(["gsm8k"], task_manager=TaskManager())["gsm8k"]
    snapshots = datasets.load_dataset("parquet", data_files={
        split: str(raw / f"{split}-00000-of-00001.parquet") for split in ("train", "test")})
    observed = {}
    for split, count in (("train", 7473), ("test", 1319)):
        actual, pinned = task.dataset[split], snapshots[split]
        rows = lambda ds: [(r["question"], r["answer"]) for r in ds]
        assert len(actual) == len(pinned) == count
        assert rows(actual) == rows(pinned), split
        digest = hashlib.sha256(json.dumps(rows(actual), ensure_ascii=False).encode()).hexdigest()
        observed[split] = dict(rows=count, content_sha256=digest, fingerprint=actual._fingerprint)
    import lm_eval
    task_yaml = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml"
    assert sha(task_yaml) == runtime_pins["gsm8k_task_yaml_sha256"]
    report["gsm8k"] = dict(revision="740312add88f781978c0658806c59bc2815b9866",
                          raw_sha256=hashes, splits=observed, task_yaml_sha256=sha(task_yaml))
(evidence / "verified-inputs.json").write_text(json.dumps(report, indent=2) + "\n")

"""Stage a hash-pinned producer TE wheel, then reuse the timeout workload."""

import hashlib
import json
import os
import shutil
from pathlib import Path

from pd_timeout_probe import main as run_probe
from pd_timeout_probe import validate_probe


def stage_context(repo, contract):
    validate_probe(contract)
    spec = contract["mooncake_rebuild"]
    source = repo / spec["context"]
    manifest = json.loads((source / "manifest.json").read_text())
    arm = spec["arm"]
    if arm not in ("base", "patched"):
        raise ValueError("Expected base or patched producer")
    assert manifest["base_image"] == contract["image"]
    assert "container" not in contract["roles"]["decode"]
    selected = manifest["arms"][arm]
    wheel = Path(spec["artifact_root"]) / arm / manifest["wheel_name"]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != selected["wheel_sha256"]:
        raise ValueError("Rebuilt wheel hash mismatch")
    probe = repo / contract["timeout_probe"]["context"]
    probe_manifest = json.loads((probe / "manifest.json").read_text())
    context = repo / f".sa-srt-runtime/te5840-{arm}-worker"
    context.mkdir()
    for name in ("prepare_worker.py", "atomesh_timeout_probe.py"):
        assert (
            hashlib.sha256((probe / name).read_bytes()).hexdigest()
            == probe_manifest["files"][name]
        )
        shutil.copyfile(probe / name, context / name)
    verify = source / "verify_worker.py"
    assert hashlib.sha256(verify.read_bytes()).hexdigest() == manifest["verify_sha256"]
    shutil.copyfile(verify, context / "verify_worker.py")
    shutil.copyfile(wheel, context / wheel.name)
    runtime_manifest = {
        "base_image": contract["image"],
        "engine_sha256": selected["engine_sha256"],
        "instrumented_connector_sha256": probe_manifest[
            "instrumented_connector_sha256"
        ],
        "base_module_sha256": manifest["base_module_sha256"],
        "arm": arm,
        "build": manifest,
    }
    (context / "expected.json").write_text(
        json.dumps(runtime_manifest, indent=2) + "\n"
    )
    (context / "Dockerfile").write_text(
        "FROM " + contract["image"] + "\n"
        "COPY prepare_worker.py /opt/atomesh-timeout/prepare_worker.py\n"
        "COPY atomesh_timeout_probe.py /usr/local/lib/python3.12/dist-packages/atomesh_timeout_probe.py\n"
        "RUN python3 /opt/atomesh-timeout/prepare_worker.py\n"
        f"COPY {wheel.name} /opt/atomesh-timeout/{wheel.name}\n"
        "RUN python3 -m pip install --no-index --no-deps --force-reinstall "
        f"/opt/atomesh-timeout/{wheel.name}\n"
        "COPY verify_worker.py /opt/atomesh-timeout/prepare_worker.py\n"
        "COPY expected.json /opt/atomesh-timeout/expected.json\n"
        "RUN python3 /opt/atomesh-timeout/prepare_worker.py --verify\n"
    )
    runtime_manifest["files"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in context.iterdir()
        if p.is_file()
    }
    (context / "manifest.json").write_text(
        json.dumps(runtime_manifest, indent=2) + "\n"
    )
    contract["timeout_probe"]["context"] = str(context.relative_to(repo))
    return contract


def main():
    repo = Path(os.environ["REPO_ROOT"])
    contract = json.loads(
        (repo / os.environ["ATOMESH_ENV_ATOMESH_PR5090_CONTRACT"]).read_text()
    )
    contract = stage_context(repo, contract)
    relative = ".sa-srt-runtime/te5840-contract.json"
    (repo / relative).write_text(json.dumps(contract, indent=2) + "\n")
    os.environ["ATOMESH_ENV_ATOMESH_PR5090_CONTRACT"] = relative
    return run_probe()


if __name__ == "__main__":
    raise SystemExit(main())

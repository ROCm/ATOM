"""Run one leased-node PP/DEP A/B: fixed P, fixed router, only D image changes."""

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys


def pair_contract(base, arm, image, manifest):
    if arm not in ("base", "candidate") or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", image
    ):
        raise ValueError("Expected a known arm and a built image reference")
    contract = copy.deepcopy(base)
    contract["case_name"] += "-" + arm
    contract["pair_arm"] = arm
    for role in ("prefill", "decode"):
        key = (
            "candidate_module_sha256"
            if arm == "candidate" and role == "decode"
            else "base_module_sha256"
        )
        contract["roles"][role]["env"]["ATOMESH_EXPECTED_NATIVE_SHA"] = manifest[key]
    if arm == "candidate":
        contract["roles"]["decode"]["container"] = image
    return contract


def main():
    repo, root = Path(os.environ["REPO_ROOT"]), Path(os.environ["RUN_DIR"])
    scripts = repo / ".github/scripts/atomesh"
    base = json.loads(
        (repo / os.environ["ATOMESH_ENV_ATOMESH_PR5090_CONTRACT"]).read_text()
    )
    context = repo / base["paired_worker_image"]["context"]
    manifest = json.loads((context / "manifest.json").read_text())
    assert manifest["base_image"] == base["image"]
    assert (context / "Dockerfile").read_text().splitlines() == [
        "FROM " + base["image"],
        "COPY rocm_native.py " + manifest["module_path"],
    ]
    assert manifest["validation"]["bitwise_equal_cases"] == 8
    assert manifest["validation"]["release_rc"] == 0
    assert (
        hashlib.sha256((context / "rocm_native.py").read_bytes()).hexdigest()
        == manifest["candidate_module_sha256"]
    )
    state = {
        "phase": "worker_build",
        "current_arm": None,
        "arms": {},
        "complete": False,
    }

    def save(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = root / "pair-state.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "pair-state.json")
        print("[mxfp8-pair] " + json.dumps(updates), flush=True)

    def guard(phase):
        subprocess.run(
            [sys.executable, str(scripts / "pd_gpool_guard.py"), phase], check=True
        )

    save()
    try:
        guard("worker-build")
        build = root / "worker-build"
        build.mkdir()
        with (build / "build.log").open("w") as log:
            subprocess.run(
                [
                    "docker",
                    "build",
                    "--network=none",
                    "--iidfile",
                    str(build / "image-id.txt"),
                    str(context),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=900,
                check=True,
            )
        image = (build / "image-id.txt").read_text().strip()
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", image)
        for label, reference in (("base", base["image"]), ("candidate", image)):
            actual = subprocess.check_output(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "--label",
                    "srtctl.job=" + os.environ["SRTCTL_JOB_ID"],
                    "--entrypoint",
                    "sha256sum",
                    reference,
                    manifest["module_path"],
                ],
                text=True,
                timeout=60,
            ).split()[0]
            assert actual == manifest[f"{label}_module_sha256"], (label, actual)
        (build / "runtime.json").write_text(
            json.dumps(
                {
                    "base_image": base["image"],
                    "candidate_image_ref": image,
                    "manifest": manifest,
                    "file_hashes_verified_in_images": True,
                },
                indent=2,
            )
            + "\n"
        )
        save(candidate_image_ref=image)
        for arm in ("base", "candidate"):
            guard("pair-" + arm)
            # The first service must have relinquished its ports before the next arm.
            for port in (6100, 6132, 7200, 8000):
                with socket.socket() as listener:
                    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    listener.bind(("0.0.0.0", port))
            output = root / arm
            (output / "bin").mkdir(parents=True)
            shutil.copyfile(root / "bin/nvcc", output / "bin/nvcc")
            (output / "bin/nvcc").chmod(0o755)
            relative = f".sa-srt-runtime/mxfp8-pair-{arm}.json"
            (repo / relative).write_text(
                json.dumps(pair_contract(base, arm, image, manifest), indent=2) + "\n"
            )
            save(phase="running", current_arm=arm)
            env = dict(
                os.environ,
                RUN_DIR=str(output),
                ATOMESH_ENV_ATOMESH_PR5090_CONTRACT=relative,
            )
            code = subprocess.run(
                ["bash", str(scripts / "pd_srt_functional.sh")], env=env, check=False
            ).returncode
            state["arms"][arm] = {"exit_code": code, "directory": arm}
            save()
            if code:
                raise RuntimeError(f"{arm} workload failed: {code}")
            assert not (output / "srt-cleanup-remaining.txt").read_text().strip()
        save(phase="completed", complete=True)
        return 0
    except BaseException as error:
        save(phase="failed", error=repr(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())

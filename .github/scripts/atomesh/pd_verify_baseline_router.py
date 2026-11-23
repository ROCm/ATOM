"""Verify the pinned vLLM Router image without exposing GPU devices."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    reference = contract["router_images"]["baseline"]
    assert re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", reference)
    assert contract["router_arm"] == "baseline"
    args.output.mkdir(parents=True, exist_ok=True)
    inspect = subprocess.run(
        ["docker", "image", "inspect", reference],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode:
        with (args.output / "image-pull.log").open("w") as log:
            subprocess.run(
                ["docker", "pull", reference],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
                check=True,
            )
        inspect = subprocess.run(
            ["docker", "image", "inspect", reference],
            capture_output=True,
            text=True,
            check=True,
        )
    image = json.loads(inspect.stdout)[0]
    digest = reference.rsplit("@", 1)[1]
    assert any(ref.endswith("@" + digest) for ref in image["RepoDigests"])
    command = (
        'set -eu; binary="$(command -v vllm-router)"; '
        'sha256sum "$binary"; vllm-router --version; vllm-router --help'
    )
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--label",
            f"srtctl.job={args.job_id}",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--entrypoint",
            "bash",
            image["Id"],
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    output = result.stdout + result.stderr
    (args.output / "runtime-check.txt").write_text(output)
    match = re.search(r"^([0-9a-f]{64})\s+(/\S+)$", output, re.MULTILINE)
    assert match is not None
    version = contract["baseline_router_version"]
    assert re.search(r"^vllm-router " + re.escape(version) + r"$", output, re.MULTILINE)
    assert "--kv-connector" in output and "mooncake" in output
    manifest = {
        "kind": "baseline",
        "base_image": reference,
        "image_id": image["Id"],
        "repo_digests": image["RepoDigests"],
        "version": version,
        "binary_sha256": match[1],
        "binary_path": match[2],
        "runtime_verified": True,
    }
    (args.output / "runtime.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[router-baseline] verified {image['Id']} version={version}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Resolve one optional AITER wheel for every point in a benchmark run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import zipfile
from email.parser import Parser
from html import escape
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit, urlunsplit


def validate_request(value, aiter_commit=""):
    """Accept latest, an immutable ROCm/aiter artifact ID, or a wheel URL."""
    value = value.strip()
    if not value:
        return ""
    if aiter_commit.strip():
        raise ValueError("Use only one of aiter_wheel and aiter_commit")
    if value == "latest" or re.fullmatch(r"artifact:[1-9][0-9]*", value):
        return value
    parts = urlsplit(value)
    name = unquote(parts.path.rsplit("/", 1)[-1])
    if (
        parts.scheme != "https"
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.fragment
        or any(c.isspace() for c in value)
        or not re.fullmatch(r"amd_aiter-[A-Za-z0-9_.+\-]+\.whl", name)
    ):
        raise ValueError(
            "aiter_wheel must be latest, artifact:<ROCm/aiter artifact ID>, "
            "or an HTTPS amd_aiter .whl URL"
        )
    path = "/".join(
        quote(unquote(segment), safe="") for segment in parts.path.split("/")
    )
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def wheel_metadata(path):
    """Read identity without importing or executing the downloaded wheel."""
    with zipfile.ZipFile(path) as archive:
        candidates = [
            info
            for info in archive.infolist()
            if info.filename.endswith(".dist-info/METADATA")
        ]
        if len(candidates) != 1 or candidates[0].file_size > 1024 * 1024:
            raise ValueError("Expected one bounded wheel METADATA file")
        metadata = Parser().parsestr(archive.read(candidates[0]).decode("utf-8"))
    if metadata.get("Name", "").lower().replace("_", "-") != "amd-aiter":
        raise ValueError("Selected wheel is not amd-aiter")
    version = metadata.get("Version")
    if not version or not re.fullmatch(r"[A-Za-z0-9_.+!\-]+", version):
        raise ValueError("Selected wheel has no valid package version")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return version, digest.hexdigest()


def prepare_wheel(requested, output_dir, config_dir, python_tag="cp312"):
    """Download once on CPU, then pin replay inputs and save the wheel checksum."""
    selector = validate_request(requested)
    if not selector:
        return None
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("amd_aiter*.whl"):
        old.unlink()
    resolved = selector
    if selector.startswith("https://"):
        name = unquote(urlsplit(selector).path.rsplit("/", 1)[-1])
        subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--show-error",
                "--location",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "2",
                "--connect-timeout",
                "30",
                "--max-time",
                "540",
                "--output",
                str(output / name),
                selector,
            ],
            check=True,
        )
    else:
        with tempfile.TemporaryDirectory() as temporary:
            result_file = Path(temporary) / "outputs"
            env = {
                **os.environ,
                "ATOM_PYTHON_TAG": python_tag,
                "AITER_WHEEL_OUTPUT_DIR": str(output),
                "GITHUB_OUTPUT": str(result_file),
                "AITER_WHEEL_DOWNLOAD_MODE": "resolve",
            }
            if selector.startswith("artifact:"):
                env.update(
                    AITER_WHEEL_DOWNLOAD_MODE="workflow_artifact",
                    AITER_WORKFLOW_ARTIFACT_REPO="ROCm/aiter",
                    AITER_WORKFLOW_ARTIFACT_ID=selector.split(":", 1)[1],
                )
            subprocess.run(
                ["bash", str(Path(__file__).with_name("download_aiter_wheel.sh"))],
                check=True,
                env=env,
                cwd=temporary,
            )
            values = dict(
                line.split("=", 1) for line in result_file.read_text().splitlines()
            )
            resolved = values.get("aiter_wheel_url") or (
                "artifact:"
                + (
                    values.get("aiter_artifact_id")
                    or values.get("aiter_workflow_artifact_id", "")
                )
            )
            resolved = validate_request(resolved)
    wheels = list(output.glob("amd_aiter*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one downloaded amd-aiter wheel")
    wheel = wheels[0]
    version, digest = wheel_metadata(wheel)
    record = {
        "requested": requested,
        "pinned": resolved,
        "filename": wheel.name,
        "version": version,
        "sha256": digest,
    }
    (output / "selection.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "SHA256SUMS").write_text(f"{digest}  {wheel.name}\n")
    config = Path(config_dir)
    for filename, field, value in (
        ("run-config.json", "aiter_wheel_resolution", record),
        ("dispatch-inputs.json", "aiter_wheel", resolved),
    ):
        path = config / filename
        content = json.loads(path.read_text())
        content[field] = value
        path.write_text(json.dumps(content, indent=2) + "\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
            stream.write(
                "\n### AITER wheel selected for every point\n\n<pre>"
                + escape(json.dumps(record, indent=2))
                + "</pre>\n"
            )
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="aiter-whl")
    parser.add_argument("--config-dir", default="agentic-run-config")
    args = parser.parse_args()
    prepare_wheel(
        os.environ.get("AITER_WHEEL_REQUEST", ""), args.output_dir, args.config_dir
    )


if __name__ == "__main__":
    main()

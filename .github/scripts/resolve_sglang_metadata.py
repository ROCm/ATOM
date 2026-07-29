#!/usr/bin/env python3
"""Export SGLang version metadata from the Docker release workflow."""

from __future__ import annotations

import os
import re
from pathlib import Path

METADATA_SOURCE = Path(".github/workflows/docker-release.yaml")


def read_metadata(name: str, text: str) -> str:
    match = re.search(rf'^\s*{name}:\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"Failed to read {name} from {METADATA_SOURCE}")
    return match.group(1)


def main() -> None:
    text = METADATA_SOURCE.read_text(encoding="utf-8")
    sglang_ref = read_metadata("SGLANG_REF", text)
    sglang_version = read_metadata("SGLANG_VERSION", text)

    github_env = os.environ.get("GITHUB_ENV")
    if not github_env:
        raise SystemExit("GITHUB_ENV is not set")

    with Path(github_env).open("a", encoding="utf-8") as env_file:
        env_file.write(f"SGLANG_REF={sglang_ref}\n")
        env_file.write(f"SGLANG_VERSION={sglang_version}\n")

    print(f"Resolved SGLang metadata: ref={sglang_ref}, version={sglang_version}")


if __name__ == "__main__":
    main()

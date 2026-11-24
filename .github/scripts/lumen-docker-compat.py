#!/usr/bin/env python3
"""Delegate Docker CLI calls to Docker or Podman for Lumen CI launchers."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def strip_source_mounts(args: list[str]) -> list[str]:
    """Use source baked into the CI image instead of shadowing its extensions."""
    if os.environ.get("LUMEN_CI_USE_IMAGE_SOURCE") != "1":
        return list(args)
    if not args or args[0] != "run":
        return list(args)

    translated: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        spec: str | None = None
        consumed = 1
        if argument in {"-v", "--volume"} and index + 1 < len(args):
            spec = args[index + 1]
            consumed = 2
        elif argument.startswith("--volume="):
            spec = argument.split("=", 1)[1]

        if spec is not None:
            fields = spec.rsplit(":", 2)
            destination = fields[-2] if fields[-1] in {"ro", "rw"} else fields[-1]
            if destination in {"/workspace/ATOM", "/workspace/Lumen"}:
                index += consumed
                continue

        translated.append(argument)
        if consumed == 2:
            translated.append(args[index + 1])
        index += consumed
    return translated


def translate_args(engine_path: str, args: list[str]) -> list[str]:
    """Translate Docker group flags when a launcher is delegated to Podman."""
    if Path(engine_path).name.lower() not in {"podman", "podman.exe"}:
        return list(args)
    if not args or args[0] != "run":
        return list(args)

    translated: list[str] = []
    keep_groups_added = False
    index = 0
    while index < len(args):
        argument = args[index]
        group: str | None = None
        consumed = 1
        if argument == "--group-add" and index + 1 < len(args):
            group = args[index + 1]
            consumed = 2
        elif argument.startswith("--group-add="):
            group = argument.split("=", 1)[1]

        if group in {"video", "render"}:
            if not keep_groups_added:
                translated.extend(("--group-add", "keep-groups"))
                keep_groups_added = True
            index += consumed
            continue

        translated.append(argument)
        if consumed == 2:
            translated.append(args[index + 1])
        index += consumed

    return translated


def main() -> None:
    engine_path = os.environ.get("CONTAINER_ENGINE_PATH")
    if not engine_path:
        raise SystemExit("CONTAINER_ENGINE_PATH is required")
    arguments = translate_args(engine_path, strip_source_mounts(sys.argv[1:]))
    os.execv(engine_path, [engine_path, *arguments])


if __name__ == "__main__":
    main()

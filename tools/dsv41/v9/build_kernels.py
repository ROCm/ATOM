#!/usr/bin/env python3
"""Rebuild the V9 HIP backends into an external directory; qualification is separate."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from prepare import HERE, reconstruct, sha


def build(aiter, output, hipcc):
    lock = json.loads((HERE / "LOCK.json").read_text())
    for name, expected in lock["native_header_sha256"].items():
        if sha(aiter / name) != expected:
            raise ValueError(f"Native header differs from measured dependency: {name}")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    shutil.copytree(aiter / "csrc/include", source / "native_include")
    reconstruct(
        json.loads((HERE / "kernels/ordered_header.json").read_text()), aiter, source
    )
    for name in ("ordered_rs.cu", "push_join.cu"):
        shutil.copy2(HERE / "kernels" / name, source / name)
    commands = []
    for name in ("ordered_rs", "push_join"):
        command = [
            hipcc,
            "-std=c++17",
            "--offload-arch=gfx950",
            "-O3",
            "-fPIC",
            "-shared",
            "-DUSE_ROCM",
            "-DENABLE_CK=0",
            "-I" + str(source / "native_include"),
            str(source / (name + ".cu")),
            "-o",
            str(output / (name + ".so")),
        ]
        with (output / (name + ".log")).open("x") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, check=False
            )
        commands.append({"command": command, "returncode": result.returncode})
        (output / "BUILD.json").write_text(
            json.dumps(
                {
                    "commands": commands,
                    "source_sha256": {
                        str(path.relative_to(source)): sha(path)
                        for path in source.rglob("*")
                        if path.is_file()
                    },
                    "qualification": "Recompiled binaries need numerical/graph checks before service; not inherited performance",
                },
                indent=2,
            )
            + "\n"
        )
        if result.returncode:
            raise RuntimeError(f"{name} compilation failed; retain the build log")
    return {
        "output": str(output),
        "binary_sha256": {
            name: sha(output / name) for name in ("ordered_rs.so", "push_join.so")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    args = parser.parse_args()
    print(json.dumps(build(args.aiter.resolve(), args.output.resolve(), args.hipcc)))

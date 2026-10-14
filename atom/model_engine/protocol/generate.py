# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate the checked-in Python binding of the engine wire protocol.

    python atom/model_engine/protocol/generate.py          # rewrite engine_v1_pb2.py
    python atom/model_engine/protocol/generate.py --check  # fail if it is stale

Run it after editing engine_v1.proto and commit both files. `--check` needs no
compiler: the binding records the sha256 of the .proto it was generated from.

Compiles with `grpc_tools.protoc` when it is importable, with a `protoc` on
PATH otherwise, and refuses code that carries a protobuf runtime version gate
(grpcio-tools >= 1.65, protoc >= 26): such a binding would not import under the
older protobuf runtimes a plugin host may pin. protoc 3.21 / grpcio-tools < 1.65
produce code that imports under any protobuf from 3.20 up.

Imports nothing from atom, so it runs from a bare checkout.
"""

import argparse
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTO = Path("atom/model_engine/protocol/engine_v1.proto")  # from ROOT
BINDING = PROTO.with_name("engine_v1_pb2.py")
_STAMP = re.compile(r'^PROTO_SHA256 = "([0-9a-f]{64})"$', re.MULTILINE)


def proto_sha256(root: Path = ROOT) -> str:
    return hashlib.sha256((root / PROTO).read_bytes()).hexdigest()


def _compile(root: Path, out: Path) -> None:
    args = [f"-I{root}", f"--python_out={out}", str(root / PROTO)]
    try:
        from grpc_tools import protoc
    except ImportError:
        protoc = None
    if protoc is not None:
        if protoc.main(["protoc", *args]) != 0:
            raise RuntimeError(f"grpc_tools.protoc failed on {PROTO}")
        return
    if shutil.which("protoc") is None:
        raise RuntimeError(
            f"cannot compile {PROTO}: install grpcio-tools<1.65 or put protoc on PATH"
        )
    subprocess.run(["protoc", *args], check=True)


def generate(root: Path = ROOT) -> Path:
    """Write the binding next to the .proto; return its path."""
    with tempfile.TemporaryDirectory() as out:
        _compile(root, Path(out))
        code = (Path(out) / BINDING).read_text()
    if "ValidateProtobufRuntimeVersion" in code:
        raise RuntimeError(
            "this compiler emits a protobuf runtime version gate; "
            "use protoc 3.21-25 or grpcio-tools<1.65"
        )
    # What `protocol.binding` and `--check` compare against the .proto.
    code += f'\nPROTO_SHA256 = "{proto_sha256(root)}"\n'
    target = root / BINDING
    target.write_text(code)
    # A .proto edit that leaves the descriptor alone (a comment) regenerates a
    # file of the same size; within the same second its old .pyc still passes
    # Python's mtime+size check and would be loaded instead.
    Path(importlib.util.cache_from_source(str(target))).unlink(missing_ok=True)
    return target


def check(root: Path = ROOT) -> bool:
    """Whether the checked-in binding was generated from the current .proto."""
    binding = root / BINDING
    stamp = _STAMP.search(binding.read_text()) if binding.exists() else None
    return stamp is not None and stamp.group(1) == proto_sha256(root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        if check():
            return 0
        print(
            f"{BINDING} is stale or missing; run {Path(__file__).relative_to(ROOT)}",
            file=sys.stderr,
        )
        return 1
    print(generate())
    return 0


if __name__ == "__main__":
    sys.exit(main())

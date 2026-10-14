# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate the Python binding of the engine wire protocol.

    python atom/model_engine/protocol/generate.py

`pip install` runs this too (setup.py, by path, before the package is
importable -- so this file imports nothing from it), and it only needs running
by hand after editing engine_v1.proto in an editable checkout. The binding is
not checked in.

Compiles with `grpc_tools.protoc` when it is importable -- the isolated build
environment pins a grpcio-tools whose generated code carries no runtime version
gate, so a wheel imports under any protobuf from 3.20 up -- and with a `protoc`
on PATH otherwise.
"""

import hashlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROTO = Path("atom/model_engine/protocol/engine_v1.proto")  # from the repo root
BINDING = PROTO.with_name("engine_v1_pb2.py")


def proto_sha256(proto: Path) -> str:
    return hashlib.sha256(proto.read_bytes()).hexdigest()


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
            f"cannot compile {PROTO}: install grpcio-tools or put protoc on PATH"
        )
    subprocess.run(["protoc", *args], check=True)


def generate(root: Path) -> Path:
    """Write the binding next to the .proto; return its path."""
    root = Path(root).resolve()
    with tempfile.TemporaryDirectory() as out:
        _compile(root, Path(out))
        code = (Path(out) / BINDING).read_text()
    # What `protocol.binding` compares against the .proto on import, so an
    # edit to the .proto without regenerating fails loudly, not on the wire.
    code += f'\nPROTO_SHA256 = "{proto_sha256(root / PROTO)}"\n'
    target = root / BINDING
    target.write_text(code)
    # A .proto edit that leaves the descriptor alone (a comment) regenerates a
    # file of the same size; within the same second its old .pyc still passes
    # Python's mtime+size check and would be loaded instead.
    Path(importlib.util.cache_from_source(str(target))).unlink(missing_ok=True)
    return target


if __name__ == "__main__":
    print(generate(Path(__file__).resolve().parents[3]))
    sys.exit(0)

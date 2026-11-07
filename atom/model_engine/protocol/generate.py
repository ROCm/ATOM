# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regenerate the checked-in Python binding of the engine wire protocol.

    python atom/model_engine/protocol/generate.py

Run it after editing engine_v1.proto and commit both files.

Compiles with `grpc_tools.protoc` when it is importable and with a `protoc` on
PATH otherwise, and refuses code that carries a protobuf runtime version gate
(grpcio-tools >= 1.65, protoc >= 26): such a binding would not import under the
older protobuf runtimes a plugin host may pin. protoc 3.21 / grpcio-tools < 1.65
produce code that imports under any protobuf from 3.20 up.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTO = Path("atom/model_engine/protocol/engine_v1.proto")  # from ROOT
BINDING = PROTO.with_name("engine_v1_pb2.py")


def _compile(out: Path) -> None:
    args = [f"-I{ROOT}", f"--python_out={out}", str(ROOT / PROTO)]
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


def main() -> int:
    with tempfile.TemporaryDirectory() as out:
        _compile(Path(out))
        code = (Path(out) / BINDING).read_text()
    if "ValidateProtobufRuntimeVersion" in code:
        print(
            "this compiler emits a protobuf runtime version gate; "
            "use protoc 3.21-25 or grpcio-tools<1.65",
            file=sys.stderr,
        )
        return 1
    (ROOT / BINDING).write_text(code)
    print(ROOT / BINDING)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regenerate the checked-in Python bindings of the engine wire protocol.

    python scripts/gen_engine_protocol.py          # rewrite engine_v1_pb2.py
    python scripts/gen_engine_protocol.py --check  # fail if it is stale

The binding is generated with protoc 3.21: its gencode carries no runtime
version gate, so it imports under any protobuf runtime from 3.20 up, which is
what keeps ATOM installable next to whatever protobuf a plugin host pins.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_PROTOC = "libprotoc 3.21."
PROTO = Path("atom/model_engine/protocol/engine_v1.proto")
BINDING = PROTO.with_name("engine_v1_pb2.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    version = subprocess.check_output([args.protoc, "--version"], text=True).strip()
    if not version.startswith(EXPECTED_PROTOC):
        parser.error(f"expected {EXPECTED_PROTOC}x, found {version}")

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as out:
        subprocess.run(
            [args.protoc, f"-I{root}", f"--python_out={out}", str(PROTO)],
            cwd=root,
            check=True,
        )
        generated = (Path(out) / BINDING).read_bytes()
    target = root / BINDING
    if args.check:
        if not target.exists() or target.read_bytes() != generated:
            print(f"{BINDING} is stale; run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    target.write_bytes(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())

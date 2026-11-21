# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rewrite the committed Python modules of atom_grpc.proto and engine_v1.proto:
`python -m atom.entrypoints.grpc.generate`, after editing either proto."""

import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# The modules' generator: its protoc sets the grpcio and protobuf floors they
# check at import, which pyproject.toml declares.
GRPCIO_TOOLS = "1.78.0"
PROTOS = {  # path from ROOT: protoc outputs
    "atom/model_engine/zmq_protocol/engine_v1.proto": ("--python_out",),
    "atom/entrypoints/grpc/atom_grpc.proto": ("--python_out", "--grpc_python_out"),
}
MODULES = (
    "atom/model_engine/zmq_protocol/engine_v1_pb2.py",
    "atom/entrypoints/grpc/atom_grpc_pb2.py",
    "atom/entrypoints/grpc/atom_grpc_pb2_grpc.py",
)


def generated() -> dict[str, bytes]:
    """Each module in MODULES, by its path from ROOT, as protoc writes it."""
    try:
        found = version("grpcio-tools")
    except PackageNotFoundError:
        found = None
    if found != GRPCIO_TOOLS:
        raise RuntimeError(f"needs grpcio-tools=={GRPCIO_TOOLS}, found {found}")
    from grpc_tools import protoc

    with tempfile.TemporaryDirectory() as out:
        for proto, outputs in PROTOS.items():
            args = [f"-I{ROOT}", *(f"{o}={out}" for o in outputs), str(ROOT / proto)]
            if protoc.main(["protoc", *args]) != 0:
                raise RuntimeError(f"protoc failed on {proto}")
        return {module: (Path(out) / module).read_bytes() for module in MODULES}


def main() -> None:
    for module, code in generated().items():
        (ROOT / module).write_bytes(code)
        print(module)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The generated engine_v1 binding, checked against the .proto it came from.

engine_v1_pb2.py is generated at install time (setup.py) and not checked in.
Import it through here: a missing or stale binding then fails with how to fix
it, instead of as a bare ImportError or as messages the peer cannot read.
"""

import hashlib
from pathlib import Path

_PROTO = Path(__file__).with_name("engine_v1.proto")
_HINT = (
    "regenerate it with `python atom/model_engine/protocol/generate.py`, "
    "or reinstall ATOM (`pip install -e .`)"
)

try:
    import atom.model_engine.protocol.engine_v1_pb2 as pb
except ModuleNotFoundError as exc:
    if exc.name != "atom.model_engine.protocol.engine_v1_pb2":
        raise
    raise ImportError(f"the engine protocol binding is missing; {_HINT}") from exc

# Only where the .proto is present: an installed wheel always ships both.
if _PROTO.exists():
    _expected = hashlib.sha256(_PROTO.read_bytes()).hexdigest()
    if getattr(pb, "PROTO_SHA256", None) != _expected:
        raise ImportError(
            f"engine_v1_pb2.py was generated from another engine_v1.proto; {_HINT}"
        )

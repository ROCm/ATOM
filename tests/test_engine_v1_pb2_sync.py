# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The checked-in `engine_v1_pb2.py` is `engine_v1.proto` compiled.

Rust compiles the .proto at build time while Python imports the checked-in
binding, so an edit to one without `generate.py` would leave the two halves
speaking different schemas. Checked without protoc: every message's fields,
by name and number, as the .proto spells them and as the binding has them.
"""

import re
from pathlib import Path

from atom.model_engine.protocol import engine_v1_pb2 as pb

PROTO = Path(pb.__file__).with_name("engine_v1.proto")

_BLOCK = re.compile(r"^\s*(message|enum|oneof)\s+(\w+)\s*\{")
_FIELD = re.compile(
    r"^\s*(?:repeated\s+|optional\s+)?(?:map<[^>]+>|[\w.]+)\s+(\w+)\s*=\s*(\d+)\s*[;\[]"
)


def proto_fields() -> set[tuple[str, str, int]]:
    """(message, field, number) for every field the .proto declares."""
    fields, stack = set(), []
    for line in PROTO.read_text().splitlines():
        line = line.split("//", 1)[0]
        block = _BLOCK.match(line)
        if block:
            if "}" not in line:  # `message Shutdown {}` opens and closes
                stack.append(block.groups())
            continue
        if line.strip().startswith("}"):
            stack.pop()
            continue
        field = _FIELD.match(line)
        if field and stack and stack[-1][0] != "enum":
            message = ".".join(name for kind, name in stack if kind == "message")
            fields.add((message, field.group(1), int(field.group(2))))
    return fields


def binding_fields() -> set[tuple[str, str, int]]:
    fields = set()

    def walk(message, prefix):
        if message.GetOptions().map_entry:
            return
        name = f"{prefix}{message.name}"
        for field in message.fields:
            fields.add((name, field.name, field.number))
        for nested in message.nested_types:
            walk(nested, f"{name}.")

    for message in pb.DESCRIPTOR.message_types_by_name.values():
        walk(message, "")
    return fields


def test_the_binding_is_the_proto_compiled():
    in_proto, in_binding = proto_fields(), binding_fields()
    assert in_proto, "parsed no fields out of engine_v1.proto"
    assert in_proto == in_binding, (
        "engine_v1_pb2.py is stale; run atom/model_engine/protocol/generate.py\n"
        f"only in .proto: {sorted(in_proto - in_binding)}\n"
        f"only in binding: {sorted(in_binding - in_proto)}"
    )

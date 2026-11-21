# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The checked-in `engine_v1_pb2.py` is `engine_v1.proto` compiled.

Rust compiles the .proto at build time while Python imports the checked-in
binding, so an edit to one without `generate.py` would leave the two halves
speaking different schemas. Checked without protoc: every message's fields,
by name and number, as the .proto spells them and as the binding has them.

With protoc on PATH the whole descriptor is compared too -- types, labels,
oneofs, enums -- against the .proto compiled now; CI (`CI` set) must have
protoc, so that check cannot quietly turn into a skip.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from google.protobuf import descriptor_pb2

from atom.model_engine.zmq_protocol import engine_v1_pb2 as pb

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
        "engine_v1_pb2.py is stale; run atom/model_engine/zmq_protocol/generate.py\n"
        f"only in .proto: {sorted(in_proto - in_binding)}\n"
        f"only in binding: {sorted(in_binding - in_proto)}"
    )


def _without_json_names(message) -> None:
    """Drop what `--descriptor_set_out` fills in and `--python_out` strips."""
    for field in message.field:
        field.ClearField("json_name")
    for nested in message.nested_type:
        _without_json_names(nested)


def test_the_binding_carries_the_descriptor_protoc_compiles(tmp_path):
    protoc = shutil.which("protoc")
    if protoc is None:
        if os.environ.get("CI"):
            pytest.fail("CI runs this check with protoc on PATH (protobuf-compiler)")
        pytest.skip("protoc is not on PATH")
    root = PROTO.parents[3]
    out = tmp_path / "engine_v1.desc"
    subprocess.run(
        [
            protoc,
            f"-I{root}",
            f"--descriptor_set_out={out}",
            str(PROTO.relative_to(root)),
        ],
        cwd=root,
        check=True,
    )
    compiled = descriptor_pb2.FileDescriptorSet.FromString(out.read_bytes()).file[0]
    for message in compiled.message_type:
        _without_json_names(message)
    checked_in = descriptor_pb2.FileDescriptorProto.FromString(
        pb.DESCRIPTOR.serialized_pb
    )
    assert (
        compiled == checked_in
    ), "engine_v1_pb2.py is stale; run atom/model_engine/zmq_protocol/generate.py"

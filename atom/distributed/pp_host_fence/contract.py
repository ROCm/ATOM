# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Pure CPU wire/ownership planning; no tensor import, allocation or transport."""

from math import prod

KEYS = ("hidden_states", "block_residual", "pp_aux")


def plan_submission(schema, descriptors, owned=()):
    if set(descriptors) != set(KEYS):
        raise ValueError(
            "fixed K3 edge requires exactly hidden_states/block_residual/pp_aux"
        )
    if not set(owned).issubset(KEYS):
        raise ValueError("unknown owned key")
    hidden_shape = descriptors["hidden_states"].get("shape", ())
    if len(hidden_shape) != 2:
        raise ValueError("hidden_states must be a matrix")
    rows = hidden_shape[0]
    expected = schema.fields(rows)
    fields = []
    for key in KEYS:
        descriptor = descriptors[key]
        shape = expected[key]
        if (
            tuple(descriptor.get("shape", ())) != shape
            or descriptor.get("dtype") != "bfloat16"
            or descriptor.get("device") != "cuda"
            or descriptor.get("bytes") != prod(shape) * 2
            or type(descriptor.get("contiguous")) is not bool
        ):
            raise ValueError("wrong edge field descriptor: " + key)
        if rows == 0:
            action = "metadata-only"
        elif key not in owned:
            action = "clone-contiguous-on-caller-stream"
        elif descriptor["contiguous"]:
            action = "retain-owned-source"
        else:
            action = "contiguous-copy-and-retain-source-until-producer"
        fields.append(
            {
                "key": key,
                "shape": list(shape),
                "dtype": "bfloat16",
                "bytes": prod(shape) * 2,
                "action": action,
            }
        )
    return {
        "rows": rows,
        "fields": fields,
        "copyBytes": sum(x["bytes"] for x in fields),
        "zeroRowsMetadataRequired": rows == 0,
        "producerEventAfterPreparation": True,
    }


def ready_metadata(edge, sequence, rows, schema_id):
    if (
        type(sequence) is not int
        or sequence < 0
        or type(rows) is not int
        or not 0 <= rows <= 16384
    ):
        raise ValueError("invalid sequence or row count")
    if (
        type(edge) is not int
        or edge not in (0, 1, 2, 4, 5, 6)
        or not isinstance(schema_id, str)
        or not schema_id
    ):
        raise ValueError("unsupported IPC edge/schema identity")
    return {
        "kind": "READY",
        "edge": edge,
        "sequence": sequence,
        "slot": sequence % 3,
        "rows": rows,
        "schemaId": schema_id,
        "hostCopyComplete": True,
    }


def validate_ready(meta, edge, expected_sequence, schema_id):
    if set(meta) != {
        "kind",
        "edge",
        "sequence",
        "slot",
        "rows",
        "schemaId",
        "hostCopyComplete",
    }:
        raise ValueError("unexpected wire fields")
    expected = ready_metadata(edge, expected_sequence, meta["rows"], schema_id)
    if (
        meta != expected
        or any(
            type(meta[key]) is not int for key in ("edge", "sequence", "slot", "rows")
        )
        or meta["hostCopyComplete"] is not True
    ):
        raise ValueError("stale, incomplete or wrong-edge READY")
    return meta["rows"]

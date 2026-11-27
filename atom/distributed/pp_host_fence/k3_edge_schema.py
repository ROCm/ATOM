# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU schema/topology contract for the supported PP8 K3 transport.

No handle exchange, allocation, model patch or GPU import occurs here. The
model adapter uses these per-edge decisions consistently at both
ends; a global IPC flag is insufficient for a two-host pipeline.
"""

from dataclasses import dataclass
from math import prod

SPLIT = (7, 13, 13, 13, 13, 13, 13, 8)
SLOTS = 3
MAX_ROWS = 16384


@dataclass(frozen=True)
class EdgeSchema:
    hidden: int
    blocks: int
    aux_taps: int
    max_rows: int = MAX_ROWS

    def fields(self, rows):
        if type(rows) is not int or not 0 <= rows <= self.max_rows:
            raise ValueError("row count outside the fixed K3 capacity")
        return {
            "hidden_states": (rows, self.hidden),
            "block_residual": (rows, self.blocks, self.hidden),
            "pp_aux": (rows, self.aux_taps * self.hidden),
        }

    def metadata(self):
        fields, offset = [], 0
        for key, shape in self.fields(self.max_rows).items():
            nbytes = prod(shape) * 2
            fields.append(
                {
                    "key": key,
                    "shape": list(shape),
                    "dtype": "bfloat16",
                    "offset": offset,
                    "capacityBytes": nbytes,
                }
            )
            offset += nbytes
        return {
            "maxRows": self.max_rows,
            "fields": fields,
            "slotBytes": offset,
            "slots": SLOTS,
            "ringBytes": offset * SLOTS,
        }

    def validate(self, fields, rows):
        expected = self.fields(rows)
        if set(fields) != set(expected):
            raise ValueError(
                "K3 edge keys differ; never rename block_residual to residual"
            )
        for key, shape in expected.items():
            actual = fields[key]
            if (
                tuple(actual["shape"]) != shape
                or actual["dtype"] != "bfloat16"
                or actual["contiguous"] is not True
                or actual["bytes"] != prod(shape) * 2
            ):
                raise ValueError("K3 field shape/dtype/stride/bytes mismatch: " + key)


def edge_schemas(main_config, draft_config):
    cfg = main_config["text_config"]
    if main_config["model_type"] != "kimi_k3" or cfg["num_hidden_layers"] != sum(SPLIT):
        raise ValueError("wrong fixed model or layer partition")
    hidden = cfg["hidden_size"]
    block_size = cfg["attn_res_block_size"]
    taps = draft_config["target_layer_ids"]
    if (
        hidden != 7168
        or block_size != 12
        or taps != [2, 23, 47, 71, 89]
        or draft_config["target_hidden_size"] != hidden
        or draft_config["target_num_hidden_layers"] != cfg["num_hidden_layers"]
    ):
        raise ValueError("model/draft schema differs from the source-bound reference")
    end, schemas = 0, []
    for layers in SPLIT[:-1]:
        end += layers
        # KimiLinearModel starts with an empty bank. AttnRes.maybe_close_block
        # appends at layer indices 0, 12, ... below this exclusive stage end.
        blocks = (end + block_size - 1) // block_size
        schemas.append(EdgeSchema(hidden, blocks, sum(tap < end for tap in taps)))
    return schemas


def routes(identities):
    """Select links from gathered actual kernel host/boot/device identities."""
    if len(identities) != 8 or {item["ppRank"] for item in identities} != set(range(8)):
        raise ValueError("exactly eight distinct pipeline stages required")
    ordered = sorted(identities, key=lambda item: item["ppRank"])
    for item in ordered:
        if any(
            not isinstance(item.get(key), str)
            or not item[key]
            or item[key] == "unknown"
            for key in ("hostname", "bootId", "gpuUuid")
        ):
            raise ValueError(
                "unknown topology must not authorize a HIP handle exchange"
            )
    devices = [(item["hostname"], item["bootId"], item["gpuUuid"]) for item in ordered]
    if len(set(devices)) != 8:
        raise ValueError("pipeline aliases the same physical device")
    return [
        {
            "src": rank,
            "dst": rank + 1,
            "transport": (
                "hip-ipc-memory"
                if (ordered[rank]["hostname"], ordered[rank]["bootId"])
                == (ordered[rank + 1]["hostname"], ordered[rank + 1]["bootId"])
                else "rccl"
            ),
            "requiresValidatedBackend": True,
        }
        for rank in range(7)
    ]

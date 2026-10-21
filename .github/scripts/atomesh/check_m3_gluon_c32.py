#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refuse changes to the user's gather C32 recipe beyond the kernel source pin."""

import argparse
import json
from pathlib import Path


def compare(matrix):
    baseline = json.loads(
        Path(__file__).with_name("m3_gluon_c32_baseline.json").read_text()
    )
    old = baseline["include"][0]
    cells = matrix["include"]
    if len(cells) != 1 or cells[0]["name"] != old["name"]:
        raise ValueError("Select only the C32 case from gather run 39034151209")
    new = json.loads(json.dumps(cells[0]))
    refs = []
    for role in ("common", "prefill", "decode", "router"):
        key = "ATOM_AITER_SCATTER_REF"
        refs.append(new["env"][role][key])
        new["env"][role][key] = old["env"][role][key]
    if len(set(refs)) != 1 or refs[0] == old["env"]["common"]["ATOM_AITER_SCATTER_REF"]:
        raise ValueError("Expected one new kernel pin across roles")
    # Scheduling may avoid a node that failed host-memory/RDMA preparation.
    # Preserve the saved baseline and reject every other configuration change.
    excluded = new["runner"]["exclude_nodes"]
    if "pit2-p03-g41" in excluded:
        print("[gluon-c32-config] excluding g41 after host-memory/RDMA failures")
    new["runner"]["exclude_nodes"] = [n for n in excluded if n != "pit2-p03-g41"]
    if new != old:
        different = [
            key for key in old.keys() | new.keys() if old.get(key) != new.get(key)
        ]
        raise ValueError(f"Non-kernel configuration changed: {different}")
    print(
        "[gluon-c32-config] matches run 39034151209 C32; only AITER kernel pin differs: "
        + refs[0]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    args = parser.parse_args()
    compare(json.loads(args.matrix.read_text()))


if __name__ == "__main__":
    main()

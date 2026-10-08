"""Check Mooncake auto-matched primary HCAs before loading model weights."""

import argparse
import json
import os
from pathlib import Path


def inspect_primaries(root, indices, configured_device=""):
    devices = [x.strip() for x in configured_device.split(",") if x.strip()]
    if not devices:
        for index in indices:
            # Keep the connector's rdmaN-before-ionic_N selection order.
            rdma = f"rdma{index}"
            ionic = f"ionic_{index}"
            devices.append(rdma if (root / rdma).exists() else ionic)
    records = []
    for device in dict.fromkeys(devices):
        states = {}
        for state_file in sorted((root / device).glob("ports/*/state")):
            try:
                states[state_file.parent.name] = state_file.read_text().strip()
            except OSError as error:
                states[state_file.parent.name] = f"unreadable: {error}"
        active = any(
            value.partition(":")[0].strip() == "4" for value in states.values()
        )
        records.append({"device": device, "states": states, "active": active})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kv-transfer-config", required=True)
    args = parser.parse_args()
    config = json.loads(args.kv_transfer_config)
    if (
        config.get("kv_connector") != "mooncake"
        or config.get("protocol", "rdma").lower() != "rdma"
        or os.environ.get("ATOM_MOONCAKE_MATCHED_RAILS", "").strip().lower() != "auto"
    ):
        return 0
    visible = (
        os.environ.get("ROCR_VISIBLE_DEVICES")
        or os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    if not visible:
        raise ValueError("RDMA preflight requires an explicit visible GPU list")
    records = inspect_primaries(
        Path("/sys/class/infiniband"),
        [int(value) for value in visible.split(",") if value],
        config.get("ib_device", "") or os.environ.get("ATOM_MOONCAKE_IB_DEVICE", ""),
    )
    for record in records:
        print("[rdma-preflight] " + json.dumps(record), flush=True)
    if not records or any(not record["active"] for record in records):
        print(
            "[rdma-preflight][FAIL] Primary HCA has no readable ACTIVE port", flush=True
        )
        return 1
    print(
        "[rdma-preflight][OK] All selected primary HCAs have an ACTIVE port", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

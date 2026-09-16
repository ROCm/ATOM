#!/usr/bin/env python3
"""Select HIP IPC for same-host Mooncake connectors at server launch time."""

import argparse
import json


def use_hip(config: dict) -> None:
    """Preserve connector composition while removing RDMA-only settings."""
    if config.get("kv_connector") == "mooncake":
        config["protocol"] = "hip"
        for key in ("ib_device", "ib_enable_alternate_hca", "ib_rail_offset"):
            config.pop(key, None)
    for child in config.get("connectors", []):
        if isinstance(child, dict):
            use_hip(child)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="KV transfer JSON after runtime expansion")
    args = parser.parse_args()
    config = json.loads(args.config)
    use_hip(config)
    print(json.dumps(config, separators=(",", ":")))


if __name__ == "__main__":
    main()

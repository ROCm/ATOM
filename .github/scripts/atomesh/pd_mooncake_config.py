#!/usr/bin/env python3
"""Configure native Mooncake direct P/D and a prefill CPU Store."""

import json
import os
import sys


def connector(role):
    if role not in ("prefill", "decode"):
        raise ValueError(f"Unsupported role: {role}")
    direct = {
        "kv_connector": "MooncakeConnector",
        "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
        "kv_connector_extra_config": {"mooncake_protocol": "rdma", "num_workers": 1},
    }
    if role == "decode":
        return {**direct, "kv_load_failure_policy": "fail"}
    return {
        "kv_connector": "MultiConnector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "fail",
        "kv_connector_extra_config": {
            "connectors": [
                direct,
                {
                    "kv_connector": "MooncakeStoreConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "load_async": True,
                        "cache_prefix": "k3-main-bacbbe187-dspark3-dcp8-fp8-20261009",
                    },
                },
            ]
        },
    }


def store():
    return {
        "mode": "embedded",
        "metadata_server": "P2PHANDSHAKE",
        "master_server_address": f"127.0.0.1:{os.environ['MOONCAKE_MASTER_PORT']}",
        "global_segment_size": os.environ["ATOMESH_VLLM_MOONCAKE_GLOBAL_SEGMENT_SIZE"],
        "local_buffer_size": os.environ["ATOMESH_VLLM_MOONCAKE_LOCAL_BUFFER_SIZE"],
        "protocol": "rdma",
        "device_name": "",
        "enable_offload": False,
    }


if __name__ == "__main__":
    result = connector(sys.argv[2]) if sys.argv[1] == "connector" else store()
    print(json.dumps(result, indent=2))

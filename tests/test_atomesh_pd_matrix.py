import importlib.util
import json
from pathlib import Path


def _load_pd_matrix():
    path = (
        Path(__file__).resolve().parents[1]
        / ".github"
        / "scripts"
        / "atomesh"
        / "pd_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("atomesh_pd_matrix", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_node_hip_env_adds_direct_mooncake_configs():
    pd_matrix = _load_pd_matrix()
    env = {"common": {}, "prefill": {}, "decode": {}, "router": {}}

    pd_matrix._single_node_hip_env(env, {"tp": 4})

    prefill = json.loads(env["prefill"]["PREFILL_KV_TRANSFER_CONFIG"])
    decode = json.loads(env["decode"]["DECODE_KV_TRANSFER_CONFIG"])
    assert prefill == {
        "kv_role": "kv_producer",
        "kv_connector": "mooncake",
        "protocol": "hip",
        "proxy_ip": "${ROLE_IP}",
        "handshake_port": 6301,
    }
    assert decode == {
        "kv_role": "kv_consumer",
        "kv_connector": "mooncake",
        "protocol": "hip",
        "proxy_ip": "${ROLE_IP}",
        "handshake_port": 6305,
    }


def test_single_node_hip_env_updates_nested_mooncake_only():
    pd_matrix = _load_pd_matrix()
    env = {
        "common": {},
        "prefill": {
            "PREFILL_KV_TRANSFER_CONFIG": json.dumps(
                {
                    "kv_connector": "multi",
                    "connectors": [
                        {
                            "kv_connector": "mooncake",
                            "kv_role": "kv_producer",
                            "ib_enable_alternate_hca": True,
                            "ib_rail_offset": 4,
                        },
                        {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                    ],
                }
            )
        },
        "decode": {},
        "router": {},
    }

    pd_matrix._single_node_hip_env(env, {"tp": 4})

    prefill = json.loads(env["prefill"]["PREFILL_KV_TRANSFER_CONFIG"])
    mooncake, lmcache = prefill["connectors"]
    assert mooncake["protocol"] == "hip"
    assert "ib_enable_alternate_hca" not in mooncake
    assert "ib_rail_offset" not in mooncake
    assert lmcache == {"kv_connector": "lmcache_offload", "kv_role": "offload"}

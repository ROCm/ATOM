"""Exercise the real server launch functions without starting model servers."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh"


@pytest.mark.parametrize("single_node", [False, True])
@pytest.mark.parametrize("port_offset", [0, 10000])
@pytest.mark.parametrize("nested", [False, True])
def test_launch_transport_and_role_ports(single_node, port_offset, nested):
    source = (SCRIPT_DIR / "pd_server_atom.sh").read_text()
    functions = "\n".join(
        re.search(rf"^{name}\(\) \{{.*?^\}}", source, re.MULTILINE | re.DOTALL).group()
        for name in (
            "apply_prefixed_env",
            "apply_role_env",
            "start_prefill",
            "start_decode",
        )
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ATOMESH_")
    }
    env.update(
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        ATOMESH_SCRIPT_DIR=str(SCRIPT_DIR),
        SINGLE_NODE_PD=str(int(single_node)),
        HANDSHAKE_PORT=str(6301 + port_offset),
        PREFILL_PORT=str(8010 + port_offset),
        DECODE_PORT=str(8020 + port_offset),
        PREFILL_DP_MASTER_PORT="29500",
        PREFILL_DP_BASE_PORT="29600",
        DECODE_DP_MASTER_PORT="29700",
        DECODE_DP_BASE_PORT="29800",
        USE_EXPLICIT_DP_PORTS="1",
        NODE_RANK="0",
        host_ip="127.0.0.1",
        host_name="test-node",
        HIP_VISIBLE_DEVICES="0,1,2,3",
        MAX_NUM_SEQS="16",
        BENCH_MAX_CONCURRENCY="16",
        DECODE_MAX_NUM_SEQS="",
        DECODE_MAX_NUM_BATCHED_TOKENS="",
        ISL_LIST="128",
        OSL="32",
        PREFILL_SERVER_ARGS="",
        DECODE_SERVER_ARGS="",
        RUNTIME_LOG_DIR="/unused",
        PREFILL_KV_TRANSFER_CONFIG="",
        DECODE_KV_TRANSFER_CONFIG="",
    )
    if nested:
        prefill = {
            "kv_connector": "multi",
            "connectors": [
                {
                    "kv_connector": "mooncake",
                    "kv_role": "kv_producer",
                    "protocol": "rdma",
                    "proxy_ip": "${ROLE_IP}",
                    "handshake_port": "${HANDSHAKE_PORT}",
                    "ib_device": "ionic_0",
                    "ib_enable_alternate_hca": True,
                    "ib_rail_offset": 4,
                },
                {"kv_connector": "lmcache_offload", "kv_role": "offload"},
            ],
        }
        decode = dict(prefill["connectors"][0], kv_role="kv_consumer")
        # Match the unquoted integer template used by models_atomesh.yaml.
        for role, config in (("PREFILL", prefill), ("DECODE", decode)):
            env[f"ATOMESH_{role}_ENV_{role}_KV_TRANSFER_CONFIG"] = json.dumps(
                config
            ).replace('"${HANDSHAKE_PORT}"', "${HANDSHAKE_PORT}")

    harness = r"""
set -euo pipefail
ROLE_ENV_NAMES=()
server_common=(); prefill_parallel=(); decode_parallel=()
prefill_cudagraph_args=(); decode_cudagraph_args=()
reset_lmcache_disk() { :; }
build_server_cache_env() { :; }
dump_launch_info() { :; }
start_logged_process() {
  shift 3
  python3 -c 'import sys; print("LAUNCH_JSON=" + sys.argv[sys.argv.index("--kv-transfer-config") + 1])' "$@"
}
"""
    harness += functions + "\nstart_prefill prefill\n"
    harness += 'start_decode decode "$DECODE_PORT" "$((HANDSHAKE_PORT + 4))"\n'
    result = subprocess.run(
        ["bash", "-c", harness], env=env, check=True, text=True, capture_output=True
    )
    configs = [
        json.loads(line.removeprefix("LAUNCH_JSON="))
        for line in result.stdout.splitlines()
        if line.startswith("LAUNCH_JSON=")
    ]
    assert len(configs) == 2
    if nested:
        assert configs[0]["connectors"][1] == {
            "kv_connector": "lmcache_offload",
            "kv_role": "offload",
        }
        configs[0] = configs[0]["connectors"][0]
    for index, config in enumerate(configs):
        assert config.get("protocol", "rdma") == ("hip" if single_node else "rdma")
        assert config["handshake_port"] == 6301 + port_offset + 4 * index
        assert config["proxy_ip"] == "127.0.0.1"
        if single_node:
            assert (
                not {"ib_device", "ib_enable_alternate_hca", "ib_rail_offset"}
                & config.keys()
            )
        elif nested:
            assert config["ib_device"] == "ionic_0"


def test_non_mooncake_config_is_preserved():
    config = {"kv_connector": "moriio", "protocol": "rdma", "handshake_port": 7000}
    result = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "pd_hip_config.py"), json.dumps(config)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(result.stdout) == config

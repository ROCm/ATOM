# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-role online quant configs, from the cell JSON to each server's flags."""

import copy
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
BASE = {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head"]}


def submit_mapping():
    """pd_submit.sh's cell-to-environment mapping, runnable on its own."""
    source = (SCRIPTS / "pd_submit.sh").read_text()
    start = source.index("python3 - <<'PY'\nimport json\nimport os\nimport shlex")
    start = source.index("\n", start) + 1
    return source[start : source.index("\nPY\n", start)]


class SubmitOnlineQuantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = dict(
            os.environ,
            ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
            ATOMESH_SLURM_ACCOUNT="",
            ATOMESH_SLURM_PARTITION="",
            ATOMESH_MODEL_ROOT="/models",
            ATOMESH_LOG_ROOT="/logs",
            ATOMESH_1P1D_NODES="",
            ATOMESH_2P1D_NODES="",
            ATOMESH_PD_RANK_MAPPING_POLICY="none",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "matrix.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "pd_matrix.py"),
                    "--suite",
                    "nightly",
                    "--model",
                    "Kimi-K3-MXFP4",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            cls.cell = json.loads(output.read_text())["include"][0]
        cls.mapping = submit_mapping()

    def exports(self, shared_extra=None, prefill_extra=None, decode_extra=None):
        cell = copy.deepcopy(self.cell)
        cell["server_args"]["online_quant_config"] = json.dumps(BASE)
        for cfg, extra in (
            (cell["server_args"], shared_extra),
            (cell["service"]["prefill"], prefill_extra),
            (cell["service"]["decode"], decode_extra),
        ):
            cfg.pop("online_quant_extra_exclude_layer", None)
            if extra is not None:
                cfg["online_quant_extra_exclude_layer"] = extra
        result = subprocess.run(
            [sys.executable, "-c", self.mapping],
            env=dict(os.environ, CELL_JSON=json.dumps(cell)),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        exports = {}
        for line in result.stdout.splitlines():
            words = shlex.split(line)
            if words[:1] == ["export"]:
                key, _, value = words[1].partition("=")
                exports[key] = value
            else:
                exports.setdefault("_shell", []).append(line)
        return exports

    def excludes(self, exports, role):
        return json.loads(exports[f"{role}_ONLINE_QUANT_CONFIG"])["exclude_layer"]

    def test_roles_share_the_config_without_extras(self):
        exports = self.exports()
        self.assertEqual(json.loads(exports["PREFILL_ONLINE_QUANT_CONFIG"]), BASE)
        self.assertEqual(json.loads(exports["DECODE_ONLINE_QUANT_CONFIG"]), BASE)

    def test_extras_append_shared_first_and_stay_on_their_role(self):
        exports = self.exports(
            shared_extra=["context_proj"],
            prefill_extra=["*self_attn.f_b_proj"],
            decode_extra=["*shared_experts.down_proj*"],
        )
        self.assertEqual(
            self.excludes(exports, "PREFILL"),
            ["lm_head", "context_proj", "*self_attn.f_b_proj"],
        )
        self.assertEqual(
            self.excludes(exports, "DECODE"),
            ["lm_head", "context_proj", "*shared_experts.down_proj*"],
        )
        self.assertEqual(
            json.loads(exports["PREFILL_ONLINE_QUANT_CONFIG"])["global_quant_config"],
            "ptpc_fp8",
        )
        self.assertNotIn("_shell", exports)

    def test_extras_without_a_config_fail_the_cell(self):
        cell = copy.deepcopy(self.cell)
        cell["server_args"].pop("online_quant_config", None)
        cell["service"]["decode"]["online_quant_extra_exclude_layer"] = ["x"]
        result = subprocess.run(
            [sys.executable, "-c", self.mapping],
            env=dict(os.environ, CELL_JSON=json.dumps(cell)),
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertIn("without an online_quant_config", result.stdout)
        self.assertEqual(result.stdout.splitlines()[-1], "exit 1")


class LaunchOnlineQuantTest(unittest.TestCase):
    STUBS = """
apply_role_env() { :; }
reset_lmcache_disk() { :; }
start_mooncake_store() { :; }
start_mooncake_store_decode_owners() { :; }
mooncake_store_running() { return 1; }
mooncake_store_requested() { return 1; }
build_server_cache_env() { local -n out="$3"; out=(); }
build_torch_profiler_args() { local -n out="$2"; out=(); }
kv_transfer_config() { echo '{}'; }
dump_launch_info() { :; }
start_logged_process() { shift 3; printf 'ARG:%s\\n' "$@"; }
"""

    @classmethod
    def setUpClass(cls):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        functions = []
        for name in ("build_online_quant_args", "start_prefill", "start_decode"):
            start = source.index(f"{name}() {{")
            functions.append(source[start : source.index("\n}\n", start) + 3])
        cls.functions = "\n".join(functions)

    def launch(self, role, **env):
        script = self.functions + self.STUBS + f"""
server_common=(--model /m)
prefill_parallel=(); decode_parallel=()
prefill_cudagraph_args=(); decode_cudagraph_args=()
start_{role} {role}-rank-0
"""
        result = subprocess.run(
            ["bash", "-eu", "-c", script],
            env=dict(
                os.environ,
                **{
                    "PREFILL_ONLINE_QUANT_CONFIG": '{"p":1}',
                    "DECODE_ONLINE_QUANT_CONFIG": '{"d":1}',
                    "PREFILL_SERVER_ARGS": "",
                    "DECODE_SERVER_ARGS": "",
                    **env,
                },
                NODE_RANK="0",
                host_ip="10.0.0.1",
                host_name="n0",
                HIP_VISIBLE_DEVICES="0",
                PREFILL_PORT="8010",
                DECODE_PORT="8020",
                HANDSHAKE_PORT="6301",
                PREFILL_DP_MASTER_PORT="1",
                PREFILL_DP_BASE_PORT="2",
                DECODE_DP_MASTER_PORT="3",
                DECODE_DP_BASE_PORT="4",
                USE_EXPLICIT_DP_PORTS="0",
                MAX_NUM_SEQS="8",
                DECODE_MAX_NUM_SEQS="",
                DECODE_MAX_NUM_BATCHED_TOKENS="",
                BENCH_MAX_CONCURRENCY="8",
                ISL_LIST="1048576",
                OSL="1024",
                PREFILL_KV_TRANSFER_CONFIG="",
                DECODE_KV_TRANSFER_CONFIG="",
                RUNTIME_LOG_DIR="/tmp",
                ATOM_SOURCE_ENV="",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return [
            line[4:] for line in result.stdout.splitlines() if line.startswith("ARG:")
        ]

    def quant_values(self, args):
        return [
            args[i + 1] for i, arg in enumerate(args) if arg == "--online_quant_config"
        ]

    def test_each_role_launches_with_its_own_config(self):
        self.assertEqual(self.quant_values(self.launch("prefill")), ['{"p":1}'])
        self.assertEqual(self.quant_values(self.launch("decode")), ['{"d":1}'])

    def test_a_flag_in_extra_args_still_comes_last(self):
        args = self.launch(
            "prefill",
            PREFILL_SERVER_ARGS='--enforce-eager --online_quant_config {"x":1}',
        )
        self.assertEqual(self.quant_values(args), ['{"p":1}', '{"x":1}'])

    def test_no_config_no_flag(self):
        args = self.launch("decode", DECODE_ONLINE_QUANT_CONFIG="")
        self.assertEqual(self.quant_values(args), [])


if __name__ == "__main__":
    unittest.main()

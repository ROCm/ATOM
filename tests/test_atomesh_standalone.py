# SPDX-License-Identifier: MIT
"""Exercise the standalone catalog through env export and real launch builders."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
CASE = "kimi-k3-mxfp4-standalone-tp8-dcp8-agentic-lmcache-state-1m-c48"
spec = importlib.util.spec_from_file_location("pd_matrix", SCRIPTS / "pd_matrix.py")
pd_matrix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pd_matrix)


class StandaloneTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.env = dict(
            os.environ,
            ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd",
            ATOMESH_SLURM_ACCOUNT="",
            ATOMESH_SLURM_PARTITION="",
            ATOMESH_MODEL_ROOT="/models",
            ATOMESH_LOG_ROOT=str(self.directory),
            ATOMESH_NODE_POOL="node-a,node-b",
            ATOMESH_1P1D_NODES="",
            ATOMESH_SINGLE_NODE="auto",
            ATOMESH_PD_RANK_MAPPING_POLICY="none",
        )
        self.config = pd_matrix.load_config(
            ROOT / ".github/benchmark/models_atomesh.yaml"
        )

    def cells(self, case=CASE):
        with patch.dict(os.environ, self.env):
            return pd_matrix.build_cells(
                self.config,
                suite="nightly",
                model_filter={"Kimi-K3-MXFP4"},
                case_filter={case},
                benchmark_kind_filter=None,
                override_image=None,
                override_benchmark_concurrency=None,
                override_eval_concurrency=None,
            )

    def test_one_node_and_no_decode_worker(self):
        for runner in (
            "atomesh-cicd",
            "atomesh-cicd-mi350",
            "atomesh-cicd-mi355-crusoe",
        ):
            with self.subTest(runner=runner):
                self.env["ATOMESH_SLURM_SUBMIT_RUNNER"] = runner
                self.env["ATOMESH_1P1D_NODES"] = "node-a,node-b"
                (cell,) = self.cells()
                self.assertEqual(cell["num_nodes"], 1)
                self.assertEqual(cell["service"]["decode"]["workers"], 0)
                self.assertEqual(cell["pd_worker_layout"], "standalone")
                self.assertEqual(cell["concurrency"], [48])
        self.env["ATOMESH_SLURM_SUBMIT_RUNNER"] = "atomesh-cicd"
        self.env["ATOMESH_SINGLE_NODE"] = "node-b"
        self.assertEqual(self.cells()[0]["nodes"], ["node-b"])

    def test_existing_pd_case_still_uses_two_nodes(self):
        (cell,) = self.cells("kimi-k3-mxfp4-1p1d-tp8-dcp8-agentic-lmcache-1m-c48")
        self.assertEqual(cell["num_nodes"], 2)
        self.assertEqual(cell["service"]["decode"]["workers"], 1)
        self.assertNotIn("kv_transfer_config", cell["server_args"])

    def test_standalone_rejects_multiple_servers_and_oversized_workers(self):
        case = next(
            c
            for c in self.config["models"]["Kimi-K3-MXFP4"]["suites"]["nightly"]
            if c["name"] == CASE
        )
        case["prefill"]["workers"] = 2
        with self.assertRaisesRegex(ValueError, "exactly one ATOM server"):
            self.cells()
        case["prefill"]["workers"] = 1
        case["prefill"]["extra_args"] = "--pipeline-parallel-size 2"
        with self.assertRaisesRegex(ValueError, "fit on one"):
            self.cells()

    def test_mesh_setup_is_skipped_on_both_slurm_paths(self):
        source = (SCRIPTS / "pd_slurm_job.sh").read_text()
        markers = ("  local mesh_binary=", "      mesh_binary=")
        for marker in markers:
            start = source.index(marker)
            end = source.index("\n", source.index("fi\n", start))
            block = source[start:end].replace("local mesh_binary=", "mesh_binary=")
            result = subprocess.run(
                [
                    "bash",
                    "-eu",
                    "-c",
                    'bash() { echo "Unexpected Mesh setup" >&2; exit 99; }\n'
                    "rank=0\nATOMESH_PD_WORKER_LAYOUT=standalone\n" + block,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("Unexpected Mesh", result.stderr)

    def test_standalone_resource_count_is_eight_gpus(self):
        with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
            spec = importlib.util.spec_from_file_location(
                "standalone_results", SCRIPTS / "process_result.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        resources = module.topology_resources(
            {
                "topology": "standalone_tp8_dcp8",
                "prefill_workers": 1,
                "decode_workers": 0,
                "prefill_tp": 8,
                "decode_tp": 8,
            },
            {},
        )
        self.assertEqual(resources["total_gpu"], 8)
        self.assertEqual(resources["num_decode_gpu"], 0)

    def launch(self):
        (cell,) = self.cells()
        submit = (SCRIPTS / "pd_submit.sh").read_text()
        export_code = submit.split("python3 - <<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
        exports = subprocess.run(
            [sys.executable, "-c", export_code],
            env=self.env | {"CELL_JSON": json.dumps(cell)},
            check=True,
            text=True,
            capture_output=True,
        ).stdout
        capture = self.directory / "capture.py"
        capture.write_text("""import json, os, sys
from pathlib import Path
kind, *args = sys.argv[1:]
root = Path(os.environ['CAPTURE_DIR'])
env = {k: v for k, v in os.environ.items() if k.startswith(('ATOM_', 'AITER_', 'LMCACHE_', 'OFFLOAD_'))}
(root / (kind + '.json')).write_text(json.dumps({'args': args, 'env': env}))
if kind == 'benchmark':
    out = Path(args[args.index('--output-artifact-dir') + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'profile_export_aiperf.json').write_text('{}')
""")
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        source = source.replace(
            "rm -rf /root/.cache/atom/* 2>/dev/null || true",
            ": # No shared cache cleanup in CPU tests",
        )
        source = source.replace(
            'ATOMESH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            "ATOMESH_SCRIPT_DIR=" + shlex.quote(str(SCRIPTS)),
        )
        stubs = r"""
dump_launch_info() { :; }
start_logged_process() {
  local pid_var="$1"
  shift 2
  "$TEST_PYTHON" "$CAPTURE_SCRIPT" launch "$@"
  printf -v "$pid_var" '%s' "$$"
}
wait_http() { "$TEST_PYTHON" "$CAPTURE_SCRIPT" health "$@"; }
cleanup_processes() { :; }
start_router() { echo 'Unexpected Mesh router' >&2; exit 91; }
start_prefill() { echo 'Unexpected P/D server' >&2; exit 92; }
start_decode() { echo 'Unexpected P/D server' >&2; exit 93; }
ensure_aiperf() { :; }
write_aiperf_dashboard_json() { :; }
write_aiperf_chrome_trace() { :; }
python3() {
  if [[ "$1" == */observability/collect_metrics.py ]]; then
    shift
    "$TEST_PYTHON" "$CAPTURE_SCRIPT" benchmark "$@"
  else
    "$TEST_PYTHON" "$@"
  fi
}
"""
        source = source.replace(
            "\nwrite_metadata\n", "\n" + stubs + "\nwrite_metadata\n"
        )
        script = (
            exports
            + """
export xP="$PREFILL_WORKERS" yD="$DECODE_WORKERS"
export PREFILL_TP_SIZE="$PREFILL_TP" DECODE_TP_SIZE="$DECODE_TP"
"""
            + source
        )
        path = self.directory / "launch.sh"
        path.write_text(script)
        result = subprocess.run(
            ["bash", str(path)],
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
            env=self.env
            | {
                "TEST_PYTHON": sys.executable,
                "CAPTURE_DIR": str(self.directory),
                "CAPTURE_SCRIPT": str(capture),
                "RUN_DIR": str(self.directory / "run"),
                "ATOMESH_WORKER_CACHE_BASE": str(self.directory / "cache"),
                "NODE_RANK": "0",
                "IPADDRS": "127.0.0.1",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return {
            name: json.loads((self.directory / f"{name}.json").read_text())
            for name in ("launch", "benchmark", "health")
        }

    def test_real_launch_and_client_connect_directly_with_original_recipe(self):
        captures = self.launch()
        args = captures["launch"]["args"]
        for flag, value in {
            "-tp": "8",
            "--decode-context-parallel-size": "8",
            "--data-parallel-size": "1",
            "--max-num-seqs": "96",
            "--max-num-batched-tokens": "8192",
            "--block-size": "128",
            "--state-checkpoint-interval-tokens": "-1",
            "--kv_cache_dtype": "fp8",
            "--cudagraph-mode": "FULL",
            "--level": "3",
            "--server-port": "8000",
        }.items():
            self.assertEqual(args.count(flag), 1, flag)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(
            json.loads(args[args.index("--kv-transfer-config") + 1]),
            {"kv_connector": "lmcache_offload", "kv_role": "offload"},
        )
        self.assertEqual(
            json.loads(args[args.index("--cudagraph-capture-sizes") + 1]),
            list(range(1, 97)),
        )
        self.assertFalse(
            {
                "--method",
                "--draft-model",
                "--num-speculative-tokens",
                "--spec-decode-acceptance-length",
                "--pd-disaggregation",
                "--no-enable_prefix_caching",
            }
            & set(args)
        )
        env = captures["launch"]["env"]
        for key, value in {
            "ATOM_KDA_SPARE_STATE_CHECKPOINTS": "1",
            "ATOM_KDA_SPARE_STATE_RESERVE": "8",
            "ATOM_ENABLE_REPLAYSSM": "0",
            "ATOM_GDN_SSM_DTYPE": "fp16",
            "ATOM_STATE_CHECKPOINT_DEMAND": "0",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": "128",
            "LMCACHE_CHUNK_SIZE": "1024",
            "LMCACHE_LOCAL_CPU": "True",
            "OFFLOAD_KV_FOR_HYBRID": "1",
            "OFFLOAD_WARMUP": "1",
        }.items():
            self.assertEqual(env[key], value)
        client = captures["benchmark"]["args"]
        for flag, value in {
            "--url": "http://127.0.0.1:8000",
            "--standalone": "127.0.0.1:8000",
            "--server-metrics": "http://127.0.0.1:8000/metrics",
            "--concurrency": "48",
            "--benchmark-duration": "3600",
            "--max-context-length": "1048576",
            "--failed-request-threshold": "0.1",
            "--agentic-warmup-grace-period": "1800",
        }.items():
            self.assertEqual(client[client.index(flag) + 1], value)
        self.assertNotIn("--mesh", client)
        self.assertNotIn("--decode", client)
        self.assertEqual(captures["health"]["args"][0], "http://127.0.0.1:8000/health")


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for MoRI IO PD accuracy cells and their launcher support."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
SERVER_SCRIPT = SCRIPTS / "pd_server_atom.sh"
SPUR_NODES = ",".join(
    f"ml-ai-ubuntu-gpu-mi350x8-2304gb-fabric-{index}"
    for index in (5, 6, 7, 8, 13, 14, 15, 16)
)


def script_section(start: str, end: str) -> str:
    source = SERVER_SCRIPT.read_text()
    return source[source.index(start) : source.index(end)]


def generate_cells(model: str) -> list[dict]:
    """Run the matrix over one model's accuracy suite on the Spur lane."""
    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "cells.json"
        subprocess.run(
            [
                "python3",
                str(SCRIPTS / "pd_matrix.py"),
                "--suite",
                "accuracy",
                "--model",
                model,
                "--output",
                str(output),
            ],
            cwd=ROOT,
            env=dict(
                os.environ,
                ATOMESH_SLURM_SUBMIT_RUNNER="atomesh-cicd-mi350",
                ATOMESH_SLURM_ACCOUNT="",
                ATOMESH_SLURM_PARTITION="",
                ATOMESH_LOG_ROOT="/data/${USER}/logs/ATOMESH_LOG/",
                ATOMESH_MODEL_ROOT="/data/models2",
                ATOMESH_1P1D_NODES=SPUR_NODES,
                ATOMESH_2P1D_NODES=SPUR_NODES,
                ATOMESH_PD_RANK_MAPPING_POLICY="idx2idx",
            ),
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return json.loads(output.read_text())["include"]


class MoriIOCellTest(unittest.TestCase):
    """The catalog cell must reach the launcher as a MoRI IO gsm8k-only run."""

    @classmethod
    def setUpClass(cls):
        cells = generate_cells("Kimi-K2.5-MXFP4")
        cls.cell = next(cell for cell in cells if "moriio" in cell["id"])

    def test_cell_selects_moriio_for_both_roles(self):
        env = self.cell["env"]
        for role in ("common", "prefill", "decode"):
            self.assertEqual(env[role]["KV_CONNECTOR"], "moriio")
        # The model-level Mooncake JSON must not survive into a MoRI IO cell.
        self.assertEqual(env["decode"].get("DECODE_KV_TRANSFER_CONFIG"), "")
        self.assertEqual(env["prefill"].get("PREFILL_KV_TRANSFER_CONFIG"), None)

    def test_cell_is_gsm8k_only_across_two_nodes(self):
        self.assertTrue(self.cell["run_eval"])
        self.assertTrue(self.cell["eval_only"])
        self.assertEqual(self.cell["accuracy"]["task"], "gsm8k")
        self.assertEqual(self.cell["accuracy"]["threshold"], 0.90)
        # Chat completions, so the template's thinking switch applies.
        self.assertEqual(self.cell["accuracy"]["model_type"], "local-chat-completions")
        self.assertEqual(self.cell["accuracy"]["endpoint"], "chat/completions")
        self.assertTrue(self.cell["accuracy"]["apply_chat_template"])
        # lm_eval scores content, which a model cut off mid-thought leaves empty.
        self.assertIn(
            '--default-chat-template-kwargs {"thinking":false}',
            self.cell["server_args"]["extra_args"],
        )
        # Prefill and decode own a node each, so KV crosses the fabric.
        self.assertEqual(self.cell["num_nodes"], 2)
        self.assertEqual(self.cell["service"]["prefill"]["tp"], 8)
        self.assertEqual(self.cell["service"]["decode"]["tp"], 8)


class DeepSeekV3AttributionCellsTest(unittest.TestCase):
    """The DeepSeek-V3 pair exists to attribute KV corruption to a connector.

    That only works if the two cells are otherwise identical, so this pins the
    difference down to the connector rather than trusting the YAML merge key.
    """

    @classmethod
    def setUpClass(cls):
        cells = generate_cells("DeepSeek-V3")
        cls.cells = {cell["env"]["common"]["KV_CONNECTOR"]: cell for cell in cells}

    def test_both_connectors_are_present(self):
        self.assertEqual(set(self.cells), {"mooncake", "moriio"})

    def test_the_pair_differs_only_by_the_connector(self):
        mooncake, moriio = self.cells["mooncake"], self.cells["moriio"]
        # Labels are expected to differ; nothing that reaches a server is.
        labels = {"id", "name", "topology", "display_topology", "log_root"}
        differing = {
            key
            for key in set(mooncake) | set(moriio)
            if mooncake.get(key) != moriio.get(key)
        }
        self.assertEqual(differing - labels, {"env"})
        for role in ("common", "prefill", "decode"):
            without = [
                {k: v for k, v in cell["env"][role].items() if k != "KV_CONNECTOR"}
                for cell in (mooncake, moriio)
            ]
            self.assertEqual(without[0], without[1])

    def test_neither_cell_pins_a_kv_transfer_config(self):
        # An explicit JSON would override the built-in config and defeat the
        # comparison, since only one of the two would be honoured.
        for cell in self.cells.values():
            self.assertIsNone(cell["env"]["prefill"].get("PREFILL_KV_TRANSFER_CONFIG"))
            self.assertIsNone(cell["env"]["decode"].get("DECODE_KV_TRANSFER_CONFIG"))

    def test_cells_are_two_node_tp8_gsm8k_only(self):
        for cell in self.cells.values():
            self.assertTrue(cell["eval_only"])
            self.assertEqual(cell["num_nodes"], 2)
            self.assertEqual(cell["service"]["prefill"]["tp"], 8)
            self.assertEqual(cell["service"]["decode"]["tp"], 8)
            self.assertEqual(cell["accuracy"]["task"], "gsm8k")
            self.assertEqual(cell["accuracy"]["endpoint"], "chat/completions")
            self.assertTrue(cell["accuracy"]["apply_chat_template"])
            # V3 does not think, so gsm8k answers fit well inside this.
            self.assertEqual(cell["accuracy"]["max_gen_toks"], 512)


class KvConnectorTest(unittest.TestCase):
    """KV_CONNECTOR picks the built-in config and guards role overrides."""

    @classmethod
    def setUpClass(cls):
        cls.functions = script_section(
            "prepare_kv_connector() {", 'host_ip="$(echo "${IPADDRS}"'
        )

    def run_shell(self, script, **env):
        base = {
            key: value
            for key, value in os.environ.items()
            if key not in ("KV_CONNECTOR", "ATOM_HOST_IP")
        }
        return subprocess.run(
            ["bash", "-c", "set -euo pipefail\n" + self.functions + script],
            env={**base, **env},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

    def test_built_in_config_defaults_to_mooncake(self):
        result = self.run_shell('kv_transfer_config kv_producer "" 10.0.0.1 6301')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "kv_role": "kv_producer",
                "kv_connector": "mooncake",
                "proxy_ip": "10.0.0.1",
                "handshake_port": 6301,
            },
        )

    def test_built_in_config_uses_the_selected_connector_and_port(self):
        result = self.run_shell(
            'kv_transfer_config kv_consumer "" 10.0.0.2 6309', KV_CONNECTOR="moriio"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "kv_role": "kv_consumer",
                "kv_connector": "moriio",
                "proxy_ip": "10.0.0.2",
                "handshake_port": 6309,
            },
        )

    def test_role_override_is_passed_through_untouched(self):
        override = '{"kv_role":"kv_consumer","kv_connector":"multi","connectors":[]}'
        result = self.run_shell(f"kv_transfer_config kv_consumer '{override}' ip 6301")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, override)

    def test_override_naming_another_connector_fails(self):
        override = '{"kv_role":"kv_consumer","kv_connector":"mooncake"}'
        result = self.run_shell(
            f"kv_transfer_config kv_consumer '{override}' ip 6301",
            KV_CONNECTOR="moriio",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("names 'mooncake'", result.stderr)

    def test_moriio_pins_the_advertised_host_ip(self):
        # get_ip() would otherwise pick whichever address owns the default
        # route, which is not the one the peer role dials.
        result = self.run_shell(
            'host_ip=10.19.0.2; prepare_kv_connector; echo "ip=${ATOM_HOST_IP}"',
            KV_CONNECTOR="moriio",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ip=10.19.0.2", result.stdout)

    def test_explicit_host_ip_is_preserved(self):
        result = self.run_shell(
            'host_ip=10.19.0.2; prepare_kv_connector; echo "ip=${ATOM_HOST_IP}"',
            KV_CONNECTOR="moriio",
            ATOM_HOST_IP="10.9.9.9",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ip=10.9.9.9", result.stdout)

    def test_unknown_connector_fails_before_any_server_starts(self):
        result = self.run_shell("prepare_kv_connector", KV_CONNECTOR="nixl")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported KV_CONNECTOR=nixl", result.stderr)


class ModelPathFallbackTest(unittest.TestCase):
    """Catalog paths name the HF org; some clusters store the weights flat."""

    @classmethod
    def setUpClass(cls):
        cls.functions = script_section(
            "resolve_model_path() {", 'MODEL_PATH="$(resolve_model_path'
        )

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def resolve(self, path):
        body = "set -euo pipefail\n" + self.functions + 'resolve_model_path "$1"'
        result = subprocess.run(
            ["bash", "-c", body, "_", path],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout

    def test_org_prefixed_path_falls_back_to_the_flat_layout(self):
        (self.root / "Kimi-K2.5-MXFP4").mkdir()
        flat = self.resolve(f"{self.root}/amd/Kimi-K2.5-MXFP4/")
        self.assertEqual(flat, f"{self.root}/Kimi-K2.5-MXFP4")

    def test_existing_path_is_used_as_is(self):
        (self.root / "amd/GLM-5.2-MXFP4").mkdir(parents=True)
        path = f"{self.root}/amd/GLM-5.2-MXFP4"
        self.assertEqual(self.resolve(path), path)

    def test_unknown_and_empty_paths_are_left_for_the_engine_to_report(self):
        self.assertEqual(self.resolve(""), "")
        missing = f"{self.root}/amd/Nope"
        self.assertEqual(self.resolve(missing), missing)
        # A relative path is a Hugging Face repo id, not a local directory.
        self.assertEqual(self.resolve("amd/Kimi-K2.5-MXFP4"), "amd/Kimi-K2.5-MXFP4")


class SpurSubmitWrapperTest(unittest.TestCase):
    """Only the workflow chmods the scripts, so nothing may need the exec bit."""

    def test_wrapper_runs_the_job_script_through_bash(self):
        source = (SCRIPTS / "pd_submit.sh").read_text()
        self.assertIn("printf 'exec bash %q\\n' \"${JOB_SCRIPT}\"", source)

    def test_job_script_is_not_marked_executable(self):
        # If this ever flips, the wrapper above stops being load-bearing and
        # the 0644 case silently goes untested.
        mode = (SCRIPTS / "pd_slurm_job.sh").stat().st_mode
        self.assertFalse(mode & 0o111, "pd_slurm_job.sh is executable on disk")


class JobResultTest(unittest.TestCase):
    """A job that dies before its run dir exists still has a result to record."""

    def test_resolve_reports_the_scheduler_code_without_a_run_dir(self):
        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / "missing/slurm_job-99"
            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPTS / "pd_job_result.py"),
                    "resolve",
                    "--run-dir",
                    str(run_dir),
                    "--job-id",
                    "99",
                    "--run-token",
                    "token",
                    "--num-ranks",
                    "2",
                    "--scheduler-state",
                    "FAILED",
                    "--scheduler-exit-code",
                    "126:0",
                    "--scheduler-rc",
                    "126",
                    "--spur",
                    "1",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            # A traceback here would mask the real exit code behind rc=1.
            self.assertEqual(result.returncode, 126, result.stderr)
            self.assertIn("result_state=FAILED", result.stdout)
            payload = json.loads((run_dir / "job-result.json").read_text())
        self.assertEqual(payload["scheduler"]["exit_code"], "126:0")
        self.assertEqual(payload["workload"]["state"], "UNVERIFIED")


class Gsm8kThresholdTest(unittest.TestCase):
    """A below-threshold gsm8k score must fail the job, as SWE-bench already does."""

    @classmethod
    def setUpClass(cls):
        marker = (
            'python3 - "${result_dir}" "${eval_conc}" "${EVAL_THRESHOLD}" <<\'PY\'\n'
        )
        source = SERVER_SCRIPT.read_text()
        start = source.index(marker) + len(marker)
        cls.checker = source[start : source.index("\nPY\n", start)]

    def score(self, results, threshold):
        with tempfile.TemporaryDirectory() as temp:
            result_dir = Path(temp) / "20260101_gsm8k_1p1d_c64"
            (result_dir / "model").mkdir(parents=True)
            (result_dir / "model/results_now.json").write_text(
                json.dumps({"results": results})
            )
            return subprocess.run(
                ["python3", "-", str(result_dir), "64", threshold],
                input=self.checker,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )

    def test_score_at_or_above_threshold_passes(self):
        result = self.score({"gsm8k": {"exact_match,flexible-extract": 0.9356}}, "0.90")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[eval] PASS", result.stdout)

    def test_score_below_threshold_fails(self):
        result = self.score({"gsm8k": {"exact_match,flexible-extract": 0.0053}}, "0.90")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[eval] FAIL", result.stdout + result.stderr)

    def test_missing_metric_fails_instead_of_reporting_na(self):
        result = self.score({"gsm8k": {}}, "0.90")
        self.assertEqual(result.returncode, 1)
        self.assertIn("[eval] FAIL", result.stdout + result.stderr)

    def test_no_threshold_keeps_the_run_green(self):
        result = self.score({"gsm8k": {"exact_match,flexible-extract": 0.0053}}, "")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("[eval] FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()

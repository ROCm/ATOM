# SPDX-License-Identifier: MIT
"""Exercise the launch argv, including quoting and the legacy AIPerf path."""

import os
import re
import subprocess
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_server_atom.sh"
)


class AgentXCommandTest(unittest.TestCase):
    def command(self, version):
        source = SCRIPT.read_text()
        builder = source.split("build_agentic_client_command() {", 1)[1].split(
            "\nrun_aiperf_agentic_benchmark() {", 1
        )[0]
        builder = "build_agentic_client_command() {" + builder
        # Populate legacy scalar options; model/path values deliberately contain spaces.
        env = dict(os.environ)
        for name in re.findall(r"\$\{([A-Z_]+)", builder):
            env[name] = "1"
        env.update(
            AIPERF_AGENTX_VERSION=version,
            AIPERF_VENV="/tmp/client env",
            MODEL_PATH="/models/model with spaces",
            MODEL_SERVED_NAME="MiniMaxAI/MiniMax-M3.1-Flash",
            ROUTER_PORT="8000",
            AIPERF_BENCHMARK_DURATION="3600",
            AIPERF_SCENARIO="inferencex-agentx-mvp",
            AIPERF_PUBLIC_DATASET="semianalysis_cc_traces_weka_062126",
        )
        shell = (
            "set -euo pipefail\n"
            + builder
            + "\nserver_metrics_args=(--server-metrics http://p:8010/metrics http://d:8020/metrics)\n"
            + "unsafe_args=(); chat_template_args=(--apply-chat-template)\n"
            + "cmd=(); build_agentic_client_command cmd 48 '/tmp/run results'\n"
            + "printf '%s\\0' \"${cmd[@]}\"\n"
        )
        result = subprocess.run(
            ["bash", "-c", shell], env=env, capture_output=True, check=True
        )
        return result.stdout.decode().rstrip("\0").split("\0")

    def test_agentx_keeps_recipe_defaults_and_scrapes_both_roles(self):
        args = self.command("1.0.12")
        self.assertEqual(args[:2], ["/tmp/client env/bin/agentx", "profile"])
        for option, value in {
            "--model": "MiniMaxAI/MiniMax-M3.1-Flash",
            "--tokenizer": "/models/model with spaces",
            "--concurrency": "48",
            "--benchmark-duration": "3600",
            "--output-artifact-dir": "/tmp/run results",
        }.items():
            self.assertEqual(args[args.index(option) + 1], value)
        index = args.index("--server-metrics")
        self.assertEqual(
            args[index + 1 : index + 3],
            ["http://p:8010/metrics", "http://d:8020/metrics"],
        )
        self.assertNotIn("--scenario", args)
        self.assertNotIn("--trajectory-start-min-ratio", args)
        self.assertNotIn("--warmup-requests-per-lane", args)

    def test_existing_aiperf_command_keeps_its_explicit_scenario(self):
        args = self.command("")
        self.assertEqual(args[:2], ["/tmp/client env/bin/aiperf", "profile"])
        self.assertEqual(args[args.index("--scenario") + 1], "inferencex-agentx-mvp")
        self.assertIn("--warmup-requests-per-lane", args)
        self.assertIn("--apply-chat-template", args)
        self.assertEqual(
            args[args.index("--tokenizer") + 1], "/models/model with spaces"
        )


if __name__ == "__main__":
    unittest.main()

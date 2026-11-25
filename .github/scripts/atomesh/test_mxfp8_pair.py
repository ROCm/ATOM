import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pd_mxfp8_pair
from pd_mxfp8_pair import pair_contract
from pd_srt_functional import native_source_guard


class PairedWorkerTest(unittest.TestCase):
    def test_pair_runs_sequentially_and_stops_when_second_arm_must_yield(self):
        for yield_second in (False, True):
            with (
                self.subTest(yield_second=yield_second),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                repo, output = root / "repo", root / "run"
                context = repo / "context"
                context.mkdir(parents=True)
                (repo / ".sa-srt-runtime").mkdir()
                (output / "bin").mkdir(parents=True)
                (output / "bin/nvcc").write_text("stub")
                body = b"candidate source"
                manifest = {
                    "base_image": "base-image",
                    "module_path": "/worker.py",
                    "base_module_sha256": "a" * 64,
                    "candidate_module_sha256": hashlib.sha256(body).hexdigest(),
                    "validation": {"bitwise_equal_cases": 8, "release_rc": 0},
                }
                (context / "manifest.json").write_text(json.dumps(manifest))
                (context / "Dockerfile").write_text(
                    "FROM base-image\nCOPY rocm_native.py /worker.py\n"
                )
                (context / "rocm_native.py").write_bytes(body)
                base = {
                    "image": "base-image",
                    "case_name": "pair",
                    "paired_worker_image": {"context": "context"},
                    "roles": {"prefill": {"env": {}}, "decode": {"env": {}}},
                }
                (repo / "contract.json").write_text(json.dumps(base))
                calls = []

                def run(args, **kwargs):
                    if args[:2] == ["docker", "build"]:
                        Path(args[args.index("--iidfile") + 1]).write_text(
                            "sha256:" + "c" * 64
                        )
                    if args[-1] == "pair-candidate" and yield_second:
                        raise subprocess.CalledProcessError(1, args)
                    if args[0] == "bash":
                        arm = Path(kwargs["env"]["RUN_DIR"])
                        calls.append(arm.name)
                        self.assertEqual(kwargs["env"]["JOB_ID"], "one-pair-job")
                        (arm / "srt-cleanup-remaining.txt").write_text("")
                    return SimpleNamespace(returncode=0)

                def digest(args, **kwargs):
                    key = (
                        "base_module_sha256"
                        if "base-image" in args
                        else "candidate_module_sha256"
                    )
                    return manifest[key] + " /worker.py\n"

                env = {
                    "REPO_ROOT": str(repo),
                    "RUN_DIR": str(output),
                    "ATOMESH_ENV_ATOMESH_PR5090_CONTRACT": "contract.json",
                    "SRTCTL_JOB_ID": "one-pair-job-sa-functional",
                    "JOB_ID": "one-pair-job",
                }
                with (
                    patch.dict(os.environ, env),
                    patch.object(pd_mxfp8_pair.subprocess, "run", side_effect=run),
                    patch.object(
                        pd_mxfp8_pair.subprocess, "check_output", side_effect=digest
                    ),
                    patch.object(pd_mxfp8_pair.socket, "socket"),
                ):
                    if yield_second:
                        with self.assertRaises(subprocess.CalledProcessError):
                            pd_mxfp8_pair.main()
                    else:
                        self.assertEqual(pd_mxfp8_pair.main(), 0)
                state = json.loads((output / "pair-state.json").read_text())
                self.assertEqual(
                    calls, ["base"] if yield_second else ["base", "candidate"]
                )
                self.assertEqual(state["complete"], not yield_second)

    def test_only_decode_image_and_expected_hash_change_between_arms(self):
        folder = (
            Path(__file__).resolve().parents[3] / ".github/benchmark/pr5090-router-ab"
        )
        original = json.loads(
            (folder / "single-node-pp4-dep4-mxfp8-pair.json").read_text()
        )
        snapshot = copy.deepcopy(original)
        manifest = json.loads((folder / "worker-mxfp8/manifest.json").read_text())
        base = pair_contract(original, "base", "sha256:" + "a" * 64, manifest)
        candidate = pair_contract(original, "candidate", "sha256:" + "a" * 64, manifest)
        self.assertEqual(original, snapshot)
        self.assertEqual(base["roles"]["prefill"], candidate["roles"]["prefill"])
        self.assertNotIn("container", base["roles"]["decode"])
        self.assertEqual(
            candidate["roles"]["decode"].pop("container"), "sha256:" + "a" * 64
        )
        candidate["roles"]["decode"]["env"]["ATOMESH_EXPECTED_NATIVE_SHA"] = manifest[
            "base_module_sha256"
        ]
        candidate["case_name"], candidate["pair_arm"] = (
            base["case_name"],
            base["pair_arm"],
        )
        self.assertEqual(candidate, base)
        self.assertTrue(base["diagnostic_only"])
        self.assertEqual(base["duration"], 900)

    @unittest.skipUnless(shutil.which("bash"), "bash required")
    def test_worker_source_hash_guard_rejects_wrong_image_contents(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source with spaces.py"
            path.write_bytes(b"source sentinel\n")
            module = path.as_posix()
            if os.name == "nt":
                module = "/" + module[0].lower() + module[2:]
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            for digest, code in ((expected, 0), ("0" * 64, 1), ("", 0)):
                result = subprocess.run(
                    [
                        shutil.which("bash"),
                        "-c",
                        "set -euo pipefail\n" + native_source_guard(module),
                    ],
                    env=dict(os.environ, ATOMESH_EXPECTED_NATIVE_SHA=digest),
                    capture_output=True,
                )
                self.assertEqual(result.returncode, code, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()

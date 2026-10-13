"""CPU regressions for RDMA startup failures seen on TW nodes."""

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / ".github/scripts/atomesh"
spec = importlib.util.spec_from_file_location(
    "preflight", SCRIPTS / "pd_rdma_preflight.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class RdmaPreflightTest(unittest.TestCase):
    def make_port(self, root, name, state):
        target = root / name / "ports/1/state"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(state)

    def test_active_and_down_primaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_port(root, "rdma0", "4: ACTIVE")
            self.make_port(root, "rdma1", "1: DOWN")
            self.make_port(root, "ionic_1", "4: ACTIVE")
            records = preflight.inspect_primaries(root, [0, 1, 2])
            self.assertEqual([r["active"] for r in records], [True, False, False])
            self.assertEqual(records[1]["device"], "rdma1")

    def test_ionic_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_port(root, "ionic_0", "4: ACTIVE")
            self.assertTrue(preflight.inspect_primaries(root, [0])[0]["active"])
            self.assertTrue(
                preflight.inspect_primaries(root, [7], "ionic_0")[0]["active"]
            )

    def test_unreadable_state_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_port(root, "rdma0", "4: ACTIVE")
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                record = preflight.inspect_primaries(root, [0])[0]
            self.assertFalse(record["active"])
            self.assertIn("unreadable", record["states"]["1"])

    def test_failed_peer_stops_waiter(self):
        source = (SCRIPTS / "pd_server_atom.sh").read_text()
        function = source.split("check_peer_failures() {", 1)[1].split("\n}\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rank-rc-1").write_text("1\n")
            script = "check_peer_failures() {" + function + "\n}\ncheck_peer_failures"
            result = subprocess.run(
                ["bash", "-c", 'RUN_DIR="$1"\n' + script, "test", str(root)],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("peer worker exited", result.stderr)


if __name__ == "__main__":
    unittest.main()

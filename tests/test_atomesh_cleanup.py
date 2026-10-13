"""Keep recovery cleanup scoped to the selected previous jobs."""

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/atomesh/pd_cleanup.py"
spec = importlib.util.spec_from_file_location("cleanup", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CleanupTest(unittest.TestCase):
    def inventory(self):
        return "\n".join(
            json.dumps({"ID": str(index), "Names": name})
            for index, name in enumerate(
                [
                    "atomesh-cell-5816-0",
                    "atomesh-cell-5816-1-eval",
                    "atomesh-cell-5837-0",
                    "unrelated-5816-0",
                    "atomesh-cell-15816-0",
                ]
            )
        )

    def test_only_explicit_job_names_match(self):
        selected = module.select_containers(self.inventory(), {"5816"})
        self.assertEqual([item["ID"] for item in selected], ["0", "1"])

    def test_empty_selection_does_not_touch_docker(self):
        with patch.object(module.subprocess, "run") as run:
            module.cleanup(set())
        run.assert_not_called()

    def test_failed_cleanup_is_not_silently_ignored(self):
        result = subprocess.CompletedProcess([], 0, self.inventory(), "")
        with (
            patch.object(module.subprocess, "run", return_value=result),
            self.assertRaisesRegex(RuntimeError, "still running"),
        ):
            module.cleanup({"5816"})

    def test_successful_cleanup_only_stops_selected_containers(self):
        present = subprocess.CompletedProcess([], 0, self.inventory(), "")
        gone = subprocess.CompletedProcess([], 0, "", "")
        with patch.object(
            module.subprocess,
            "run",
            side_effect=[present, gone, gone, gone, gone, gone],
        ) as run:
            module.cleanup({"5816"})
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [command[-1] for command in commands if command[1] in ("stop", "rm")],
            ["0", "0", "1", "1"],
        )


if __name__ == "__main__":
    unittest.main()

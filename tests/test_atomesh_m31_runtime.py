# SPDX-License-Identifier: MIT
"""A pinned AITER environment retains dependencies from an image's own venv."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / ".github/scripts/atomesh/setup_m31_aiter.sh"


class M31RuntimeSetupTest(unittest.TestCase):
    def test_nested_venv_retains_image_packages_and_editable_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "image"
            child = root / "job"
            subprocess.run(
                ["python3", "-m", "venv", "--without-pip", str(base)], check=True
            )
            purelib = subprocess.check_output(
                [
                    str(base / "bin/python"),
                    "-c",
                    "import sysconfig; print(sysconfig.get_path('purelib'))",
                ],
                text=True,
            ).strip()
            package_dir = Path(purelib)
            (package_dir / "image_dependency.py").write_text("value = 42\n")
            editable = root / "editable"
            editable.mkdir()
            (editable / "image_editable.py").write_text("value = 17\n")
            (package_dir / "image_editable.pth").write_text(str(editable) + "\n")
            env = dict(os.environ, ATOMESH_M31_AITER_REF="")
            env["PATH"] = str(base / "bin") + os.pathsep + env["PATH"]
            subprocess.run(
                [
                    "bash",
                    "-euc",
                    'source "$1"; create_m31_aiter_venv "$2"',
                    "bash",
                    str(SETUP),
                    str(child),
                ],
                env=env,
                check=True,
            )
            subprocess.run(
                [
                    str(child / "bin/python"),
                    "-c",
                    (
                        "import image_dependency, image_editable, sys; "
                        "assert image_dependency.value == 42; "
                        "assert image_editable.value == 17; "
                        "assert sys.prefix == sys.argv[1]"
                    ),
                    str(child),
                ],
                check=True,
            )

    def test_moving_ref_is_rejected_before_fetch(self):
        result = subprocess.run(
            ["bash", "-euc", 'source "$1"', "bash", str(SETUP)],
            env=dict(os.environ, ATOMESH_M31_AITER_REF="main"),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a full commit SHA", result.stderr)


if __name__ == "__main__":
    unittest.main()

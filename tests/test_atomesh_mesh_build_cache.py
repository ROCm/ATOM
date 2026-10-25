# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""build_mesh.sh recovers from a fallback CARGO_HOME whose unpacked crates were cleaned."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_MESH = ROOT / ".github/scripts/atomesh/build_mesh.sh"

# Stands in for cargo when the image's Rust is unusable: `--version` works only
# with the fallback CARGO_HOME. A build fails the way cargo does when a crate's
# .cargo-ok marker survived but its files did not, or with any FAIL_WITH error.
FAKE_CARGO = """
import os, sys
from pathlib import Path
args = sys.argv[1:]
if args == ['--version']:
    sys.exit(0 if os.environ.get('CARGO_HOME') else 1)
with open(os.environ['TEST_CARGO_CALLS'], 'a') as calls:
    calls.write('build\\n')
if os.environ.get('FAIL_WITH'):
    print(os.environ['FAIL_WITH'], file=sys.stderr)
    sys.exit(101)
for crate in Path(os.environ['CARGO_HOME'], 'registry', 'src').glob('*/*'):
    if (crate / '.cargo-ok').exists() and not (crate / 'Cargo.toml').exists():
        print(f"error: failed to read `{crate / 'Cargo.toml'}`", file=sys.stderr)
        sys.exit(101)
out = Path(args[args.index('--target-dir') + 1]) / args[args.index('--profile') + 1]
out.mkdir(parents=True, exist_ok=True)
(out / 'atomesh').write_text('binary')
(out / 'atomesh').chmod(0o755)
"""


class FallbackCargoHomeTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.tmp = Path(directory.name)
        source = self.tmp / "source"
        source.mkdir()
        (source / "Cargo.toml").write_text("[package]\n")
        (source / "Cargo.lock").write_text("lock\n")
        proto = self.tmp / "model_engine" / "protocol" / "engine_v1.proto"
        proto.parent.mkdir(parents=True)
        proto.write_text('syntax = "proto3";\n')
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        cargo = bin_dir / "cargo"
        cargo.write_text("#!" + sys.executable + "\n" + FAKE_CARGO)
        cargo.chmod(0o755)
        self.cache = self.tmp / "cache"
        self.crate = (
            self.cache
            / "cargo-home/registry/src/index.crates.io-1949cf8c6b5b557f/adler2-2.0.1"
        )
        self.calls = self.tmp / "cargo-calls"
        env = {k: v for k, v in os.environ.items() if k != "CARGO_HOME"}
        self.env = {
            **env,
            "PATH": str(bin_dir) + os.pathsep + env["PATH"],
            "TMPDIR": str(self.tmp),
            "ATOMESH_MESH_TARGET_DIR": str(self.cache),
            "ATOMESH_MESH_SOURCE_COMMIT": "reviewed-commit",
            "TEST_CARGO_CALLS": str(self.calls),
        }
        self.command = ["bash", str(BUILD_MESH), str(source), str(self.tmp / "logs")]

    def build(self, **env):
        return subprocess.run(
            self.command,
            env={**self.env, **env},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    def builds(self):
        return self.calls.read_text().count("build")

    def test_cleaned_crate_sources_are_unpacked_again_and_the_build_retried(self):
        self.crate.mkdir(parents=True)
        (self.crate / ".cargo-ok").write_text('{"v":1}')
        (self.crate / "src").mkdir()
        result = self.build()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("unpacking them again", result.stderr)
        self.assertEqual(self.builds(), 2)
        self.assertFalse((self.cache / "cargo-home/registry/src").exists())
        self.assertTrue((self.tmp / "logs" / "atomesh").is_file())

    def test_other_build_failures_are_not_retried(self):
        self.crate.mkdir(parents=True)
        (self.crate / ".cargo-ok").write_text('{"v":1}')
        (self.crate / "Cargo.toml").write_text("[package]\n")
        result = self.build(FAIL_WITH="error[E0425]: cannot find value `x`")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("E0425", result.stderr)
        self.assertEqual(self.builds(), 1)
        self.assertTrue((self.crate / "Cargo.toml").exists())


if __name__ == "__main__":
    unittest.main()

"""CPU-only checks for paired runtime installation and rollback."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "docker/rocm-runtime/switch_runtime.py"
SPEC = importlib.util.spec_from_file_location("switch_runtime", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runtime)


class RuntimeSwitchTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runtime"
        self.rocm = Path(self.temporary.name) / "rocm"
        self.lib = self.rocm / "lib"
        self.patched = self.root / "patched/lib"
        self.lib.mkdir(parents=True)
        self.patched.mkdir(parents=True)
        (self.root / "build-info.json").write_text('{"enabled": true}')
        for name, soname in zip(runtime.FAMILIES, runtime.SONAMES):
            original = self.lib / (soname + ".original")
            original.write_bytes(("stock " + name).encode())
            # Exercise absolute aliases, which must not point back into the live
            # SDK once they have been backed up.
            (self.lib / soname).symlink_to(original)
            (self.lib / name).symlink_to(soname)
            replacement = self.patched / (soname + ".patched")
            replacement.write_bytes(("patched " + name).encode())
            (self.patched / soname).symlink_to(replacement.name)
            (self.patched / name).symlink_to(soname)
        (self.lib / "librccl.so").write_bytes(b"leave RCCL alone")
        self.original = {p.name: p.read_bytes() for p in runtime.libraries(self.lib)}

    def select(self, mode):
        runtime.select_runtime(self.root, self.rocm, mode)

    def test_patched_aliases_and_repeated_stock_roundtrip(self):
        for _ in range(2):
            self.select("patched")
            for name, soname in zip(runtime.FAMILIES, runtime.SONAMES):
                aliases = [p for p in runtime.libraries(self.lib) if p.name.startswith(name)]
                self.assertTrue(all(p.read_bytes() == ("patched " + name).encode()
                                    for p in aliases))
                self.assertEqual(len({p.stat().st_ino for p in aliases}), 1)
                self.assertTrue((self.lib / (soname + ".original")).exists())
            self.select("stock")
            self.assertEqual({p.name: p.read_bytes() for p in runtime.libraries(self.lib)},
                             self.original)
        self.assertEqual((self.lib / "librccl.so").read_bytes(), b"leave RCCL alone")
        self.assertEqual((self.root / "active").read_text().strip(), "stock")

    def test_missing_half_of_pair_does_not_modify_original(self):
        (self.patched / runtime.SONAMES[1]).unlink()
        with self.assertRaises(RuntimeError):
            self.select("patched")
        self.assertEqual({p.name: p.read_bytes() for p in runtime.libraries(self.lib)},
                         self.original)
        self.assertFalse((self.root / "stock").exists())

    def test_corrupt_backup_is_rejected_before_switch(self):
        self.select("patched")
        before = {p.name: p.read_bytes() for p in runtime.libraries(self.lib)}
        (self.root / "stock/lib" / runtime.SONAMES[0]).write_bytes(b"corrupt")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            self.select("stock")
        self.assertEqual({p.name: p.read_bytes() for p in runtime.libraries(self.lib)}, before)

    def test_stock_rollback_does_not_require_intact_patched_pair(self):
        self.select("patched")
        (self.patched / runtime.SONAMES[1]).unlink()
        self.select("stock")
        self.assertEqual({p.name: p.read_bytes() for p in runtime.libraries(self.lib)},
                         self.original)

    def test_disabled_rebuild_restores_inherited_stock_backup(self):
        self.select("patched")
        (self.root / "build-info.json").write_text('{"enabled": false, "restore_stock": true}')
        self.select("patched")
        self.assertEqual({p.name: p.read_bytes() for p in runtime.libraries(self.lib)},
                         self.original)
        self.assertEqual((self.root / "active").read_text().strip(), "stock")

    def test_stock_without_backup_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "No stock runtime backup"):
            self.select("stock")

    def test_disabled_build_preserves_original(self):
        (self.root / "build-info.json").write_text(json.dumps(dict(enabled=False)))
        self.select("patched")
        self.assertEqual({p.name: p.read_bytes() for p in runtime.libraries(self.lib)},
                         self.original)
        self.assertFalse((self.root / "stock").exists())


if __name__ == "__main__":
    unittest.main()

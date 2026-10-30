"""Check which pinned KDA loading fix the in-container verifier accepts."""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_c48_port_patch import RELATIVE_PATH, verify_port_patch
from pd_c48_quant_patch import verify_quant_patch

GAP = Path(__file__).resolve().parents[3] / "benchmark/c48-model-gap"


def fake_vllm(temp, kda_dir):
    root = Path(temp)
    (root / "models/kimi_k3/amd").mkdir(parents=True)
    (root / "model_executor/model_loader/reload").mkdir(parents=True)
    shutil.copy(GAP / kda_dir / "kda.py", root / "models/kimi_k3/amd/kda.py")
    shutil.copy(GAP / "online-quant-fix/meta.py", root / "model_executor/model_loader/reload/meta.py")
    return root


class QuantPatchTest(unittest.TestCase):
    def test_port_patch_accepts_both_engines(self):
        for source, engine in ((GAP / "moriio_common.py", "81198e97ba7eee2a22540caaa756b7fdddcb4d93"),
                               (GAP / "n3966924d/moriio_common.py", "3966924d566340e971a6da313e790d05492aa505")):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / RELATIVE_PATH
                path.parent.mkdir(parents=True)
                shutil.copy(source, path)
                self.assertEqual(verify_port_patch(Path(temp))["base_engine_sha"], engine)
                with open(path, "a") as f:
                    f.write("\n")
                with self.assertRaisesRegex(RuntimeError, "prepared read-only source mount"):
                    verify_port_patch(Path(temp))

    def test_both_pinned_kda_files_pass_digest_check(self):
        # Outside the container nothing is bind-mounted, so a matching digest
        # surfaces as the read-only mount error instead of "fix missing".
        for kda_dir in ("online-quant-fix", "inproj-pad128", "n3966924d/inproj-pad128"):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaisesRegex(RuntimeError, "read-only mount"):
                    verify_quant_patch(fake_vllm(temp, kda_dir))

    def test_unpinned_kda_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = fake_vllm(temp, "inproj-pad128")
            with open(root / "models/kimi_k3/amd/kda.py", "a") as f:
                f.write("\n")
            with self.assertRaisesRegex(RuntimeError, "fix missing"):
                verify_quant_patch(root)


if __name__ == "__main__":
    unittest.main()

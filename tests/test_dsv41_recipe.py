"""CPU checks for source selection and native/candidate process isolation."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dsv41_recipe", ROOT / "tools/dsv41/run.py"
)
recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recipe)


class RecipeTests(unittest.TestCase):
    def env(self, mode):
        return recipe.environment(
            SimpleNamespace(
                mode=mode,
                output=Path("/tmp/new-evidence"),
                model_source=Path("/tmp/subject"),
                aiter_source=Path("/tmp/aiter"),
                label="check",
                action="benchmark",
                cache_dir=None,
            )
        )

    def test_native_is_rebased_main_configuration(self):
        env = self.env("native")
        for key in (
            "DSV41_OWNER_SERVING",
            "DSV41_STEP12_OWNER_EP4",
            "DSV41_OWNER_ACCEPTED_PREFILL",
            "ATOM_DSV41_ROW_SHARDED_MHC",
            "ATOM_DSV41_EP_PROFILED_PLACEMENT",
            "ATOM_DSV41_PREFILL_QUANTIZED_FFN",
        ):
            self.assertEqual(env[key], "0", key)
        self.assertEqual(env["ATOM_DSV41_PREFILL_TAIL_MODE"], "off")

    def test_candidate_preserves_complete_package(self):
        env = self.env("candidate")
        for key in (
            "DSV41_OWNER_SERVING",
            "DSV41_STEP12_OWNER_EP4",
            "DSV41_OWNER_STABLE_GRAPHS",
            "ATOM_DSV41_EP_PROFILED_PLACEMENT",
            "ATOM_DSV41_PREFILL_LOCAL_ROUTER",
            "ATOM_DSV41_PREFILL_ROUTE_PACKET",
            "ATOM_DSV41_PREFILL_QUANTIZED_FFN",
        ):
            self.assertEqual(env[key], "1", key)
        self.assertEqual(env["ATOM_DSV41_PREFILL_TAIL_MODE"], "settled")

    def test_outputs_and_imports_do_not_reuse_an_experiment(self):
        env = self.env("candidate")
        for key in (
            "DSV41_OWNER_SERVING_OUTPUT",
            "DSV41_OWNER_SERVING_CONTROL",
            "DSV41_OWNER_GRAPH_PHASE_FILE",
            "DSV41_COMGR_RECEIPT_DIR",
        ):
            self.assertTrue(env[key].startswith("/tmp/new-evidence"), key)
        self.assertEqual(
            env["PYTHONPATH"].split(":"),
            [str(recipe.HERE / "acceptance"), "/tmp/subject", "/tmp/aiter"],
        )
        self.assertEqual(env["DSV41_REBASE_SERVICE_SERVER"], "0")


if __name__ == "__main__":
    unittest.main()

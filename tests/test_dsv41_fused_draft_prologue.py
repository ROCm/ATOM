"""CPU routing contracts for the opt-in draft prologue; no GPU imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "atom/models/deepseek_v41_plugin/dspark.py"


def load_method(name, **scope):
    tree = ast.parse(MODEL.read_text())
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    exec(  # noqa: S102 - repository guard only, without GPU imports
        compile(ast.Module(body=[node], type_ignores=[]), str(MODEL), "exec"), scope
    )
    return scope[name]


class PrologueRoutingTest(unittest.TestCase):
    def test_default_off_and_local_embedding_only(self):
        options = SimpleNamespace(fused_prologue=False)
        host = SimpleNamespace(dspark_options=lambda: options)
        guard = load_method("_fused_prologue_ok", plugin_host=host)
        weight = SimpleNamespace(is_cuda=True, dim=lambda: 2, stride=lambda _: 1)
        model = SimpleNamespace(embed=SimpleNamespace(weight=weight, tp_size=1))
        self.assertFalse(guard(model))
        options.fused_prologue = True
        self.assertTrue(guard(model))
        for tp in (0, 2, 8):
            model.embed.tp_size = tp
            self.assertFalse(guard(model))
        model.embed.tp_size = 1
        weight.is_cuda = False
        self.assertFalse(guard(model))
        weight.is_cuda = True
        weight.stride = lambda _: 2
        self.assertFalse(guard(model))
        self.assertFalse(guard(SimpleNamespace()))
        self.assertFalse(guard(SimpleNamespace(embed=None)))

    def test_flag_defaults_to_off(self):
        from atom import plugin_recipe

        self.assertFalse(plugin_recipe.DSparkOptions().fused_prologue)
        doc = {
            "schema": plugin_recipe.RECIPE_SCHEMA_V2,
            "arch": "DeepseekV41ForCausalLM",
            "dspark_options": {"fused_prologue": True},
        }
        self.assertTrue(plugin_recipe.resolve([doc]).dspark_options.fused_prologue)

    def test_state_is_optional_and_both_call_paths_remain(self):
        tree = ast.parse(MODEL.read_text())
        hidden = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "draft_hidden"
        )
        defaults = dict(
            zip([a.arg for a in hidden.args.kwonlyargs], hidden.args.kw_defaults)
        )
        self.assertIsNone(ast.literal_eval(defaults["state"]))
        backbone = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "block_backbone"
        )
        names = {
            n.func.id
            for n in ast.walk(backbone)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        self.assertTrue({"fused_draft_prologue", "draft_step_indices"} <= names)
        text = ast.unparse(hidden)
        self.assertIn("if state is None:", text)
        self.assertIn("self.embed(anchor_ids)", text)
        self.assertIn("SinglePassHCState(*state)", text)


if __name__ == "__main__":
    unittest.main()

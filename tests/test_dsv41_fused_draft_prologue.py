"""CPU routing contracts for the opt-in draft prologue; no GPU imports."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "atom/models/deepseek_v41/dspark.py"


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
        env = SimpleNamespace(ATOM_DSPARK_FUSED_PROLOGUE=False)
        guard = load_method("_fused_prologue_ok", envs=env)
        weight = SimpleNamespace(is_cuda=True, dim=lambda: 2, stride=lambda _: 1)
        model = SimpleNamespace(embed=SimpleNamespace(weight=weight, tp_size=1))
        self.assertFalse(guard(model))
        env.ATOM_DSPARK_FUSED_PROLOGUE = True
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
        text = (ROOT / "atom/utils/envs.py").read_text()
        self.assertIn('os.getenv("ATOM_DSPARK_FUSED_PROLOGUE", "0") == "1"', text)

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

import ast
import pathlib

from atom.model_engine.block_manager import BlockManager


def test_every_block_manager_method_the_scheduler_calls_exists():
    """A guard restored from an earlier revision called
    `BlockManager.cancel_state_load`, a method this branch had deleted -- an
    AttributeError on the state-only admission path, which unit tests could not
    reach because `test_scheduler.py` does not import in this environment.
    Static check instead: cheap, and it does not need the scheduler to run.
    """
    # Anchored on this file, not on the CWD: nothing in the tree pins pytest's
    # rootdir, so a relative path made the test silently vacuous from any other
    # directory.
    root = pathlib.Path(__file__).resolve().parents[1]
    src = (root / "atom" / "model_engine" / "scheduler.py").read_text()
    called = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not isinstance(fn, ast.Attribute):
            continue
        obj = fn.value
        # self.block_manager.<name>(...)  /  bm.<name>(...)
        if (isinstance(obj, ast.Attribute) and obj.attr == "block_manager") or (
            isinstance(obj, ast.Name) and obj.id == "bm"
        ):
            called.add(fn.attr)
    assert called, "found no block_manager calls; the matcher is broken"
    missing = sorted(n for n in called if not hasattr(BlockManager, n))
    assert (
        not missing
    ), f"scheduler calls BlockManager methods that do not exist: {missing}"

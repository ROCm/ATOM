"""CPU checks for graph retirement before communicator teardown."""

import ast
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
SERVICE = ROOT / "atom/models/deepseek_v41/owner_service"
spec = importlib.util.spec_from_file_location(
    "owner_lifecycle", SERVICE / "lifecycle.py"
)
lifecycle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lifecycle)


class OwnerLifecycleTests(unittest.TestCase):
    def test_exit_orders_all_ranks_before_native_and_installs_once(self):
        events = []

        class Runner:
            still_running = True

            def exit(self):
                events.append("native")
                self.still_running = False
                return "result"

        group = SimpleNamespace(cpu_group=object())
        dist = SimpleNamespace(
            barrier=lambda **kw: events.append(("barrier", kw["group"]))
        )
        modules = {
            "torch": SimpleNamespace(
                distributed=dist,
                cuda=SimpleNamespace(synchronize=lambda: events.append("sync")),
            ),
            "torch.distributed": dist,
            "aiter.dist.parallel_state": SimpleNamespace(get_tp_group=lambda: group),
            "atom.model_engine.model_runner": SimpleNamespace(ModelRunner=Runner),
            "atom.utils.graph_holders": SimpleNamespace(
                release_registered_graphs=lambda: events.append("release")
            ),
        }
        with (
            patch.dict("sys.modules", modules),
            patch.object(lifecycle.gc, "collect", lambda: events.append("collect")),
        ):
            lifecycle.install()
            installed = Runner.exit
            lifecycle.install()
            self.assertIs(Runner.exit, installed)
            runner = Runner()
            self.assertEqual(runner.exit(), "result")
            self.assertEqual(
                events,
                ["sync", "release", "collect", ("barrier", group.cpu_group), "native"],
            )
            events.clear()
            runner.exit()
            self.assertEqual(events, ["native"])

    def test_owner_graphs_reset_before_pool_release_and_preserve_seal(self):
        path = SERVICE / "owner_serving.py"
        cls = next(
            node
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.ClassDef) and node.name == "OwnerServing"
        )
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "release_graphs"
        )
        namespace = {}
        exec(  # noqa: S102 - actual method, without importing the GPU stack
            compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
            namespace,
        )
        events = []
        owner = SimpleNamespace(graphs={}, graph_pool=object(), sealed=True)

        def reset():
            self.assertIsNotNone(owner.graph_pool)
            events.append("reset")

        owner.graphs = {
            key: (SimpleNamespace(reset=reset), None, None) for key in (1, 2)
        }
        self.assertEqual(namespace["release_graphs"](owner), 2)
        self.assertEqual(events, ["reset", "reset"])
        self.assertIsNone(owner.graph_pool)
        self.assertEqual(owner.graphs, {})
        self.assertTrue(owner.sealed)
        self.assertEqual(namespace["release_graphs"](owner), 0)


if __name__ == "__main__":
    unittest.main()

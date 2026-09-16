# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The engine-core managers must agree on the state their output thread reads.

`CoreManager` spawns its engines one way and `DisaggCoreManager` another, so
the subclass cannot run the base `__init__`. It used to hand-copy the field
block instead, and the copy drifted: `_flush_stream_batch_fn` was added to the
copy and to the API server that assigns it, but not to the base class. Nothing
failed loudly -- the server path assigns the hook before the first request and
the disagg path had it from the copy, so only the offline entrypoint was left
without it. There, the output thread raised `AttributeError` on the first
streamed token and died, the engine kept producing tokens nobody collected,
and CI reported it a day later as a 60-minute timeout.

These tests pin the invariant rather than that one field: whatever the output
thread reads off `self`, both managers must provide.
"""

import ast
import pathlib
import unittest
from types import SimpleNamespace

from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

MODULE_PATH = pathlib.Path(CoreManager.__module__.replace(".", "/") + ".py")
SOURCE = pathlib.Path(__file__).resolve().parent.parent / MODULE_PATH
TREE = ast.parse(SOURCE.read_text())


def _self_attributes_read_by(func_name: str) -> set[str]:
    """Names read off `self` inside `func_name`, wherever it is nested.

    Derived from the source rather than hardcoded so that a newly added read
    is covered the day it lands -- which is the whole failure mode here.
    """
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return {
                n.attr
                for n in ast.walk(node)
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Name)
                and n.value.id == "self"
                and isinstance(n.ctx, ast.Load)
            }
    raise AssertionError(f"{func_name} not found in {SOURCE}")


def _calls_shared_initialiser(cls) -> bool:
    """Whether `cls.__init__` routes through `_init_shared_state`."""
    for node in ast.walk(TREE):
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__init__":
                    return any(
                        isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Attribute)
                        and c.func.attr == "_init_shared_state"
                        for c in ast.walk(fn)
                    )
    raise AssertionError(f"{cls.__name__}.__init__ not found")


def _bare_manager(cls, **kwargs):
    """A manager with its shared state set up and no engine processes spawned."""
    mgr = cls.__new__(cls)
    mgr._init_shared_state(
        SimpleNamespace(dp_load_balance="round_robin"),
        label=cls.__name__,
        local_engine_count=kwargs.get("local_engine_count", 2),
    )
    return mgr


class OutputThreadStateTest(unittest.TestCase):
    """Fields the output thread reads must exist before the thread starts."""

    def setUp(self):
        self.managers = [_bare_manager(CoreManager), _bare_manager(DisaggCoreManager)]
        self.addCleanup(self._terminate)

    def _terminate(self):
        for mgr in self.managers:
            mgr.ctx.term()

    def test_every_attribute_the_output_thread_reads_is_initialised(self):
        reads = _self_attributes_read_by("process_outputs_socket")
        self.assertIn(
            "_flush_stream_batch_fn",
            reads,
            "the regression this guards is gone; retarget or drop this test",
        )
        for mgr in self.managers:
            for attr in sorted(reads):
                # Methods resolve on the class, so only instance state is at
                # risk of being missed by an initialiser.
                if callable(getattr(type(mgr), attr, None)):
                    continue
                self.assertTrue(
                    hasattr(mgr, attr),
                    f"{type(mgr).__name__} never initialises self.{attr}, which "
                    f"process_outputs_socket reads -- the output thread will "
                    f"die on the first output and take every response with it",
                )

    def test_stream_flush_hook_defaults_to_absent(self):
        """It stays None until the API server resolves it; offline never does."""
        for mgr in self.managers:
            self.assertIsNone(mgr._flush_stream_batch_fn)

    def test_load_accounting_is_sized_for_the_engine_count(self):
        mgr = _bare_manager(CoreManager, local_engine_count=4)
        self.addCleanup(mgr.ctx.term)
        self.assertEqual(mgr._rank_reqs, [0] * 4)
        self.assertEqual(mgr._rank_tokens, [0] * 4)
        self.assertEqual(mgr.local_engine_count, 4)


class SharedInitialiserIsTheOnlyPathTest(unittest.TestCase):
    """Neither manager may go back to hand-copying the field block."""

    def test_both_managers_call_the_shared_initialiser(self):
        for cls in (CoreManager, DisaggCoreManager):
            self.assertTrue(
                _calls_shared_initialiser(cls),
                f"{cls.__name__}.__init__ must call _init_shared_state rather "
                f"than assigning the shared fields itself; a second copy of "
                f"that block is what drifted last time",
            )

    def test_subclass_does_not_override_the_shared_initialiser(self):
        self.assertIs(
            DisaggCoreManager._init_shared_state,
            CoreManager._init_shared_state,
            "overriding _init_shared_state reintroduces the two-copies problem",
        )


if __name__ == "__main__":
    unittest.main()

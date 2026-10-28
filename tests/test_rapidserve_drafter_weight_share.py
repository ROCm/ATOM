# SPDX-License-Identifier: MIT
"""Sharing the DRAFTER's weights across a rapidserve pair.

The target has always travelled by CUDA IPC: decode builds it on meta and
`import_model_weight_ipc_handles` fills every parameter from prefill's pages.
The drafter did not. It was loaded from the checkpoint on BOTH sides, so a GPU
already sharing weights, KV and per-request state still held two copies of the
draft model -- and decode's copy is part of what prefill's `non_torch` term
measures, which is what sizes down the KV pool.

Two things make the drafter different from the target, and both are the reason
this needed its own path rather than one more argument to the existing one:

- its tensors are REAL on decode, not meta, so the import replaces them and the
  originals are reclaimed; and
- it owns mutable scratch as well as weights. Some DSpark flavours register a
  private rolling target-KV window per layer (`reset_kv_cache`), and aliasing
  that would have both processes writing one buffer. Only parameters are
  shared.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
import torch

RUNNER = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/model_runner.py"
)


def _method(owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(RUNNER.read_text(), filename=str(RUNNER))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


def _src(owner: str, name: str) -> str:
    out = ast.get_source_segment(RUNNER.read_text(), _method(owner, name))
    assert out is not None
    return out


# ── Only the weights travel, never the scratch ───────────────────────────


class TestBuffersAreNotShared:
    """A drafter's private rolling KV window must stay private.

    `reset_kv_cache` is a no-op on DeepSeek-V4 DSpark -- its draft KV is paged
    into the shared pool -- but the sibling flavour allocates a real per-layer
    ring, and nothing in the export can tell a weight from scratch by shape.
    """

    class _Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.buffer_walks = 0

        def named_buffers(self, *a, **kw):
            self.buffer_walks += 1
            return iter(())

    def test_buffers_are_walked_by_default(self):
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        m = self._Spy()
        ipc.export_model_weight_handles(m)
        assert m.buffer_walks == 1

    def test_include_buffers_false_skips_them_entirely(self):
        """Not filtered afterwards -- never walked. A drafter's scratch must
        not even be offered a handle."""
        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        m = self._Spy()
        ipc.export_model_weight_handles(m, include_buffers=False)
        assert m.buffer_walks == 0

    def test_the_drafter_is_exported_without_buffers(self):
        src = _src("RapidServeModelRunner", "export_model_weight_ipc_handles")
        drafter_call = src[src.index('payload["drafter"]') :]
        assert "include_buffers=False" in drafter_call

    def test_the_target_still_gets_its_buffers(self):
        """The target has no private scratch -- it is the model both sides run
        -- and its buffers (RoPE caches and the like) are wanted."""
        src = _src("RapidServeModelRunner", "export_model_weight_ipc_handles")
        target_call = src[src.index('"target":') : src.index('payload["drafter"]')]
        assert "include_buffers" not in target_call


# ── The two modules cannot share one key space ───────────────────────────


def test_target_and_drafter_travel_under_separate_keys():
    """Both name their tensors `model.layers.N...`. Merged into one dict the
    target's handles would silently overwrite the draft's, and decode would
    alias draft parameters onto target storage -- same shapes, wrong values,
    no error anywhere."""
    src = _src("RapidServeModelRunner", "export_model_weight_ipc_handles")
    assert '"target":' in src
    assert 'payload["drafter"]' in src

    imp = _src("RapidServeModelRunner", "import_model_weight_ipc_handles")
    assert 'payload["target"]' in imp
    assert 'payload.get("drafter")' in imp


# ── One accessor, so export and import cannot disagree ───────────────────


class TestWeightRoot:
    def test_it_unwraps_a_compiled_module(self):
        """`torch.compile` prefixes every parameter name with `_orig_mod.`,
        and the halves do not necessarily wrap alike -- prefill forces
        `enforce_eager`. Keys that disagree intersect with nothing, and the
        import is silent about replacing nothing."""
        mr = pytest.importorskip("atom.model_engine.model_runner")

        inner = torch.nn.Linear(2, 2)
        wrapper = torch.nn.Module()
        wrapper._orig_mod = inner
        assert mr.RapidServeModelRunner._ipc_weight_root(wrapper) is inner

    def test_a_plain_module_passes_through(self):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        m = torch.nn.Linear(2, 2)
        assert mr.RapidServeModelRunner._ipc_weight_root(m) is m

    def test_both_sides_go_through_it(self):
        """The whole point of the accessor: one definition of "the drafter"."""
        exp = _src("RapidServeModelRunner", "export_model_weight_ipc_handles")
        imp = _src("RapidServeModelRunner", "_import_drafter_weights")
        assert "_ipc_drafter_root()" in exp
        assert "_ipc_drafter_root()" in imp


class TestDrafterRoot:
    @staticmethod
    def _runner(drafter=None):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        obj = object.__new__(mr.RapidServeModelRunner)
        obj.rank = 0
        if drafter is not None:
            obj.drafter = drafter
        return obj

    def test_none_without_a_drafter(self):
        assert self._runner()._ipc_drafter_root() is None

    def test_none_when_the_drafter_has_no_model(self):
        assert (
            self._runner(drafter=type("D", (), {"model": None})())._ipc_drafter_root()
            is None
        )

    def test_it_unwraps_a_compiled_drafter(self):
        inner = torch.nn.Linear(2, 2)
        wrapper = torch.nn.Module()
        wrapper._orig_mod = inner
        runner = self._runner(drafter=type("D", (), {"model": wrapper})())
        assert runner._ipc_drafter_root() is inner


# ── The import reports whether it actually took ──────────────────────────


class TestImportIsVerified:
    @staticmethod
    def _runner(drafter_model=None):
        mr = pytest.importorskip("atom.model_engine.model_runner")

        obj = object.__new__(mr.RapidServeModelRunner)
        obj.rank = 3
        if drafter_model is not None:
            obj.drafter = type("D", (), {"model": drafter_model})()
        return obj

    def test_no_handles_is_a_no_op(self):
        """A pair running without speculation ships no drafter section."""
        self._runner(torch.nn.Linear(2, 2))._import_drafter_weights(None)

    def test_handles_with_no_local_drafter_warns(self, caplog):
        """Prefill speculating while decode does not is a configuration fault,
        not something to swallow."""
        import logging

        with caplog.at_level(logging.WARNING, logger="atom"):
            self._runner()._import_drafter_weights({"__param__w": object()})
        assert "no drafter" in caplog.text

    def test_replacing_nothing_is_reported_not_assumed(self, caplog, monkeypatch):
        """THE failure this guards. `import_model_weights` looks each handle
        name up in the consumer's module; a key space that does not line up
        matches nothing, replaces nothing and raises nothing. Decode then keeps
        its checkpoint-loaded copy -- the same VALUES, so no output is wrong --
        and the GPU quietly holds two sets of draft weights, which is exactly
        the thing this feature exists to stop."""
        import logging

        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        # `_import_drafter_weights` imports this at call time, so patching the
        # module attribute is what the real call will see.
        monkeypatch.setattr(ipc, "import_model_weights", lambda *a, **kw: None)

        runner = self._runner(torch.nn.Linear(4, 4))
        with caplog.at_level(logging.WARNING, logger="atom"):
            runner._import_drafter_weights({"__param__weight": object()})
        assert "replaced NOTHING" in caplog.text

    def test_a_real_replacement_is_reported(self, caplog, monkeypatch):
        import logging

        ipc = pytest.importorskip("atom.model_engine.ipc_utils")

        def _swap(module, handles):
            # Stand in for the IPC alias: replace the parameter with a
            # different allocation, which is what moves the data pointer.
            module.weight = torch.nn.Parameter(torch.zeros_like(module.weight))

        monkeypatch.setattr(ipc, "import_model_weights", _swap)

        runner = self._runner(torch.nn.Linear(4, 4))
        with caplog.at_level(logging.INFO, logger="atom"):
            runner._import_drafter_weights({"__param__weight": object()})
        assert "drafter aliased" in caplog.text
        assert "replaced NOTHING" not in caplog.text


# ── Ordering against the existing rebind ─────────────────────────────────


def _call_line(owner: str, method: str, attr: str) -> int:
    """Line of a real call to `self.<attr>` inside <owner>.<method>.

    On the AST, never on the source text: this method's docstring and comments
    both NAME the calls being ordered, so a substring search matches prose and
    passes whatever the code does. That is not hypothetical -- the first
    version of these two tests compared a docstring mention at offset 899
    against a comment at 1693 and would have passed with the calls in either
    order.
    """
    fn = _method(owner, method)
    lines = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == attr
    ]
    assert lines, f"no call to {attr} in {owner}.{method}"
    return min(lines)


def test_the_drafter_import_precedes_the_shared_rebind():
    """The drafter's embed/head are bound to the TARGET's module objects, and
    `_rebind_drafter_shared_weights_after_import` is what re-points them at the
    storage this import installed -- and the only thing that checks the binding
    landed. It has to have the last word, so the drafter import runs first.
    """
    owner, method = "RapidServeModelRunner", "import_model_weight_ipc_handles"
    assert _call_line(owner, method, "_import_drafter_weights") < _call_line(
        owner, method, "_rebind_drafter_shared_weights_after_import"
    )


def test_the_drafter_import_precedes_the_collect():
    """The replaced copies are freed by that collect; after it they would sit
    in the allocator until the next one -- which is the memory this feature
    exists to return."""
    fn = _method("RapidServeModelRunner", "import_model_weight_ipc_handles")
    collect = min(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "collect"
    )
    assert (
        _call_line(
            "RapidServeModelRunner",
            "import_model_weight_ipc_handles",
            "_import_drafter_weights",
        )
        < collect
    )

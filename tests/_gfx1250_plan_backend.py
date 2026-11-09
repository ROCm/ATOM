# SPDX-License-Identifier: MIT
"""A plain-integer stand-in for FlyDSL's traced Int32 (``fx.max`` / ``fx.min`` / ``I32`` / ``cmp.select``), so the CPU
tests run the gfx1250 step kernel's plan functions (kernels/index_plan.py, kernels/moe_plan.py) on Python ints.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

KERNELS = (
    Path(__file__).resolve().parents[1]
    / "atom/models/deepseek_v41/mono/gfx1250_single/kernels"
)


class B:
    """A comparison result: ``.select`` and the bitwise and / or the kernel code uses on traced booleans."""

    def __init__(self, v):
        self.v = bool(v)

    def select(self, a, b):
        return a if self.v else b

    def __and__(self, o):
        return B(self.v and _b(o))

    def __or__(self, o):
        return B(self.v or _b(o))

    def __bool__(self):
        return self.v


def _b(x):
    return x.v if isinstance(x, B) else bool(x)


class Z(int):
    """An int whose arithmetic stays Z and whose comparisons give B."""

    def _w(f):
        return lambda a, b: Z(f(int(a), int(b)))

    __add__ = _w(int.__add__)
    __radd__ = _w(int.__radd__)
    __sub__ = _w(int.__sub__)
    __rsub__ = _w(int.__rsub__)
    __mul__ = _w(int.__mul__)
    __rmul__ = _w(int.__rmul__)

    def __floordiv__(a, b):
        assert int(a) >= 0 and int(b) > 0, (
            a,
            b,
        )  # the device truncates: only non-negative operands are allowed
        return Z(int(a) // int(b))

    def __mod__(a, b):
        assert int(a) >= 0 and int(b) > 0, (a, b)
        return Z(int(a) % int(b))

    __lt__ = lambda a, b: B(int(a) < int(b))
    __le__ = lambda a, b: B(int(a) <= int(b))
    __gt__ = lambda a, b: B(int(a) > int(b))
    __ge__ = lambda a, b: B(int(a) >= int(b))
    __eq__ = lambda a, b: B(int(a) == int(b))
    __hash__ = int.__hash__


FX = types.SimpleNamespace(
    max=lambda a, b: Z(max(int(a), int(b))),
    min=lambda a, b: Z(min(int(a), int(b))),
    Int32=Z,
)


def load(name):
    """kernels/<name>.py with its FlyDSL names bound to the integer backend (no FlyDSL / GPU needed)."""
    if "flydsl.expr" not in sys.modules:
        try:
            importlib.import_module("flydsl.expr")
        except ImportError:
            fl = types.ModuleType("flydsl")
            fl.expr = types.ModuleType("flydsl.expr")
            fl.expr.Int32 = Z
            sys.modules["flydsl"], sys.modules["flydsl.expr"] = fl, fl.expr
    spec = importlib.util.spec_from_file_location(
        f"gfx1250_{name}", KERNELS / f"{name}.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    if hasattr(m, "fx"):
        m.fx, m.I32 = FX, Z
    return m

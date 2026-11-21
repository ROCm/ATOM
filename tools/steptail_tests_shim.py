# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A pytest plugin (``-p steptail_tests_shim``): binds the ``tests`` package to
this tree's ``tests/`` where another ``tests`` package comes first on
``sys.path`` (the DSpark tests import each other as ``tests.<module>``). No
test file is changed. Used by ``steptail_pytest.py`` with ``SHIM=1``."""

import importlib.util
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MINE = os.path.join(ROOT, "tests")

_spec = importlib.util.find_spec("tests")
_places = [] if _spec is None else list(_spec.submodule_search_locations or [])
if [os.path.realpath(p) for p in _places] != [os.path.realpath(MINE)]:
    _module = types.ModuleType("tests")
    _module.__path__ = [MINE]
    sys.modules["tests"] = _module

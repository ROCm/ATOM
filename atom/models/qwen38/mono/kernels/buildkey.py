# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build-key values that stay put when a parameter class grows (or loses) a defaulted field.

``Program.key()`` is the compile cache's key and the kernel symbol's digest. A stage's ``key`` holds the Python
constants its ``emit`` closes over: mostly a block's parameter dataclass. :func:`stable` keys a dataclass by its class
name and the fields whose value differs from the field's default, so a defaulted field is keyed only where a program
sets it.
"""

from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from enum import Enum


def _is_default(value, f) -> bool:
    if f.default is not MISSING:
        default = f.default
    elif f.default_factory is not MISSING:
        default = f.default_factory()
    else:
        return False
    if type(value) is not type(default):  # (True is not 1, 1.0 is not 1)
        return False
    try:
        return bool(value == default)
    except Exception:  # noqa: BLE001  (a value without a plain equality: always keyed)
        return False


def stable(x):
    """``x`` as a key value: a dataclass -> (class name, ((field, value) for every field off its default)); tuples,
    lists, dicts and sets element-wise; an enum by its class and member name; anything else as is.
    """
    if is_dataclass(x) and not isinstance(x, type):
        return (
            type(x).__qualname__,
            tuple(
                (f.name, stable(getattr(x, f.name)))
                for f in fields(x)
                if not _is_default(getattr(x, f.name), f)
            ),
        )
    if isinstance(x, Enum):
        return f"{type(x).__qualname__}.{x.name}"
    if isinstance(x, (tuple, list)):
        return tuple(stable(v) for v in x)
    if isinstance(x, dict):
        return tuple(
            sorted(((k, stable(v)) for k, v in x.items()), key=lambda kv: repr(kv[0]))
        )
    if isinstance(x, (set, frozenset)):
        return tuple(sorted((stable(v) for v in x), key=repr))
    return x

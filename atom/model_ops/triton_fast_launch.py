# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Launch a Triton kernel through the compiled kernel its first launch returned.

``kernel[grid](...)`` re-derives every argument's specialization, looks the
compiled kernel up and rebuilds the launch on each call: tens of microseconds
of host time a launch in a busy server, which sits on the critical path of a
speculative decode step. ``FastLaunch`` keys the compiled kernel on the same
properties Triton specializes on -- a tensor's dtype, 16-byte alignment and
whether 32-bit offsets reach its whole storage; an integer's width class and,
unless the parameter is ``do_not_specialize``, whether it is 1 or a multiple
of 16; every constexpr's value and launch option -- and launches a hit
directly. A miss, or an argument it does not recognise, takes the ordinary
``kernel[grid](...)`` path.
"""

import logging

import torch

logger = logging.getLogger("atom")


def _int_class(v: int) -> int:
    if -(2**31) <= v < 2**31:
        return 0
    if 0 <= v < 2**32:
        return 1
    if -(2**63) <= v < 2**63:
        return 2
    return 3


class FastLaunch:
    def __init__(self, fn, *, small_tensors: bool = False):
        """``small_tensors``: the caller guarantees every tensor argument's
        storage is under 2 GB, the property the AMD backend's 32-bit buffer
        offsets specialize on, so the key does not read the storage size."""
        self.fn = fn
        self._small = small_tensors
        self._cache: dict = {}
        params = getattr(fn, "params", None)
        if params is None:  # the Triton interpreter: no compiled kernels
            self.enabled = False
            self._constexpr, self._constexpr_set = (), frozenset()
            return
        self._constexpr = tuple(p.name for p in params if p.is_constexpr)
        self._constexpr_set = frozenset(self._constexpr)
        n = len(params) - len(self._constexpr)
        # Positional (non-constexpr) parameters first, constexprs last: the
        # layout this helper relies on to append the constexpr values.
        assert all(not p.is_constexpr for p in params[:n]), fn
        self._specialized = tuple(not p.do_not_specialize for p in params[:n])
        self.enabled = True

    def _key(self, args, constexprs):
        key = [constexprs]
        for a, spec in zip(args, self._specialized):
            if isinstance(a, torch.Tensor):
                # AMD also specializes on a storage that 32-bit offsets reach
                # (buffer ops).
                key.append(
                    (a.dtype, a.data_ptr() & 15 == 0)
                    if self._small
                    else (
                        a.dtype,
                        a.data_ptr() & 15 == 0,
                        a.untyped_storage().size() < 2**31,
                    )
                )
            elif isinstance(a, bool):
                key.append(("b", a) if spec else "b")
            elif isinstance(a, int):
                key.append(
                    (_int_class(a), a == 1, a % 16 == 0) if spec else _int_class(a)
                )
            elif isinstance(a, float):
                key.append("f")
            else:
                return None
        return tuple(key)

    def __call__(self, grid, *args, **kwargs):
        """``fn[grid](*args, **kwargs)``: ``args`` the non-constexpr
        parameters in order, ``kwargs`` every constexpr and launch option."""
        values = tuple(kwargs[name] for name in self._constexpr)
        options = tuple(
            sorted((k, v) for k, v in kwargs.items() if k not in self._constexpr_set)
        )
        key = self._key(args, (values, options)) if self.enabled else None
        compiled = self._cache.get(key) if key is not None else None
        if compiled is not None:
            try:
                # The compiled kernel's runner takes a 3-D grid.
                compiled[(*grid, 1, 1)[:3]](*args, *values)
                return
            except (TypeError, IndexError, AttributeError, ValueError) as e:
                # A Triton whose compiled-kernel launcher this does not fit
                # raises before launching: the ordinary path from now on.
                logger.warning("FastLaunch off for %s: %r", self.fn, e)
                self.enabled = False
                self._cache.clear()
        compiled = self.fn[grid](*args, **kwargs)
        if key is not None and compiled is not None:
            self._cache[key] = compiled

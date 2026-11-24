# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project

"""The XCD swizzle of AITER's preshuffle GEMMs (``xcd_remap_bx_by``), kept from its shared MFMA preshuffle helpers."""

import flydsl.expr as fx
from flydsl._mlir.dialects.arith import CmpIPredicate


def xcd_remap_bx_by(
    bx,
    by,
    c_m,
    *,
    tile_m: int,
    tile_n: int,
    N: int,
    xcd_swizzle: int,
    num_xcds: int = 8,
):
    """Remap (bx, by) for L2-cache reuse via XCD swizzle.

    No-op when ``xcd_swizzle <= 0``. Otherwise:
      1. Linearize the original (bx, by) grid round-robin across ``num_xcds``
         XCDs so that contiguous workgroup ids stay on the same XCD.
      2. Re-tile that 1-D order with an M-major group of size ``xcd_swizzle``,
         folding the tail group when ``gy`` does not divide evenly.

    Designed to be called inside a ``@flyc.kernel`` immediately after::

        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx, by = xcd_remap_bx_by(bx, by, c_m, tile_m=..., tile_n=..., N=...,
                                 xcd_swizzle=xcd_swizzle)

    ``c_m`` is the dynamic ``fx.Index`` for runtime ``M``; ``tile_m``,
    ``tile_n``, ``N`` and ``xcd_swizzle`` are compile-time Python ints.
    """
    if xcd_swizzle <= 0:
        return bx, by

    _c1 = fx.arith.constant(1, index=True)
    _c_tm = fx.arith.constant(tile_m, index=True)
    _gx = fx.arith.constant(N // tile_n, index=True)
    _gy = (c_m + _c_tm - _c1) // _c_tm

    _linear_id = bx * _gx + by
    _num_wgs = _gx * _gy

    _c_xcds = fx.arith.constant(num_xcds, index=True)
    _q = _num_wgs // _c_xcds
    _r = _num_wgs % _c_xcds
    _xcd = _linear_id % _c_xcds
    _in_xcd = _linear_id // _c_xcds
    _xcd_lt_r = fx.arith.cmpi(CmpIPredicate.ult, _xcd, _r)
    _clip = fx.arith.select(_xcd_lt_r, _xcd, _r)
    _wgid = _xcd * _q + _clip + _in_xcd

    _c_wgm = fx.arith.constant(xcd_swizzle, index=True)
    _num_wgid_in_group = _c_wgm * _gx
    _group_id = _wgid // _num_wgid_in_group
    _first_pid_m = _group_id * _c_wgm
    _remaining_m = _gy - _first_pid_m
    _cmp_m = fx.arith.cmpi(CmpIPredicate.ult, _remaining_m, _c_wgm)
    _group_size_m = fx.arith.select(_cmp_m, _remaining_m, _c_wgm)

    _wgid_in_group = _wgid % _num_wgid_in_group
    new_bx = _first_pid_m + (_wgid_in_group % _group_size_m)
    new_by = _wgid_in_group // _group_size_m
    return new_bx, new_by

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact NVFP4 quantization with scale reciprocals staged in LDS during the GEMM.

Entries 0-255 contain the same correctly rounded 1/(gs * decode_e4m3(byte)) as the standalone insert's quantizer
(``norm_rope_nvfp4_insert_d128_rot64_p128/kernel.py`` ``_quant4``);
zero scale encodings contain zero. Entry 256 contains the same correctly rounded 1/gs.
No arithmetic, rounding or quantization threshold changes: only its scheduling and storage change.
"""

import flydsl.expr as fx
from amd_kernels.attention.norm_rope_nvfp4_insert_d128_rot64_p128.kernel import (
    F448, RCP6, _e2m1x2, _e4m3_byte, _f32, _maxnum, _minnum, _xmax,
)


def _quant4_table(xs, table):
    """``_quant4``'s four codes and group scale, using the current call's exact reciprocal table."""
    a = [fx.math.absf(x) for x in xs]
    m = _maxnum(_maxnum(a[0], a[1]), _maxnum(a[2], a[3]))
    m = _xmax(_xmax(m, 1), 2)
    inv_gs = fx.Int32(table[256]).bitcast(fx.Float32)
    sf = _minnum((m * _f32(RCP6)) * inv_gs, _f32(F448))
    byte = _e4m3_byte(sf)
    quant = fx.Int32(table[byte]).bitcast(fx.Float32)
    y = [x * quant for x in xs]
    return _e2m1x2(y[2], y[3], _e2m1x2(y[0], y[1], fx.Int32(0), 0), 1), byte

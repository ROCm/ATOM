# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""AMD buffer load / store (FlyDSL's typed-pointer generation: ``make_buffer_ptr`` and ``BufferCopy`` atoms).

Buffer operations use a scalar base pointer and per-thread offsets. ``create_buffer_resource_from_addr`` returns a
byte-addressed buffer-descriptor pointer; ``buffer_load`` / ``buffer_store`` take element offsets.
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing

_MAX_NUM_RECORDS = 0xFFFFFFFF


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    - a bare single-result OpView (has .result), e.g. ``arith.TruncIOp(...)``
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "result"):
            value = value.result
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


def _as_num_records(num_records_bytes) -> fx.Int64 | None:
    """Normalize a descriptor byte count to ``fx.Int64`` for ``make_buffer_ptr``."""
    if num_records_bytes is None:
        return None
    if isinstance(num_records_bytes, int):
        num_records_bytes = max(0, min(int(num_records_bytes), _MAX_NUM_RECORDS))
    return fx.Int64(num_records_bytes)


@dsl_loc_tracing
def _byte_buffer_ptr(global_ptr, num_records_bytes):
    """Wrap a global pointer in a byte-addressed buffer-resource pointer."""
    ptr = fx.recast_iter(fx.Int8, global_ptr)
    return fx.rocdl.make_buffer_ptr(
        ptr, num_records_bytes=_as_num_records(num_records_bytes)
    )


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value, *, num_records_bytes: int | ir.Value | None = None
):
    """A buffer resource (``!fly.ptr<i8, BufferDesc>``) whose base is the raw i64 device address ``addr_i64``;
    ``num_records_bytes``: the bound for the hardware's out-of-range check (None: none).
    """
    ptr_ty = fx.PointerType.get(
        T.i8(), address_space=fx.AddressSpace.Global, alignment=16
    )
    base_ptr = fx.inttoptr(ptr_ty, fx.Int64(_unwrap_value(addr_i64)))
    return _byte_buffer_ptr(base_ptr, num_records_bytes)


@dsl_loc_tracing
def buffer_load(rsrc, offset, vec_width: int = 4, dtype=None, cache_modifier: int = 0):
    """``vec_width`` elements of ``dtype`` (an ``ir`` type or a DSL Numeric class) at element ``offset`` of the
    buffer ``rsrc`` (scalar or vector); ``cache_modifier``: the cache-policy bits."""
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    if hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    offset = fx.Int32(_unwrap_value(offset)).ir_value()

    # Convert the API's element offset to the buffer instruction's byte offset.
    element_bytes = dtype.width // 8
    offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)

    src = fx.make_view(
        rsrc + fx.Int32(offset), fx.make_layout(vec_width * element_bytes, 1)
    )
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(vec_width * dtype.width, cache_modifier), fx.Int8
    )
    reg = fx.make_rmem_tensor(
        fx.make_layout(vec_width, 1), fx.Numeric.from_ir_type(dtype)
    )
    fx.copy(atom, src, reg)
    loaded = fx.memref_load_vec(reg)
    result = _unwrap_value(loaded[0] if vec_width == 1 else loaded)
    if result.type != result_type:
        if vec_width == 1:
            result = (
                fx.Numeric.from_ir_type(result.type)(result)
                .bitcast(fx.Numeric.from_ir_type(dtype))
                .ir_value()
            )
        else:
            result = (
                fx.Vector(result).bitcast(fx.Numeric.from_ir_type(dtype)).ir_value()
            )
    return result


@dsl_loc_tracing
def buffer_store(data, rsrc, offset, cache_modifier: int = 0):
    """Store ``data`` (scalar or vector) at element ``offset`` of the buffer ``rsrc``; ``cache_modifier``: the
    cache-policy bits."""
    # Keep `rsrc` as a !fly.ptr<i8, BufferDesc> for the copy atom.
    data = _unwrap_value(data)
    offset = fx.Int32(_unwrap_value(offset)).ir_value()

    data_type = data.type
    if hasattr(data_type, "element_type"):  # Vector type
        element_type = data_type.element_type
        vec_width = data_type.shape[0]
    else:  # Scalar type
        element_type = data_type
        vec_width = 1
    element_bytes = element_type.width // 8

    # Convert element offsets to the buffer instruction's byte offset.
    offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    dst = fx.make_view(
        rsrc + fx.Int32(offset), fx.make_layout(vec_width * element_bytes, 1)
    )
    atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy(vec_width * element_type.width, cache_modifier), fx.Int8
    )
    reg = fx.make_rmem_tensor(
        fx.make_layout(vec_width, 1), fx.Numeric.from_ir_type(element_type)
    )
    fx.memref_store_vec(data, reg)
    fx.copy(atom, reg, dst)

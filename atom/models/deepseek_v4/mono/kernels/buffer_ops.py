# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications: ATOM extraction and removal of unused memref APIs, 2026-10-22.

"""AMD Buffer Load/Store Operations - High-level Python API

This module provides high-level Python wrappers for AMD CDNA3/CDNA4 buffer operations.
Buffer operations use a scalar base pointer and per-thread offsets for efficient memory access.

``create_buffer_resource_from_addr`` wraps a raw device address in a byte-addressed descriptor.

Example:
    >>> from atom.models.deepseek_v4.mono.kernels import buffer_ops
    >>> from flydsl.expr import arith
    >>> from flydsl._mlir.extras import types as T
    >>>
    >>> # Create buffer resource from a raw device address
    >>> rsrc = buffer_ops.create_buffer_resource_from_addr(addr)
    >>>
    >>> # Compute offset
    >>> offset = row * arith.index(4096) + col
    >>>
    >>> # Buffer load (4xf32)
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>>
    >>> # Buffer store
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing

__all__ = [
    "buffer_load",
    "buffer_store",
    "create_buffer_resource_from_addr",
]

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


@dsl_loc_tracing
def _create_i32_constant(value: int) -> ir.Value:
    """Create a signless i32 constant from a packed 32-bit value."""
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    return fx.Int32(value).ir_value()


@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


def _as_num_records(num_records_bytes) -> fx.Int64 | None:
    """Normalize a descriptor byte count to ``fx.Int64`` for ``make_buffer_ptr``."""
    if num_records_bytes is None:
        return None
    if isinstance(num_records_bytes, int):
        num_records_bytes = max(0, min(int(num_records_bytes), _MAX_NUM_RECORDS))
    return fx.Int64(num_records_bytes)


@dsl_loc_tracing
def _byte_buffer_ptr(global_ptr, num_records_bytes, base_byte_offset=None):
    """Wrap a global pointer in a byte-addressed buffer-resource pointer."""
    ptr = fx.recast_iter(fx.Int8, global_ptr)
    if base_byte_offset is not None:
        if isinstance(base_byte_offset, ir.Value) and not isinstance(
            base_byte_offset.type, ir.IndexType
        ):
            base_byte_offset = fx.Int32(base_byte_offset)
        ptr = ptr + base_byte_offset
    return fx.rocdl.make_buffer_ptr(
        ptr, num_records_bytes=_as_num_records(num_records_bytes)
    )


@dsl_loc_tracing
def _add_soffset_bytes(byte_offset: ir.Value, soffset_bytes) -> ir.Value:
    """Fold ``soffset_bytes`` into the byte offset."""
    if soffset_bytes is None:
        return byte_offset
    return (fx.Int32(byte_offset) + fx.Int32(soffset_bytes)).ir_value()


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        Byte-addressed buffer-descriptor pointer (``!fly.ptr<i8, BufferDesc>``).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    ptr_ty = fx.PointerType.get(
        T.i8(), address_space=fx.AddressSpace.Global, alignment=16
    )
    base_ptr = fx.inttoptr(ptr_ty, fx.Int64(_unwrap_value(addr_i64)))
    return _byte_buffer_ptr(base_ptr, num_records_bytes)


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    soffset_bytes: int | ir.Value | None = None,
    is_scalar: bool = False,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer descriptor pointer from :func:`create_buffer_resource_from_addr`
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1 or 4;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    # Scalar (uniform) loads return raw i32 dwords; force the element type so the
    # element->byte offset math below uses 4 and the result type is i32 / v4i32.
    if is_scalar:
        if vec_width not in (1, 4):
            raise ValueError(
                f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
            )
        if mask is not None or soffset_bytes is not None:
            raise ValueError(
                "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
            )
        dtype = T.i32()
    elif dtype is None:
        dtype = T.f32()
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    offset = fx.Int32(_unwrap_value(offset)).ir_value()

    # Convert the API's element offset to the buffer instruction's byte offset.
    element_bytes = dtype.width // 8
    offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = (
            fx.Boolean(mask).select(fx.Int32(offset), fx.Int32(0x7FFFFFFF)).ir_value()
        )

    result_type = dtype if vec_width == 1 else ir.VectorType.get([vec_width], dtype)

    # Scalar/uniform load path: s.buffer.load is an SMEM instruction with no CopyOp type,
    # so it stays on the raw intrinsic and needs the raw v4i32 resource. Returns i32
    # (vec_width 1) or v4i32 (vec_width 4).
    if is_scalar:
        rsrc_v4 = _ptr8_to_v4i32(fx.rocdl.get_buffer_rsrc(rsrc))
        cache_policy = _create_i32_constant(cache_modifier)
        suffix = "i32" if vec_width == 1 else "v4i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [rsrc_v4, offset, cache_policy],
            [],
            [],
        )

    offset = _add_soffset_bytes(offset, soffset_bytes)
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
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: int | ir.Value | None = None,
    offset_is_bytes: bool = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.


    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer descriptor pointer from :func:`create_buffer_resource_from_addr`
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
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

    # Convert element offsets unless the caller already supplied bytes.
    if not offset_is_bytes:
        offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = (
            fx.Boolean(mask).select(fx.Int32(offset), fx.Int32(0x7FFFFFFF)).ir_value()
        )

    offset = _add_soffset_bytes(offset, soffset_bytes)
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

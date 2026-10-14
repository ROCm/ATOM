# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Frames and values: how bulk data and dict-shaped fields ride a multipart.

A message is ``[Envelope, frame 1, frame 2, ...]``. `FrameWriter` places each
buffer an envelope refers to: a large one becomes a frame of its own and is
handed to ZMQ without a copy; small ones are packed, aligned, into one shared
frame so a batch of short prompts is not hundreds of tiny frames. On the other
end `FrameReader` hands out views of the received frames, so a decoded tensor
aliases the ZMQ message rather than a copy of it.

`encode_value` / `decode_value` carry the dict-shaped fields (multimodal data,
KV-transfer params, metrics) as a typed tree. Types are matched exactly: a
tuple stays a tuple, a numpy scalar stays a numpy scalar, and anything the
tree cannot express -- a subclass included -- is pickled into a frame rather
than silently converted.
"""

import array
import pickle
import sys

import numpy as np

from atom.model_engine.protocol.binding import pb

if sys.byteorder != "little":  # pragma: no cover - every supported host is LE
    raise ImportError("the engine wire protocol assumes a little-endian host")

PROTOCOL_VERSION = 1
# Buffers at least this large get a frame of their own and are sent without a
# copy; pyzmq copies anything smaller into the message regardless.
OWN_FRAME_BYTES = 64 * 1024
_ALIGN = 64


class ProtocolError(ValueError):
    """A message that does not decode under this protocol version."""


class FrameWriter:
    """Collects the frames one message's buffers live in."""

    def __init__(self, own_frame_bytes: int = OWN_FRAME_BYTES):
        self._own_frame_bytes = own_frame_bytes
        self._frames: list = []
        self._packed: bytearray | None = None
        self._packed_index = 0
        # id(key) -> (frame, offset, nbytes), for bytes several refs share:
        # n > 1 siblings carry one prompt and one set of media tensors.
        # `_keep` pins the keys so their ids stay unique for the writer's life.
        self._placed: dict[int, tuple[int, int, int]] = {}
        self._keep: list = []

    def add(self, buffer, ref, *, key=None) -> None:
        """Point `ref` (a BufferRef) at the bytes of `buffer`.

        `key`, when given, is an object whose identity names those bytes: a
        later `add` with the same key reuses the placement instead of sending
        them again.
        """
        if key is not None and id(key) in self._placed:
            ref.frame, ref.offset, ref.nbytes = self._placed[id(key)]
            return
        view = memoryview(buffer)
        if not view.c_contiguous:
            raise ProtocolError("buffers on the wire must be C-contiguous")
        view = view.cast("B")
        nbytes = view.nbytes
        if nbytes >= self._own_frame_bytes:
            self._frames.append(view)
            placed = (len(self._frames), 0, nbytes)
        else:
            if self._packed is None:
                self._packed = bytearray()
                self._frames.append(None)  # filled in by `frames()`
                self._packed_index = len(self._frames)
            packed = self._packed
            packed.extend(b"\0" * (-len(packed) % _ALIGN))
            offset = len(packed)
            packed.extend(view)
            placed = (self._packed_index, offset, nbytes)
        ref.frame, ref.offset, ref.nbytes = placed
        if key is not None:
            self._placed[id(key)] = placed
            self._keep.append(key)

    def frames(self, envelope: "pb.Envelope") -> list:
        envelope.protocol_version = PROTOCOL_VERSION
        # Deterministic: map entries in key order, so one message has one
        # encoding whichever order its maps were filled in.
        frames = [envelope.SerializeToString(deterministic=True), *self._frames]
        if self._packed is not None:
            frames[self._packed_index] = self._packed
        return frames


class FrameReader:
    """Views into the frames of one received message."""

    def __init__(self, frames: list):
        self._buffers = [
            frame.buffer if hasattr(frame, "buffer") else memoryview(frame)
            for frame in frames
        ]

    def view(self, ref) -> memoryview:
        if not 1 <= ref.frame < len(self._buffers):
            raise ProtocolError(f"buffer refers to missing frame {ref.frame}")
        buffer = self._buffers[ref.frame]
        end = ref.offset + ref.nbytes
        if end > buffer.nbytes:
            raise ProtocolError(
                f"buffer [{ref.offset}, {end}) overruns frame {ref.frame} "
                f"({buffer.nbytes} bytes)"
            )
        return buffer[ref.offset : end]

    def int32_array(self, ref) -> array.array:
        view = self.view(ref)
        if view.nbytes % 4:
            raise ProtocolError("an int32 buffer must be a multiple of 4 bytes")
        out = array.array("i")
        out.frombytes(view)
        return out

    def float64_array(self, ref) -> array.array:
        view = self.view(ref)
        if view.nbytes % 8:
            raise ProtocolError("a float64 buffer must be a multiple of 8 bytes")
        out = array.array("d")
        out.frombytes(view)
        return out


def decode_envelope(frames: list) -> "pb.Envelope":
    if not frames:
        raise ProtocolError("empty message")
    head = frames[0]
    envelope = pb.Envelope()
    envelope.ParseFromString(head.bytes if hasattr(head, "bytes") else bytes(head))
    if envelope.protocol_version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version {envelope.protocol_version}, expected "
            f"{PROTOCOL_VERSION}"
        )
    return envelope


# ---------------------------------------------------------------- tensors --

_NUMPY_DTYPES = {
    np.dtype(np.bool_): pb.BOOL,
    np.dtype(np.uint8): pb.UINT8,
    np.dtype(np.int8): pb.INT8,
    np.dtype(np.int16): pb.INT16,
    np.dtype(np.uint16): pb.UINT16,
    np.dtype(np.int32): pb.INT32,
    np.dtype(np.uint32): pb.UINT32,
    np.dtype(np.int64): pb.INT64,
    np.dtype(np.uint64): pb.UINT64,
    np.dtype(np.float16): pb.FLOAT16,
    np.dtype(np.float32): pb.FLOAT32,
    np.dtype(np.float64): pb.FLOAT64,
    np.dtype(np.complex64): pb.COMPLEX64,
    np.dtype(np.complex128): pb.COMPLEX128,
}
_NUMPY_FROM_WIRE = {wire: dtype for dtype, wire in _NUMPY_DTYPES.items()}
_torch_dtypes: dict | None = None
_torch_from_wire: dict | None = None


def _torch_tables():
    global _torch_dtypes, _torch_from_wire
    if _torch_dtypes is None:
        import torch

        names = {
            "bool": pb.BOOL,
            "uint8": pb.UINT8,
            "int8": pb.INT8,
            "int16": pb.INT16,
            "uint16": pb.UINT16,
            "int32": pb.INT32,
            "uint32": pb.UINT32,
            "int64": pb.INT64,
            "uint64": pb.UINT64,
            "float16": pb.FLOAT16,
            "bfloat16": pb.BFLOAT16,
            "float32": pb.FLOAT32,
            "float64": pb.FLOAT64,
            "complex64": pb.COMPLEX64,
            "complex128": pb.COMPLEX128,
        }
        _torch_dtypes = {
            getattr(torch, name): wire
            for name, wire in names.items()
            if hasattr(torch, name)
        }
        _torch_from_wire = {wire: dtype for dtype, wire in _torch_dtypes.items()}
    return _torch_dtypes, _torch_from_wire


def _is_torch_tensor(value) -> bool:
    module = type(value).__module__
    if not module.startswith("torch"):
        return False
    import torch

    return type(value) is torch.Tensor


def _encode_numpy(value, msg, writer: FrameWriter, kind) -> bool:
    wire = _NUMPY_DTYPES.get(value.dtype.newbyteorder("="))
    if wire is None:
        return False
    if value.dtype.byteorder == ">":
        value = value.astype(value.dtype.newbyteorder("<"))
    original = value
    if not value.flags.c_contiguous:
        # Not `np.ascontiguousarray`, which turns a 0-d array into a 1-d one.
        value = value.copy(order="C")
    msg.dtype = wire
    msg.kind = kind
    msg.shape.extend(value.shape)
    writer.add(value.reshape(-1).view(np.uint8), msg.data, key=original)
    return True


def _encode_torch(value, msg, writer: FrameWriter) -> bool:
    import torch

    dtypes, _ = _torch_tables()
    wire = dtypes.get(value.dtype)
    if wire is None or value.layout != torch.strided or value.is_quantized:
        return False
    tensor = value.detach()
    if tensor.device.type != "cpu":
        msg.device = str(tensor.device)
        tensor = tensor.cpu()
    tensor = tensor.resolve_conj().resolve_neg().contiguous()
    msg.dtype = wire
    msg.kind = pb.TORCH
    msg.requires_grad = value.requires_grad
    msg.shape.extend(tensor.shape)
    writer.add(tensor.reshape(-1).view(torch.uint8).numpy(), msg.data, key=value)
    return True


def _decode_tensor(msg, reader: FrameReader):
    view = reader.view(msg.data)
    shape = tuple(msg.shape)
    if msg.kind == pb.TORCH:
        import torch

        _, from_wire = _torch_tables()
        dtype = from_wire.get(msg.dtype)
        if dtype is None:
            raise ProtocolError(f"unknown torch dtype {msg.dtype}")
        if view.nbytes:
            tensor = torch.frombuffer(view, dtype=dtype).reshape(shape)
        else:
            tensor = torch.empty(shape, dtype=dtype)
        if msg.device:
            tensor = tensor.to(msg.device)
        if msg.requires_grad:
            tensor.requires_grad_(True)
        return tensor
    dtype = _NUMPY_FROM_WIRE.get(msg.dtype)
    if dtype is None:
        raise ProtocolError(f"unknown numpy dtype {msg.dtype}")
    out = np.frombuffer(view, dtype=dtype).reshape(shape)
    if msg.kind == pb.NUMPY_SCALAR:
        return out[()]
    return out


# ---------------------------------------------------------------- values ---

_INT64_MIN, _INT64_MAX = -(1 << 63), (1 << 63) - 1


def encode_value(value, msg: "pb.Value", writer: FrameWriter) -> None:
    """Fill `msg` with `value`."""
    kind = type(value)
    if value is None:
        msg.null_value = True
    elif kind is bool:
        msg.bool_value = value
    elif kind is int:
        if _INT64_MIN <= value <= _INT64_MAX:
            msg.int_value = value
        else:
            msg.big_int_value = str(value)
    elif kind is float:
        msg.float_value = value
    elif kind is str:
        msg.string_value = value
    elif kind is bytes:
        msg.bytes_value = value
    elif kind is list or kind is tuple:
        items = msg.list_value if kind is list else msg.tuple_value
        items.SetInParent()
        for item in value:
            encode_value(item, items.items.add(), writer)
    elif kind is dict:
        entries = msg.map_value
        entries.SetInParent()
        for key, item in value.items():
            entry = entries.entries.add()
            encode_value(key, entry.key, writer)
            encode_value(item, entry.value, writer)
    elif not _encode_array(value, msg.tensor_value, writer):
        writer.add(
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL), msg.pickled_value
        )


def _encode_array(value, msg, writer: FrameWriter) -> bool:
    """Fill a Tensor `msg` from an ndarray, numpy scalar or tensor, if it is
    one with a wire dtype; leave it untouched and say so otherwise."""
    if type(value) is np.ndarray:
        return _encode_numpy(value, msg, writer, pb.NUMPY)
    if isinstance(value, np.generic):
        return _encode_numpy(np.asarray(value), msg, writer, pb.NUMPY_SCALAR)
    if _is_torch_tensor(value):
        return _encode_torch(value, msg, writer)
    return False


def decode_value(msg: "pb.Value", reader: FrameReader):
    which = msg.WhichOneof("kind")
    if which == "null_value":
        return None
    if which == "bool_value":
        return msg.bool_value
    if which == "int_value":
        return msg.int_value
    if which == "big_int_value":
        return int(msg.big_int_value)
    if which == "float_value":
        return msg.float_value
    if which == "string_value":
        return msg.string_value
    if which == "bytes_value":
        return msg.bytes_value
    if which == "list_value":
        return [decode_value(item, reader) for item in msg.list_value.items]
    if which == "tuple_value":
        return tuple(decode_value(item, reader) for item in msg.tuple_value.items)
    if which == "map_value":
        return {
            decode_value(entry.key, reader): decode_value(entry.value, reader)
            for entry in msg.map_value.entries
        }
    if which == "tensor_value":
        return _decode_tensor(msg.tensor_value, reader)
    if which == "pickled_value":
        return pickle.loads(reader.view(msg.pickled_value))
    raise ProtocolError("a value carries no kind")


def encode_optional_value(value, parent, field: str, writer: FrameWriter) -> None:
    """Set `parent.field` unless `value` is None, which stays absent."""
    if value is not None:
        encode_value(value, getattr(parent, field), writer)


def decode_optional_value(parent, field: str, reader: FrameReader):
    if not parent.HasField(field):
        return None
    return decode_value(getattr(parent, field), reader)

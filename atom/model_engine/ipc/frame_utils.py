"""Bounded headers and borrowed multipart attachments for engine IPC."""

from __future__ import annotations

import array
import math
import sys
from dataclasses import dataclass, field
from typing import ClassVar

from atom.model_engine.engine_core_protocol import EngineCoreRequestType

from .proto import common_pb2 as common


class ProtocolError(ValueError):
    pass


@dataclass(slots=True)
class EncodedMessage:
    frames: list
    owners: list = field(default_factory=list)


@dataclass(slots=True)
class DecodedMessage:
    kind: object
    payload: object
    metadata: dict
    owners: list = field(default_factory=list)


class Fields:
    @staticmethod
    def get(value, name, default=None):
        return (
            value.get(name, default)
            if isinstance(value, dict)
            else getattr(value, name, default)
        )

    @staticmethod
    def kind(value):
        return (
            value.name
            if isinstance(value, EngineCoreRequestType)
            else str(value).upper()
        )

    @staticmethod
    def kind_value(name):
        return getattr(EngineCoreRequestType, name, name)

    @staticmethod
    def present(message, names):
        return {
            name: getattr(message, name) for name in names if message.HasField(name)
        }

    @staticmethod
    def assign(message, value, names):
        for name in names:
            item = Fields.get(value, name)
            if item is not None:
                setattr(message, name, item)


class DTypes:
    NAMES: ClassVar[dict] = {
        common.INT32: ("int32", 4),
        common.INT64: ("int64", 8),
        common.FLOAT32: ("float32", 4),
        common.FLOAT64: ("float64", 8),
        common.FLOAT16: ("float16", 2),
        common.BFLOAT16: ("bfloat16", 2),
        common.UINT8: ("uint8", 1),
        common.INT8: ("int8", 1),
        common.BOOL: ("bool", 1),
        common.UINT16: ("uint16", 2),
        common.UINT32: ("uint32", 4),
        common.UINT64: ("uint64", 8),
        common.INT16: ("int16", 2),
        common.FLOAT8_E4M3FN: ("float8_e4m3fn", 1),
        common.FLOAT8_E4M3FNUZ: ("float8_e4m3fnuz", 1),
        common.FLOAT8_E5M2: ("float8_e5m2", 1),
        common.FLOAT8_E5M2FNUZ: ("float8_e5m2fnuz", 1),
    }

    @classmethod
    def encode(cls, value):
        name = str(value).removeprefix("torch.").removesuffix("_le")
        for code, (candidate, _) in cls.NAMES.items():
            if name == candidate:
                return code
        raise ProtocolError(f"unsupported tensor dtype: {name}")

    @classmethod
    def size(cls, code):
        if code not in cls.NAMES:
            raise ProtocolError(f"unknown dtype: {code}")
        return cls.NAMES[code][1]


class MetaCodec:
    """Builtin values as ``MetaValue``.

    The default mode is JSON-shaped: a tuple arrives as a list, keys must be
    strings and floats finite. ``exact`` keeps every value
    ``collective_rpc.is_plain_rpc_value`` accepts as it was -- tuples, dicts
    keyed by any plain value, non-finite floats -- at that check's depth.
    """

    MAX_DEPTH = 16
    EXACT_MAX_DEPTH = 64
    MAX_ITEMS = 16384
    MAX_BYTES = 1 << 20

    @classmethod
    def _check_depth(cls, depth, exact):
        if depth > (cls.EXACT_MAX_DEPTH if exact else cls.MAX_DEPTH):
            raise ProtocolError("metadata nesting limit exceeded")

    @classmethod
    def encode(cls, target, value, depth=0, *, exact=False):
        cls._check_depth(depth, exact)
        if value is None:
            target.null_value = True
        elif type(value) is bool:
            target.bool_value = value
        elif type(value) is int:
            if -(1 << 63) <= value < 0:
                target.signed_value = value
            elif 0 <= value < 1 << 64:
                target.unsigned_value = value
            else:
                raise ProtocolError("metadata integer exceeds 64 bits")
        elif type(value) is float:
            if not exact and not math.isfinite(value):
                raise ProtocolError("metadata float must be finite")
            target.double_value = value
        elif type(value) in (str, bytes):
            if len(value) > cls.MAX_BYTES:
                raise ProtocolError("metadata byte limit exceeded")
            setattr(
                target,
                "string_value" if isinstance(value, str) else "bytes_value",
                value,
            )
        elif type(value) in (list, tuple):
            if len(value) > cls.MAX_ITEMS:
                raise ProtocolError("metadata item limit exceeded")
            items = (
                target.tuple_value
                if exact and type(value) is tuple
                else target.list_value
            )
            items.SetInParent()
            for item in value:
                cls.encode(items.values.add(), item, depth + 1, exact=exact)
        elif type(value) is dict:
            if exact and any(type(key) is not str for key in value):
                if len(value) > cls.MAX_ITEMS:
                    raise ProtocolError("metadata item limit exceeded")
                target.map_value.SetInParent()
                for key, item in value.items():
                    entry = target.map_value.entries.add()
                    cls.encode(entry.key, key, depth + 1, exact=True)
                    cls.encode(entry.value, item, depth + 1, exact=True)
            else:
                cls.encode_object(target.object_value, value, depth + 1, exact=exact)
        else:
            raise ProtocolError(f"unsupported metadata type: {type(value).__name__}")

    @classmethod
    def encode_object(cls, target, value, depth=0, *, exact=False):
        if type(value) is not dict or len(value) > cls.MAX_ITEMS:
            raise ProtocolError("metadata object must be a bounded string-keyed dict")
        cls._check_depth(depth, exact)
        target.SetInParent()
        for key, item in value.items():
            if type(key) is not str:
                raise ProtocolError("metadata object keys must be strings")
            cls.encode(target.fields[key], item, depth + 1, exact=exact)

    @classmethod
    def decode(cls, value, depth=0, *, exact=False):
        cls._check_depth(depth, exact)
        kind = value.WhichOneof("value")
        if kind is None:
            raise ProtocolError("metadata value is missing")
        if kind == "null_value":
            return None
        if kind == "object_value":
            return cls.decode_object(value.object_value, depth + 1, exact=exact)
        if kind in ("list_value", "tuple_value"):
            items = getattr(value, kind).values
            if len(items) > cls.MAX_ITEMS:
                raise ProtocolError("metadata item limit exceeded")
            if kind == "tuple_value" and not exact:
                raise ProtocolError("tuple metadata needs exact mode")
            decoded = [cls.decode(item, depth + 1, exact=exact) for item in items]
            return tuple(decoded) if kind == "tuple_value" else decoded
        if kind == "map_value":
            entries = value.map_value.entries
            if not exact:
                raise ProtocolError("non-string metadata keys need exact mode")
            if len(entries) > cls.MAX_ITEMS:
                raise ProtocolError("metadata item limit exceeded")
            result = {}
            for entry in entries:
                key = cls.decode(entry.key, depth + 1, exact=True)
                try:
                    result[key] = cls.decode(entry.value, depth + 1, exact=True)
                except TypeError:
                    raise ProtocolError("metadata map key is unhashable") from None
            return result
        result = getattr(value, kind)
        if kind in ("string_value", "bytes_value") and len(result) > cls.MAX_BYTES:
            raise ProtocolError("metadata byte limit exceeded")
        if kind == "double_value" and not exact and not math.isfinite(result):
            raise ProtocolError("metadata float must be finite")
        return result

    @classmethod
    def decode_object(cls, value, depth=0, *, exact=False):
        if len(value.fields) > cls.MAX_ITEMS:
            raise ProtocolError("metadata object limit exceeded")
        cls._check_depth(depth, exact)
        return {
            key: cls.decode(item, depth + 1, exact=exact)
            for key, item in value.fields.items()
        }

    @staticmethod
    def encode_number(target, value):
        if value is None:
            target.null_value = True
        elif type(value) is bool:
            target.bool_value = value
        elif type(value) is int:
            if -(1 << 63) <= value < 0:
                target.signed_value = value
            elif 0 <= value < 1 << 64:
                target.unsigned_value = value
            else:
                raise ProtocolError("counter exceeds 64 bits")
        elif type(value) is float and math.isfinite(value):
            target.double_value = value
        else:
            raise ProtocolError("counter must be a finite number")

    @staticmethod
    def decode_number(value):
        kind = value.WhichOneof("value")
        if kind is None:
            raise ProtocolError("counter value is missing")
        if kind == "null_value":
            return None
        result = getattr(value, kind)
        if type(result) is float and not math.isfinite(result):
            raise ProtocolError("counter must be finite")
        return result


class FrameWriter:
    def __init__(self, max_attachments=1024, max_batch_bytes=512 << 20):
        self.frames = []
        self.owners = []
        self._indices = {}
        self.max_attachments = max_attachments
        self.max_batch_bytes = max_batch_bytes
        self.nbytes = 0

    @staticmethod
    def byte_view(value):
        view = memoryview(value.buffer if hasattr(value, "buffer") else value)
        if not view.c_contiguous:
            raise ProtocolError("attachments must be contiguous")
        return view.cast("B").toreadonly()

    def buffer(self, target, value, dtype, *, identity=None, owner=None):
        view = self.byte_view(value)
        size = DTypes.size(dtype)
        if view.nbytes % size:
            raise ProtocolError("attachment length is not aligned to its dtype")
        key = identity if identity is not None else id(value)
        index = self._indices.get(key)
        if index is None:
            if len(self.frames) >= self.max_attachments:
                raise ProtocolError("attachment count limit exceeded")
            if view.nbytes > self.max_batch_bytes - self.nbytes:
                raise ProtocolError("attachment byte limit exceeded")
            index = len(self.frames)
            self._indices[key] = index
            self.frames.append(view)
            self.nbytes += view.nbytes
            self.owners.append(value if owner is None else owner)
        elif len(self.frames[index]) != view.nbytes:
            raise ProtocolError("attachment identity reused with another size")
        target.buffer_id = str(index)
        target.logical_length = view.nbytes // size
        target.dtype = dtype
        target.byte_order = "little"
        target.raw.frame_index = index
        target.raw.nbytes = view.nbytes

    @staticmethod
    def shm(target, descriptor):
        if descriptor.dtype != "int32_le":
            raise ProtocolError("shared prompt descriptor must use int32_le")
        target.dtype = common.INT32
        target.byte_order = "little"
        target.logical_length = descriptor.nbytes // 4
        target.buffer_id = (
            f"{descriptor.arena_id}:{descriptor.allocation_id}:{descriptor.generation}"
        )
        target.shm.arena_id = descriptor.arena_id
        target.shm.arena_epoch = descriptor.epoch
        target.shm.allocation_id = descriptor.allocation_id
        target.shm.allocation_generation = descriptor.generation
        target.shm.offset_bytes = descriptor.offset
        target.shm.nbytes = descriptor.nbytes
        FrameReader.validate(target)

    def packed(self, target, values, typecode, dtype):
        if (
            isinstance(values, array.array)
            and values.typecode == typecode
            or isinstance(values, memoryview)
            and values.format == typecode
        ):
            owner = values
        else:
            owner = array.array(typecode, values)
        if sys.byteorder != "little":
            owner = array.array(typecode, owner)
            owner.byteswap()
        self.buffer(target, owner, dtype)


class FrameReader:
    def __init__(self, frames):
        self.frames = frames
        self.used = set()

    @staticmethod
    def validate(ref):
        storage = ref.WhichOneof("storage")
        if storage is None:
            raise ProtocolError("buffer storage is missing")
        size = DTypes.size(ref.dtype)
        if ref.byte_order != "little":
            raise ProtocolError("only little-endian buffers are supported")
        part = getattr(ref, storage)
        if part.offset_bytes % size or part.nbytes % size:
            raise ProtocolError("buffer slice is not aligned to its dtype")
        if (
            ref.logical_length > ((1 << 63) - 1) // size
            or ref.logical_length * size != part.nbytes
        ):
            raise ProtocolError("buffer length does not match its dtype")
        if storage == "shm" and (
            not part.arena_id or not part.arena_epoch or not part.allocation_generation
        ):
            raise ProtocolError("shared buffer identity is incomplete")
        return storage

    def view(self, ref, *, dtype=None):
        storage = self.validate(ref)
        if dtype is not None and ref.dtype != dtype:
            raise ProtocolError("unexpected buffer dtype")
        if storage != "raw":
            raise ProtocolError("raw attachment required")
        part = ref.raw
        if part.frame_index >= len(self.frames):
            raise ProtocolError("attachment index is out of bounds")
        frame = self.frames[part.frame_index]
        view = FrameWriter.byte_view(frame)
        if part.offset_bytes > len(view) or part.nbytes > len(view) - part.offset_bytes:
            raise ProtocolError("attachment slice is out of bounds")
        self.used.add(part.frame_index)
        return view[part.offset_bytes : part.offset_bytes + part.nbytes], frame

    def prompt(self, ref):
        from .buffer_types import BufferDescriptor, PromptHandle

        if ref.dtype != common.INT32:
            raise ProtocolError("prompt must contain int32 tokens")
        if self.validate(ref) == "shm":
            part = ref.shm
            return BufferDescriptor(
                arena_id=part.arena_id,
                epoch=part.arena_epoch,
                allocation_id=part.allocation_id,
                generation=part.allocation_generation,
                offset=part.offset_bytes,
                nbytes=part.nbytes,
            )
        view, owner = self.view(ref, dtype=common.INT32)
        if sys.byteorder != "little":
            raise ProtocolError("borrowed int32 prompts require a little-endian host")
        return PromptHandle.from_buffer(view.cast("i"), owner=owner)

    def finish(self):
        if self.used != set(range(len(self.frames))):
            raise ProtocolError("message contains unused attachments")


class ChannelCodec:
    SCHEMA_ID = 0
    MAJOR = 1
    MINOR = 0
    MESSAGE = None

    def __init__(
        self,
        *,
        max_header_bytes=4 << 20,
        max_attachments=1024,
        max_batch_size=256,
        max_batch_bytes=512 << 20,
    ):
        self.max_header_bytes = max_header_bytes
        self.max_attachments = max_attachments
        self.max_batch_size = max_batch_size
        self.max_batch_bytes = max_batch_bytes

    def _new(
        self, *, session_id="", engine_id="", message_id=0, in_reply_to=0, **ignored
    ):
        message = self.MESSAGE(
            schema_id=self.SCHEMA_ID,
            protocol_major=self.MAJOR,
            protocol_minor=self.MINOR,
            manager_session_id=session_id,
            engine_instance_id=engine_id,
            message_id=message_id,
        )
        if "in_reply_to" in message.DESCRIPTOR.fields_by_name:
            message.in_reply_to = in_reply_to
        return message, FrameWriter(self.max_attachments, self.max_batch_bytes)

    def _finish(self, message, writer):
        message.attachment_count = len(writer.frames)
        if message.ByteSize() > self.max_header_bytes:
            raise ProtocolError("IPC header exceeds size limit")
        return EncodedMessage(
            [message.SerializeToString(), *writer.frames], writer.owners
        )

    def _parse(self, frames, *, session_id=None, engine_id=None):
        if not frames:
            raise ProtocolError("IPC header is missing")
        header = FrameWriter.byte_view(frames[0])
        if (
            len(header) > self.max_header_bytes
            or len(frames) - 1 > self.max_attachments
        ):
            raise ProtocolError("IPC message exceeds configured limits")
        if (
            sum(FrameWriter.byte_view(frame).nbytes for frame in frames[1:])
            > self.max_batch_bytes
        ):
            raise ProtocolError("attachment byte limit exceeded")
        message = self.MESSAGE()
        try:
            message.ParseFromString(header)
        except Exception as error:
            raise ProtocolError("malformed protobuf header") from error
        if message.schema_id != self.SCHEMA_ID or message.protocol_major != self.MAJOR:
            raise ProtocolError("incompatible IPC schema identity or major version")
        if message.attachment_count != len(frames) - 1:
            raise ProtocolError("IPC attachment count mismatch")
        if session_id is not None and message.manager_session_id != session_id:
            raise ProtocolError("IPC manager session mismatch")
        if engine_id is not None and message.engine_instance_id != engine_id:
            raise ProtocolError("IPC engine instance mismatch")
        metadata = {
            "session_id": message.manager_session_id,
            "engine_id": message.engine_instance_id,
            "message_id": message.message_id,
            "in_reply_to": getattr(message, "in_reply_to", 0),
        }
        return message, FrameReader(frames[1:]), metadata

    def _batch(self, values):
        if not 0 < len(values) <= self.max_batch_size:
            raise ProtocolError("IPC batch size is outside configured limits")

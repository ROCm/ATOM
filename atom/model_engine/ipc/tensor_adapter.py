"""Known CPU array formats and the versioned PyTorch CUDA reducer ABI."""

from __future__ import annotations

import inspect
import math

from .frame_utils import DTypes, ProtocolError
from .proto import common_pb2 as common


class TensorAdapter:
    CUDA_REDUCER_ABI = 1
    MAX_DIMS = 32
    CUDA_PARAMETERS = (
        "tensor_cls",
        "tensor_size",
        "tensor_stride",
        "tensor_offset",
        "storage_cls",
        "dtype",
        "storage_device",
        "storage_handle",
        "storage_size_bytes",
        "storage_offset_bytes",
        "requires_grad",
        "ref_counter_handle",
        "ref_counter_offset",
        "event_handle",
        "event_sync_required",
    )

    @classmethod
    def encode(cls, target, value, writer, *, name=""):
        target.name = name
        if type(value).__module__.startswith("torch"):
            import torch

            if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
                raise ProtocolError("only strided CPU tensors are supported")
            if value.device.type != "cpu":
                raise ProtocolError("GPU weights require the explicit CUDA IPC utility")
            owner = value.detach().contiguous()
            target.dtype = DTypes.encode(owner.dtype)
            target.shape.extend(owner.shape)
            target.array_kind = common.TensorDescriptor.TORCH
            raw = owner.reshape(-1).view(torch.uint8).numpy()
        else:
            import numpy as np

            owner = np.asarray(value)
            target.dtype = DTypes.encode(owner.dtype.name)
            target.shape.extend(owner.shape)
            if owner.dtype.byteorder == ">":
                owner = owner.astype(owner.dtype.newbyteorder("<"))
            owner = np.ascontiguousarray(owner)
            target.array_kind = common.TensorDescriptor.NUMPY
            raw = owner
        if len(target.shape) > cls.MAX_DIMS:
            raise ProtocolError("tensor dimension limit exceeded")
        writer.buffer(
            target.buffer,
            raw,
            target.dtype,
            identity=("tensor", id(value)),
            owner=owner,
        )

    @classmethod
    def decode(cls, source, reader):
        if len(source.shape) > cls.MAX_DIMS or source.dtype != source.buffer.dtype:
            raise ProtocolError("tensor shape or dtype is invalid")
        if math.prod(source.shape) != source.buffer.logical_length:
            raise ProtocolError("tensor shape does not match buffer length")
        raw, _ = reader.view(source.buffer, dtype=source.dtype)
        name = DTypes.NAMES[source.dtype][0]
        if source.array_kind == common.TensorDescriptor.TORCH:
            import torch

            dtype = getattr(torch, name, None)
            if dtype is None:
                raise ProtocolError(f"local torch lacks dtype {name}")
            if not raw.nbytes:
                return torch.empty(tuple(source.shape), dtype=dtype)
            # Torch has no read-only tensor type; downstream workers own their staging.
            return torch.frombuffer(raw, dtype=dtype).reshape(tuple(source.shape))
        if source.array_kind != common.TensorDescriptor.NUMPY:
            raise ProtocolError("unknown CPU array kind")
        import numpy as np

        try:
            dtype = np.dtype(name).newbyteorder("<")
        except TypeError as error:
            raise ProtocolError(f"numpy cannot represent dtype {name}") from error
        return np.frombuffer(raw, dtype=dtype).reshape(tuple(source.shape))

    @classmethod
    def encode_cuda(cls, target, handle):
        import torch
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        if (
            tuple(inspect.signature(rebuild_cuda_tensor).parameters)
            != cls.CUDA_PARAMETERS
        ):
            raise ProtocolError(
                "local PyTorch CUDA reducer has an unsupported signature"
            )

        if not isinstance(handle, tuple) or len(handle) != 2:
            raise ProtocolError("CUDA IPC handle must use the supported reducer ABI")
        function, args = handle
        if (
            function is not rebuild_cuda_tensor
            or not isinstance(args, tuple)
            or len(args) != 15
        ):
            raise ProtocolError(
                "unsupported CUDA IPC reducer ABI; expected rebuild_cuda_tensor/15"
            )
        if args[0] is not torch.Tensor or args[4] is not torch.storage.TypedStorage:
            raise ProtocolError("unsupported CUDA IPC tensor/storage class")
        target.reducer_abi = cls.CUDA_REDUCER_ABI
        target.dtype = DTypes.encode(args[5])
        target.shape.extend(args[1])
        target.stride.extend(args[2])
        target.tensor_offset = args[3]
        target.device = args[6]
        target.storage_handle = args[7]
        target.storage_size_bytes = args[8]
        target.storage_offset_bytes = args[9]
        target.requires_grad = args[10]
        target.ref_counter_handle = args[11]
        target.ref_counter_offset = args[12]
        target.event_handle = args[13]
        target.event_sync_required = args[14]
        cls.validate_cuda(target)

    @classmethod
    def validate_cuda(cls, source):
        if source.reducer_abi != cls.CUDA_REDUCER_ABI:
            raise ProtocolError("unsupported CUDA IPC reducer ABI version")
        if len(source.shape) != len(source.stride) or len(source.shape) > cls.MAX_DIMS:
            raise ProtocolError("invalid CUDA IPC tensor dimensions")
        if source.device < 0 or any(stride < 0 for stride in source.stride):
            raise ProtocolError("invalid CUDA IPC device or stride")
        size = DTypes.size(source.dtype)
        extent = source.tensor_offset
        if all(source.shape):
            extent += 1 + sum(
                (length - 1) * stride
                for length, stride in zip(source.shape, source.stride)
            )
        if extent * size > source.storage_size_bytes:
            raise ProtocolError("CUDA IPC tensor exceeds its storage")
        if not source.storage_handle or len(source.storage_handle) > 4096:
            raise ProtocolError("invalid CUDA IPC storage handle")
        if len(source.ref_counter_handle) > 4096 or len(source.event_handle) > 4096:
            raise ProtocolError("invalid CUDA IPC auxiliary handle")

    @classmethod
    def decode_cuda(cls, source):
        cls.validate_cuda(source)
        import torch
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        if (
            tuple(inspect.signature(rebuild_cuda_tensor).parameters)
            != cls.CUDA_PARAMETERS
        ):
            raise ProtocolError(
                "local PyTorch CUDA reducer has an unsupported signature"
            )

        name = DTypes.NAMES[source.dtype][0]
        dtype = getattr(torch, name, None)
        if dtype is None:
            raise ProtocolError(f"local torch lacks dtype {name}")
        return rebuild_cuda_tensor, (
            torch.Tensor,
            tuple(source.shape),
            tuple(source.stride),
            source.tensor_offset,
            torch.storage.TypedStorage,
            dtype,
            source.device,
            source.storage_handle,
            source.storage_size_bytes,
            source.storage_offset_bytes,
            source.requires_grad,
            source.ref_counter_handle,
            source.ref_counter_offset,
            source.event_handle,
            source.event_sync_required,
        )

    @staticmethod
    def encode_bucket(target, bucket):
        for name, value in bucket.items():
            item = target.add(
                name=name,
                dtype=DTypes.encode(value["dtype"]),
                offset=value["offset"],
                nbytes=value["nbytes"],
            )
            item.shape.extend(value["shape"])
            if math.prod(item.shape) * DTypes.size(item.dtype) != item.nbytes:
                raise ProtocolError("weight shape does not match byte length")

    @staticmethod
    def decode_bucket(source):
        result = {}
        for item in source:
            if (
                item.name in result
                or math.prod(item.shape) * DTypes.size(item.dtype) != item.nbytes
            ):
                raise ProtocolError("invalid or duplicate weight slice")
            if item.offset % DTypes.size(item.dtype):
                raise ProtocolError("unaligned weight slice")
            result[item.name] = {
                "shape": tuple(item.shape),
                "dtype": "torch." + DTypes.NAMES[item.dtype][0],
                "offset": item.offset,
                "nbytes": item.nbytes,
            }
        return result

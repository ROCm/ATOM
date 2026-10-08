"""Typed cancellation, utility, shutdown, and probe messages."""

from typing import ClassVar

from .frame_utils import (
    ChannelCodec,
    DecodedMessage,
    Fields,
    FrameReader,
    FrameWriter,
    MetaCodec,
    ProtocolError,
)
from .proto import control_pb2 as control
from .tensor_adapter import TensorAdapter


class ControlCodec(ChannelCodec):
    SCHEMA_ID = 0x41544354
    MESSAGE = control.ControlMessage
    EMPTY = frozenset(
        (
            "clear_kv_cache",
            "start_profile",
            "stop_profile",
            "get_mtp_stats",
            "get_mtp_statistics",
            "get_cache_statistics",
            "get_weight_sync_status",
        )
    )
    # Fields every utility command may carry, kept on ``UtilityRequest`` itself
    # and handed back to the handler inside its args.
    COMMON: ClassVar[tuple] = ("request_id", "timeout", "deadline", "sync_id")
    TAGS: ClassVar[dict] = {"weights": control.WEIGHTS, "kv_cache": control.KV_CACHE}

    def encode(self, kind, payload, *, metadata=None, **context):
        kind = Fields.kind(kind)
        metadata = metadata or {}
        message, writer = self._new(**context)
        if kind in ("ABORT", "CANCEL"):
            ids = (
                payload.get("request_ids", []) if isinstance(payload, dict) else payload
            )
            if isinstance(ids, int):
                ids = [ids]
            message.cancel.request_ids.extend(ids)
            if isinstance(payload, dict):
                message.cancel.admission_ids.extend(payload.get("admission_ids", []))
                message.cancel.submission_sequence = payload.get(
                    "submission_sequence", 0
                )
            message.cancel.SetInParent()
        elif kind == "SHUTDOWN":
            message.shutdown.reason = Fields.get(payload, "reason", "") or ""
        elif kind == "BUFFER_PROBE":
            message.buffer_probe.probe_id = payload["probe_id"]
            descriptor = payload.get("buffer", payload.get("descriptor"))
            if descriptor is not None:
                FrameWriter.shm(message.buffer_probe.buffer, descriptor)
            message.buffer_probe.expected = payload.get("expected", b"")
            if payload.get("registration") is not None:
                Fields.assign(
                    message.buffer_probe.registration,
                    payload["registration"],
                    ("root_dir", "arena_id", "epoch", "capacity_bytes"),
                )
        elif kind == "UTILITY":
            command, args = payload
            args = dict(args or {})
            utility = message.utility
            for name in self.COMMON:
                value = args.pop(name, None)
                if value is not None:
                    setattr(utility, name, value)
            utility.expected_ranks.extend(metadata.get("expected_ranks", []))
            self._encode_utility(utility, command, args, writer, metadata)
        else:
            raise ProtocolError(f"unsupported control kind: {kind}")
        return self._finish(message, writer)

    def _encode_utility(self, utility, command, args, writer, metadata):
        if command in self.EMPTY:
            if args:
                raise ProtocolError(f"{command} does not accept arguments")
            getattr(utility, command).SetInParent()
        elif command == "update_weights":
            target = utility.update_weights
            target.flush_cache = args.get("flush_cache", True)
            for name, tensor in args.get("named_tensors", []):
                TensorAdapter.encode(
                    target.named_tensors.add(), tensor, writer, name=name
                )
        elif command == "update_weights_shm":
            target = utility.update_weights_shm
            target.shm_name = args.get("shm_name", "")
            target.is_last = args.get("is_last", True)
            TensorAdapter.encode_bucket(target.bucket, args.get("bucket_meta", {}))
        elif command == "update_weights_ipc":
            if metadata.get("same_host") is False:
                raise ProtocolError("CUDA IPC handles cannot be sent to another host")
            target = utility.update_weights_ipc
            target.is_last = args.get("is_last", True)
            if args.get("ipc_handle") is not None:
                TensorAdapter.encode_cuda(target.ipc_handle, args["ipc_handle"])
            for gpu, handle in (args.get("ipc_handles") or {}).items():
                entry = target.ipc_handles.add(gpu=gpu)
                TensorAdapter.encode_cuda(entry.tensor, handle)
            TensorAdapter.encode_bucket(target.bucket, args.get("bucket_meta", {}))
        elif command in ("release_memory", "resume_memory"):
            target = getattr(utility, command)
            target.SetInParent()
            for tag in args.get("tags", ["weights", "kv_cache"]):
                if tag not in self.TAGS:
                    raise ProtocolError(f"unknown resource tag: {tag}")
                target.tags.append(self.TAGS[tag])
        elif command == "configure_hidden_states":
            target = utility.configure_hidden_states
            target.aux_layer_ids.extend(args.get("aux_layer_ids", []))
            target.config_version = 1
            MetaCodec.encode_object(
                target.mooncake_config, args.get("mooncake_config", {})
            )
        elif command == "abort_request":
            utility.abort_request.req_id = args["req_id"]
        elif command == "collective_rpc":
            target = utility.collective_rpc
            target.method = args["method"]
            target.barrier = bool(args.get("barrier", False))
            MetaCodec.encode(target.args, tuple(args.get("args", ())), exact=True)
            MetaCodec.encode(target.kwargs, dict(args.get("kwargs") or {}), exact=True)
        elif command == "finish_weight_sync":
            target = utility.finish_weight_sync
            target.completed_cmd = args.get("completed_cmd") or ""
            target.completed_request_id = args.get("completed_request_id") or ""
        elif command == "discard_failed_weight_sync":
            target = utility.discard_failed_weight_sync
            target.failed_cmd = args.get("failed_cmd") or ""
            if args.get("error") is not None:
                target.error = str(args["error"])
        else:
            raise ProtocolError(f"unknown utility command: {command}")

    def decode(self, frames, **context):
        message, reader, metadata = self._parse(frames, **context)
        kind = message.WhichOneof("payload")
        if kind == "cancel":
            metadata.update(
                admission_ids=list(message.cancel.admission_ids),
                submission_sequence=message.cancel.submission_sequence,
            )
            payload, name = list(message.cancel.request_ids), "ABORT"
        elif kind == "shutdown":
            payload, name = {"reason": message.shutdown.reason}, "SHUTDOWN"
        elif kind == "buffer_probe":
            probe = message.buffer_probe
            if probe.HasField("buffer") and FrameReader.validate(probe.buffer) != "shm":
                raise ProtocolError("buffer probe requires shared memory")
            payload = {
                "probe_id": probe.probe_id,
                "buffer": (
                    reader.prompt(probe.buffer) if probe.HasField("buffer") else None
                ),
                "expected": probe.expected,
                "registration": (
                    {
                        name: getattr(probe.registration, name)
                        for name in ("root_dir", "arena_id", "epoch", "capacity_bytes")
                    }
                    if probe.HasField("registration")
                    else None
                ),
            }
            name = "BUFFER_PROBE"
        elif kind == "utility":
            utility = message.utility
            metadata.update(expected_ranks=list(utility.expected_ranks))
            command, args = self._decode_utility(utility, reader)
            for name in self.COMMON:
                if name == "request_id" or utility.HasField(name):
                    value = getattr(utility, name)
                    if value != "":
                        args[name] = value
            payload = command, args
            name = "UTILITY"
        else:
            raise ProtocolError("control payload is missing or unsupported")
        reader.finish()
        return DecodedMessage(
            Fields.kind_value(name), payload, metadata, list(reader.frames)
        )

    def _decode_utility(self, utility, reader):
        command = utility.WhichOneof("command")
        if command is None:
            raise ProtocolError("utility command is missing or unsupported")
        source = getattr(utility, command)
        if command in self.EMPTY:
            args = {}
        elif command == "update_weights":
            args = {
                "named_tensors": [
                    (item.name, TensorAdapter.decode(item, reader))
                    for item in source.named_tensors
                ],
                "flush_cache": source.flush_cache,
            }
        elif command == "update_weights_shm":
            args = {
                "shm_name": source.shm_name,
                "bucket_meta": TensorAdapter.decode_bucket(source.bucket),
                "is_last": source.is_last,
            }
        elif command == "update_weights_ipc":
            handles = {}
            for entry in source.ipc_handles:
                if entry.gpu in handles:
                    raise ProtocolError("duplicate CUDA IPC device")
                handles[entry.gpu] = TensorAdapter.decode_cuda(entry.tensor)
            args = {
                "ipc_handle": (
                    TensorAdapter.decode_cuda(source.ipc_handle)
                    if source.HasField("ipc_handle")
                    else None
                ),
                "ipc_handles": handles or None,
                "bucket_meta": TensorAdapter.decode_bucket(source.bucket),
                "is_last": source.is_last,
            }
        elif command in ("release_memory", "resume_memory"):
            tags = {value: name for name, value in self.TAGS.items()}
            if any(tag not in tags for tag in source.tags):
                raise ProtocolError("unsupported resource tag")
            args = {"tags": [tags[tag] for tag in source.tags]}
        elif command == "abort_request":
            args = {"req_id": source.req_id}
        elif command == "collective_rpc":
            call_args = MetaCodec.decode(source.args, exact=True)
            kwargs = MetaCodec.decode(source.kwargs, exact=True)
            if type(call_args) is not tuple or type(kwargs) is not dict:
                raise ProtocolError("collective_rpc needs tuple args and dict kwargs")
            args = {
                "method": source.method,
                "args": call_args,
                "kwargs": kwargs,
                "barrier": source.barrier,
            }
        elif command == "finish_weight_sync":
            args = {
                "completed_cmd": source.completed_cmd,
                "completed_request_id": source.completed_request_id or None,
            }
        elif command == "discard_failed_weight_sync":
            args = {"failed_cmd": source.failed_cmd}
            if source.HasField("error"):
                args["error"] = source.error
        else:
            if source.config_version != 1:
                raise ProtocolError("unsupported Mooncake configuration version")
            args = {
                "aux_layer_ids": list(source.aux_layer_ids),
                "mooncake_config": MetaCodec.decode_object(source.mooncake_config),
            }
        return command, args

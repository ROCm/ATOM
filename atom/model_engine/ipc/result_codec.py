"""Output-only protobuf codec for the Core result socket."""

from typing import ClassVar

from atom.model_engine.request import RequestOutput

from .frame_utils import ChannelCodec, DecodedMessage, Fields, MetaCodec, ProtocolError
from .proto import common_pb2 as common
from .proto import result_pb2 as result


class StatisticsCodec:
    MTP = ("enabled", "total_draft_tokens", "total_accepted_tokens", "acceptance_rate")
    CACHE = (
        "enabled",
        "requests",
        "cached_tokens",
        "compressed_tokens",
        "wanted_tokens",
        "reusable_tokens",
        "full_tokens",
        "offload_tokens",
    )
    METRICS = (
        "enabled",
        "role",
        "requests_running",
        "requests_waiting",
        "requests_parked_kv_load",
        "requests_partial_prefill",
        "requests_finished",
        "prompt_tokens",
        "generation_tokens",
        "preemptions",
    )
    KV = ("used", "free", "total", "indexed", "evictable", "vacant")
    SCHEDULER = ("running", "waiting", "waiting_kv")

    @classmethod
    def encode(cls, target, values, names, *, skip=()):
        target.SetInParent()
        for name, value in values.items():
            if name in skip:
                continue
            if name in names:
                if value is None:
                    target.null_fields.append(name)
                else:
                    setattr(target, name, value)
            else:
                MetaCodec.encode_number(target.counters[name], value)

    @classmethod
    def decode(cls, source, names):
        values = Fields.present(source, names)
        for name in source.null_fields:
            if name not in names or name in values:
                raise ProtocolError("invalid null metric field")
            values[name] = None
        for name, value in source.counters.items():
            if name in names:
                raise ProtocolError("extension counter shadows a typed metric")
            values[name] = MetaCodec.decode_number(value)
        return values

    @classmethod
    def encode_mtp(cls, target, values):
        cls.encode(target, values, cls.MTP, skip=("distribution",))
        target.has_distribution = "distribution" in values
        for key, value in values.get("distribution", {}).items():
            if type(key) is not int or type(value) is not int:
                raise ProtocolError(
                    "MTP distribution must have integer keys and counters"
                )
            target.distribution[key] = value

    @classmethod
    def decode_mtp(cls, source):
        values = cls.decode(source, cls.MTP)
        if source.has_distribution:
            values["distribution"] = dict(source.distribution)
        return values

    @classmethod
    def encode_metrics(cls, target, values):
        nested = {"mtp", "cache", "offload", "scheduler_metrics"}
        nested.update("kv_blocks_" + name for name in cls.KV)
        cls.encode(target, values, cls.METRICS, skip=nested)
        if "mtp" in values:
            cls.encode_mtp(target.mtp, values["mtp"])
        if "cache" in values:
            cls.encode(target.cache, values["cache"], cls.CACHE)
        for name in cls.KV:
            if "kv_blocks_" + name in values:
                setattr(target.kv_pool, name, values["kv_blocks_" + name])
        if "scheduler_metrics" in values:
            target.scheduler_metrics.SetInParent()
            Fields.assign(
                target.scheduler_metrics, values["scheduler_metrics"], cls.SCHEDULER
            )
        for name, value in values.get("offload", {}).items():
            MetaCodec.encode_number(target.offload[name], value)
        target.has_offload = "offload" in values

    @classmethod
    def decode_metrics(cls, source):
        values = cls.decode(source, cls.METRICS)
        if source.HasField("mtp"):
            values["mtp"] = cls.decode_mtp(source.mtp)
        if source.HasField("cache"):
            values["cache"] = cls.decode(source.cache, cls.CACHE)
        if source.HasField("kv_pool"):
            values.update(
                {
                    "kv_blocks_" + name: value
                    for name, value in Fields.present(source.kv_pool, cls.KV).items()
                }
            )
        if source.HasField("scheduler_metrics"):
            values["scheduler_metrics"] = Fields.present(
                source.scheduler_metrics, cls.SCHEDULER
            )
        if source.has_offload:
            values["offload"] = {
                name: MetaCodec.decode_number(value)
                for name, value in source.offload.items()
            }
        return values


class ResultCodec(ChannelCodec):
    SCHEMA_ID = 0x41545253
    MESSAGE = result.ResultMessage
    COMPLETION_FIELDS = (
        "id",
        "return_logprobs",
        "num_prompt_tokens",
        "num_cached_tokens",
        "arrive_time",
        "first_token_time",
        "leave_time",
        "sibling_index",
        "prefill_start_time",
        "prefill_end_time",
    )
    COMPLETION_OPTIONAL = ("leave_reason", "external_request_id", "parent_request_id")
    RELEASE_FIELDS = (
        "arena_id",
        "arena_epoch",
        "allocation_id",
        "allocation_generation",
        "admission_id",
        "request_id",
        "rank",
        "reason",
    )
    VERSIONS: ClassVar[dict] = {
        "request_version": 0x41545251,
        "control_version": 0x41544354,
        "result_version": SCHEMA_ID,
    }

    def encode(self, kind, payload, *, metadata=None, **context):
        name = Fields.kind(kind)
        metadata = metadata or {}
        message, writer = self._new(**context)
        if name == "READY":
            self._encode_ready(message.ready, payload or {})
        elif name == "STREAM":
            self._batch(payload)
            for value in payload:
                if isinstance(value, tuple):
                    request_id, value = value
                    if request_id != Fields.get(value, "request_id"):
                        raise ProtocolError("stream request IDs disagree")
                target = message.stream.outputs.add()
                Fields.assign(
                    target,
                    value,
                    ("request_id", "finished", "finish_reason", "num_cached_tokens"),
                )
                target.output_tokens.extend(Fields.get(value, "output_tokens", ()))
                kv = Fields.get(value, "kv_transfer_params_output")
                if kv is not None:
                    MetaCodec.encode_object(target.kv_transfer_params_output, kv)
        elif name in ("ADD", "COMPLETION"):
            self._batch(payload)
            for value in payload:
                target = message.completion.outputs.add()
                Fields.assign(
                    target, value, self.COMPLETION_FIELDS + self.COMPLETION_OPTIONAL
                )
                writer.packed(
                    target.completion_token_ids,
                    Fields.get(value, "completion_token_ids", ()),
                    "i",
                    common.INT32,
                )
                logprobs = Fields.get(value, "logprobs", ())
                if target.return_logprobs or len(logprobs):
                    writer.packed(target.logprobs, logprobs, "d", common.FLOAT64)
                kv = Fields.get(value, "kv_transfer_params_output")
                if kv is not None:
                    MetaCodec.encode_object(target.kv_transfer_params_output, kv)
        elif name == "METRICS":
            StatisticsCodec.encode_metrics(message.metrics, payload)
        elif name == "UTILITY_RESPONSE":
            target = message.utility_response
            target.command = payload["cmd"]
            target.request_id = payload.get("request_id") or ""
            target.rank = payload.get("rank", metadata.get("rank", 0))
            if payload.get("error") is not None:
                target.error = str(payload["error"])
            value = payload.get("result")
            if target.command == "get_mtp_statistics" and value is not None:
                StatisticsCodec.encode_mtp(target.mtp, value)
            elif target.command == "get_cache_statistics" and value is not None:
                StatisticsCodec.encode(target.cache, value, StatisticsCodec.CACHE)
            elif target.command == "collective_rpc" and "error" not in payload:
                self._encode_rpc_result(target.collective_rpc, payload)
            else:
                try:
                    MetaCodec.encode(target.acknowledgement, value, exact=True)
                except ProtocolError as exc:
                    # Answer the caller instead of failing the whole channel.
                    target.acknowledgement.Clear()
                    target.acknowledgement.null_value = True
                    target.error = f"utility result cannot be sent: {exc}"
        elif name in ("BUFFER_RELEASE", "RELEASE"):
            Fields.assign(message.release, payload, self.RELEASE_FIELDS)
            message.release.SetInParent()
        elif name in ("ERROR", "EXECUTOR_FAILED"):
            if isinstance(payload, str):
                payload = {"message": payload}
            Fields.assign(
                message.error, payload, ("code", "message", "request_id", "command_id")
            )
            message.error.SetInParent()
        elif name in ("STOPPED", "SHUTDOWN"):
            message.stopped.reason = Fields.get(payload, "reason", "") or ""
        elif name == "BUFFER_PROBE_RESULT":
            Fields.assign(
                message.buffer_probe_result,
                payload,
                ("probe_id", "accessible", "error", "digest", "arena_epoch"),
            )
            message.buffer_probe_result.SetInParent()
        else:
            raise ProtocolError(f"unsupported result kind: {name}")
        return self._finish(message, writer)

    @staticmethod
    def _encode_rpc_result(target, payload):
        target.method = payload.get("method") or ""
        target.worker_world_size = payload.get("worker_world_size", 0)
        for entry in payload.get("results", ()):
            item = target.results.add(tp_rank=entry.get("tp_rank", -1))
            if entry.get("error") is not None:
                item.error = str(entry["error"])
            # The worker returns only `is_plain_rpc_value` data, which exact
            # metadata carries as it was -- unless it is past the size limits,
            # which fail that rank rather than the whole reply.
            try:
                MetaCodec.encode(item.value, entry.get("value"), exact=True)
            except ProtocolError as exc:
                item.value.Clear()
                item.value.null_value = True
                item.error = f"return value cannot be sent: {exc}"

    @staticmethod
    def _decode_rpc_result(source):
        world_size = source.worker_world_size
        return {
            "method": source.method,
            "worker_world_size": world_size,
            # Kept for callers of the older reply; prefer worker_world_size.
            "tp_world_size": world_size,
            "results": [
                {
                    "tp_rank": item.tp_rank,
                    "value": MetaCodec.decode(item.value, exact=True),
                    "error": item.error if item.HasField("error") else None,
                }
                for item in source.results
            ],
        }

    def _encode_ready(self, target, payload):
        for name, schema_id in self.VERSIONS.items():
            version = getattr(target, name)
            supplied = payload.get(name, {})
            version.schema_id = supplied.get("schema_id", schema_id)
            version.major = supplied.get("major", 1)
            version.minor = supplied.get("minor", 0)
        Fields.assign(target, payload, ("max_pool_tokens", "role", "host_id", "rank"))
        target.buffer_modes.extend(payload.get("buffer_modes", ["raw", "shm"]))
        if payload.get("arena") is not None:
            Fields.assign(
                target.arena,
                payload["arena"],
                ("root_dir", "arena_id", "epoch", "capacity_bytes"),
            )

    def decode(self, frames, **context):
        message, reader, metadata = self._parse(frames, **context)
        kind = message.WhichOneof("payload")
        if kind == "ready":
            payload = self._decode_ready(message.ready)
            name = "READY"
        elif kind == "stream":
            self._batch(message.stream.outputs)
            payload = [
                RequestOutput(
                    request_id=item.request_id,
                    output_tokens=list(item.output_tokens),
                    finished=item.finished,
                    finish_reason=(
                        item.finish_reason if item.HasField("finish_reason") else None
                    ),
                    kv_transfer_params_output=(
                        MetaCodec.decode_object(item.kv_transfer_params_output)
                        if item.HasField("kv_transfer_params_output")
                        else None
                    ),
                    num_cached_tokens=item.num_cached_tokens,
                )
                for item in message.stream.outputs
            ]
            payload = [(value.request_id, value) for value in payload]
            name = "STREAM"
        elif kind == "completion":
            self._batch(message.completion.outputs)
            payload = []
            for item in message.completion.outputs:
                values = {
                    field: getattr(item, field) for field in self.COMPLETION_FIELDS
                }
                values.update(
                    {
                        field: getattr(item, field) if item.HasField(field) else None
                        for field in self.COMPLETION_OPTIONAL
                    }
                )
                tokens, _ = reader.view(item.completion_token_ids, dtype=common.INT32)
                logprobs = (
                    reader.view(item.logprobs, dtype=common.FLOAT64)[0].cast("d")
                    if item.HasField("logprobs")
                    else ()
                )
                values.update(
                    completion_token_ids=tokens.cast("i"),
                    logprobs=logprobs,
                    leave_reason=(
                        item.leave_reason if item.HasField("leave_reason") else None
                    ),
                    kv_transfer_params_output=(
                        MetaCodec.decode_object(item.kv_transfer_params_output)
                        if item.HasField("kv_transfer_params_output")
                        else None
                    ),
                )
                if item.return_logprobs and len(logprobs) != len(
                    values["completion_token_ids"]
                ):
                    raise ProtocolError(
                        "completion logprobs length does not match generated tokens"
                    )
                payload.append(values)
            name = "ADD"
        elif kind == "metrics":
            payload, name = StatisticsCodec.decode_metrics(message.metrics), "METRICS"
        elif kind == "utility_response":
            target = message.utility_response
            metadata.update(rank=target.rank)
            payload = {
                "cmd": target.command,
                "request_id": target.request_id or None,
                "rank": target.rank,
            }
            result_kind = target.WhichOneof("result")
            if result_kind == "mtp":
                payload["result"] = StatisticsCodec.decode_mtp(target.mtp)
            elif result_kind == "cache":
                payload["result"] = StatisticsCodec.decode(
                    target.cache, StatisticsCodec.CACHE
                )
            elif result_kind == "collective_rpc":
                payload.update(self._decode_rpc_result(target.collective_rpc))
            elif result_kind == "acknowledgement":
                payload["result"] = MetaCodec.decode(target.acknowledgement, exact=True)
            else:
                raise ProtocolError("utility result is missing")
            if target.HasField("error"):
                payload["error"] = target.error
            name = "UTILITY_RESPONSE"
        elif kind == "release":
            payload = {
                field: getattr(message.release, field) for field in self.RELEASE_FIELDS
            }
            name = "BUFFER_RELEASE"
        elif kind == "error":
            payload = {
                "code": message.error.code,
                "message": message.error.message,
                "command_id": message.error.command_id,
            }
            if message.error.HasField("request_id"):
                payload["request_id"] = message.error.request_id
            name = "ERROR"
        elif kind == "stopped":
            payload, name = {"reason": message.stopped.reason}, "STOPPED"
        elif kind == "buffer_probe_result":
            payload = {
                field: getattr(message.buffer_probe_result, field)
                for field in (
                    "probe_id",
                    "accessible",
                    "error",
                    "digest",
                    "arena_epoch",
                )
            }
            name = "BUFFER_PROBE_RESULT"
        else:
            raise ProtocolError("result payload is missing or unsupported")
        reader.finish()
        return DecodedMessage(
            Fields.kind_value(name), payload, metadata, list(reader.frames)
        )

    def _decode_ready(self, source):
        values = Fields.present(source, ("max_pool_tokens",))
        for name, schema_id in self.VERSIONS.items():
            version = getattr(source, name)
            if version.schema_id != schema_id or version.major != 1:
                raise ProtocolError(f"incompatible Ready {name}")
            values[name] = {
                "schema_id": version.schema_id,
                "major": version.major,
                "minor": version.minor,
            }
        values.update(
            role=source.role,
            host_id=source.host_id,
            rank=source.rank,
            buffer_modes=list(source.buffer_modes),
        )
        if source.HasField("arena"):
            values["arena"] = {
                name: getattr(source.arena, name)
                for name in ("root_dir", "arena_id", "epoch", "capacity_bytes")
            }
        return values

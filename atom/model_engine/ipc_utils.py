# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
CUDA / ROCm IPC helpers for sharing GPU tensors across processes.

Uses tensor._share_cuda_() / UntypedStorage._new_shared_cuda() for the
low-level IPC handle path (hipIpcGetMemHandle / hipIpcOpenMemHandle on ROCm).
Both processes must be on the same physical GPU device.

Phase 1 (KV cache sharing):
  - export_kv_cache_handle  — called by PrefillEngineCore after allocate_kv_cache()
  - import_kv_cache         — called by DecodeEngineCore at startup

Phase 2 (weight sharing):
  - export_model_weight_handles  — called by PrefillEngineCore after load_model()
  - import_model_weights         — called by DecodeEngineCore at startup (frees own copy)
"""

import base64
import importlib
import logging
from enum import Enum

import numpy as np
import torch
import torch.nn as nn
from google.protobuf import empty_pb2, json_format, struct_pb2

from atom.proto.engine import disagg_proto
from atom.proto.engine import engine_core_proto

logger = logging.getLogger("atom")
WIRE_VERSION = 1


def _export_tensor(t: torch.Tensor) -> dict:
    """Serialize a CUDA tensor to a dict that can be pickled and sent cross-process.

    Uses tensor._share_cuda_() which calls hipIpcGetMemHandle on ROCm.
    Returns metadata needed to reconstruct the tensor on the other side.
    """
    t = t.contiguous()
    share_cuda_args = t.untyped_storage()._share_cuda_()
    return {
        "share_cuda_args": share_cuda_args,
        "dtype": t.dtype,
        "shape": t.shape,
        "stride": t.stride(),
        "storage_offset": t.storage_offset(),
    }


def _import_tensor(meta: dict) -> torch.Tensor:
    """Reconstruct a CUDA tensor from the dict produced by _export_tensor.

    Calls UntypedStorage._new_shared_cuda() which calls hipIpcOpenMemHandle.
    """
    storage = torch.UntypedStorage._new_shared_cuda(*meta["share_cuda_args"])
    t = torch.empty(0, dtype=meta["dtype"], device="cuda")
    t.set_(storage, meta["storage_offset"], meta["shape"], meta["stride"])
    return t


# ---------------------------------------------------------------------------
# KV cache (Phase 1)
# ---------------------------------------------------------------------------


def export_kv_cache_handle(
    kv_cache: torch.Tensor, kv_scale: torch.Tensor | None = None
) -> dict:
    """Export kv_cache (and optionally kv_scale for fp8) as CUDA IPC handles.

    Must be called from the process that allocated the tensor (prefill).
    Returns a dict that can be pickled and sent over ZMQ to the decode process.
    """
    result = {"kv_cache": _export_tensor(kv_cache)}
    if kv_scale is not None:
        result["kv_scale"] = _export_tensor(kv_scale)
    return result


def import_kv_cache(meta: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reconstruct kv_cache (and kv_scale if present) from CUDA IPC handles.

    Must be called from the consumer process (decode).
    Returns (kv_cache, kv_scale) — kv_scale is None when not fp8.
    The returned tensors share GPU memory with prefill's allocation — no copy.
    """
    kv_cache = _import_tensor(meta["kv_cache"])
    kv_scale = _import_tensor(meta["kv_scale"]) if "kv_scale" in meta else None
    return kv_cache, kv_scale


# ---------------------------------------------------------------------------
# Model weights (Phase 2)
# ---------------------------------------------------------------------------


def export_model_weight_handles(model: nn.Module) -> dict:
    """Export all model parameter tensors as CUDA IPC handles.

    Also exports MLA weight-absorbed tensors (W_K/W_K_scale/W_V/W_V_scale)
    which are plain tensor attributes set by process_weights_after_loading(),
    not nn.Parameters, so named_parameters() misses them.

    Must be called from the process that allocated the weights (prefill),
    after load_model() completes.  Returns a dict {key: meta_dict}.
    """
    handles = {}
    # Parameters. remove_duplicate=False so a Parameter registered under multiple
    # names (e.g. e_score_correction_bias, shared by gate + experts) is exported
    # under EVERY name — otherwise the consumer only materializes one of the
    # aliased registrations and the other stays on meta.
    for name, param in model.named_parameters(remove_duplicate=False):
        handles[f"__param__{name}"] = _export_tensor(param.data)
    # Registered buffers (non-persistent included).
    for name, buf in model.named_buffers():
        if isinstance(buf, torch.Tensor) and buf.is_cuda and buf.numel() > 0:
            handles[f"__buf__{name}"] = _export_tensor(buf)
    # Plain tensor attributes set by process_weights_after_loading() — e.g. the
    # MLA absorbed W_K/W_V — which are neither Parameters nor registered buffers.
    for mod_name, mod in model.named_modules():
        for attr, val in list(mod.__dict__.items()):
            if (
                isinstance(val, torch.Tensor)
                and not isinstance(val, nn.Parameter)
                and val.is_cuda
                and val.numel() > 0
            ):
                key = f"{mod_name}.{attr}" if mod_name else attr
                handles[f"__attr__{key}"] = _export_tensor(val)

    return handles


def import_model_weights(model: nn.Module, handles: dict) -> None:
    """Replace model parameters with views into another process's GPU allocation.

    Also restores MLA absorbed tensors exported by export_model_weight_handles.

    Must be called from the consumer process (decode) after receiving the
    handles dict from the producer (prefill).  After this call the decode
    model's parameters point into prefill's GPU memory — zero additional bytes
    are allocated.  The decode process's original weight tensors are freed when
    their reference counts drop to zero.
    """
    modules = dict(model.named_modules())
    # remove_duplicate=False to match the export and to materialize every
    # registration of a shared Parameter (see export note).
    params = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers())

    for key, meta in handles.items():
        t = _import_tensor(meta)
        if key.startswith("__param__"):
            # Rebuild the Parameter around the imported CUDA view (set_data fails
            # for meta->cuda). Create the slot if the consumer's meta model lacks
            # it (process_weights_after_loading may add params on the producer).
            name = key[len("__param__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            rg = params[name].requires_grad if name in params else False
            mod._parameters[attr] = nn.Parameter(t, requires_grad=rg)
        elif key.startswith("__buf__"):
            # Keep decode's locally-built real buffers (e.g. RoPE caches built
            # during construction); only fill buffers it is missing or left on
            # meta (those created inside process_weights_after_loading).
            name = key[len("__buf__") :]
            existing = buffers.get(name)
            if existing is None or existing.is_meta:
                parent, _, attr = name.rpartition(".")
                mod = modules.get(parent, model)
                if mod is not None:
                    mod._buffers[attr] = t
        elif key.startswith("__attr__"):
            # Plain tensor attribute (e.g. MLA W_K/W_V from process_weights).
            name = key[len("__attr__") :]
            parent, _, attr = name.rpartition(".")
            mod = modules.get(parent, model)
            if mod is not None:
                setattr(mod, attr, t)

    leftover = [n for n, p in model.named_parameters() if p.is_meta] + [
        n for n, b in model.named_buffers() if isinstance(b, torch.Tensor) and b.is_meta
    ]
    if leftover:
        logger.warning(
            f"[WT-IMPORT] {len(leftover)} tensors still on meta after import "
            f"(not exported by producer): {leftover[:12]}"
        )


class EngineCoreIpcCodec:
    """Protobuf codecs for EngineCore request, response, and stream frames."""

    @staticmethod
    def _to_struct(value: dict | None) -> struct_pb2.Struct:
        message = struct_pb2.Struct()
        if value is not None:
            json_format.ParseDict(value, message)
        return message

    @staticmethod
    def _from_struct(message: struct_pb2.Struct) -> dict | None:
        if not message.fields:
            return None
        return json_format.MessageToDict(message)

    @staticmethod
    def _to_utility_value(value):
        """Convert utility payloads to lossless protobuf-Struct-compatible data."""
        if isinstance(value, bytes):
            return {
                "__atom_ipc_type__": "bytes",
                "value": base64.b64encode(value).decode("ascii"),
            }
        if isinstance(value, tuple):
            return {
                "__atom_ipc_type__": "tuple",
                "value": [EngineCoreIpcCodec._to_utility_value(item) for item in value],
            }
        if isinstance(value, int) and not isinstance(value, bool):
            # Struct stores numbers as doubles, which cannot exactly represent
            # pointer-sized CUDA IPC metadata or every int64 sequence id.
            return {"__atom_ipc_type__": "int", "value": str(value)}
        if isinstance(value, list):
            return [EngineCoreIpcCodec._to_utility_value(item) for item in value]
        if isinstance(value, torch.dtype):
            return {
                "__atom_ipc_type__": "torch_dtype",
                "value": str(value).removeprefix("torch."),
            }
        if isinstance(value, torch.device):
            return {
                "__atom_ipc_type__": "torch_device",
                "value": str(value),
            }
        if isinstance(value, type) or callable(value):
            module = getattr(value, "__module__", "")
            qualname = getattr(value, "__qualname__", "")
            if not module.startswith("torch") or not qualname or "<locals>" in qualname:
                raise TypeError(
                    "Utility IPC only supports importable PyTorch callables/types, "
                    f"got {value!r}"
                )
            return {
                "__atom_ipc_type__": "torch_symbol",
                "kind": "type" if isinstance(value, type) else "callable",
                "module": module,
                "qualname": qualname,
            }
        if isinstance(value, dict):
            return {
                "__atom_ipc_type__": "dict",
                "value": [
                    {
                        "key": EngineCoreIpcCodec._to_utility_value(key),
                        "value": EngineCoreIpcCodec._to_utility_value(item),
                    }
                    for key, item in value.items()
                ],
            }
        return value

    @staticmethod
    def _from_utility_value(value):
        if isinstance(value, list):
            return [EngineCoreIpcCodec._from_utility_value(item) for item in value]
        if isinstance(value, dict):
            value_type = value.get("__atom_ipc_type__")
            if value_type == "bytes":
                return base64.b64decode(value["value"])
            if value_type == "tuple":
                return tuple(
                    EngineCoreIpcCodec._from_utility_value(item)
                    for item in value["value"]
                )
            if value_type == "int":
                return int(value["value"])
            if value_type == "torch_dtype":
                dtype = getattr(torch, value["value"], None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError(f"Unknown torch dtype {value['value']!r}")
                return dtype
            if value_type == "torch_device":
                return torch.device(value["value"])
            if value_type == "torch_symbol":
                module = value["module"]
                if not module.startswith("torch"):
                    raise ValueError(f"Refusing non-PyTorch utility symbol {module!r}")
                if value["kind"] == "callable" and (
                    module,
                    value["qualname"],
                ) != (
                    "torch.multiprocessing.reductions",
                    "rebuild_cuda_tensor",
                ):
                    raise ValueError(
                        "Refusing unexpected PyTorch utility callable "
                        f"{module}.{value['qualname']}"
                    )
                symbol = importlib.import_module(module)
                for part in value["qualname"].split("."):
                    symbol = getattr(symbol, part)
                return symbol
            if value_type == "dict":
                return {
                    EngineCoreIpcCodec._from_utility_value(item["key"]): (
                        EngineCoreIpcCodec._from_utility_value(item["value"])
                    )
                    for item in value["value"]
                }
            return {
                key: EngineCoreIpcCodec._from_utility_value(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _to_tensor(value: torch.Tensor | np.ndarray) -> engine_core_proto.Tensor:
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected tensor or ndarray, got {type(value).__name__}")
        value = value.detach().to(device="cpu").contiguous()
        return engine_core_proto.Tensor(
            dtype=str(value.dtype).removeprefix("torch."),
            shape=value.shape,
            data=value.view(torch.uint8).numpy().tobytes(),
        )

    @staticmethod
    def _from_tensor(message: engine_core_proto.Tensor) -> torch.Tensor:
        dtype = getattr(torch, message.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unknown torch tensor dtype {message.dtype!r}")
        shape = tuple(message.shape)
        if not message.data:
            return torch.empty(shape, dtype=dtype)
        return (
            torch.frombuffer(bytearray(message.data), dtype=dtype)
            .reshape(shape)
            .clone()
        )

    @staticmethod
    def _to_multimodal_data(value: dict | None) -> engine_core_proto.MultimodalData:
        message = engine_core_proto.MultimodalData()
        if value is not None:
            for key, tensor in value.items():
                message.tensors[key].CopyFrom(EngineCoreIpcCodec._to_tensor(tensor))
        return message

    @staticmethod
    def _from_multimodal_data(
        message: engine_core_proto.MultimodalData,
    ) -> dict[str, torch.Tensor] | None:
        if not message.tensors:
            return None
        return {
            key: EngineCoreIpcCodec._from_tensor(tensor)
            for key, tensor in message.tensors.items()
        }

    @staticmethod
    def _to_ndarray(value: np.ndarray) -> engine_core_proto.Ndarray:
        value = np.ascontiguousarray(value)
        return engine_core_proto.Ndarray(
            dtype=value.dtype.str, shape=value.shape, data=value.tobytes()
        )

    @staticmethod
    def _from_ndarray(message: engine_core_proto.Ndarray) -> np.ndarray:
        if not message.dtype:
            return np.array([], dtype=np.int32)
        return np.frombuffer(message.data, dtype=np.dtype(message.dtype)).reshape(
            tuple(message.shape)
        ).copy()

    @staticmethod
    def _sequence_status_value(status: Enum) -> int:
        return engine_core_proto.SequenceStatus.Value(
            f"SEQUENCE_STATUS_{status.name}"
        )

    @staticmethod
    def _sequence_type_value(sequence_type: Enum) -> int:
        return engine_core_proto.SequenceType.Value(f"SEQUENCE_TYPE_{sequence_type.name}")

    @staticmethod
    def encode_sequence(sequence) -> engine_core_proto.Sequence:
        if isinstance(sequence, engine_core_proto.Sequence):
            message = engine_core_proto.Sequence()
            message.CopyFrom(sequence)
            return message
        sampling = engine_core_proto.SamplingParameters(
            temperature=sequence.temperature,
            top_k=sequence.top_k,
            top_p=sequence.top_p,
            max_tokens=sequence.max_tokens,
            ignore_eos=sequence.ignore_eos,
            stop_strings=sequence.stop_strings or [],
            return_logprobs=sequence.return_logprobs,
        )
        message = engine_core_proto.Sequence(
            id=sequence.id,
            status=EngineCoreIpcCodec._sequence_status_value(sequence.status),
            type=EngineCoreIpcCodec._sequence_type_value(sequence.type),
            block_size=sequence.block_size,
            token_ids=sequence.token_ids,
            output_tokens=sequence.output_tokens,
            block_table=sequence.block_table,
            stop_token_sequences=[
                engine_core_proto.TokenSequence(values=token_ids)
                for token_ids in sequence.stop_token_sequences
            ],
            last_token=sequence.last_token,
            num_tokens=sequence.num_tokens,
            num_prompt_tokens=sequence.num_prompt_tokens,
            num_cached_tokens=sequence.num_cached_tokens,
            num_hashed_tokens=sequence.num_hashed_tokens,
            num_compressed_hit_blocks=sequence.num_compressed_hit_blocks,
            num_wanted_hit_blocks=sequence.num_wanted_hit_blocks,
            checkpoint_demand_pos=sequence.checkpoint_demand_pos,
            checkpoint_demand_counted=sequence.checkpoint_demand_counted,
            checkpoint_demand_declined=sequence.checkpoint_demand_declined,
            last_checkpoint_pos=sequence.last_checkpoint_pos,
            prefix_cache_hit_tokens=sequence.prefix_cache_hit_tokens,
            is_partial_prefill=sequence.is_partial_prefill,
            per_req_cache_group=sequence.per_req_cache_group,
            state_fork_src=sequence.state_fork_src,
            is_first_decode=sequence.is_first_decode,
            prefix_hashes_published=sequence.prefix_hashes_published,
            logprobs=sequence.logprobs,
            num_placeholder_tokens=sequence.num_placeholder_tokens,
            spec_token_ids=EngineCoreIpcCodec._to_ndarray(sequence.spec_token_ids),
            arrive_time=sequence.arrive_time,
            first_token_time=sequence.first_token_time,
            leave_time=sequence.leave_time,
            leave_reason=sequence.leave_reason,
            sampling=sampling,
            kv_transfer_params=EngineCoreIpcCodec._to_struct(sequence.kv_transfer_params),
            kv_transfer_params_output=EngineCoreIpcCodec._to_struct(
                sequence.kv_transfer_params_output
            ),
            multimodal_data=EngineCoreIpcCodec._to_multimodal_data(
                sequence.multimodal_data
            ),
            mrope_position_delta=sequence.mrope_position_delta,
            has_per_req_cache=sequence.has_per_req_cache,
            num_draft_tokens=sequence.num_draft_tokens,
            needs_independent_noise=sequence.needs_independent_noise,
            sibling_index=sequence.sibling_index,
            num_rejected=sequence.num_rejected,
            num_bonus_tokens=sequence.num_bonus_tokens,
        )
        if sequence.external_request_id is not None:
            message.external_request_id = sequence.external_request_id
        if sequence.dspark_next_ell is not None:
            message.dspark_next_ell = sequence.dspark_next_ell
        if sequence.parent_request_id is not None:
            message.parent_request_id = sequence.parent_request_id
        if sequence.data_parallel_rank is not None:
            message.data_parallel_rank = sequence.data_parallel_rank
        if sequence.mrope_positions is not None:
            message.mrope_positions.CopyFrom(
                EngineCoreIpcCodec._to_ndarray(sequence.mrope_positions)
            )
        return message

    @staticmethod
    def decode_sequence(message: engine_core_proto.Sequence):
        from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
        from atom.sampling_params import SamplingParams

        sampling = SamplingParams(
            temperature=message.sampling.temperature,
            top_k=message.sampling.top_k,
            top_p=message.sampling.top_p,
            max_tokens=message.sampling.max_tokens,
            ignore_eos=message.sampling.ignore_eos,
            stop_strings=list(message.sampling.stop_strings) or None,
            n=message.sampling.n or 1,
            logprobs=message.sampling.logprobs
            if message.sampling.HasField("logprobs")
            else (True if message.sampling.return_logprobs else None),
        )
        sequence = Sequence(
            token_ids=list(message.token_ids),
            block_size=message.block_size,
            sampling_params=sampling,
            stop_token_sequences=[list(item.values) for item in message.stop_token_sequences],
            id=message.id,
            kv_transfer_params=EngineCoreIpcCodec._from_struct(message.kv_transfer_params),
            num_draft_tokens=message.num_draft_tokens,
            has_per_req_cache=message.has_per_req_cache,
            needs_independent_noise=message.needs_independent_noise,
            parent_request_id=message.parent_request_id
            if message.HasField("parent_request_id")
            else None,
            sibling_index=message.sibling_index,
            request_id=message.external_request_id
            if message.HasField("external_request_id")
            else None,
            multimodal_data=EngineCoreIpcCodec._from_multimodal_data(
                message.multimodal_data
            ),
            mrope_positions=EngineCoreIpcCodec._from_ndarray(message.mrope_positions)
            if message.mrope_positions.dtype and message.mrope_positions.data
            else None,
            mrope_position_delta=message.mrope_position_delta,
            data_parallel_rank=message.data_parallel_rank
            if message.HasField("data_parallel_rank")
            else None,
        )
        # Sequence.__init__ treats zero as an absent id; protobuf sequence ids
        # are valid int64 values and must roundtrip exactly.
        sequence.id = message.id
        sequence.status = SequenceStatus[
            engine_core_proto.SequenceStatus.Name(message.status).removeprefix(
                "SEQUENCE_STATUS_"
            )
        ]
        sequence.type = SequenceType[
            engine_core_proto.SequenceType.Name(message.type).removeprefix(
                "SEQUENCE_TYPE_"
            )
        ]
        for name in (
            "last_token", "num_tokens", "num_prompt_tokens", "num_cached_tokens",
            "num_hashed_tokens", "num_compressed_hit_blocks", "num_wanted_hit_blocks",
            "checkpoint_demand_pos", "checkpoint_demand_counted",
            "checkpoint_demand_declined", "last_checkpoint_pos",
            "prefix_cache_hit_tokens", "is_partial_prefill", "per_req_cache_group",
            "state_fork_src", "is_first_decode", "prefix_hashes_published",
            "num_placeholder_tokens", "arrive_time", "first_token_time", "leave_time",
            "leave_reason", "has_per_req_cache", "num_draft_tokens",
            "needs_independent_noise", "sibling_index", "num_rejected", "num_bonus_tokens",
        ):
            setattr(sequence, name, getattr(message, name))
        sequence.output_tokens.extend(message.output_tokens)
        sequence.block_table.extend(message.block_table)
        sequence.logprobs.extend(message.logprobs)
        sequence.spec_token_ids = EngineCoreIpcCodec._from_ndarray(message.spec_token_ids)
        sequence.kv_transfer_params_output = EngineCoreIpcCodec._from_struct(
            message.kv_transfer_params_output
        )
        sequence.dspark_next_ell = (
            message.dspark_next_ell if message.HasField("dspark_next_ell") else None
        )
        return sequence

    @staticmethod
    def encode_add_request(sequences: list) -> bytes:
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            add_request=engine_core_proto.AddRequest(
                sequences=[EngineCoreIpcCodec.encode_sequence(seq) for seq in sequences]
            ),
        ).SerializeToString()

    @staticmethod
    def decode_add_request(message: engine_core_proto.AddRequest) -> list:
        return [
            EngineCoreIpcCodec.decode_sequence(sequence)
            for sequence in message.sequences
        ]

    @staticmethod
    def encode_add_response(sequences: list) -> bytes:
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            add_response=engine_core_proto.AddResponse(
                sequences=[EngineCoreIpcCodec.encode_sequence(seq) for seq in sequences]
            ),
        ).SerializeToString()

    @staticmethod
    def decode_add_response(message: engine_core_proto.AddResponse) -> list:
        return [
            EngineCoreIpcCodec.decode_sequence(sequence)
            for sequence in message.sequences
        ]

    @staticmethod
    def decode_engine_core_envelope(frame: bytes):
        message = engine_core_proto.EngineCoreEnvelope()
        message.ParseFromString(frame)
        if message.wire_version != WIRE_VERSION:
            raise ValueError(f"unsupported engine-core protobuf version {message.wire_version}")
        return message

    @staticmethod
    def encode_utility_command(payload: dict) -> bytes:
        command = payload.get("cmd")
        if not isinstance(command, str):
            raise ValueError("utility payload must contain a string 'cmd'")
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            utility_command=engine_core_proto.UtilityCommand(
                command=command,
                arguments=EngineCoreIpcCodec._to_struct(
                    EngineCoreIpcCodec._to_utility_value(payload)
                ),
            ),
        ).SerializeToString()

    @staticmethod
    def decode_utility_command(message: engine_core_proto.UtilityCommand) -> dict:
        payload = EngineCoreIpcCodec._from_struct(message.arguments) or {}
        payload = EngineCoreIpcCodec._from_utility_value(payload)
        payload["cmd"] = message.command
        return payload

    @staticmethod
    def encode_shutdown() -> bytes:
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            shutdown=empty_pb2.Empty(),
        ).SerializeToString()

    @staticmethod
    def encode_ready(payload: dict | None) -> bytes:
        ready = engine_core_proto.ReadySignal()
        if payload and payload.get("max_pool_tokens") is not None:
            ready.max_pool_tokens = payload["max_pool_tokens"]
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION, ready=ready
        ).SerializeToString()

    @staticmethod
    def encode_metrics(payload: dict) -> bytes:
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            metrics=engine_core_proto.MetricsSnapshot(
                values=EngineCoreIpcCodec._to_struct(payload)
            ),
        ).SerializeToString()

    @staticmethod
    def decode_metrics(message: engine_core_proto.MetricsSnapshot) -> dict:
        return EngineCoreIpcCodec._from_struct(message.values) or {}

    @staticmethod
    def encode_utility_response(command: str, result) -> bytes:
        result_message = struct_pb2.Value()
        json_format.ParseDict(
            EngineCoreIpcCodec._to_utility_value(result), result_message
        )
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            utility_response=engine_core_proto.UtilityResponse(
                command=command, result=result_message
            ),
        ).SerializeToString()

    @staticmethod
    def decode_utility_response(message: engine_core_proto.UtilityResponse):
        return EngineCoreIpcCodec._from_utility_value(
            json_format.MessageToDict(message.result)
        )

    @staticmethod
    def encode_stream(outputs: list[tuple[int, object]]) -> bytes:
        stream_outputs = []
        for sequence_id, output in outputs:
            encoded = engine_core_proto.RequestOutput(
                request_id=output.request_id,
                output_tokens=output.output_tokens,
                finished=output.finished,
                num_cached_tokens=output.num_cached_tokens,
                kv_transfer_params_output=EngineCoreIpcCodec._to_struct(
                    output.kv_transfer_params_output
                ),
            )
            if output.finish_reason is not None:
                encoded.finish_reason = output.finish_reason
            stream_outputs.append(
                engine_core_proto.StreamOutput(sequence_id=sequence_id, output=encoded)
            )
        return engine_core_proto.EngineCoreEnvelope(
            wire_version=WIRE_VERSION,
            stream=engine_core_proto.StreamChunk(outputs=stream_outputs),
        ).SerializeToString()

    @staticmethod
    def decode_stream(message: engine_core_proto.StreamChunk) -> list[tuple[int, object]]:
        from atom.model_engine.request import RequestOutput

        return [
            (
                output.sequence_id,
                RequestOutput(
                    request_id=output.output.request_id,
                    output_tokens=list(output.output.output_tokens),
                    finished=output.output.finished,
                    finish_reason=output.output.finish_reason
                    if output.output.HasField("finish_reason")
                    else None,
                    kv_transfer_params_output=EngineCoreIpcCodec._from_struct(
                        output.output.kv_transfer_params_output
                    ),
                    num_cached_tokens=output.output.num_cached_tokens,
                ),
            )
            for output in message.outputs
        ]

class DisaggIpcCodec:
    """Protobuf codecs for prefill/decode disaggregation frames."""

    @staticmethod
    def encode_weight_handles(paths: list[str]) -> bytes:
        return disagg_proto.WeightIpcBootstrap(rank_file_paths=paths).SerializeToString()

    @staticmethod
    def decode_weight_handles(frame: bytes) -> list[str]:
        message = disagg_proto.WeightIpcBootstrap()
        message.ParseFromString(frame)
        return list(message.rank_file_paths)

    @staticmethod
    def encode_kv_cache_handles(paths: list[str], num_blocks: int) -> bytes:
        return disagg_proto.KvCacheIpcBootstrap(
            rank_file_paths=paths, num_kvcache_blocks=num_blocks
        ).SerializeToString()

    @staticmethod
    def decode_kv_cache_handles(frame: bytes) -> tuple[list[str], int]:
        message = disagg_proto.KvCacheIpcBootstrap()
        message.ParseFromString(frame)
        return list(message.rank_file_paths), message.num_kvcache_blocks

    @staticmethod
    def encode_acknowledgement() -> bytes:
        return disagg_proto.WeightIpcAck(acknowledged=True).SerializeToString()

    @staticmethod
    def decode_acknowledgement(frame: bytes) -> None:
        message = disagg_proto.WeightIpcAck()
        message.ParseFromString(frame)
        if not message.acknowledged:
            raise ValueError("PD weight IPC bootstrap was not acknowledged")

    @staticmethod
    def encode_block_assignment(assignment) -> bytes:
        return disagg_proto.DisaggEnvelope(
            wire_version=WIRE_VERSION,
            block_assignment=assignment,
        ).SerializeToString()

    @staticmethod
    def encode_prefill_done(done) -> bytes:
        return disagg_proto.DisaggEnvelope(
            wire_version=WIRE_VERSION,
            prefill_done=done,
        ).SerializeToString()

    @staticmethod
    def encode_abort(seq_id: int) -> bytes:
        return disagg_proto.DisaggEnvelope(
            wire_version=WIRE_VERSION, abort_seq_id=seq_id
        ).SerializeToString()

    @staticmethod
    def decode_assignment_or_abort(frame: bytes):
        message = disagg_proto.DisaggEnvelope()
        message.ParseFromString(frame)
        if message.wire_version != WIRE_VERSION:
            raise ValueError("unsupported PD protobuf version")
        match message.WhichOneof("payload"):
            case "block_assignment":
                assignment = disagg_proto.BlockAssignment()
                assignment.CopyFrom(message.block_assignment)
                return "block_assignment", assignment
            case "abort_seq_id":
                return "abort", message.abort_seq_id
            case _:
                raise ValueError("expected PD block assignment or abort")

    @staticmethod
    def decode_prefill_done(frame: bytes):
        message = disagg_proto.DisaggEnvelope()
        message.ParseFromString(frame)
        if message.wire_version != WIRE_VERSION:
            raise ValueError("unsupported PD protobuf version")
        if message.WhichOneof("payload") != "prefill_done":
            raise ValueError("expected PD prefill done")
        done = disagg_proto.PrefillDone()
        done.CopyFrom(message.prefill_done)
        return done

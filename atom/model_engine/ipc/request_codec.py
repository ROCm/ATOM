"""Admission-only protobuf codec for the Manager request socket."""

import array
import math

from atom.model_engine.prefix_hash import ShippedChain
from atom.sampling_params import SamplingParams

from .frame_utils import ChannelCodec, DecodedMessage, Fields, MetaCodec, ProtocolError
from .proto import common_pb2 as common
from .proto import request_pb2 as request
from .tensor_adapter import TensorAdapter


class RequestCodec(ChannelCodec):
    SCHEMA_ID = 0x41545251
    MESSAGE = request.RequestMessage
    OPTIONAL = (
        "external_request_id",
        "parent_request_id",
        "data_parallel_rank",
        "dp_session_id",
        "dp_parent_session_id",
    )
    SCALARS = (
        "id",
        "sibling_index",
        "needs_independent_noise",
        "arrive_time",
        "mrope_position_delta",
    )
    TENSORS = frozenset(
        (
            "pixel_values",
            "image_grid_thw",
            "token_types",
            "image_sizes",
            "pixel_attention_mask",
            "video_grid_thw",
            "pixel_values_videos",
            "audio_features",
            "input_features",
            "attention_mask",
        )
    )

    @staticmethod
    def _validate_sampling(value):
        if not math.isfinite(value.temperature) or value.temperature < 0:
            raise ProtocolError("sampling temperature must be finite and nonnegative")
        if not math.isfinite(value.top_p) or not 0 < value.top_p <= 1:
            raise ProtocolError("sampling top_p must be in (0, 1]")
        if value.top_k != -1 and value.top_k < 1:
            raise ProtocolError("sampling top_k must be -1 or positive")
        if value.n < 1:
            raise ProtocolError("sampling n must be positive")

    def encode(self, kind, payload, *, buffer_refs=None, **context):
        if Fields.kind(kind) not in ("ADD", "SUBMIT"):
            raise ProtocolError("request socket only accepts admission batches")
        self._batch(payload)
        message, writer = self._new(**context)
        for value in payload:
            item = message.submit.requests.add()
            Fields.assign(item, value, self.SCALARS + self.OPTIONAL)
            item.admission_id = str(Fields.get(value, "admission_id", item.id))
            item.submission_sequence = Fields.get(
                value, "submission_sequence", context.get("message_id", 0)
            )
            prompt = Fields.get(value, "prompt")
            descriptor = None if buffer_refs is None else buffer_refs.get(item.id)
            if descriptor is not None:
                writer.shm(item.prompt, descriptor)
            else:
                view = prompt.view
                writer.buffer(
                    item.prompt,
                    view,
                    common.INT32,
                    identity=("prompt", prompt.buffer_identity),
                    owner=prompt,
                )
            sampling = Fields.get(
                value, "sampling_params", Fields.get(value, "sampling")
            )
            if sampling is None:
                raise ProtocolError("admission sampling parameters are missing")
            Fields.assign(
                item.sampling,
                sampling,
                ("temperature", "top_k", "top_p", "max_tokens", "ignore_eos", "n"),
            )
            self._validate_sampling(item.sampling)
            logprobs = Fields.get(sampling, "logprobs")
            if type(logprobs) is bool:
                item.sampling.logprobs_bool = logprobs
            elif type(logprobs) is int and logprobs >= 0:
                item.sampling.logprobs_count = logprobs
            elif logprobs is not None:
                raise ProtocolError(
                    "sampling logprobs must be null, bool, or a nonnegative integer"
                )
            stops = Fields.get(sampling, "stop_strings")
            item.sampling.has_stop_strings = stops is not None
            if stops is not None:
                item.sampling.stop_strings.extend(
                    [stops] if isinstance(stops, str) else stops
                )
            for tokens in Fields.get(value, "stop_token_sequences", ()) or ():
                item.stop_token_sequences.add().tokens.extend(tokens)
            kv = Fields.get(value, "kv_transfer_params")
            if kv is not None:
                MetaCodec.encode_object(item.kv_transfer_params, kv)
            media = Fields.get(value, "multimodal_data")
            if media is not None:
                self._encode_media(item.multimodal_data, media, writer)
            mrope = Fields.get(value, "mrope_positions")
            if mrope is not None:
                TensorAdapter.encode(
                    item.mrope_positions, mrope, writer, name="mrope_positions"
                )
            chain = Fields.get(value, "prompt_hash_chain")
            if chain is not None:
                target = item.prompt_hash_chain
                target.hash_block_size = chain.hash_block_size
                target.cache_seed = chain.cache_seed
                target.num_prompt_tokens = chain.num_prompt_tokens
                target.fingerprint = chain.fingerprint
                target.hashes.extend(chain.hashes)
            carrier = Fields.get(value, "_atom_trace_carrier")
            if carrier:
                trace = item.trace
                trace.carrier.update(carrier)
                trace.model = Fields.get(value, "_atom_trace_model") or ""
                trace_request_id = Fields.get(value, "_atom_trace_request_id")
                if trace_request_id is not None:
                    trace.request_id = trace_request_id
        return self._finish(message, writer)

    def _encode_media(self, target, media, writer):
        target.SetInParent()
        extra = {}
        for name, value in media.items():
            if name in self.TENSORS and value is not None:
                TensorAdapter.encode(target.tensors.add(), value, writer, name=name)
            elif name == "embedding_spans":
                for start, length in value:
                    target.embedding_spans.add(start=start, length=length)
            elif name == "cache_seed":
                target.cache_seed = value
            else:
                extra[name] = value
        if extra:
            MetaCodec.encode_object(target.metadata, extra)

    def decode(self, frames, **context):
        message, reader, metadata = self._parse(frames, **context)
        if not message.HasField("submit"):
            raise ProtocolError("request SubmitBatch is missing")
        self._batch(message.submit.requests)
        payload = []
        seen = set()
        for item in message.submit.requests:
            if item.id in seen or not item.HasField("sampling"):
                raise ProtocolError(
                    "duplicate admission ID or missing sampling parameters"
                )
            seen.add(item.id)
            self._validate_sampling(item.sampling)
            logprobs_kind = item.sampling.WhichOneof("logprobs")
            sampling = SamplingParams(
                temperature=item.sampling.temperature,
                top_k=item.sampling.top_k,
                top_p=item.sampling.top_p,
                max_tokens=item.sampling.max_tokens,
                ignore_eos=item.sampling.ignore_eos,
                n=item.sampling.n,
                stop_strings=(
                    list(item.sampling.stop_strings)
                    if item.sampling.has_stop_strings
                    else None
                ),
                logprobs=(
                    None
                    if logprobs_kind is None
                    else getattr(item.sampling, logprobs_kind)
                ),
            )
            value = {name: getattr(item, name) for name in self.SCALARS}
            value.update(
                {
                    name: getattr(item, name) if item.HasField(name) else None
                    for name in self.OPTIONAL
                }
            )
            value.update(
                prompt=reader.prompt(item.prompt),
                sampling_params=sampling,
                stop_token_sequences=[
                    list(stop.tokens) for stop in item.stop_token_sequences
                ],
                admission_id=item.admission_id,
                submission_sequence=item.submission_sequence,
                kv_transfer_params=(
                    MetaCodec.decode_object(item.kv_transfer_params)
                    if item.HasField("kv_transfer_params")
                    else None
                ),
                multimodal_data=(
                    self._decode_media(item.multimodal_data, reader)
                    if item.HasField("multimodal_data")
                    else None
                ),
                mrope_positions=(
                    TensorAdapter.decode(item.mrope_positions, reader)
                    if item.HasField("mrope_positions")
                    else None
                ),
            )
            if item.HasField("prompt_hash_chain"):
                chain = item.prompt_hash_chain
                # Only a well-formed chain is decoded; whether it fits this
                # engine's prompt and block size is `_adopt_shipped_chain`'s.
                value["prompt_hash_chain"] = ShippedChain(
                    chain.hash_block_size,
                    chain.cache_seed,
                    chain.num_prompt_tokens,
                    chain.fingerprint,
                    array.array("Q", chain.hashes),
                )
            if item.HasField("trace") and item.trace.carrier:
                value.update(
                    _atom_trace_carrier=dict(item.trace.carrier),
                    _atom_trace_model=item.trace.model,
                    _atom_trace_request_id=(
                        item.trace.request_id
                        if item.trace.HasField("request_id")
                        else None
                    ),
                )
            payload.append(value)
        reader.finish()
        return DecodedMessage(
            Fields.kind_value("ADD"), payload, metadata, list(reader.frames)
        )

    def _decode_media(self, media, reader):
        result = (
            MetaCodec.decode_object(media.metadata)
            if media.HasField("metadata")
            else {}
        )
        for tensor in media.tensors:
            if tensor.name not in self.TENSORS or tensor.name in result:
                raise ProtocolError("unknown or duplicate multimodal tensor")
            result[tensor.name] = TensorAdapter.decode(tensor, reader)
        result["embedding_spans"] = [
            (span.start, span.length) for span in media.embedding_spans
        ]
        if media.HasField("cache_seed"):
            result["cache_seed"] = media.cache_seed
        return result

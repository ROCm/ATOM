# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The CoreManager <-> EngineCore messages, as msgspec structs.

A message is one ZMQ frame (after the routing identity a ROUTER adds): the
msgpack array ``[kind, body]``. ``kind`` is the `EngineCoreRequestType` byte as
an int; ``body`` is one of the structs below, every one ``array_like``, so a
struct is a msgpack array of its fields in declaration order. That is also how
`rmp-serde` encodes a Rust struct, which is what the atomesh client
(`atom/mesh/src/atom_engine/wire.rs`) mirrors. The rules a change keeps:

- Fields are only ever appended, each with a default. A reader fills a missing
  trailing field with its default and ignores one it does not know, so either
  side may be a version ahead.
- Bulk numbers ride as little-endian ``bytes`` (int32 tokens, float64
  logprobs, uint64 hashes): one copy each way, as a pickled ``array`` was.
  A field its reader consumes at once is a ``memoryview``, decoded as a view
  of the received frame rather than a copy of it; it keeps that frame alive,
  so a message held for long (`Completion`) keeps ``bytes``.
- What only Python reads (utility commands and replies, metrics, DP
  bookkeeping) is ``Pickled``: the very object the engine pickled before.
- ``Ready.protocol_version`` is checked by the client at connect.

Whatever only Python produces and reads -- KV-transfer params, multimodal data
and MRoPE positions, with their tensors, arrays and tuples -- is pickled inside
its field, so it arrives exactly as it left. The media's tensors and arrays
ride beside their pickle as raw bytes: copied once each way, without torch's
own serialization.
"""

import msgspec

from atom.model_engine.engine_core_protocol import EngineCoreRequestType

PROTOCOL_VERSION = 2

# The wire's kind for each request type, and back.
KIND = {t: t.value[0] for t in EngineCoreRequestType}
REQUEST_TYPE = {kind: t for t, kind in KIND.items()}

# `Completion.reject_code`: why the Core refused a request it never ran.
FINISHED = 0
REJECT_INVALID = 1  # the request itself is bad
REJECT_OVERLOADED = 2  # retrying later may succeed
REJECT_INTERNAL = 3  # the engine failed it


class _Body(msgspec.Struct, array_like=True, gc=False):
    pass


class RequiredChannel(_Body, frozen=True):
    """`sampling_params.RequiredChannel`'s fields."""

    after: tuple[int, ...]
    opening: tuple[int, ...]
    opens_at: int
    end_ids: tuple[int, ...]


class Sampling(_Body):
    """`SamplingParams`' fields, in its declaration order."""

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop_strings: list[str] | None = None
    n: int = 1
    logprobs: bool | int | None = None
    required_channel: RequiredChannel | None = None
    top_logprobs: int = 0
    json_schema: str | None = None
    structured_tool_format: str | None = None
    structured_reasoning_end: str = ""
    max_tokens_is_default: bool = False


class PromptHashChain(_Body):
    """`prefix_hash.ShippedChain`, its hashes as uint64 bytes."""

    hash_block_size: int
    cache_seed: int
    num_prompt_tokens: int
    fingerprint: int
    hashes: bytes


class TraceContext(_Body):
    """What `tracing.wire_context` gives for a traced request."""

    carrier: dict[str, str]
    model: str
    request_id: str | None = None


class SubmitItem(_Body):
    """One sequence to admit: the `Sequence` constructor's arguments and what
    the frontend sets on a sequence before it sends it."""

    seq_id: int
    prompt: memoryview  # int32 token ids
    # `Sampling`, decoded per item: a bad one refuses its own request only.
    sampling: msgspec.Raw
    stop_token_sequences: list[list[int]] = []
    kv_transfer_params: memoryview | None = None  # the pickled dict
    # `(multimodal_data, mrope_positions)` pickled without its tensors' and
    # arrays' bytes, which are `multimodal_buffers` (`codec.pickle_media`).
    multimodal: bytes | None = None
    multimodal_buffers: list[memoryview] | None = None
    mrope_position_delta: int = 0
    # The media's, when multimodal. A decode leg whose prefill kept the media
    # takes `cache_seed` and `mrope_position_delta` from `kv_transfer_params`
    # instead (`sequence.leaves_media_on_prefill`).
    cache_seed: int | None = None
    needs_independent_noise: bool = False
    parent_request_id: str | None = None
    sibling_index: int = 0
    external_request_id: str | None = None
    data_parallel_rank: int | None = None
    dp_session_id: str | None = None
    dp_parent_session_id: str | None = None
    arrive_time: float = 0.0
    prompt_hash_chain: PromptHashChain | None = None
    trace: TraceContext | None = None
    dp_est_cached_tokens: int = 0
    # Why the client could not encode this request; the Core refuses it with
    # this reason, so it ends like any other refused request.
    refused: str | None = None


class Submit(_Body):
    items: list[SubmitItem]


class Abort(_Body):
    """Abort these sequences, as the ``abort_request`` utility command does."""

    seq_ids: list[int]


class Empty(_Body):
    """SHUTDOWN, either way."""


class Pickled(_Body):
    """A payload only Python reads, pickled."""

    data: memoryview


class Ready(_Body):
    protocol_version: int = PROTOCOL_VERSION
    max_pool_tokens: int | None = None
    weight_syncs_unavailable: dict[str, str] = {}


class StreamItem(_Body):
    """One sequence's step: `RequestOutput`'s fields, by position."""

    seq_id: int
    output_tokens: list[int]
    finished: bool = False
    finish_reason: str | None = None
    kv_transfer_params_output: bytes | None = None  # the pickled dict
    num_cached_tokens: int = 0
    logprobs: list[tuple] | None = None


class Stream(_Body):
    items: list[StreamItem]


class Completion(_Body):
    """A finished sequence -- or one the Core refused (`reject_code`).

    The client reads it where it read the finished `Sequence` itself, so the
    properties give the names that code reads.
    """

    seq_id: int
    completion_tokens: bytes = b""  # int32
    logprobs_bytes: bytes | None = None  # float64, one per completion token
    return_logprobs: bool = False
    num_prompt_tokens: int = 0
    num_cached_tokens: int = 0
    arrive_time: float = 0.0
    first_token_time: float = 0.0
    leave_reason: str = ""
    prefill_completed: bool = False
    json_schema: str | None = None
    structured_tool_format: str | None = None
    reject_code: int = FINISHED
    # Set by the client when it hands the result out.
    leave_time: float = 0.0

    @property
    def id(self) -> int:
        return self.seq_id

    @property
    def completion_token_ids(self) -> list[int]:
        return memoryview(self.completion_tokens).cast("i").tolist()

    @property
    def num_completion_tokens(self) -> int:
        return len(self.completion_tokens) // 4

    @property
    def logprobs(self) -> list[float]:
        if self.logprobs_bytes is None:
            return []
        return memoryview(self.logprobs_bytes).cast("d").tolist()

    @property
    def rejected(self) -> bool:
        return self.reject_code != FINISHED


class Completions(_Body):
    items: list[Completion]


# The body each kind carries, by direction.
INPUT_BODY = {
    EngineCoreRequestType.ADD: Submit,
    EngineCoreRequestType.ABORT: Abort,
    EngineCoreRequestType.SHUTDOWN: Empty,
    EngineCoreRequestType.UTILITY: Pickled,
    EngineCoreRequestType.DP_PREFIX_CREDIT: Pickled,
}
OUTPUT_BODY = {
    EngineCoreRequestType.ADD: Completions,
    EngineCoreRequestType.STREAM: Stream,
    EngineCoreRequestType.READY: Ready,
    EngineCoreRequestType.SHUTDOWN: Empty,
    EngineCoreRequestType.UTILITY_RESPONSE: Pickled,
    EngineCoreRequestType.METRICS: Pickled,
    EngineCoreRequestType.PREFILL_BACKLOG: Pickled,
    EngineCoreRequestType.DP_REUSE: Pickled,
}

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""What a request must pass to be admitted, and the `Sequence` a Core builds
for one.

`check_sampling` is what both the frontend's `InputOutputProcessor` and the
EngineCore refuse first. The Core checks every client again, since a client
that is not this repo's frontend (the atomesh Rust runtime) skipped the
frontend's checks.

`build_sequence` is the Core's half: it turns a decoded request
(`zmq_protocol.codec.AdmissionSpec`) into the `Sequence` the scheduler has
always taken.
"""

from dataclasses import replace

from atom.model_engine.sequence import PER_REQ_CACHE_MODEL_TYPES, Sequence
from atom.observability import tracing
from atom.sampling_params import EngineChannel, SamplingParams
from atom.structured_output import is_structured
from atom.structured_output.validation import validate_execution_options
from atom.utils import envs


def check_sampling(config, params: SamplingParams, *, kv_transfer, multimodal) -> None:
    """Refuse sampling options this engine cannot serve for a request: its
    logprob options, and a structured-output grammar beside a required
    channel. What both the frontend and the Core check first."""
    params.validate_logprob_options(
        config, kv_transfer=kv_transfer, multimodal=multimodal
    )
    if is_structured(params) and params.required_channel is not None:
        raise ValueError(
            "A request cannot carry both a structured-output grammar "
            "(json_schema or structured_tool_format) and a required_channel"
        )


class AdmissionContext:
    """How a Core turns a decoded request into its `Sequence`.

    Holds what the Core derives from its own config -- the client used to
    derive the same from its copy of it -- and fills in what a client may
    leave out (the REQUESTS section of engine_v1.proto). Built once per Core,
    at startup.
    """

    def __init__(self, config):
        speculative = config.speculative_config
        self.config = config
        self.block_size = config.kv_cache_block_size
        self.num_draft_tokens = (
            speculative.num_speculative_tokens if speculative is not None else 0
        )
        self.has_per_req_cache = (
            config.hf_config.model_type in PER_REQ_CACHE_MODEL_TYPES
        )
        # This Core's own: its engine loop holds one channel.
        self.engine_channel = EngineChannel(config)


def build_sequence(spec, context: AdmissionContext) -> Sequence:
    """The `Sequence` the client would have built for `spec`, a decoded
    `AdmissionSpec`.

    Refuses what the frontend refuses before it sends a request, in its
    order: `check_sampling`, a structured request's execution options, and
    the model's own validation; then holds a required channel to this Core's.
    A structured request's grammar is the client's to have compiled
    (`LLMEngine.prepare_structured_request`).
    """
    kwargs = dict(spec.kwargs)
    multimodal_data = kwargs["multimodal_data"]
    kv_transfer = kwargs["kv_transfer_params"]
    params = kwargs["sampling_params"]
    check_sampling(
        context.config,
        params,
        kv_transfer=kv_transfer,
        multimodal=multimodal_data is not None,
    )
    if is_structured(params):
        validate_execution_options(
            context.config,
            params,
            kv_transfer=kv_transfer,
            multimodal=multimodal_data is not None,
            relaxed_mtp=envs.ATOM_ENABLE_RELAXED_MTP,
        )
    validate_request = getattr(context.config.hf_config, "validate_request", None)
    if validate_request is not None:
        validate_request(
            num_draft_tokens=context.num_draft_tokens,
            multimodal_data=multimodal_data,
            enable_dp_attention=context.config.enable_dp_attention,
        )
    if params.required_channel is not None:
        kwargs["sampling_params"] = replace(
            params,
            required_channel=context.engine_channel.hold(params.required_channel),
        )
    if multimodal_data is not None and kwargs["mrope_positions"] is None:
        # A client that cannot run the model's MRoPE code leaves them out.
        from atom.multimodal.registry import get_mrope_input_positions

        (
            kwargs["mrope_positions"],
            kwargs["mrope_position_delta"],
        ) = get_mrope_input_positions(
            context.config, list(kwargs["token_ids"]), multimodal_data
        )
    # The client hashed the media already; hand the seed to the constructor
    # the way it reads one, then take it back out, so the sequence's
    # multimodal_data is the dict the client sent.
    lend_seed = (
        spec.cache_seed is not None
        and multimodal_data is not None
        and "cache_seed" not in multimodal_data
    )
    if lend_seed:
        multimodal_data["cache_seed"] = spec.cache_seed
    try:
        seq = Sequence(
            block_size=context.block_size,
            num_draft_tokens=context.num_draft_tokens,
            has_per_req_cache=context.has_per_req_cache,
            id=spec.seq_id,
            **kwargs,
        )
    finally:
        if lend_seed:
            del multimodal_data["cache_seed"]
    seq.max_tokens = spec.max_tokens
    seq.arrive_time = spec.arrive_time
    seq.dp_est_cached_tokens = spec.dp_est_cached_tokens
    # The BlockManager checks a shipped chain before it trusts it.
    seq.prompt_hash_chain = spec.prompt_hash_chain
    if spec.trace is not None:
        tracing.adopt_context(seq, *spec.trace)
    return seq

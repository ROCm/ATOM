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
always taken. `validate_for_model` and `mrope_for` are what the frontend's
`InputOutputProcessor` runs for the same request.
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


def num_draft_tokens(config) -> int:
    """The draft width a sequence carries under `config`'s speculation."""
    speculative = config.speculative_config
    return speculative.num_speculative_tokens if speculative is not None else 0


def validate_for_model(config, *, num_draft_tokens, multimodal_data) -> None:
    """The model's own refusals (`hf_config.validate_request`), if it has any."""
    validate_request = getattr(config.hf_config, "validate_request", None)
    if validate_request is not None:
        validate_request(
            num_draft_tokens=num_draft_tokens,
            multimodal_data=multimodal_data,
            enable_dp_attention=config.enable_dp_attention,
        )


def mrope_for(config, token_ids, multimodal_data):
    """`(mrope_positions, mrope_position_delta)` of a prompt: `(None, 0)` for
    a text one."""
    if multimodal_data is None:
        return None, 0
    from atom.multimodal.registry import get_mrope_input_positions

    return get_mrope_input_positions(config, token_ids, multimodal_data)


class AdmissionContext:
    """How a Core turns a decoded request into its `Sequence`.

    Holds what the Core derives from its own config -- the client used to
    derive the same from its copy of it -- and fills in what a client may
    leave out (the REQUESTS section of engine_v1.proto). Built once per Core,
    at startup.
    """

    def __init__(self, config):
        self.config = config
        self.block_size = config.kv_cache_block_size
        self.num_draft_tokens = num_draft_tokens(config)
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
    validate_for_model(
        context.config,
        num_draft_tokens=context.num_draft_tokens,
        multimodal_data=multimodal_data,
    )
    if params.required_channel is not None:
        kwargs["sampling_params"] = replace(
            params,
            required_channel=context.engine_channel.hold(params.required_channel),
        )
    if multimodal_data is not None and kwargs["mrope_positions"] is None:
        # A client that cannot run the model's MRoPE code leaves them out.
        kwargs["mrope_positions"], kwargs["mrope_position_delta"] = mrope_for(
            context.config, list(kwargs["token_ids"]), multimodal_data
        )
    seq = Sequence(
        block_size=context.block_size,
        num_draft_tokens=context.num_draft_tokens,
        has_per_req_cache=context.has_per_req_cache,
        id=spec.seq_id,
        # The client hashed the media already.
        cache_seed=spec.cache_seed,
        **kwargs,
    )
    seq.max_tokens = spec.max_tokens
    seq.arrive_time = spec.arrive_time
    seq.dp_est_cached_tokens = spec.dp_est_cached_tokens
    # The BlockManager checks a shipped chain before it trusts it.
    seq.prompt_hash_chain = spec.prompt_hash_chain
    if spec.trace is not None:
        tracing.adopt_context(seq, *spec.trace)
    return seq

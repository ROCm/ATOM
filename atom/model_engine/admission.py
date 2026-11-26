# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""What a request must pass to be admitted, and the `Sequence` a Core builds
for one.

`check_sampling` is what both the frontend's `InputOutputProcessor` and the
EngineCore refuse first. The Core checks every client again, since a client
that is not this repo's frontend (the atomesh Rust runtime) skipped the
frontend's checks.

`build_sequence` is the Core's half: it turns a decoded request
(`zmq_protocol.messages.SubmitItem`) into the `Sequence` the scheduler has
always taken. `validate_for_model` and `mrope_for` are what the frontend's
`InputOutputProcessor` runs for the same request.
"""

from dataclasses import replace

from atom.model_engine.sequence import (
    PER_REQ_CACHE_MODEL_TYPES,
    Sequence,
    leaves_media_on_prefill,
)
from atom.model_engine.zmq_protocol import codec
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


def media_kept_on_prefill(kv_transfer, multimodal_data) -> bool:
    """Whether a request is a decode leg whose prefill kept the media
    (`Sequence`'s own test)."""
    return (
        kv_transfer is not None
        and multimodal_data is None
        and leaves_media_on_prefill(kv_transfer)
    )


def validate_for_model(
    config, *, num_draft_tokens, multimodal_data, kv_transfer=None
) -> None:
    """The model's own refusals (`hf_config.validate_request`), if it has any,
    and RapidServe's of a decode leg whose prefill kept the media."""
    media_on_prefill = media_kept_on_prefill(kv_transfer, multimodal_data)
    if media_on_prefill and config.enable_rapidserve:
        # RapidServe prefills every request on this GPU, with no media.
        raise ValueError(
            "RapidServe cannot serve an image decode leg whose prefill "
            "kept the image"
        )
    validate_request = getattr(config.hf_config, "validate_request", None)
    if validate_request is not None:
        validate_request(
            num_draft_tokens=num_draft_tokens,
            multimodal_data=multimodal_data,
            media_on_prefill_instance=media_on_prefill,
            enable_dp_attention=config.enable_dp_attention,
        )


def check_prompt(token_ids, *, kv_transfer, multimodal_data) -> None:
    """Refuse an empty prompt, and a decode leg's handed-off MRoPE delta that
    runs past its prompt."""
    # Checked on the tokens, not the text: "" encodes to [BOS] on Llama-3,
    # which runs, and to nothing on Qwen3 or gpt-oss.
    if len(token_ids) == 0:
        raise ValueError("The decoder prompt cannot be empty")
    if media_kept_on_prefill(kv_transfer, multimodal_data):
        # An image prompt's positions never run past its text's, so the
        # delta its prefill hands over is at most 0; decode positions then
        # stay in [1, max_model_len).
        delta = kv_transfer.get("mrope_position_delta", 0)
        if not 1 - len(token_ids) <= delta <= 0:
            raise ValueError(
                "kv_transfer_params['mrope_position_delta'] must be in "
                f"[{1 - len(token_ids)}, 0] for a {len(token_ids)}-token prompt"
            )


def mrope_for(config, token_ids, multimodal_data, *, uses_mrope: bool):
    """`(mrope_positions, mrope_position_delta)` of a prompt: `(None, 0)` for a
    text one, and for a model without MRoPE (`registry.uses_mrope`), which
    has none to give -- without copying the prompt to ask."""
    if multimodal_data is None or not uses_mrope:
        return None, 0
    from atom.multimodal.registry import get_mrope_input_positions

    return get_mrope_input_positions(config, token_ids, multimodal_data)


class AdmissionContext:
    """How a Core turns a decoded request into its `Sequence`.

    Holds what the Core derives from its own config -- the client used to
    derive the same from its copy of it -- and fills in what a client may
    leave out (`messages.SubmitItem`'s defaults). Built once per Core, at
    startup.
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
        # Whether a multimodal request's MRoPE positions are this model's to
        # compute; a model without MRoPE skips copying the prompt for them.
        from atom.multimodal.registry import uses_mrope

        self.uses_mrope = uses_mrope(config)


def build_sequence(item, context: AdmissionContext) -> Sequence:
    """The `Sequence` the client built for `item`, a decoded
    `messages.SubmitItem`.

    Refuses what the frontend refuses before it sends a request, in its
    order: `check_sampling`, a structured request's execution options, the
    model's own validation and `check_prompt`; then holds a required channel to this Core's.
    A structured request's grammar is the client's to have compiled
    (`LLMEngine.prepare_structured_request`).
    """
    if item.refused is not None:
        raise ValueError(item.refused)
    params = codec.sampling_params(item)
    multimodal_data, mrope_positions = codec.media(item)
    kv_transfer = codec.unpickled(item.kv_transfer_params)
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
        kv_transfer=kv_transfer,
    )
    if params.required_channel is not None:
        params = replace(
            params,
            required_channel=context.engine_channel.hold(params.required_channel),
        )
    token_ids = codec.prompt_token_ids(item)
    check_prompt(token_ids, kv_transfer=kv_transfer, multimodal_data=multimodal_data)
    mrope_position_delta = item.mrope_position_delta
    if multimodal_data is not None and mrope_positions is None:
        # A client that cannot run the model's MRoPE code leaves them out.
        mrope_positions, mrope_position_delta = mrope_for(
            context.config,
            # A list only for a model that reads one.
            list(token_ids) if context.uses_mrope else token_ids,
            multimodal_data,
            uses_mrope=context.uses_mrope,
        )
    seq = _sequence(
        item,
        context,
        token_ids,
        params,
        kv_transfer,
        multimodal_data,
        mrope_positions,
        mrope_position_delta,
    )
    seq.arrive_time = item.arrive_time
    seq.dp_est_cached_tokens = item.dp_est_cached_tokens
    # The BlockManager checks a shipped chain before it trusts it.
    seq.prompt_hash_chain = codec.shipped_chain(item)
    if item.trace is not None:
        tracing.adopt_context(
            seq, item.trace.carrier, item.trace.model, item.trace.request_id
        )
    return seq


def _sequence(
    item,
    context,
    token_ids,
    params,
    kv_transfer,
    multimodal_data,
    mrope_positions,
    delta,
) -> Sequence:
    return Sequence(
        token_ids,
        context.block_size,
        params,
        item.stop_token_sequences,
        id=item.seq_id,
        kv_transfer_params=kv_transfer,
        num_draft_tokens=context.num_draft_tokens,
        has_per_req_cache=context.has_per_req_cache,
        needs_independent_noise=item.needs_independent_noise,
        parent_request_id=item.parent_request_id,
        sibling_index=item.sibling_index,
        request_id=item.external_request_id,
        multimodal_data=multimodal_data,
        mrope_positions=mrope_positions,
        mrope_position_delta=delta,
        data_parallel_rank=item.data_parallel_rank,
        dp_session_id=item.dp_session_id,
        dp_parent_session_id=item.dp_parent_session_id,
        # The client hashed the media already.
        cache_seed=item.cache_seed,
    )

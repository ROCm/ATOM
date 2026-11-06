# SPDX-License-Identifier: MIT
"""Row-owner prefill on the runtime model (``ATOM_DSV41_ROW_SHARDED_MHC=1``).

The shared runtime (`..runtime`) calls into this module from four hooks only:
`configure` after the layers are built, `build_stages` beside the native early
stage, `select_early` per forward, and `prepare_replay` / `tail_gatherer` for
the decoder's first layer. Everything else -- the owner blocks, their guarded
ops and the owner early stage -- lives here, so a deployment without the flag
imports none of it.

An owned layer (``range(2, decoder_start)``) runs its mHC, attention and FFN
on this rank's quarter of the rows (`row_sharded_mhc`); only the KV latent,
the compressor rows and the FFN input/output cross ranks.
"""

import logging
import os

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.utils import envs
from atom.utils.backends import set_model_tag
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context

from ..bounded_replay import late_layer_start, replayable
from ..runtime import RuntimeBlock, _rows_first, _Stage
from .admission import owner_prefill_enabled
from .row_sharded_mhc import eligible, gather_norm, localize, replicate

logger = logging.getLogger("atom")

TP_SIZE = 4
TAIL_MODES = ("off", "settled", "delayed")
TAIL_TRANSPORTS = ("packet", "fields")


def _flag(name):
    return os.environ.get(name, "0") == "1"


# -- guarded ops ---------------------------------------------------------------


def _fake_attention(hidden, layer_name, hidden_fp8=None, hidden_scale=None):
    return torch.empty_like(hidden)


@torch_compile_guard(mutates_args=[], gen_fake=_fake_attention)
def v41_owned_attention(
    hidden: torch.Tensor,
    layer_name: str,
    hidden_fp8: torch.Tensor | None = None,
    hidden_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """`runtime.v41_attention` for owner rows: ``hidden`` is this rank's rows."""
    from .attention import forward

    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return hidden.new_zeros(hidden.shape)
    layer, rope = context.no_compile_layers[layer_name]
    return forward(
        layer.attn,
        hidden,
        metadata.cache,
        metadata.step,
        rope,
        hidden_fp8=hidden_fp8,
        hidden_scale=hidden_scale,
    )


def _fake_moe(hidden, layer_name, row_sharded):
    if not row_sharded:
        return torch.empty_like(hidden)
    return hidden.new_empty(
        ((hidden.shape[0] + TP_SIZE - 1) // TP_SIZE, *hidden.shape[1:])
    )


# Op names are part of the preflight contract (`original_rows`, `layer_name`).
@torch_compile_guard(mutates_args=[], gen_fake=_fake_moe)
def v41_owned_moe(
    hidden: torch.Tensor, layer_name: str, row_sharded: bool
) -> torch.Tensor:
    """The FFN on every row; with ``row_sharded`` its sum reduce-scattered back
    to the owners instead of all-reduced."""
    # Looked up per call: the guarded service patches this seam.
    from . import row_sharded_mhc

    layer, _ = get_forward_context().no_compile_layers[layer_name]
    if not row_sharded:
        return layer.ffn(hidden)
    moe = layer.ffn
    shared = moe.shared_experts(hidden) if moe.shared_experts is not None else None
    routed = moe.experts(hidden, moe.router_logits(hidden))
    return row_sharded_mhc.v41_mhc_reduce_scatter(
        routed if shared is None else routed + shared
    )


def _fake_prequantized_moe(hidden, layer_name, original_rows):
    return torch.empty_like(hidden)


@torch_compile_guard(mutates_args=[], gen_fake=_fake_prequantized_moe)
def v41_owned_prequantized_moe(
    hidden: torch.Tensor, layer_name: str, original_rows: int
) -> torch.Tensor:
    """The FFN from owner rows (`prefill_ffn`) where its shape is supported,
    else gather them and run `v41_owned_moe`."""
    # Looked up per call: the guarded service patches this seam.
    from . import prefill_ffn

    context = get_forward_context()
    layer, _ = context.no_compile_layers[layer_name]
    if prefill_ffn.supported(layer.ffn, hidden, original_rows, context.attn_metadata):
        return prefill_ffn.prequantized_owned_ffn(layer.ffn, hidden, original_rows)
    return v41_owned_moe(gather_norm(hidden, original_rows), layer_name, True)


def local_engram_rows(rows, embeddings, image_mask, width):
    """This rank's Engram rows, padded to the owner capacity ``rows``: zero
    embeddings, and an image mask of ones (no Engram on padding)."""
    from aiter.dist.parallel_state import get_tp_group

    start = get_tp_group().rank_in_group * rows
    valid = min(rows, max(0, width - start))

    def take(value, dim, pad):
        shape = list(value.shape)
        shape[dim] = rows - valid
        return torch.cat((value.narrow(dim, start, valid), pad(shape)), dim=dim)

    embeddings = take(embeddings, -2, embeddings.new_zeros)
    if image_mask is not None:
        image_mask = take(image_mask, -1, image_mask.new_ones)
    return embeddings, image_mask


# -- blocks and stages ---------------------------------------------------------


class OwnedRuntimeBlock(RuntimeBlock):
    """An owned layer: `RuntimeBlock` on replicated rows, owner rows when
    called with ``row_sharded``."""

    row_sharded_mhc = True
    prequantized_prefill_ffn = False

    def forward(
        self,
        state,
        cache,
        step,
        rope,
        embeddings=None,
        image_mask=None,
        queries=None,
        *,
        row_sharded=False,
    ):
        if not row_sharded:
            return super().forward(
                state, cache, step, rope, embeddings, image_mask, queries
            )
        if queries is not None:
            raise RuntimeError("query-row cutting requires a replicated mHC state")
        original_rows = (
            state.original_rows if state.row_sharded else state.residual.shape[-3]
        )
        normed, residual, pre, post, comb = self.prepare_attention(
            localize(state), embeddings, image_mask
        )
        if isinstance(normed, tuple):
            output = v41_owned_attention(normed[0], self.layer_name, *normed[1:])
        else:
            output = v41_owned_attention(normed, self.layer_name)
        normed, residual, pre, post, comb = self.prepare_ffn(
            output, residual, pre, post, comb
        )
        flat = normed.reshape(-1, normed.shape[-1])
        if self.prequantized_prefill_ffn:
            output = v41_owned_prequantized_moe(flat, self.layer_name, original_rows)
        else:
            output = v41_owned_moe(
                gather_norm(flat, original_rows), self.layer_name, True
            )
        return SinglePassHCState(
            residual,
            pre,
            output.view_as(normed),
            post,
            comb,
            row_sharded=True,
            original_rows=original_rows,
        )

    def ffn_forward(self, output, residual, pre, post, comb):
        normed, residual, pre, post, comb = self.prepare_ffn(
            output, residual, pre, post, comb
        )
        flat = normed.reshape(-1, normed.shape[-1])
        output = v41_owned_moe(flat, self.layer_name, False).view_as(normed)
        return self.finish_ffn(output, residual, pre, post, comb)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "inputs_embeds": 0})
class _EarlyOwnedLayers(_Stage):
    """`runtime._EarlyLayers` with the owned layers on owner rows.

    ``retain_mode``: ``"off"`` hands the decoder replicated rows; otherwise the
    owner state is kept (``"settled"`` applies its pending post first) and the
    replay tail gathers only the rows it needs (`tail.gather_owner_tail`)."""

    def __init__(self, *, retain_mode="off", **kwargs):
        super().__init__(**kwargs)
        self.retain_mode = retain_mode

    def forward(self, input_ids, inputs_embeds=None):
        m = self.owner
        hidden = (m.embed(input_ids) if inputs_embeds is None else inputs_embeds)[None]
        m.begin_forward(hidden, None)
        state = SinglePassHCState.from_embeddings(hidden, m.config.hc_mult)
        for i in range(self.first, self.last):
            layer = m.layers[i]
            if isinstance(layer, OwnedRuntimeBlock):
                state = layer(
                    state,
                    None,
                    None,
                    m._rope(i),
                    None,
                    image_mask=None,
                    row_sharded=True,
                )
            else:
                state = layer(
                    replicate(state), None, None, m._rope(i), None, image_mask=None
                )
        if self.retain_mode == "off":
            state = replicate(state)
        elif self.retain_mode == "settled":
            state = state.settle()
        return _rows_first(state.fields())


# -- runtime hooks -------------------------------------------------------------


def _validate(model, config):
    if (
        config.tensor_parallel_size != TP_SIZE
        or config.parallel_config.data_parallel_size != 1
        or config.enable_dp_attention
        or envs.ATOM_DSV41_SIDE_STREAMS
        or config.enable_tbo
        or config.prefill_context_parallel_size != 1
        or config.decode_context_parallel_size != 1
        or config.index_cache_dtype != "fp4"
        or not (
            config.kv_cache_dtype == "bf16"
            or (config.kv_cache_dtype == "fp8" and model.owner_service_prefill)
        )
        or (config.enable_expert_parallel and model.owner_ep4_contract is None)
        or (config.speculative_config is not None and not model.owner_service_prefill)
        or not getattr(config, "enable_decoder_swa_bounded_replay", False)
    ):
        raise ValueError(
            "row-sharded prefill requires TP4 BF16/FP8 KV or verified BF16 EP4, "
            "FP4 index and bounded replay; speculation requires the explicit "
            "DSpark5 owner deployment"
        )


def configure(model, config):
    """After the layers are built: validate the deployment and turn the owned
    layers into `OwnedRuntimeBlock`s."""
    owner_service = os.environ.get("DSV41_OWNER_SERVING") == "1"
    retained_prefill = _flag("DSV41_OWNER_ACCEPTED_PREFILL")
    fixed_placement = _flag("ATOM_DSV41_EP_PROFILED_PLACEMENT")
    model.owner_ep4_contract = None
    if fixed_placement and not config.enable_expert_parallel:
        raise ValueError("Fixed placement requires the EP4 owner deployment")
    if config.enable_expert_parallel:
        from .ep import validate_owner_ep4

        if not owner_prefill_enabled(
            config,
            owner_service=owner_service,
            retained_prefill=retained_prefill,
            ep4_verified=True,
        ):
            raise ValueError(
                "EP4 row-sharded prefill requires the DSpark5 owner deployment"
            )
        if fixed_placement:
            from .placement import install_owner_placement

            install_owner_placement(model, config)
        model.owner_ep4_contract = validate_owner_ep4(model, config)
    model.owner_service_prefill = owner_prefill_enabled(
        config,
        owner_service=owner_service,
        retained_prefill=retained_prefill,
        ep4_verified=model.owner_ep4_contract is not None,
    )
    model.prefill_tail_mode = envs.ATOM_DSV41_PREFILL_TAIL_MODE
    model.prefill_tail_transport = envs.ATOM_DSV41_PREFILL_TAIL_TRANSPORT
    if model.prefill_tail_mode not in TAIL_MODES:
        raise ValueError("Invalid prefill owner-tail mode")
    if model.prefill_tail_transport not in TAIL_TRANSPORTS:
        raise ValueError("Invalid prefill owner-tail transport")
    _validate(model, config)
    prequantized = _flag("ATOM_DSV41_PREFILL_QUANTIZED_FFN")
    local_router = _flag("ATOM_DSV41_PREFILL_LOCAL_ROUTER")
    route_packet = model.owner_ep4_contract is not None and _flag(
        "ATOM_DSV41_PREFILL_ROUTE_PACKET"
    )
    split = late_layer_start(config.hf_config) - 1
    for i in range(2, split):
        layer = model.layers[i]
        if layer.ffn.is_hash_layer or layer.ffn._use_comm_fused_dispatch:
            raise ValueError(
                "row-sharded prefill requires non-hash native routed experts"
            )
        layer.__class__ = OwnedRuntimeBlock
        layer.prequantized_prefill_ffn = prequantized
        layer.ffn.prefill_local_router = local_router
        layer.ffn.prefill_route_packet = route_packet
    logger.info(
        "row-sharded mHC/index: encoder layers 2..%s, TP4, M>=4096; "
        "tail mode %s, transport %s",
        split - 1,
        model.prefill_tail_mode,
        model.prefill_tail_transport,
    )


def build_stages(model, config):
    """The owner early stages, each under its own compile-cache tag."""
    with set_model_tag("backbone_early_owned"):
        model.early_owned = _EarlyOwnedLayers(
            atom_config=config, owner=model, last=model.decoder_start
        )
    if model.prefill_tail_mode != "off":
        with set_model_tag("backbone_early_owned_retained"):
            model.early_owned_retained = _EarlyOwnedLayers(
                atom_config=config,
                owner=model,
                last=model.decoder_start,
                retain_mode=model.prefill_tail_mode,
            )


def select_early(model, forward, running_tokens):
    """The early stage for this forward, and whether it hands over owner rows.

    Resolved before `begin_forward` and every state write."""
    if getattr(forward.attn_metadata, "image_mask", None) is not None or not eligible(
        running_tokens, TP_SIZE
    ):
        return model.early, False
    if model.prefill_tail_mode != "off" and replayable(forward, running_tokens):
        return model.early_owned_retained, True
    return model.early_owned, False


def prepare_replay(layer, state, cache, step, rope):
    """`Block.prepare_attention` plus the compressor for the decoder's first
    layer on retained owner rows; returns ``(normed, *seam)``."""
    from .attention import compress_batch

    prepared = layer.prepare_attention(localize(state), None, None)
    normed = prepared[0]
    compress_batch(
        layer.attn,
        normed[0] if isinstance(normed, tuple) else normed,
        cache,
        step,
        rope,
    )
    return prepared


def tail_gatherer(model, plan, running_tokens):
    """The decoder's `gather_tail`: the replay tail's rows from their owners."""
    from .tail import gather_owner_tail

    def gather_tail(*fields, out=None):
        return gather_owner_tail(
            plan.tail_rows_host,
            *fields,
            original_rows=running_tokens,
            transport=model.prefill_tail_transport,
            out=out,
        )

    return gather_tail

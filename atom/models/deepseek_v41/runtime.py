# SPDX-License-Identifier: MIT
"""ModelRunner interface over the V4.1 multimodal backbone."""

import logging

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import nn

from atom.model_ops.attentions.token_layout.gather import gather_rows
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.utils.backends import set_model_tag
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context

from .bounded_replay import (
    build_replay_plan,
    carry_selections,
    decoder_replay_unsupported,
    late_layer_start,
    replayable,
    rows_of,
)
from .model import Block
from .multimodal import DeepseekV41MultimodalModel

logger = logging.getLogger("atom")


def _fake_layer_output(hidden, layer_name):
    return torch.empty_like(hidden)


# The boundary ops carry a dependency on hidden so the compiler retains and
# orders the stream fork/join with the model's tensor work.
@torch_compile_guard(mutates_args=["hidden"], gen_fake=lambda hidden: None)
def v41_begin_forward(hidden: torch.Tensor) -> None:
    """Read live request state on every execution, including graph capture."""
    metadata = get_forward_context().attn_metadata
    metadata.step.begin_forward()
    if not metadata.step.requests:
        return
    if hidden.shape[-2] != metadata.step.width:
        raise ValueError("Token rows disagree with the width this step declared")
    stage = getattr(metadata.engram_embeddings, "stage", None)
    if stage is not None:
        stage()


@torch_compile_guard(mutates_args=["hidden"], gen_fake=lambda hidden: None)
def v41_end_forward(hidden: torch.Tensor) -> None:
    metadata = get_forward_context().attn_metadata
    if not metadata.step.requests:
        hidden.zero_()
        return
    rows = metadata.engram_embeddings
    if getattr(rows, "stage", None) is not None:
        rows.join()


@torch_compile_guard(mutates_args=[], gen_fake=_fake_layer_output)
def v41_attention(hidden: torch.Tensor, layer_name: str) -> torch.Tensor:
    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return torch.zeros_like(hidden)
    layer, rope = context.no_compile_layers[layer_name]
    # `hidden` arrives normed: the mHC seam (`pre_delayed`) applies the norm
    # before this boundary, so only the attention body runs here.
    # A bounded replay's layer 39 outputs some rows of `hidden`
    # (`metadata.queries`); it runs outside the compiled graphs.
    return layer.attn(
        hidden, None, metadata.cache, metadata.step, rope, metadata.queries
    )


@torch_compile_guard(mutates_args=[], gen_fake=_fake_layer_output)
def v41_engram(residual: torch.Tensor, layer_name: str) -> torch.Tensor:
    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return residual.clone()
    layer, _ = context.no_compile_layers[layer_name]
    embeddings = metadata.engram_embeddings.get(layer.engram.layer_id)
    return Block.engram_forward(layer, residual, embeddings, metadata.image_mask)


class RuntimeBlock(Block):
    """Serving uses the same guarded ops in eager and compiled execution."""

    def attention_forward(self, normed, cache, step, rope):
        return v41_attention(normed, self.layer_name)

    def engram_forward(self, residual, embeddings, image_mask):
        return v41_engram(residual, self.layer_name)


class _Stage(nn.Module):
    """A compiled piece of the runtime model's forward.

    The layers stay the owner's submodules (that is where the weights load);
    this holds the owner outside the module tree so parameter traversal does
    not reach them twice.
    """

    def __init__(self, *, atom_config, owner, first=0, last=0):
        super().__init__()
        self.__dict__["owner"] = owner
        self.first, self.last = first, last

    def run(self, state):
        m = self.owner
        for i in range(self.first, self.last):
            state = m.layers[i](state, None, None, m._rope(i), None, image_mask=None)
        return state


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class _Backbone(_Stage):
    """The whole forward as one graph: every V4.1 deployment that does not
    replay, exactly the graph the runtime model compiled before."""

    def forward(self, input_ids, positions, inputs_embeds=None):
        return self.owner.forward_hidden(
            input_ids.unsqueeze(0),
            None,
            None,
            image_mask=get_forward_context().attn_metadata.image_mask,
            inputs_embeds=(
                None if inputs_embeds is None else inputs_embeds.unsqueeze(0)
            ),
        ).squeeze(0)


def _rows_first(tensors):
    return tuple(None if t is None else t.squeeze(0) for t in tensors)


def _batched(tensors):
    return tuple(None if t is None else t[None] for t in tensors)


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "inputs_embeds": 0})
class _EarlyLayers(_Stage):
    """Under bounded replay: embedding, the Engram fork and the layers before
    the last KV source, on every row; the mHC state comes back as its five
    tensors, rows first."""

    def forward(self, input_ids, inputs_embeds=None):
        m = self.owner
        hidden = (m.embed(input_ids) if inputs_embeds is None else inputs_embeds)[None]
        m.begin_forward(hidden, None)
        state = self.run(SinglePassHCState.from_embeddings(hidden, m.config.hc_mult))
        return _rows_first(state.fields())


@support_torch_compile(
    dynamic_arg_dims={
        "residual": 0,
        "pre_mix": 0,
        "pending": 0,
        "post_mix": 0,
        "combination": 0,
    }
)
class _LateLayers(_Stage):
    """Under bounded replay: the decoder layers between its KV source and its
    last layer, on whichever rows they are handed -- the forward's, or a
    replay's tail. The mHC state in and out, rows first."""

    def forward(self, residual, pre_mix, pending, post_mix, combination):
        state = SinglePassHCState(
            *_batched((residual, pre_mix, pending, post_mix, combination))
        )
        return _rows_first(self.run(state).fields())


class DeepseekV41RuntimeModel(DeepseekV41MultimodalModel):
    # Weights arrive through the shared loader, the way V4's do, so the
    # renames, the packed projections and the expert mapping are declared once
    # as tables on `DeepseekV41ForCausalLM` and inherited here rather than
    # restated per model. Engram's mmap tables still come from
    # `model_loader.deepseek_v41.engram_tables`, which the Engram runtime
    # imports directly and does not route through here.
    #
    # With decoder SWA bounded replay (on by default) the forward is two
    # compiled graphs, the encoder and the decoder layers between its KV
    # source and its last layer, with those two layers run between and after
    # them: a prefill runs the decoder on each request's tail
    # (`bounded_replay.py`); decode runs everything on every row. With
    # --no-decoder-swa-bounded-replay it is one graph (`_Backbone`).

    block_cls = RuntimeBlock

    def __init__(self, atom_config):
        config = atom_config
        super().__init__(
            config.hf_config,
            max_length=config.max_model_len,
            online_quant_config=config.online_quant_config,
        )
        for i, layer in enumerate(self.layers):
            atom_config.compilation_config.static_forward_context[layer.layer_name] = (
                layer,
                self._rope(i),
            )
        self.replay = False
        self.late_aux_layers, self.aux_buffers = (), None
        if getattr(config, "enable_decoder_swa_bounded_replay", False):
            reason = decoder_replay_unsupported(config.hf_config)
            if reason is None:
                self._build_replay_stages(config)
            else:
                logger.warning("Decoder SWA bounded replay is off: %s.", reason)
        if not self.replay:
            self.backbone = _Backbone(atom_config=config, owner=self)

    def _build_replay_stages(self, config):
        hf = config.hf_config
        layers, split = hf.num_hidden_layers, late_layer_start(hf)
        self.replay = True
        self.decoder_start, self.last_layer = split - 1, layers - 1
        self.decoder_specs = self.topology[self.decoder_start : layers]
        self.early = _EarlyLayers(
            atom_config=config, owner=self, last=self.decoder_start
        )
        # Its own compile-cache tag: two graphs of one model must not share one.
        with set_model_tag("backbone_late"):
            self.late = _LateLayers(
                atom_config=config, owner=self, first=split, last=self.last_layer
            )
        logger.info(
            "Decoder SWA bounded replay: a prefill finishing its prompt runs "
            "the decoder (layers %d..%d) on its last window rows only and layer "
            "%d's query on its last row; one that does not stops at layer %d's "
            "compression",
            self.decoder_start,
            self.last_layer,
            self.last_layer,
            self.decoder_start,
        )

    def begin_forward(self, hidden, engram_embeddings):
        v41_begin_forward(hidden)

    def end_forward(self, hidden, engram_embeddings):
        v41_end_forward(hidden)

    def decoder_replay_active(self) -> bool:
        """Whether prefills replay: fixed at launch, since a prompt's earlier
        chunks skip the decoder and the scheduler backs its last one off for
        it (`Config.min_final_prefill_rows`)."""
        return self.replay

    def set_aux_hidden_state_rows(self, layer_ids, buffers):
        """DSpark's aux capture buffers, one per id in ``layer_ids`` -- the
        tensors its hooks write. A replay's decoder layers capture the tail's
        rows only, which ``_replayed`` moves to their forward rows; it runs the
        decoder's KV source outside its module, so no capture may hook it."""
        if not self.replay:
            return
        if self.decoder_start in layer_ids:
            raise ValueError(
                f"decoder bounded replay cannot capture layer {self.decoder_start}"
            )
        start = self.late.first
        self.late_aux_layers = tuple(
            k for k, layer in enumerate(layer_ids) if layer >= start
        )
        self.aux_buffers = buffers

    def _rope(self, i):
        return self.global_rope if self.topology[i].ratio else self.window_rope

    def _layer(self, i, state, queries=None):
        return self.layers[i](
            state, None, None, self._rope(i), None, image_mask=None, queries=queries
        )

    def _late_layers(self, state):
        return SinglePassHCState(*_batched(self.late(*_rows_first(state.fields()))))

    def forward(self, input_ids, positions, inputs_embeds=None):
        """Tensor-only serving entry; guarded ops read the live forward context."""
        if not self.replay:
            return self.backbone(input_ids, positions, inputs_embeds)
        state = SinglePassHCState(*_batched(self.early(input_ids, inputs_embeds)))
        if replayable(get_forward_context(), input_ids.numel()):
            return self._replayed(state, input_ids.numel())
        state = self._late_layers(self._layer(self.decoder_start, state))
        hidden = self._layer(self.last_layer, state).collapse()
        self.end_forward(hidden, None)
        return hidden.squeeze(0)

    def _replayed(self, state, num_tokens):
        """The forward's rows cut as `bounded_replay` lays them out: layer 20
        compresses every row; the requests that finish their prompts run the
        decoder on their tails, layer 39's query on their last row, which alone
        comes out. Every other row is zero: only each request's last row is
        read (the LM head's), and a chunk that holds none has its logits
        discarded."""
        metadata = get_forward_context().attn_metadata
        step, cache = metadata.step, metadata.cache
        plan = step.replay = build_replay_plan(step, cache.geometry.window_size)
        layer, rope = self.layers[self.decoder_start], self._rope(self.decoder_start)
        normed, *seam = layer.prepare_attention(state, None, None)
        del state
        layer.attn._compress_batch(normed, cache, step, rope)
        full = normed.new_zeros(num_tokens, self.config.hidden_size)
        if plan.tail is not None:
            carry_selections(step, plan, self.decoder_specs)
            normed, *seam = gather_rows(plan.tail_rows, normed, *seam, dim=1)
            with rows_of(metadata, plan.tail, rows=plan.tail_rows):
                output = layer.attention_forward(normed, None, None, rope)
                tail = self._late_layers(layer.ffn_forward(output, *seam))
                with rows_of(metadata, plan.tail, plan.q39, plan.q39.rows):
                    last = self._layer(self.last_layer, tail, plan.q39)
            self._move_aux_rows(plan.tail_rows)
            full[plan.last_rows] = last.collapse().squeeze(0)
        self.end_forward(full, None)
        return full

    def _move_aux_rows(self, rows):
        """Decoder-layer aux captures wrote the tail's rows at the buffer's head;
        move each to its forward ``rows``. The other rows hold stale values:
        the draft's context write puts them in ring slots whose positions lie
        below its window, which no draft block reads."""
        n = rows.numel()
        for k in self.late_aux_layers:
            buffer = self.aux_buffers[k]
            buffer[rows] = buffer[:n].clone()

    def compute_logits(self, hidden):
        return self.head.get_logits(self.norm(hidden))

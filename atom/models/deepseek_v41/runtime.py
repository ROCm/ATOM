# SPDX-License-Identifier: MIT
"""ModelRunner interface over the V4.1 multimodal backbone."""

import logging

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import nn

from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.utils.backends import set_model_tag
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context

from .bounded_replay import (
    build_late_layer_tail,
    decoder_replay_unsupported,
    late_layer_start,
    late_layer_tail,
    replay_rows,
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
    return Block.attention_forward(layer, hidden, metadata.cache, metadata.step, rope)


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
            spec, layer = m.topology[i], m.layers[i]
            rope = m.global_rope if spec.ratio else m.window_rope
            state = layer(state, None, None, rope, None, image_mask=None)
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


@support_torch_compile(dynamic_arg_dims={"input_ids": 0, "inputs_embeds": 0})
class _EarlyLayers(_Stage):
    """Under bounded replay: embedding, the Engram fork and the layers before
    the split, on every row; the mHC state comes back as its five tensors,
    rows first."""

    def forward(self, input_ids, inputs_embeds=None):
        m = self.owner
        hidden = (m.embed(input_ids) if inputs_embeds is None else inputs_embeds)[None]
        m.begin_forward(hidden, None)
        state = self.run(SinglePassHCState.from_embeddings(hidden, m.config.hc_mult))
        return tuple(
            None if t is None else t.squeeze(0)
            for t in (
                state.residual,
                state.pre_mix,
                state.pending,
                state.post_mix,
                state.combination,
            )
        )


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
    """Under bounded replay: the layers from the split on, the collapse and
    the Engram join, on whichever rows they are handed -- the forward's, or a
    replay's tail."""

    def forward(self, residual, pre_mix, pending, post_mix, combination):
        state = SinglePassHCState(
            *(
                None if t is None else t[None]
                for t in (residual, pre_mix, pending, post_mix, combination)
            )
        )
        hidden = self.run(state).collapse()
        self.owner.end_forward(hidden, None)
        return hidden.squeeze(0)


class DeepseekV41RuntimeModel(DeepseekV41MultimodalModel):
    # Weights arrive through the shared loader, the way V4's do, so the
    # renames, the packed projections and the expert mapping are declared once
    # as tables on `DeepseekV41ForCausalLM` and inherited here rather than
    # restated per model. Engram's mmap tables still come from
    # `model_loader.deepseek_v41.engram_tables`, which the Engram runtime
    # imports directly and does not route through here.
    #
    # With decoder SWA bounded replay (on by default) the forward is two
    # compiled graphs split after the last KV-source layer, so a prefill can
    # run the late one on each request's tail (`bounded_replay.py`); decode
    # runs both on every row. With --no-decoder-swa-bounded-replay it is one
    # graph (`_Backbone`).

    block_cls = RuntimeBlock

    def __init__(self, atom_config):
        config = atom_config
        super().__init__(
            config.hf_config,
            max_length=config.max_model_len,
            online_quant_config=config.online_quant_config,
        )
        for spec, layer in zip(self.topology, self.layers):
            rope = self.global_rope if spec.ratio else self.window_rope
            atom_config.compilation_config.static_forward_context[layer.layer_name] = (
                layer,
                rope,
            )
        self.replay = False
        # Turned off at runtime by a caller that reads every row of the
        # model's output or of a late layer's (TorchSpec hidden-state export).
        self.replay_enabled = True
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
        self.late_specs = self.topology[split:layers]
        self.early = _EarlyLayers(atom_config=config, owner=self, last=split)
        # Its own compile-cache tag: two graphs of one model must not share one.
        with set_model_tag("backbone_late"):
            self.late = _LateLayers(
                atom_config=config, owner=self, first=split, last=layers
            )
        logger.info(
            "Decoder SWA bounded replay: prefill runs layers %d..%d on each "
            "request's last window-ring rows",
            split,
            layers - 1,
        )

    def begin_forward(self, hidden, engram_embeddings):
        v41_begin_forward(hidden)

    def end_forward(self, hidden, engram_embeddings):
        v41_end_forward(hidden)

    def set_decoder_replay(self, enabled: bool) -> bool:
        """Allow or forbid replaying prefills; returns whether replay is now
        active. Forbidding it keeps the two graphs but runs the late one on
        every row, so every row of every layer is computed."""
        self.replay_enabled = enabled
        return self.replay and enabled

    def set_aux_hidden_state_rows(self, layer_ids, buffers):
        """DSpark's aux capture buffers, one per id in ``layer_ids`` -- the
        tensors its hooks write. A replay's late layers capture the tail's rows
        only, which ``_late_on_tail`` moves to their forward rows."""
        if self.replay:
            start = self.late.first
            self.late_aux_layers = tuple(
                k for k, layer in enumerate(layer_ids) if layer >= start
            )
            self.aux_buffers = buffers

    def forward(self, input_ids, positions, inputs_embeds=None):
        """Tensor-only serving entry; guarded ops read the live forward context."""
        if not self.replay:
            return self.backbone(input_ids, positions, inputs_embeds)
        state = self.early(input_ids, inputs_embeds)
        forward = get_forward_context()
        ring = replay_rows(forward, state[0].shape[0]) if self.replay_enabled else None
        if ring is None:
            return self.late(*state)
        tail = build_late_layer_tail(forward.attn_metadata.step, ring, self.late_specs)
        tail_state = SinglePassHCState(*state).take_rows(tail.token_indices)
        # Release the early graph's every-row state before the late layers run;
        # only the tail's rows are needed from here on.
        del state
        return self._late_on_tail(tail_state, tail, input_ids.numel())

    def _late_on_tail(self, tail_state, tail, num_tokens):
        rows = tail.token_indices
        with late_layer_tail(get_forward_context().attn_metadata, tail):
            hidden = self.late(*tail_state.fields())
        # Late-layer aux captures wrote the tail's rows at the buffer's head;
        # move each to its forward row. The other rows hold values nothing
        # reads: the draft's context write takes each request's last ring rows,
        # all of them in the tail.
        if self.late_aux_layers:
            buffers = self.aux_buffers
            n = rows.numel()
            for k in self.late_aux_layers:
                buffers[k].index_copy_(0, rows, buffers[k][:n].clone())
        # Rows outside the tail are zero: only each request's last row is read
        # (the LM head's), and it is in the tail. `compute_logits` still norms
        # every row; restricting it needs a per-forward signal that survives
        # CUDA-graph replay, which skips this Python.
        out = hidden.new_zeros(num_tokens, hidden.shape[-1])
        out.index_copy_(0, rows, hidden)
        return out

    def compute_logits(self, hidden):
        return self.head.get_logits(self.norm(hidden))

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
    cut_layer,
    cut_layer_rows,
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


def _fake_attention(hidden, layer_name, kv_written=False):
    return torch.empty_like(hidden)


@torch_compile_guard(mutates_args=[], gen_fake=_fake_attention)
def v41_attention(
    hidden: torch.Tensor, layer_name: str, kv_written: bool = False
) -> torch.Tensor:
    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return torch.zeros_like(hidden)
    layer, rope = context.no_compile_layers[layer_name]
    # `hidden` arrives normed: the mHC seam (`pre_delayed`) applies the norm
    # before this boundary, so only the attention body runs here.
    return Block.attention_forward(
        layer, hidden, metadata.cache, metadata.step, rope, kv_written
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

    def attention_forward(self, normed, cache, step, rope, kv_written=False):
        return v41_attention(normed, self.layer_name, kv_written)

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

    def rope(self, i):
        m = self.owner
        return m.global_rope if m.topology[i].ratio else m.window_rope

    def run(self, state):
        layers = self.owner.layers
        for i in range(self.first, self.last):
            state = layers[i](state, None, None, self.rope(i), None, image_mask=None)
        return state


def _rows_first(tensors):
    return tuple(None if t is None else t.squeeze(0) for t in tensors)


def _batched(tensors):
    return tuple(None if t is None else t[None] for t in tensors)


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
    """Under bounded replay, on every row: embedding, the Engram fork, the
    layers before the cut layer, and the cut layer's attention seam. Returns
    the seam's five tensors, rows first -- the cut layer's attention input."""

    def forward(self, input_ids, inputs_embeds=None):
        m = self.owner
        hidden = (m.embed(input_ids) if inputs_embeds is None else inputs_embeds)[None]
        m.begin_forward(hidden, None)
        state = self.run(SinglePassHCState.from_embeddings(hidden, m.config.hc_mult))
        return _rows_first(m.layers[self.last].prepare_attention(state, None, None))


@support_torch_compile(
    dynamic_arg_dims={
        "normed": 0,
        "residual": 0,
        "pre_mix": 0,
        "post_mix": 0,
        "combination": 0,
    }
)
class _CutLayer(_Stage):
    """Under bounded replay: the cut layer from its attention on, its cache
    writes done, on the rows it is handed -- every row, or the extended tail
    (`cut_layer_rows`). Returns the mHC state's five tensors, rows first."""

    def forward(self, normed, residual, pre_mix, post_mix, combination):
        seam = _batched((normed, residual, pre_mix, post_mix, combination))
        state = self.owner.layers[self.first](
            None, None, None, self.rope(self.first), None, seam=seam
        )
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
    """Under bounded replay: the layers from the split on, the collapse and
    the Engram join, on whichever rows they are handed -- the forward's, or a
    replay's tail."""

    def forward(self, residual, pre_mix, pending, post_mix, combination):
        state = SinglePassHCState(
            *_batched((residual, pre_mix, pending, post_mix, combination))
        )
        hidden = self.run(state).collapse()
        self.owner.end_forward(hidden, None)
        return hidden.squeeze(0)


def _take_rows(tensors, rows):
    return tuple(None if t is None else t.index_select(0, rows) for t in tensors)


class DeepseekV41RuntimeModel(DeepseekV41MultimodalModel):
    # Weights arrive through the shared loader, the way V4's do, so the
    # renames, the packed projections and the expert mapping are declared once
    # as tables on `DeepseekV41ForCausalLM` and inherited here rather than
    # restated per model. Engram's mmap tables still come from
    # `model_loader.deepseek_v41.engram_tables`, which the Engram runtime
    # imports directly and does not route through here.
    #
    # With decoder SWA bounded replay (on by default) the forward is three
    # compiled graphs cut inside the last KV-source layer: the early layers
    # and the cut layer's cache writes, the rest of the cut layer, the late
    # layers. A prefill runs the second on each request's extended tail and
    # the third on its tail (`bounded_replay.py`); decode runs all three on
    # every row. With --no-decoder-swa-bounded-replay it is one graph
    # (`_Backbone`).

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
        layers, cut, split = hf.num_hidden_layers, cut_layer(hf), late_layer_start(hf)
        self.replay = True
        self.cut_specs = self.topology[cut:layers]
        self.late_specs = self.topology[split:layers]
        self.early = _EarlyLayers(atom_config=config, owner=self, last=cut)
        # Their own compile-cache tags: graphs of one model must not share one.
        with set_model_tag("backbone_cut"):
            self.cut = _CutLayer(atom_config=config, owner=self, first=cut, last=split)
        with set_model_tag("backbone_late"):
            self.late = _LateLayers(
                atom_config=config, owner=self, first=split, last=layers
            )
        logger.info(
            "Decoder SWA bounded replay: prefill runs layer %d's attention and "
            "FFN on each request's last window-ring + window rows and layers "
            "%d..%d on its last window-ring rows",
            cut,
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
            if self.cut.first in layer_ids:
                # its rows are split across two graphs, neither holding them all
                raise ValueError(
                    f"Bounded replay cannot capture aux hidden states at layer "
                    f"{self.cut.first}, the layer it cuts"
                )
            start = self.late.first
            self.late_aux_layers = tuple(
                k for k, layer in enumerate(layer_ids) if layer >= start
            )
            self.aux_buffers = buffers

    def forward(self, input_ids, positions, inputs_embeds=None):
        """Tensor-only serving entry; guarded ops read the live forward context."""
        if not self.replay:
            return self.backbone(input_ids, positions, inputs_embeds)
        seam = self.early(input_ids, inputs_embeds)
        forward = get_forward_context()
        self._write_cut_kv(forward, seam[0])
        ring = replay_rows(forward, seam[0].shape[0]) if self.replay_enabled else None
        if ring is None:
            return self.late(*self.cut(*seam))
        metadata = forward.attn_metadata
        extended = build_late_layer_tail(
            metadata.step, cut_layer_rows(metadata.cache.geometry), self.cut_specs
        )
        # Release the early graph's every-row seam as soon as the extended
        # tail's rows are out of it; only those are needed from here on.
        seam = _take_rows(seam, extended.token_indices)
        with late_layer_tail(metadata, extended):
            state = self.cut(*seam)
            del seam
            tail = build_late_layer_tail(extended.step, ring, self.late_specs)
            state = _take_rows(state, tail.token_indices)
            with late_layer_tail(metadata, tail):
                hidden = self.late(*state)
        return self._scatter_tail(
            hidden, extended.token_indices[tail.token_indices], input_ids.numel()
        )

    def _write_cut_kv(self, forward, normed):
        """The cut layer's cache writes, on every row (`Block.write_kv`).

        Eager, between the graphs: the writes are opaque to the compiler
        either way, and inside the early graph Inductor's first-use log of a
        new op stringifies its input's IR, twenty layers deep, which does not
        finish. A CUDA-graph capture of the forward records them like the
        graphs around them.
        """
        metadata = forward.attn_metadata
        if not metadata.step.requests:
            return
        layer = self.layers[self.cut.first]
        rope = self.cut.rope(self.cut.first)
        Block.write_kv(layer, normed[None], metadata.cache, metadata.step, rope)

    def _scatter_tail(self, hidden, rows, num_tokens):
        """The late layers' output, and aux captures, at their forward rows."""
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

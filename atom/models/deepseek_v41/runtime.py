# SPDX-License-Identifier: MIT
"""ModelRunner interface over the V4.1 multimodal backbone."""

import logging

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from torch import nn

from atom.model_ops.attentions.token_layout.gather import gather_rows
from atom.model_ops.deepseek_v41.mhc import SinglePassHCState
from atom.model_ops.engram.device.staging import EngramStagedRows
from atom.utils.backends import set_model_tag
from atom.utils.decorators import support_torch_compile
from atom.utils.forward_context import get_forward_context

from .bounded_replay import (
    build_replay_plan,
    carry_selections,
    decoder_replay_unsupported,
    late_layer_start,
    placeholder_rows_of,
    planned_stage_tokens,
    replayable,
    rows_of,
    run_decoder_stages,
)
from .config import validate_tbo_parallelism
from .model import Block
from .multimodal import DeepseekV41MultimodalModel

logger = logging.getLogger("atom")


def _fake_layer_output(hidden, layer_name, hidden_fp8=None, hidden_scale=None):
    return torch.empty_like(hidden)


# The boundary ops carry a dependency on hidden so the compiler retains and
# orders the stream fork/join with the model's tensor work.
@torch_compile_guard(mutates_args=["hidden"], gen_fake=lambda hidden: None)
def v41_begin_forward(hidden: torch.Tensor) -> None:
    """Read live request state on every execution, including graph capture."""
    context = get_forward_context()
    metadata = context.attn_metadata
    metadata.step.begin_forward()
    if not metadata.step.requests:
        return
    # Checked before the staged path returns, so it covers both. What the
    # width governs is how every later index into this batch is computed, so a
    # mismatch is not confined to staging: skipping the check for the plugin
    # left its main path to discover the same disagreement as a reshape error
    # inside a layer, or not at all, with padding rows read as real ones.
    if hidden.shape[-2] != metadata.step.width:
        raise ValueError("Token rows disagree with the width this step declared")
    if getattr(metadata, "staged_outside_forward", False):
        # The plugin staged this step in the metadata builder, where no graph
        # can capture it. Re-staging here would advance the cursor and reset
        # the slots a second time.
        return
    if isinstance(metadata.engram_embeddings, EngramStagedRows):
        metadata.engram_embeddings.stage()


@torch_compile_guard(mutates_args=["hidden"], gen_fake=lambda hidden: None)
def v41_end_forward(hidden: torch.Tensor) -> None:
    """Closes what `v41_begin_forward` opened."""
    metadata = get_forward_context().attn_metadata
    if not metadata.step.requests:
        hidden.zero_()
        return
    if getattr(metadata, "staged_outside_forward", False):
        # Staged and joined by the metadata builder, before this forward. Its
        # fork was opened and closed out there; waiting on it from in here
        # would be a captured stream waiting on uncaptured work.
        return
    rows = metadata.engram_embeddings
    if isinstance(rows, EngramStagedRows):
        rows.join()


@torch_compile_guard(mutates_args=[], gen_fake=_fake_layer_output)
def v41_attention(
    hidden: torch.Tensor,
    layer_name: str,
    hidden_fp8: torch.Tensor | None = None,
    hidden_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    context = get_forward_context()
    metadata = context.attn_metadata
    if not metadata.step.requests:
        return torch.zeros_like(hidden)
    layer, rope = context.no_compile_layers[layer_name]
    # `hidden` arrives normed: the mHC seam (`pre_delayed`) applies the norm
    # before this boundary, so only the attention body runs here.
    # A bounded replay's layer 39 outputs some rows of `hidden`
    # (`metadata.queries`); it runs outside the compiled graphs.
    options = {} if hidden_fp8 is None else {"hidden_fp8": hidden_fp8}
    return layer.attn(
        hidden,
        hidden_scale,
        metadata.cache,
        metadata.step,
        rope,
        metadata.queries,
        **options,
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
        if isinstance(normed, tuple):
            hidden, hidden_fp8, hidden_scale = normed
            return v41_attention(hidden, self.layer_name, hidden_fp8, hidden_scale)
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


def _gather_attention_rows(rows, normed, *seam):
    """Keep BF16 hidden, optional fused quantization and mHC state aligned."""
    if isinstance(normed, tuple):
        hidden, quant, scale, *seam = gather_rows(
            rows,
            normed[0],
            normed[1],
            normed[2].view(torch.uint8),
            *seam,
            dim=1,
        )
        return (hidden, quant, scale.view(torch.float8_e8m0fnu)), *seam
    return gather_rows(rows, normed, *seam, dim=1)


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
        # Config is built before legacy DP kwargs and DPA rank expansion apply.
        # Validate their final width here, before allocating model parameters.
        if getattr(config, "enable_tbo", False):
            validate_tbo_parallelism(config)
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
            "compression (under DP attention it joins a peer's decoder "
            "collectives on one placeholder row)",
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
        forward, num_tokens = get_forward_context(), input_ids.numel()
        plan = None
        if replayable(forward, num_tokens):
            step = forward.attn_metadata.step
            window = forward.attn_metadata.cache.geometry.window_size
            plan = step.replay = build_replay_plan(step, window)
        self._check_stage_report(forward, plan, num_tokens)
        if plan is not None:
            return self._replayed(state, num_tokens, plan)
        hidden = run_decoder_stages(
            forward,
            lambda: self._layer(self.decoder_start, state),
            self._late_layers,
            lambda late: self._layer(self.last_layer, late),
        ).collapse()
        self.end_forward(hidden, None)
        return hidden.squeeze(0)

    def _check_stage_report(self, forward, plan, num_tokens):
        """Under DP attention, check this forward runs the token counts its
        rank reported to the group (`decoder_stage_report`): a rank that ran
        others would leave its peers waiting in a collective. The replay
        decision itself is the reported one (`replayable`); this guards the
        counts, computed from the batch on one side and the plan on the
        other."""
        stages = getattr(forward, "dp_stages", None)
        if stages is None:
            return
        reported = tuple(
            stages.stages[name].local_tokens for name in ("decoder", "last_layer")
        )
        planned = planned_stage_tokens(plan, num_tokens)
        if planned != reported:
            raise RuntimeError(
                f"this DP rank runs {planned} decoder/last-layer tokens but "
                f"reported {reported} to its DP group"
            )

    def _replayed(self, state, num_tokens, plan):
        """The forward's rows cut as `bounded_replay` lays them out: layer 20
        compresses every row; the requests that finish their prompts run the
        decoder on their tails, layer 39's query on their last row, which alone
        comes out. Every other row is zero: only each request's last row is
        read (the LM head's), and a chunk that holds none has its logits
        discarded.

        Under DP attention each stage's MoE collectives are sized as the
        group agreed (`run_decoder_stages`), and a rank with no tail joins
        them on one placeholder row while a peer has decoder work."""
        forward = get_forward_context()
        metadata = forward.attn_metadata
        step, cache = metadata.step, metadata.cache
        layer, rope = self.layers[self.decoder_start], self._rope(self.decoder_start)
        normed, *seam = layer.prepare_attention(state, None, None)
        del state
        hidden = normed[0] if isinstance(normed, tuple) else normed
        layer.attn._compress_batch(hidden, cache, step, rope)
        full = hidden.new_zeros(num_tokens, self.config.hidden_size)
        if plan.tail is not None:
            carry_selections(step, plan, self.decoder_specs)
            normed, *seam = _gather_attention_rows(plan.tail_rows, normed, *seam)

            def last_layer(tail):
                with rows_of(metadata, plan.tail, plan.q39, plan.q39.rows):
                    return self._layer(self.last_layer, tail, plan.q39)

            with rows_of(metadata, plan.tail, rows=plan.tail_rows):
                last = run_decoder_stages(
                    forward,
                    lambda: layer.ffn_forward(
                        layer.attention_forward(normed, None, None, rope), *seam
                    ),
                    self._late_layers,
                    last_layer,
                )
            self._move_aux_rows(plan.tail_rows)
            full[plan.last_rows] = last.collapse().squeeze(0)
        elif forward.dp_stages is not None and forward.dp_stages.active:
            self._join_decoder_collectives(forward, layer, rope, normed, seam)
        self.end_forward(full, None)
        return full

    def _join_decoder_collectives(self, forward, layer, rope, normed, seam):
        """Join the decoder's DP collectives without a tail of one's own.

        A rank whose prefill finishes no prompt runs none of the decoder, but
        while a peer has decoder work its MoE gathers count on this rank once
        per layer. It runs the decoder on its first row under a request-less
        step (`placeholder_rows_of`): attention returns zeros and writes no
        cache, so the row only feeds the collectives, and its output is
        dropped. V4.1 rejects EPLB (`validate_runtime_config`), so the row's
        routing reaches no expert-load statistics."""
        metadata = forward.attn_metadata
        hidden = normed[0] if isinstance(normed, tuple) else normed
        rows = torch.zeros(1, dtype=torch.int32, device=hidden.device)
        normed, *seam = _gather_attention_rows(rows, normed, *seam)
        with placeholder_rows_of(metadata, rows):
            run_decoder_stages(
                forward,
                lambda: layer.ffn_forward(
                    layer.attention_forward(normed, None, None, rope), *seam
                ),
                self._late_layers,
                lambda late: self._layer(self.last_layer, late),
            )

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

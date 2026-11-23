# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark pieces of the SGLang plugin that are not SGLang models.

Target side (`V41DSparkTargetSurface`): what SGLang's ``DSparkWorkerV2`` reads
off the target wrapper (``lm_head`` with ``weight``, ``get_input_embeddings``,
``set_dspark_layers_to_capture``), ATOM's own aux taps on target layers 37-39,
and the draft's context-KV write inside the target forward that produced the
rows -- native ``DSparkProposer.compute_draft_kv``'s place and contract.

Draft side (`draft_block_inputs`, `v41_dspark_draft_forward_context`): one
draft block per forward, a TARGET_VERIFY of five rows a request on the draft
runner, translated into ATOM's ``block_backbone`` call. Every value comes from
the current ``ForwardBatch`` and the current runner's pool; nothing is carried
from the target's verify.
"""

from __future__ import annotations

import logging
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import torch

from atom.plugin.sglang.deepseek_v41_bridge import (
    V41_DSPARK_DRAFT_TOKENS,
    _is_target_verify,
    install_deepseek_v41_dspark_commit_patch,
)

logger = logging.getLogger("atom")


# --- Target ---


class V41TargetLMHeadView:
    """The target ``lm_head`` SGLang's DSpark worker reads.

    Not a module: ATOM's head is already a parameter of the target model, and
    a second module holding it would register the weight twice. ``weight`` is
    resolved on every read, so it is the loaded (and post-processed) tensor.
    ``atom_backbone`` / ``atom_wrapper`` are what the ATOM draft wrapper needs
    from it in ``attach_shared_modules``.
    """

    def __init__(self, wrapper) -> None:
        self._wrapper = weakref.ref(wrapper)

    @property
    def atom_wrapper(self):
        return self._wrapper()

    @property
    def atom_backbone(self):
        # V4.1's `model` property is the backbone itself; embed and head live
        # on it.
        return self.atom_wrapper.model

    @property
    def weight(self) -> torch.Tensor:
        return self.atom_backbone.head.weight

    @property
    def tp_size(self) -> int:
        return int(self.atom_backbone.head.tp_size)

    @property
    def org_vocab_size(self) -> int:
        return int(self.atom_backbone.head.num_embeddings)

    @property
    def num_embeddings_per_partition(self) -> int:
        return int(self.atom_backbone.head.num_embeddings_per_partition)


class V41DSparkTargetSurface:
    """The DSpark half of a V4.1 target wrapper; one per wrapper.

    Aux capture is ATOM's own: `install_aux_taps` with the attached draft's
    native ``target_aux_capture_spec`` (the settled residual entering the
    layer, averaged over the hc streams). The taps write into a per-forward
    dict, cleared at forward start, so nothing is allocated outside the
    profiled budget and a missing tap is an error, not a stale row.
    """

    def __init__(self, wrapper) -> None:
        self._wrapper = weakref.ref(wrapper)
        self.layer_ids: tuple[int, ...] | None = None
        self.draft = None
        self._rows: dict[int, torch.Tensor] = {}

    @property
    def wrapper(self):
        return self._wrapper()

    # -- SGLang-facing ---------------------------------------------------

    def set_layers_to_capture(self, layer_ids) -> None:
        wrapper = self.wrapper
        hf = wrapper.atom_config.hf_config
        if layer_ids is None:
            raise ValueError("DeepSeek-V4.1 DSpark needs explicit capture layer ids")
        ids = tuple(int(i) for i in layer_ids)
        expected = tuple(int(i) for i in hf.dspark_target_layer_ids)
        if ids != expected:
            raise ValueError(
                f"DeepSeek-V4.1 DSpark taps layers {expected} (the checkpoint's "
                f"dspark_target_layer_ids); SGLang asked for {ids}"
            )
        if self.layer_ids is not None:
            if self.layer_ids == ids:
                return
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark taps are installed on {self.layer_ids}"
            )
        if self.draft is None:
            # SGLang 0.5.22 builds the DSpark worker (which attaches the
            # draft) before the target's attention backends ask for the taps.
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark taps asked for before the draft was attached"
            )
        late = _replay_late_start(wrapper.model)
        if late is not None and min(ids) < late:
            # A replay runs layers before `late` on every row (and layer
            # late - 1 outside its module): the taps would disagree on rows.
            raise ValueError(
                f"DeepSeek-V4.1 DSpark taps {ids} under decoder SWA bounded "
                f"replay must all be decoder layers >= {late}; set "
                "ATOM_SGLANG_V41_DECODER_REPLAY=0"
            )
        from atom.spec_decode.drafter import install_aux_taps

        # The draft's own spec, as native's drafter installs it (input taps;
        # it refuses a tap at an Engram layer, read before its injection).
        spec = self.draft.target_aux_capture_spec(ids, int(hf.hidden_size))
        install_aux_taps(wrapper.model, spec, ids, self._make_hook)
        self.layer_ids = ids
        wrapper.capture_aux_hidden_states = True
        logger.info(
            "ATOM V4.1 DSpark aux taps layers=%s capture=input "
            "extract=_target_layer_input",
            ids,
        )

    def get_input_embeddings(self):
        return self.wrapper.model.embed

    def attach_draft(self, draft) -> None:
        """The ATOM draft model whose context window this target writes.
        A construction-time link (`attach_shared_modules`), not step state."""
        if self.draft is not None and self.draft is not draft:
            raise RuntimeError("A DeepSeek-V4.1 DSpark draft is already attached")
        self.draft = draft

    # -- per forward -----------------------------------------------------

    def _make_hook(self, layer_id, extract):
        from atom.spec_decode.drafter import _aux_hook

        def write(tensor, tokens):
            if (
                not isinstance(tokens, slice)
                or tokens.start != 0
                or tokens.stop != tensor.shape[0]
            ):
                raise RuntimeError(
                    f"DeepSeek-V4.1 DSpark tap {layer_id} wrote rows {tokens}; "
                    "the plugin forward is one batch from row 0"
                )
            if layer_id in self._rows:
                raise RuntimeError(
                    f"DeepSeek-V4.1 DSpark tap {layer_id} fired twice in a forward"
                )
            self._rows[layer_id] = tensor

        return _aux_hook(extract, write)

    def begin_forward(self) -> None:
        if self.layer_ids is None:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark target forward before SGLang set the "
                "capture layers (set_dspark_layers_to_capture)"
            )
        self._rows.clear()

    def take_aux(self, rows: int) -> torch.Tensor:
        """This forward's ``[rows, len(taps) * hidden]`` aux block, in tap
        order (SGLang's FULL capture packs the list the same way)."""
        missing = [i for i in self.layer_ids if i not in self._rows]
        if missing:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark taps {missing} did not fire this forward"
            )
        taps = [self._rows[i] for i in self.layer_ids]
        self._rows.clear()
        for layer_id, tap in zip(self.layer_ids, taps):
            if tap.dim() != 2 or int(tap.shape[0]) != rows:
                raise RuntimeError(
                    f"DeepSeek-V4.1 DSpark tap {layer_id} has shape "
                    f"{tuple(tap.shape)}, the forward {rows} rows"
                )
        return torch.cat(taps, dim=-1)

    def finish_forward(self, hidden_states, positions, forward_batch) -> torch.Tensor:
        """After the target model, inside its ATOM forward context: write the
        draft's context KV for every row this forward ran and hand the aux
        block to SGLang's logits processor (``aux_hidden_states``).

        Every row the late layers ran, as native ``compute_draft_kv`` writes
        every scheduled row: a radix hit's replay rows ``[P - R, P)``
        included, which ride in this forward (``hidden_states`` /
        ``positions`` are the widened rows, and the wrapper gathers SGLang's
        rows out of the returned aux afterwards). That is what rebuilds the
        draft's windows below the hit.

        Under decoder SWA bounded replay (``step.replay``) the late layers --
        the taps' -- ran on each request's last ``window`` rows alone, and
        ATOM's ``write_context_kv`` writes exactly those under the tail's
        step. A fused hit's span ``[P - R, L)`` is at least ``R >= window``
        rows (`deepseek_v41_bridge.v41_extend_replays`), so its tail is
        ``[L - window, L)``: the first draft block's whole context, whether it
        lies in SGLang's rows or reaches into the replay rows."""
        from atom.utils.forward_context import get_forward_context

        forward = get_forward_context()
        metadata = forward.attn_metadata
        step = metadata.step
        rows = int(hidden_states.shape[0])
        mode = forward_batch.forward_mode
        if _is_target_verify(mode):
            width = int(forward_batch.spec_info.draft_token_num)
            if rows != int(forward_batch.batch_size) * width:
                raise RuntimeError(
                    f"DeepSeek-V4.1 DSpark verify returned {rows} rows for "
                    f"{forward_batch.batch_size} requests x {width}"
                )
        elif mode.is_decode():
            raise RuntimeError("DeepSeek-V4.1 DSpark runs no plain DECODE")
        if int(step.running_tokens) != rows or int(positions.shape[0]) != rows:
            raise RuntimeError(
                f"DeepSeek-V4.1 DSpark forward of {rows} rows, step "
                f"{step.running_tokens}, positions {int(positions.shape[0])}"
            )
        if metadata.served_by_round:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark on SGLang writes the draft context from "
                "the plain forward only (no decode round)"
            )
        # The taps exist only with a draft attached (`set_layers_to_capture`).
        plan = step.replay
        if plan is None:
            aux = self.take_aux(rows)
        else:
            if _is_target_verify(mode):
                raise RuntimeError("DeepSeek-V4.1 DSpark verify ran a bounded replay")
            aux = tail_aux_to_forward_rows(
                self.take_aux(int(plan.tail.running_tokens)), plan, rows
            )
        # Native's drafter call: under a replay it takes `aux[plan.tail_rows]`
        # and writes them under the tail's step, so only the rows above are
        # read; the rest are zero (no draft block reads them). ATOM takes the
        # taps as a sequence (`DSparkDraftModel.write_context_kv`); this one
        # block is them already joined, as a PP relay hands them, which ATOM
        # projects as is.
        self.draft.write_context_kv([aux], positions)
        return aux


def tail_aux_to_forward_rows(tail_aux, plan, rows: int) -> torch.Tensor:
    """A replay's aux block ``[tail width, F]`` (the late-layer taps ran on
    the tail alone) as the forward's ``[rows, F]``: each real tail row moved
    to its forward row (``plan.tail_rows[:plan.filled]``), as native's
    ``DeepseekV41RuntimeModel._move_aux_rows`` moves its capture buffers.
    Every other row is zero: the forward computed no aux for it."""
    width = int(plan.tail.running_tokens)
    if tail_aux.dim() != 2 or int(tail_aux.shape[0]) != width:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark replay aux {tuple(tail_aux.shape)} is not the "
            f"tail's {width} rows"
        )
    filled = int(plan.filled)
    if filled > width or filled > rows:
        raise RuntimeError(
            f"DeepSeek-V4.1 DSpark replay fills {filled} tail rows of {width} "
            f"in a forward of {rows}"
        )
    aux = tail_aux.new_zeros(rows, tail_aux.shape[1])
    if filled:
        aux[plan.tail_rows[:filled]] = tail_aux[:filled]
    return aux


def _replay_late_start(model) -> int | None:
    """The first layer a replay runs on the tail alone (`_LateLayers`), or
    None where the runtime model does not replay."""
    return int(model.late.first) if model.replay else None


def install_v41_dspark_target_surface(wrapper) -> V41DSparkTargetSurface:
    """Give a V4.1 target wrapper what SGLang's DSpark worker reads off it.

    Instance attributes, set only for a DSpark deployment: on every other
    model the wrapper keeps ``hasattr(model, "lm_head")`` False and has no
    ``set_dspark_layers_to_capture``, exactly as before."""
    surface = V41DSparkTargetSurface(wrapper)
    wrapper._v41_dspark = surface
    wrapper.lm_head = V41TargetLMHeadView(wrapper)
    wrapper.get_input_embeddings = surface.get_input_embeddings
    wrapper.set_dspark_layers_to_capture = surface.set_layers_to_capture
    wrapper.attach_dspark_draft = surface.attach_draft
    install_deepseek_v41_dspark_commit_patch()
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark target surface: lm_head view, input "
        "embeddings, capture layers, in-forward draft context write"
    )
    return surface


# --- Draft ---


def draft_block_inputs(
    forward_batch, input_ids, positions, *, noise_token_id: int, width=None
):
    """The draft block of one draft forward, from the current ``ForwardBatch``.

    SGLang runs the block as a TARGET_VERIFY of ``draft_token_num`` (5) rows a
    request: ``[bonus, noise x 4]`` at positions ``p..p+4``, with ``p`` the
    committed length (``seq_lens``). ATOM's ``block_backbone`` takes the bonus
    as the anchor token and its anchor position as the last processed target
    row, ``p - 1``, whose block runs at ``anchor + 1 .. anchor + 5``.
    """
    width = V41_DSPARK_DRAFT_TOKENS if width is None else int(width)
    mode = forward_batch.forward_mode
    if not _is_target_verify(mode):
        raise NotImplementedError(
            f"DeepSeek-V4.1 DSpark draft runs only its block (TARGET_VERIFY); got {mode}"
        )
    # SGLang's block is ``gamma`` (5) rows a request (`spec_info.draft_token_num`).
    bs = int(forward_batch.batch_size)
    ids = input_ids.view(bs, width)
    pos = positions.view(bs, width)
    return SimpleNamespace(
        bs=bs,
        width=width,
        anchor_ids=ids[:, 0].contiguous(),
        # A no-op on a live row (position >= 1). A graph's padding row (SGLang
        # zero-pads positions and req_pool_indices) then drafts what native's
        # `_stage_anchors_kernel` pads with: anchor position 0 in STATE slot
        # 0, the never-allocated padding row.
        anchor_positions=(pos[:, 0] - 1).clamp_min(0).contiguous(),
        slots=forward_batch.req_pool_indices[:bs].to(torch.int32).contiguous(),
        positions=positions.reshape(-1),
    )


@contextmanager
def v41_dspark_draft_forward_context(atom_config: Any, *, cache, block):
    """ATOM's forward context for one draft block: the current draft pool's
    cache, the block's own STATE slots, a decode-shaped context of
    ``bs * width`` rows marked as a draft pass. No target step rides along
    (``step`` is None): the block reads committed windows only
    (`read_windows` refuses while a tentative step is pending)."""
    from atom.utils.forward_context import (
        AttentionMetaData,
        AttnState,
        Context,
        DPMetadata,
        RoundService,
        reset_forward_context,
        set_forward_context,
    )

    rows = block.bs * block.width
    metadata = AttentionMetaData(max_seqlen_q=block.width, state=AttnState.DECODE)
    metadata.cache = cache
    metadata.step = None
    metadata.state_slot_out = block.slots
    metadata.served_by_round = RoundService.NONE
    metadata.queries = None
    metadata.image_mask = None
    metadata.dummy = False
    context = Context(
        positions=block.positions,
        is_prefill=False,
        is_dummy_run=False,
        scheduled_bs=block.bs,
        scheduled_tokens=rows,
        running_bs=block.bs,
        running_tokens=rows,
        is_draft=True,
        running_tokens_are_unified=True,
    )
    set_forward_context(
        attn_metadata=metadata,
        atom_config=atom_config,
        context=context,
        running_tokens_per_dp_rank=DPMetadata.gather_running_tokens(
            atom_config.parallel_config, rows
        ),
    )
    try:
        yield metadata
    finally:
        reset_forward_context()

# SPDX-License-Identifier: MIT
"""DeepSeek-V4.1 DSpark pieces of the SGLang plugin that are not SGLang models.

Target side (`V41DSparkTargetSurface`): what SGLang's ``DSparkWorkerV2`` reads
off the target wrapper (``lm_head`` with ``weight``, ``get_input_embeddings``,
``set_dspark_layers_to_capture``), ATOM's own aux taps on target layers 37-39,
and the draft's context-KV write inside the target forward that produced the
rows -- native ``DSparkProposer.compute_draft_kv``'s place and contract.
"""

from __future__ import annotations

import logging
import weakref

import torch

from atom.plugin.sglang.deepseek_v41_bridge import (
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
        block to SGLang's logits processor (``aux_hidden_states``)."""
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
        if (
            metadata.served_by_round
            or step.replay is not None
            or step.decoder is not None
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark on SGLang writes the draft context from "
                "the plain forward only (no decode round, no bounded replay)"
            )
        # The taps exist only with a draft attached (`set_layers_to_capture`).
        aux = self.take_aux(rows)
        # ATOM takes the taps as a sequence (`DSparkDraftModel.write_context_kv`);
        # this one block is them already joined, as a PP relay hands them,
        # which ATOM projects as is.
        self.draft.write_context_kv([aux], positions)
        return aux


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

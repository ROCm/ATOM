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

import numpy as np
import torch

from atom.plugin.sglang.deepseek_v41_bridge import (
    V41_DSPARK_DRAFT_TOKENS,
    _cpu_ints,
    _is_target_verify,
    install_deepseek_v41_dspark_commit_patch,
    v41_dspark_checks_enabled,
)

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------


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
        wrapper = self._wrapper()
        if wrapper is None:
            raise RuntimeError("The DeepSeek-V4.1 target wrapper is gone")
        return wrapper

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

    Aux capture is ATOM's own (`install_aux_taps` with the draft's
    ``target_aux_capture_spec`` convention: the settled residual entering the
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
        wrapper = self._wrapper()
        if wrapper is None:
            raise RuntimeError("The DeepSeek-V4.1 target wrapper is gone")
        return wrapper

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
        # The draft's own rule (`DeepseekV41DSpark.target_aux_capture_spec`):
        # an input tap at an Engram layer would read before its injection.
        engram = {int(i) for i in (getattr(hf, "engram_layer_ids", None) or ())}
        if engram.intersection(ids):
            raise ValueError("V4.1 DSpark input taps cannot precede Engram injection")
        from atom.models.deepseek_v41.dspark import DeepseekV41DSpark
        from atom.spec_decode.drafter import AuxCaptureSpec, install_aux_taps

        spec = AuxCaptureSpec(
            ids,
            int(hf.hidden_size),
            DeepseekV41DSpark._target_layer_input,
            capture="input",
        )
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
            or getattr(step, "replay", None) is not None
            or getattr(step, "decoder", None) is not None
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark on SGLang writes the draft context from "
                "the plain forward only (no decode round, no bounded replay)"
            )
        aux = self.take_aux(rows)
        if self.draft is None:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark target forward ran with no draft attached"
            )
        self.draft.write_context_kv(aux, positions)
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
    log_v41_dspark_runtime_sources()
    return surface


def log_v41_dspark_runtime_sources() -> None:
    """G0 at startup: the files this rank actually imported for the path."""
    import importlib
    import inspect
    import os

    modules = (
        "sglang.srt.speculative.dspark_components.dspark_worker_v2",
        "sglang.srt.speculative.dspark_components.dspark_draft",
        "sglang.srt.speculative.dspark_components.dspark_verify",
        "sglang.srt.speculative.dspark_components.dspark_kv_inject",
        "sglang.srt.model_executor.runner.eager_runner",
        "atom.plugin.sglang.deepseek_v41_bridge",
        "atom.plugin.sglang.deepseek_v41_dspark",
        "atom.plugin.sglang.models.deepseek_v41_dspark_wrapper",
        "atom.models.deepseek_v41.dspark",
    )
    sources = {}
    for name in modules:
        try:
            sources[name.rsplit(".", 1)[-1]] = inspect.getfile(
                importlib.import_module(name)
            )
        except Exception as exc:  # noqa: BLE001 - a log line, not a gate
            sources[name.rsplit(".", 1)[-1]] = f"<{type(exc).__name__}>"
    try:
        import sglang

        version = getattr(sglang, "__version__", "?")
    except Exception:  # noqa: BLE001
        version = "?"
    logger.info(
        "ATOM DeepSeek-V4.1 DSpark runtime: sglang=%s RAGGED_VERIFY_MODE=%s "
        "checks=%s sources=%s",
        version,
        os.environ.get("SGLANG_RAGGED_VERIFY_MODE", "static"),
        v41_dspark_checks_enabled(),
        sources,
    )


# ---------------------------------------------------------------------------
# Draft
# ---------------------------------------------------------------------------


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
    spec_width = int(getattr(forward_batch.spec_info, "draft_token_num", 0) or 0)
    if spec_width != width:
        raise ValueError(
            f"DeepSeek-V4.1 DSpark draft block width {spec_width} != {width}"
        )
    bs = int(forward_batch.batch_size)
    rows = bs * width
    if int(input_ids.numel()) != rows or int(positions.numel()) != rows:
        raise ValueError(
            f"DeepSeek-V4.1 DSpark draft block of {bs} requests x {width}: "
            f"input_ids {int(input_ids.numel())}, positions {int(positions.numel())}"
        )
    ids = input_ids.view(bs, width)
    pos = positions.view(bs, width)
    # The device-reading checks sync, so never under a graph capture or its
    # warmups (SGLang's capture batch has zero positions and slots anyway);
    # a replayed block is checked by the backend before the replay.
    if v41_dspark_checks_enabled() and not _in_graph_capture():
        # The draft block's ForwardBatch carries no req_pool_indices_cpu, so
        # this reads the device indices back: bring-up only.
        slots_host = _cpu_ints(
            getattr(forward_batch, "req_pool_indices_cpu", None),
            forward_batch.req_pool_indices,
            bs,
        )
        if np.unique(slots_host).size != bs or np.any(slots_host <= 0):
            raise ValueError(
                "DeepSeek-V4.1 DSpark draft needs one live req_pool_idx per "
                f"request; got {slots_host}"
            )
        seq = forward_batch.seq_lens[:bs].to(device=pos.device, dtype=pos.dtype)
        expected = seq[:, None] + torch.arange(
            width, device=pos.device, dtype=pos.dtype
        )
        if not torch.equal(pos, expected):
            raise ValueError(
                "DeepSeek-V4.1 DSpark draft positions are not seq_lens + "
                f"0..{width - 1}"
            )
        if bool((pos[:, 0] < 1).any()):
            raise ValueError("DeepSeek-V4.1 DSpark draft block without a prefix")
        if width > 1 and not bool((ids[:, 1:] == int(noise_token_id)).all()):
            raise ValueError(
                "DeepSeek-V4.1 DSpark draft block columns 1.. are not the "
                f"noise token {noise_token_id}"
            )
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


def _in_graph_capture() -> bool:
    """SGLang's capture mode (both warmups and the capture) or a stream
    capture in progress."""
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return True
    try:
        from sglang.srt.model_executor.runner import get_is_capture_mode
    except ImportError:  # older layouts
        try:
            from sglang.srt.model_executor.cuda_graph_runner import (
                get_is_capture_mode,
            )
        except ImportError:
            return False
    return bool(get_is_capture_mode())


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

import abc
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from aiter.dist.parallel_state import get_pp_group
from torch import nn

from atom.config import Config
from atom.model_loader.loader import load_model
from atom.spec_decode.draft_graph import DraftGraph
from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname
from atom.utils.forward_context import (
    DPMetadata,
    SpecDecodeMetadata,
    get_forward_context,
    set_forward_context,
)

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class AuxCaptureSpec:
    """Declarative spec for drafter-owned target aux-hidden-state capture.

    A drafter declares WHICH target decoder layers to tap and HOW to turn each
    tapped layer's forward output into the ``[N, hidden_size]`` aux tensor it
    consumes; the base ``Drafter`` owns the generic forward-hook + buffer
    machinery. This keeps the target model agnostic — a different drafter can be
    run against the same target with zero model-side changes.
    """

    layer_ids: tuple[int, ...]
    hidden_size: int
    # (layer_output, layer_module) -> [N, hidden_size], or None to skip this call.
    extract: Callable[[Any, nn.Module], torch.Tensor | None]


# Descent bound for the `.model` wrapper chain below — big enough for every
# nesting we ship (UBatchWrapper -> ForCausalLM -> Model), small enough that a
# self-referential `.model` cannot spin.
_MAX_WRAPPER_DEPTH = 8


def _descend_wrappers(target_model: nn.Module, pick):
    """Walk the ``.model`` wrapper chain until ``pick(module)`` returns non-None.

    A fixed two-level walk is not enough: aux capture is armed AFTER the optional
    TBO wrap, so ``target_model`` may be a ``UBatchWrapper`` and the attribute
    sits one level deeper. Same unwrap ``EagleProposer.arm_aux_capture`` does.
    """
    module = getattr(target_model, "language_model", target_model)
    for _ in range(_MAX_WRAPPER_DEPTH):
        found = pick(module)
        if found is not None:
            return found
        inner = getattr(module, "model", None)
        if inner is None or inner is module:
            break
        module = inner
    return None


def _resolve_decoder_layers(target_model: nn.Module) -> nn.Module:
    """Unwrap the target's decoder-layer stack (handles the multimodal wrapper
    and any number of ``.model`` wrappers, e.g. the TBO ``UBatchWrapper``)."""
    layers = _descend_wrappers(target_model, lambda m: getattr(m, "layers", None))
    if layers is None:
        raise ValueError(
            "could not resolve the target decoder layers for aux capture "
            f"(no `.layers` within {_MAX_WRAPPER_DEPTH} `.model` levels of "
            f"{type(target_model).__name__})."
        )
    return layers


def _resolve_embedding(target_model: nn.Module) -> nn.Module:
    """Unwrap the target's token-embedding module (for the ``-1`` aux tap).

    The reference convention (deepspec ``extract_context_feature``) uses layer id
    ``-1`` to mean the embedding output (``hidden_states[0]``). Model wrappers name
    it either ``embed_tokens`` (standard) or ``embed`` (DeepSeek-V4)."""

    def _pick(module):
        for name in ("embed_tokens", "embed"):
            embed = getattr(module, name, None)
            if embed is not None:
                return embed
        return None

    embed = _descend_wrappers(target_model, _pick)
    if embed is None:
        raise ValueError(
            "could not resolve the target embedding module for aux capture of "
            "layer id -1 (expected model.embed_tokens or model.embed)."
        )
    return embed


def _identity_extract(output: Any, _module: nn.Module) -> torch.Tensor:
    """Aux extractor for the ``-1`` (embedding) tap: the embedding output is
    already ``[N, hidden]``, so take it verbatim (unwrapping a (tensor, scale)
    tuple if a quantized embedding returns one)."""
    return output[0] if isinstance(output, tuple) else output


# Draft-model architecture registry: maps a draft checkpoint's architecture
# string to the ATOM wrapper class. Covers all drafter flavors (serial MTP,
# EAGLE3, DSpark), so it lives on the shared base rather than a per-flavor file.
support_draft_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "DeepseekV4MTPModel": "atom.models.deepseek_v4_mtp.DeepseekV4MTP",
    "DeepseekV4DSparkModel": "atom.models.deepseek_v4_dspark.DeepseekV4DSpark",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
    "MiMoV2MTPModel": "atom.models.mimo_v2_mtp.MiMoV2MTP",
    "MiMoV2FlashMTPModel": "atom.models.mimo_v2_mtp.MiMoV2MTP",
    "Qwen3_5MTPModel": "atom.models.qwen3_5_mtp.Qwen3_5MTP",
    "Eagle3LlamaModel": "atom.models.eagle3_llama.Eagle3LlamaModel",
    "Eagle3DeepseekMLAModel": "atom.models.eagle3_deepseek_mla.Eagle3DeepseekMLAModel",
    "K3DSparkModel": "atom.models.kimi_k3_dspark.KimiK3DSpark",
}


class Drafter(abc.ABC):
    """Abstract speculative-decode drafter.

    ModelRunner depends on this contract, not on the concrete flavor
    (serial-MTP / EAGLE3 via ``EagleProposer``, or block-parallel via
    ``DSparkProposer``). The base holds the flavor-neutral scaffolding both
    concretes share — buffer setup, DP-metadata refresh, weight loading /
    sharing, anchor-index preparation, and spec-decode metadata — while the
    flavor hooks (``_resolve_mtp_k``, ``_build_draft_model``) and the hot
    ``propose`` are subclass-supplied.

    The capability properties (``is_block_drafter`` / ``uses_confidence_schedule``
    / ``verify_scheduler``) are the typed replacements for the historical
    ``getattr(drafter, "dspark_*")`` and ``method == "eagle3"`` flavor probes
    ModelRunner used to scatter through its code.
    """

    def __init__(self, atom_config: Config, device: torch.device, runner):
        self.config = atom_config
        self.speculative_config = self.config.speculative_config
        self.runner = runner
        self.device = device
        self.dtype = self.config.torch_dtype
        self.max_num_tokens = self.config.max_num_batched_tokens

        # Flavor-specific verify horizon. MUST be resolved before the buffers
        # below are sized: target_logits_indices is (max_bs * mtp_k) wide.
        self.mtp_k: int = self._resolve_mtp_k()

        # Flavor-specific draft-model construction (plain MTP / eagle3 / DSpark).
        draft_hf = self.speculative_config.draft_model_hf_config
        model_class = resolve_obj_by_qualname(
            support_draft_model_arch_dict[draft_hf.architectures[0]]
        )
        self.model = self._build_draft_model(model_class)

        # Drafter-owned aux capture state (armed by arm_aux_capture).
        self._captures_aux = False
        self._aux_buffers: list[torch.Tensor] = []

        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        max_bs = self.config.max_num_seqs
        self.cu_num_draft_tokens = CpuGpuBuffer(max_bs, **i32_kwargs)
        self.target_logits_indices = CpuGpuBuffer(max_bs * self.mtp_k, **i64_kwargs)
        self.bonus_logits_indices = CpuGpuBuffer(max_bs, **i64_kwargs)

        self._build_draft_graphs()

    # ---- draft passes ----
    def _declare_draft_graphs(self) -> tuple[DraftGraph, ...]:
        """Declare this drafter's warmable forwards. Opt-in.

        A flavor whose model cannot answer the warmup/forward/epilogue must
        declare nothing rather than merely decline to pad: warmup runs before
        the pad and capture gates.
        """
        return ()

    def _build_draft_graphs(self) -> None:
        self.draft_graphs: tuple[DraftGraph, ...] = tuple(
            pass_.bind(self.config, self.device)
            for pass_ in self._declare_draft_graphs()
        )

    @torch.inference_mode()
    def warmup_draft_graphs(self, build_context, stream) -> None:
        """Run every declared pass once per captured size, paying its JIT here.

        aiter's flydsl hgemm builds a kernel per tile config, in-process, so a
        batch first seen mid-serve stalls that step and every restart pays
        again: `hipModuleLoadData` per rank went 6 -> 0 on the 16k/20/50c
        reproducer, with this and `propose`'s padding.

        Warming `runner.capture_sizes` is the point -- `propose` runs at the batch
        the target ran, which `ForwardMode.decide` picks out of that same list,
        so warmed and reachable are one set by construction rather than two
        that drift.

        `build_context(bs=...)` synthesizes one decode batch. The runner passes
        it already bound, because which backends take a `max_q_len` is the
        runner's to know. Call INSIDE `graph_capture()` (it arms the custom
        all-reduce for the optional capture) and after `allocate_kv_cache`, so
        that context has real ring slots and `is_dummy_run=False`.

        Whatever `propose` sets on the context, this must set too. A capture
        bakes every Python branch taken while it is made, so a guard that reads
        the context is decided once, here, for every replay -- `is_draft` gates
        the aux-capture hook away from the draft's own forward, and warming
        without it records that hook clobbering the buffer it protects. The
        row counts are the same rule and the sharper case: they size the MoE
        all_gather that goes INTO the recording, so a synthetic context left
        describing the target bakes a collective the pass never runs at.
        """
        if not self.draft_graphs:
            return
        runner = self.runner
        capture_sizes = sorted(runner.capture_sizes)  # capture leaves it descending
        pool = runner.graph_pool
        start = time.time()
        for pass_ in self.draft_graphs:
            for bs in capture_sizes:
                attn_metadata, context = build_context(bs=bs)
                context.is_draft = True
                # Every declared pass uses the synthetic decode context from
                # the target capture builder. Stated, not inherited, per the
                # rule above.
                context.is_prefill = False
                context.running_tokens_are_unified = True
                # The synthetic batch is full, so the pass's scheduled and
                # running counts are the same number. Width belongs to the
                # pass: Eagle step 0 is full_q rows/seq while its mid-step is 1.
                local_tokens = bs * pass_.tokens_per_seq
                context.scheduled_tokens = local_tokens
                context.running_tokens = local_tokens
                set_forward_context(
                    attn_metadata=attn_metadata,
                    atom_config=self.config,
                    context=context,
                    num_tokens=local_tokens,
                )
                pool = pass_.warmup(bs, pool=pool, stream=stream)
        runner.graph_pool = pool
        torch.cuda.synchronize()
        if runner.rank == 0:
            logger.info(
                "Draft passes warmed at %s in %.2fs: %s",
                capture_sizes,
                time.time() - start,
                [g.name for g in self.draft_graphs],
            )

    # ---- flavor hooks ----
    @property
    def draft_tokens_per_seq(self) -> int:
        """The flavor's default draft width per sequence.

        1 for a serial flavor; a block flavor drafts its whole width in one
        pass and overrides. A declared graph copies this into its own
        ``tokens_per_seq`` because one flavor may declare passes of different
        widths (Eagle step 0 and its mid-step).
        """
        return 1

    @abc.abstractmethod
    def _resolve_mtp_k(self) -> int:
        """Return the per-request verify horizon (drafted tokens per step).

        Called before buffer allocation in ``__init__``; a subclass may also
        stash flavor-specific config (e.g. DSpark's block size) as a side
        effect here so the value is available when buffers are sized.
        """

    def _build_draft_model(self, model_class) -> nn.Module:
        """Construct the draft ``nn.Module``. Default builds a plain draft from
        the target config; ``EagleProposer`` overrides for the eagle3 arch."""
        return model_class(self.config)

    def _aux_capture_spec(self, target_model: nn.Module) -> AuxCaptureSpec | None:
        """Declare which target layers to tap for aux hidden states and how to
        extract them. Default: no capture. Hook-based drafters (DSpark) override
        this; the base ``arm_aux_capture`` turns the spec into forward hooks."""
        return None

    @abc.abstractmethod
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        num_reject_tokens: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Draft this step's speculative tokens for every request in the batch.

        Aux hidden states (EAGLE3/DSpark) are drafter-owned: a concrete reads
        them from its own buffers via ``aux_for`` — ModelRunner passes none.
        """

    # ---- capability surface (typed replacements for flavor probes) ----
    @property
    def is_block_drafter(self) -> bool:
        """True for block-parallel drafters (DSpark); False for serial MTP/EAGLE3."""
        return False

    @property
    def uses_confidence_schedule(self) -> bool:
        """True when the drafter drives variable-length (confidence) verification."""
        return False

    @property
    def verify_scheduler(self):
        """Confidence-schedule verify planner (DSpark Level B), or None when the
        drafter uses fixed-length verification."""
        return None

    @property
    def draft_kv_duplicates_propose(self) -> bool:
        """Is `compute_draft_kv` the pass propose's first draft step redoes?

        True (EAGLE) means the two must never both run: the second is a
        collective the peers on `dummy_execution` do not mirror. False (DSpark)
        means it writes storage propose only reads, so it always runs.
        """
        return False

    # ---- aux-hidden-state ownership (drafter-owned, hook-based) ----
    def arm_aux_capture(self, target_model: nn.Module) -> None:
        """Install drafter-owned forward hooks on the target's decoder layers per
        this drafter's ``_aux_capture_spec``, writing each tapped layer's aux
        tensor into a preallocated buffer in-place (cudagraph-safe). Called once
        after ``load_model``. No-op when the drafter declares no spec."""
        spec = self._aux_capture_spec(target_model)
        if spec is None:
            return
        self._aux_buffers = [
            torch.zeros(
                self.max_num_tokens,
                spec.hidden_size,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in spec.layer_ids
        ]
        layers = _resolve_decoder_layers(target_model)
        n_layers = len(layers)
        # Reference convention (deepspec validate_target_layer_ids): ids must be
        # strictly increasing and each in {-1} U [0, n_layers). Each id resolves to
        # a (module, extract) tap funneled through the SAME hook: -1 taps the
        # embedding output (hidden_states[0], taken verbatim); k taps decoder
        # layer k's output through the drafter's mHC extract.
        prev = None
        for buf_idx, lid in enumerate(spec.layer_ids):
            if not (lid == -1 or 0 <= lid < n_layers):
                raise ValueError(
                    f"aux capture layer id {lid} out of range "
                    f"{{-1}} U [0,{n_layers})."
                )
            if prev is not None and lid <= prev:
                raise ValueError(
                    "aux capture layer ids must be strictly increasing, got "
                    f"{spec.layer_ids}."
                )
            prev = lid
            module, extract = (
                (_resolve_embedding(target_model), _identity_extract)
                if lid == -1
                else (layers[lid], spec.extract)
            )
            module.register_forward_hook(self._make_aux_hook(buf_idx, extract))
        self._captures_aux = True
        logger.info(
            f"{type(self).__name__} aux capture on target layers: {spec.layer_ids}"
        )

    def _make_aux_hook(self, buf_idx: int, extract):
        buffer = self._aux_buffers[buf_idx]

        def _hook(module, _inputs, output):
            ctx = get_forward_context().context
            # Target forwards only. The `-1` tap sits on the embedding, which the
            # draft model shares — its forward_spec would otherwise clobber these
            # rows with noise-token embeddings.
            if ctx.is_draft:
                return
            tensor = extract(output, module)
            if tensor is None:
                return
            # In-place write into the fixed buffer (cudagraph-safe). Offset is 0
            # except under TBO, where this hook fires once per micro-batch on a
            # disjoint token slice — writing at row 0 unconditionally would make
            # the ubatches overwrite each other.
            off = ctx.ubatch_token_offset
            buffer[off : off + tensor.shape[0]].copy_(tensor)

        return _hook

    def aux_for(self, target_hidden_states: torch.Tensor) -> list[torch.Tensor] | None:
        """The target aux hidden states captured during the last forward,
        row-aligned to ``target_hidden_states`` (the tensor they accompany), or
        None if this drafter captures no aux. The fixed capture buffers are
        sliced to ``target_hidden_states``' row count — the caller never juggles
        token counts."""
        if not self._captures_aux:
            return None
        n = target_hidden_states.shape[0]
        return [buf[:n] for buf in self._aux_buffers]

    def anchors_to_gpu(self, anchors: list[int]) -> torch.Tensor:
        """Scheduler-supplied anchors as an int32 GPU tensor, without blocking.

        `-1` entries survive as-is; callers read them as "sampling supplies it".
        Uses ``forward_vars`` so the PP ring clones it per in-flight slot.
        """
        n = len(anchors)
        buf = self.runner.forward_vars["draft_next_tokens"]
        buf.np[:n] = anchors
        return buf.copy_to_gpu(n)

    def compute_draft_kv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: list[int] | None,
    ) -> None:
        """Absorb one target forward into whatever context this drafter keeps.

        Called after EVERY target forward, including the ones that sample
        nothing. `propose()` is reached only from a forward that samples, so a
        context maintained there covers a chunked prefill's final chunk alone.

        `next_token_ids` is the token one position past this forward, per seq:
        -1 where sampling supplies it, None on unlabelled batches.
        """
        return

    # ---- shared machinery ----
    @staticmethod
    def _share_if_not_loaded(
        owner: nn.Module,
        attr: str,
        source: nn.Module,
        loaded: set[str],
        param_key: str,
        label: str,
    ):
        """Replace *owner.attr* with *source* if the weight was not loaded."""
        if param_key not in loaded and getattr(owner, attr, None) is not None:
            logger.info(
                f"MTP {label} not loaded from checkpoint, "
                "sharing from the target model."
            )
            delattr(owner, attr)
            setattr(owner, attr, source)

    def load_model(self, target_model: nn.Module) -> None:
        # Three drafter flavors cover three of the four combinations:
        #   eagle3          standalone ckpt, independent embed + lm_head
        #   K3 DSpark       standalone ckpt, SHARED embed + lm_head (Kimi-K3:
        #                   the checkpoint ships neither)
        #   MTP / V4 DSpark weights inside the target ckpt, shared embed/lm_head
        # so they are decided separately below rather than by one method probe.
        spec = self.speculative_config
        standalone_ckpt = spec.method == "eagle3" or spec.use_dspark_with_draft()
        loaded = load_model(
            self.model,
            spec.model if standalone_ckpt else self.config.model,
            spec.draft_model_hf_config,
            self.config.load_dummy,
            not standalone_ckpt,
        )

        if spec.method == "eagle3":
            logger.info(
                "Eagle3 draft model loaded from %s (independent embed/lm_head)",
                spec.model,
            )
            return

        # Resolve the base model (unwrap multimodal wrapper if present)
        target_base = getattr(target_model, "language_model", target_model)

        # Model-specific share hook escape valve. Models whose embed/lm_head
        # naming doesn't match the standard `model.embed_tokens` /
        # `lm_head` convention (e.g. DeepSeek-V4 uses `model.embed` /
        # `model.head`) implement `share_with_target(target_base)` to do
        # their own setattr-rebinding and short-circuit the default path.
        if hasattr(self.model, "share_with_target"):
            self.model.share_with_target(target_base, loaded)
            if self.is_block_drafter and hasattr(self.model, "reset_kv_cache"):
                # Allocate DSpark's private rolling target-KV window now that the
                # device/dtype and max concurrency are known.
                self.model.reset_kv_cache(
                    self.config.max_num_seqs, self.device, self.dtype
                )
            return

        # Share embed_tokens with the target model. Match on the *logical* vocab
        # (num_embeddings) and hidden dim rather than the stored weight shape, so a
        # replicated target embed ([vocab, hidden], ATOM_REPLICATE_VOCAB_EMBED) is
        # still shared onto a TP-sharded draft embed ([vocab/tp, hidden]) — the
        # draft then reuses the target's replicated table (no post-embed
        # all-reduce). When both are sharded this is identical to the old check.
        draft_embed = self.model.model.embed_tokens
        target_embed = target_base.model.embed_tokens
        draft_vocab = getattr(draft_embed, "num_embeddings", None)
        target_vocab = getattr(target_embed, "num_embeddings", None)
        if (
            get_pp_group().world_size == 1
            and draft_vocab is not None
            and draft_vocab == target_vocab
            and draft_embed.weight.shape[1] == target_embed.weight.shape[1]
        ):
            logger.info(
                "Assuming the EAGLE head shares the same vocab embedding"
                " with the target model."
            )
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_base.model.embed_tokens

        # Share lm_head from target if not loaded from checkpoint.
        # Case 1: per-layer shared_head.head (DeepSeek MTP)
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
            # ModuleDict uses string keys (actual layer indices like "61"),
            # ModuleList uses integer indices.
            layer_items = (
                layers.items() if hasattr(layers, "items") else enumerate(layers)
            )
            for key, layer in layer_items:
                if hasattr(layer, "shared_head"):
                    self._share_if_not_loaded(
                        layer.shared_head,
                        "head",
                        target_base.lm_head,
                        loaded,
                        f"model.layers.{key}.shared_head.head.weight",
                        "shared_head.head",
                    )
        # Case 2: top-level lm_head (Qwen3.5 / Qwen3-Next MTP)
        self._share_if_not_loaded(
            self.model,
            "lm_head",
            target_base.lm_head,
            loaded,
            "lm_head.weight",
            "lm_head",
        )

    def _publish_draft_shape(
        self,
        forward_context,
        scheduled_tokens: int,
        running_tokens: int,
        *,
        running_tokens_are_unified: bool,
    ) -> None:
        """Re-point the forward context at the pass about to run.

        A draft reuses the target's context, whose counts are the verified
        tokens -- not the draft's own width. Anything sizing a collective off
        the context (MoE's `pad_for_all_gather` above all) would then use the
        wrong height, so the draft states its shape here rather than letting
        each consumer special-case `is_draft`.

        Both units: `scheduled_tokens` is how many rows carry a real request,
        `running_tokens` how many the pass runs -- they differ exactly when the
        batch was widened.

        Both flags below are the caller's to answer, because this seam cannot
        see either. `running_tokens_are_unified` claims every OTHER rank runs
        this height too -- true where it is `context.running_bs` (`decide`'s
        reduction) times a config width, false where it came off this rank's
        own batch or token stream. `is_prefill` is left alone for the same
        reason: a pass on its own `[bs, T]` shape must clear it, one carrying
        the target's stream must not.
        """
        context = forward_context.context
        context.scheduled_tokens = scheduled_tokens
        context.running_tokens = running_tokens
        parallel_config = self.config.parallel_config
        # A group of one is uniform whatever it runs; only the table needs peers.
        if parallel_config.data_parallel_size <= 1:
            return
        context.running_tokens_are_unified = running_tokens_are_unified
        # The answer travels, not the table it implies -- `make` owns both ways
        # of reaching one, and skipping its all_reduce is why this is worth
        # stating at all.
        forward_context.dp_metadata = DPMetadata.make(
            parallel_config,
            running_tokens,
            unified=running_tokens_are_unified,
        )

    def prepare_inputs(
        self,
        scheduled_bs: int,
        # [scheduled_bs] each request's anchor offset WITHIN its own segment;
        # for a verified request that is its accepted-draft count. None ->
        # every segment's last row, what a step that verified nothing wants.
        anchor_in_seq: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Anchor row per request = its segment start + `anchor_in_seq`.

        Reading forward from the start is what keeps this length-free; counting
        back from the segment end needs the segment length, which DSpark's
        ragged verify makes a per-request number.
        """
        cu_seqlens_q = get_forward_context().attn_metadata.cu_seqlens_q
        cu_seqlens_q = cu_seqlens_q[: scheduled_bs + 1]

        if anchor_in_seq is None:
            anchor_in_seq = cu_seqlens_q[1:] - cu_seqlens_q[:-1] - 1
        token_indices = cu_seqlens_q[:-1] + anchor_in_seq

        # Defensive clamp to the valid flat-token range [0, total_tokens-1].
        # Under DSpark flat-ragged CUDA graph, the drain-phase corner (tiny /
        # mixed batches) can drive an anchor index just out of range; the anchor
        # only seeds the DRAFT (a wrong anchor lowers acceptance but never
        # corrupts the verified/target output — losslessness is preserved), so
        # clamping is safe and avoids an index_select GPU fault. No-op on the
        # normal path where indices are already in range.
        if self.is_block_drafter:
            upper = (cu_seqlens_q[-1] - 1).clamp_(min=0)
            token_indices = token_indices.clamp_(min=0)
            torch.minimum(token_indices, upper, out=token_indices)

        return token_indices

    def calc_spec_decode_metadata(
        self,
        num_sampled_tokens: np.ndarray,
        cu_num_sampled_tokens: np.ndarray,
        input_ids: torch.Tensor,
    ) -> SpecDecodeMetadata:
        scheduled_bs = len(num_sampled_tokens)

        # num_draft = num_sampled - 1 per request. num_sampled_tokens is the
        # per-seq token count for THIS forward (anchor + drafts). In Phase 1 that
        # is mtp_k+1 (full). With DSpark plan Y the q-bucket already shrank it to
        # q (uniform), so deriving num_draft from num_sampled keeps draft / target
        # / bonus indices consistent with the actual forward layout — no separate
        # mtp_k constant that could desync (the A-bug: 98%->52%).
        num_draft_tokens = np.asarray(num_sampled_tokens, dtype=np.int32) - 1
        np.clip(num_draft_tokens, 0, self.mtp_k, out=num_draft_tokens)
        sum_drafted_tokens = int(num_draft_tokens.sum())

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self.runner._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # Do the CPU -> GPU copy.
        self.target_logits_indices.np[:sum_drafted_tokens] = target_logits_indices
        self.cu_num_draft_tokens.np[:scheduled_bs] = cu_num_draft_tokens
        self.bonus_logits_indices.np[:scheduled_bs] = bonus_logits_indices
        target_logits_indices = self.target_logits_indices.copy_to_gpu(
            sum_drafted_tokens
        )
        cu_num_draft_tokens = self.cu_num_draft_tokens.copy_to_gpu(scheduled_bs)
        bonus_logits_indices = self.bonus_logits_indices.copy_to_gpu(scheduled_bs)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = torch.index_select(input_ids[1:], 0, target_logits_indices)

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_spec_steps=self.mtp_k,
            num_draft_tokens_np=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
        )
        return metadata

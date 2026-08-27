import logging

import torch
from torch import nn
from torch.profiler import record_function

from atom.distributed.dcp_utils import get_dcp_rank, get_dcp_world_size
from atom.spec_decode.draft_graph import DraftGraph, StagedInput
from atom.spec_decode.drafter import AuxCaptureSpec, Drafter
from atom.spec_decode.dspark_verify import VerifyScheduler
from atom.utils import envs
from atom.utils.block_convert import kv_indices_generate_triton
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


class DSparkProposer(Drafter):
    """DSpark block-parallel drafter (sibling of ``EagleProposer``).

    Unlike the serial Eagle/MTP loop (the draft model run ``mtp_k`` times),
    DSpark generates the whole block in a single ``forward_spec`` backbone
    pass; the sequential dependency lives in the lightweight Markov head. The
    verify length defaults to the checkpoint's ``dspark_block_size`` and may be
    driven by a confidence schedule (variable-length, Level B) verification.
    """

    def __init__(self, atom_config, device: torch.device, runner):
        super().__init__(atom_config, device, runner)
        # Confidence-scheduled verification (Level B, variable-length verify) is
        # DSpark-only. The ell (per-request verify length) machinery lives in a
        # reusable VerifyScheduler; propose() feeds it the confidence head and
        # the next step's calc_spec_decode_metadata consumes the ell map.
        # Private on purpose: the public surface is the base's
        # `uses_confidence_schedule`. A public `dspark_*` attribute is what let
        # `getattr(drafter, "dspark_*", False)` probes keep silently working.
        self._confidence_schedule = bool(self.config.dspark.confidence_schedule)
        self._verify_scheduler = (
            VerifyScheduler(runner) if self._confidence_schedule else None
        )
        # The draft shares the target's block tables and its KV lives in the same
        # paged pool, so it inherits the pool's DCP sharding: the block pass must
        # address and size itself in LOCAL (per-rank) terms. 1 when -dcp is unset.
        self.dcp_world_size = get_dcp_world_size()
        self.dcp_rank = get_dcp_rank()
        if self._with_draft:
            self._init_draft_block_buffers()

    def _declare_draft_graphs(self):
        """The block pass.

        ``block`` is the only pure pass either drafter has: it gathers the
        rolling window and writes nothing back. The KV write is a separate call
        (``compute_draft_kv``), which is exactly why the block can be padded
        at all -- purity is a property of what a pass does, not of DSpark.

        That KV write is per-token, not per-sequence, and is NOT declared:
        warming it needs a prefill-shaped synthetic forward context, which the
        capture builder cannot produce. Declaring it with a stub forward would
        create a pass that claims to be warmable and is not.

        The separate-draft flavor declares nothing for the same reason: it drafts
        through ``_propose_with_draft``, and the warmup/forward/epilogue below
        reach for ``model.window_size`` and ``model.model.head_and_sample``,
        which its checkpoint does not carry. Declining to pad is not enough --
        warmup runs before that gate.

        No ``mtp_k`` floor, unlike eagle's: the block drafts its whole width in
        ONE pass, so there is no step 1+ to be absent at ``mtp_k == 1``.
        """
        self.block = None
        if self._with_draft:
            return ()
        self.block = DraftGraph(
            forward=self._block_backbone,
            epilogue=self._block_head,
            capture_epilogue=True,
            inputs={
                "anchor_ids": StagedInput(dtype=torch.int32),
                "anchor_positions": StagedInput(dtype=torch.int64),
            },
            pads=self._block_may_pad,
            warmup_inputs=self._block_warmup_inputs,
        )
        return (self.block,)

    def _block_warmup_inputs(self, running_bs, *, anchor_positions, **_):
        """A plausible warmup batch: anchors past the window, real ring slots.

        Anchors at position ``window`` leave every window slot valid, which is
        what a steady-state decode draws. Warming at position 0 would mask all
        but the last slot and compile a shape serving never asks for.
        """
        fc = get_forward_context()
        assert not fc.context.is_dummy_run, (
            "warmup needs a real forward context; a dummy one bakes the "
            "all-zero rolling window and only shows up as lost acceptance"
        )
        anchor_positions.fill_(int(self.model.window_size))

    @property
    def _block_may_pad(self) -> bool:
        """Whether the block may take fabricated rows.

        Off under EPLB: pad rows carry fabricated tokens through the draft's full
        MoE, and ``select_experts_with_record`` counts every row it routes into
        the expert-load histogram that drives online expert migration.
        """
        return not self.config.eplb_enable

    def _block_head(self, out, running_bs, *, anchor_ids, **_):
        """The block's epilogue: LM head, then the sequential Markov sampler.

        Nothing here resists capture -- the sampler is a fixed-trip loop over
        the draft width, and the LM head's one data-dependent step (its
        prefill last-token slice) is already suppressed for a draft. Under TP
        it does all_gather the vocab shard; ``capture_epilogue`` says whether
        that one collective is captured with the rest.

        It must be WARMED regardless: the head has its own per-shape flydsl
        builder, and leaving it out of the warm is exactly how
        `hipModuleLoadData` went 0 -> 4 on the reproducer once.
        """
        normed, hc_hidden = out
        return self.model.model.head_and_sample(normed, hc_hidden, anchor_ids)

    def _block_backbone(self, running_bs, *, anchor_ids, anchor_positions):
        """The block's forward: the parallel backbone over the whole draft width.

        Nothing of the target's metadata is installed. The ring slots the model
        reads off the forward context are already this length: `prepare_decode`
        publishes them at the padded batch, which is the batch this runs at.
        """
        return self.model.model(anchor_ids, anchor_positions, self.draft_tokens_per_seq)

    def _init_draft_block_buffers(self) -> None:
        """Preallocate the block-pass metadata the separate-draft path rebinds."""
        max_bs = self.config.max_num_seqs
        t = self.mtp_k
        i64 = {"dtype": torch.int64, "device": self.device}
        i32 = {"dtype": torch.int32, "device": self.device}
        # Block absolute positions, [max_bs, T]; passed into forward_spec.
        self._blk_positions = torch.zeros(max_bs, t, **i64)
        # Flat slot mapping for the block rows, [max_bs * T].
        self._blk_slots = torch.zeros(max_bs * t, **i64)
        # Per-request KV length = anchor + 1 + T.
        self._blk_ctx_lens = torch.zeros(max_bs, **i32)
        # Every request contributes exactly T query rows, so cu_seqlens_q is a
        # constant ramp — build it once and only ever slice it.
        self._blk_cu_seqlens_q = torch.arange(0, (max_bs + 1) * t, step=t, **i32)
        # Constant 1..T ramp used to expand anchors into block positions.
        self._blk_offsets = torch.arange(1, t + 1, **i64)
        self._blk_last_page_lens = torch.ones(max_bs, **i32)
        self._blk_kv_indptr = torch.zeros(max_bs + 1, **i32)
        # kv_indptr[-1] = sum(ctx_lens) = sum(anchor + 1 + T). An anchor can sit
        # at max_model_len - 1, so each request contributes up to
        # max_model_len + T entries -- pad by max_bs * T so the unchecked
        # kv_indices_generate_triton write can never run past the buffer.
        self._blk_kv_indices = torch.zeros(
            max_bs * (self.config.max_model_len + t), **i32
        )

        self._blk_dtype_q = None

        # Persistent (ps=1) MLA-decode metadata for the block pass.
        self._blk_ps_bufs = None

    def _init_block_persistent_buffers(self, dtype_q, dtype_kv):
        """Allocate the block-pass persistent MLA metadata buffers once.

        Mirrors MLAAttentionBackend's constructor (aiter_mla.py:150-197): the
        buffer sizes come from get_mla_metadata_info_v1 for this draft's head
        count / dtypes / block width, and get_mla_metadata_v1 fills them in
        place each step."""
        if self._blk_ps_bufs is not None:
            return self._blk_ps_bufs
        from atom.model_ops.attentions.aiter_mla import (
            _MLA_META_SUPPORTS_MAX_SPLIT,
            _MLA_SPLIT_BUDGET_AUTO,
            get_mla_metadata_info_v1,
        )

        max_bs = self.config.max_num_seqs
        # padded_num_heads lives on the draft MLA module (see
        # mla_min_query_heads), matching the gqa ratio the asm kernel dispatches
        # on -- read it rather than recomputing, so the work descriptors planned
        # here describe the kernel that will actually run. The ModelRunner itself
        # has no such attribute.
        impl = self.model.layers[0].self_attn.mla_attn.impl
        if self.dcp_world_size > 1:
            # DCP decode all-gathers on the head dim, so the descriptors must be
            # planned for the padded GATHERED width.
            self._blk_padded_heads = impl.dcp_kernel_num_heads
        else:
            self._blk_padded_heads = impl.padded_num_heads
        # max_split_per_batch only exists in newer aiter builds; feature-detect
        # it (as aiter_mla does) so old builds don't hit a TypeError. Cache the
        # kwargs so the sizing (info) and fill (get_mla_metadata_v1) calls agree.
        # The block pass is a bs=1, non-causal (msk0) decode over the FULL target
        # context, so a hardcoded 16 pins it to 16 of the machine's clusters and
        # starves it at long ctx (~392us at 256k). Take the budget from the
        # machine like the rest of the decode path (_MLA_SPLIT_BUDGET_AUTO=-1).
        self._blk_split_kwargs = (
            {"max_split_per_batch": _MLA_SPLIT_BUDGET_AUTO}
            if _MLA_META_SUPPORTS_MAX_SPLIT
            else {}
        )
        (
            (wmd_sz, wmd_ty),
            (wip_sz, wip_ty),
            (wis_sz, wis_ty),
            (rip_sz, rip_ty),
            (rfm_sz, rfm_ty),
            (rpm_sz, rpm_ty),
        ) = get_mla_metadata_info_v1(
            max_bs,
            self.mtp_k,  # max_seqlen_qo = block width T
            self._blk_padded_heads,
            dtype_q,
            dtype_kv,
            is_sparse=False,
            fast_mode=True,
            **self._blk_split_kwargs,
        )
        dev = self.device
        self._blk_ps_bufs = {
            "work_meta_data": torch.empty(wmd_sz, dtype=wmd_ty, device=dev),
            "work_indptr": torch.empty(wip_sz, dtype=wip_ty, device=dev),
            "work_info_set": torch.empty(wis_sz, dtype=wis_ty, device=dev),
            "reduce_indptr": torch.empty(rip_sz, dtype=rip_ty, device=dev),
            "reduce_final_map": torch.empty(rfm_sz, dtype=rfm_ty, device=dev),
            "reduce_partial_map": torch.empty(rpm_sz, dtype=rpm_ty, device=dev),
        }
        return self._blk_ps_bufs

    @property
    def _with_draft(self) -> bool:
        """DSpark given a separate --draft-model, vs the V4 draft that ships
        inside the target checkpoint.

        The two agree on everything the block algorithm cares about -- block
        width, Markov sampling, confidence, verification -- and differ only in
        where the draft weights come from and how the target context reaches
        them (paged dual-source KV vs a private rolling window).
        """
        return self.speculative_config.use_dspark_with_draft()

    def _build_draft_model(self, model_class) -> nn.Module:
        if not self._with_draft:
            # V4: the draft is part of the target checkpoint and shares its
            # config wholesale, so it inherits the target's compilation level
            # and its `_DSparkInner` is compiled (see deepseek_v4_dspark.py).
            model = model_class(self.config)
            if envs.ATOM_DSPARK_DISABLE_COMPILE:
                # Flip the decorator's own bypass rather than handing the draft a
                # cloned config with NO_COMPILATION (what the with-draft branch
                # below does). A cloned compilation_config would no longer be the
                # object get_current_atom_config() returns, splitting the shared
                # static_forward_context registry. This flag is read at the top of
                # the decorator's __call__ (decorators.py:505), so it degrades to
                # a plain self.forward(...) with no other side effects.
                model.model.do_not_compile = True
                logger.info("DSpark draft: torch.compile disabled by env.")
            return model

        # Standalone draft: build from the DRAFT's own hf_config, exactly as
        # EagleProposer does for eagle3. Shallow-copy rather than deepcopy --
        # atom_config can hold non-picklable cuda.Stream objects, and only
        # hf_config / compilation_config are mutated here.
        import copy

        from atom.config import CompilationLevel

        draft_hf = self.speculative_config.draft_model_hf_config
        draft_atom_config = copy.copy(self.config)
        draft_atom_config.hf_config = draft_hf
        draft_atom_config.compilation_config = copy.copy(self.config.compilation_config)
        draft_atom_config.compilation_config.level = CompilationLevel.NO_COMPILATION
        model = model_class(
            draft_atom_config,
            layer_offset=self.config.hf_config.num_hidden_layers,
        )
        # An MLA draft stores the same 576-wide latent (kv_lora_rank 512 +
        # qk_rope_head_dim 64) as an MLA target's own layers, so it binds into
        # the TARGET's pool as extra rows -- the target builder already sizes
        # and addresses them (see `_num_cache_rows` / `build_kv_cache_tensor`),
        # and the draft inherits `--kv_cache_dtype` for free that way.
        #
        # An MHA draft has no such row to borrow and needs the sibling pool.
        # Same fork, same spelling as EagleProposer.
        draft_is_mla = bool(getattr(draft_hf, "kv_lora_rank", None))
        if not draft_is_mla:
            from atom.spec_decode.eagle3_kv_builder import Eagle3DraftBuilder

            # ModelRunner keys its draft-pool allocation and per-module binding
            # off the presence of this attribute.
            self.runner.eagle3_draft_builder = Eagle3DraftBuilder(self.runner, draft_hf)
        return model

    @property
    def draft_tokens_per_seq(self) -> int:
        """The whole block, in one pass -- capped at the rolling window so the
        `[window ++ draft]` KV the block attends to stays bounded."""
        return min(self.mtp_k, int(self.model.window_size))

    def _resolve_mtp_k(self) -> int:
        draft_cfg = self.speculative_config.draft_model_hf_config
        num_spec = self.speculative_config.num_speculative_tokens
        # V4-Pro-DSpark records its training block width in the config;
        # Kimi-K3-DSpark does not (the draft is width-agnostic in its weights),
        # so there the block IS whatever --num-speculative-tokens says.
        block_size = getattr(draft_cfg, "dspark_block_size", None)
        if not block_size and not num_spec:
            raise ValueError(
                "DSpark needs a draft block width: this draft config carries no "
                "`dspark_block_size`, so pass --num-speculative-tokens "
                "(7 for Kimi-K3-DSpark)."
            )
        self.dspark_block_size = int(block_size or num_spec)
        # num_speculative_tokens may be unset when the config supplies the
        # width; default to the full block (a static verify length == block).
        return num_spec or self.dspark_block_size

    def _resolve_dtype_q(self, forward_context) -> "tuple[torch.dtype, bool]":
        """q_out dtype for the draft's MLA decode, read from its bound cache.

        Returns ``(dtype, final)``; ``final`` is False when the pool is not
        allocated yet, so the caller uses the answer for this step without
        caching it.

        `attention_mla.forward_impl` allocates q_out with this and then hands
        both it and `kv_cache_data[f"layer_{layer_num}"].k_cache` to
        `fused_qk_rope_concat_and_cache_mla`, whose kernel derives the KV dtype
        from the tensor and rejects a bf16 cache paired with an fp8 q_out
        (cache_kernels.cu:4209). Taking q_out's dtype from that same tensor
        makes the pair agree by construction, whatever the tensor turns out
        to be.
        """
        from aiter import dtypes

        # d_dtypes maps the "auto" cache dtype to None, and torch.empty would
        # silently read that as float32 rather than the model dtype.
        from_config = dtypes.d_dtypes.get(self.config.kv_cache_dtype) or self.dtype

        layer_num = self.model.layers[0].self_attn.mla_attn.layer_num
        cache_data = forward_context.kv_cache_data or {}
        entry = cache_data.get(f"layer_{layer_num}")
        bound = getattr(entry, "k_cache", None) if entry is not None else None
        if bound is None or bound.numel() == 0:
            # warmup_model() runs before allocate_kv_cache(), so on that pass
            # there is no pool to read. Answer from the config and return None
            # for `final` so the caller does not cache a warmup-time guess.
            return from_config, False
        if bound.dtype != from_config:
            logger.warning(
                "DSpark draft layer_%d is bound to a %s KV cache, but "
                "--kv_cache_dtype=%s implies %s. Using the bound tensor's dtype "
                "for q_out so the fused write agrees -- but the two should not "
                "be able to differ, since the draft binds into the pool the "
                "engine allocated from that same flag. Check that layer_%d "
                "resolved to a draft row and not to a target layer.",
                layer_num,
                bound.dtype,
                self.config.kv_cache_dtype,
                from_config,
                layer_num,
            )
        return bound.dtype, True

    # ---- Drafter capability surface ----
    @property
    def is_block_drafter(self) -> bool:
        return True

    @property
    def uses_confidence_schedule(self) -> bool:
        return self._confidence_schedule

    @property
    def verify_scheduler(self):
        return self._verify_scheduler

    # ---- aux-hidden-state ownership (declarative; base owns the hook machinery) ----
    def _aux_capture_spec(self, target_model: nn.Module) -> AuxCaptureSpec:
        """DSpark taps the configured target layers and reconstructs each one's
        post-layer hidden state. The base registers the forward hooks."""
        draft_cfg = self.speculative_config.draft_model_hf_config
        layer_ids = tuple(
            int(i) for i in getattr(draft_cfg, "dspark_target_layer_ids", ())
        )
        if not layer_ids:
            raise ValueError(
                "DSpark requires dspark_target_layer_ids on the draft config."
            )
        return AuxCaptureSpec(
            layer_ids=layer_ids,
            hidden_size=self.config.hf_config.hidden_size,
            extract=self._extract_layer_hidden,
        )

    @staticmethod
    def _extract_layer_hidden(output, block: nn.Module):
        """Reconstruct a target layer's post-layer hidden state ``[N, dim]``.

        Every DSpark draft is trained on the reference HF model's
        ``output.hidden_states[layer_id + 1]`` -- the plain residual stream after
        layer ``layer_id``. ATOM's targets do not hand that tensor back directly:
        each optimizes its residual bookkeeping differently, so the layer's
        return value is a different shape per family. Dispatch on that return
        rather than on the drafter flavor -- the reconstruction is a property of
        the TARGET's layer protocol, and a standalone draft could in principle be
        trained against a V4 target or vice versa.

        Returning ``None`` skips the capture for this call (the base hook
        treats it as "nothing to record").
        """
        # A target that bookkeeps its residual stream in a non-obvious way can
        # own the reconstruction itself; preferred, since it then changes in
        # lockstep with that layer's forward(). Kimi-K3 does this.
        own = getattr(block, "aux_hidden_state", None)
        if own is not None:
            return own(output)

        # DeepSeek-V4: an HCState carrying the multi-hidden-connection residual
        # [N, hc, dim]; the aux tensor is its mean over the hc axis.
        if hasattr(output, "residual"):
            residual = output.residual
            if residual is None:
                return None
            x_prev = getattr(output, "x_prev", None)
            post = getattr(output, "post_mix", None)
            comb = getattr(output, "comb_mix", None)
            if x_prev is not None and post is not None and comb is not None:
                residual = block.hc_post(x_prev, residual, post, comb)
            return residual.mean(dim=1)

        # A tuple means the layer carries residual bookkeeping we cannot
        # interpret from here -- which component is the residual stream, and
        # which are deferred addends, is that layer's private convention (K3
        # returns four tensors, three of which must be summed). Guessing would
        # feed the draft a silently wrong aux tensor, so require the target to
        # say. Fail loudly instead.
        if isinstance(output, tuple):
            raise TypeError(
                f"{type(block).__name__}.forward returns a "
                f"{len(output)}-tuple but the layer defines no "
                "`aux_hidden_state(output)`. A drafter cannot reconstruct the "
                "post-layer hidden state from an unknown tuple convention -- "
                "add that method to the layer (see KimiDecoderLayer)."
            )

        # Plain residual stream (no special bookkeeping).
        return output

    def compute_draft_kv(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        next_token_ids: list[int] | None,
    ) -> None:
        """Absorb the target context into the draft's KV, for EVERY DSpark flavor.

        Called once by the runner right after each target forward, while the
        forward context still holds the TARGET's slot mapping / cu_seqlens. Each
        backbone overrides ``model.write_context_kv`` for its own storage:
          * V4 (inline draft): scatter into a private rolling target-KV window.
          * Kimi-K3 (standalone draft): scatter into the paged sibling latent
            pool at the verified tokens' slots.
        Both take the same ``(main_hidden_all, positions)`` contract, so this
        hook stays flavor-agnostic -- the per-request geometry (window span vs
        slot mapping) lives inside the model, read off the live forward context.

        Every scheduled row is written, prefill and decode alike: the read side
        gathers by absolute position without checking what was written, so
        anything left unwritten shows the slot's previous occupant. Rejected
        rows are harmless -- they land on future positions, unread until the
        step that accepts them rewrites them.

        `next_token_ids` is unused: DSpark drafts from aux hidden states.
        """
        del next_token_ids
        aux_hidden_states = self.aux_for(hidden_states)
        if aux_hidden_states is None:
            return
        forward_context = get_forward_context()
        bs = forward_context.context.scheduled_bs
        main_hidden_all = torch.cat(aux_hidden_states, dim=-1)
        with record_function(f"draft_kv[bs={bs} tok={main_hidden_all.shape[0]}]"):
            self.model.write_context_kv(main_hidden_all, positions)

    def propose(
        self,
        # [num_tokens] (unused: DSpark seeds from the verified anchor, not the
        # full target token stream)
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size] (unused: DSpark reads aux_hidden_states)
        target_hidden_states: torch.Tensor,
        # [batch] (unused on this path)
        num_reject_tokens: torch.Tensor,
        next_token_ids: torch.Tensor,  # [batch] verified anchor token x0
        last_token_indices: torch.Tensor,  # [batch] flat index of each anchor row
    ) -> torch.Tensor:
        """DSpark block drafting: ONE parallel backbone pass + Markov sampling.

        Unlike the serial Eagle/MTP path (a python loop running the draft model
        mtp_k times), DSpark generates the whole block in a single forward_spec
        call. The sequential dependency lives inside the lightweight Markov head,
        not in repeated heavyweight backbone passes.

        GPU-VERIFY: this path needs an MI3xx run against the reference DSpark to
        confirm (a) the rolling target-KV window is populated correctly across
        prefix-cache hits, and (b) the sampled block matches the reference.
        """
        forward_context = get_forward_context()
        context = forward_context.context
        attn_metadata = forward_context.attn_metadata
        context.is_draft = True
        bs = context.scheduled_bs

        # Drafter-owned aux: our own forward-hook capture buffers, row-aligned to
        # the target hidden states.
        aux_hidden_states = self.aux_for(target_hidden_states)
        if aux_hidden_states is None:
            raise RuntimeError(
                "DSpark requires target auxiliary hidden states from "
                "dspark_target_layer_ids; none were captured."
            )
        # aux is validated here (drafting requires it) but the target context is
        # already in the draft's KV: `compute_draft_kv` absorbed it right
        # after the target forward, uniformly for every flavor. propose() only
        # needs the anchor to seed the block.

        # Anchor token x0 per request = the just-verified target token, located
        # at last_token_indices in the flat batch.
        # Seatbelt: markov_w1 is a raw nn.Embedding, so a -1 anchor traps it.
        anchor_ids = next_token_ids.clamp(0, int(self.model.vocab_size) - 1)
        anchor_positions = torch.index_select(target_positions, 0, last_token_indices)

        if self._with_draft:
            return self._propose_with_draft(
                forward_context,
                attn_metadata,
                bs,
                anchor_ids,
                anchor_positions,
            )

        # The rolling target-KV window is filled by `compute_draft_kv`,
        # which the runner calls after every target forward.
        #
        # Draft width = the verify horizon mtp_k (num_speculative_tokens). This
        # may exceed dspark_block_size (the training default); the DSpark weights
        # carry no per-width parameters, so the wider block is drafted in one
        # pass with positions past block_size RoPE-extrapolated. Capped at the
        # rolling window so [window ++ draft] KV stays bounded.
        #
        # Width-agnostic in the WEIGHTS, not in the OUTPUT: block attention is
        # bidirectional, so every draft token depends on T. Acceptance rates and
        # confidence calibration are not comparable across K.
        window = int(self.model.window_size)
        num_draft = self.draft_tokens_per_seq
        # forward_spec sizes the block off anchor_ids. context.scheduled_bs counts
        # only one half of a mixed prefill+decode step, so it is not that B.
        real_bs = anchor_ids.shape[0]
        # Agreed first, and on EVERY step. The batch a pass runs at has to be
        # one number for the whole DP group, or half of it replays a recorded
        # collective while the rest issue a differently sized one. Not
        # conditioned on prefill-vs-decode either: which of the two a rank is
        # doing is its own business, so a rank that skipped this would leave
        # the others waiting in the exchange.
        running_bs = self.block.target_running_bs(real_bs, context)
        # The target already replayed a padded graph, but none of that padding
        # reaches here: its pad rows end at the graph boundary. `anchor_ids`
        # comes from the sampler, which runs after the graph and only over real
        # rows. So the block pads its own inputs. The target's `state_slot_out`
        # needs no help, and by an identity rather than an agreement:
        # `prepare_decode` publishes it at the `bs` it was built with, which is
        # `context.running_bs` -- the same number `target_running_bs` just
        # returned.
        staged = self.block.stage(
            running_bs,
            {"anchor_ids": anchor_ids, "anchor_positions": anchor_positions},
        )
        # ...and the fabricated rows must not scatter their draft KV. Their ring
        # slot is the 0 `prepare_decode` fills that tail with, which is a real
        # position, so the write would land in another request's window. Here
        # and not inside the block: `run` may REPLAY, and then nothing in the
        # block's Python runs at all.
        self.model.model.index_buffers(num_draft, window, self.device).mask_pad_tail(
            self.runner.attn_metadata_builder.row_ids, real_bs, running_bs
        )
        self._publish_draft_shape(
            forward_context,
            scheduled_tokens=real_bs * num_draft,
            running_tokens=running_bs * num_draft,
        )
        label = self.block.label(real_bs, running_bs, context)
        with record_function(f"propose_dspark[{label} T={num_draft}]"):
            draft_token_ids, confidence = self.block.run(running_bs, context, **staged)
        draft_token_ids = draft_token_ids[:real_bs, : self.mtp_k]
        if confidence is not None:
            # compute_ell takes its batch size from confidence.shape and zips the
            # resulting ell against batch.req_ids by position.
            confidence = confidence[:real_bs]
        # Confidence-scheduled verification. The hardware-aware prefix scheduler
        # consumes the confidence head to pick a per-request verify length
        # ell_r. We compute ell here and stash it; the actual variable-length
        # verification (Level B) is applied downstream by truncating each
        # request's scheduled spec tokens to ell_r, which frees batch capacity
        # instead of the no-op in-block masking of Level A.
        if self.verify_scheduler is not None and confidence is not None:
            with record_function(f"dspark_sched[bs={real_bs}]"):
                self.verify_scheduler.set_last_ell(
                    self.verify_scheduler.compute_ell(confidence[:, : self.mtp_k])
                )
        elif self.verify_scheduler is not None:
            self.verify_scheduler.set_last_ell(None)
        return draft_token_ids

    # ---- separate-draft-model path (Kimi-K3) --------------------------------

    def _propose_with_draft(
        self,
        forward_context,
        attn_metadata,
        bs: int,
        anchor_ids: torch.Tensor,  # [bs]
        anchor_positions: torch.Tensor,  # [bs]
    ) -> torch.Tensor:
        """Kimi-K3 DSpark: one non-causal block pass over the paged latent cache.

        The target context is already in the draft's latent cache (absorbed by
        `compute_draft_kv`); this only builds the draft block's own metadata
        (addressed by slot mapping) and runs the block. Same shape as the V4
        block pass -- T queries per request against a paged KV cache.
        """
        T = self.mtp_k
        block_size = self.runner.block_size
        # warmup_model() runs at the end of ModelRunner.__init__, BEFORE
        # allocate_kv_cache(), so on a dummy run there is no paged state: the
        # draft's kv_cache is still the empty init tensor and attn_metadata's
        # slot_mapping / block_tables are unset. Everything that touches paged
        # state is skipped below; the block forward still runs, because warmup
        # doubles as the memory-profiling pass and omitting the draft would
        # leave its activations out of the KV budget.
        is_dummy = forward_context.context.is_dummy_run

        # The DSpark draft block is bidirectional: every one of the T draft
        # positions attends the whole block, so the MLA decode runs non-causal.
        # The target rebuilds its own metadata (causal defaults True), so this
        # never leaks back.
        attn_metadata.causal = False

        block_positions = self._blk_positions[:bs]  # [bs, T] view, stable
        torch.add(
            anchor_positions.view(bs, 1),
            self._blk_offsets.view(1, T),
            out=block_positions,
        )

        if not is_dummy:
            block_tables = self.runner.forward_vars["block_tables"].gpu[:bs]
            slots = self._blk_slots[: bs * T].view(bs, T)  # stable
            # Each request's KV spans [0, anchor] (context) ++ the T block rows.
            ctx_lens = self._blk_ctx_lens[:bs]  # stable
            global_lens = anchor_positions + (1 + T)
            if self.dcp_world_size > 1:
                # DCP shards KV token-wise round-robin, so one block table entry
                # covers block_size*dcp_world_size GLOBAL tokens and rank r holds
                # only the positions p with p % dcp_world_size == r, packed
                # densely into its page. Rows this rank does not own are written
                # to -1 (dropped).
                virtual_block = block_size * self.dcp_world_size
                page_idx = torch.div(
                    block_positions, virtual_block, rounding_mode="floor"
                )
                local_off = torch.div(
                    torch.remainder(block_positions, virtual_block),
                    self.dcp_world_size,
                    rounding_mode="floor",
                )
                slots.copy_(
                    torch.where(
                        torch.remainder(block_positions, self.dcp_world_size)
                        == self.dcp_rank,
                        torch.gather(block_tables, 1, page_idx) * block_size
                        + local_off,
                        -1,
                    )
                )
                # Of L global tokens this rank stores ceil((L - r) / dcp_world_size).
                ctx_lens.copy_(
                    torch.div(
                        global_lens + (self.dcp_world_size - 1 - self.dcp_rank),
                        self.dcp_world_size,
                        rounding_mode="floor",
                    )
                )
            else:
                # slot = page_id * block_size + offset_in_page, derived on-device
                # from the block table so there is no host sync.
                page_idx = torch.div(block_positions, block_size, rounding_mode="floor")
                slots.copy_(torch.gather(block_tables, 1, page_idx))
                slots.mul_(block_size)
                slots.add_(torch.remainder(block_positions, block_size))
                ctx_lens.copy_(global_lens)

            attn_metadata.slot_mapping = slots.view(-1)
            attn_metadata.block_tables = block_tables
            attn_metadata.cu_seqlens_q = self._blk_cu_seqlens_q[: bs + 1]
            attn_metadata.context_lens = ctx_lens
            attn_metadata.max_seqlen_q = T
            # Upper bound rather than a .max() host sync: every context_len is
            # at most the target pass's longest sequence plus the block.
            attn_metadata.max_seqlen_k = int(attn_metadata.max_seqlen_k) + T
            kv_indptr = self._blk_kv_indptr[: bs + 1]
            # kv_indptr[0] stays 0 (zero-init, never written). cumsum promotes
            # integers to int64, so land it through copy_ rather than out=.
            kv_indptr[1:].copy_(torch.cumsum(ctx_lens, dim=0))
            # The generator walks block_tables row by row up to this bound, and
            # under DCP those rows are LOCAL: one entry covers
            # block_size*dcp_world_size global tokens. max_seqlen_k stays global
            # (prepare_decode keeps it that way too), so handing it over unscaled
            # would run the generator dcp_world_size times past each row's valid
            # entries and emit slots from unmapped pages. Overestimating is fine
            # -- kv_indptr caps the real per-seq count.
            index_max_k = (
                attn_metadata.max_seqlen_k // self.dcp_world_size + 1
                if self.dcp_world_size > 1
                else attn_metadata.max_seqlen_k
            )
            kv_indices_generate_triton(
                block_tables,
                self._blk_kv_indices,
                kv_indptr,
                block_size,
                index_max_k,
            )
            attn_metadata.kv_indptr = kv_indptr
            attn_metadata.kv_indices = self._blk_kv_indices
            attn_metadata.kv_last_page_lens = self._blk_last_page_lens[:bs]

            # Build persistent (ps=1) MLA-decode metadata
            from atom.model_ops.attentions.aiter_mla import get_mla_metadata_v1

            dtype_q, _ = self._resolve_dtype_q(forward_context)
            dtype_kv = dtype_q
            ps = self._init_block_persistent_buffers(dtype_q, dtype_kv)
            get_mla_metadata_v1(
                attn_metadata.cu_seqlens_q,  # seqlens_qo_indptr
                kv_indptr,  # seqlens_kv_indptr
                attn_metadata.kv_last_page_lens,  # kv_last_page_lens
                self._blk_padded_heads,
                1,  # nhead_kv
                False,  # is_causal (non-causal block)
                ps["work_meta_data"],
                ps["work_info_set"],
                ps["work_indptr"],
                ps["reduce_indptr"],
                ps["reduce_final_map"],
                ps["reduce_partial_map"],
                page_size=block_size,
                kv_granularity=max(block_size, 16),
                max_seqlen_qo=T,
                uni_seqlen_qo=T,
                fast_mode=True,
                dtype_q=dtype_q,
                dtype_kv=dtype_kv,
                **self._blk_split_kwargs,
            )
            attn_metadata.work_meta_data = ps["work_meta_data"]
            attn_metadata.work_indptr = ps["work_indptr"]
            attn_metadata.work_info_set = ps["work_info_set"]
            attn_metadata.reduce_indptr = ps["reduce_indptr"]
            attn_metadata.reduce_final_map = ps["reduce_final_map"]
            attn_metadata.reduce_partial_map = ps["reduce_partial_map"]

        # The separate-draft path pads nothing, so both heights are `bs * T`.
        self._publish_draft_shape(
            forward_context, scheduled_tokens=bs * T, running_tokens=bs * T
        )

        # The block pass is ALWAYS decode-shaped -- T queries per request against
        # a paged KV cache -- even on a step where the target just prefilled. But
        # `is_prefill` is a property of the step, not of the model being run, so
        # on a prefill step it is still True here and every MLA layer would take
        # its prefill branch (attention_mla.py:1238, and again at :1443), which
        # reads cu_seqlens_k / chunk_meta / _gather_cached_kv_b_proj -- none of
        # which the retarget above touches, because none of them describe this
        # batch. Force the decode shape.
        #
        # EagleProposer does the same (eagle_proposer.py:358) but only from its
        # SECOND draft step: its first step deliberately reuses the target's own
        # layout. DSpark has exactly one block pass and it is never that shape,
        # so this is unconditional. Not restored afterwards -- the Context is
        # rebuilt per forward, the same reason `is_draft` above is not restored.
        forward_context.context.is_prefill = False

        dtype_q = self._blk_dtype_q
        if dtype_q is None:
            dtype_q, final = self._resolve_dtype_q(forward_context)
            if final:
                self._blk_dtype_q = dtype_q
        forward_context.attn_metadata.dtype_q = dtype_q

        # ---- 3. Block pass + Markov sampling ---------------------------------
        with record_function(f"propose_dspark[bs={bs} T={T}]"):
            draft_token_ids, confidence = self.model.forward_spec(
                anchor_ids,
                block_positions.view(-1),
                T,
            )

        if self.verify_scheduler is not None:
            self.verify_scheduler.set_last_ell(
                self.verify_scheduler.compute_ell(confidence[:, :T])
                if confidence is not None
                else None
            )
        return draft_token_ids[:, :T]

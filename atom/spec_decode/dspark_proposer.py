import logging

import torch
from torch import nn
from torch.profiler import record_function

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
        if self._with_draft:
            self._init_draft_block_buffers()

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
                "for q_out so the fused write agrees, but the draft's sibling "
                "pool and the requested cache dtype disagree -- check that "
                "layer_%d maps to eagle3_kv_cache and not to a target layer.",
                layer_num,
                bound.dtype,
                self.config.kv_cache_dtype,
                from_config,
                layer_num,
            )
        return bound.dtype, True

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
        from atom.spec_decode.eagle3_kv_builder import Eagle3DraftBuilder

        draft_hf = self.speculative_config.draft_model_hf_config
        draft_atom_config = copy.copy(self.config)
        draft_atom_config.hf_config = draft_hf
        draft_atom_config.compilation_config = copy.copy(self.config.compilation_config)
        draft_atom_config.compilation_config.level = CompilationLevel.NO_COMPILATION
        model = model_class(
            draft_atom_config,
            layer_offset=self.config.hf_config.num_hidden_layers,
        )
        # The draft owns a sibling KV pool. It stores the MLA latent
        # (kv_lora_rank 512 + qk_rope_head_dim 64 = 576 per token), and the
        # Kimi-K3 target has NO pool of that shape to borrow: being a
        # kimi_linear hybrid it goes through the GDN/KDA builder, which
        # allocates its full-attention layers as split K/V at
        # head_dim = qk_nope + qk_rope = 192, not as a compressed latent. So
        # sharing is not merely suboptimal, it is a shape mismatch --
        # concat_and_cache_mla asserts `kv_cache.size(2) == 576`.
        #
        # Eagle3DraftBuilder covers both layouts and picks MLA off the draft
        # config's `kv_lora_rank`. ModelRunner keys its draft-pool allocation
        # and per-module binding off the presence of this attribute.
        self.runner.eagle3_draft_builder = Eagle3DraftBuilder(self.runner, draft_hf)
        return model

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

        # Kimi-K3: (prefix_sum, pending_add, block_residual). ATOM's port defers
        # the FFN residual add across the layer boundary so the next layer can
        # fuse it into apply_attn_res; the HF reference adds it before returning
        # (`prefix_sum = prefix_sum + hidden_states`), and THAT sum is what
        # transformers records and what the draft was trained on. So add it back.
        # `pending_add` is None on the layers that already folded it in.
        if isinstance(output, tuple):
            carrier, pending = output[0], output[1]
            if carrier is None:
                return None
            return carrier if pending is None else carrier + pending

        # Plain residual stream (no special bookkeeping).
        return output

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
        bs = context.batch_size

        # Drafter-owned aux: our own forward-hook capture buffers, row-aligned to
        # the target hidden states.
        aux_hidden_states = self.aux_for(target_hidden_states)
        if aux_hidden_states is None:
            raise RuntimeError(
                "DSpark requires target auxiliary hidden states from "
                "dspark_target_layer_ids; none were captured."
            )
        # Concatenate the configured target layers -> [num_tokens, dim*L].
        main_hidden_all = torch.cat(aux_hidden_states, dim=-1)

        # Anchor token x0 per request = the just-verified target token, located
        # at last_token_indices in the flat batch.
        anchor_ids = next_token_ids
        anchor_positions = torch.index_select(target_positions, 0, last_token_indices)

        if self._with_draft:
            return self._propose_with_draft(
                forward_context,
                attn_metadata,
                bs,
                main_hidden_all,
                target_positions,
                anchor_ids,
                anchor_positions,
            )

        # Populate the rolling target-KV window: write EVERY scheduled row, on
        # prefill and decode alike. A spec-decode step advances the anchor by
        # 1 + accepted drafts, so writing only the anchor would leave most slots
        # at their zero-init value — and the read side gathers them regardless,
        # since validity comes from absolute position, not from what was written.
        # Rejected rows are harmless: swa_write is content-addressed by position
        # and the draft only reads [anchor-window+1, anchor], so they land on
        # future positions, unread until the step that accepts them rewrites them.
        #
        # write_per_batch is the grid's y extent; the kernel writes the LAST
        # min(tok_n, WRITE_PER_BATCH) rows of each span. It must cover
        # window + mtp_k, not just window: the anchor sits up to mtp_k rows
        # before the span end, so a narrower bound drops the anchor's own row.
        # Not clamped by max_seqlen_q the way the V4 target clamps its own
        # swa_write — that was tried and measured worse (GSM8K 0.936/0.941 vs
        # 0.942-0.950), so the over-provisioned grid stays.
        write_per_batch = (
            int(attn_metadata.max_seqlen_q)
            if envs.ATOM_SWA_FULL_RETAIN
            else int(self.model.window_size) + int(self.mtp_k)
        )
        with record_function(f"dspark_ctx_kv[bs={bs} tok={main_hidden_all.shape[0]}]"):
            self.model.precompute_context_kv(
                main_hidden_all,
                target_positions,
                attn_metadata.cu_seqlens_q[: bs + 1],
                write_per_batch=write_per_batch,
            )
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
        num_draft = min(self.mtp_k, window)
        self._refresh_dp_metadata(forward_context, bs * num_draft)
        with record_function(f"dspark[bs={bs} T={num_draft}]"):
            draft_token_ids, confidence = self.model.forward_spec(
                anchor_ids,
                anchor_positions,
                num_draft=num_draft,
            )
        draft_token_ids = draft_token_ids[:, : self.mtp_k]
        # Confidence-scheduled verification. The hardware-aware prefix scheduler
        # consumes the confidence head to pick a per-request verify length
        # ell_r. We compute ell here and stash it; the actual variable-length
        # verification (Level B) is applied downstream by truncating each
        # request's scheduled spec tokens to ell_r, which frees batch capacity
        # instead of the no-op in-block masking of Level A.
        if self.verify_scheduler is not None and confidence is not None:
            with record_function(f"dspark_sched[bs={bs}]"):
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
        main_hidden_all: torch.Tensor,  # [num_tokens, hidden * num_aux]
        target_positions: torch.Tensor,  # [num_tokens]
        anchor_ids: torch.Tensor,  # [bs]
        anchor_positions: torch.Tensor,  # [bs]
    ) -> torch.Tensor:
        """Kimi-K3 DSpark: paged dual-source context + one non-causal block pass.

        Structurally the same two steps as the V4 path -- populate the draft's
        view of the target context, then draft the block -- but the context
        lives in the shared paged latent cache rather than a private rolling
        window, so both steps are addressed by slot mapping instead of by
        position-within-a-window.
        """
        T = self.mtp_k
        block_size = self.runner.block_size
        num_tokens = main_hidden_all.shape[0]
        # warmup_model() runs at the end of ModelRunner.__init__, BEFORE
        # allocate_kv_cache(), so on a dummy run there is no paged state: the
        # draft's kv_cache is still the empty init tensor and attn_metadata's
        # slot_mapping / block_tables are unset. Everything that touches paged
        # state is skipped below; the block forward still runs, because warmup
        # doubles as the memory-profiling pass and omitting the draft would
        # leave its activations out of the KV budget.
        is_dummy = forward_context.context.is_dummy_run

        # ---- 1. Context rows -------------------------------------------------
        if not is_dummy:
            with record_function(f"dspark_ctx_kv[bs={bs} tok={num_tokens}]"):
                self.model.write_context_kv(
                    main_hidden_all,
                    target_positions,
                    attn_metadata.slot_mapping[:num_tokens],
                )

        # ---- 2. Block metadata ----------------------------------------------
        block_positions = self._blk_positions[:bs]  # [bs, T] view, stable
        torch.add(
            anchor_positions.view(bs, 1),
            self._blk_offsets.view(1, T),
            out=block_positions,
        )

        if not is_dummy:
            block_tables = self.runner.forward_vars["block_tables"].gpu[:bs]
            # slot = page_id * block_size + offset_in_page, derived on-device
            # from the block table so there is no host sync.
            page_idx = torch.div(block_positions, block_size, rounding_mode="floor")
            slots = self._blk_slots[: bs * T].view(bs, T)  # stable
            slots.copy_(torch.gather(block_tables, 1, page_idx))
            slots.mul_(block_size)
            slots.add_(torch.remainder(block_positions, block_size))

            # Each request's KV spans [0, anchor] (context) ++ the T block rows.
            ctx_lens = self._blk_ctx_lens[:bs]  # stable
            ctx_lens.copy_(anchor_positions + (1 + T))

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
            kv_indices_generate_triton(
                block_tables,
                self._blk_kv_indices,
                kv_indptr,
                block_size,
                attn_metadata.max_seqlen_k,
            )
            attn_metadata.kv_indptr = kv_indptr
            attn_metadata.kv_indices = self._blk_kv_indices
            attn_metadata.kv_last_page_lens = self._blk_last_page_lens[:bs]

            attn_metadata.work_meta_data = None
            attn_metadata.work_indptr = None
            attn_metadata.work_info_set = None
            attn_metadata.reduce_indptr = None
            attn_metadata.reduce_final_map = None
            attn_metadata.reduce_partial_map = None

        self._refresh_dp_metadata(forward_context, bs * T)

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
        with record_function(f"dspark[bs={bs} T={T}]"):
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

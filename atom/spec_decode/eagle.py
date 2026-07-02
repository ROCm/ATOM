import copy
import logging
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from aiter import dtypes
from aiter.dist.parallel_state import get_pp_group
from atom.config import CompilationLevel, Config, KVCacheTensor
from atom.model_loader.loader import load_model
from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname
from atom.utils import envs
from atom.utils.forward_context import (
    DPMetadata,
    SpecDecodeMetadata,
    get_forward_context,
)
from torch.profiler import record_function

logger = logging.getLogger("atom")


support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "DeepseekV4MTPModel": "atom.models.deepseek_v4_mtp.DeepseekV4MTP",
    "DeepseekV4DSparkModel": "atom.models.deepseek_v4_dspark.DeepseekV4DSpark",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
    "MiMoV2MTPModel": "atom.models.mimo_v2_mtp.MiMoV2MTP",
    "MiMoV2FlashMTPModel": "atom.models.mimo_v2_mtp.MiMoV2MTP",
    "Qwen3_5MTPModel": "atom.models.qwen3_5_mtp.Qwen3_5MTP",
    "Eagle3LlamaModel": "atom.models.eagle3_llama.Eagle3LlamaModel",
    "Eagle3DeepseekMLAModel": "atom.models.eagle3_deepseek_mla.Eagle3DeepseekMLAModel",
}


class Eagle3DraftBuilder:
    """KV cache subsystem for an Eagle3 MHA draft alongside a non-MHA target.

    Implements the same subset of `AttentionMetadataBuilder` hooks that
    ModelRunner consults during KV pool sizing and per-module binding —
    `compute_block_bytes`, `allocate_kv_cache_tensors`, and
    `build_kv_cache_tensor` — so the draft's independent non-MLA cache
    fits the post-#659 builder protocol without leaking into the target's
    builder. The draft does NOT drive prepare_decode/prepare_prefill;
    it piggybacks on the target builder's metadata flow during propose.
    """

    def __init__(self, model_runner, draft_hf):
        self.model_runner = model_runner
        self.draft_hf = draft_hf
        self.block_size = model_runner.block_size
        self.num_kv_heads = draft_hf.num_key_value_heads // model_runner.world_size
        self.num_layers = draft_hf.num_hidden_layers
        self.head_dim = draft_hf.head_dim
        self._next_layer_id = 0  # consumed by build_kv_cache_tensor
        self.num_blocks = 0  # set in allocate_kv_cache_tensors

    def compute_block_bytes(self) -> int:
        """Per-block bytes for the draft's independent non-MLA KV cache."""
        kv_dtype_size = dtypes.d_dtypes[
            self.model_runner.config.kv_cache_dtype
        ].itemsize
        bb = (
            2
            * self.num_layers
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * kv_dtype_size
        )
        if self.model_runner.config.kv_cache_dtype == "fp8":
            # fp8 KV cache needs an extra per-(layer, block, kv_head) scale
            # tensor (one fp32 per element) to dequantize fp8 → bf16 at
            # attention time. Reserve that space alongside the cache.
            bb += (
                2
                * self.num_layers
                * self.block_size
                * self.num_kv_heads
                * dtypes.fp32.itemsize
            )
        return bb

    def allocate_kv_cache_tensors(self, num_kv_heads, num_draft_layers) -> dict:
        """Allocate the draft's [2, L, blocks, block_size, kv_heads, head_dim]
        cache and matching fp32 scale; ModelRunner setattr's both onto itself
        under namespaced keys so they don't collide with the target builder's
        `kv_cache` / `kv_scale`.
        """
        runner = self.model_runner
        config = runner.config
        # Draft's block budget scales with the target pool: same total token
        # capacity, just paged at the draft's own block size.
        self.num_blocks = (
            config.num_kvcache_blocks * runner.block_size // self.block_size
        )
        cache = torch.zeros(
            2,
            self.num_layers,
            self.num_blocks,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=dtypes.d_dtypes[config.kv_cache_dtype],
            device="cuda",
        )
        scale = torch.zeros(
            2,
            self.num_layers,
            self.num_blocks,
            self.num_kv_heads,
            self.block_size,
            dtype=dtypes.fp32,
            device="cuda",
        )
        logger.info(f"Allocated Eagle3 draft KV cache: {cache.shape}")
        return {"eagle3_kv_cache": cache, "eagle3_kv_scale": scale}

    def build_kv_cache_tensor(self, layer_id: int, module):
        """Bind one Eagle3 draft attention module to its slice of the
        independent draft KV cache. Returns None for non-MHA modules so
        ModelRunner falls through to the target builder.
        """
        if not (
            hasattr(module, "base_attention")
            and hasattr(module, "use_mla")
            and not module.use_mla
        ):
            return None
        runner = self.model_runner
        idx = self._next_layer_id
        self._next_layer_id += 1
        cache = runner.eagle3_kv_cache
        x = 16 // cache.element_size()
        k_cache = cache[0, idx].view(
            self.num_blocks,
            self.num_kv_heads,
            self.head_dim // x,
            self.block_size,
            x,
        )
        v_cache = cache[1, idx].view(
            self.num_blocks,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
        )
        module.max_model_len = runner.config.max_model_len
        if runner.config.kv_cache_dtype == "fp8":
            module.k_scale = runner.eagle3_kv_scale[0, idx]
            module.v_scale = runner.eagle3_kv_scale[1, idx]
        module.k_cache = k_cache
        module.v_cache = v_cache
        return KVCacheTensor(
            layer_num=layer_id,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=getattr(module, "k_scale", None),
            v_scale=getattr(module, "v_scale", None),
        )

    def get_kv_transfer_tensors(self) -> list:
        from atom.kv_transfer.disaggregation.types import KVTransferRegion

        runner = self.model_runner
        if not hasattr(runner, "eagle3_kv_cache"):
            return []

        regions: list[KVTransferRegion] = []
        cache = runner.eagle3_kv_cache
        for layer_id in range(self.num_layers):
            for kv in range(2):
                t = cache[kv, layer_id]
                regions.append(
                    KVTransferRegion(
                        base_addr=t.data_ptr(),
                        total_bytes=t.numel() * t.element_size(),
                        unit_bytes=t.stride(0) * t.element_size(),
                    )
                )
        scale = runner.eagle3_kv_scale
        if (
            self.model_runner.config.kv_cache_dtype == "fp8"
            and scale is not None
            and scale.numel() > 0
        ):
            for layer_id in range(self.num_layers):
                for kv in range(2):
                    t = scale[kv, layer_id]
                    regions.append(
                        KVTransferRegion(
                            base_addr=t.data_ptr(),
                            total_bytes=t.numel() * t.element_size(),
                            unit_bytes=t.stride(0) * t.element_size(),
                        )
                    )
        return regions


class EagleProposer:

    def __init__(
        self,
        atom_config: Config,
        device: torch.device,
        runner,
    ):
        self.config = atom_config
        self.speculative_config = self.config.speculative_config
        # DSpark is a parallel block drafter: the verify length is the draft
        # block size, drafted in a single backbone pass (not mtp_k serial steps).
        self.use_dspark = bool(
            getattr(self.speculative_config, "use_dspark", lambda: False)()
        )
        if self.use_dspark:
            draft_cfg = self.speculative_config.draft_model_hf_config
            self.dspark_block_size = int(getattr(draft_cfg, "dspark_block_size"))
            # num_speculative_tokens may be unset for DSpark; default to the
            # full block (Phase 1 uses a static verify length == block size).
            self.mtp_k: int = (
                self.speculative_config.num_speculative_tokens
                or self.dspark_block_size
            )
            # Phase 2: confidence-scheduled verification (Level B, variable-length
            # verify). propose() stores the scheduler-chosen ell here; the next
            # step's calc_spec_decode_metadata consumes it (eager-only for now).
            self.dspark_confidence_schedule = bool(
                envs.ATOM_DSPARK_CONFIDENCE_SCHEDULE
            )
            self._dspark_last_ell: Optional[torch.Tensor] = None
            # req_id -> ell map from the PREVIOUS step's propose(), used to re-map
            # ell onto the next step's (possibly reordered) batch by req_id.
            self._dspark_ell_by_req: dict = {}
            # SPS(B) throughput profile + STS temperatures are bound later
            # (engine warmup / checkpoint). Until then: a synthetic monotone SPS
            # stub and T=1 (uncalibrated) keep the path lossless and testable.
            self.dspark_sps_table: Optional[torch.Tensor] = None
            self.dspark_sts_temperatures: Optional[torch.Tensor] = None
        else:
            self.mtp_k: int = self.speculative_config.num_speculative_tokens or 0

        self.runner = runner
        self.dtype = self.config.torch_dtype
        self.max_model_len = self.config.max_model_len
        self.block_size = self.config.kv_cache_block_size
        self.max_num_tokens = self.config.max_num_batched_tokens
        self.use_cuda_graph = (
            self.config.compilation_config.level == CompilationLevel.PIECEWISE
            and not self.config.enforce_eager
        )
        self.cudagraph_batch_sizes = list(
            reversed(self.config.compilation_config.cudagraph_capture_sizes)
        )

        self.device = device
        draft_model_hf_config = self.speculative_config.draft_model_hf_config
        model_class = resolve_obj_by_qualname(support_eagle_model_arch_dict[draft_model_hf_config.architectures[0]])  # type: ignore

        if self.speculative_config.method == "eagle3":
            # Eagle3 draft has its own architecture, so build it from the
            # draft hf_config. Disable torch.compile for the draft to avoid
            # Dynamo tracing issues with the separate KV cache binding.
            # Shallow-copy instead of deepcopy: with MLA targets (K2.6), the
            # atom_config holds non-picklable cuda.Stream objects under
            # downstream fields that deepcopy can't traverse. We only mutate
            # hf_config and compilation_config.level on the draft, so
            # isolating just those two attrs is sufficient.
            draft_atom_config = copy.copy(atom_config)
            draft_atom_config.hf_config = draft_model_hf_config
            draft_atom_config.compilation_config = copy.copy(
                atom_config.compilation_config
            )
            draft_atom_config.compilation_config.level = CompilationLevel.NO_COMPILATION
            # Draft attention layer_num must continue from the target model's
            # layer count so it maps to the correct kv_cache_data entry.
            self.model = model_class(
                draft_atom_config,
                layer_offset=atom_config.hf_config.num_hidden_layers,
            )
            # MHA draft (e.g. K2.5 LlamaForCausalLMEagle3): owns an independent
            # non-MLA KV cache via Eagle3DraftBuilder, attached to the runner.
            # MLA draft (e.g. K2.6 EAGLE 3.1): same MLA shape as target, so
            # it piggybacks on the target's MLA pool (model_runner accounts
            # for the +1 draft layer via num_nextn_predict_layers default).
            draft_is_mla = bool(getattr(draft_model_hf_config, "kv_lora_rank", None))
            if not draft_is_mla:
                runner.eagle3_draft_builder = Eagle3DraftBuilder(
                    runner, draft_model_hf_config
                )
        else:
            self.model = model_class(self.config)

        self._draft_argmax_fused = hasattr(self.model, "compute_draft_token")

        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        max_bs = self.config.max_num_seqs
        self.arrange_bs = torch.arange(max_bs + 1, **i32_kwargs)
        self.cu_num_draft_tokens = CpuGpuBuffer(max_bs, **i32_kwargs)
        self.target_logits_indices = CpuGpuBuffer(max_bs * self.mtp_k, **i64_kwargs)
        self.bonus_logits_indices = CpuGpuBuffer(max_bs, **i64_kwargs)

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
        if self.speculative_config.method == "eagle3":
            load_model(
                self.model,
                self.speculative_config.model,
                self.speculative_config.draft_model_hf_config,
                self.config.load_dummy,
                False,
            )
            logger.info(
                "Eagle3 draft model loaded from %s (independent embed/lm_head)",
                self.speculative_config.model,
            )
            return

        # MTP: load from the target model checkpoint and share embeddings/lm_head.
        loaded = load_model(
            self.model,
            self.config.model,
            self.speculative_config.draft_model_hf_config,
            self.config.load_dummy,
            True,
        )

        # Resolve the base model (unwrap multimodal wrapper if present)
        target_base = getattr(target_model, "language_model", target_model)

        # Model-specific share hook escape valve. Models whose embed/lm_head
        # naming doesn't match the standard `model.embed_tokens` /
        # `lm_head` convention (e.g. DeepSeek-V4 uses `model.embed` /
        # `model.head`) implement `share_with_target(target_base)` to do
        # their own setattr-rebinding and short-circuit the default path.
        if hasattr(self.model, "share_with_target"):
            self.model.share_with_target(target_base, loaded)
            if self.use_dspark and hasattr(self.model, "reset_kv_cache"):
                # Allocate DSpark's private rolling target-KV window now that the
                # device/dtype and max concurrency are known.
                self.model.reset_kv_cache(
                    self.config.max_num_seqs, self.device, self.dtype
                )
            return

        # Share embed_tokens with the target model
        if (
            get_pp_group().world_size == 1
            and self.model.model.embed_tokens.weight.shape
            == target_base.model.embed_tokens.weight.shape
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

    def _refresh_dp_metadata(self, forward_context, num_local_tokens: int) -> None:
        parallel_config = self.config.parallel_config
        if parallel_config.data_parallel_size <= 1:
            return
        forward_context.context.dp_uniform_decode = False
        forward_context.dp_metadata = DPMetadata.make(
            parallel_config,
            num_local_tokens,
        )

    def _propose_dspark(
        self,
        *,
        target_token_ids: torch.Tensor,   # [num_tokens]
        target_positions: torch.Tensor,   # [num_tokens]
        num_reject_tokens: torch.Tensor,  # [batch]
        next_token_ids: torch.Tensor,     # [batch] verified anchor token x0
        last_token_indices: torch.Tensor,  # [batch] flat index of each anchor row
        aux_hidden_states: Optional[list[torch.Tensor]],
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
        anchor_positions = torch.index_select(
            target_positions, 0, last_token_indices
        )
        main_hidden = torch.index_select(main_hidden_all, 0, last_token_indices)
        # Rolling-KV cache rows MUST be the persistent per-request state slot,
        # NOT a fresh arange(bs). The DSpark window is a ring buffer that
        # accumulates target KV across decode steps; under continuous batching a
        # request's position in the batch drifts between steps, so arange(bs)
        # would scatter each request's history across different rows (reading
        # back stale/foreign KV -> first-token rejection spikes, acceptance
        # collapse). state_slot_mapping is the same stable per-seq slot the V4
        # target uses for its own SWA cache, so DSpark's window stays aligned.
        state_slot = getattr(attn_metadata, "state_slot_mapping", None)
        if state_slot is not None:
            cache_indices = state_slot[:bs].to(torch.long)
        else:
            cache_indices = torch.arange(
                bs, device=anchor_ids.device, dtype=torch.long
            )

        # Prefill warmup: seed each request's rolling window with the last
        # min(seq_len, window) target tokens BEFORE drafting. Right after
        # prefill the window is otherwise empty (only the anchor would be
        # written), so the first draft block sees almost no target context and
        # rejects early. Writing the prefill tail lifts first-block acceptance
        # to the steady-state level. Decode steps skip this (the ring buffer is
        # already populated from prior steps).
        if context.is_prefill:
            cu_seqlens_q = getattr(attn_metadata, "cu_seqlens_q", None)
            if cu_seqlens_q is not None:
                window = int(self.model.model.mtp[0].window_size)
                seqlens = cu_seqlens_q[1 : bs + 1] - cu_seqlens_q[:bs]
                write_per_batch = int(min(int(seqlens.max().item()), window))
                self.model.precompute_context_kv(
                    main_hidden_all,
                    target_positions,
                    cache_indices,
                    cu_seqlens_q=cu_seqlens_q[: bs + 1],
                    write_per_batch=write_per_batch,
                )

        # Refresh the rolling target-KV window with the new anchor row, then
        # draft the block in a single backbone pass.
        self.model.precompute_context_kv(
            main_hidden, anchor_positions, cache_indices
        )
        draft_token_ids, confidence = self.model.forward_spec(
            anchor_ids, main_hidden, anchor_positions, cache_indices
        )
        draft_token_ids = draft_token_ids[:, : self.mtp_k]
        # Phase 2: confidence-scheduled verification. The hardware-aware prefix
        # scheduler (paper Algorithm 1) consumes the confidence head to pick a
        # per-request verify length ell_r. We compute ell here and stash it; the
        # actual variable-length verification (Level B) is applied downstream by
        # truncating each request's scheduled spec tokens to ell_r, which frees
        # batch capacity instead of the no-op in-block masking of Level A.
        if self.dspark_confidence_schedule and confidence is not None:
            self._dspark_last_ell = self._compute_schedule_ell(
                confidence[:, : self.mtp_k]
            )
        else:
            self._dspark_last_ell = None
        return draft_token_ids

    def _compute_schedule_ell(
        self,
        confidence: torch.Tensor,  # [bs, L] per-position acceptance probs
    ) -> torch.Tensor:
        """Run the Hardware-Aware Prefix Scheduler (paper Algorithm 1) and return
        the per-request verify length ``ell`` as an int tensor [bs].

        This ONLY computes ell — it does not touch the draft tokens. The actual
        variable-length verification (Level B) consumes ell downstream to size
        each request's verification batch, which is where the throughput win
        comes from. Kept sync-free (no .item()/.tolist()) for the decode hot path.
        """
        from atom.spec_decode.dspark_scheduler import schedule_prefix_lengths_tensor

        bs, L = confidence.shape
        sps_table = self.dspark_sps_table
        if sps_table is None:
            # Synthetic monotone-decreasing SPS stub until real calibration lands.
            sps_table = torch.linspace(
                1.0, 0.1, steps=bs * (L + 1) + 1, device=confidence.device
            )
        ell_t = schedule_prefix_lengths_tensor(
            confidence.detach(),
            sps_table,
            sts_temperatures=self.dspark_sts_temperatures,
        )
        if envs.ATOM_DSPARK_DEBUG_SCHEDULE:
            self._dspark_dbg_step = getattr(self, "_dspark_dbg_step", 0) + 1
            if self._dspark_dbg_step % 50 == 1:
                avg_ell = float(ell_t.float().mean())
                trunc = float((ell_t < L).float().mean())
                logger.info(
                    "DSpark schedule[step %d]: bs=%d L=%d avg_ell=%.2f "
                    "trunc_rate=%.1f%%",
                    self._dspark_dbg_step, bs, L, avg_ell, trunc * 100.0,
                )
        self._record_dspark_shadow_savings(ell_t, bs, L)
        return ell_t

    def _record_dspark_shadow_savings(
        self, ell_t: torch.Tensor, bs: int, mtp_k: int
    ) -> None:
        """Shadow-mode savings accounting (no effect on verification / output).

        Quantifies, per concurrency level, how many target-verify tokens the
        scheduler WOULD save vs the static mtp_k baseline, under two policies:

          * per-request (paper Algorithm 1): each request verifies ell_r
              saved = sum_r (mtp_k - ell_r)
          * batch-uniform L (our CUDA-graph-friendly simplification, L=max ell_r)
              saved = bs * (mtp_k - max_r ell_r)

        The gap between the two is exactly what batch-uniform L gives up by
        making every request match the strongest one. Paper Figure 8 shows the
        budget only shrinks at HIGH concurrency, so we bucket by bs to see where
        (and whether) either policy actually saves on this deployment.
        """
        if not getattr(self, "dspark_confidence_schedule", False):
            return
        baseline = bs * mtp_k  # static: every request verifies mtp_k draft tokens
        per_req_verified = int(ell_t.sum().item())
        uniform_verified = bs * int(ell_t.max().item())
        per_req_saved = baseline - per_req_verified
        uniform_saved = baseline - uniform_verified

        st = getattr(self, "_dspark_shadow", None)
        if st is None:
            st = {}  # bs -> [steps, baseline_sum, per_req_saved_sum, uniform_saved_sum]
            self._dspark_shadow = st
        rec = st.setdefault(bs, [0, 0, 0, 0])
        rec[0] += 1
        rec[1] += baseline
        rec[2] += per_req_saved
        rec[3] += uniform_saved

        self._dspark_shadow_step = getattr(self, "_dspark_shadow_step", 0) + 1
        if self._dspark_shadow_step % 100 == 0:
            for cbs in sorted(st):
                steps, base, pr, uni = st[cbs]
                if base == 0:
                    continue
                logger.info(
                    "DSpark shadow-savings bs=%d: steps=%d | per-request saves "
                    "%.1f%% of verify | batch-uniform-L saves %.1f%% | "
                    "uniform keeps %.1f%% of the per-request win",
                    cbs, steps,
                    100.0 * pr / base,
                    100.0 * uni / base,
                    (100.0 * uni / pr) if pr > 0 else 0.0,
                )

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch]
        num_reject_tokens: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
        aux_hidden_states: Optional[list[torch.Tensor]] = None,
    ) -> torch.Tensor:

        if self.use_dspark:
            return self._propose_dspark(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                num_reject_tokens=num_reject_tokens,
                next_token_ids=next_token_ids,
                last_token_indices=last_token_indices,
                aux_hidden_states=aux_hidden_states,
            )

        forward_context = get_forward_context()
        context = forward_context.context
        attn_metadata = forward_context.attn_metadata
        bs = context.batch_size
        context.is_draft = True

        assert self.runner is not None

        input_ids = target_token_ids
        # input_ids[last_token_indices] = next_token_ids
        input_ids.scatter_(0, last_token_indices, next_token_ids)
        positions = target_positions + 1

        # Eagle3: project concatenated aux hidden states through fc
        if aux_hidden_states is not None:
            concat_aux = torch.cat(aux_hidden_states, dim=-1)
            hidden_states = self.model.combine_hidden_states(concat_aux)
        else:
            hidden_states = target_hidden_states

        draft_token_ids = torch.empty(
            bs, self.mtp_k, dtype=next_token_ids.dtype, device=next_token_ids.device
        )
        if envs.ATOM_DEBUG_FORCE_SKIP_DRAFT_MODEL:
            draft_token_ids.fill_(-1)
        var = self.runner.forward_vars
        target_uses_mla = self.runner.use_mla
        # Eaale3 only support mha currently
        draft_uses_mha = hasattr(self.runner, "eagle3_draft_builder")

        # Eagle3 MHA reuses target metadata, but the target may be MLA.  Keep
        # write slots sized to this draft pass, and when prefix cache is active
        # restore logical block ids: MLA prefill expands block_tables by
        # block_ratio for its physical block_size=1 pool, while the draft MHA
        # cache is indexed by runner.block_size blocks.
        if draft_uses_mha:
            attn_metadata.slot_mapping = var["slot_mapping"].gpu[: len(input_ids)]
            attn_metadata.block_tables = var["block_tables"].gpu[:bs]

        # Backends that expose flat per-seq kv_indices/kv_indptr (MLA, MHA)
        # wire them through eagle's mid-step block; V4 has block_tables +
        # context_lens instead (its v4_kv_indices_{swa,csa,hca} are per-token
        # non-equivalent). Hoisted out of the loop so the value is bound for
        # every iteration (used at i>=1 too, even though i==0 sets it).
        has_flat_kv = "kv_indices" in var

        for i in range(self.mtp_k):
            with record_function(f"draft[{i}/{self.mtp_k} bs={bs}]"):
                # Re-sync DP token
                self._refresh_dp_metadata(forward_context, input_ids.shape[0])
                ret_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                )

                sample_hidden_states = (
                    torch.index_select(ret_hidden_states, 0, last_token_indices)
                    if i == 0
                    else ret_hidden_states
                )
                # Distributed argmax (all-gather [N, 2] not [N, vocab]) when the
                # draft supports it; token-identical to compute_logits().argmax().
                if self._draft_argmax_fused:
                    new_draft_ids = self.model.compute_draft_token(sample_hidden_states)
                else:
                    logits = self.model.compute_logits(sample_hidden_states)
                    new_draft_ids = logits.argmax(dim=-1)
                draft_token_ids[:, i] = new_draft_ids

                if i < self.mtp_k - 1:
                    do_attn_metadata_update = (
                        not context.is_prefill
                        # TODO: FIX this condition after we support3 attention head numbers=32
                        and self.runner.attn_metadata_builder.num_attention_heads != 32
                    )
                    if i == 0:
                        i0_max_seqlen_q = attn_metadata.max_seqlen_q
                        attn_metadata.max_seqlen_q = 1
                        slot_mapping = var["slot_mapping"].gpu[
                            : bs * attn_metadata.max_seqlen_q
                        ]
                        cu_seqlens_q = var["cu_seqlens_q"].gpu[: bs + 1]
                        attn_metadata.cu_seqlens_q = cu_seqlens_q
                        attn_metadata.slot_mapping = slot_mapping
                        if has_flat_kv:
                            kv_indptr = var["kv_indptr"].gpu[: bs + 1]
                            kv_indices = var["kv_indices"].gpu
                            attn_metadata.kv_indptr = kv_indptr
                            attn_metadata.kv_indices = kv_indices
                        if target_uses_mla:
                            kv_last_page_lens = var["kv_last_page_lens"].gpu[:bs]
                            attn_metadata.kv_last_page_lens = kv_last_page_lens
                        # block_tables, context_lens, and sparse_kv_indptr are
                        # needed by both MHA and MLA+sparse attention
                        attn_metadata.block_tables = var["block_tables"].gpu[:bs]
                        attn_metadata.context_lens = var["context_lens"].gpu[:bs]
                        if "sparse_kv_indptr" in var:
                            attn_metadata.sparse_kv_indptr = var[
                                "sparse_kv_indptr"
                            ].gpu[: bs + 1]
                        cu_seqlens_q[: bs + 1] = self.arrange_bs[: bs + 1]
                        if target_uses_mla and has_flat_kv:
                            # MLA: block_size=1, kv_indptr tracks tokens
                            kv_indptr[1 : bs + 1] -= torch.cumsum(
                                num_reject_tokens, dim=0
                            )
                        if positions.ndim == 1:
                            positions = torch.index_select(
                                positions, 0, last_token_indices
                            )
                        else:
                            # MRoPE positions keep the token axis last (e.g.
                            # [3, num_tokens] for Qwen3.5), so select columns
                            # instead of indexing dim 0.
                            positions = torch.index_select(
                                positions, positions.ndim - 1, last_token_indices
                            )
                        context.is_prefill = False

                    # update metadata
                    attn_metadata.max_seqlen_k += 1
                    fuse_mtp = positions.ndim == 1 and getattr(
                        self.runner.attn_metadata_builder,
                        "fuse_mtp_decode_position_update",
                        False,
                    )
                    if fuse_mtp:
                        mtp_decode_kwargs = {
                            "update_context_lens": True,
                            "positions_out": positions,
                        }
                    else:
                        attn_metadata.context_lens[:bs] += 1
                        positions += 1
                        mtp_decode_kwargs = {}
                    workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                        bs,
                        (
                            attn_metadata.max_seqlen_q
                            if not do_attn_metadata_update
                            else i0_max_seqlen_q
                        ),
                        attn_metadata.max_seqlen_k,
                        positions,
                        only_update=do_attn_metadata_update,
                        num_reject_tokens=num_reject_tokens if i == 0 else None,
                        **mtp_decode_kwargs,
                    )
                    for k, v in workinfos.items():
                        attn_metadata.__dict__[k] = v
                    if has_flat_kv and "slot_mapping" not in workinfos:
                        # MLA/MHA path: slot derived from flat kv_indices.
                        slot_mapping[:] = kv_indices[kv_indptr[1 : bs + 1] - 1]

                    input_ids = new_draft_ids
                    hidden_states = sample_hidden_states

        # self.runner.debug(f"final {draft_token_ids=}")
        # [batch_size, mtp_k]
        return draft_token_ids

    def prepare_inputs(
        self,
        scheduled_bs: int,
        # [batch_size]
        last_token_offset: int | torch.Tensor,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        cu_seqlens_q = attn_metadata.cu_seqlens_q
        # context_lens = attn_metadata.context_lens

        # Only use decode sequences' context_lens and cu_seqlens_q (num_rejected_tokens length matches decode sequences)
        # These may contain padding, so we need to slice to match num_rejected_tokens length
        # context_lens = context_lens[:scheduled_bs]
        # cu_seqlens_q has length scheduled_bs + 1 (includes 0 at start)
        cu_seqlens_q = cu_seqlens_q[: scheduled_bs + 1]

        # Calculate new sequence lengths
        # context_lens += 1

        token_indices = cu_seqlens_q[1:] - last_token_offset

        return token_indices

    def record_dspark_ell(self, req_ids: Sequence) -> None:
        """Stash this step's ell keyed by req_id (called after propose()).

        ell was computed in propose() ordered by THIS step's decode batch. We
        save {req_id: ell} so the NEXT step can re-map it onto its own (possibly
        reordered) batch by req_id — batch position is not stable across steps
        under continuous batching.
        """
        ell = getattr(self, "_dspark_last_ell", None)
        if ell is None:
            self._dspark_ell_by_req = {}
            return
        ell_np = ell.detach().to("cpu").numpy().astype(np.int32)
        n = min(len(req_ids), ell_np.shape[0])
        self._dspark_ell_by_req = {req_ids[i]: int(ell_np[i]) for i in range(n)}

    def _dspark_uniform_verify_len(self, req_ids: Sequence) -> int:
        """Batch-level uniform verify length L for this step (paper §5.2 top-K).

        Re-maps the previous step's per-req ell onto the current batch by req_id,
        then takes L = max over the batch (round up to a single uniform length so
        the verify block stays a regular matrix — simplest, and the shape Level B
        graph buckets will need anyway). Requests with no prior ell (just-started)
        fall back to mtp_k, so a fresh request never gets under-verified.

        Returns L in 1..mtp_k. L == mtp_k means "no truncation this step".
        """
        by_req = getattr(self, "_dspark_ell_by_req", None)
        if not by_req:
            return self.mtp_k
        L = 0
        for rid in req_ids:
            # Missing req (new this step) -> mtp_k so it is fully verified.
            L = max(L, by_req.get(rid, self.mtp_k))
        return int(min(max(L, 1), self.mtp_k))

    def _dspark_per_request_ell(self, req_ids: Sequence) -> np.ndarray:
        """Per-request verify length [bs] re-mapped onto the current batch.

        This is the production policy (paper Algorithm 1 per-request, not the
        batch-uniform L of §13 which only captured 15% of the win). For each
        request in THIS step's batch order, look up the ell its own previous
        step produced (keyed by req_id, so continuous-batching reorders are
        handled). A request with no prior ell (new this step) gets mtp_k so it
        is never under-verified. Values clamped to 1..mtp_k.
        """
        n = len(req_ids)
        out = np.full(n, self.mtp_k, dtype=np.int32)
        by_req = getattr(self, "_dspark_ell_by_req", None)
        if not by_req:
            return out
        for i, rid in enumerate(req_ids):
            v = by_req.get(rid)
            if v is not None:
                out[i] = min(max(int(v), 1), self.mtp_k)
        return out

    def _dspark_verify_lengths(self, scheduled_bs: int) -> np.ndarray:
        """Per-request verify length for this step's verification.

        NOT WIRED YET (Level B). calc_spec_decode_metadata intentionally uses a
        static mtp_k instead of calling this, because shrinking num_draft_tokens
        alone (without also shrinking model_runner tokens_per_seq + all the
        bonus/target index math) desynchronizes the flat-batch indices and
        breaks losslessness (GSM8K 98% -> 52%). This helper keeps the correct
        ell + alignment-guard logic ready for the full Level B wiring (which must
        change tokens_per_seq, KV reservation, and CUDA graph buckets together).

        Returns the scheduler-chosen ell_r when available and consistent with
        the current batch; otherwise the static mtp_k. Eager-only: variable
        query length is incompatible with the fixed-shape decode CUDA graph.

        Alignment guard: ell is produced by the previous step's propose() ordered
        by that step's batch. If the batch size changed (continuous batching
        added/removed requests) we cannot safely reuse it, so fall back to mtp_k.
        Both paths are lossless; the fallback only forgoes truncation that step.
        """
        static = np.full(scheduled_bs, self.mtp_k, dtype=np.int32)
        if not getattr(self, "dspark_confidence_schedule", False):
            return static
        if not self.config.enforce_eager:
            # Variable-length verify needs eager (fixed decode graph otherwise).
            return static
        ell = getattr(self, "_dspark_last_ell", None)
        if ell is None or ell.numel() != scheduled_bs:
            return static
        # ell in 0..mtp_k; clamp to >=1 (verifying 0 draft tokens degenerates the
        # index math; 1 keeps at least the first speculative position).
        ell_np = ell.detach().to("cpu").numpy().astype(np.int32)
        np.clip(ell_np, 1, self.mtp_k, out=ell_np)
        return ell_np

    def calc_spec_decode_metadata(
        self,
        num_sampled_tokens: np.ndarray,
        cu_num_sampled_tokens: np.ndarray,
        input_ids: torch.Tensor,
    ) -> SpecDecodeMetadata:
        scheduled_bs = len(num_sampled_tokens)

        # Verify length is STATIC mtp_k here. The DSpark scheduler still computes
        # ell_r in propose() and stashes it on self._dspark_last_ell, but this
        # metadata MUST stay at mtp_k:
        #   target forward always produces mtp_k+1 logits per request
        #   (model_runner tokens_per_seq = num_spec_tokens+1 is fixed by the
        #   input build + decode CUDA graph shape). num_sampled / cu_num_sampled /
        #   bonus_logits_indices are all derived from that mtp_k+1 layout. If we
        #   shrink num_draft_tokens to ell here WITHOUT also shrinking
        #   tokens_per_seq, the draft/target/bonus indices desynchronize across
        #   the flat batch (one request's bonus lands on another's logits) ->
        #   wrong outputs (observed: GSM8K 98% -> 52%).
        # Real variable-length verification (Level B) requires changing
        # tokens_per_seq + every downstream index together (paper §5.2: top-K
        # batch-capacity K + two-steps-prior causal barrier). Until then we keep
        # mtp_k (lossless, parity Phase 1) and only collect ell for Level B.
        num_draft_tokens = np.full(scheduled_bs, self.mtp_k, dtype=np.int32)
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
        # self.debug(f"{target_logits_indices=}")

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

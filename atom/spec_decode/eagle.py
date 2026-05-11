import copy
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from aiter import dtypes
from aiter.dist.parallel_state import get_pp_group
from atom.config import CompilationLevel, Config, KVCacheTensor
from atom.model_loader.loader import load_model
from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname
from atom.utils.forward_context import SpecDecodeMetadata, get_forward_context
from torch.profiler import record_function

logger = logging.getLogger("atom")


support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
    "MiMoV2FlashMTPModel": "atom.models.mimo_v2_flash_mtp.MiMoV2FlashMTP",
    "Qwen3_5MTPModel": "atom.models.qwen3_5_mtp.Qwen3_5MTP",
    "Eagle3LlamaModel": "atom.models.eagle3_llama.Eagle3LlamaModel",
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


class EagleProposer:

    def __init__(
        self,
        atom_config: Config,
        device: torch.device,
        runner,
    ):
        self.config = atom_config
        self.speculative_config = self.config.speculative_config
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
            # Eagle3 draft model has its own architecture (Llama, not MLA),
            # so it must be constructed with the draft model's hf_config.
            # Also disable torch.compile for the draft model to avoid
            # Dynamo tracing issues with the separate KV cache binding.
            draft_atom_config = copy.deepcopy(atom_config)
            draft_atom_config.hf_config = draft_model_hf_config
            draft_atom_config.compilation_config.level = CompilationLevel.NO_COMPILATION
            # Draft attention layer_num must continue from the target model's
            # layer count so it maps to the correct kv_cache_data entry.
            self.model = model_class(
                draft_atom_config,
                layer_offset=atom_config.hf_config.num_hidden_layers,
            )
            # Attach the draft's KV-cache builder to the runner. ModelRunner
            # consults `runner.eagle3_draft_builder` from `_compute_block_bytes`
            # / `allocate_kv_cache` to size + allocate + bind the draft's
            # independent non-MLA cache through the standard builder protocol.
            runner.eagle3_draft_builder = Eagle3DraftBuilder(
                runner, draft_model_hf_config
            )
        else:
            self.model = model_class(self.config)

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
            # Eagle3: load from a separate draft model checkpoint with
            # independent embed_tokens and lm_head (no sharing).
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
        var = self.runner.forward_vars
        target_uses_mla = self.runner.use_mla
        # Eaale3 only support mha currently
        draft_uses_mha = hasattr(self.runner, "eagle3_draft_builder")

        # Eagle3 MLA: re-slice slot_mapping to len(input_ids).
        # Target's MLA prepare_decode sized it
        # to bs*max_q_len; after rejection len(input_ids) may be smaller,
        # and the MHA cache-write kernel asserts slot_mapping <= q.
        # Other fields (block_tables, context_lens, slot_mapping values
        # themselves) are already in a draft-compatible format because
        # MLA's prepare_decode uses the same runner.block_size as the draft.
        if draft_uses_mha:
            attn_metadata.slot_mapping = var["slot_mapping"].gpu[: len(input_ids)]

        for i in range(self.mtp_k):
            with record_function(f"draft[{i}/{self.mtp_k} bs={bs}]"):
                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                )
                # Eagle3 draft (the only draft_uses_mha case under narrow
                # semantics) returns (post_norm, pre_norm); MTP drafts return
                # a single hidden tensor.
                if draft_uses_mha:
                    ret_hidden_states, ret_hidden_prenorm = model_output
                else:
                    ret_hidden_states = model_output
                    ret_hidden_prenorm = None

                sample_hidden_states = (
                    torch.index_select(ret_hidden_states, 0, last_token_indices)
                    if i == 0
                    else ret_hidden_states
                )
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
                        kv_indptr = var["kv_indptr"].gpu[: bs + 1]
                        kv_indices = var["kv_indices"].gpu
                        slot_mapping = var["slot_mapping"].gpu[
                            : bs * attn_metadata.max_seqlen_q
                        ]
                        cu_seqlens_q = var["cu_seqlens_q"].gpu[: bs + 1]
                        attn_metadata.kv_indptr = kv_indptr
                        attn_metadata.kv_indices = kv_indices
                        attn_metadata.cu_seqlens_q = cu_seqlens_q
                        attn_metadata.slot_mapping = slot_mapping
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
                        if target_uses_mla:
                            # MLA: block_size=1, kv_indptr tracks tokens
                            kv_indptr[1 : bs + 1] -= torch.cumsum(
                                num_reject_tokens, dim=0
                            )
                        positions = torch.gather(positions, 0, last_token_indices)
                        context.is_prefill = False

                    # update metadata
                    attn_metadata.max_seqlen_k += 1
                    # Update context_lens for each draft step (needed by both
                    # MHA attention and MLA+sparse indexer)
                    attn_metadata.context_lens[:bs] += 1
                    workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                        bs,
                        (
                            attn_metadata.max_seqlen_q
                            if not do_attn_metadata_update
                            else i0_max_seqlen_q
                        ),
                        attn_metadata.max_seqlen_k,
                        only_update=do_attn_metadata_update,
                        num_reject_tokens=num_reject_tokens if i == 0 else None,
                    )
                    for k, v in workinfos.items():
                        attn_metadata.__dict__[k] = v
                    slot_mapping[:] = kv_indices[kv_indptr[1 : bs + 1] - 1]

                    input_ids = new_draft_ids
                    positions += 1
                    if ret_hidden_prenorm is not None:
                        hidden_states = (
                            torch.index_select(
                                ret_hidden_prenorm, 0, last_token_indices
                            )
                            if i == 0
                            else ret_hidden_prenorm
                        )
                    else:
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

    def calc_spec_decode_metadata(
        self,
        num_sampled_tokens: np.ndarray,
        cu_num_sampled_tokens: np.ndarray,
        input_ids: torch.Tensor,
    ) -> SpecDecodeMetadata:
        scheduled_bs = len(num_sampled_tokens)
        sum_drafted_tokens = self.mtp_k * scheduled_bs

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        num_draft_tokens = np.full(scheduled_bs, self.mtp_k, dtype=np.int32)
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

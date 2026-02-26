import logging

import numpy as np
import torch
import torch.nn as nn
from aiter.dist.parallel_state import get_pp_group
from atom.config import CompilationLevel, Config
from atom.model_loader.loader import load_model
from atom.utils import CpuGpuBuffer, resolve_obj_by_qualname
from atom.utils.forward_context import SpecDecodeMetadata, get_forward_context

logger = logging.getLogger("atom")


support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": "atom.models.deepseek_mtp.DeepSeekMTP",
    "Qwen3NextMTPModel": "atom.models.qwen3_next_mtp.Qwen3NextMTP",
}


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
        self.model = model_class(self.config)

        i32_kwargs = {"dtype": torch.int32, "device": self.device}
        i64_kwargs = {"dtype": torch.int64, "device": self.device}
        max_bs = self.config.max_num_seqs
        self.arrange_bs = torch.arange(max_bs + 1, **i32_kwargs)
        self.cu_num_draft_tokens = CpuGpuBuffer(max_bs, **i32_kwargs)
        self.target_logits_indices = CpuGpuBuffer(max_bs * self.mtp_k, **i64_kwargs)
        self.bonus_logits_indices = CpuGpuBuffer(max_bs, **i64_kwargs)

    def load_model(self, target_model: nn.Module) -> None:

        load_model(
            self.model,
            self.config.model,
            self.speculative_config.draft_model_hf_config,
            self.config.load_dummy,
            True,
        )

        # share embed_tokens with the target model if needed
        if (
            get_pp_group().world_size == 1
            and self.model.model.embed_tokens.weight.shape
            == target_model.model.embed_tokens.weight.shape
        ):
            logger.info(
                "Assuming the EAGLE head shares the same vocab embedding"
                " with the target model."
            )
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = target_model.model.embed_tokens
        else:
            logger.info(
                "The EAGLE head's vocab embedding will be loaded separately"
                " from the target model."
            )

        if self.config.speculative_config.method != "eagle3" and hasattr(
            target_model, "lm_head"
        ):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_model.lm_head

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
    ) -> torch.Tensor:

        forward_context = get_forward_context()
        context = forward_context.context
        attn_metadata = forward_context.attn_metadata
        context.is_draft = True
        bs = context.batch_size

        assert self.runner is not None
        input_ids = target_token_ids
        # input_ids[last_token_indices] = next_token_ids
        input_ids.scatter_(0, last_token_indices, next_token_ids)
        positions = target_positions
        hidden_states = target_hidden_states

        draft_token_ids = torch.empty(
            bs, self.mtp_k, dtype=next_token_ids.dtype, device=next_token_ids.device
        )
        # return draft_token_ids.fill_(1) # for debug
        var = self.runner.forward_vars
        for i in range(self.mtp_k):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
            )
            sample_hidden_states = (
                ret_hidden_states[last_token_indices] if i == 0 else ret_hidden_states
            )
            logits = self.model.compute_logits(sample_hidden_states)
            new_draft_ids = logits.argmax(dim=-1)
            draft_token_ids[:, i] = new_draft_ids

            if i < self.mtp_k - 1:
                if i == 0:
                    kv_indptr = var["kv_indptr"].gpu[: bs + 1]
                    kv_indices = var["kv_indices"].gpu
                    slot_mapping = var["slot_mapping"].gpu[:bs]
                    kv_last_page_lens = var["kv_last_page_lens"].gpu[:bs]
                    attn_metadata.kv_indptr = kv_indptr
                    attn_metadata.kv_indices = kv_indices
                    attn_metadata.slot_mapping = slot_mapping
                    attn_metadata.kv_last_page_lens = kv_last_page_lens
                    positions = positions[last_token_indices]
                    attn_metadata.max_seqlen_q = 1
                    attn_metadata.cu_seqlens_q[: bs + 1] = self.arrange_bs[: bs + 1]
                    kv_indptr[1 : bs + 1] -= torch.cumsum(num_reject_tokens, dim=0)
                    context.is_prefill = False

                # update metadata
                attn_metadata.max_seqlen_k += 1
                workinfos = self.runner.attn_metadata_builder.prepare_mtp_decode(
                    bs, attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k
                )
                for k, v in workinfos.items():
                    attn_metadata.__dict__[k] = v
                slot_mapping[:] = kv_indices[kv_indptr[1 : bs + 1] - 1]
                input_ids = new_draft_ids
                positions += 1
                hidden_states = sample_hidden_states

        # self.runner.debug(f"{draft_token_ids=}")
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

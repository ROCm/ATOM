import logging

import numpy as np
import torch
import torch.nn as nn
from aiter.dist.parallel_state import get_pp_group
from atom.config import CompilationLevel, Config
from atom.model_engine.scheduler import ScheduledBatch
from atom.model_loader.loader import load_model
from atom.models.deepseek_mtp import DeepSeekMTP
from atom.utils.forward_context import AttentionMetaData, get_forward_context

logger = logging.getLogger("atom")


support_eagle_model_arch_dict = {
    "DeepSeekMTPModel": DeepSeekMTP,
}

class EagleProposer:
    def __init__(
        self,
        atom_config: Config,
        device: torch.device,
        runner=None,
    ):
        self.config = atom_config
        self.speculative_config = self.config.speculative_config
        self.mtp_k = self.speculative_config.num_speculative_tokens

        self.runner = runner
        self.dtype = self.config.hf_config.torch_dtype
        self.max_model_len = self.config.max_model_len
        self.block_size = self.config.kv_cache_block_size
        self.max_num_tokens = self.config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
        self.token_arange_gpu = torch.arange(
            self.max_num_tokens,
            device=device,
            dtype=torch.int32,
        )
        self.use_cuda_graph = (self.config.compilation_config.level == CompilationLevel.PIECEWISE
                               and not self.config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.config.compilation_config.cudagraph_capture_sizes))

        self.device = device
        draft_model_hf_config = self.speculative_config.draft_model_hf_config
        self.model = support_eagle_model_arch_dict[draft_model_hf_config.architectures[0]](self.config)

        self.hidden_size = getattr(self.config.hf_config, "hidden_size", 7168)
        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)

        max_batch_size = self.config.max_num_seqs
        self.arange = torch.arange(
            # We need +1 here because the arange is used to set query_start_loc,
            # which has one more element than batch_size.
            max_batch_size + 1,
            device=device,
            dtype=torch.int32,
        )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)


    def load_model(self, target_model: nn.Module) -> None:

        load_model(self.model, self.config.model, self.speculative_config.draft_model_hf_config, self.config.load_dummy, True)

        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1 \
                and self.model.model.embed_tokens.weight.shape \
            == target_model.model.embed_tokens.weight.shape:
            logger.info(
                "Assuming the EAGLE head shares the same vocab embedding"
                " with the target model.")
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = (
                target_model.model.embed_tokens)
        else:
            logger.info(
                "The EAGLE head's vocab embedding will be loaded separately"
                " from the target model.")

        if self.config.speculative_config.method != "eagle3" and \
                hasattr(target_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_model.lm_head

    @torch.inference_mode()
    def dummy_run(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> None:
        input_ids = self.input_ids[:num_tokens]

        self.model(
            input_ids=input_ids,
            positions=self.positions[:num_tokens],
            hidden_states=self.hidden_states[:num_tokens],
            inputs_embeds=None,
        )


    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # batch: ScheduledBatch,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        bs = next_token_ids.shape[0]

        forward_context = get_forward_context()
        forward_context.context.is_draft = True
        attn_metadata = forward_context.attn_metadata
        last_token_indices = attn_metadata.cu_seqlens_q[1:] - 1

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[:num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states

        input_ids = self.input_ids[:num_input_tokens]

        ret_hidden_states = self.model(
            input_ids=input_ids,
            positions=self.positions[:num_input_tokens],
            hidden_states=self.hidden_states[:num_input_tokens],
            inputs_embeds=None,
        )
        last_hidden_states = ret_hidden_states
        hidden_states = last_hidden_states

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]

        draft_token_ids = logits.argmax(dim=-1)

        # Early exit if there is only one draft token to be generated.
        if self.mtp_k == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)
        else:
            raise ValueError(
                f"num_speculative_tokens={self.mtp_k} is not supported. "
                f"Only num_speculative_tokens=1 is currently supported."
            )

    def prepare_inputs(
            self,
            # [batch_size]
            num_rejected_tokens: torch.Tensor
        ):
            """
            This function is used to prepare the inputs for the spec decode.
            It updates the attn_metadata to account for the rejected
            tokens (and newly sampled tokens). It also returns the token indices
            of the tokens that should be fed to the speculator.
            """
            # E.g.
            #  attn_metadata.cu_seqlens_q: [0, q1, q1 + q2, q1 + q2 + q3]
            #  attn_metadata.context_lens: [s1, s2, s3]
            #  num_rejected_tokens: [n1, n2, n3]
            # This function computes the intermediate values:
            #  num_tokens_per_req: [q1 - n1, q2 - n2, q3 - n3]
            # And returns:
            #  attn_metadata.cu_seqlens_q:
            #       [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
            #  attn_metadata.context_lens:
            #       [s1 - n1 + 1, s2 - n2 + 1, s3 - n3 + 1]
            #  token_indices: [0, 1, ..., q1 - n1 - 1,
            #                 q1, q1 + 1, ..., q1 + q2 - n2 - 1,
            #                 q1 + q2, q1 + q2 + 1, ..., q1 + q2 + q3 - n3 - 1]

            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

            device = attn_metadata.cu_seqlens_q.device
            # Ensure num_rejected_tokens is on the correct device
            if num_rejected_tokens.device != device:
                num_rejected_tokens = num_rejected_tokens.to(device, non_blocking=True)

            cu_seqlens_q = attn_metadata.cu_seqlens_q
            context_lens = attn_metadata.context_lens

            # Only use decode sequences' context_lens and cu_seqlens_q (num_rejected_tokens length matches decode sequences)
            # These may contain padding, so we need to slice to match num_rejected_tokens length
            scheduled_bs = len(num_rejected_tokens)
            context_lens = context_lens[:scheduled_bs]
            # cu_seqlens_q has length scheduled_bs + 1 (includes 0 at start)
            cu_seqlens_q = cu_seqlens_q[:scheduled_bs + 1]

            # Calculate new sequence lengths
            new_context_lens = context_lens - num_rejected_tokens + 1

            # [0, q1, q1 + q2, q1 + q2 + q3] -> [q1, q2, q3]
            new_query_len_per_req = (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
            # [q1, q2, q3] -> [q1 - n1, q2 - n2, q3 - n3]
            new_num_tokens_per_req = new_query_len_per_req - num_rejected_tokens

            # [q1 - n1, q2 - n2, q3 - n3] ->
            # [0, q1 - n1, q1 + q2 - n1 - n2, q1 + q2 + q3 - n1 - n2 - n3]
            new_cu_seqlens_q = torch.zeros(
                cu_seqlens_q.shape,
                dtype=torch.int32,
                device=device)
            new_cu_seqlens_q[1:] = torch.cumsum(new_num_tokens_per_req, dim=0)

            total_num_tokens = new_cu_seqlens_q[-1].item()

            # Create expanded query start locations for token indexing
            new_cu_seqlens_q_expanded = torch.repeat_interleave(
                new_cu_seqlens_q[:-1], new_num_tokens_per_req)

            # Create token offsets within each request
            token_offsets = self.token_arange_gpu[:total_num_tokens] - new_cu_seqlens_q_expanded

            # Expand old starting positions to match token pattern
            old_cu_seqlens_q_expanded = torch.repeat_interleave(
                cu_seqlens_q[:-1], new_num_tokens_per_req)

            # Final token indices
            token_indices = token_offsets + old_cu_seqlens_q_expanded

            # Create new attention metadata
            spec_common_attn_metadata = AttentionMetaData(
                cu_seqlens_q=new_cu_seqlens_q,
                cu_seqlens_k=attn_metadata.cu_seqlens_k,
                max_seqlen_q=new_num_tokens_per_req.max().item(),
                max_seqlen_k=attn_metadata.max_seqlen_k,
                min_seqlen_q=new_num_tokens_per_req.min().item(),
                slot_mapping=attn_metadata.slot_mapping[token_indices],
                context_lens=new_context_lens,
                block_tables=attn_metadata.block_tables,
                dropout_p=attn_metadata.dropout_p,
                kv_indptr=attn_metadata.kv_indptr,
                kv_indices=attn_metadata.kv_indices,
                kv_last_page_lens=attn_metadata.kv_last_page_lens,
                cu_seqlen_ks=attn_metadata.cu_seqlen_ks,
                cu_seqlen_ke=attn_metadata.cu_seqlen_ke,
                sparse_kv_indptr=attn_metadata.sparse_kv_indptr,
                work_meta_data=attn_metadata.work_meta_data,
                work_indptr=attn_metadata.work_indptr,
                work_info_set=attn_metadata.work_info_set,
                reduce_indptr=attn_metadata.reduce_indptr,
                reduce_final_map=attn_metadata.reduce_final_map,
                reduce_partial_map=attn_metadata.reduce_partial_map,
                block_tables_converted=attn_metadata.block_tables_converted,
                kv_indices_converted=attn_metadata.kv_indices_converted,
                sparse_cu_seqlens_q=attn_metadata.sparse_cu_seqlens_q,
                token_to_seq_idxs=attn_metadata.token_to_seq_idxs,
            )

            # Update the forward context
            forward_context.attn_metadata = spec_common_attn_metadata

            return token_indices

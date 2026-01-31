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
        self.use_cuda_graph = (
            self.config.compilation_config.level == CompilationLevel.PIECEWISE
            and not self.config.enforce_eager
        )
        self.cudagraph_batch_sizes = list(
            reversed(self.config.compilation_config.cudagraph_capture_sizes)
        )

        self.device = device
        draft_model_hf_config = self.speculative_config.draft_model_hf_config
        self.model = support_eagle_model_arch_dict[
            draft_model_hf_config.architectures[0]
        ](self.config)

        self.hidden_size = getattr(self.config.hf_config, "hidden_size", 7168)
        # persistent buffers for cuda graph
        self.positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size), dtype=self.dtype, device=device
        )

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

    @torch.inference_mode()
    def dummy_run(
        self,
        input_ids: torch.Tensor,
        num_tokens: int,
    ) -> None:
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
        # [batch]
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]

        forward_context = get_forward_context()
        forward_context.context.is_draft = True
        attn_metadata = forward_context.attn_metadata

        assert self.runner is not None
        input_ids = target_token_ids
        input_ids[last_token_indices] = next_token_ids

        ret_hidden_states = self.model(
            input_ids=input_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=None,
        )
        last_hidden_states = ret_hidden_states
        hidden_states = last_hidden_states

        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        # positions = target_positions[last_token_indices]
        # hidden_states = hidden_states[last_token_indices]

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
        scheduled_bs: int,
        # [batch_size]
        num_bonus_tokens: torch.Tensor,
    ) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        cu_seqlens_q = attn_metadata.cu_seqlens_q
        context_lens = attn_metadata.context_lens

        # Only use decode sequences' context_lens and cu_seqlens_q (num_rejected_tokens length matches decode sequences)
        # These may contain padding, so we need to slice to match num_rejected_tokens length
        context_lens = context_lens[:scheduled_bs]
        # cu_seqlens_q has length scheduled_bs + 1 (includes 0 at start)
        cu_seqlens_q = cu_seqlens_q[: scheduled_bs + 1]

        # Calculate new sequence lengths
        context_lens += 1

        token_indices = cu_seqlens_q[1:] - (1 + self.mtp_k - num_bonus_tokens)

        return token_indices

from atom.config import CompilationLevel, Config
import torch
import torch.nn as nn
import numpy as np
from atom.model_loader.loader import load_model
from atom.models.deepseek_mtp import DeepSeekMTP

from aiter.dist.parallel_state import get_pp_group


import logging
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
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
        self.mtp_k = self.num_speculative_tokens + 1

        self.runner = runner
        self.dtype = self.config.hf_config.torch_dtype
        self.max_model_len = self.config.max_model_len
        self.block_size = self.config.kv_cache_block_size
        self.max_num_tokens = self.config.max_num_batched_tokens
        self.token_arange_np = np.arange(self.max_num_tokens)
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

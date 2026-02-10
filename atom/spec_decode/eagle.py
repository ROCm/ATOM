import logging
import numpy as np
import itertools
import torch
import torch.nn as nn
from aiter.dist.parallel_state import get_pp_group
from atom.config import CompilationLevel, Config
from atom.model_loader.loader import load_model
from atom.models.deepseek_mtp import DeepSeekMTP
from atom.model_engine.scheduler import ScheduledBatch
from atom.utils.forward_context import get_forward_context, AttentionMetaData
from atom.utils.block_convert import (
    block_table_convert_triton,
    kv_indices_convert_triton,
)

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
        self.mtp_k: int = self.speculative_config.num_speculative_tokens or 0

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
        batch: ScheduledBatch,
    ) -> torch.Tensor:

        forward_context = get_forward_context()
        context = forward_context.context
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
        for i in range(self.mtp_k):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
            )
            if i == 0:
                sample_hidden_states = ret_hidden_states[last_token_indices]
            else:
                sample_hidden_states = ret_hidden_states
            logits = self.model.compute_logits(sample_hidden_states)
            new_draft_ids = logits.argmax(dim=-1)
            draft_token_ids[:, i] = new_draft_ids

            if batch.is_dummy_run:
                return draft_token_ids

            if i < self.mtp_k - 1:
                # update metadata
                input_ids = new_draft_ids
                if i == 0:
                    positions = positions[last_token_indices] + 1
                else:
                    positions = positions + 1
                hidden_states = sample_hidden_states

                max_seqlen_q = 1
                context.positions = positions
                context.is_prefill = False

                context_lens = positions + 1
                slot_mapping = [
                    block_table[pos // self.runner.block_size] * self.runner.block_size
                    + (pos % self.runner.block_size)
                    for block_table, seq_len in zip(batch.block_tables, context_lens)
                    for pos in range(seq_len - max_seqlen_q, seq_len)
                ]

                var = self.runner.forward_vars

                scheduled_bs = batch.total_seqs_num
                var["slot_mapping"].np[: scheduled_bs * max_seqlen_q] = slot_mapping
                var["slot_mapping"].np[
                    scheduled_bs * max_seqlen_q : bs * max_seqlen_q
                ] = -1

                block_size = self.runner.attn_metadata_builder.block_size
                block_ratio = self.runner.attn_metadata_builder.block_ratio

                num_blocks_per_seq = [
                    (ctx + block_size - 1) // block_size
                    for ctx in context_lens.tolist()
                ]
                kv_indptr = np.cumsum(num_blocks_per_seq)
                sum_blocks = kv_indptr[-1]
                sum_blocks_before_converted = sum(
                    [(i + block_ratio - 1) // block_ratio for i in num_blocks_per_seq]
                )

                def prepare_kv_indices():
                    dst = var["kv_indices"].np
                    offset = 0
                    for bt in batch.block_tables:
                        n = len(bt)
                        dst[offset : offset + n] = bt
                        offset += n

                prepare_kv_indices()
                var["kv_indptr"].np[1 : scheduled_bs + 1] = kv_indptr
                var["kv_indptr"].np[scheduled_bs + 1 : bs + 1] = sum_blocks
                var["kv_last_page_lens"].np[: scheduled_bs] = 1
                var["kv_last_page_lens"].np[scheduled_bs : bs] = 0

                # Set cu_seqlens_q for decode mode (1 query token per sequence)
                # cu_seqlens_q should be [0, 1, 2, 3, ..., bs]
                var["cu_seqlens_q"].np[: bs + 1] = np.arange(bs + 1, dtype=np.int32)

                vars_used = [
                    ("slot_mapping", bs * max_seqlen_q),
                    ("cu_seqlens_q", bs + 1),
                    ("kv_indptr", bs + 1),
                    ("kv_indices", sum_blocks),
                    ("kv_last_page_lens", bs),
                ]
                ctx = {el: var[el].copy_to_gpu(num) for el, num in vars_used}
                ctx_mla_ps = (
                    self.runner.attn_metadata_builder.set_mla_persistent_worker_buffers(
                        bs, 1
                    )
                )
                ctx.update(ctx_mla_ps)
                if block_ratio > 1:
                    kv_indices_convert_triton(
                        var["kv_indices"].gpu[:sum_blocks_before_converted],
                        var["kv_indices_converted"].gpu[:sum_blocks],
                        var["kv_indptr"].gpu[: bs + 1],
                        block_ratio,
                        block_size,
                    )
                    ctx["kv_indices_converted"] = var["kv_indices_converted"].gpu[:sum_blocks]

                    if "block_tables" in ctx:
                        block_table_convert_triton(
                            var["block_tables"].gpu[:bs],
                            var["block_tables_converted"].gpu[:bs],
                            var["context_lens"].gpu[:bs],
                            block_ratio,
                        )
                        ctx["block_tables_converted"] = var["block_tables_converted"].gpu[:bs]

                attn_metadata = AttentionMetaData(
                    dropout_p=0.0,
                    max_seqlen_q=max_seqlen_q,
                    context_lens=context_lens,
                    **ctx,
                )
                forward_context.attn_metadata = attn_metadata

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
        context_lens = attn_metadata.context_lens

        # Only use decode sequences' context_lens and cu_seqlens_q (num_rejected_tokens length matches decode sequences)
        # These may contain padding, so we need to slice to match num_rejected_tokens length
        context_lens = context_lens[:scheduled_bs]
        # cu_seqlens_q has length scheduled_bs + 1 (includes 0 at start)
        cu_seqlens_q = cu_seqlens_q[: scheduled_bs + 1]

        # Calculate new sequence lengths
        context_lens += 1

        token_indices = cu_seqlens_q[1:] - last_token_offset

        return token_indices

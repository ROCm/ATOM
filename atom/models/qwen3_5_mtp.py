# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3.5 MTP model."""

import logging
import re

import torch
from aiter import QuantType
from aiter.dist.parallel_state import get_tp_group
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.config import Config
from atom.model_ops.embed_head import (
    ParallelLMHead,
    ReplicatedEmbedding,
    VocabParallelEmbedding,
)
from atom.model_ops.linear import ColumnParallelLinear, ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.model_ops.utils import atom_parameter
from atom.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    get_qwen3_5_text_config,
)
from atom.models.utils import IntermediateTensors
from atom.utils import envs

from .utils import maybe_prefix

logger = logging.getLogger("atom")


# Up to here (decode and draft steps) the replicated fc runs whole on each rank.
_REPLICATED_FC_MAX_TOKENS = 1024


# @support_torch_compile
class Qwen3_5MultiTokenPredictor(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()

        quant_config = atom_config.quant_config

        config = get_qwen3_5_text_config(atom_config)

        self.config = config

        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        # Replicated fc (ATOM_MTP_REPLICATED_FC): every rank has the whole fc
        # output without the all-gather. Its input is the same on every rank (the
        # post-collective target hidden and the draft token's embedding), so is
        # its output.
        self.replicated_fc = (
            envs.ATOM_MTP_REPLICATED_FC and get_tp_group().world_size > 1
        )
        fc_cls = ReplicatedLinear if self.replicated_fc else ColumnParallelLinear
        self.fc = fc_cls(
            self.config.hidden_size * 2,
            self.config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )

        self.layers = torch.nn.ModuleList(
            Qwen3_5DecoderLayer(
                atom_config,
                layer_type="full_attention",
                prefix=f"{prefix}.layers.{idx}",
                layer_num=idx,
            )
            for idx in range(
                self.mtp_start_layer_idx,
                self.mtp_start_layer_idx + self.num_mtp_layers,
            )
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def _fc(self, x: torch.Tensor) -> torch.Tensor:
        if not self.replicated_fc:
            return get_tp_group().all_gather(self.fc(x))
        tp = get_tp_group()
        if (
            x.shape[0] <= _REPLICATED_FC_MAX_TOKENS
            or tp.world_size == 1
            or self.fc.quant_type.value != QuantType.No.value
            or getattr(self.fc.weight, "is_shuffled", False)
        ):
            return self.fc(x)
        # A prefill chunk: the full GEMM would be TP times the FLOPs, so this
        # rank's rows of the weight (what ColumnParallelLinear holds) and the
        # all-gather, as without the flag.
        rows = self.fc.weight.shape[0] // tp.world_size
        shard = self.fc.weight[tp.rank_in_group * rows :][:rows]
        return tp.all_gather(tgemm.mm(x, shard, None, otype=x.dtype))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:

        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)
        assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
        inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
        hidden_states = self._fc(hidden_states)
        residual = None

        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# @support_torch_compile
class Qwen3_5MTP(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }
    weights_mapping = {"mtp.": "model."}
    # On a decode step, step 0 runs the target's verify stream at the target's
    # own recorded shape, so it can replay a graph too
    # (EagleProposer._step0_graphed).
    draft_step0_graph = True

    def remap_mtp_weight_name(self, name: str) -> str | None:
        """Filter MTP weights; remap (mtp.* → model.*) is via weights_mapping."""
        shared_weight_names = ["embed_tokens", "lm_head"]

        # MTP-specific weights
        if name.startswith("mtp."):
            return name

        # Shared weights loaded into both target and draft
        if any(key in name for key in shared_weight_names):
            return name

        # Skip target model weights
        return None

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = get_qwen3_5_text_config(atom_config)
        self.config = config

        # Reindex MTP exclude entries on a copy: checkpoint uses 0-based
        # indices (mtp.layers.0.*) but the model uses absolute indices
        # starting at num_hidden_layers (mtp.layers.60.*).
        mtp_start = config.num_hidden_layers
        mtp_atom_config = atom_config
        if atom_config.quant_config is not None and mtp_start > 0:
            import copy

            pat = re.compile(r"^mtp\.layers\.(\d+)\.")
            new_excludes = []
            changed = False
            for entry in atom_config.quant_config.exclude_layers:
                m = pat.match(entry)
                if m:
                    changed = True
                    old_idx = int(m.group(1))
                    entry = pat.sub(f"mtp.layers.{mtp_start + old_idx}.", entry)
                new_excludes.append(entry)
            if changed:
                mtp_atom_config = copy.copy(atom_config)
                mtp_qc = copy.copy(atom_config.quant_config)
                mtp_qc.exclude_layers = list(dict.fromkeys(new_excludes))
                mtp_atom_config.quant_config = mtp_qc

        self.model = Qwen3_5MultiTokenPredictor(
            atom_config=mtp_atom_config, prefix=maybe_prefix(prefix, "mtp")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def post_share_with_target(self, target_base: nn.Module) -> None:
        """Replicate the draft's embedding once it is final
        (ATOM_QWEN35_MTP_REPLICATED_EMBED).

        Runs after the drafter shared the target's embed, so it is whatever the
        draft really reads. The table is all-gathered from the sharded copy
        rather than re-read from the checkpoint, so the replicated rows are
        those very bits. A no-op with the flag off.
        """
        tp = get_tp_group()
        if not (envs.ATOM_QWEN35_MTP_REPLICATED_EMBED and tp.world_size > 1):
            return
        sharded = self.model.embed_tokens
        assert isinstance(sharded, VocabParallelEmbedding), type(sharded)
        full = tp.all_gather(sharded.weight.data, dim=0)
        assert full.shape[0] == sharded.num_embeddings, tuple(full.shape)
        # Built on meta, then handed the gathered table: no second full-size
        # allocation for a weight that is overwritten anyway.
        with torch.device("meta"):
            replicated = ReplicatedEmbedding(full.shape[0], full.shape[1])
        replicated.weight = atom_parameter(full)
        # Only the draft's attribute is rebound; the target keeps its shard.
        self.model.embed_tokens = replicated
        logger.info(
            "Qwen3.5 MTP embedding: REPLICATED (%d x %d per rank, no "
            "post-embed all-reduce)",
            full.shape[0],
            full.shape[1],
        )
        torch.cuda.empty_cache()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, hidden_states, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.lm_head(hidden_states)

    def compute_draft_ids(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """Greedy draft token ids via distributed argmax — each rank reduces its
        own vocab shard and only [N, 2] is all-gathered, instead of the full
        [N, vocab] that compute_logits() would gather. Token-identical to
        compute_logits(...).argmax(-1): the draft path never hits the LM head's
        prefill last-token slice (is_draft is set for the whole propose loop),
        so both see the same rows.
        """
        return self.lm_head.compute_argmax_token(hidden_states, out=out)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts or 0),
        )

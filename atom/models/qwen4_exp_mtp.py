# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next's shared-projection, multi-stream MTP drafter."""

import copy

import torch
from torch import nn

from atom.config import Config
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.qwen4_exp.hyperconnection import (
    Qwen4ExpGroupedRMSNorm,
    Qwen4ExpHyperConnection,
)
from atom.models.qwen4_exp import (
    Qwen4ExpDecoderLayer,
    Qwen4ExpForConditionalGeneration,
    _Qwen4ExpQuantizationConfig,
)


class Qwen4ExpMultiTokenPredictor(nn.Module):
    def __init__(self, atom_config: Config):
        super().__init__()
        config = copy.deepcopy(atom_config.hf_config)
        mtp = getattr(config, "mtp", {}) or {}
        layer_types = mtp.get("layer_types", ["full_attention"])
        if getattr(config, "mtp_num_hidden_layers", 1) != 1 or len(layer_types) != 1:
            raise ValueError("Qwen3.8-Flash-Next MTP requires one reusable draft layer")
        if layer_types[0] not in ("full_attention", "qwen_sparse_attention"):
            raise ValueError("Qwen3.8-Flash-Next MTP requires a QSA draft layer")
        if getattr(config, "mtp_use_dedicated_embeddings", False):
            raise ValueError("Dedicated Qwen MTP embeddings are not supported")
        if mtp.get("mtp_use_hidden_state_from_layer") is not None:
            raise ValueError("Qwen MTP currently consumes the final target HC state")
        config.ple_layer_ids = []
        if "rope_theta" in mtp:
            config.rope_parameters = {
                **config.rope_parameters,
                "rope_theta": mtp["rope_theta"],
            }
        draft_config = copy.copy(atom_config)
        draft_config.hf_config = config
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        quant_config = _Qwen4ExpQuantizationConfig(atom_config.quant_config)
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        for name in ("fc_embedding", "fc_hidden"):
            setattr(
                self,
                name,
                ReplicatedLinear(
                    config.hidden_size,
                    config.hidden_size,
                    bias=False,
                    quant_config=quant_config,
                    prefix=f"mtp.{name}",
                ),
            )
        self.pre_fc_norm_embedding = Qwen4ExpGroupedRMSNorm(
            config.hidden_size, config.hidden_size, config.rms_norm_eps
        )
        # Unlike the per-branch HC norms, this norm spans the whole HC bundle.
        hc_dim = config.hidden_size * config.hc_count
        self.pre_fc_norm_hidden = Qwen4ExpGroupedRMSNorm(
            hc_dim, hc_dim, config.rms_norm_eps
        )
        self.layers = nn.ModuleList(
            [
                Qwen4ExpDecoderLayer(
                    draft_config,
                    layer_types[0],
                    prefix="mtp.layers.0",
                    layer_num=self.mtp_start_layer_idx,
                    quant_config=quant_config,
                )
            ]
        )
        self.hyper_connection_mixer = Qwen4ExpHyperConnection(
            config.hidden_size,
            config.hc_count,
            config.hc_lowrank,
            has_block_inject=False,
            eps=config.rms_norm_eps,
            prefix="mtp.hyper_connection_mixer",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n = hidden_states.shape[0]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        hidden = self.pre_fc_norm_hidden(hidden_states.reshape(n, -1))
        hidden = self.fc_hidden(hidden.reshape(-1, self.hidden_size))
        hidden = hidden.view(n, self.hc_count, self.hidden_size) + embedding[:, None, :]
        hidden = self.layers[0](positions, hidden.flatten(1), None)
        return hidden.view(n, self.hc_count, self.hidden_size)


class Qwen4ExpMTP(nn.Module):
    weights_mapping = Qwen4ExpForConditionalGeneration.weights_mapping
    packed_modules_mapping = Qwen4ExpForConditionalGeneration.packed_modules_mapping
    disable_fused_shared_loading = True
    get_expert_mapping = Qwen4ExpForConditionalGeneration.get_expert_mapping

    def __init__(self, atom_config: Config):
        super().__init__()
        self.config = atom_config.hf_config
        self.model = Qwen4ExpMultiTokenPredictor(atom_config)
        self.lm_head = ParallelLMHead(
            self.config.vocab_size, self.config.hidden_size, prefix="lm_head"
        )

    def remap_mtp_weight_name(self, name: str) -> str | None:
        if not name.startswith("mtp."):
            return None
        return "model." + name[len("mtp.") :]

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, hidden_states, inputs_embeds)

    def _head_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.hyper_connection_mixer.mix(hidden_states.flatten(1))[0]

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.lm_head(self._head_input(hidden_states))

    def compute_draft_ids(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head.compute_argmax_token(
            self._head_input(hidden_states), out=out
        )

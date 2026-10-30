# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next's shared-projection, multi-stream MTP drafter."""

import copy
import json
import os

import torch
from torch import nn

from atom.config import Config
from atom.model_loader.weight_utils import local_model_dir
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
from atom.quant_spec import LayerQuantConfig


def _mtp_weights_quantized(model: str) -> bool:
    """Whether any MTP weight in the checkpoint carries a quantization scale.

    Quark exports list excluded modules one by one and can miss some: the
    Qwen3.8-Flash-Next Quark MXFP4 checkpoint stores the whole MTP head in BF16
    but leaves ``mtp.fc_embedding`` and ``mtp.layers.0.mlp.experts`` off its
    exclude list. Without a readable index the configured policy is kept.
    """
    index = os.path.join(local_model_dir(model), "model.safetensors.index.json")
    try:
        with open(index) as f:
            names = json.load(f)["weight_map"]
    except (OSError, ValueError, KeyError):
        return True
    mtp = [n for n in names if n.startswith("mtp.")]
    return not mtp or any("_scale" in n.rsplit(".", 1)[-1] for n in mtp)


class _UnquantizedMTPQuantizationConfig(_Qwen4ExpQuantizationConfig):
    """Resolves every MTP layer as excluded from quantization."""

    def get_layer_quant_config(
        self, layer_name, use_online_quant=False, *, check_children=False
    ):
        if layer_name.startswith("mtp."):
            return LayerQuantConfig(quant_dtype=self.torch_dtype)
        return super().get_layer_quant_config(
            layer_name, use_online_quant, check_children=check_children
        )


class Qwen4ExpMultiTokenPredictor(nn.Module):
    # The enclosing MTP model opts in after constructing the predictor.
    compact_draft = False

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
        quant_cls = (
            _Qwen4ExpQuantizationConfig
            if _mtp_weights_quantized(atom_config.model)
            else _UnquantizedMTPQuantizationConfig
        )
        quant_config = quant_cls(atom_config.quant_config)
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
        selected_rows: torch.Tensor | None = None,
        with_head_input: bool = False,
    ):
        n = hidden_states.shape[0]
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        hidden = self.pre_fc_norm_hidden(hidden_states.reshape(n, -1))
        hidden = self.fc_hidden(hidden.reshape(-1, self.hidden_size))
        hidden = hidden.view(n, self.hc_count, self.hidden_size) + embedding[:, None, :]
        if self.compact_draft and n <= 16:
            from atom.models.qwen4_exp_mono.draft.compact import (
                forward as compact_forward,
            )

            if with_head_input:
                from atom.models.qwen4_exp_mono.draft.tail import (
                    forward as tail_forward,
                )

                return tail_forward(self, positions, hidden.flatten(1), selected_rows)
            hidden = compact_forward(
                self.layers[0], positions, hidden.flatten(1), selected_rows
            )
        else:
            assert selected_rows is None
            hidden = self.layers[0](positions, hidden.flatten(1), None)
        hidden = hidden.view(-1, self.hc_count, self.hidden_size)
        if with_head_input:
            return hidden, self.hyper_connection_mixer.mix(hidden.flatten(1))[0]
        return hidden


class Qwen4ExpMTP(nn.Module):
    weights_mapping = Qwen4ExpForConditionalGeneration.weights_mapping
    packed_modules_mapping = Qwen4ExpForConditionalGeneration.packed_modules_mapping
    disable_fused_shared_loading = True
    fused_shared_experts = 0
    get_expert_mapping = Qwen4ExpForConditionalGeneration.get_expert_mapping
    draft_graph_supports_mrope = True
    reuse_draft_graph_step_buffers = True

    @staticmethod
    def draft_graph_hidden_state_shape(draft_hf) -> tuple[int, ...]:
        """The draft carries the whole hyper-connection bundle between steps."""
        return (draft_hf.hc_count, draft_hf.hidden_size)

    def __init__(self, atom_config: Config):
        super().__init__()
        self.config = atom_config.hf_config
        self.model = Qwen4ExpMultiTokenPredictor(atom_config)
        self.supports_selected_rows = os.getenv("QWEN_DRAFT_COMPACT", "0") == "1"
        self.model.compact_draft = self.supports_selected_rows
        self.draft_returns_head_input = os.getenv("QWEN_DRAFT_TAIL", "0") == "1"
        assert not self.draft_returns_head_input or self.supports_selected_rows
        if self.supports_selected_rows:
            assert self.model.layers[0].tp_size == 1

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

    def forward_selected(self, input_ids, positions, hidden_states, selected_rows):
        return self.model(
            input_ids, positions, hidden_states, selected_rows=selected_rows
        )

    def forward_with_head(self, input_ids, positions, hidden_states):
        return self.model(input_ids, positions, hidden_states, with_head_input=True)

    def forward_selected_with_head(
        self, input_ids, positions, hidden_states, selected_rows
    ):
        return self.model(
            input_ids,
            positions,
            hidden_states,
            selected_rows=selected_rows,
            with_head_input=True,
        )

    def prepare_draft_head(self):
        self._screen_head = None
        self._use_screen_head = os.environ.get("QWEN_HEAD_SCREEN") == "1"
        if "QWEN_HEAD_SCREEN" not in os.environ:
            return
        import logging

        from atom.models.qwen4_exp_mono.head.screen import HeadScreen

        assert self.lm_head.tp_size == 1 and self.lm_head.bias is None
        assert self.lm_head.weight.shape == (248320, 2560)
        self._screen_head = HeadScreen(self.lm_head.weight)
        if os.environ.get("QWEN_GLOBAL_HEAD") == "1":
            from atom.models.qwen4_exp_mono.head.binding import prepare

            self._screen_head = prepare(self._screen_head)
        logging.getLogger("atom").info(
            "QWEN_HEAD_SCREEN_BOUND %s",
            json.dumps(
                {
                    "rows": 1,
                    "vocab": 248320,
                    "enabled": self._use_screen_head,
                    "original_weight_dtype": str(self.lm_head.weight.dtype),
                    "screen_only": type(self._screen_head).__name__,
                    "refinement": "original_bf16",
                    "same_workspace_allocated_on_both_arms": True,
                }
            ),
        )

    def _draft_argmax(self, head_input, *, out):
        if self._use_screen_head and head_input.shape == (1, 2560):
            return self._screen_head.run(head_input, out=out)
        return self.lm_head.compute_argmax_token(head_input, out=out)

    def compute_draft_ids_with_head(self, head_input, *, out):
        return self._draft_argmax(head_input, out=out)

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
        return self._draft_argmax(self._head_input(hidden_states), out=out)

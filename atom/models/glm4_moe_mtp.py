import torch
from torch import nn
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.moe import FusedMoE
from atom.models.utils import IntermediateTensors
from atom.utils.decorators import support_torch_compile

from .deepseek_mtp import rewrite_spec_layer_name
from .glm4_moe import (
    ENABLE_ALLREDUCE_RMSNORM_FUSION,
    Glm4MoeDecoderLayer,
    get_spec_layer_idx_from_weight_name,
)
from .utils import mask_pos0_inputs_embeds, maybe_prefix


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # Output norm of the MTP layer -- see the matching comment in
        # atom/models/deepseek_mtp.py SharedHead. Applied at the end of
        # Glm4MoeMultiTokenPredictorLayer.forward so the hidden recycled into
        # the next draft step is post-final-norm, and so it can absorb the
        # MoE's pending all-reduce plus the residual-add into one kernel.
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            fused_allreduce=ENABLE_ALLREDUCE_RMSNORM_FUSION,
        )
        self.head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "head"),
        )


class Glm4MoeMultiTokenPredictorLayer(nn.Module):
    def __init__(self, atom_config: Config, prefix: str) -> None:
        super().__init__()

        config = atom_config.hf_config
        self.config = config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=atom_config.quant_config
        )

        self.mtp_block = Glm4MoeDecoderLayer(
            config=config,
            atom_config=atom_config,
            prefix=prefix,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        assert inputs_embeds is not None
        masked_inputs_embeds = inputs_embeds
        inputs_embeds = self.enorm(masked_inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)

        hidden_states = self.eh_proj(
            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        # When allreduce+RMSNorm fusion is on, Glm4MoeDecoderLayer leaves its
        # final MoE down_proj output as an un-reduced TP partial sum, deferring
        # the all-reduce to the *next* layer's fused input_layernorm. The MTP
        # block is the last layer, so there is no next layer to complete it --
        # shared_head.norm completes it instead: its fused branch fires on
        # exactly the same condition (fused_allreduce AND tp_size > 1). When the
        # fusion is off the MoE already reduced internally and the norm just
        # does the add, so neither path double-reduces.
        #
        # See the matching comment in deepseek_mtp.py: on the fused branch aiter
        # only collapses this into a single kernel below its own size gate, and
        # falls back to all_reduce + a separate norm above it. Numerically
        # equivalent either way; only the launch count differs.
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        return hidden_states


class Glm4MoeMultiTokenPredictor(nn.Module):
    def __init__(self, *, atom_config: Config, prefix: str = ""):
        super().__init__()
        config = atom_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): Glm4MoeMultiTokenPredictorLayer(
                    atom_config, f"{prefix}.layers.{idx}"
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        # Set by the vLLM plugin only; see mask_pos0_inputs_embeds.
        self.mask_pos0_inputs_embeds = False

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if self.mask_pos0_inputs_embeds:
            inputs_embeds = mask_pos0_inputs_embeds(inputs_embeds, positions)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )

    def _mtp_layer(self, spec_step_idx: int) -> Glm4MoeMultiTokenPredictorLayer:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)]

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Already post-final-norm (shared_head.norm runs at the end of
        # Glm4MoeMultiTokenPredictorLayer.forward), so this is a bare LM head.
        return self._mtp_layer(spec_step_idx).shared_head.head(hidden_states)

    def compute_draft_ids(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        # Same bare-LM-head input as compute_logits, but reduced per vocab shard
        # so only [N, 2] crosses TP instead of the full [N, vocab].
        head = self._mtp_layer(spec_step_idx).shared_head.head
        return head.compute_argmax_token(hidden_states)


@support_torch_compile
class Glm4MoeMTP(nn.Module):
    # ATOM format: checkpoint_weight_name -> (model_param_name, shard_id)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config = atom_config.hf_config

        self.model = Glm4MoeMultiTokenPredictor(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "model")
        )

    def remap_mtp_weight_name(self, name: str) -> str | None:
        # GLM-4 MoE MTP shares the rewrite rules with DeepSeek MTP:
        #   - shared scalars (embed_tokens) → top-level model.*
        #   - per-layer scalars (enorm/hnorm/eh_proj/shared_head) → kept verbatim
        #   - decoder block weights (self_attn/mlp/...) → model.layers.{i}.mtp_block.*
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is None:
            return None
        return rewrite_spec_layer_name(spec_layer, name)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.model.compute_logits(hidden_states, spec_step_idx)

    def compute_draft_ids(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Greedy draft token ids via distributed argmax — only [N, 2] is
        all-gathered instead of the full [N, vocab] logits. Token-identical to
        compute_logits(...).argmax(-1): plugin mode skips the LM head's prefill
        last-token slice outright, so both see the same rows.

        Reached through the vLLM plugin's ``get_top_tokens``, not through
        EagleProposer -- Glm4MoeMTPModel is absent from
        ``support_draft_model_arch_dict``, so the native drafter cannot build
        this class. The plugin calls it unconditionally, though, so the method
        has to exist.
        """
        return self.model.compute_draft_ids(hidden_states, spec_step_idx)

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

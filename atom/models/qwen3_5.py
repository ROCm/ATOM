from collections.abc import Iterable

import torch
from torch import nn


from atom.config import QuantizationConfig, Config

from atom.model_ops.topK import is_rocm_aiter_fusion_shared_expert_enabled
from atom.model_ops.utils import atom_parameter
from atom.utils.decorators import support_torch_compile

from atom.model_ops.embed_head import VocabParallelEmbedding, ParallelLMHead
from atom.model_config.qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

from atom.model_config.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.linear import (
    MergedColumnParallelLinear,
)
from atom.plugin.prepare import is_vllm
from atom.model_ops.layernorm import GemmaRMSNorm as Qwen3_5RMSNorm
from atom.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextGatedDeltaNet,
    Qwen3NextModel,
    Qwen3NextSparseMoeBlock,
    Qwen3NextMLP,
    Qwen3NextDecoderLayer,
)

from atom.models.utils import (
    IntermediateTensors,
    PPMissingLayer,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    extract_layer_index,
)
from atom.model_ops.split_chunk import (
    fused_split_chunk_zeros,
    fused_split_chunk_zeros_qwen3_5_qkvzba,
)

if is_vllm():
    from vllm.model_executor.layers.mamba.mamba_utils import (
        MambaStateShapeCalculator,
        MambaStateDtypeCalculator,
        MambaStateCopyFunc,
        MambaStateCopyFuncCalculator,
    )


def get_qwen3_5_text_config(atom_config: Config):
    hf_config = atom_config.hf_config
    return hf_config.text_config if hasattr(hf_config, "text_config") else hf_config


# Qwen3.5 MoE models have some checkpoints where expert weights are fused together in BF16 format, so we need special handling to load those weights into our per-expert parameters.
def detect_fused_expert_format(weight_name: str) -> bool:
    """Detect if weight is from fused expert checkpoint (BF16 format)."""
    # Qwen3.5 BF16 has: experts.gate_up_proj, experts.down_proj
    # Qwen3.5 FP8 has: experts.0.gate_proj, experts.0.up_proj, experts.0.down_proj
    return "experts.gate_up_proj" in weight_name or (
        "experts.down_proj" in weight_name
        and ".experts." in weight_name
        and weight_name.count(".experts.") == 1
    )


def get_fused_expert_mapping() -> list[tuple[str, str, str]]:
    """Return mapping for fused expert weights (BF16 format)."""
    # (param_name, weight_name, shard_id)
    return [
        ("experts.w13_weight", "experts.gate_up_proj", "w1"),  # Will be chunked
        ("experts.w2_weight", "experts.down_proj", "w2"),
    ]


def load_fused_expert_weights(
    original_name: str,
    name: str,
    params_dict: dict,
    loaded_weight: torch.Tensor,
    shard_id: str,
    num_experts: int,
) -> bool:
    """Load fused expert weights (BF16 format) into per-expert parameters.

    Args:
        original_name: Original weight name from checkpoint (e.g., "experts.gate_up_proj")
        name: Mapped parameter name (e.g., "experts.w13_weight")
        params_dict: Model parameters dict
        loaded_weight: The weight tensor to load
        shard_id: Shard identifier ("w1", "w2", "w3")
        num_experts: Number of experts

    Returns:
        True if weights were loaded successfully
    """
    param = params_dict[name]
    weight_loader = param.weight_loader
    loaded_local_expert = False

    # Special handling for gate_up_proj: chunk into gate and up
    if "gate_up_proj" in original_name:
        gate_weight, up_weight = loaded_weight.chunk(2, dim=-2)
        # Load gate part (w1)
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    gate_weight[expert_id],
                    name,
                    "w1",
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(param, gate_weight[expert_id], name, "w1", expert_id)
                loaded_local_expert = True
        # Load up part (w3)
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    up_weight[expert_id],
                    name,
                    "w3",
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(param, up_weight[expert_id], name, "w3", expert_id)
                loaded_local_expert = True
    else:
        # down_proj or other weights - no chunking
        for expert_id in range(num_experts):
            try:
                success = weight_loader(
                    param,
                    loaded_weight[expert_id],
                    name,
                    shard_id,
                    expert_id,
                    return_success=True,
                )
                if success:
                    loaded_local_expert = True
            except TypeError:
                weight_loader(
                    param, loaded_weight[expert_id], name, shard_id, expert_id
                )
                loaded_local_expert = True

    return loaded_local_expert


class Qwen3_5GatedDeltaNet(Qwen3NextGatedDeltaNet):

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:

        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
        )

    def create_qkvzba_proj(self, quant_config, prefix):
        if self.quant_config.global_quant_config.quant_dtype == torch.bfloat16:
            self.in_proj_qkvzba = MergedColumnParallelLinear(
                input_size=self.hidden_size,
                output_sizes=[
                    self.key_dim,
                    self.key_dim,
                    self.value_dim,
                    self.value_dim,
                    self.num_v_heads,
                    self.num_v_heads,
                ],
                bias=False,
                quant_config=quant_config,
                prefix=prefix,
            )
        else:
            self.in_proj_qkvz = self.create_qkvz_proj(
                hidden_size=self.hidden_size,
                key_dim=self.key_dim,
                value_dim=self.value_dim,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_qkvz",
            )

            self.in_proj_ba = self.create_ba_proj(
                hidden_size=self.hidden_size,
                num_v_heads=self.num_v_heads,
                quant_config=quant_config,
                prefix=f"{prefix}.in_proj_ba",
            )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        raise NotImplementedError(
            "Qwen3.5 Series dont need to fix query key value ordering"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection
        # ============================================================
        if hasattr(self, "in_proj_qkvzba"):
            qkvzba = self.in_proj_qkvzba(hidden_states)
            k_heads_after_tp = self.num_k_heads // self.tp_size
            v_heads_after_tp = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a, core_attn_out = fused_split_chunk_zeros_qwen3_5_qkvzba(
                qkvzba,
                k_heads_after_tp,
                v_heads_after_tp,
                self.head_k_dim,
                self.head_v_dim,
            )
        else:
            mixed_qkvz = self.in_proj_qkvz(hidden_states)
            ba = self.in_proj_ba(hidden_states)

            qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
            z_size = self.value_dim // self.tp_size
            num_v_heads_tp = self.num_v_heads // self.tp_size

            mixed_qkv, z, b, a, core_attn_out = fused_split_chunk_zeros(
                mixed_qkvz, ba, qkv_size, z_size, self.head_v_dim, num_v_heads_tp
            )

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)

        # ============================================================
        # Part 3: Output Projection
        # ============================================================
        core_attn_out, maybe_scale = self.norm(core_attn_out, z)
        output[:num_tokens] = self.out_proj(core_attn_out, x_scale=maybe_scale)


class Qwen3_5DecoderLayer(Qwen3NextDecoderLayer):
    def __init__(
        self,
        atom_config,
        layer_type: str,
        prefix: str = "",
        layer_num: int = 0,
    ) -> None:
        super(Qwen3NextDecoderLayer, self).__init__()

        config = get_qwen3_5_text_config(atom_config)
        quant_config = atom_config.quant_config
        speculative_config = atom_config.speculative_config

        self.layer_type = layer_type
        self.layer_idx = layer_num

        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(
                atom_config,
                quant_config=quant_config,
                speculative_config=speculative_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                atom_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen3NextSparseMoeBlock(
                config,
                atom_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            raise ValueError(f"Invalid model_type {config.model_type}")

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = atom_parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                )
            )
            self.ffn_layer_scale = atom_parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                )
            )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3_5Model(Qwen3NextModel):
    def __init__(self, *, atom_config, prefix: str = ""):
        super(Qwen3NextModel, self).__init__()
        config: Qwen3_5TextConfig | Qwen3_5MoeTextConfig = get_qwen3_5_text_config(
            atom_config
        )

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str, layer_num: int):
            return Qwen3_5DecoderLayer(
                atom_config=atom_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                layer_num=layer_num,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3_5ForCausalLMBase(nn.Module):

    def __init__(self, atom_config: Config, prefix: str = ""):
        config: Qwen3_5MoeTextConfig = get_qwen3_5_text_config(atom_config)
        self.atom_config = atom_config

        self.quant_config = atom_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                prefix=maybe_prefix(prefix, "lm_head"),
            )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.lm_head(hidden_states)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLMBase):
    def __init__(self, atom_config: Config, prefix: str = ""):
        config: Qwen3_5MoeTextConfig = get_qwen3_5_text_config(atom_config)
        config.n_shared_experts = 1
        config.n_routed_experts = config.num_experts
        super().__init__(atom_config=atom_config, prefix=prefix)
        self.config = config

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if is_rocm_aiter_fusion_shared_expert_enabled()
                else 0
            ),
        )


class Qwen3_5ForConditionalGenerationTextOnly(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
        "in_proj_z": ("in_proj_qkvz", 3),
        "in_proj_b": ("in_proj_ba", 0),
        "in_proj_a": ("in_proj_ba", 1),
        ".gate.": (".gate.", 0),
        "shared_expert_gate": ("gate", 1),
    }
    weights_mapping = {
        "model.language_model.": "language_model.model.",
        "lm_head.": "language_model.lm_head.",
    }
    quant_exclude_name_mapping = {
        "model.language_model.": "model.",
    }
    skip_weight_prefixes = ["model.visual."]

    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config = atom_config.hf_config
        self.visual = PPMissingLayer()
        self.language_model = Qwen3_5ForCausalLM(atom_config=atom_config, prefix="")
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_: object,
    ):
        return self.language_model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)


class Qwen3_5MoeForConditionalGenerationTextOnly(
    Qwen3_5ForConditionalGenerationTextOnly
):
    def __init__(self, atom_config: Config, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = atom_config.hf_config
        self.visual = PPMissingLayer()
        self.language_model = Qwen3_5MoeForCausalLM(atom_config=atom_config, prefix="")
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )

    def detect_fused_expert_format(self, weight_name: str) -> bool:
        """Detect if weight is from fused expert checkpoint (BF16 format)."""
        # Qwen3.5 BF16 has: experts.gate_up_proj, experts.down_proj
        # Qwen3.5 FP8 has: experts.0.gate_proj, experts.0.up_proj, experts.0.down_proj
        return detect_fused_expert_format(weight_name)

    def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
        """Return mapping for fused expert weights (BF16 format)."""
        # (param_name, weight_name, shard_id)
        return get_fused_expert_mapping()

    def load_fused_expert_weights(
        self,
        original_name: str,
        name: str,
        params_dict: dict,
        loaded_weight: torch.Tensor,
        shard_id: str,
        num_experts: int,
    ) -> bool:
        """Load fused expert weights (BF16 format) into per-expert parameters.

        Args:
            original_name: Original weight name from checkpoint (e.g., "experts.gate_up_proj")
            name: Mapped parameter name (e.g., "experts.w13_weight")
            params_dict: Model parameters dict
            loaded_weight: The weight tensor to load
            shard_id: Shard identifier ("w1", "w2", "w3")
            num_experts: Number of experts

        Returns:
            True if weights were loaded successfully
        """
        return load_fused_expert_weights(
            original_name,
            name,
            params_dict,
            loaded_weight,
            shard_id,
            num_experts,
        )

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.language_model.get_expert_mapping()


########################################################
# Qwen3_5-Dense
########################################################

# ConditionalGeneration model scope should only works on plugin mode
if is_vllm():
    from vllm.config import VllmConfig
    from vllm.model_executor.models.qwen3_vl import (
        Qwen3VLMultiModalProcessor,
        Qwen3VLDummyInputsBuilder,
        Qwen3_VisionTransformer,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ProcessingInfo,
        Qwen3_5MoeProcessingInfo,
    )

    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5ForConditionalGeneration as vLLMQwen3_5,
    )
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5MoeForConditionalGeneration as vLLMQwen3_5Moe,
    )
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.model_executor.models.interfaces import IsHybrid
    from atom.model_loader.loader import load_model_in_plugin_mode, WeightsMapper
    from atom.plugin.vllm.model_wrapper import ATOMForConditionalGeneration

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5ProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5ForConditionalGeneration_(vLLMQwen3_5):
        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "gate_up_proj": ["gate_proj", "up_proj"],  # BF16 models: fused → split
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )
        hf_to_vllm_mapper = hf_to_atom_mapper

        def __init__(self, atom_config: Config, prefix: str = "model"):
            # protocols have not __init__ method, so we need to use nn.Module.__init__
            nn.Module.__init__(self)
            config: Qwen3_5Config = atom_config.hf_config
            vllm_config = atom_config.plugin_config.vllm_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config
            self.atom_config = atom_config
            if (
                self.atom_config.quant_config.global_quant_config.quant_dtype
                == torch.bfloat16
            ):
                self.packed_modules_mapping.pop("in_proj_qkv")
                self.packed_modules_mapping.pop("in_proj_b")
                self.packed_modules_mapping.pop("in_proj_a")
                self.packed_modules_mapping["in_proj_qkv"] = (
                    "in_proj_qkvzba",
                    (0, 1, 2),
                )
                self.packed_modules_mapping["in_proj_z"] = ("in_proj_qkvzba", (3))
                self.packed_modules_mapping["in_proj_b"] = ("in_proj_qkvzba", (4))
                self.packed_modules_mapping["in_proj_a"] = ("in_proj_qkvzba", (5))

            self.config = config
            self.multimodal_config = multimodal_config
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self.video_pruning_rate = multimodal_config.video_pruning_rate
            self.is_multimodal_pruning_enabled = (
                multimodal_config.is_multimodal_pruning_enabled()
            )
            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

            with self._mark_language_model(vllm_config):
                self.language_model = Qwen3_5ForCausalLM(
                    atom_config=atom_config,
                    prefix=maybe_prefix("", "language_model"),
                )
            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            # load weights in plugin mode and discard passed weights generator
            # here prefix is "model." because Qwen3ForCausalLM is constructed in model
            # wrapper class, so the name of loaded weights are prefixed with "model.".
            # The vLLM will check the name of the loaded weights to make sure all the
            # weights are loaded correctly
            loaded_weights_record = load_model_in_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="model.",
                weights_mapper=self.hf_to_atom_mapper,
            )
            return loaded_weights_record

    ########################################################
    # Qwen3_5-MoE
    ########################################################

    class Qwen3_5MoeForConditionalGeneration_(vLLMQwen3_5Moe):
        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "gate_up_proj": ["gate_proj", "up_proj"],  # BF16 models: fused → split
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.visual.": "visual.",
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )

        def __init__(self, atom_config: Config, prefix: str = "model"):
            # protocols have not __init__ method, so we need to use nn.Module.__init__
            nn.Module.__init__(self)
            self.atom_config = atom_config
            vllm_config = atom_config.plugin_config.vllm_config
            atom_config.hf_config.text_config.n_shared_experts = 1
            atom_config.hf_config.text_config.n_routed_experts = (
                atom_config.hf_config.text_config.num_experts
            )
            config: Qwen3_5MoeConfig = atom_config.hf_config
            quant_config = vllm_config.quant_config
            multimodal_config = vllm_config.model_config.multimodal_config
            if (
                self.atom_config.quant_config.global_quant_config.quant_dtype
                == torch.bfloat16
            ):
                self.packed_modules_mapping.pop("in_proj_qkv")
                self.packed_modules_mapping.pop("in_proj_b")
                self.packed_modules_mapping.pop("in_proj_a")
                self.packed_modules_mapping["in_proj_qkv"] = (
                    "in_proj_qkvzba",
                    (0, 1, 2),
                )
                self.packed_modules_mapping["in_proj_z"] = ("in_proj_qkvzba", (3))
                self.packed_modules_mapping["in_proj_b"] = ("in_proj_qkvzba", (4))
                self.packed_modules_mapping["in_proj_a"] = ("in_proj_qkvzba", (5))

            self.config = config
            self.multimodal_config = multimodal_config
            self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
            self.video_pruning_rate = multimodal_config.video_pruning_rate
            self.is_multimodal_pruning_enabled = (
                multimodal_config.is_multimodal_pruning_enabled()
            )

            with self._mark_tower_model(vllm_config, {"image", "video"}):
                self.visual = Qwen3_VisionTransformer(
                    config.vision_config,
                    norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )

            with self._mark_language_model(vllm_config):
                self.language_model = Qwen3_5MoeForCausalLM(
                    atom_config=atom_config, prefix=maybe_prefix("", "language_model")
                )

            self.make_empty_intermediate_tensors = (
                self.language_model.make_empty_intermediate_tensors
            )

        def detect_fused_expert_format(self, weight_name: str) -> bool:
            """Detect if weight is from fused expert checkpoint (BF16 format)."""
            # Qwen3.5 BF16 has: experts.gate_up_proj, experts.down_proj
            # Qwen3.5 FP8 has: experts.0.gate_proj, experts.0.up_proj, experts.0.down_proj
            return detect_fused_expert_format(weight_name)

        def get_fused_expert_mapping(self) -> list[tuple[str, str, str]]:
            """Return mapping for fused expert weights (BF16 format)."""
            # (param_name, weight_name, shard_id)
            return get_fused_expert_mapping()

        def load_fused_expert_weights(
            self,
            original_name: str,
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ) -> bool:
            """Load fused expert weights (BF16 format) into per-expert parameters.

            Args:
                original_name: Original weight name from checkpoint (e.g., "experts.gate_up_proj")
                name: Mapped parameter name (e.g., "experts.w13_weight")
                params_dict: Model parameters dict
                loaded_weight: The weight tensor to load
                shard_id: Shard identifier ("w1", "w2", "w3")
                num_experts: Number of experts

            Returns:
                True if weights were loaded successfully
            """
            return load_fused_expert_weights(
                original_name,
                name,
                params_dict,
                loaded_weight,
                shard_id,
                num_experts,
            )

        def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
            # load weights in plugin mode and discard passed weights generator
            # here prefix is "model." because Qwen3ForCausalLM is constructed in model
            # wrapper class, so the name of loaded weights are prefixed with "model.".
            # The vLLM will check the name of the loaded weights to make sure all the
            # weights are loaded correctly
            loaded_weights_record = load_model_in_plugin_mode(
                model=self,
                config=self.atom_config,
                prefix="model.",
                weights_mapper=self.hf_to_atom_mapper,
                load_fused_expert_weights_fn=self.load_fused_expert_weights,
            )
            return loaded_weights_record

        def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
            return self.language_model.get_expert_mapping()

        def embed_multimodal(self, **kwargs):
            return super().embed_multimodal(**kwargs)

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5ProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5ForConditionalGeneration(ATOMForConditionalGeneration, IsHybrid):

        packed_modules_mapping = {
            "q_proj": ("qkv_proj", "q"),
            "k_proj": ("qkv_proj", "k"),
            "v_proj": ("qkv_proj", "v"),
            "gate_proj": ("gate_up_proj", 0),
            "up_proj": ("gate_up_proj", 1),
            "gate_up_proj": ["gate_proj", "up_proj"],  # BF16 models: fused → split
            "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
            "in_proj_z": ("in_proj_qkvz", 3),
            "in_proj_b": ("in_proj_ba", 0),
            "in_proj_a": ("in_proj_ba", 1),
            ".gate.": (".gate.", 0),
            "shared_expert_gate": ("gate", 1),
        }

        hf_to_atom_mapper = WeightsMapper(
            orig_to_new_prefix={
                "lm_head.": "language_model.lm_head.",
                "model.language_model.": "language_model.model.",
            }
        )
        hf_to_vllm_mapper = hf_to_atom_mapper

        def embed_multimodal(self, **kwargs):
            return self.model.embed_multimodal(**kwargs)

        @classmethod
        def get_placeholder_str(cls, modality: str, i: int) -> str | None:
            if modality.startswith("image"):
                return "<|vision_start|><|image_pad|><|vision_end|>"
            if modality.startswith("video"):
                return "<|vision_start|><|video_pad|><|vision_end|>"

            raise ValueError("Only image or video modality is supported")

        @classmethod
        def get_mamba_state_dtype_from_config(
            cls,
            vllm_config: "VllmConfig",
        ) -> tuple[torch.dtype, torch.dtype]:
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

        @classmethod
        def get_mamba_state_shape_from_config(
            cls, vllm_config: "VllmConfig"
        ) -> tuple[tuple[int, int], tuple[int, int]]:
            parallel_config = vllm_config.parallel_config
            hf_config = vllm_config.model_config.hf_text_config
            tp_size = parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )
            return MambaStateShapeCalculator.gated_delta_net_state_shape(
                tp_size,
                hf_config.linear_num_key_heads,
                hf_config.linear_num_value_heads,
                hf_config.linear_key_head_dim,
                hf_config.linear_value_head_dim,
                hf_config.linear_conv_kernel_dim,
                num_spec,
            )

        @classmethod
        def get_mamba_state_copy_func(
            cls,
        ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

        def load_weights(
            self,
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> set[str]:
            return self.model.load_weights(weights)

    @MULTIMODAL_REGISTRY.register_processor(
        Qwen3VLMultiModalProcessor,
        info=Qwen3_5MoeProcessingInfo,
        dummy_inputs=Qwen3VLDummyInputsBuilder,
    )
    class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration, IsHybrid):

        def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
            return self.model.get_expert_mapping()

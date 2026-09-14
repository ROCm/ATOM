import fnmatch
from typing import ClassVar

import numpy as np
import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from torch import nn

from atom.config import Config
from atom.model_ops.base_attention import LinearAttention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.qwen4_exp.hyperconnection import (
    Qwen4ExpHyperConnection,
)
from atom.model_ops.qwen4_exp.ops.gated import sigmoid_mul, sigmoid_rmsnorm
from atom.model_ops.qwen4_exp.ple_layer import Qwen4ExpPLELayer
from atom.model_ops.qwen4_exp.qsa_attention import (
    Qwen4ExpAttention,
)
from atom.model_ops.utils import atom_parameter
from atom.models.qwen3_next import Qwen3NextMLP, mamba_v2_sharded_weight_loader
from atom.models.utils import (
    IntermediateTensors,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.quant_spec import LayerQuantConfig
from atom.utils.forward_context import get_forward_context


class _Qwen4ExpQuantizationConfig:
    """Read-only policy view for HF and ATOM names, without rewriting regexes.

    Exclusions take precedence over ordered layer rules, as in QuantizationConfig.
    GDN policies also recognize the packed name, but the four checkpoint shards
    remain separate queries until their compatibility has been checked.
    """

    def __init__(self, config):
        self._config = config

    def __getattr__(self, name):
        return getattr(self._config, name)

    def get_layer_quant_config(
        self, layer_name, use_online_quant=False, *, check_children=False
    ):
        names = [layer_name]
        if layer_name.rsplit(".", 1)[-1] in (
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
        ):
            names.append(layer_name.rsplit(".", 1)[0] + ".in_proj_qkvzba")
        for name in tuple(names):
            if name.startswith("model."):
                names.append("model.language_model." + name[len("model.") :])
            elif name.startswith("visual."):
                names.append("model." + name)
        field = "online_" if use_online_quant else ""
        excludes = getattr(self._config, field + "exclude_layers")
        if any(
            self._config._is_excluded(name, excludes, check_children=check_children)
            for name in names
        ):
            return LayerQuantConfig(quant_dtype=self.torch_dtype)
        for pattern, spec in getattr(self._config, field + "layer_pattern_specs"):
            if any(
                (
                    name in pattern
                    if "*" not in pattern
                    else fnmatch.fnmatch(name, pattern)
                )
                for name in names
            ):
                return spec
        return getattr(self._config, field + "global_spec")


class Qwen4ExpRMSNormGated(nn.Module):
    """GDN's sigmoid gate with the checkpoint's BF16 cast boundaries."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=None):
        super().__init__()
        self.weight = atom_parameter(torch.empty(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return sigmoid_rmsnorm(x, z, self.weight, self.eps)


def install_stacked_expert_loaders(experts: FusedMoE) -> None:
    """Let `FusedMoE` accept whole-layer `[E, ...]` expert tensors as well.

    Qwen3.8-Flash-Next ships in two expert layouts and the port has to read both:

    * the internal BF16 checkpoint stacks a whole layer into one tensor --
      `experts.gate_up_proj [E, 2I, H]` with gate and up as contiguous HALVES
      (confirmed against the reference, which chunks dim 1 into `w1`/`w3`) and
      `experts.down_proj [E, H, I]`;
    * the released FP8 checkpoint stores one tensor per expert per projection
      (`experts.0.gate_proj.weight` + `weight_scale_inv`), which is the layout
      `FusedMoE`'s own loader already handles.

    So this WRAPS the stock loader rather than replacing it: a 3D tensor
    arriving with no `shard_id` is the stacked form and is written here;
    everything else -- including every FP8 scale -- falls through untouched.
    Replacing it outright would make the per-expert path call a two-argument
    function with five arguments.

    Both axes of the stacked form may be sharded, and which one is depends on
    the parallel mode: expert parallelism cuts the expert axis and leaves the
    intermediate whole, tensor parallelism the reverse. Both ranks come from
    the MoE's own parallel config rather than the model's TP group, which is
    not the same thing here.
    """
    moe_parallel = experts.moe_parallel_config
    tp_rank = moe_parallel.tp_rank
    expert_map = getattr(experts, "expert_map", None)
    if expert_map is None:
        expert_slice = slice(None)
    else:
        local = torch.nonzero(expert_map >= 0).flatten()
        first, last = int(local[0]), int(local[-1])
        if last - first + 1 != local.numel():
            raise NotImplementedError(
                "Qwen3.8-Flash-Next expert loading needs a contiguous expert-parallel range"
            )
        expert_slice = slice(first, last + 1)

    def stacked_w13(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        intermediate = param.data.shape[1] // 2
        full = loaded_weight.shape[1] // 2
        stacked = loaded_weight[expert_slice]
        for half in range(2):
            source = stacked.narrow(
                1, half * full + tp_rank * intermediate, intermediate
            )
            target = param.data.narrow(1, half * intermediate, intermediate)
            target.copy_(source.to(device=target.device, dtype=target.dtype))

    def stacked_w2(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        intermediate = param.data.shape[2]
        source = loaded_weight[expert_slice].narrow(
            2, tp_rank * intermediate, intermediate
        )
        param.data.copy_(source.to(device=param.device, dtype=param.dtype))

    def wrap(param: nn.Parameter, stacked_loader):
        stock = param.weight_loader

        def loader(param, loaded_weight, *args, **kwargs):
            # The stacked form is the only one that arrives as a bare 3D
            # tensor with no shard id; the per-expert form always carries one.
            shard_id = kwargs.get("shard_id", args[1] if len(args) > 1 else "")
            if loaded_weight.dim() == 3 and not shard_id:
                stacked_loader(param, loaded_weight)
            else:
                stock(param, loaded_weight, *args, **kwargs)

        return loader

    experts.w13_weight.weight_loader = wrap(experts.w13_weight, stacked_w13)
    experts.w2_weight.weight_loader = wrap(experts.w2_weight, stacked_w2)


class _UnfusedSharedExpertConfig:
    """Config view that hides `n_shared_experts` from `FusedMoE`.

    ATOM synthesizes `n_shared_experts=1` for any checkpoint carrying
    `shared_expert` tensors, and `FusedMoE` then reserves a 513th expert slot
    to fuse it into. Qwen3.8-Flash-Next's routed experts arrive as one fixed `[512, ...]`
    stack with no room for that slot, so the shared expert stays a standalone
    MLP here -- which is also what the reference implementation does. Every
    other attribute passes straight through.
    """

    def __init__(self, config) -> None:
        self._config = config

    def __getattr__(self, name: str):
        if name == "n_shared_experts":
            return 0
        return getattr(self._config, name)


class Qwen4ExpSparseMoeBlock(nn.Module):
    """512 routed experts, top-10, plus one sigmoid-gated shared expert.

    `mlp.gate.weight [512, 2560]` and `mlp.shared_expert_gate.weight [1, 2560]`
    merge into one replicated projection, exactly as Qwen3-Next does: the tail
    column is the shared expert's gate.
    """

    def __init__(self, config, quant_config, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = int(config.num_experts)
        if self.tp_size > self.n_routed_experts:
            raise ValueError(
                f"TP {self.tp_size} exceeds the expert count {self.n_routed_experts}"
            )

        self.gate = MergedReplicatedLinear(
            config.hidden_size,
            [self.n_routed_experts, 1],
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert = Qwen3NextMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_act,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.shared_expert",
        )
        self.experts = FusedMoE(
            num_experts=self.n_routed_experts,
            top_k=int(config.num_experts_per_tok),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            use_grouped_topk=False,
            has_bias=False,
            prefix=f"{prefix}.experts",
            config=_UnfusedSharedExpertConfig(config),
            shared_expert_prefix=f"{prefix}.shared_expert",
        )
        install_stacked_expert_loaders(self.experts)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, orig_shape[-1])
        logits = self.gate(hidden_states)
        routed = self.experts(
            hidden_states=hidden_states,
            router_logits=logits[:, : self.n_routed_experts],
        )
        shared = self.shared_expert(hidden_states)
        out = sigmoid_mul(shared, logits[:, self.n_routed_experts :], routed)
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.view(orig_shape)


class Qwen4ExpLinearAttention(nn.Module):
    """Gated DeltaNet, on the 36 `linear_attention` layers.

    Same recurrence as Qwen3-Next -- ATOM's `LinearAttention` / `GatedDeltaNet`
    run it unchanged. Separate checkpoint projections are packed into the
    existing MergedColumnParallelLinear at load time, with Q/K/V sharded
    independently. The forward consumes zero-copy [Q|K|V|Z|B|A] views.
    """

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    # Kept out of the class-level model mapping: the runner must not collapse
    # original quantization policies before this layer has checked every shard.
    packed_modules_mapping: ClassVar[dict] = {
        ".in_proj_qkv": (".in_proj_qkvzba", (0, 1, 2)),
        ".in_proj_z": (".in_proj_qkvzba", 3),
        ".in_proj_b": (".in_proj_qkvzba", 4),
        ".in_proj_a": (".in_proj_qkvzba", 5),
    }

    def __init__(
        self, atom_config, config, quant_config=None, prefix: str = ""
    ) -> None:
        super().__init__()
        if quant_config is not None:
            # B/A have 48 output rows, so separately quantized checkpoint
            # shards cannot share packed block scales. Published FP8 weights
            # exclude GDN; reject other source/online layouts before loading.
            for projection in ("qkv", "z", "b", "a"):
                name = f"{prefix}.in_proj_{projection}"
                for online in (False, True) if quant_config.online_quant else (False,):
                    policy = quant_config.get_layer_quant_config(
                        name, use_online_quant=online
                    )
                    if policy.is_quantized:
                        raise ValueError(
                            "Qwen3.8-Flash-Next requires unquantized GDN input "
                            f"projections; {name} is quantized "
                            f"({'online' if online else 'source'} policy)"
                        )
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.config = config
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.num_k_heads = int(config.linear_num_key_heads)
        self.num_v_heads = int(config.linear_num_value_heads)
        self.head_k_dim = int(config.linear_key_head_dim)
        self.head_v_dim = int(config.linear_value_head_dim)
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_kernel_size = int(config.linear_conv_kernel_dim)
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.activation = config.hidden_act
        if self.num_k_heads % self.tp_size or self.num_v_heads % self.tp_size:
            raise ValueError(
                f"TP={self.tp_size} must divide the GDN heads "
                f"(k={self.num_k_heads}, v={self.num_v_heads})"
            )

        self.in_proj_qkvzba = MergedColumnParallelLinear(
            self.hidden_size,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
                self.value_dim,
                self.num_v_heads,
                self.num_v_heads,
            ],
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.in_proj_qkvzba",
        )
        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)
        delattr(self.conv1d.weight, "weight_loader")
        # The depthwise conv runs over the [q | k | v] stack, so its channels
        # shard exactly like the projection above.
        self.conv1d.weight.weight_loader = mamba_v2_sharded_weight_loader(
            [
                (self.key_dim, 0, False),
                (self.key_dim, 0, False),
                (self.value_dim, 0, False),
            ],
            self.tp_size,
            self.tp_rank,
        )

        self.dt_bias = atom_parameter(torch.ones(self.num_v_heads // self.tp_size))
        self.A_log = atom_parameter(torch.empty(self.num_v_heads // self.tp_size))
        self.norm = Qwen4ExpRMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            dtype=atom_config.torch_dtype,
        )
        self.attn = LinearAttention(
            self.hidden_size,
            self.num_v_heads,
            self.num_k_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.key_dim,
            self.value_dim,
            dt_bias=self.dt_bias,
            A_log=self.A_log,
            conv1d=self.conv1d,
            activation=self.activation,
            layer_num=extract_layer_index(prefix),
            prefix=prefix,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        v_heads = self.num_v_heads // self.tp_size
        projected = self.in_proj_qkvzba(hidden_states)
        mixed_qkv, z, b, a = projected.split(
            [
                self.conv_dim // self.tp_size,
                self.value_dim // self.tp_size,
                v_heads,
                v_heads,
            ],
            dim=-1,
        )
        z = z.view(num_tokens, v_heads, self.head_v_dim)

        core_attn_out = torch.empty_like(z)
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)
        core_attn_out = self.norm(core_attn_out, z)
        return self.out_proj(core_attn_out)


class Qwen4ExpDecoderLayer(nn.Module):
    """PLE (layer 1 only) -> HC(attn) -> attn -> HC(mlp) -> MoE.

    The two hyper-connections carry the norms this checkpoint has instead of
    `input_layernorm` / `post_attention_layernorm`, and both the attention and
    the MLP write into all four residual streams through `combine`.
    """

    def __init__(
        self,
        atom_config: Config,
        layer_type: str,
        prefix: str = "",
        layer_num: int = 0,
        quant_config=None,
    ) -> None:
        super().__init__()
        config = atom_config.hf_config
        self.layer_type = layer_type
        self.layer_idx = layer_num
        self.tp_size = get_tensor_model_parallel_world_size()

        # `ple_layer_ids` is 1-based, so [2] puts the PLE on layers.1.
        self.ple = None
        ple_layer_ids = sorted(set(getattr(config, "ple_layer_ids", []) or []))
        if (layer_num + 1) in ple_layer_ids:
            self.ple = Qwen4ExpPLELayer(
                config,
                max_total_tokens=atom_config.max_num_batched_tokens,
                max_num_reqs=atom_config.max_num_seqs,
                ple_dense_layer_id=ple_layer_ids.index(layer_num + 1),
                quant_config=quant_config,
                prefix=f"{prefix}.ple",
            )

        hc_kwargs = {
            "hidden_size": config.hidden_size,
            "hc_count": config.hc_count,
            "hc_lowrank": config.hc_lowrank,
            "eps": config.rms_norm_eps,
        }
        self.attn_hyper_connection = Qwen4ExpHyperConnection(
            **hc_kwargs, prefix=f"{prefix}.attn_hyper_connection"
        )
        self.mlp_hyper_connection = Qwen4ExpHyperConnection(
            **hc_kwargs, prefix=f"{prefix}.mlp_hyper_connection"
        )

        # The checkpoint writes "full_attention" for the layers that actually
        # run QSA; transformers' own config normalizes that to
        # "qwen_sparse_attention". Accept both so the port does not depend on
        # which of the two produced this config.
        if layer_type == "linear_attention":
            self.linear_attn = Qwen4ExpLinearAttention(
                atom_config,
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif layer_type in ("full_attention", "qwen_sparse_attention"):
            self.self_attn = Qwen4ExpAttention(
                config,
                atom_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                layer_num=layer_num,
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen4ExpSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.ple is not None:
            ple_metadata = get_forward_context().attn_metadata.ple_metadata
            # `None` only on the warmup/profiling forwards that run before the
            # state pool exists; a served token always has metadata.
            if ple_metadata is not None:
                hidden_states = hidden_states + self.ple.forward_with_state(
                    hidden_states,
                    input_ids,
                    ple_metadata,
                )
            else:
                # Profile the real embedding, projections and convolution.
                # The production pool is not allocated yet: one synthetic
                # request provides the activation/workspace peak to budget.
                starts = torch.tensor(
                    [0, hidden_states.shape[0]],
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                context = torch.full(
                    (1, self.ple.short_conv_dilation - 1),
                    self.ple.ple_embedding.eos_token_id,
                    device=hidden_states.device,
                    dtype=torch.int64,
                )
                hidden_states = hidden_states + self.ple(
                    hidden_states,
                    input_ids,
                    starts,
                    context,
                )

        mixed, residual = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            sub_output = self.linear_attn(mixed)
        else:
            sub_output = self.self_attn(positions, mixed)
        if self.tp_size > 1:
            sub_output = tensor_model_parallel_all_reduce(sub_output)
        hidden_states = self.attn_hyper_connection.combine(sub_output, residual)

        mixed, residual = self.mlp_hyper_connection.mix(hidden_states)
        # The MoE block owns its own all-reduce.
        return self.mlp_hyper_connection.combine(self.mlp(mixed), residual)


class Qwen4ExpModel(nn.Module):
    def __init__(
        self, atom_config: Config, prefix: str = "", quant_config=None
    ) -> None:
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        self.hc_count = int(config.hc_count)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Qwen4ExpDecoderLayer(
                atom_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                layer_num=layer_num,
                quant_config=quant_config,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        # PP is rejected before model construction. The final mixer only reduces
        # the residual streams; it has no block-injection projection.
        self.hyper_connection_mixer = Qwen4ExpHyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            has_block_inject=False,
            eps=config.rms_norm_eps,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_count
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds=None,
    ) -> torch.Tensor:
        """Replicate the input into HC residual streams, then run the blocks."""
        if intermediate_tensors is not None:
            raise ValueError("Qwen3.8-Flash-Next does not support pipeline parallelism")
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.get_input_embeddings(input_ids)
        ).repeat(1, self.hc_count)

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions, hidden_states, input_ids)

        mixed, _ = self.hyper_connection_mixer.mix(hidden_states)
        return mixed


class Qwen4ExpForConditionalGeneration(nn.Module):
    """Qwen3.8-Flash-Next. The MTP draft layer is skipped at load; everything else runs.

    The vision tower is only built when the engine was given a multimodal
    config, so a text-only deployment neither allocates its 0.9 GB nor pays
    for the mRoPE position cache the QSA side caches would then need.
    """

    # `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b` and the separate
    # `q_proj` / `k_proj` / `v_proj` all land on parameters of the same name or
    # on a packed one, so only the MoE and the model prefix need rewriting.
    weights_mapping: ClassVar[dict[str, str]] = {
        "model.language_model.": "model.",
        "model.visual.": "visual.",
        ".mlp.experts.gate_up_proj": ".mlp.experts.w13_weight",
        ".mlp.experts.down_proj": ".mlp.experts.w2_weight",
    }
    # Keys are dot-anchored because the QSA indexer's `index_qk_proj` contains
    # a bare "k_proj" and would otherwise be rewritten into a parameter that
    # does not exist -- a silently dropped weight rather than an error.
    # N-gram shard mappings are derived per instance from the checkpoint config.
    packed_modules_mapping: ClassVar[dict] = {
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
        "shared_expert_gate": ("gate", 1),
        ".gate.": (".gate.", 0),
    }
    # `model.visual.` is added at construction time when the tower is absent.
    skip_weight_prefixes: ClassVar[list[str]] = [
        "mtp.",  # MTP draft layer: not ported
    ]
    # The shared expert stays a standalone module: the routed experts arrive
    # as one stacked tensor with no slot to fuse it into.
    disable_fused_shared_loading: ClassVar[bool] = True

    @staticmethod
    def get_mrope_input_positions(
        atom_config: Config,
        input_tokens: list[int],
        multimodal_data: dict,
    ) -> tuple[np.ndarray | None, int]:
        """Per-request T/H/W positions for an image or video prompt.

        Qwen3.8-Flash-Next's vision token ids and spatial merge match Qwen3.5's, so the
        shared builder produces the same layout.
        """
        multimodal_config = atom_config.multimodal_config
        if multimodal_config is None or not any(
            key in multimodal_data for key in ("image_grid_thw", "video_grid_thw")
        ):
            return None, 0
        vision_config = getattr(multimodal_config, "vision_config", None)
        if vision_config is None:
            return None, 0
        from atom.models.qwen3_5 import build_qwen3_5_mrope_input_positions

        return build_qwen3_5_mrope_input_positions(
            input_tokens,
            multimodal_data.get("image_grid_thw"),
            multimodal_data.get("video_grid_thw"),
            image_token_id=int(getattr(multimodal_config, "image_token_id", 248056)),
            video_token_id=int(getattr(multimodal_config, "video_token_id", 248057)),
            vision_start_token_id=int(
                getattr(multimodal_config, "vision_start_token_id", 248053)
            ),
            vision_end_token_id=int(
                getattr(multimodal_config, "vision_end_token_id", 248054)
            ),
            spatial_merge_size=int(getattr(vision_config, "spatial_merge_size", 2)),
        )

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        from atom.model_ops.attentions.qwen4_exp_attn import Qwen4ExpBackend

        # These constraints follow the model's forward signature and kernels,
        # not a blacklist of untested serving modes in the shared Config.
        if atom_config.pipeline_parallel_size > 1:
            raise ValueError("Qwen3.8-Flash-Next does not support pipeline parallelism")
        mode = atom_config.compilation_config.cudagraph_mode
        if (
            not atom_config.enforce_eager
            and mode is not None
            and mode.requires_piecewise_compilation()
        ):
            raise ValueError("Qwen3.8-Flash-Next has no piecewise compilation support")
        if atom_config.torch_dtype != torch.bfloat16:
            raise ValueError("Qwen3.8-Flash-Next requires BF16 activations")
        if atom_config.hf_config.output_gate_type != "sigmoid":
            raise ValueError("Qwen3.8-Flash-Next requires output_gate_type=sigmoid")
        Qwen4ExpBackend.validate_config(atom_config)
        config = atom_config.hf_config
        self.config = config
        self.packed_modules_mapping = {
            **Qwen4ExpLinearAttention.packed_modules_mapping,
            **self.packed_modules_mapping,
            **{
                f".ngram_embedding.shard_{shard}.": (".ngram_embedding.", shard)
                for shard in range(int(config.split_ngram_parts))
            },
        }
        self.atom_config = atom_config
        self.quant_config = (
            _Qwen4ExpQuantizationConfig(atom_config.quant_config)
            if atom_config.quant_config is not None
            else None
        )
        multimodal_config = atom_config.multimodal_config
        if multimodal_config is not None:
            from atom.models.qwen3_5_vl import Qwen3VisionTransformer

            self.visual = Qwen3VisionTransformer(
                multimodal_config.vision_config,
                norm_eps=float(getattr(config, "rms_norm_eps", 1e-6)),
            )
            self.image_token_id = int(
                getattr(multimodal_config, "image_token_id", 248056)
            )
            self.video_token_id = int(
                getattr(multimodal_config, "video_token_id", 248057)
            )
        else:
            self.visual = None
            self.skip_weight_prefixes = [*self.skip_weight_prefixes, "model.visual."]
        self.model = Qwen4ExpModel(
            atom_config=atom_config,
            prefix=maybe_prefix(prefix, "model"),
            quant_config=self.quant_config,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        self.embed_tokens = self.model.embed_tokens
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def get_vision_embeddings(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        if self.visual is None:
            raise RuntimeError("this engine was built without a vision tower")
        return self.visual(pixel_values, grid_thw)

    def merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        vision_embeds: torch.Tensor,
    ) -> torch.Tensor:
        mask = (input_ids == self.image_token_id) | (input_ids == self.video_token_id)
        num_slots = int(mask.sum())
        if num_slots != vision_embeds.shape[0]:
            # A bare `inputs_embeds[mask] = ...` reports this as an opaque
            # broadcast error. Say which side is short: the encoder produced
            # one row per merged patch, so a mismatch means the prompt's
            # placeholder run and the image grid disagree.
            raise ValueError(
                f"vision embeddings ({vision_embeds.shape[0]}) do not match the "
                f"{num_slots} image/video placeholder tokens in this forward "
                f"({input_ids.numel()} tokens total). The encoder runs over the "
                "whole prompt, so this also fires if a multimodal prefill was "
                "chunked."
            )
        inputs_embeds[mask] = vision_embeds.to(inputs_embeds.dtype)
        return inputs_embeds

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor):
        return self.lm_head(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        """Route `experts.N.{gate,up,down}_proj.*` onto the fused parameters.

        The released FP8 checkpoint names experts individually; the internal
        BF16 one stacks them, and those names simply never match an entry here
        and fall through to the stacked loader instead. Declaring the mapping
        therefore serves both layouts, and it covers the FP8 `weight_scale`
        tensors too, since they share the projection's name.
        """
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=int(self.config.num_experts),
        )

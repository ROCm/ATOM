"""Qwen3.8-Flash-Next (`qwen3_8_flash_next` / `Qwen3_8FlashNextForConditionalGeneration`).

`qwen3_8_flash_next` is Qwen3.8-Flash-Next under a different name: the two are the same
architecture, and this port follows the reference implementation in
`/app/wef` (branch `qwen38next`, `vllm/models/qwen3_8_flash_next/`).

Shape of the model (BF16 sizes on disk, 360 GB total):

    hidden 2560, 48 layers, layer_types = 3x linear_attention + 1x QSA
      MoE            512 experts, top-10, inter 640, + shared expert   246.6 GB
      PLE @ layer 1  n-gram table 320M rows x 160, 128 shards          102.4 GB
      linear_attn    36 layers, GDN: 16 k-heads / 48 v-heads, dim 128    4.2 GB
      self_attn      12 layers, 24 q / 2 kv heads, head_dim 256          1.3 GB
                     + QSA indexer: 4 heads x 128, compress 4, budget 2048
      HyperConnection  4 residual streams, low-rank 320                  1.3 GB
      visual         27-layer ViT, 1152, patch 16, merger -> 2560        0.9 GB
      MTP            1 layer (own HC + self_attn + MoE)                  ~0 GB

Four things make it unlike anything already in ATOM:

  * **Hyper-Connections replace the residual.** There is no `input_layernorm`
    or `post_attention_layernorm`; each sub-layer is wrapped in a
    `mix`/`combine` pair over `hc_count = 4` parallel residual streams, and the
    tensor threaded between layers is the flat `[tokens, 4 * 2560]` bundle.
  * **PLE on layer 1** reads a 102 GB hashed n-gram table and adds the result
    to that bundle before the block runs -- 28% of the weights sitting under
    one `+`, feeding all 47 layers above it.
  * **QSA** replaces dense attention on all 12 full-attention layers, with
    three paged caches each.
  * **GDN** on the other 36, packed as separate `in_proj_{qkv,z,a,b}` rather
    than Qwen3-Next's fused `in_proj_qkvz` / `in_proj_ba`.

SCOPE: text and vision. The 27-layer ViT is wired up (it is structurally a
Qwen3-VL tower, so ATOM's `Qwen3VisionTransformer` runs it unchanged) and
mRoPE positions flow through attention, the QSA indexer, and the QSA
compressed-key position cache. The MTP draft layer is present in the
checkpoint and still skipped at load.
"""

from typing import ClassVar

import torch
import torch.nn.functional as F
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from torch import nn
from transformers.activations import ACT2FN

from atom.config import Config
from atom.model_ops.base_attention import LinearAttention
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import RMSNormGated
from atom.model_ops.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    RowParallelLinear,
)
from atom.model_ops.moe import FusedMoE
from atom.model_ops.qwen3_8_flash_next.hyperconnection import Qwen3_8FlashNextHyperConnection
from atom.model_ops.qwen3_8_flash_next.ple import Qwen3_8FlashNextPLELayer
from atom.model_ops.qwen3_8_flash_next.qsa_attention import Qwen3_8FlashNextAttention
from atom.model_ops.utils import atom_parameter
from atom.models.qwen3_next import mamba_v2_sharded_weight_loader
from atom.models.utils import (
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from atom.utils.forward_context import get_forward_context

# `split_ngram_parts` in the checkpoint config: the n-gram table ships as
# this many row slices.
_NGRAM_TABLE_SHARDS = 128


class Qwen3_8FlashNextGDNQKVLinear(ColumnParallelLinear):
    """`in_proj_qkv` as `[q_all | k_all | v_all]`, sharded per head.

    Qwen3-Next packs its GDN input projection group-major, which makes a plain
    contiguous tensor-parallel split land on head boundaries by accident.
    Qwen3.8-Flash-Next packs the three regions back to back instead, so each has to be
    sliced separately -- a contiguous split would cut through the middle of q
    and hand rank 1 a mixture of q and k.
    """

    def __init__(
        self,
        input_size: int,
        head_k_dim: int,
        head_v_dim: int,
        num_k_heads: int,
        num_v_heads: int,
        bias: bool = False,
        quant_config=None,
        prefix: str = "",
        **kwargs,
    ) -> None:
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        super().__init__(
            input_size,
            2 * self.key_dim + self.value_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=prefix,
        )

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        key_shard = self.key_dim // tp_size
        value_shard = self.value_dim // tp_size
        regions = (
            (0, key_shard, 0),
            (self.key_dim, key_shard, key_shard),
            (2 * self.key_dim, value_shard, 2 * key_shard),
        )
        for source_base, shard, target_base in regions:
            param.weight_loader_process(
                param.data.narrow(0, target_base, shard),
                loaded_weight.narrow(0, source_base + rank * shard, shard),
            )


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


class Qwen3_8FlashNextMLP(nn.Module):
    """Shared-expert MLP: gate_proj/up_proj [640, 2560], down_proj [2560, 640]."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Qwen3.8-Flash-Next expects a silu MLP, got {hidden_act}")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3_8FlashNextSparseMoeBlock(nn.Module):
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
        self.shared_expert = Qwen3_8FlashNextMLP(
            config.hidden_size,
            config.shared_expert_intermediate_size,
            config.hidden_act,
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
        shared = F.sigmoid(logits[:, self.n_routed_experts :]) * shared
        out = shared + routed
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out.view(orig_shape)


class Qwen3_8FlashNextLinearAttention(nn.Module):
    """Gated DeltaNet, on the 36 `linear_attention` layers.

    Same recurrence as Qwen3-Next -- ATOM's `LinearAttention` / `GatedDeltaNet`
    run it unchanged -- but the checkpoint keeps the input projections apart
    (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) where Qwen3-Next
    fuses them into `in_proj_qkvz` and `in_proj_ba`.
    """

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def __init__(self, atom_config, config, quant_config=None, prefix: str = "") -> None:
        super().__init__()
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

        self.in_proj_qkv = Qwen3_8FlashNextGDNQKVLinear(
            self.hidden_size,
            self.head_k_dim,
            self.head_v_dim,
            self.num_k_heads,
            self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_qkv",
        )
        # z, a and b are all v-head-major, so a contiguous column split is
        # already a per-head split.
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size,
            self.value_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_z",
        )
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_b",
        )
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size,
            self.num_v_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_a",
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
        setattr(
            self.conv1d.weight,
            "weight_loader",
            mamba_v2_sharded_weight_loader(
                [(self.key_dim, 0, False), (self.key_dim, 0, False), (self.value_dim, 0, False)],
                self.tp_size,
                self.tp_rank,
            ),
        )

        self.dt_bias = atom_parameter(torch.ones(self.num_v_heads // self.tp_size))
        self.A_log = atom_parameter(torch.empty(self.num_v_heads // self.tp_size))
        # `output_gate_type` is "sigmoid" for Qwen3.8-Flash-Next where Qwen3-Next
        # leaves it at SiLU; it gates BOTH this norm and the QSA attention
        # output. Getting it wrong is silent -- both are smooth (0, 1)-ish
        # gates and nothing about the shapes changes.
        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"
        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            group_size=None,
            norm_before_gate=True,
            dtype=atom_config.torch_dtype,
            quant_config=None,
            activation=output_gate_type,
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
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).view(num_tokens, v_heads, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        core_attn_out = torch.empty_like(z)
        core_attn_out = self.attn(mixed_qkv, b, a, core_attn_out)
        core_attn_out, maybe_scale = self.norm(core_attn_out, z)
        return self.out_proj(core_attn_out, x_scale=maybe_scale)


class Qwen3_8FlashNextDecoderLayer(nn.Module):
    """PLE (layer 1 only) -> HC(attn) -> attn -> HC(mlp) -> MoE.

    The two hyper-connections carry the norms this checkpoint has instead of
    `input_layernorm` / `post_attention_layernorm`, and both the attention and
    the MLP write into all four residual streams through `combine`.
    """

    def __init__(
        self, atom_config: Config, layer_type: str, prefix: str = "", layer_num: int = 0
    ) -> None:
        super().__init__()
        config = atom_config.hf_config
        quant_config = atom_config.quant_config
        self.layer_type = layer_type
        self.layer_idx = layer_num
        self.tp_size = get_tensor_model_parallel_world_size()

        # `ple_layer_ids` is 1-based, so [2] puts the PLE on layers.1.
        self.ple = None
        ple_layer_ids = sorted(set(getattr(config, "ple_layer_ids", []) or []))
        if (layer_num + 1) in ple_layer_ids:
            self.ple = Qwen3_8FlashNextPLELayer(
                config,
                max_total_tokens=atom_config.max_num_batched_tokens,
                max_num_reqs=atom_config.max_num_seqs,
                ple_dense_layer_id=ple_layer_ids.index(layer_num + 1),
                num_spec_tokens=0,
                quant_config=quant_config,
                prefix=f"{prefix}.ple",
            )

        hc_kwargs = {
            "hidden_size": config.hidden_size,
            "hc_count": config.hc_count,
            "hc_lowrank": config.hc_lowrank,
            "eps": config.rms_norm_eps,
        }
        self.attn_hyper_connection = Qwen3_8FlashNextHyperConnection(**hc_kwargs)
        self.mlp_hyper_connection = Qwen3_8FlashNextHyperConnection(**hc_kwargs)

        # The checkpoint writes "full_attention" for the layers that actually
        # run QSA; transformers' own config normalizes that to
        # "qwen_sparse_attention". Accept both so the port does not depend on
        # which of the two produced this config.
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_8FlashNextLinearAttention(
                atom_config,
                config,
                quant_config=quant_config,
                prefix=f"{prefix}.linear_attn",
            )
        elif layer_type in ("full_attention", "qwen_sparse_attention"):
            self.self_attn = Qwen3_8FlashNextAttention(
                config,
                atom_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
                layer_num=layer_num,
            )
        else:
            raise ValueError(f"Invalid layer_type {layer_type}")

        self.mlp = Qwen3_8FlashNextSparseMoeBlock(config, quant_config, prefix=f"{prefix}.mlp")

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
                    ple_metadata.conv_state,
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


class Qwen3_8FlashNextModel(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        config = atom_config.hf_config
        self.config = config
        self.hc_count = int(config.hc_count)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix, layer_num=None: Qwen3_8FlashNextDecoderLayer(
                atom_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
                layer_num=layer_num,
            ),
            prefix=f"{prefix}.layers",
            layer_num_offset=0,
        )
        # Final reduction of the four residual streams: the same parameters as
        # a layer's hyper-connection minus `block_inject_weight`, since it only
        # ever reduces.
        self.hyper_connection_mixer = Qwen3_8FlashNextHyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            hc_lowrank=config.hc_lowrank,
            has_block_inject=False,
            eps=config.rms_norm_eps,
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size * config.hc_count
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None
    ):
        hidden_states = (
            inputs_embeds
            if inputs_embeds is not None
            else self.get_input_embeddings(input_ids)
        )
        # Widen to the four residual streams by plain replication, matching the
        # reference's `hidden_states.repeat(1, hc_count)`.
        hidden_states = hidden_states.repeat(1, self.hc_count)
        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states = layer(positions, hidden_states, input_ids)
        mixed, _ = self.hyper_connection_mixer.mix(hidden_states)
        return mixed


class Qwen3_8FlashNextForConditionalGeneration(nn.Module):
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
    # `shard_N.` keys route the 128 slices of the n-gram table to the shard
    # loader, which needs the slice index the plain path cannot pass.
    packed_modules_mapping: ClassVar[dict] = {
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
        "shared_expert_gate": ("gate", 1),
        ".gate.": (".gate.", 0),
        **{
            f".ngram_embedding.shard_{shard}.": (".ngram_embedding.", shard)
            for shard in range(_NGRAM_TABLE_SHARDS)
        },
    }
    # `model.visual.` is added at construction time when the tower is absent.
    skip_weight_prefixes: ClassVar[list[str]] = [
        "mtp.",  # MTP draft layer: not ported
    ]
    # The shared expert stays a standalone module: the routed experts arrive
    # as one stacked tensor with no slot to fuse it into.
    disable_fused_shared_loading: ClassVar[bool] = True
    # The FP8 checkpoint's `modules_to_not_convert` names layers by their
    # CHECKPOINT path; these are the same prefix rewrites `weights_mapping`
    # applies to the weights themselves. Without them every excluded module
    # (all of GDN, the shared experts, PLE's projections) is built as an FP8
    # layer, its BF16 weight is loaded into it, and the FP8 weight shuffle
    # then fails on a tensor that was never quantized.
    quant_exclude_name_mapping: ClassVar[dict[str, str]] = {
        "model.language_model.": "model.",
        "model.visual.": "visual.",
    }

    @staticmethod
    def get_mrope_input_positions(
        atom_config: Config,
        input_tokens: list[int],
        multimodal_data: dict,
    ) -> tuple["np.ndarray | None", int]:
        """Per-request T/H/W positions for an image or video prompt.

        Qwen3.8-Flash-Next's vision token ids and spatial merge match Qwen3.5's, so the
        shared builder produces the same layout.
        """
        multimodal_config = atom_config.multimodal_config
        if multimodal_config is None or "image_grid_thw" not in multimodal_data:
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
        config = atom_config.hf_config
        self.config = config
        self.atom_config = atom_config
        self.quant_config = atom_config.quant_config
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
        self.model = Qwen3_8FlashNextModel(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "model")
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
        mask = (input_ids == self.image_token_id) | (
            input_ids == self.video_token_id
        )
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


# Re-exported so the parity tests can keep importing it from the model module.
__all__ = [
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextHyperConnection",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from aiter import QuantType, dtypes
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config, PretrainedConfig

from atom.config import Config, QuantizationConfig
from atom.model_ops.embed_head import (
    ParallelLMHead,
    ReplicatedEmbedding,
    VocabParallelEmbedding,
)
from atom.model_ops.fused_mtp_prologue import (
    fused_mtp_embedding_dual_rmsnorm_fp8_quant,
)
from atom.model_ops.layernorm import RMSNorm, fused_dual_rmsnorm_cat
from atom.model_ops.linear import ReplicatedLinear
from atom.model_ops.moe import FusedMoE
from atom.models.utils import IntermediateTensors
from atom.utils.decorators import support_torch_compile

from .deepseek_v2 import (
    ENABLE_ALLREDUCE_RMSNORM_FUSION,
    DeepseekV2DecoderLayer,
    _can_fuse_indexer_wk_weights_proj,
    use_replicated_vocab_embed,
)
from .utils import ckpt_has_tensor_suffix, maybe_prefix


class SharedHead(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()
        # Output norm of the MTP layer -- the draft's counterpart of the
        # backbone's final norm. Applied at the end of
        # DeepSeekMultiTokenPredictorLayer.forward rather than deferred to
        # compute_logits, so the hidden recycled into the next draft step is
        # post-final-norm, matching what the target hands draft step 0. On the
        # fused path it also absorbs mtp_block's pending all-reduce and
        # residual-add into a single kernel (mirrors Eagle3LlamaModel.norm).
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


class DeepSeekMultiTokenPredictorLayer(nn.Module):
    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_idx: int,
        alt_stream: torch.cuda.Stream | None = None,
    ) -> None:
        super().__init__()

        config = atom_config.hf_config
        self.config = config

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=maybe_prefix(prefix, "eh_proj"),
        )

        self.shared_head = SharedHead(
            config=config, prefix=prefix, quant_config=atom_config.quant_config
        )

        quant_config = atom_config.quant_config

        self.mtp_block = DeepseekV2DecoderLayer(
            prefix=prefix,
            config=self.config,
            cache_config=atom_config.kv_cache_dtype,
            quant_config=quant_config,
            layer_num=layer_idx,
            is_mtp_block=True,
            alt_stream=alt_stream,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        spec_step_index: int = 0,
        eh_input_quant: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Returns the POST-final-norm hidden of this MTP layer.

        Draft step 0 is fed the target's post-final-norm hidden, so every later
        step must be fed this. Returning the pre-norm block output instead would
        skip shared_head.norm: RMSNorm's rescale is idempotent but its
        per-channel weight is not, so the draft would consume an input the layer
        was never trained on and the error would compound down the draft chain.
        """
        if eh_input_quant is not None:
            assert inputs_embeds is None
            eh_input, eh_input_scale = eh_input_quant
            hidden_states = self.eh_proj(eh_input, x_scale=eh_input_scale)
        else:
            assert inputs_embeds is not None
            # Fused enorm(inputs_embeds) ++ hnorm(previous_hidden_states) in a
            # single Triton launch (folds the two RMSNorms + torch.cat).
            eh_input = fused_dual_rmsnorm_cat(
                inputs_embeds,
                self.enorm.weight,
                previous_hidden_states,
                self.hnorm.weight,
                self.enorm.eps,
            )
            hidden_states = self.eh_proj(eh_input)

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        # mtp_block's mlp is built with `reduce_results=not fuse_ar_input_norm`
        # (deepseek_v2.py), so it leaves an un-reduced TP partial sum exactly
        # when ENABLE_ALLREDUCE_RMSNORM_FUSION is on -- the same condition that
        # makes shared_head.norm take RMSNorm's fused branch. With the fusion
        # off the mlp already reduced and the norm just does the add. Neither
        # path needs an explicit all-reduce here, and neither double-reduces.
        #
        # On the fused branch aiter MAY collapse all-reduce + residual-add +
        # norm into one kernel, but only under its own size gate
        # (communicator_cuda.py: n <= 16384, total_bytes < 64 MiB, world_size
        # != 6) -- roughly 4681 tokens at hidden 7168 bf16. Above that, and at
        # TP=6, it falls back to all_reduce + a separate Triton RMSNorm. The
        # fallback is numerically equivalent, so only the launch count differs,
        # and the draft's step-0 prefill is usually on the fallback side of it.
        #
        # The unconditional all_reduce this replaces was wrong for
        # ENABLE_ALLREDUCE_RMSNORM_FUSION=0: the mlp had already reduced, so the
        # MTP output came out scaled by tp_size. Default is 1, hence unnoticed.
        hidden_states, _ = self.shared_head.norm(hidden_states, residual)
        return hidden_states


class DeepSeekMultiTokenPredictor(nn.Module):
    def __init__(
        self,
        *,
        atom_config: Config,
        prefix: str = "",
    ):
        super().__init__()
        config = atom_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers
        self.alt_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream()
            if torch.cuda.is_available()
            and getattr(config, "n_shared_experts", None) is not None
            else None
        )
        # to map the exact layer index from weights
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): DeepSeekMultiTokenPredictorLayer(
                    atom_config,
                    f"{prefix}.layers.{idx}",
                    layer_idx=idx,
                    alt_stream=self.alt_stream,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )
        if use_replicated_vocab_embed(config):
            # GLM-5.2 MTP: full table per rank, no post-embedding all-reduce.
            # (Shared with the target's replicated embed by EagleProposer at load.)
            self.embed_tokens = ReplicatedEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        eh_input_quant = None
        can_fuse_prologue = (
            inputs_embeds is None
            and isinstance(self.embed_tokens, ReplicatedEmbedding)
            and input_ids.ndim == 1
            and previous_hidden_states.ndim == 2
            and input_ids.shape[0] == previous_hidden_states.shape[0]
            and previous_hidden_states.is_contiguous()
            and self.embed_tokens.weight.ndim == 2
            and self.embed_tokens.weight.shape[1] == previous_hidden_states.shape[1]
            and self.embed_tokens.weight.dtype == previous_hidden_states.dtype
            and layer.enorm.weight.shape == (previous_hidden_states.shape[1],)
            and layer.hnorm.weight.shape == (previous_hidden_states.shape[1],)
            and layer.eh_proj.quant_type.value == QuantType.per_Token.value
            and layer.eh_proj.params_dtype == dtypes.fp8
            and getattr(layer.eh_proj, "input_scale", None) is None
        )
        if can_fuse_prologue:
            eh_input_quant = fused_mtp_embedding_dual_rmsnorm_fp8_quant(
                input_ids,
                self.embed_tokens.weight,
                previous_hidden_states,
                layer.enorm.weight,
                layer.hnorm.weight,
                layer.enorm.eps,
            )
        elif inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        return layer(
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
            eh_input_quant,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """``hidden_states`` is already post-final-norm (shared_head.norm runs at
        the end of DeepSeekMultiTokenPredictorLayer.forward), so this is a bare
        LM head -- norming again here would double-norm."""
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return mtp_layer.shared_head.head(hidden_states)

    def compute_draft_ids(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Greedy draft token ids via distributed argmax over the TP-sharded vocab —
        avoids all-gathering the full [N, vocab] logits every draft step.

        Feeds the same shared head as compute_logits (and, like it, takes an
        already post-final-norm ``hidden_states`` -- shared_head.norm runs at the
        end of DeepSeekMultiTokenPredictorLayer.forward), but reduces each rank's
        logit shard to (max_val, global_idx) and all-gathers only [N, 2] instead
        of the O(vocab) logits. Token-identical to
        compute_logits(...).argmax(-1).
        """
        current_step_idx = spec_step_idx % self.num_mtp_layers
        mtp_layer = self.layers[str(self.mtp_start_layer_idx + current_step_idx)]
        return mtp_layer.shared_head.head.compute_argmax_token(hidden_states, out=out)

    def set_skip_topk(self, skip: bool) -> None:
        """Toggle ``skip_topk`` on MTP sparse-attention layers.

        Used by ``EagleProposer`` for ``index_share_for_mtp_iteration``: draft
        step 0 sets ``skip=False`` (compute indexer top-k), steps 1+ set
        ``skip=True`` (reuse step 0's ``sparse_kv_indices_buffer``).
        Matches vLLM ``DeepSeekMultiTokenPredictor.set_skip_topk``.
        """
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is None:
                continue
            self_attn = getattr(mtp_block, "self_attn", None)
            if self_attn is None or not hasattr(self_attn, "skip_topk"):
                continue
            if getattr(self_attn, "indexer", None) is not None:
                self_attn.skip_topk = skip

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        """Gather sparse top-k rows at ``slot_ids`` to the front of each buffer."""
        num_slots = slot_ids.numel()
        for layer in self.layers.values():
            mtp_block = getattr(layer, "mtp_block", None)
            if mtp_block is None:
                continue
            self_attn = getattr(mtp_block, "self_attn", None)
            if self_attn is None:
                continue
            mla_attn = getattr(self_attn, "mla_attn", None)
            if mla_attn is None:
                continue
            sparse_buf = getattr(mla_attn, "sparse_kv_indices_buffer", None)
            if sparse_buf is not None and sparse_buf.numel() > 0:
                sparse_buf[:num_slots] = sparse_buf[slot_ids]


@support_torch_compile
class DeepSeekMTP(nn.Module):
    def __init__(self, atom_config: Config, prefix: str = ""):
        super().__init__()
        self.config = atom_config.hf_config

        # Several MTP checkpoints (DeepSeek R1/V3/V3.2 FP8 + the Quark mixed
        # MXFP4/FP8 variants) store eh_proj as BF16 with no weight_scale even
        # though their HF quantization_config does not list eh_proj in the
        # exclude set. Without this guard ReplicatedLinear is built with the
        # global FP8/MXFP4 spec, the BF16 weight is cast into the FP8 slot
        # against an uninitialized weight_scale, and MTP accept rate collapses.
        # GLM-FP8 ckpts already list eh_proj explicitly (this becomes a no-op);
        # GLM-5.1-MXFP4 truly quantizes eh_proj and ships weight_scale on disk
        # so the check below leaves the global spec in effect.
        if atom_config.quant_config is not None and not ckpt_has_tensor_suffix(
            atom_config.model, "eh_proj.weight_scale"
        ):
            atom_config.quant_config.apply_default_exclude_layers(["*.eh_proj"])

        if hasattr(self.config, "q_lora_rank") and self.config.q_lora_rank is not None:
            self.packed_modules_mapping = {
                "q_a_proj": ("fused_qkv_a_proj", 0),
                "kv_a_proj_with_mqa": ("fused_qkv_a_proj", 1),
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }
        else:
            self.packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj": ("gate_up_proj", 1),
            }

        model_prefix = maybe_prefix(prefix, "model")
        if hasattr(self.config, "index_topk"):
            indexer_prefixes = [
                f"{model_prefix}.layers.{idx}.self_attn.indexer"
                for idx in range(
                    self.config.num_hidden_layers,
                    self.config.num_hidden_layers
                    + self.config.num_nextn_predict_layers,
                )
            ]
            if _can_fuse_indexer_wk_weights_proj(
                self.config,
                atom_config.quant_config,
                indexer_prefixes,
            ):
                self.packed_modules_mapping.update(
                    {
                        "indexer.wk": ("indexer.wk_weights_proj", 0),
                        "indexer.weights_proj": ("indexer.wk_weights_proj", 1),
                    }
                )

        self.model = DeepSeekMultiTokenPredictor(
            atom_config=atom_config,
            prefix=model_prefix,
        )

    def remap_mtp_weight_name(self, name: str) -> str | None:
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
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Distributed greedy argmax for the MTP draft rollout (GLM-5.2).

        Every draft model implements compute_draft_ids; this one does the argmax
        distributed rather than via compute_logits().argmax(-1), so the draft
        never all-gathers the full [N, vocab] logits — only the packed [N, 2]
        per-rank reductions. Token-identical either way. See
        DeepSeekMultiTokenPredictor.compute_draft_ids.
        """
        return self.model.compute_draft_ids(hidden_states, spec_step_idx, out=out)

    def set_skip_topk(self, skip: bool) -> None:
        self.model.set_skip_topk(skip)

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        self.model.compact_topk_indices(slot_ids)

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


def get_spec_layer_idx_from_weight_name(
    config: DeepseekV2Config | DeepseekV3Config, weight_name: str
) -> int | None:
    if (
        hasattr(config, "num_nextn_predict_layers")
        and config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx+i}."):
                return layer_idx + i
    return None


def rewrite_spec_layer_name(spec_layer: int, name: str) -> str:
    """
    Rewrite the weight name to match the format of the original model.
    Add .mtp_block for modules in transformer layer block for spec layer
    and rename shared layer weights to be top level.
    """
    spec_layer_weight_names = [
        "embed_tokens",
        "enorm",
        "hnorm",
        "eh_proj",
        "shared_head",
    ]
    shared_weight_names = ["embed_tokens"]
    spec_layer_weight = False
    shared_weight = False
    for weight_name in spec_layer_weight_names:
        if weight_name in name:
            spec_layer_weight = True
            if weight_name in shared_weight_names:
                shared_weight = True
            break
    if not spec_layer_weight:
        # treat rest weights as weights for transformer layer block
        name = name.replace(
            f"model.layers.{spec_layer}.", f"model.layers.{spec_layer}.mtp_block."
        )
    elif shared_weight:
        # treat shared weights as top level weights
        name = name.replace(f"model.layers.{spec_layer}.", "model.")
    return name

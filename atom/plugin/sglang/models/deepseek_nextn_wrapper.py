"""ATOM DeepSeek NextN wrapper for SGLang external loading.

This keeps SGLang's draft architecture name (`DeepseekV3ForCausalLMNextN`)
so ModelRegistry can override the upstream implementation, but delegates the
actual draft core to ATOM's `DeepSeekMTP`.
"""

import copy
import logging
import re
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

from atom.config import QuantizationConfig as AtomQuantizationConfig
from atom.config import SpeculativeConfig
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.plugin.sglang.models.deepseek_mla import (
    setup_deepseek_for_sglang,
)
from atom.plugin.sglang.runtime import (
    SGLangPluginRuntime,
    plugin_runtime_scope,
)

logger = logging.getLogger("atom.plugin.sglang.models")


def _sync_replaced_weights() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _replace_weight(module: nn.Module, attr_name: str, weight) -> None:
    if hasattr(module, attr_name):
        delattr(module, attr_name)
    setattr(module, attr_name, weight)


def _materialize_dummy_hidden_states(
    hidden_states: torch.Tensor,
    *,
    length: int,
) -> torch.Tensor:
    shape = (length, *hidden_states.shape[1:])
    return hidden_states.new_zeros(shape)


def _set_runtime_layer_id(layer_module: nn.Module, layer_id: int) -> None:
    if hasattr(layer_module, "layer_id"):
        layer_module.layer_id = layer_id
    if hasattr(layer_module, "layer_num"):
        layer_module.layer_num = layer_id


def _retag_mtp_runtime_layer_ids(model: nn.Module) -> None:
    """Retag MTP runtime layer ids to draft-local indices.

    ATOM's DeepSeekMTP keeps checkpoint/global layer numbering (e.g. 61, 62...)
    in module prefixes so weight remapping still works. SGLang's draft KV cache,
    however, allocates layers using draft-local indices (0..num_nextn_layers-1).
    Rebind only the runtime ids used by the attention/KV-cache path.
    """

    for local_layer_id, (global_layer_id, mtp_layer) in enumerate(
        model.model.layers.items()
    ):
        mtp_block = mtp_layer.mtp_block
        self_attn = mtp_block.self_attn

        _set_runtime_layer_id(self_attn, local_layer_id)
        indexer = getattr(self_attn, "indexer", None)
        if indexer is not None:
            k_cache = getattr(indexer, "k_cache", None)
            if k_cache is not None and hasattr(k_cache, "prefix"):
                k_cache.prefix = re.sub(
                    rf"\.layers\.{re.escape(str(global_layer_id))}\.",
                    f".layers.{local_layer_id}.",
                    k_cache.prefix,
                    count=1,
                )

        for attr_name in ("mla_attn", "attn_non_absorbed", "attn_mha"):
            attn_obj = getattr(self_attn, attr_name, None)
            if attn_obj is None:
                continue
            _set_runtime_layer_id(attn_obj, local_layer_id)
            nested_attn = getattr(attn_obj, "attn", None)
            if nested_attn is not None:
                _set_runtime_layer_id(nested_attn, local_layer_id)


def _install_local_nextn_weight_remap(model: nn.Module) -> None:
    """Teach a standalone NextN checkpoint's local layer names to ATOM MTP."""

    from atom.models.deepseek_mtp import rewrite_spec_layer_name

    original_remap_mtp_weight_name = model.remap_mtp_weight_name
    config = model.config

    def remap_mtp_weight_name(name: str) -> str | None:
        num_nextn_layers = getattr(config, "num_nextn_predict_layers", 0)
        for local_idx in range(num_nextn_layers):
            local_prefix = f"model.layers.{local_idx}."
            if name.startswith(local_prefix):
                spec_layer = config.num_hidden_layers + local_idx
                global_layer_name = name.replace(
                    local_prefix,
                    f"model.layers.{spec_layer}.",
                    1,
                )
                return rewrite_spec_layer_name(spec_layer, global_layer_name)
        return original_remap_mtp_weight_name(name)

    model.remap_mtp_weight_name = remap_mtp_weight_name


class DeepseekV3ForCausalLMNextN(nn.Module):
    """SGLang-compatible draft wrapper backed by ATOM's `DeepSeekMTP`."""

    draft_model_name = "DeepSeek MTP"

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()

        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size
        self._is_glm_moe_dsa_nextn = (
            str(getattr(config, "model_type", "")).lower() == "glm_moe_dsa"
        )
        if self._is_glm_moe_dsa_nextn:
            self.draft_model_name = "GLM DSA MTP"

        with plugin_runtime_scope(framework="sglang"):
            self.atom_config = generate_atom_config_for_plugin_mode(config)

        # Draft workers need ATOM's MTP-specific config semantics rather than the
        # default target-model translation used by the generic plugin wrapper.
        server_args = get_global_server_args()
        draft_model_path = (
            server_args.speculative_draft_model_path or server_args.model_path
        )
        use_standalone_draft = (
            server_args.speculative_draft_model_path is not None
            and server_args.speculative_draft_model_path != server_args.model_path
        )
        self.use_standalone_draft = use_standalone_draft
        self.atom_config.model = draft_model_path
        if use_standalone_draft and hasattr(config, "quantization_config"):
            # Keep the target-derived structural config (num_hidden_layers=61,
            # expert counts, etc.) but use the standalone NextN checkpoint's
            # quantization metadata so FP8 attention scales are materialized.
            self.atom_config.hf_config.quantization_config = copy.deepcopy(
                config.quantization_config
            )
        if self._is_glm_moe_dsa_nextn:
            n_predict = int(
                getattr(self.atom_config.hf_config, "num_nextn_predict_layers", 1) or 1
            )
            self.atom_config.hf_config.update(
                {
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                }
            )
        else:
            SpeculativeConfig.hf_config_override(
                self.atom_config.hf_config, model_path=draft_model_path
            )
        if use_standalone_draft:
            self.atom_config.quant_config = AtomQuantizationConfig(
                self.atom_config.hf_config,
                self.atom_config.online_quant_config,
            )
        self._prepare_atom_config_for_nextn(
            config=config,
            draft_model_path=draft_model_path,
            use_standalone_draft=use_standalone_draft,
        )

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            from atom.plugin.register import (
                init_aiter_dist,
                register_ops_to_sglang,
                set_attn_cls,
            )
            from atom.models.deepseek_mtp import DeepSeekMTP

            register_ops_to_sglang(atom_config=self.atom_config)
            set_attn_cls()
            init_aiter_dist(config=self.atom_config)

            self.model = DeepSeekMTP(atom_config=self.atom_config)
            if self.use_standalone_draft:
                _install_local_nextn_weight_remap(self.model)
            self.model.atom_config = self.atom_config
            setup_deepseek_for_sglang(self.model)
            _retag_mtp_runtime_layer_ids(self.model)

        self.logits_processor = LogitsProcessor(config)
        self.lm_head = self._first_mtp_layer().shared_head.head

    def _prepare_atom_config_for_nextn(
        self,
        *,
        config,
        draft_model_path: str,
        use_standalone_draft: bool,
    ) -> None:
        del config, draft_model_path, use_standalone_draft
        if not getattr(self, "_is_glm_moe_dsa_nextn", False):
            return None

        # GLM-5.x quant configs name the DSA indexer projection as
        # `indexers_proj`; ATOM's shared DSA/MTP module uses `indexer.weights_proj`.
        # Also run the standard DeepSeek/GLM packed-module remap so excludes for
        # q_a_proj/kv_a_proj follow ATOM's fused_qkv_a_proj module in MTP blocks.
        quant_config = getattr(self.atom_config, "quant_config", None)
        if quant_config is not None:
            quant_config.remap_layer_name(
                self.atom_config.hf_config,
                quant_exclude_name_mapping={
                    "indexers_proj": "indexer.weights_proj",
                },
            )
        return None

    def _mtp_layers(self):
        return list(self.model.model.layers.values())

    def _first_mtp_layer(self):
        layers = self._mtp_layers()
        if not layers:
            raise ValueError("DeepSeekMTP does not contain any draft layers")
        return layers[0]

    def get_embed_and_head(self):
        return self.model.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        self.set_embed(embed)
        for mtp_layer in self._mtp_layers():
            _replace_weight(mtp_layer.shared_head.head, "weight", head)
        self.lm_head = self._first_mtp_layer().shared_head.head
        _sync_replaced_weights()

    def set_embed(self, embed):
        _replace_weight(self.model.model.embed_tokens, "weight", embed)
        _sync_replaced_weights()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        if forward_batch.spec_info is None:
            raise ValueError(
                f"{self.draft_model_name} draft forward requires speculative info"
            )

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            with SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
                input_embeds=input_embeds,
            ) as runtime:
                model_input_ids = runtime.input_ids
                model_input_embeds = runtime.input_embeds
                num_model_tokens = int(runtime.positions.shape[0])
                if (
                    torch.is_tensor(model_input_ids)
                    and model_input_ids.shape[0] != num_model_tokens
                ):
                    if num_model_tokens % int(model_input_ids.shape[0]) == 0:
                        model_input_ids = model_input_ids.repeat_interleave(
                            num_model_tokens // int(model_input_ids.shape[0]), dim=0
                        )
                    elif int(model_input_ids.shape[0]) == 1:
                        model_input_ids = model_input_ids.expand(num_model_tokens)
                    else:
                        raise RuntimeError(
                            f"{self.draft_model_name} draft input_ids/positions "
                            "layout mismatch: "
                            f"input_ids={tuple(model_input_ids.shape)}, "
                            f"positions={tuple(runtime.positions.shape)}"
                        )
                if (
                    torch.is_tensor(model_input_embeds)
                    and model_input_embeds.shape[0] != num_model_tokens
                ):
                    if num_model_tokens % int(model_input_embeds.shape[0]) == 0:
                        model_input_embeds = model_input_embeds.repeat_interleave(
                            num_model_tokens // int(model_input_embeds.shape[0]), dim=0
                        )
                    elif int(model_input_embeds.shape[0]) == 1:
                        model_input_embeds = model_input_embeds.expand(
                            num_model_tokens, -1
                        )
                    else:
                        raise RuntimeError(
                            f"{self.draft_model_name} draft input_embeds/positions "
                            "layout mismatch: "
                            f"input_embeds={tuple(model_input_embeds.shape)}, "
                            f"positions={tuple(runtime.positions.shape)}"
                        )

                model_hidden_states = forward_batch.spec_info.hidden_states
                if runtime.forward_batch is not forward_batch:
                    model_hidden_states = _materialize_dummy_hidden_states(
                        model_hidden_states,
                        length=num_model_tokens,
                    )
                elif (
                    torch.is_tensor(model_hidden_states)
                    and model_hidden_states.shape[0] != num_model_tokens
                ):
                    tokens_per_req = int(
                        getattr(
                            getattr(runtime.forward_batch, "spec_info", None),
                            "num_tokens_per_req",
                            0,
                        )
                        or 0
                    )
                    if (
                        tokens_per_req > 0
                        and model_hidden_states.shape[0] * tokens_per_req
                        == num_model_tokens
                    ):
                        model_hidden_states = model_hidden_states.repeat_interleave(
                            tokens_per_req, dim=0
                        )
                    elif model_hidden_states.shape[0] == 1:
                        model_hidden_states = model_hidden_states.expand(
                            num_model_tokens, -1
                        )
                    else:
                        raise RuntimeError(
                            f"{self.draft_model_name} draft-extend hidden layout "
                            "mismatch: "
                            f"hidden={tuple(model_hidden_states.shape)}, "
                            f"input_tokens={num_model_tokens}, "
                            f"tokens_per_req={tokens_per_req}"
                        )
                hidden_states = self.model(
                    input_ids=model_input_ids,
                    positions=runtime.positions,
                    hidden_states=model_hidden_states,
                    inputs_embeds=model_input_embeds,
                )

            if self.pp_group.is_last_rank:
                hidden_states = runtime.trim_output(hidden_states)
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                )
            return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        del weights
        from atom.model_loader.loader import load_model

        server_args = get_global_server_args()
        draft_model_path = (
            server_args.speculative_draft_model_path or server_args.model_path
        )
        self.atom_config.model = draft_model_path
        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            return load_model(
                model=self.model,
                model_name_or_path=draft_model_path,
                hf_config=self.atom_config.hf_config,
                load_dummy=self.atom_config.load_dummy,
                spec_decode=True,
            )


class GlmMoeDsaForCausalLMNextN(DeepseekV3ForCausalLMNextN):
    """SGLang-compatible GLM-5.2 MTP draft wrapper backed by ATOM `DeepSeekMTP`."""

    draft_model_name = "GLM DSA MTP"

    def _prepare_atom_config_for_nextn(
        self,
        *,
        config,
        draft_model_path: str,
        use_standalone_draft: bool,
    ) -> None:
        return super()._prepare_atom_config_for_nextn(
            config=config,
            draft_model_path=draft_model_path,
            use_standalone_draft=use_standalone_draft,
        )


EntryClass = [DeepseekV3ForCausalLMNextN, GlmMoeDsaForCausalLMNextN]

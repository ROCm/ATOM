"""ATOM DeepSeek NextN wrapper for SGLang external loading.

This keeps SGLang's draft architecture name (`DeepseekV3ForCausalLMNextN`)
so ModelRegistry can override the upstream implementation, but delegates the
actual draft core to ATOM's `DeepSeekMTP`.
"""

import logging
from typing import Iterable, Optional, Tuple

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import get_global_server_args

from atom.config import SpeculativeConfig
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
    setup_deepseek_for_sglang,
)
from atom.plugin.sglang.models.base_model_wrapper import (
    _current_forward_batch,
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

    for local_layer_id, mtp_layer in enumerate(model.model.layers.values()):
        mtp_block = mtp_layer.mtp_block
        self_attn = mtp_block.self_attn

        _set_runtime_layer_id(self_attn, local_layer_id)

        for attr_name in ("mla_attn", "attn_mha"):
            attn_obj = getattr(self_attn, attr_name, None)
            if attn_obj is None:
                continue
            _set_runtime_layer_id(attn_obj, local_layer_id)
            nested_attn = getattr(attn_obj, "attn", None)
            if nested_attn is not None:
                _set_runtime_layer_id(nested_attn, local_layer_id)


class DeepseekV3ForCausalLMNextN(nn.Module):
    """SGLang-compatible draft wrapper backed by ATOM's `DeepSeekMTP`."""

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

        with plugin_runtime_scope(framework="sglang"):
            self.atom_config = generate_atom_config_for_plugin_mode(config)

        # Draft workers need ATOM's MTP-specific config semantics rather than the
        # default target-model translation used by the generic plugin wrapper.
        SpeculativeConfig.hf_config_override(self.atom_config.hf_config)

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
            self.model.atom_config = self.atom_config
            setup_deepseek_for_sglang(self.model)
            _retag_mtp_runtime_layer_ids(self.model)

        self.logits_processor = LogitsProcessor(config)
        self.lm_head = self._first_mtp_layer().shared_head.head

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
        del kwargs
        if forward_batch.spec_info is None:
            raise ValueError("DeepSeek MTP draft forward requires speculative info")

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            token = _current_forward_batch.set(forward_batch)
            try:
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=forward_batch.spec_info.hidden_states,
                    inputs_embeds=input_embeds,
                )
            finally:
                _current_forward_batch.reset(token)

            if self.pp_group.is_last_rank:
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

EntryClass = [DeepseekV3ForCausalLMNextN]

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang NextN wrapper for Qwen3.8-Flash-Next MTP.

SGLang 0.5.17 rewrites known MTP families to ``*NextN`` / ``*MTP`` EntryClass
names. Flash is missing from that table, so the recognition patch maps
``Qwen4ExpForConditionalGeneration`` → ``Qwen4ExpForCausalLMNextN``. Compute
stays Native ``atom.models.qwen4_exp_mtp.Qwen4ExpMTP`` (one reusable QSA
layer, shared embed/lm_head, ``mtp.*`` weights).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch import nn

from atom.config import SpeculativeConfig
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.plugin.sglang.models.qwen4_exp import (
    _lm_positions_for_runtime,
    _sequence_positions,
    flatten_qwen4_exp_hc,
    reshape_qwen4_exp_hc,
)
from atom.plugin.sglang.qwen4_exp_bridge import qwen4_exp_metadata_from_forward_batch
from atom.plugin.sglang.runtime import (
    SGLangPluginRuntime,
    plugin_runtime_scope,
)
from atom.plugin.sglang.runtime.attention_backend_resolver import resolve_sglang_runtime

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
    hidden_states: torch.Tensor, *, length: int
) -> torch.Tensor:
    shape = (length, *hidden_states.shape[1:])
    return hidden_states.new_zeros(shape)


def _draft_rope_positions(positions: torch.Tensor) -> torch.Tensor:
    """Native draft RoPE is one ahead of the logical cache slots."""
    return positions + 1


class Qwen4ExpForCausalLMNextN(nn.Module):
    """SGLang-facing draft name; Native ``Qwen4ExpMTP`` does the work."""

    sglang_skip_quant_config = True
    packed_modules_mapping: ClassVar[dict] = {}
    skip_weight_prefixes: ClassVar[list[str]] = []
    disable_fused_shared_loading = True

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        from atom.models.qwen4_exp_mtp import Qwen4ExpMTP
        from atom.plugin.register import init_aiter_dist, register_ops_to_sglang
        from atom.plugin.sglang.models.qwen4_exp import (
            apply_prepare_qwen4_exp_adaptations,
        )

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size

        with plugin_runtime_scope(framework="sglang"):
            self.atom_config = generate_atom_config_for_plugin_mode(config)

        SpeculativeConfig.hf_config_override(
            self.atom_config.hf_config, model_path=self.atom_config.model
        )
        apply_prepare_qwen4_exp_adaptations(
            self.atom_config, "Qwen4ExpForCausalLMNextN"
        )

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            register_ops_to_sglang(atom_config=self.atom_config)
            init_aiter_dist(config=self.atom_config)
            self.model = Qwen4ExpMTP(self.atom_config)
            self.model.atom_config = self.atom_config

        self.lm_head = self.model.lm_head
        self.packed_modules_mapping = dict(
            getattr(self.model, "packed_modules_mapping", {})
        )
        self.disable_fused_shared_loading = bool(
            getattr(self.model, "disable_fused_shared_loading", True)
        )
        self.logits_processor = LogitsProcessor(
            config,
            skip_all_gather=bool(self.atom_config.enable_dp_attention),
        )
        hf = self.atom_config.hf_config
        self.hidden_size = int(getattr(hf, "hidden_size", config.hidden_size))
        self.hc_count = int(getattr(hf, "hc_count", 4) or 4)

    def get_embed_and_head(self):
        return self.model.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed, head):
        _replace_weight(self.model.model.embed_tokens, "weight", embed)
        _replace_weight(self.lm_head, "weight", head)
        _sync_replaced_weights()

    def set_embed(self, embed):
        _replace_weight(self.model.model.embed_tokens, "weight", embed)
        _sync_replaced_weights()

    def _align_spec_hidden(
        self, hidden_states: torch.Tensor, *, length: int
    ) -> torch.Tensor:
        if hidden_states.shape[0] == length:
            return hidden_states
        raise RuntimeError(
            "Flash MTP draft hidden layout mismatch: "
            f"hidden={tuple(hidden_states.shape)}, tokens={length}"
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        del kwargs
        if forward_batch.spec_info is None:
            raise ValueError("Flash MTP draft forward requires speculative info")

        seq_positions = _sequence_positions(positions)
        with (
            plugin_runtime_scope(framework="sglang", atom_config=self.atom_config),
            SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=seq_positions,
                input_ids=input_ids,
                input_embeds=input_embeds,
                set_forward_context=True,
            ) as runtime,
        ):
            from atom.utils.forward_context import get_forward_context

            ctx = get_forward_context()
            prev_is_draft = None
            if ctx.context is not None:
                prev_is_draft = bool(ctx.context.is_draft)
                ctx.context.is_draft = True

            try:
                pools = resolve_sglang_runtime(runtime.forward_batch)
                runtime.forward_batch.req_to_token_pool = pools.req_to_token_pool
                runtime.forward_batch.token_to_kv_pool = pools.token_to_kv_pool
                qwen4 = qwen4_exp_metadata_from_forward_batch(
                    self.atom_config,
                    runtime.forward_batch,
                    runtime.positions,
                    model=self.model,
                    gdn_metadata=None,
                )
                if ctx.attn_metadata is not None:
                    ctx.attn_metadata.qsa_metadata = qwen4.qsa_metadata
                    ctx.attn_metadata.ple_metadata = qwen4.ple_metadata

                model_hidden = forward_batch.spec_info.hidden_states
                if runtime.forward_batch is not forward_batch:
                    model_hidden = _materialize_dummy_hidden_states(
                        model_hidden, length=int(runtime.positions.shape[0])
                    )
                else:
                    model_hidden = self._align_spec_hidden(
                        model_hidden,
                        length=int(runtime.input_ids.shape[0]),
                    )
                model_hidden = reshape_qwen4_exp_hc(
                    model_hidden, hidden_size=self.hidden_size, hc_count=self.hc_count
                )
                rope_positions = _draft_rope_positions(
                    _lm_positions_for_runtime(runtime)
                )
                hidden = self.model(
                    runtime.input_ids,
                    rope_positions,
                    model_hidden,
                    runtime.input_embeds,
                )
            finally:
                ctx = get_forward_context()
                if ctx.context is not None and prev_is_draft is not None:
                    ctx.context.is_draft = prev_is_draft

            hidden = runtime.trim_output(hidden)

        if not self.pp_group.is_last_rank:
            return hidden
        flat = flatten_qwen4_exp_hc(hidden)
        mixed = self.model._head_input(hidden)
        return self.logits_processor(
            input_ids,
            mixed,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=flat,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        from atom.model_loader.loader import load_model

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            return load_model(
                model=self.model,
                model_name_or_path=self.atom_config.model,
                hf_config=self.atom_config.hf_config,
                load_dummy=self.atom_config.load_dummy,
                spec_decode=True,
            )


EntryClass = [Qwen4ExpForCausalLMNextN]

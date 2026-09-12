# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang EntryClass for Qwen3.8-Flash-Next / Qwen4Exp (eager text).

Native compute: atom.models.qwen3_8_flash_next (PR #2048).
Metadata: ForwardBatch → QSA + PLE bridge. GDN uses the existing SGLang GDN
context (#2067 path). Do not hang this architecture on Qwen3_5* EntryClass.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig as SGLangQuantizationConfig,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    PPProxyTensors,
)
from torch import nn

from atom.model_loader.loader import WeightsMapper, load_model_in_plugin_mode
from atom.plugin.sglang.attention_backend.attention_gdn import SGLangGDNForwardContext
from atom.plugin.sglang.qwen3_8_flash_next_bridge import (
    attach_flash_metadata,
    flash_metadata_from_forward_batch,
)
from atom.plugin.sglang.runtime import (
    SGLangForwardBatchMetadata,
    SGLangPluginRuntime,
    plugin_runtime_scope,
)

try:
    from atom.models.qwen3_8_flash_next import (
        Qwen3_8FlashNextForConditionalGeneration as _NativeFlash,
    )
except ImportError as exc:  # pragma: no cover - until Native #2048 is on the tree
    _NativeFlash = None
    _NATIVE_IMPORT_ERROR = exc
else:
    _NATIVE_IMPORT_ERROR = None


def _require_native() -> type[nn.Module]:
    if _NativeFlash is None:
        raise ImportError(
            "Qwen3.8-Flash-Next SGLang plugin needs Native "
            "atom.models.qwen3_8_flash_next from ATOM PR #2048 "
            f"(import failed: {_NATIVE_IMPORT_ERROR})"
        ) from _NATIVE_IMPORT_ERROR
    return _NativeFlash


_FLASH_HF_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.language_model.": "model.",
        "model.visual.": "visual.",
        "lm_head.": "lm_head.",
    },
)


def apply_prepare_flash_adaptations(atom_config: Any, model_arch: str) -> None:
    del model_arch
    native = _require_native()
    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is None:
        return
    quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=dict(getattr(native, "packed_modules_mapping", {})),
        weights_mapper=_FLASH_HF_MAPPER,
        quant_exclude_name_mapping=dict(
            getattr(native, "quant_exclude_name_mapping", {})
        ),
    )


class Qwen4ExpForConditionalGeneration(nn.Module):
    """SGLang-facing name matches checkpoint `architectures`."""

    sglang_skip_quant_config = True
    packed_modules_mapping = getattr(_NativeFlash, "packed_modules_mapping", {})
    hf_to_sglang_mapper = _FLASH_HF_MAPPER

    def __init__(
        self,
        config: Any,
        quant_config: SGLangQuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del prefix
        super().__init__()
        from atom.plugin.sglang.prepare import prepare_model

        native_cls = _require_native()
        atom_lm = prepare_model(config=config)
        if atom_lm is None:
            raise ValueError("ATOM failed to build Qwen3.8-Flash-Next")
        if not isinstance(atom_lm, native_cls):
            logger_name = type(atom_lm).__name__
            raise TypeError(
                "Flash plugin expected Native Qwen3_8FlashNextForConditionalGeneration, "
                f"got {logger_name}. Do not route Qwen4Exp through Qwen3_5*."
            )

        self.pp_group = get_pp_group()
        self.config = atom_lm.config
        self.atom_config = atom_lm.atom_config
        self.quant_config = quant_config or atom_lm.atom_config.quant_config
        self.model = atom_lm.model
        self.lm_head = atom_lm.lm_head
        self.make_empty_intermediate_tensors = atom_lm.make_empty_intermediate_tensors
        self.logits_processor = LogitsProcessor(
            self.config,
            skip_all_gather=bool(self.atom_config.enable_dp_attention),
        )
        self.__dict__["_atom_lm"] = atom_lm
        self.packed_modules_mapping = dict(
            getattr(native_cls, "packed_modules_mapping", {})
        )

    def get_input_embeddings(self, input_ids: torch.Tensor | None = None):
        if input_ids is None:
            return self.model.embed_tokens
        return self.model.get_input_embeddings(input_ids)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self._atom_lm.get_expert_mapping()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return load_model_in_plugin_mode(
            model=self,
            config=self.atom_config,
            prefix="",
            weights_mapper=self.hf_to_sglang_mapper,
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput | PPProxyTensors | torch.Tensor:
        del kwargs
        with (
            plugin_runtime_scope(framework="sglang", atom_config=self.atom_config),
            SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
                input_embeds=input_embeds,
                set_forward_context=True,
            ) as runtime,
        ):
            metadata = SGLangForwardBatchMetadata.build(
                runtime.forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            with SGLangGDNForwardContext.bind(metadata):
                from atom.utils.forward_context import get_forward_context

                ctx = get_forward_context()
                gdn_md = getattr(ctx.attn_metadata, "gdn_metadata", None)
                flash = flash_metadata_from_forward_batch(
                    self.atom_config,
                    runtime.forward_batch,
                    runtime.positions,
                    model=self._atom_lm,
                    input_ids=runtime.input_ids,
                    gdn_metadata=gdn_md,
                )
                if ctx.attn_metadata is not None:
                    attach_flash_metadata(
                        ctx.attn_metadata, flash.qsa_metadata, flash.ple_metadata
                    )
                hidden = self._atom_lm(
                    runtime.input_ids,
                    runtime.positions,
                    None,
                    runtime.input_embeds,
                )
            hidden = runtime.trim_output(hidden)

        if not self.pp_group.is_last_rank:
            return hidden
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)


# SGLang discovers this module's EntryClass. Checkpoint architectures field
# is Qwen4ExpForConditionalGeneration — not Qwen3_5*.
EntryClass = [Qwen4ExpForConditionalGeneration]

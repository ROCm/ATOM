# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""SGLang EntryClass for Qwen3.8-Flash-Next / Qwen4Exp.

Native compute: atom.models.qwen4_exp (PR #2048).
Metadata: ForwardBatch → QSA + PLE bridge. GDN uses the existing SGLang GDN
context (#2067 path). Vision uses Native ``get_vision_embeddings`` on prefill,
same as ATOM ``model_runner``. Do not hang this architecture on Qwen3_5*
EntryClass.
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
from atom.plugin.sglang.qwen4_exp_bridge import (
    qwen4_exp_metadata_from_forward_batch,
)
from atom.plugin.sglang.runtime import (
    SGLangForwardBatchMetadata,
    SGLangPluginRuntime,
    plugin_runtime_scope,
)

try:
    from atom.models.qwen4_exp import (
        Qwen4ExpForConditionalGeneration as _NativeQwen4Exp,
    )
except ImportError as exc:  # pragma: no cover - until Native #2048 is on the tree
    _NativeQwen4Exp = None
    _NATIVE_IMPORT_ERROR = exc
else:
    _NATIVE_IMPORT_ERROR = None


def _require_native() -> type[nn.Module]:
    if _NativeQwen4Exp is None:
        raise ImportError(
            "Qwen4Exp SGLang plugin needs Native "
            "atom.models.qwen4_exp from ATOM PR #2048 "
            f"(import failed: {_NATIVE_IMPORT_ERROR})"
        ) from _NATIVE_IMPORT_ERROR
    return _NativeQwen4Exp


_QWEN4_EXP_HF_MAPPER = WeightsMapper(
    orig_to_new_prefix={
        "model.language_model.": "model.",
        "model.visual.": "visual.",
        "lm_head.": "lm_head.",
    },
)


def apply_prepare_qwen4_exp_adaptations(atom_config: Any, model_arch: str) -> None:
    del model_arch
    native = _require_native()
    quant_config = getattr(atom_config, "quant_config", None)
    if quant_config is None:
        return

    hf = getattr(atom_config.hf_config, "text_config", None) or atom_config.hf_config
    ngram_parts = int(getattr(hf, "split_ngram_parts", 128) or 128)
    # Native GDN must see the four checkpoint shards before packing. Folding
    # them onto ``in_proj_qkvzba`` here rewrites PTPC exclude ``in_proj_b/a``
    # onto the same name as quantized ``in_proj_qkv/z``, so the layer builds
    # unquantized ``qkvzba`` and drops ``weight_scale`` (72 tensors / 36 GDN).
    packed = {
        **dict(getattr(native, "packed_modules_mapping", {})),
        **{
            f".ngram_embedding.shard_{shard}.": (".ngram_embedding.", shard)
            for shard in range(ngram_parts)
        },
    }
    quant_config.remap_layer_name(
        atom_config.hf_config,
        packed_modules_mapping=packed,
        weights_mapper=_QWEN4_EXP_HF_MAPPER,
        quant_exclude_name_mapping=dict(
            getattr(native, "quant_exclude_name_mapping", {})
        ),
    )


def _sequence_positions(positions: torch.Tensor) -> torch.Tensor:
    """1-D sequence index. QSA grouping must not see 3-row mRoPE."""
    if positions.ndim == 2 and positions.shape[0] in (1, 3):
        return positions[0]
    return positions.reshape(-1)


def _lm_positions_for_runtime(runtime: SGLangPluginRuntime) -> torch.Tensor:
    """Native QSA RoPE wants 3-row mRoPE when SGLang computed it."""
    positions = runtime.positions
    if getattr(runtime, "_is_dummy_run", False):
        return positions
    mrope = getattr(runtime.forward_batch, "mrope_positions", None)
    if not torch.is_tensor(mrope) or mrope.ndim != 2 or int(mrope.shape[0]) != 3:
        return positions
    n = int(positions.shape[-1] if positions.ndim == 2 else positions.shape[0])
    if int(mrope.shape[-1]) < n:
        return positions
    return mrope[:, :n]


def _skip_visual_prefixes(prefixes: list[str], has_visual: bool) -> list[str]:
    """Skip ``visual.*`` after the plugin mapper when Native built no tower."""
    out = list(prefixes)
    if not has_visual:
        for skip in ("model.visual.", "visual."):
            if skip not in out:
                out.append(skip)
    return out


def _cat_mm_field(items: list[Any], names: tuple[str, ...]) -> torch.Tensor:
    tensors: list[torch.Tensor] = []
    for item in items:
        value = None
        for name in names:
            value = getattr(item, name, None)
            if value is not None:
                break
        if value is None:
            continue
        tensors.append(value if torch.is_tensor(value) else torch.as_tensor(value))
    if not tensors:
        raise ValueError(f"Qwen4Exp multimodal items are missing {names[0]}")
    return torch.cat(tensors, dim=0)


def _should_embed_mm(forward_batch: ForwardBatch) -> bool:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None:
        if callable(getattr(mode, "is_decode", None)) and mode.is_decode():
            return False
        is_verify = getattr(mode, "is_target_verify", None)
        if callable(is_verify) and is_verify():
            return False
    contains = getattr(forward_batch, "contains_mm_inputs", None)
    if callable(contains):
        return bool(contains())
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    return bool(mm_inputs) and any(item is not None for item in mm_inputs)


class Qwen4ExpForConditionalGeneration(nn.Module):
    """SGLang-facing name matches checkpoint `architectures`."""

    sglang_skip_quant_config = True
    packed_modules_mapping = getattr(_NativeQwen4Exp, "packed_modules_mapping", {})
    hf_to_sglang_mapper = _QWEN4_EXP_HF_MAPPER

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
            raise ValueError("ATOM failed to build Qwen4Exp")
        if not isinstance(atom_lm, native_cls):
            logger_name = type(atom_lm).__name__
            raise TypeError(
                "Qwen4Exp plugin expected Native Qwen4ExpForConditionalGeneration, "
                f"got {logger_name}. Do not route Qwen4Exp through Qwen3_5*."
            )

        self.pp_group = get_pp_group()
        self.config = atom_lm.config
        self.atom_config = atom_lm.atom_config
        self.quant_config = quant_config or atom_lm.atom_config.quant_config
        self.model = atom_lm.model
        self.lm_head = atom_lm.lm_head
        self.make_empty_intermediate_tensors = atom_lm.make_empty_intermediate_tensors
        # Register Native vision so ``visual.*`` parameters exist after the
        # plugin mapper rewrites ``model.visual.`` → ``visual.``.
        self.visual = getattr(atom_lm, "visual", None)
        self.skip_weight_prefixes = _skip_visual_prefixes(
            list(getattr(atom_lm, "skip_weight_prefixes", ["mtp."])),
            self.visual is not None,
        )
        # Loader reads this flag on the SGLang wrapper, not Native atom_lm.
        # Flash keeps shared_expert as a standalone module: routed experts
        # arrive as one stacked tensor with no extra slot to fuse into.
        # Without this, plugin load defaults to False and rewrites
        # shared_expert.* onto experts.{n_routed_experts}.* — a missing param.
        self.disable_fused_shared_loading = bool(
            getattr(atom_lm, "disable_fused_shared_loading", True)
        )
        self.logits_processor = LogitsProcessor(
            self.config,
            skip_all_gather=bool(self.atom_config.enable_dp_attention),
        )
        self.__dict__["_atom_lm"] = atom_lm
        # Native __init__ merges GDN in_proj packing and n-gram shards onto the
        # instance mapping. The class-level dict only has QSA qkv / MoE gate_up,
        # so copying native_cls here leaves in_proj_qkvzba and PLE ngram at init.
        self.packed_modules_mapping = dict(
            getattr(
                atom_lm, "packed_modules_mapping", native_cls.packed_modules_mapping
            )
        )

    def get_input_embeddings(self, input_ids: torch.Tensor | None = None):
        if input_ids is None:
            return self.model.embed_tokens
        return self.model.get_input_embeddings(input_ids)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self._atom_lm.get_expert_mapping()

    def pad_input_ids(self, input_ids: list[int], mm_inputs: Any) -> list[int]:
        from sglang.srt.managers.mm_utils import (
            MultiModalityDataPaddingPatternMultimodalTokens,
        )

        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[Any]) -> torch.Tensor:
        return self._vision_feature(items, ("image_grid_thw", "grid_thw"))

    def get_video_feature(self, items: list[Any]) -> torch.Tensor:
        return self._vision_feature(
            items, ("video_grid_thw", "image_grid_thw", "grid_thw")
        )

    def _vision_feature(
        self, items: list[Any], grid_names: tuple[str, ...]
    ) -> torch.Tensor:
        if self.visual is None:
            raise RuntimeError("this engine was built without a vision tower")
        pixel_values = _cat_mm_field(items, ("feature", "pixel_values")).to(
            device=self.visual.device, dtype=self.visual.dtype
        )
        grid_thw = _cat_mm_field(items, grid_names).to(device=self.visual.device)
        return self._atom_lm.get_vision_embeddings(pixel_values, grid_thw)

    def _embed_qwen4_exp_mm(
        self,
        input_ids: torch.Tensor | None,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor | None:
        """Scatter Native vision embeddings onto SGLang pad_values.

        Do not call ``general_mm_embed_routine(language_model=self)``: that
        would recurse into this forward. Native ``merge_multimodal_embeddings``
        looks for image/video token ids, which ``pad_input_ids`` already
        replaced with per-image pad hashes.
        """
        if input_ids is None or not _should_embed_mm(forward_batch):
            return None
        from sglang.srt.managers.mm_utils import embed_mm_inputs

        mm_inputs = getattr(forward_batch, "mm_inputs", None) or []
        mm_inputs_list = [mm for mm in mm_inputs if mm is not None]
        if not mm_inputs_list:
            return None
        prefix_lens = list(getattr(forward_batch, "extend_prefix_lens_cpu", None) or [])
        seq_lens = list(getattr(forward_batch, "extend_seq_lens_cpu", None) or [])
        extend_prefix_lens = [
            prefix_lens[i]
            for i, mm in enumerate(mm_inputs)
            if mm is not None and i < len(prefix_lens)
        ]
        extend_seq_lens = [
            seq_lens[i]
            for i, mm in enumerate(mm_inputs)
            if mm is not None and i < len(seq_lens)
        ]
        embeds, _other = embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            input_embedding=self.model.embed_tokens,
            multimodal_model=self,
        )
        return embeds

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
            metadata = SGLangForwardBatchMetadata.build(
                runtime.forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
            )
            with SGLangGDNForwardContext.bind(metadata):
                from atom.utils.forward_context import get_forward_context

                ctx = get_forward_context()
                gdn_md = getattr(ctx.attn_metadata, "gdn_metadata", None)
                qwen4 = qwen4_exp_metadata_from_forward_batch(
                    self.atom_config,
                    runtime.forward_batch,
                    runtime.positions,
                    model=self._atom_lm,
                    gdn_metadata=gdn_md,
                )
                if ctx.attn_metadata is not None:
                    ctx.attn_metadata.qsa_metadata = qwen4.qsa_metadata
                    ctx.attn_metadata.ple_metadata = qwen4.ple_metadata
                mm_embeds = runtime.input_embeds
                if mm_embeds is None and not getattr(runtime, "_is_dummy_run", False):
                    mm_embeds = self._embed_qwen4_exp_mm(
                        runtime.input_ids, runtime.forward_batch
                    )
                hidden = self._atom_lm(
                    runtime.input_ids,
                    _lm_positions_for_runtime(runtime),
                    None,
                    mm_embeds,
                )
            hidden = runtime.trim_output(hidden)

        if not self.pp_group.is_last_rank:
            return hidden
        return self.logits_processor(input_ids, hidden, self.lm_head, forward_batch)


# SGLang discovers this module's EntryClass. Checkpoint architectures field
# is Qwen4ExpForConditionalGeneration — not Qwen3_5*.
EntryClass = [Qwen4ExpForConditionalGeneration]

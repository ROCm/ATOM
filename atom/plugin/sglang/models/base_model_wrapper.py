"""ATOM model wrappers for SGLang external model loading.

Registers model architecture classes via SGLANG_EXTERNAL_MODEL_PACKAGE,
replacing sglang's built-in implementations with ATOM-optimized versions.
"""

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, ClassVar, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger("atom.plugin.sglang.models")


_current_forward_batch: ContextVar[Optional[ForwardBatch]] = ContextVar(
    "atom_sglang_current_forward_batch", default=None
)


def get_current_forward_batch():
    return _current_forward_batch.get()


@dataclass(frozen=True)
class SGLangForwardBatchMetadata:
    """Small context object for one SGLang model forward."""

    forward_batch: Optional[ForwardBatch]
    pp_proxy_tensors: Optional[PPProxyTensors] = None
    save_kv_cache: bool = True
    _current: ClassVar[ContextVar[Optional["SGLangForwardBatchMetadata"]]] = ContextVar(
        "atom_sglang_current_forward_batch_metadata",
        default=None,
    )

    @classmethod
    def current(cls) -> Optional["SGLangForwardBatchMetadata"]:
        return cls._current.get()

    @classmethod
    def build(
        cls,
        forward_batch: Optional[
            Union[ForwardBatch, "SGLangForwardBatchMetadata"]
        ] = None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        save_kv_cache: Optional[bool] = None,
    ) -> Optional["SGLangForwardBatchMetadata"]:
        if isinstance(forward_batch, cls):
            return forward_batch
        if forward_batch is None and pp_proxy_tensors is None and save_kv_cache is None:
            return cls.current()
        return cls(
            forward_batch=forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=True if save_kv_cache is None else save_kv_cache,
        )

    @classmethod
    @contextmanager
    def bind(cls, metadata: Optional["SGLangForwardBatchMetadata"]):
        meta_token = cls._current.set(metadata)
        batch_token = _current_forward_batch.set(
            None if metadata is None else metadata.forward_batch
        )
        try:
            yield metadata
        finally:
            _current_forward_batch.reset(batch_token)
            cls._current.reset(meta_token)

    @staticmethod
    def to_intermediate_tensors(
        intermediate_tensors,
        metadata: Optional["SGLangForwardBatchMetadata"],
    ):
        if intermediate_tensors is not None or metadata is None:
            return intermediate_tensors
        pp_proxy_tensors = metadata.pp_proxy_tensors
        if pp_proxy_tensors is None:
            return intermediate_tensors
        tensors = getattr(pp_proxy_tensors, "tensors", None)
        if tensors is None:
            return intermediate_tensors
        from atom.models.utils import IntermediateTensors

        return IntermediateTensors(dict(tensors))


@dataclass(frozen=True)
class ModelArchSpec:
    wrapper_binds_gdn_context: bool = False
    apply_deepseek_patch: bool = False


_MODEL_ARCH_SPECS = {
    "DeepseekV3ForCausalLM": ModelArchSpec(apply_deepseek_patch=True),
    "Qwen3MoeForCausalLM": ModelArchSpec(),
    "Qwen3NextForCausalLM": ModelArchSpec(wrapper_binds_gdn_context=True),
}


class _AtomCausalLMBaseForSglang(nn.Module):
    """Base ATOM model wrapper conforming to sglang's model interface.

    Delegates model creation and weight loading to ATOM's plugin system,
    while providing the forward signature and LogitsProcessorOutput return
    type that sglang expects.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        logger.info("Initializing ATOM backend for %s", self.__class__.__name__)

        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.config = config
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size
        self.model_arch = getattr(config, "architectures", [""])[0]
        self.model_arch_spec = _MODEL_ARCH_SPECS.get(self.model_arch, ModelArchSpec())

        import atom

        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            raise ValueError(
                f"ATOM failed to create model for architecture {self.model_arch}"
            )

        self.logits_processor = LogitsProcessor(config)

        # Apply ds model-specific sglang patches (attn dispatch, weight hooks, etc.)
        # TODO: will remove this after sglang supports atom attention backend
        if self.model_arch_spec.apply_deepseek_patch:
            from atom.plugin.sglang.attention_backend.sgl_attention_mla import (
                setup_deepseek_for_sglang,
            )

            setup_deepseek_for_sglang(self.model)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        **model_kwargs: Any,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        metadata = SGLangForwardBatchMetadata.build(
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            save_kv_cache=model_kwargs.get("save_kv_cache"),
        )
        model_inputs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=SGLangForwardBatchMetadata.to_intermediate_tensors(
                pp_proxy_tensors, metadata
            ),
            inputs_embeds=input_embeds,
        )
        with SGLangForwardBatchMetadata.bind(metadata):
            if self.model_arch_spec.wrapper_binds_gdn_context:
                from atom.plugin.sglang.attention_backend.attention_gdn import (
                    SGLangGDNForwardContext,
                )

                with SGLangGDNForwardContext.bind(metadata):
                    hidden_states = self.model(**model_inputs)
            else:
                hidden_states = self.model(**model_inputs)

        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids,
                hidden_states,
                self.model.lm_head,
                forward_batch,
            )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The passed `weights` iterable from sglang is ignored because ATOM
        # uses its own weight loading pipeline (handling AITER-specific quant
        # formats, kv_b_proj splitting, etc.) that is incompatible with
        # sglang's default weight iterator.
        from atom.model_loader.loader import load_model_in_plugin_mode

        return load_model_in_plugin_mode(
            model=self.model, config=self.model.atom_config, prefix="model."
        )


EntryClass = []
for _name in _MODEL_ARCH_SPECS:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)

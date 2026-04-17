"""ATOM model wrappers for SGLang external model loading.

Registers model architecture classes via SGLANG_EXTERNAL_MODEL_PACKAGE,
replacing sglang's built-in implementations with ATOM-optimized versions.

To add a new model, append its architecture class name to _MODEL_NAMES.
"""

import logging
from contextvars import ContextVar
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

logger = logging.getLogger("atom.plugin.sglang.models")

# Context for patched DeepSeek attention layers that need wrapper state without
# changing every intermediate forward signature. ContextVar keeps nested or
# concurrent forwards isolated and lets us reliably restore the prior value.
_current_forward_batch: ContextVar[Optional[ForwardBatch]] = ContextVar(
    "atom_sglang_current_forward_batch", default=None
)


def get_current_forward_batch():
    return _current_forward_batch.get()


_MODEL_NAMES = [
    "DeepseekV3ForCausalLM",
    "Qwen3MoeForCausalLM",
]

_DEEPSEEK_ARCHS = {
    "DeepseekV3ForCausalLM",
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

        import atom

        # TODO: prepare_model() currently handles model construction, config
        # generation, attention backend registration, and distributed init.
        # Refactor so this wrapper only dispatches the attention backend
        # (register_ops_to_sglang + set_attn_cls), and let sglang handle
        # model construction directly
        self.model = atom.prepare_model(config=config, engine="sglang")
        if self.model is None:
            model_arch = getattr(config, "architectures", ["unknown"])[0]
            raise ValueError(
                f"ATOM failed to create model for architecture {model_arch}"
            )

        self.logits_processor = LogitsProcessor(config)

        # Apply ds model-specific sglang patches (attn dispatch, weight hooks, etc.)
        # TODO: will remove this after sglang supports atom attention backend
        arch = getattr(config, "architectures", [""])[0]
        self._uses_forward_batch_context = arch in _DEEPSEEK_ARCHS
        if arch in _DEEPSEEK_ARCHS:
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
        model_inputs = dict(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=pp_proxy_tensors,
            inputs_embeds=input_embeds,
        )
        if self._uses_forward_batch_context:
            token = _current_forward_batch.set(forward_batch)
            try:
                hidden_states = self.model(**model_inputs)
            finally:
                _current_forward_batch.reset(token)
        else:
            hidden_states = self.model(
                **model_inputs,
                forward_batch=forward_batch,
                get_embedding=get_embedding,
                pp_proxy_tensors=pp_proxy_tensors,
                **model_kwargs,
            )

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
for _name in _MODEL_NAMES:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)

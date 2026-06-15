"""ATOM model wrappers for SGLang external model loading.

Registers model architecture classes via SGLANG_EXTERNAL_MODEL_PACKAGE,
replacing sglang's built-in implementations with ATOM-optimized versions.

To add a new model, append its architecture class name to _MODEL_NAMES.
"""

import inspect
import logging
from typing import Any, Iterable, Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from atom.plugin.sglang.runtime import (
    MODEL_ARCH_SPECS,
    SGLangForwardBatchMetadata,
    SGLangPluginRuntime,
    bind_current_forward_batch,
    get_current_forward_batch,
    get_model_arch_spec,
    plugin_runtime_scope,
)

logger = logging.getLogger("atom.plugin.sglang.models")

__all__ = [
    "EntryClass",
    "SGLangForwardBatchMetadata",
    "SGLangPluginRuntime",
    "bind_current_forward_batch",
    "get_current_forward_batch",
    "plugin_runtime_scope",
]


class _ComputeLogitsHeadAdapter(nn.Module):
    """Expose ATOM `compute_logits` through SGLang's lm_head call contract."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def set_lora(self, *args: Any, **kwargs: Any) -> None:
        return None

    def apply_lora(self, *args: Any, **kwargs: Any) -> None:
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)


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
        self.model_arch_spec = get_model_arch_spec(self.model_arch)

        with plugin_runtime_scope(framework="sglang"):
            from atom.config import get_current_atom_config
            from atom.plugin.sglang.prepare import prepare_model

            self.model = prepare_model(config=config)
            self.atom_config = getattr(self.model, "atom_config", None)
            if self.atom_config is None:
                self.atom_config = get_current_atom_config()
                self.model.atom_config = self.atom_config
        # SGLang's loader invokes some quantization post-load hooks after
        # returning from this constructor/load_weights scope. Keep the
        # process-local ATOM config available, matching native model_runner.
        from atom.config import set_current_atom_config

        set_current_atom_config(self.atom_config)
        if self.model is None:
            raise ValueError(
                f"ATOM failed to create model for architecture {self.model_arch}"
            )

        if hasattr(self.model, "lm_head"):
            self.logits_head = self.model.lm_head
            logits_head_handles_all_gather = False
        elif hasattr(self.model, "compute_logits"):
            self.logits_head = _ComputeLogitsHeadAdapter(self.model)
            logits_head_handles_all_gather = True
        else:
            raise AttributeError(
                f"ATOM model {type(self.model).__name__} must define lm_head "
                "or compute_logits for SGLang logits processing"
            )

        # Under SGLang dp-attention, ATOM runtime interprets non-MoE modules
        # like lm_head with tp=1 semantics, so plugin logits must not perform
        # an extra TP all-gather after local lm_head matmul.
        plugin_skip_all_gather = bool(
            self.model.atom_config.enable_dp_attention or logits_head_handles_all_gather
        )
        self.logits_processor = LogitsProcessor(
            config, skip_all_gather=plugin_skip_all_gather
        )

        # Apply model-specific install-time adapters (attn dispatch, weight hooks, etc.).
        if self.model_arch_spec.install_adapters is not None:
            with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
                self.model_arch_spec.install_adapters(self.model)

    def _filter_model_forward_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Drop SGLang wrapper kwargs that the ATOM model forward does not accept."""
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return kwargs

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        ):
            return kwargs

        return {key: value for key, value in kwargs.items() if key in params}

    def get_embed_and_head(self):
        if hasattr(self.model, "get_embed_and_head"):
            return self.model.get_embed_and_head()

        embed_owner = (
            self.model.model
            if hasattr(self.model, "model")
            and hasattr(self.model.model, "embed_tokens")
            else self.model
        )
        return embed_owner.embed_tokens.weight, self.model.lm_head.weight

    def set_embed_and_head(self, embed, head):
        if hasattr(self.model, "set_embed_and_head"):
            return self.model.set_embed_and_head(embed, head)

        embed_owner = (
            self.model.model
            if hasattr(self.model, "model")
            and hasattr(self.model.model, "embed_tokens")
            else self.model
        )
        del embed_owner.embed_tokens.weight
        del self.model.lm_head.weight
        embed_owner.embed_tokens.weight = embed
        self.model.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_embed(self, embed):
        if hasattr(self.model, "set_embed"):
            return self.model.set_embed(embed)

        embed_owner = (
            self.model.model
            if hasattr(self.model, "model")
            and hasattr(self.model.model, "embed_tokens")
            else self.model
        )
        del embed_owner.embed_tokens.weight
        embed_owner.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

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
        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            with SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
                input_embeds=input_embeds,
                set_forward_context=not self.model_arch_spec.wrapper_binds_gdn_context,
            ) as runtime:
                if self.model_arch == "DeepseekV4ForCausalLM":
                    from atom.plugin.sglang.deepseek_v4_bridge import (
                        bind_deepseek_v4_proxy_cache_views,
                        maybe_get_proxy_pool_from_sglang_backend,
                        reset_deepseek_v4_state_slots,
                    )

                    proxy_pool, _ = maybe_get_proxy_pool_from_sglang_backend()
                    if not bind_deepseek_v4_proxy_cache_views(self.model, proxy_pool):
                        raise RuntimeError(
                            "DeepSeek-V4 SGLang proxy KV pool is not initialized"
                        )
                    from atom.utils.forward_context import get_forward_context

                    reset_slots = getattr(
                        get_forward_context().attn_metadata, "reset_slots", None
                    )
                    reset_deepseek_v4_state_slots(self.model, reset_slots)

                metadata = SGLangForwardBatchMetadata.build(
                    runtime.forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                    save_kv_cache=model_kwargs.get("save_kv_cache"),
                )
                model_inputs = dict(
                    input_ids=runtime.input_ids,
                    positions=runtime.positions,
                    intermediate_tensors=SGLangForwardBatchMetadata.to_intermediate_tensors(
                        pp_proxy_tensors, metadata
                    ),
                    inputs_embeds=runtime.input_embeds,
                )

                with SGLangForwardBatchMetadata.bind(metadata):
                    if self.model_arch_spec.wrapper_binds_gdn_context:
                        from atom.plugin.sglang.attention_backend.attention_gdn import (
                            SGLangGDNForwardContext,
                        )

                        with SGLangGDNForwardContext.bind(metadata):
                            hidden_states = self.model(
                                **self._filter_model_forward_kwargs(model_inputs)
                            )
                    elif self.model_arch_spec.uses_context_only_forward:
                        hidden_states = self.model(
                            **self._filter_model_forward_kwargs(model_inputs)
                        )
                    else:
                        model_call_kwargs = dict(
                            model_inputs,
                            forward_batch=runtime.forward_batch,
                            get_embedding=get_embedding,
                            pp_proxy_tensors=pp_proxy_tensors,
                        )
                        model_call_kwargs.update(model_kwargs)
                        hidden_states = self.model(
                            **self._filter_model_forward_kwargs(model_call_kwargs)
                        )

                hidden_states = runtime.trim_output(hidden_states)

                if self.pp_group.is_last_rank:
                    if self.model_arch == "DeepseekV4ForCausalLM" and not getattr(
                        forward_batch, "return_logprob", False
                    ):
                        if forward_batch.forward_mode.is_decode_or_idle():
                            pruned_states = hidden_states
                        elif forward_batch.forward_mode.is_extend():
                            last_index = (
                                torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
                            )
                            pruned_states = hidden_states[last_index]
                        else:
                            pruned_states = hidden_states
                        return LogitsProcessorOutput(
                            next_token_logits=self.model.compute_logits(pruned_states)
                        )
                    return self.logits_processor(
                        input_ids,
                        hidden_states,
                        self.logits_head,
                        forward_batch,
                    )
                return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The passed `weights` iterable from sglang is ignored because ATOM
        # uses its own weight loading pipeline (handling AITER-specific quant
        # formats, kv_b_proj splitting, etc.) that is incompatible with
        # sglang's default weight iterator.
        from atom.model_loader.loader import load_model_in_plugin_mode

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            return load_model_in_plugin_mode(
                model=self.model, config=self.atom_config, prefix="model."
            )


EntryClass = []
for _name in MODEL_ARCH_SPECS:
    _cls = type(_name, (_AtomCausalLMBaseForSglang,), {})
    globals()[_name] = _cls
    EntryClass.append(_cls)

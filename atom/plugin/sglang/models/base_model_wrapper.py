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

    # ATOM owns checkpoint quantization parsing, weight allocation, loading, and
    # post-load processing for these external models. SGLang should not build
    # its native quant_config from HF `quantization_config` first.
    sglang_skip_quant_config = True

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
        self.capture_aux_hidden_states = False

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

        if self.model_arch == "LlamaForCausalLMEagle3" and hasattr(
            self.model, "compute_logits"
        ):
            self.logits_head = _ComputeLogitsHeadAdapter(self.model)
            logits_head_handles_all_gather = True
        elif hasattr(self.model, "lm_head"):
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
        self.load_lm_head_from_target = getattr(
            self.model, "load_lm_head_from_target", False
        )
        self.hot_token_id = getattr(self.model, "hot_token_id", None)

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

        if self.model_arch == "DeepseekV4ForCausalLM":
            return self.model.model.embed.weight, self.model.model.head.weight

        embed_owner, head_owner = self._embed_and_head_owners()
        return embed_owner.embed_tokens.weight, head_owner.lm_head.weight

    def _embed_and_head_owners(self):
        if hasattr(self.model, "language_model"):
            language_model = self.model.language_model
            embed_owner = (
                language_model.model
                if hasattr(language_model, "model")
                and hasattr(language_model.model, "embed_tokens")
                else language_model
            )
            return embed_owner, language_model

        embed_owner = (
            self.model.model
            if hasattr(self.model, "model")
            and hasattr(self.model.model, "embed_tokens")
            else self.model
        )
        return embed_owner, self.model

    def set_embed_and_head(self, embed, head):
        if hasattr(self.model, "set_embed_and_head"):
            return self.model.set_embed_and_head(embed, head)
        if self.model_arch == "LlamaForCausalLMEagle3":
            logger.info(
                "Skip sharing target embed/lm_head for ATOM EAGLE3 draft; "
                "the draft checkpoint owns independent weights."
            )
            return None

        embed_owner, head_owner = self._embed_and_head_owners()
        del embed_owner.embed_tokens.weight
        del head_owner.lm_head.weight
        embed_owner.embed_tokens.weight = embed
        head_owner.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_embed(self, embed):
        if hasattr(self.model, "set_embed"):
            return self.model.set_embed(embed)
        if self.model_arch == "LlamaForCausalLMEagle3":
            logger.info(
                "Skip sharing target embedding for ATOM EAGLE3 draft; "
                "the draft checkpoint owns independent embedding."
            )
            return None

        embed_owner, _ = self._embed_and_head_owners()
        del embed_owner.embed_tokens.weight
        embed_owner.embed_tokens.weight = embed
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[Iterable[int]] = None):
        self.capture_aux_hidden_states = True
        if layer_ids is None:
            get_default_layers = getattr(
                self.model, "get_eagle3_aux_hidden_state_layers", None
            )
            if get_default_layers is None:
                raise AttributeError(
                    f"ATOM model {type(self.model).__name__} does not define "
                    "get_eagle3_aux_hidden_state_layers"
                )
            layer_ids = get_default_layers()

        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if hasattr(self.model, "set_eagle3_layers_to_capture"):
            return self.model.set_eagle3_layers_to_capture(layer_ids)
        if hasattr(self.model, "set_aux_hidden_state_layers"):
            return self.model.set_aux_hidden_state_layers(layer_ids)
        raise AttributeError(
            f"ATOM model {type(self.model).__name__} does not support "
            "EAGLE3 auxiliary hidden-state capture"
        )

    def _split_aux_hidden_states(self, output):
        if (
            isinstance(output, tuple)
            and len(output) == 2
            and (torch.is_tensor(output[0]) or hasattr(output[0], "tensors"))
        ):
            return output[0], output[1]
        return output, None

    def _trim_aux_hidden_states(self, runtime, aux_hidden_states):
        if aux_hidden_states is None:
            return None
        if torch.is_tensor(aux_hidden_states):
            return runtime.trim_output(aux_hidden_states)
        if isinstance(aux_hidden_states, list):
            return [
                runtime.trim_output(aux_hidden_state)
                for aux_hidden_state in aux_hidden_states
            ]
        return aux_hidden_states

    def _forward_eagle3_draft_model(self, input_ids, positions, forward_batch):
        spec_info = getattr(forward_batch, "spec_info", None)
        hidden_states = getattr(spec_info, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("EAGLE3 draft forward requires spec_info.hidden_states")
        if hidden_states.shape[-1] != self.model.config.hidden_size:
            hidden_states = self.model.combine_hidden_states(hidden_states)
        return self.model(
            input_ids=input_ids, positions=positions, hidden_states=hidden_states
        )

    def _load_eagle3_token_map(self, draft_path: str) -> None:
        """Load SGLang EAGLE3 non-parameter token-map tensors in plugin scope."""
        from atom.model_loader.loader import safetensors_weights_iterator

        for name, weight_tensor in safetensors_weights_iterator(draft_path):
            if "d2t" in name:
                base = torch.arange(
                    weight_tensor.shape[0],
                    dtype=weight_tensor.dtype,
                    device=weight_tensor.device,
                )
                self.hot_token_id = (weight_tensor + base).to(torch.int64)
                setattr(self.model, "hot_token_id", self.hot_token_id)
                logger.info(
                    "Loaded EAGLE3 draft token map %s with %d entries",
                    name,
                    self.hot_token_id.numel(),
                )
                return

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
                    if self.model_arch == "LlamaForCausalLMEagle3":
                        hidden_states = self._forward_eagle3_draft_model(
                            runtime.input_ids,
                            runtime.positions,
                            runtime.forward_batch,
                        )
                    elif self.model_arch_spec.wrapper_binds_gdn_context:
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

                hidden_states, aux_hidden_states = self._split_aux_hidden_states(
                    hidden_states
                )
                hidden_states = runtime.trim_output(hidden_states)
                aux_hidden_states = self._trim_aux_hidden_states(
                    runtime, aux_hidden_states
                )

                if self.pp_group.is_last_rank:
                    if self.model_arch == "DeepseekV4ForCausalLM":
                        return self.logits_processor(
                            input_ids,
                            hidden_states,
                            self.logits_head,
                            forward_batch,
                            aux_hidden_states=aux_hidden_states,
                            hidden_states_before_norm=hidden_states,
                        )
                    return self.logits_processor(
                        input_ids,
                        hidden_states,
                        self.logits_head,
                        forward_batch,
                        aux_hidden_states=aux_hidden_states,
                    )
                return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The passed `weights` iterable from sglang is ignored because ATOM
        # uses its own weight loading pipeline (handling AITER-specific quant
        # formats, kv_b_proj splitting, etc.) that is incompatible with
        # sglang's default weight iterator.
        if self.model_arch == "LlamaForCausalLMEagle3":
            from atom.model_loader.loader import load_model

            draft_path = None
            try:
                from sglang.srt.server_args import get_global_server_args

                server_args = get_global_server_args()
                draft_path = getattr(server_args, "speculative_draft_model_path", None)
            except Exception:
                logger.exception("Failed to resolve SGLang EAGLE3 draft model path")
            draft_path = draft_path or getattr(self.config, "_name_or_path", None)
            draft_path = draft_path or getattr(self.config, "name_or_path", None)
            if not draft_path:
                raise RuntimeError("Cannot resolve EAGLE3 draft model path")
            logger.info("Loading ATOM EAGLE3 draft weights from %s", draft_path)
            self.atom_config.model = draft_path
            self.atom_config.hf_config = self.config
            with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
                result = load_model(
                    model=self.model,
                    model_name_or_path=draft_path,
                    hf_config=self.config,
                    load_dummy=self.atom_config.load_dummy,
                    prefix="",
                    is_plugin_mode=True,
                )
            self._load_eagle3_token_map(draft_path)
            return result

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

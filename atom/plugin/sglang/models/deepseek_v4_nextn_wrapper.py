"""ATOM DeepSeek-V4 NextN wrapper for SGLang external loading.

SGLang rewrites a DeepSeek-V4 draft runner to the architecture name
``DeepseekV4ForCausalLMNextN``.  This wrapper keeps that public name while
delegating the actual MTP block to ATOM's ``DeepseekV4MTP`` implementation.
"""

import copy
import logging
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
from atom.model_ops.embed_head import VocabParallelEmbedding
from atom.models.deepseek_v4 import DeepseekV4Attention, ParallelHead
from atom.plugin.config import generate_atom_config_for_plugin_mode
from atom.plugin.sglang.runtime import (
    SGLangForwardBatchMetadata,
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


def _reshape_mtp_hidden_states(hidden_states: torch.Tensor, *, hidden_size: int):
    if hidden_states is None:
        return None
    if hidden_states.dim() == 3:
        return hidden_states
    if hidden_states.dim() != 2:
        raise ValueError(
            "DeepSeek-V4 MTP hidden_states must be rank-2 flattened or rank-3 "
            f"mHC, got shape={tuple(hidden_states.shape)}"
        )
    width = int(hidden_states.shape[-1])
    if width == hidden_size:
        return hidden_states.unsqueeze(1)
    if width % hidden_size != 0:
        raise ValueError(
            "DeepSeek-V4 MTP flattened hidden width must be divisible by "
            f"hidden_size={hidden_size}, got width={width}"
        )
    return hidden_states.view(hidden_states.shape[0], width // hidden_size, hidden_size)


def _install_deepseek_v4_mtp_adapters(model: nn.Module) -> None:
    from atom.plugin.sglang.models.deepseek_v4_attention import (
        patch_deepseek_v4_attention_for_sglang,
    )

    for module in model.modules():
        if isinstance(module, DeepseekV4Attention):
            patch_deepseek_v4_attention_for_sglang(module)


class _DeepseekV4MTPLogitsHeadAdapter(nn.Module):
    """Expose ``DeepseekV4MTP.compute_logits`` as an SGLang lm_head."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def set_lora(self, *args, **kwargs) -> None:
        return None

    def apply_lora(self, *args, **kwargs) -> None:
        return None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)


class DeepseekV4ForCausalLMNextN(nn.Module):
    """SGLang-compatible draft wrapper backed by ATOM's ``DeepseekV4MTP``."""

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
            self.atom_config.hf_config.quantization_config = copy.deepcopy(
                config.quantization_config
            )
        SpeculativeConfig.hf_config_override(
            self.atom_config.hf_config, model_path=draft_model_path
        )
        if use_standalone_draft:
            self.atom_config.quant_config = AtomQuantizationConfig(
                self.atom_config.hf_config,
                self.atom_config.online_quant_config,
            )

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            from atom.models.deepseek_v4_mtp import DeepseekV4MTP
            from atom.plugin.register import (
                init_aiter_dist,
                register_ops_to_sglang,
                set_attn_cls,
            )

            register_ops_to_sglang(atom_config=self.atom_config)
            set_attn_cls()
            init_aiter_dist(config=self.atom_config)

            self.model = DeepseekV4MTP(config=self.atom_config)
            self.model.atom_config = self.atom_config
            _install_deepseek_v4_mtp_adapters(self.model)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        self.shared_head = ParallelHead(
            config.vocab_size,
            config.hidden_size,
            norm_eps=getattr(config, "rms_norm_eps", 1e-6),
            hc_eps=getattr(config, "hc_eps", 1e-6),
        )
        self._bind_shared_modules()
        self.logits_head = _DeepseekV4MTPLogitsHeadAdapter(self.model)
        self.logits_processor = LogitsProcessor(config, skip_all_gather=True)

    def _mtp_blocks(self):
        return list(self.model.model.mtp)

    def _bind_shared_modules(self) -> None:
        for block in self._mtp_blocks():
            block.embed = self.embed_tokens
            block.head = self.shared_head

    def get_embed_and_head(self):
        return self.embed_tokens.weight, self.shared_head.weight

    def set_embed_and_head(self, embed, head):
        self.set_embed(embed)
        _replace_weight(self.shared_head, "weight", head)
        self._bind_shared_modules()
        _sync_replaced_weights()

    def set_embed(self, embed):
        _replace_weight(self.embed_tokens, "weight", embed)
        self._bind_shared_modules()
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
        del input_embeds, kwargs
        if forward_batch.spec_info is None:
            raise ValueError("DeepSeek-V4 MTP draft forward requires speculative info")

        with plugin_runtime_scope(framework="sglang", atom_config=self.atom_config):
            with SGLangPluginRuntime(
                atom_config=self.atom_config,
                forward_batch=forward_batch,
                positions=positions,
                input_ids=input_ids,
            ) as runtime:
                from atom.plugin.sglang.deepseek_v4_bridge import (
                    bind_deepseek_v4_proxy_cache_views,
                    maybe_get_proxy_pool_from_sglang_backend,
                    reset_deepseek_v4_state_slots,
                )

                proxy_pool, _ = maybe_get_proxy_pool_from_sglang_backend()
                if not bind_deepseek_v4_proxy_cache_views(self.model, proxy_pool):
                    raise RuntimeError(
                        "DeepSeek-V4 MTP SGLang proxy KV pool is not initialized"
                    )
                from atom.utils.forward_context import get_forward_context

                reset_slots = getattr(
                    get_forward_context().attn_metadata, "reset_slots", None
                )
                reset_deepseek_v4_state_slots(self.model, reset_slots)

                model_hidden_states = forward_batch.spec_info.hidden_states
                if runtime.forward_batch is not forward_batch:
                    model_hidden_states = _materialize_dummy_hidden_states(
                        model_hidden_states,
                        length=int(runtime.positions.shape[0]),
                    )
                elif (
                    torch.is_tensor(model_hidden_states)
                    and model_hidden_states.shape[0] != runtime.input_ids.shape[0]
                    and bool(
                        getattr(
                            runtime.forward_batch.forward_mode,
                            "is_draft_extend",
                            lambda **kwargs: False,
                        )(include_v2=True)
                    )
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
                        == runtime.input_ids.shape[0]
                    ):
                        model_hidden_states = model_hidden_states.repeat_interleave(
                            tokens_per_req, dim=0
                        )
                    else:
                        raise RuntimeError(
                            "DeepSeek-V4 MTP draft-extend hidden layout mismatch: "
                            f"hidden={tuple(model_hidden_states.shape)}, "
                            f"input_tokens={int(runtime.input_ids.shape[0])}, "
                            f"tokens_per_req={tokens_per_req}"
                        )
                model_hidden_states = _reshape_mtp_hidden_states(
                    model_hidden_states,
                    hidden_size=int(self.config.hidden_size),
                )

                metadata = SGLangForwardBatchMetadata.build(runtime.forward_batch)
                with SGLangForwardBatchMetadata.bind(metadata):
                    hidden_states = self.model(
                        input_ids=runtime.input_ids,
                        positions=runtime.positions,
                        hidden_states=model_hidden_states,
                    )

            if self.pp_group.is_last_rank:
                hidden_states = runtime.trim_output(hidden_states)
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.logits_head,
                    forward_batch,
                    hidden_states_before_norm=hidden_states,
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


EntryClass = [DeepseekV4ForCausalLMNextN]

from collections.abc import Iterable

import importlib
import torch
import torch.nn as nn
from aiter.dist.parallel_state import (
    get_pp_group,
    get_tp_group,
)
from vllm.config import VllmConfig
from vllm.model_executor.models.interfaces import (
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModel,
    VllmModelForTextGeneration,
)
from vllm.sequence import IntermediateTensors

import atom  # noqa: F401
from atom.plugin.config import generate_atom_config_for_plugin_mode

import logging

logger = logging.getLogger("atom")


_ATOM_MODEL_CLASSES: dict[str, str] = {
    "Qwen3ForCausalLM": "atom.models.qwen3:Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe:Qwen3MoeForCausalLM",
}


def _get_atom_model_cls(model_arch: str) -> type:
    if model_arch is not None and model_arch in _ATOM_MODEL_CLASSES:
        model_ref = _ATOM_MODEL_CLASSES[model_arch]
    else:
        raise ValueError(f"The {model_arch} is not supported by ATOM OOT backend")

    module_path, class_name = model_ref.split(":", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _prepare_env(atom_config) -> None:
    from atom.plugin.register import set_attn_cls, init_aiter_dist

    # set global attention class
    logger.info("Set global attention class")
    set_attn_cls()

    # init aiter dist for using aiter custom collective ops
    logger.info("Init aiter dist for using aiter custom collective ops")
    init_aiter_dist(config=atom_config)


class ATOMModelBase(nn.Module, VllmModel, SupportsQuant, SupportsPP):
    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.text_config = self.config.get_text_config()
        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config

        # Weights to skip in `self.load_weights`
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []

        self.atom_config = generate_atom_config_for_plugin_mode(vllm_config)

        _prepare_env(atom_config=self.atom_config)

        model_arch = vllm_config.model_config.architectures[0]
        model_cls = _get_atom_model_cls(model_arch)

        logger.info(f"Construct ATOM model {model_arch} for vLLM plugin mode")
        self.model = model_cls(self.atom_config)

        if self.model is None:
            model_arch = vllm_config.model_config.architectures[0]
            raise ValueError(
                f"The model {model_arch} is not supported by model impl backend atom"
            )

        # here init aiter dist for using aiter custom collective ops
        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        if not self.pp_group.is_first_rank:
            assert intermediate_tensors is not None
            input_ids = None
            inputs_embeds = intermediate_tensors["hidden_states"]

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        if not self.pp_group.is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return hidden_states

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        return self.model.load_weights(weights)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return logits


class ATOMForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMMoEForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...

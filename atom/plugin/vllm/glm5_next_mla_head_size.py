# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Give vLLM the MLA cache width ATOM's GLM-5.3-Flash kernels use.

vLLM 0.31 reads the checkpoint with its own `Glm5NextConfig` /
`Glm5NextTextConfig` (`vllm/transformers_utils/configs/glm5_next.py`) and
already lists `glm5_next_text` as MLA. What it gets wrong for ATOM is the
width: `get_head_size()` returns `kv_lora_rank + qk_rope_head_dim`, and
GLM-5.3-Flash keeps `qk_rope_head_dim == 0` so its INDEXER stays NoPE, while
ATOM materializes the rope block as a zero pad on the MLA side only
(`_ROPE_PAD` in atom/models/glm5_next.py, the same distinction
`aiter_mla.mla_kv_entry_dim` draws natively). The sum is then 512 where the
kernels need 576, the MLA cache is allocated a block too narrow per token, and
the first reshape onto it fails. This patch reapplies ATOM's width on vLLM's
side of the boundary, keyed on the GLM text model_type so every other model
keeps vLLM's own answer.
"""

import logging

_logger = logging.getLogger("atom")

_GLM5_TEXT_MODEL_TYPE = "glm5_next_text"


def _is_glm5_next_text(hf_text_config) -> bool:
    return getattr(hf_text_config, "model_type", None) == _GLM5_TEXT_MODEL_TYPE


def patch_vllm_mla_head_size() -> None:
    # `model_arch_config_convertor` and `vllm.config` import each other, so
    # reaching the convertor module FIRST raises ImportError on a partially
    # initialized module. Importing vllm.config up front settles the cycle;
    # in a live server it is already loaded and this is a no-op.
    import vllm.config  # noqa: F401
    from vllm.transformers_utils import model_arch_config_convertor as convertor

    base = getattr(convertor, "ModelArchConfigConvertorBase", None)
    if base is None or not hasattr(base, "get_head_size"):
        return  # a vLLM without this hook has nothing to patch
    if getattr(base, "_atom_glm5_next_patched", False):
        return

    original_get_head_size = base.get_head_size

    def get_head_size(self) -> int:
        if _is_glm5_next_text(self.hf_text_config):
            declared = getattr(self.hf_text_config, "mla_kv_entry_dim", None)
            if declared:
                return int(declared)
            # Imported lazily: this module is loaded from the platform hook for
            # every model, and atom.models.glm5_next is a heavy import.
            from atom.models.glm5_next import _ROPE_PAD

            return int(self.hf_text_config.kv_lora_rank) + _ROPE_PAD
        return original_get_head_size(self)

    base.get_head_size = get_head_size
    base._atom_glm5_next_patched = True
    _logger.info("ATOM plugin: patched vLLM MLA head size for glm5_next_text")


def install_glm5_next_mla_head_size_patch() -> None:
    """Install the patch now, or as soon as vLLM can take it.

    Runs from the PLATFORM hook, which vLLM calls while `vllm.config` is still
    being imported; reaching `model_arch_config_convertor` from there raises on
    the partially initialized module. So when that happens, defer to the
    construction of vLLM's own `Glm5NextTextConfig`: by then vLLM is fully
    imported, and it happens strictly before ModelConfig computes
    `model_arch_config`, which is what reads the head size.
    """
    try:
        patch_vllm_mla_head_size()
        return
    except ImportError as exc:
        _logger.info(
            "ATOM plugin: deferring the glm5_next MLA head-size patch to config "
            "construction (%s)",
            exc,
        )

    try:
        from vllm.transformers_utils.configs.glm5_next import Glm5NextTextConfig
    except ImportError:
        # This runs from the platform hook for every model: a vLLM that does
        # not know glm5_next cannot serve GLM-5.3-Flash, and must still load
        # the plugin for everything else.
        return

    if getattr(Glm5NextTextConfig, "_atom_patch_on_init", False):
        return
    original_init = Glm5NextTextConfig.__init__

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        patch_vllm_mla_head_size()

    Glm5NextTextConfig.__init__ = __init__
    Glm5NextTextConfig._atom_patch_on_init = True

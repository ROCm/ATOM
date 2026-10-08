"""Flash MTP adapters that SGLang 0.5.20 still does not provide.

Upstream 0.5.20 recognizes ``qwen4_exp`` (config, M-RoPE, processor,
``hybrid_gdn_config``). Those recognition shims are gone. Native MTP still
needs a draft architecture rewrite and HC hidden width: upstream's MTP
EntryClass name is not ``Qwen4ExpForCausalLMNextN``.

GDN pad sentinels live in ``qwen4_exp_gdn_pad.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("atom.plugin.sglang.qwen4_exp_recognition")

QWEN4_EXP_NEXTN_ARCH = "Qwen4ExpForCausalLMNextN"
_QWEN4_EXP_DRAFT_SOURCE_ARCHS = {
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpForCausalLM",
    # 0.5.20 ``_config_draft_model`` rewrites the checkpoint arch to this
    # before ATOM's hook. #2385 only listed the checkpoint name, which
    # 0.5.17 left unchanged, so the Native NextN wrapper never loaded.
    "Qwen4ExpForCausalLMMTP",
    QWEN4_EXP_NEXTN_ARCH,
}


def is_qwen4_exp_nextn_arch(hf: Any) -> bool:
    archs = getattr(hf, "architectures", None) or []
    return QWEN4_EXP_NEXTN_ARCH in archs


def rewrite_qwen4_exp_draft_hf_config(
    hf_config: Any, hf_text_config: Any | None = None
) -> bool:
    """Rewrite a draft HF config onto ``Qwen4ExpForCausalLMNextN``.

    Shrink the draft worker to the one reusable QSA layer so the KV pool is
    one page table, not the target's hybrid stack. SGLang still allocates a
    single full-attention KV pool; Native owns QSA compute.
    """
    text = hf_text_config or getattr(hf_config, "text_config", None) or hf_config
    archs = list(getattr(hf_config, "architectures", None) or [])
    model_type = str(
        getattr(text, "model_type", None) or getattr(hf_config, "model_type", "") or ""
    )
    if not archs:
        if not model_type.startswith("qwen4_exp"):
            return False
        archs = ["Qwen4ExpForConditionalGeneration"]
    if not any(a in _QWEN4_EXP_DRAFT_SOURCE_ARCHS for a in archs):
        return False
    mtp = getattr(text, "mtp", None) or {}
    layer_types = mtp.get("layer_types", ["full_attention"])
    n = getattr(text, "mtp_num_hidden_layers", 1)
    if (
        n != 1
        or len(layer_types) != 1
        or layer_types[0] not in {"full_attention", "qwen_sparse_attention"}
    ):
        raise ValueError("Flash MTP requires exactly one QSA draft layer")
    layer_types = ["full_attention"]
    for cfg in (hf_config, text):
        cfg.architectures = [QWEN4_EXP_NEXTN_ARCH]
        cfg.num_nextn_predict_layers = 1
        cfg.num_hidden_layers = 1
        cfg.layer_types = layer_types
        if hasattr(cfg, "ple_layer_ids"):
            cfg.ple_layer_ids = []
    logger.info(
        "Rewrote Flash draft arch to %s (layers=%s types=%s)",
        QWEN4_EXP_NEXTN_ARCH,
        n,
        layer_types,
    )
    return True


def promote_flash_draft_text_config(model_config: Any) -> bool:
    """Point the draft ``ModelConfig`` at the text config.

    0.5.20 keeps the VL ``Qwen4ExpConfig`` as ``hf_config`` and only shrinks
    ``text_config``. That object has no ``vocab_size`` / ``hidden_size`` /
    ``mtp``. The Native NextN wrapper and ``Qwen4ExpMTP`` read those fields
    on the config they are given.
    """
    hf = getattr(model_config, "hf_config", None)
    if not is_qwen4_exp_nextn_arch(hf):
        return False
    text = getattr(model_config, "hf_text_config", None) or getattr(
        hf, "text_config", None
    )
    if text is None or text is hf:
        return False
    if getattr(hf, "vocab_size", None) is not None:
        return False
    model_config.hf_config = text
    model_config.hf_text_config = text
    return True


def apply_qwen4_exp_hc_hidden_size(model_config: Any) -> bool:
    """Treat Flash ``hc_count`` like DSV4 ``hc_mult`` for EAGLE hidden width."""
    text = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    if getattr(text, "model_type", None) not in {"qwen4_exp", "qwen4_exp_text"}:
        return False
    hc = int(getattr(text, "hc_count", 0) or 0)
    hidden = int(getattr(model_config, "hidden_size", 0) or 0)
    if hc <= 1 or hidden <= 0:
        return False
    model_config.spec_hidden_size = hidden * hc
    model_config.hc_hidden_size = hidden * hc
    return True


def _patch_qwen4_exp_draft_model() -> None:
    """Map the Flash draft config to the Native ATOM NextN wrapper."""
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except Exception:  # noqa: BLE001
        return

    orig = ModelConfig._config_draft_model
    if getattr(orig, "_atom_qwen4_exp_nextn", False):
        return

    def _config_draft_model(self):
        orig(self)
        if not getattr(self, "is_draft_model", False):
            return
        rewrite_qwen4_exp_draft_hf_config(
            self.hf_config, getattr(self, "hf_text_config", None)
        )
        promote_flash_draft_text_config(self)

    _config_draft_model._atom_qwen4_exp_nextn = True  # type: ignore[attr-defined]
    ModelConfig._config_draft_model = _config_draft_model


def _patch_qwen4_exp_spec_hidden_size() -> None:
    """EAGLE stores flattened HC ``[N, hc_count * H]``, not mixed ``[N, H]``."""
    try:
        from sglang.srt.configs.model_config import ModelConfig
    except Exception:  # noqa: BLE001
        return

    orig = ModelConfig._derive_model_shapes
    if getattr(orig, "_atom_qwen4_exp_hc", False):
        return

    def _derive_model_shapes(self):
        orig(self)
        if apply_qwen4_exp_hc_hidden_size(self):
            logger.info(
                "Flash HC hidden width spec_hidden_size=%s hc_hidden_size=%s",
                self.spec_hidden_size,
                self.hc_hidden_size,
            )

    _derive_model_shapes._atom_qwen4_exp_hc = True  # type: ignore[attr-defined]
    ModelConfig._derive_model_shapes = _derive_model_shapes


def apply_qwen4_exp_recognition_patch() -> None:
    """Install Flash MTP draft-arch and HC-width adapters.

    Config, M-RoPE, processor, and hybrid GDN recognition now come from
    SGLang 0.5.20. This call remains because upstream MTP EntryClass naming
    and HC hidden width still do not match Native ATOM.
    """

    _patch_qwen4_exp_draft_model()
    _patch_qwen4_exp_spec_hidden_size()

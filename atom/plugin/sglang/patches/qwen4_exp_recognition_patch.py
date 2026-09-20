"""Qwen3.8-Flash-Next (`qwen4_exp`) recognition shims for an old SGLang.

This file exists only because **SGLang is too old to recognize Flash /
``qwen4_exp``**. This image is 0.5.17; the latest released tag **v0.5.19
still has no** ``configs/qwen4_exp.py``. Upstream ``main`` landed official
recognition in PR #37500 (2026-09-08); it is expected to ship in
**v0.5.20**.

**Remove when** SGLang is upgraded far enough to recognize Flash: delete
this file and drop ``apply_qwen4_exp_recognition_patch()`` from
``register.py``. Ready when ``sglang.srt.configs.qwen4_exp`` exists,
``hybrid_gdn_config`` matches ``Qwen4Exp*``, ``get_rope_index`` accepts
``model_type=qwen4_exp``, and ``PROCESSOR_MAPPING`` already lists
``Qwen4ExpForConditionalGeneration``. Do **not** delete ATOM's Native
compute path (``qwen4_exp.py`` / bridge) — upstream
``models/qwen4_exp.py`` does not replace Native PR #2048.

``apply_gdn_pad_sentinels`` is **not** a recognition shim. Do not drop it
just because SGLang recognizes Flash. After the upgrade, Hybrid GDN pad
for Flash is the same path as Qwen3.8-2.4T (``_replay_metadata`` /
``_forward_metadata``). Change Native GDN to consume those Hybrid
buffers directly (like 2.4T), then delete this helper and its call in
``attention_gdn.py``. See ``apply_gdn_pad_sentinels`` below.

Until that upgrade, ATOM must:

1. Allow SGLang to re-register HuggingFace model_types that
   transformers 5.16.1 already owns (otherwise ``Qwen3_5TextConfig``
   cannot be imported).
2. Register AutoConfig classes so ServerArgs can parse the checkpoint
   before Native ``get_hf_config`` runs.
3. Teach ``hybrid_gdn_config`` that Flash is a GDN hybrid, otherwise
   SGLang allocates no MambaPool / ``mamba_map``, GDN zeros out, and
   greedy text becomes garbage.
4. Map M-RoPE ``model_type=qwen4_exp`` onto the ``qwen3_5`` index formula
   (``get_rope_index`` only knows qwen3_5 / qwen2_vl on 0.5.17).
5. Own the Flash VL processor class and put it on SGLang's
   ``PROCESSOR_MAPPING``. 0.5.17 has no ``qwen4_exp`` processor.
6. Re-apply GDN CUDA-graph / DP pad sentinels for Native GDN
   (``apply_gdn_pad_sentinels``, used by ``attention_gdn.py``). Latest
   SGLang Hybrid already pads Flash the same way as 2.4T once Flash is
   a GDN hybrid; Native GDN still clones / reconstructs and can lose
   those sentinels. After upgrade: make Native GDN eat Hybrid's
   already-padded buffers, then delete this helper (not with items 1–5).

``Qwen4ExpTextConfig`` subclasses SGLang's ``Qwen3_5TextConfig`` so
``mamba2_cache_params`` / ``linear_layer_ids`` exist. Official
transformers ``Qwen4ExpTextConfig`` does not carry those SGLang fields.
Both config classes stay module-level: SGLang spawn-pickles
``ServerArgs.model_config.hf_config``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, ClassVar

import torch
from transformers import AutoConfig, PretrainedConfig

from atom.plugin.sglang.attention_backend.backend_resolver import real_batch_size

logger = logging.getLogger("atom.plugin.sglang.qwen4_exp_recognition")


def _allow_duplicate_hf_autoconfig_register() -> None:
    """Let SGLang re-register model_types that transformers 5.16.1 already owns.

    Temporary: needed because this SGLang is too old to own ``qwen4_exp``.
    Remove this helper (and this file) once SGLang recognizes Flash natively.

    Importing ``sglang.srt.configs.qwen3_5`` executes ``configs/__init__.py``,
    which calls ``AutoConfig.register("qwen3_asr", ...)``. Transformers 5.16.1
    already registered that key, so the import raises and Flash's text config
    would silently fall back to bare ``PretrainedConfig`` (no
    ``mamba2_cache_params``).
    """

    orig = AutoConfig.register
    if getattr(orig, "_atom_exist_ok", False):
        return

    def register(model_type, config, exist_ok=False):
        try:
            return orig(model_type, config, exist_ok=True)
        except TypeError:
            try:
                return orig(model_type, config)
            except ValueError as exc:
                if "already used by a Transformers config" not in str(exc):
                    raise
                return None

    register._atom_exist_ok = True  # type: ignore[attr-defined]
    AutoConfig.register = register  # type: ignore[method-assign]


_allow_duplicate_hf_autoconfig_register()

try:
    from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig as _Qwen4ExpTextBase
except Exception:  # noqa: BLE001  # pragma: no cover - register only loads under SGLang
    _Qwen4ExpTextBase = PretrainedConfig


class Qwen4ExpTextConfig(_Qwen4ExpTextBase):
    """Shim for nested ``qwen4_exp_text`` while SGLang is too old to ship it.

    Remove with this file after SGLang upgrades to a release that provides
    official ``Qwen4ExpTextConfig`` (expected >= v0.5.20 / PR #37500).

    Subclasses SGLang's Qwen3.5 text config so ``hybrid_gdn_config`` /
    MambaPool sizing see ``mamba2_cache_params``. Without that, SGLang
    allocates no mamba slots and ATOM GDN/PLE zero out → greedy garbage.
    """

    model_type = "qwen4_exp_text"

    @property
    def layers_block_type(self):
        """Prefer checkpoint ``layer_types``; map Flash QSA names to GDN pool ids."""
        layer_types = getattr(self, "layer_types", None)
        if layer_types:
            out = []
            for layer_type in layer_types:
                if layer_type in (
                    "full_attention",
                    "qwen_sparse_attention",
                    "attention",
                ):
                    out.append("attention")
                else:
                    out.append("linear_attention")
            return out
        return super().layers_block_type  # type: ignore[misc]


class Qwen4ExpConfig(PretrainedConfig):
    """Shim for ``qwen4_exp`` while SGLang is too old to ship it.

    Remove with this file after SGLang upgrades to a release that provides
    official ``Qwen4ExpConfig`` (expected >= v0.5.20 / PR #37500).

    Must stay module-level: SGLang spawn-pickles
    ``ServerArgs.model_config.hf_config``; a nested class is not picklable.
    """

    model_type = "qwen4_exp"

    def __init__(
        self,
        text_config: dict | PretrainedConfig | None = None,
        vision_config: dict | PretrainedConfig | None = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_kwargs = dict(text_config)
            text_kwargs.pop("model_type", None)
            text_config = Qwen4ExpTextConfig(**text_kwargs)
        if isinstance(vision_config, dict):
            vision_kwargs = dict(vision_config)
            vision_kwargs.pop("model_type", None)
            vision_config = PretrainedConfig(**vision_kwargs)
        self.text_config = text_config
        self.vision_config = vision_config
        # Official HF puts vision token ids on the root config. Keep them even
        # if PretrainedConfig drops unknown kwargs.
        vision_token_ids = {
            key: kwargs.pop(key)
            for key in (
                "image_token_id",
                "video_token_id",
                "vision_start_token_id",
                "vision_end_token_id",
            )
            if key in kwargs
        }
        super().__init__(**kwargs)
        for key, value in vision_token_ids.items():
            setattr(self, key, value)
        src = self.text_config
        if src is None:
            return
        for key in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "vocab_size",
            "head_dim",
            "max_position_embeddings",
            "rms_norm_eps",
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "intermediate_size",
            "image_token_id",
            "video_token_id",
            "vision_start_token_id",
            "vision_end_token_id",
        ):
            if getattr(self, key, None) is None and hasattr(src, key):
                setattr(self, key, getattr(src, key))


def _register_qwen4_exp_hf_configs() -> None:
    """Register AutoConfig aliases so this old SGLang can parse Flash ckpts.

    Temporary: SGLang 0.5.17 / 0.5.19 has no ``qwen4_exp`` mapping.
    Remove with this file after SGLang recognizes Flash natively.
    """

    def _register(model_type: str, config_cls: type) -> None:
        try:
            AutoConfig.register(model_type, config_cls, exist_ok=True)
        except TypeError:
            try:
                AutoConfig.register(model_type, config_cls)
            except ValueError as exc:
                if "already used by a Transformers config" not in str(exc):
                    raise

    _register("qwen4_exp_text", Qwen4ExpTextConfig)
    _register("qwen4_exp", Qwen4ExpConfig)


def _patch_qwen4_exp_mrope() -> None:
    """Map Flash-Next ``qwen4_exp`` onto Qwen3.5 M-RoPE index math.

    Temporary: this SGLang's ``get_rope_index`` only knows qwen3_5 / qwen2_vl.
    Remove with this file after SGLang's helper accepts ``qwen4_exp``.
    """
    try:
        from sglang.srt.layers.rotary_embedding import mrope as mrope_mod
        from sglang.srt.layers.rotary_embedding import mrope_rope_index as mri
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        return

    orig = mri.get_rope_index
    if getattr(orig, "_atom_qwen4_exp", False):
        return

    def get_rope_index_with_qwen4_exp(*args, **kwargs):
        if len(args) >= 5 and args[4] == "qwen4_exp":
            args = (*args[:4], "qwen3_5", *args[5:])
        if kwargs.get("model_type") == "qwen4_exp":
            kwargs = {**kwargs, "model_type": "qwen3_5"}
        return orig(*args, **kwargs)

    get_rope_index_with_qwen4_exp._atom_qwen4_exp = True
    mri.get_rope_index = get_rope_index_with_qwen4_exp
    if getattr(mrope_mod, "get_rope_index", None) is orig:
        mrope_mod.get_rope_index = get_rope_index_with_qwen4_exp


def _patch_hybrid_gdn_config_for_qwen4_exp() -> None:
    """Teach this old SGLang that Flash is a hybrid GDN model.

    Temporary: stock ``hybrid_gdn_config`` only matches Qwen3-Next / Qwen3.5.
    Remove with this file after SGLang recognizes ``qwen4_exp`` as GDN hybrid
    (upstream ``Qwen4ExpTextConfig`` subclasses ``Qwen3NextConfig``).
    """
    try:
        from sglang.srt.configs import hybrid_arch as ha
    except Exception:  # noqa: BLE001 - optional across SGLang versions
        logger.warning("hybrid_gdn_config patch skipped: hybrid_arch import failed")
        return

    orig = ha.hybrid_gdn_config
    if getattr(orig, "_atom_qwen4_exp", False):
        return

    def hybrid_gdn_config(model_config):
        cfg = orig(model_config)
        if cfg is not None:
            return cfg
        hf = getattr(model_config, "hf_config", None)
        if hf is None:
            return None
        text = None
        getter = getattr(hf, "get_text_config", None)
        if callable(getter):
            try:
                text = getter()
            except Exception:  # noqa: BLE001
                text = None
        if text is None:
            text = getattr(hf, "text_config", None) or hf
        mt = getattr(text, "model_type", None) or getattr(hf, "model_type", None)
        if (
            mt in ("qwen4_exp_text", "qwen4_exp")
            or isinstance(text, Qwen4ExpTextConfig)
            or isinstance(hf, Qwen4ExpConfig)
        ):
            return text
        return None

    hybrid_gdn_config._atom_qwen4_exp = True  # type: ignore[attr-defined]
    ha.hybrid_gdn_config = hybrid_gdn_config
    # ``from hybrid_arch import hybrid_gdn_config`` aliases must be rebound.
    for mod in list(sys.modules.values()):
        try:
            if getattr(mod, "hybrid_gdn_config", None) is orig:
                mod.hybrid_gdn_config = hybrid_gdn_config
        except Exception:  # noqa: BLE001, S112
            continue
    logger.info("Patched hybrid_gdn_config to recognize qwen4_exp / Qwen3.8-Flash-Next")


try:
    from sglang.srt.multimodal.processors.base_processor import (
        MultimodalSpecialTokens,
    )
    from sglang.srt.multimodal.processors.transformers_auto import (
        TransformersAutoMultimodalProcessor,
    )
except Exception:  # noqa: BLE001 - SGLang multimodal symbols are optional
    MultimodalSpecialTokens = None  # type: ignore[misc, assignment]
    TransformersAutoMultimodalProcessor = object

# Native atom.models.qwen4_exp defaults when the wrapper config omits ids.
_QWEN4_EXP_IMAGE_TOKEN_ID = 248056
_QWEN4_EXP_VIDEO_TOKEN_ID = 248057
_QWEN4_EXP_VISION_START_TOKEN_ID = 248053
_QWEN4_EXP_VISION_END_TOKEN_ID = 248054


class Qwen4ExpForConditionalGeneration:
    """PROCESSOR_MAPPING key. SGLang 0.5.17 looks up processors by class name."""


def _first_attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _qwen4_exp_mm_source_configs(hf_config: Any) -> tuple[Any, ...]:
    text = getattr(hf_config, "text_config", None)
    vision = getattr(hf_config, "vision_config", None)
    return tuple(cfg for cfg in (hf_config, text, vision) if cfg is not None)


def qwen4_exp_uses_mrope(hf_config: Any) -> bool:
    """True when Flash's text rope carries ``mrope_section`` (scaling or v5 params)."""
    for cfg in _qwen4_exp_mm_source_configs(hf_config):
        for attr in ("rope_parameters", "rope_scaling"):
            rope = getattr(cfg, attr, None) or {}
            if isinstance(rope, dict) and "mrope_section" in rope:
                return True
        if "mrope" in str(getattr(cfg, "rope_type", "")).lower():
            return True
    return False


def qwen4_exp_mm_token_id(hf_config: Any, names: tuple[str, ...], default: int) -> int:
    for cfg in _qwen4_exp_mm_source_configs(hf_config):
        value = _first_attr(cfg, names)
        if value is not None:
            return int(value)
    return int(default)


class Qwen4ExpMultimodalProcessor(TransformersAutoMultimodalProcessor):
    """HF processor path for Flash-Next text and image/video inputs.

    Temporary: SGLang 0.5.17 / 0.5.19 has no Flash VL processor. Remove
    with this file after an upgraded SGLang already maps
    ``Qwen4ExpForConditionalGeneration``. Native ViT / QSA is not this class.
    """

    models: ClassVar[list[type]] = [Qwen4ExpForConditionalGeneration]
    supports_transformers_backend = True

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        if MultimodalSpecialTokens is None:
            return
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=getattr(_processor, "image_token", None),
            video_token=getattr(_processor, "video_token", None),
            audio_token=getattr(_processor, "audio_token", None),
            image_token_id=qwen4_exp_mm_token_id(
                hf_config,
                ("image_token_id", "image_token_index", "im_token_id"),
                _QWEN4_EXP_IMAGE_TOKEN_ID,
            ),
            video_token_id=qwen4_exp_mm_token_id(
                hf_config, ("video_token_id",), _QWEN4_EXP_VIDEO_TOKEN_ID
            ),
            audio_token_id=_first_attr(hf_config, ("audio_token_id",)),
        ).build(_processor)
        self._is_mrope = qwen4_exp_uses_mrope(hf_config)
        vision_config = getattr(hf_config, "vision_config", None)
        self._spatial_merge_size = int(
            getattr(vision_config, "spatial_merge_size", 2) or 2
        )
        self._tokens_per_second = getattr(vision_config, "tokens_per_second", None)
        self._vision_start_token_id = qwen4_exp_mm_token_id(
            hf_config,
            ("vision_start_token_id", "image_start_token_id", "im_start_id"),
            _QWEN4_EXP_VISION_START_TOKEN_ID,
        )
        self._vision_end_token_id = qwen4_exp_mm_token_id(
            hf_config,
            ("vision_end_token_id", "image_end_token_id", "im_end_id"),
            _QWEN4_EXP_VISION_END_TOKEN_ID,
        )
        self._model_type = getattr(hf_config, "model_type", None) or "qwen4_exp"


Qwen4ExpTextOnlyProcessor = Qwen4ExpMultimodalProcessor


def register_qwen4_exp_processor() -> None:
    """Register Flash on SGLang's multimodal processor table.

    Temporary: SGLang 0.5.17 / 0.5.19 has no ``qwen4_exp`` processor, so
    ``get_mm_processor`` raises for ``Qwen4ExpForConditionalGeneration``.
    Remove with this file after an upgraded SGLang already maps that
    architecture name. Native ViT / QSA compute is not this table.
    """
    try:
        from sglang.srt.managers.multimodal_processor import PROCESSOR_MAPPING
    except Exception:  # noqa: BLE001 - processor mapping is optional outside SGLang
        return

    PROCESSOR_MAPPING.setdefault(
        Qwen4ExpForConditionalGeneration,
        Qwen4ExpMultimodalProcessor,
    )


def register_qwen4_exp_text_only_processor() -> None:
    """Back-compat alias for :func:`register_qwen4_exp_processor`."""

    register_qwen4_exp_processor()


def apply_gdn_pad_sentinels(
    forward_batch: Any,
    idx: torch.Tensor,
    query_start_loc: torch.Tensor,
    mode: Any,
    bs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mark CUDA-graph / DP pad rows as GDN no-write (``PAD_SLOT_ID = -1``).

    Latest SGLang Hybrid already pads Flash GDN the same way as
    Qwen3.8-2.4T (not a Flash-only helper): ``_replay_metadata`` /
    ``_forward_metadata`` write ``mamba_cache_indices = -1`` on pad
    rows. 2.4T Native GDN reads those Hybrid buffers, so it does not
    need this clone. Flash Native GDN still reconstructs or clones
    metadata and can bring back a finished request's mamba slot.

    After upgrading SGLang (expected v0.5.20 / PR #37500): keep Hybrid
    pad, change ``attention_gdn._build_gdn_metadata`` to consume
    ``linear_backend.forward_metadata`` directly (same as 2.4T), then
    delete this function and its call. Do not reconstruct pad rows from
    ``req_pool_indices``. PLE must keep aliasing Hybrid's static
    buffers — a clone cannot be captured into the CUDA graph.

    Clones so Hybrid's static CUDA-graph buffers are not mutated in place.
    """

    live_bs = real_batch_size(forward_batch)
    if live_bs < idx.shape[0]:
        # CUDA-graph / DP pad rows keep a finished request's mamba slot.
        idx = idx.clone()
        idx[live_bs:] = -1
    # Decode pads are 1 token/row, so cu_seqlens must stop at live_bs.
    # Extend reconstruct already ends at the last real request's token.
    if (
        mode.is_decode_or_idle()
        and live_bs < bs
        and query_start_loc.numel() > live_bs + 1
    ):
        query_start_loc = query_start_loc.clone()
        query_start_loc[live_bs + 1 :] = live_bs
    return idx, query_start_loc


def apply_qwen4_exp_recognition_patch() -> None:
    """Install Flash recognition shims required by this too-old SGLang.

    Exists because SGLang 0.5.17 / 0.5.19 does not recognize ``qwen4_exp``.
    After upgrading to an SGLang that recognizes Flash, delete this call and
    this file (expected v0.5.20 / PR #37500). Do not delete ATOM's Native
    compute plugin.
    """

    _register_qwen4_exp_hf_configs()
    _patch_qwen4_exp_mrope()
    _patch_hybrid_gdn_config_for_qwen4_exp()
    register_qwen4_exp_processor()

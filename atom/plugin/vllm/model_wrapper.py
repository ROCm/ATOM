from collections.abc import Iterable

import functools
import importlib
import json
import os
import types
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
    SupportsMultiModal,
    SupportsMRoPE,
    MultiModalEmbeddings,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModel,
    VllmModelForTextGeneration,
)
from vllm.sequence import IntermediateTensors
from vllm.forward_context import (
    get_forward_context as get_vllm_forward_context,
    is_forward_context_available,
)

import atom  # noqa: F401
from atom.plugin.config import (
    _generate_atom_config_from_vllm_config,
    generate_atom_config_for_plugin_mode,
)
from atom.plugin.prepare import _set_framework_backbone

import logging

logger = logging.getLogger("atom")

_MTP_MASK_INPUT_ARCH: set[str] = {
    "DeepSeekMTPModel",
    "Glm4MoeMTPModel",
}
_EAGLE3_DRAFT_ARCH_TO_ATOM_ARCH: dict[str, str] = {
    # vLLM/HF draft arch name: ATOM server-mode draft class
    "Eagle3LlamaForCausalLM": "Eagle3LlamaModel",
    "LlamaForCausalLMEagle3": "Eagle3LlamaModel",
    "Eagle3DeepseekV2ForCausalLM": "Eagle3DeepseekMLAModel",
    "Eagle3DeepseekV3ForCausalLM": "Eagle3DeepseekMLAModel",
}
_EAGLE3_ATOM_DRAFT_ARCHS: set[str] = {
    "Eagle3LlamaModel",
    "Eagle3DeepseekMLAModel",
}
# DeepSeek-V4 is a native ATOM model whose forward reads ATOM's own forward
# context (not vLLM's). It needs the V4 proxy-cache bridge wired in the plugin
# wrapper (register at init, bind + enter context per forward); see `forward`.
_DEEPSEEK_V4_ARCH = "DeepseekV4ForCausalLM"


def _probe_v4_routed_expert_dtype(model_path) -> str | None:
    """Return ``"fp4"`` / ``"fp8"`` / ``None`` for a DeepSeek-V4 checkpoint's
    routed-expert weights, read from the actual on-disk tensor dtype.

    V4 stores routed experts (``ffn.experts.*.w{1,2,3}``) as either FP4 e2m1
    (packed two-per-byte into (u)int8 + per_1x32 UE8M0 scale) or FP8 e4m3
    (per-block 128x128). The checkpoint's global ``quantization_config`` only
    describes the FP8 *projection* scheme, so the routed-expert dtype can only
    be known by reading the weight tensor itself.
    """
    if not model_path or not os.path.isdir(model_path):
        return None
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.isfile(idx_path):
        return None
    try:
        with open(idx_path) as f:
            wmap = json.load(f).get("weight_map", {})
        probe = next(
            (k for k in wmap if ".ffn.experts." in k and k.endswith(".w1.weight")),
            None,
        )
        if probe is None:
            return None
        from safetensors import safe_open

        with safe_open(os.path.join(model_path, wmap[probe]), framework="pt") as h:
            dt = str(h.get_slice(probe).get_dtype()).upper()
    except Exception:
        return None
    if dt in ("I8", "U8", "UINT8", "INT8"):
        return "fp4"  # FP4 e2m1 packed two values per byte
    if dt in ("F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ"):
        return "fp8"
    return None


def _maybe_set_v4_expert_dtype(atom_config, vllm_config) -> None:
    """Pin DeepSeek-V4 ``hf_config.expert_dtype`` from the on-disk routed-expert
    dtype so ``make_v4_quant_config`` selects the correct (FP4 vs FP8) spec.

    Checkpoints like DeepSeek-V4-Flash ship FP4 routed experts + FP8
    projections, but their global ``quantization_config`` only declares the FP8
    scheme. The model's parser-based auto-detection therefore mis-classifies the
    routed experts as FP8-block and dequantizes the FP4 expert weights wrongly,
    producing garbage output. ``expert_dtype`` is the model's documented
    override hook; we set it from the real on-disk dtype.
    """
    hf_config = getattr(atom_config, "hf_config", None)
    if hf_config is None or getattr(hf_config, "expert_dtype", None):
        return  # explicit config / prior setting wins
    model_path = getattr(getattr(vllm_config, "model_config", None), "model", None)
    dtype = _probe_v4_routed_expert_dtype(model_path)
    if dtype:
        hf_config.expert_dtype = dtype
        logger.info(
            "DeepSeek-V4: pinned expert_dtype=%s from on-disk routed-expert "
            "weights (%s)",
            dtype,
            model_path,
        )


_ATOM_MODEL_CLASSES: dict[str, str] = {
    "LlamaForCausalLM": "atom.models.llama:LlamaForCausalLM",
    "Qwen3ForCausalLM": "atom.models.qwen3:Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM": "atom.models.qwen3_moe:Qwen3MoeForCausalLM",
    "GptOssForCausalLM": "atom.models.gpt_oss:GptOssForCausalLM",
    "DeepseekV3ForCausalLM": "atom.models.deepseek_v2:DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM": "atom.models.deepseek_v2:DeepseekV3ForCausalLM",
    "Glm4MoeForCausalLM": "atom.models.glm4_moe:Glm4MoeForCausalLM",
    "GlmMoeDsaForCausalLM": "atom.models.deepseek_v2:GlmMoeDsaForCausalLM",
    "DeepSeekMTPModel": "atom.models.deepseek_mtp:DeepSeekMTP",
    "Glm4MoeMTPModel": "atom.models.glm4_moe_mtp:Glm4MoeMTP",
    "Qwen3NextForCausalLM": "atom.plugin.vllm.models.qwen3_next:Qwen3NextForCausalLM",
    "Qwen3NextMTP": "atom.models.qwen3_next_mtp:Qwen3NextMTP",
    "Qwen3_5MoeForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5MoeForConditionalGeneration_",
    "Qwen3_5ForConditionalGeneration": "atom.plugin.vllm.models.qwen3_5:Qwen3_5ForConditionalGeneration_",
    "KimiK25ForConditionalGeneration": "atom.plugin.vllm.models.kimi_k25:KimiK25ForConditionalGeneration_",
    "MiniMaxM2ForCausalLM": "atom.models.minimax_m2:MiniMaxM2ForCausalLM",
    "DeepseekV4ForCausalLM": "atom.plugin.vllm.models.deepseek_v4:DeepseekV4ForCausalLM",
    "MiniMaxM3SparseForCausalLM": "atom.models.minimax_m3:MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration": "atom.models.minimax_m3:MiniMaxM3SparseForConditionalGeneration",
    "Eagle3LlamaModel": "atom.models.eagle3_llama:Eagle3LlamaModel",
    "Eagle3DeepseekMLAModel": "atom.models.eagle3_deepseek_mla:Eagle3DeepseekMLAModel",
}


def _normalize_atom_model_arch(model_arch: str) -> str:
    return _EAGLE3_DRAFT_ARCH_TO_ATOM_ARCH.get(model_arch, model_arch)


def _is_eagle3_draft_arch(model_arch: str | None) -> bool:
    return (
        model_arch in _EAGLE3_DRAFT_ARCH_TO_ATOM_ARCH
        or model_arch in _EAGLE3_ATOM_DRAFT_ARCHS
    )


def _get_atom_model_cls(model_arch: str) -> type:
    normalized_arch = _normalize_atom_model_arch(model_arch)
    if normalized_arch is not None and normalized_arch in _ATOM_MODEL_CLASSES:
        model_ref = _ATOM_MODEL_CLASSES[normalized_arch]
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


def _safe_get_first_arch(config_like) -> str | None:
    if config_like is None:
        return None
    architectures = getattr(config_like, "architectures", None)
    if isinstance(architectures, list) and len(architectures) > 0:
        return architectures[0]
    return None


def _select_model_arch(vllm_config: VllmConfig) -> str:
    model_arch = _safe_get_first_arch(getattr(vllm_config, "model_config", None))
    if model_arch is None:
        raise ValueError("Cannot determine model architecture from vLLM model_config")
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    draft_arch = _safe_get_first_arch(draft_model_config)
    if draft_arch is None:
        return model_arch
    model_tag = None
    try:
        from vllm.compilation import backends as vllm_backends

        model_tag = getattr(vllm_backends, "model_tag", None)
    except Exception:
        pass
    if model_tag is None:
        model_tag = getattr(
            getattr(vllm_config, "compilation_config", None), "model_tag", None
        )
    if model_tag in {"eagle_head", "draft_model", "drafter"}:
        logger.info(
            f"Use draft model architecture {draft_arch} for speculative tag {model_tag}"
        )
        return draft_arch
    return model_arch


def _patch_required_act_dtype_post_load_hooks(
    module: nn.Module,
    act_dtype: torch.dtype,
) -> int:
    """Give vLLM-style post-load hooks a default dtype in plugin mode.

    ATOM's loader invokes module-level post-load hooks without arguments. Some
    vLLM modules embedded in multimodal ATOM models require an `act_dtype`
    parameter, so adapt those instances locally instead of changing the generic
    ATOM loader behavior.
    """
    import inspect

    patched = 0
    for submodule in module.modules():
        orig = getattr(submodule, "process_weights_after_loading", None)
        if orig is None or getattr(orig, "_atom_vllm_act_dtype_patched", False):
            continue

        try:
            sig = inspect.signature(orig)
        except (TypeError, ValueError):
            continue

        act_dtype_param = sig.parameters.get("act_dtype")
        if act_dtype_param is None or act_dtype_param.default is not inspect._empty:
            continue

        @functools.wraps(orig)
        def wrapped(act_dtype: torch.dtype = act_dtype, _orig=orig):
            return _orig(act_dtype)

        setattr(wrapped, "_atom_vllm_act_dtype_patched", True)
        submodule.process_weights_after_loading = wrapped
        patched += 1

    return patched


class ATOMModelBase(nn.Module, VllmModel, SupportsQuant, SupportsPP):
    # forced_model_arch: str | None = None

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from atom.config import get_current_atom_config, use_custom_atom_config

        _set_framework_backbone("vllm")

        self.cache_config = vllm_config.cache_config
        self.device_config = vllm_config.device_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.quant_config = vllm_config.quant_config
        self.vllm_compilation_config = vllm_config.compilation_config

        # Weights to skip in `self.load_weights`
        self.skip_prefixes: list[str] = []
        self.skip_substrs: list[str] = []
        self.ignore_unexpected_prefixes: list[str] = []
        self.ignore_unexpected_suffixes: list[str] = []

        self.vllm_config = vllm_config
        self.is_mtp = False
        self.is_eagle3 = False
        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config is not None:
            spec_method = speculative_config.method
            self.is_mtp = spec_method == "mtp"
            self.is_eagle3 = spec_method == "eagle3"

        main_model_arch = vllm_config.model_config.architectures[0]
        selected_model_arch = _select_model_arch(vllm_config)
        # Normalize vLLM or HF draft architecture to ATOM server-mode draft class,
        # pass through for non-draft models
        model_arch = _normalize_atom_model_arch(selected_model_arch)
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        draft_hf_config = getattr(draft_model_config, "hf_config", None)
        self.is_mtp_draft_model = self.is_mtp and selected_model_arch != main_model_arch
        self.is_eagle3_draft_model = (
            self.is_eagle3
            and selected_model_arch != main_model_arch
            and _is_eagle3_draft_arch(selected_model_arch)
        )
        self.is_spec_draft_model = self.is_mtp_draft_model or self.is_eagle3_draft_model

        if self.is_eagle3_draft_model and draft_hf_config is None:
            raise ValueError("EAGLE3 draft model config is missing hf_config")

        self.config = (
            draft_hf_config
            if self.is_eagle3_draft_model
            else vllm_config.model_config.hf_config
        )
        self.text_config = (
            self.config.get_text_config()
            if hasattr(self.config, "get_text_config")
            else self.config
        )

        if self.is_mtp_draft_model:
            # Generate separate config for main model and draft model to make sure
            # that draft model has its own compilation config rather than carried
            # over from main model. Also get the mutated hf_config from main model
            main_atom_config = get_current_atom_config()
            self.atom_config = _generate_atom_config_from_vllm_config(vllm_config)
            self.atom_config.hf_config = main_atom_config.hf_config
        elif self.is_eagle3_draft_model:
            self.atom_config = _generate_atom_config_from_vllm_config(vllm_config)
            self.atom_config.hf_config = draft_hf_config
        else:
            self.atom_config = generate_atom_config_for_plugin_mode(vllm_config)
            # root HF config so --hf-overrides survive without losing multimodal
            # sub-configs such as Kimi-K2.5's vision_config/text_config.
            self.atom_config.hf_config = self.config
        self.vllm_model_arch = selected_model_arch
        self.model_arch = model_arch
        logger.info(
            "ATOM vLLM hf config overrides: use_index_cache=%s, index_topk_freq=%s, "
            "index_topk_pattern=%s",
            getattr(self.atom_config.hf_config, "use_index_cache", None),
            getattr(self.atom_config.hf_config, "index_topk_freq", None),
            getattr(self.atom_config.hf_config, "index_topk_pattern", None),
        )
        # DeepSeek-V4's routed-expert quant scheme (FP4 vs FP8-block) is not
        # described by the checkpoint's global quantization_config, so the
        # model's auto-detection can pick the wrong spec and emit garbage. Pin
        # expert_dtype from the on-disk weights before the model (and its
        # make_v4_quant_config) is constructed.
        if model_arch == _DEEPSEEK_V4_ARCH:
            _maybe_set_v4_expert_dtype(self.atom_config, vllm_config)
        _prepare_env(atom_config=self.atom_config)
        model_cls = _get_atom_model_cls(model_arch)
        module_remapping = getattr(model_cls, "packed_modules_mapping", {})
        weights_mapper = getattr(model_cls, "hf_to_atom_mapper", {})
        self.atom_config.quant_config.remap_layer_name(
            self.atom_config.hf_config,
            packed_modules_mapping=module_remapping,
            weights_mapper=weights_mapper,
        )

        # In ATOM, quant_exclude_name_mapping is used to translate the HF module names
        # to ATOM's format. It is invoked in ATOM's model_runner initialization, but
        # lacks correspondences in vLLM. So we invoke the translation here for vLLM OOT.
        exclude_mapping = getattr(model_cls, "quant_exclude_name_mapping", {})
        # add exclude mapping for mtp layer of GLM5.
        if model_arch != main_model_arch and main_model_arch == "GlmMoeDsaForCausalLM":
            exclude_mapping.update(
                {
                    "indexers_proj": "indexer.weights_proj",
                }
            )
        if exclude_mapping and self.atom_config.quant_config is not None:
            self.atom_config.quant_config.apply_exclude_name_mapping(exclude_mapping)

        default_excludes = getattr(model_cls, "quant_default_exclude_layers", [])
        if default_excludes and self.atom_config.quant_config is not None:
            self.atom_config.quant_config.apply_default_exclude_layers(default_excludes)

        logger.info(f"Construct ATOM model {model_arch} for vLLM plugin mode")
        if self.is_spec_draft_model:
            # Draft model's layers read get_current_atom_config() to register their
            # static_forward_context, so swap out the global atom_config temporarily
            # with the draft model's atom_config so that the correct forward context
            # can be registered
            with use_custom_atom_config(self.atom_config):
                if self.is_eagle3_draft_model:
                    target_layer_num = vllm_config.model_config.get_num_layers(
                        vllm_config.parallel_config
                    )
                    logger.info(
                        "Construct EAGLE3 draft with layer_offset=%s",
                        target_layer_num,
                    )
                    self.model = model_cls(
                        self.atom_config,
                        layer_offset=target_layer_num,
                    )
                else:
                    self.model = model_cls(self.atom_config)
        else:
            self.model = model_cls(self.atom_config)

        num_patched_post_load_hooks = _patch_required_act_dtype_post_load_hooks(
            self.model,
            vllm_config.model_config.dtype,
        )
        if num_patched_post_load_hooks:
            logger.info(
                "Patched %d vLLM post-load hooks with default act_dtype "
                "inside ATOM vLLM plugin wrapper.",
                num_patched_post_load_hooks,
            )

        if model_arch in _MTP_MASK_INPUT_ARCH:
            self._adapt_mtp_layers_for_vllm()
        if self.is_eagle3_draft_model:
            self._enable_eagle3_draft_interface()
        elif self.is_eagle3 and self._eagle3_uses_aux_hidden_state():
            self._enable_eagle3_target_interface()
        if self.is_mtp or self.is_eagle3:
            # Mirror nested attributes required by vLLM speculative decoding.
            self._expose_spec_decode_attrs()

        # For sparse MLA, register the Indexer's DeepseekV32IndexerCache as
        # a virtual subclass of vLLM's AttentionLayerBase so vLLM can discover
        # it and allocate KV cache.
        self._register_indexer_caches_with_vllm()

        if self.model is None:
            raise ValueError(
                f"The model {model_arch} is not supported by model impl backend atom"
            )

        # here init aiter dist for using aiter custom collective ops
        self.pp_group = get_pp_group()
        self.tp_group = get_tp_group()

        # DeepSeek-V4 is a native ATOM model: its forward reads ATOM's *own*
        # forward context (input_ids for hash-MoE routing, indexer/attention
        # metadata), which vLLM's runner never populates. The plugin bridges
        # this — register the proxy KV layer now, then per-forward bind the
        # proxy cache views and enter `atom_deepseek_v4_forward_context`
        # (see `forward`). Other ATOM models follow vLLM's contract directly.
        self._is_deepseek_v4 = self.model_arch == _DEEPSEEK_V4_ARCH
        if self._is_deepseek_v4:
            from atom.plugin.vllm.deepseek_v4_bridge import (
                register_deepseek_v4_proxy_layer,
            )

            register_deepseek_v4_proxy_layer(vllm_config)

    # Attributes whose writes on the outer model must propagate to the
    # inner model so vLLM's weight-sharing reaches the forward path.
    _WEIGHT_SHARED_ATTRS = frozenset({"embed_tokens", "embedding", "lm_head"})
    # Attribute names under which ATOM models nest their inner backbone. Walked
    # in order at each level to build the inner-model chain. To support a model
    # that nests under a new name, add it here.
    _INNER_MODEL_ATTRS = ("model", "language_model")

    def _inner_model_chain(self, outer: nn.Module) -> list[nn.Module]:
        """`outer` followed by each nested backbone, deepest last.

        ATOM models nest their language backbone at different names and depths:
          - flat EAGLE3 draft (Eagle3LlamaModel): depth 0, attrs on `outer`
          - MTP draft / text-only target: depth 1, under `.model`
          - VL target (MiniMax-M3 ...TextOnly): depth 2,
            `.language_model` then `.model`
        Walking `_INNER_MODEL_ATTRS` at each level yields a single chain that
        covers all of them, so a wanted attribute can be found wherever it lives.
        """
        chain: list[nn.Module] = []
        node: nn.Module | None = outer
        seen: set[int] = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            chain.append(node)
            nxt = None
            for name in self._INNER_MODEL_ATTRS:
                cand = getattr(node, name, None)
                if isinstance(cand, nn.Module) and id(cand) not in seen:
                    nxt = cand
                    break
            node = nxt
        return chain

    def _expose_spec_decode_attrs(self) -> None:
        """"Bridge the extra nesting level between vLLM and ATOM for spec decode.

        vLLM reads embed_tokens / embedding / layers at ``wrapper.model.<attr>``
        and the head at ``wrapper.lm_head``. ATOM nests these under one or more
        backbone levels (see `_inner_model_chain`), so mirror the first holder
        found in the chain onto the paths vLLM reads.

        Draft vs target differ in one way:
        - Draft/MTP: vLLM *replaces* embed_tokens / lm_head with shared target
          weights, so register real submodules (`setattr`) and install a
          `__setattr__` hook that propagates those writes down to the inner
          module the forward path reads from.
        - Target: vLLM only *reads* these attrs, so mirror them as plain aliases
          (`object.__setattr__`) that stay invisible to the module tree and the
          weight loader — no ownership change, no sync hook.
        """
        model = self.model
        chain = self._inner_model_chain(model)
        is_draft = self.is_spec_draft_model
        put = setattr if is_draft else object.__setattr__

        def first_holder(attr: str) -> nn.Module | None:
            for node in chain:
                if hasattr(node, attr):
                    return node
            return None

        # (1) Mirror backbone attrs onto the outer model, and the head onto self.
        for attr in (*self._WEIGHT_SHARED_ATTRS, "layers"):
            if not hasattr(model, attr):
                holder = first_holder(attr)
                if holder is not None and holder is not model:
                    put(model, attr, getattr(holder, attr))

        if not hasattr(self, "lm_head"):
            holder = first_holder("lm_head")
            if holder is not None:
                put(self, "lm_head", getattr(holder, "lm_head"))

        # (2) Draft only: propagate vLLM's later writes on the outer model down
        #     to the inner module the forward path reads from. Create the one-off
        #     subclass only once, and only when there is an inner level to sync.
        if not is_draft:
            return
        inner = chain[1] if len(chain) > 1 else None
        if inner is None:
            return
        if getattr(model, "_atom_vllm_shared_attr_sync_patched", False):
            return
        shared = self._WEIGHT_SHARED_ATTRS
        base_setattr = model.__class__.__setattr__

        def _syncing_setattr(self_model, name, value):
            base_setattr(self_model, name, value)
            if name in shared and hasattr(inner, name):
                base_setattr(inner, name, value)

        base_setattr(model, "_atom_vllm_shared_attr_sync_patched", True)
        # Make the one-off subclass report its actual module instead of the
        # base wrapper's
        model.__class__ = type(
            model.__class__.__name__,
            (model.__class__,),
            {
                "__module__": model.__class__.__module__,
                "__setattr__": _syncing_setattr,
            },
        )

    def _register_indexer_caches_with_vllm(self):
        """Register DeepseekV32IndexerCache instances with vLLM so that:
        1. vLLM discovers them via isinstance(AttentionLayerBase) for KV cache
           allocation (get_kv_cache_spec iterates static_forward_context)
        2. bind_kv_cache() can find them in vLLM's static_forward_context to
           assign the allocated KV cache tensor
        3. The indexer's metadata lookup uses the correct prefix in vLLM's
           attn_metadata dict

        ATOM's DeepseekV32IndexerCache inherits from nn.Module (not vLLM's
        AttentionLayerBase), so we register it as a virtual subclass.
        We also register each instance in vLLM's static_forward_context using
        the same prefix convention as other attention layers (the prefix
        parameter passed at construction, e.g. 'model.layers.0...k_cache').
        """
        from atom.models.deepseek_v2 import DeepseekV32IndexerCache

        # Find indexer cache instances. module.prefix is the ATOM-internal
        # prefix set during __init__ (e.g. "model.layers.0.self_attn.indexer.k_cache").
        indexer_caches = []
        for _name, module in self.model.named_modules():
            if isinstance(module, DeepseekV32IndexerCache):
                indexer_caches.append(module)

        if not indexer_caches:
            return

        try:
            from vllm.model_executor.layers.attention_layer_base import (
                AttentionLayerBase,
            )

            # Register DeepseekV32IndexerCache as a virtual subclass of
            # AttentionLayerBase so vLLM's isinstance() check passes.
            AttentionLayerBase.register(DeepseekV32IndexerCache)
            logger.info(
                "Registered DeepseekV32IndexerCache as AttentionLayerBase "
                "virtual subclass for vLLM KV cache allocation"
            )
        except ImportError:
            logger.warning(
                "Could not import AttentionLayerBase from vLLM. "
                "Indexer cache will not be managed by vLLM."
            )
            return

        # Register each indexer cache in vLLM's static_forward_context.
        # Use module.prefix (the ATOM-internal prefix), which follows the same
        # convention as vLLM's MLAAttention layers that self-register with
        # their prefix parameter (e.g. "model.layers.0.self_attn.attn").
        vllm_sfc = self.vllm_compilation_config.static_forward_context
        for module in indexer_caches:
            # MTP draft models own a separate atom_config/static_forward_context.
            # Keep that ownership on the cache so metadata builders can bind
            # sparse buffers back to the draft modules instead of the main model.
            module.atom_config = self.atom_config
            prefix = module.prefix
            if prefix not in vllm_sfc:
                vllm_sfc[prefix] = module
                logger.info(
                    f"Registered indexer cache in vLLM static_forward_context: {prefix}"
                )
            else:
                logger.warning(
                    f"Indexer cache {prefix} already in vLLM "
                    f"static_forward_context, skipping"
                )

    def _adapt_mtp_layers_for_vllm(self) -> None:
        """Install vLLM-only MTP input masking without changing model code."""
        if not self.is_mtp_draft_model:
            return

        inner_model = getattr(self.model, "model", None)
        layers = (
            getattr(inner_model, "layers", None) if inner_model is not None else None
        )
        if layers is None:
            return

        layer_iter = layers.values() if isinstance(layers, nn.ModuleDict) else layers
        for layer in layer_iter:
            if getattr(layer, "_atom_vllm_mtp_masked", False):
                continue

            layer.forward = types.MethodType(
                self._make_vllm_mtp_layer_forward(layer.forward),
                layer,
            )
            layer._atom_vllm_mtp_masked = True

    @staticmethod
    def _make_vllm_mtp_layer_forward(original_forward):
        @functools.wraps(original_forward)
        def masked_forward(
            self_layer,
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            spec_step_index=0,
        ):
            inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
            return original_forward(
                input_ids,
                positions,
                previous_hidden_states,
                inputs_embeds,
                spec_step_index,
            )

        return masked_forward

    def _eagle3_uses_aux_hidden_state(self) -> bool:
        vllm_spec_config = getattr(self.vllm_config, "speculative_config", None)
        if getattr(vllm_spec_config, "method", None) != "eagle3":
            return False
        draft_model_config = getattr(vllm_spec_config, "draft_model_config", None)
        hf_config = getattr(draft_model_config, "hf_config", None)
        eagle_config = getattr(hf_config, "eagle_config", None)
        if isinstance(eagle_config, dict):
            return eagle_config.get("use_aux_hidden_state", True)
        return True

    def _enable_eagle3_target_interface(self) -> None:
        """Expose vLLM's SupportsEagle3 target surface by bridging to the inner
        ATOM model's server-mode aux_hidden_state interface.
        ATOM target models follow the server-mode convention, exposing
        `set_aux_hidden_state_layers` and `get_eagle3_aux_hidden_state_layers`.
        vLLM's SupportsEagle3 instead calls `set_aux_hidden_state_layers` and
        `get_eagle3_default_aux_hidden_state_layers`.
        """
        model = self.model
        if not (
            callable(getattr(model, "set_aux_hidden_state_layers", None))
            and callable(getattr(model, "get_eagle3_aux_hidden_state_layers", None))
        ):
            raise RuntimeError(
                f"Model {self.model_arch} cannot serve as an EAGLE3 target: it "
                "does not expose the ATOM server-mode aux-hidden-state interface "
                "(set_aux_hidden_state_layers / get_eagle3_aux_hidden_state_layers)."
            )
        self.supports_eagle3 = True
        self.has_own_lm_head = False
        self.has_own_embed_tokens = False
        self.set_aux_hidden_state_layers = model.set_aux_hidden_state_layers
        self.get_eagle3_default_aux_hidden_state_layers = (
            self._resolve_eagle3_aux_hidden_state_layers
        )

    def _resolve_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        # Following ATOM server mode, perfer the draft's configured IDs that
        # are already resolved from the possibly nested eagle_config by ATOM's
        # SpeculativeConfig.__post_init__, and fall back to the target model's
        # architecture default
        spec_config = getattr(self.atom_config, "speculative_config", None)
        aux_ids = list(getattr(spec_config, "eagle3_aux_layer_ids", None) or [])
        if aux_ids:
            return tuple(aux_ids)
        return tuple(self.model.get_eagle3_aux_hidden_state_layers())

    def _enable_eagle3_draft_interface(self) -> None:
        # Expose vLLM's EAGLE3 draft `combine_hidden_states` by forwarding it to
        # the inner ATOM draft model
        model = self.model
        if not callable(getattr(model, "combine_hidden_states", None)):
            raise RuntimeError(
                f"Model {self.model_arch} cannot serve as an EAGLE3 draft: it "
                "does not implement combine_hidden_states()."
            )
        self.has_own_lm_head = False
        self.has_own_embed_tokens = False
        self.combine_hidden_states = model.combine_hidden_states
        self._maybe_index_draft_attn_layer()

    def _maybe_index_draft_attn_layer(self) -> None:
        # vLLM's bind_kv_cache calls extract_layer_index which asserts that
        # each kv cache layer name contains only one integer. ATOM's
        # Eagle3LlamaModel names its decoder layer as "midlayer", so prefix
        # it with "layers.0." so that vLLM's assertion can pass
        static_forward_context = self.vllm_compilation_config.static_forward_context

        for _name, module in self.model.named_modules():
            old_name = getattr(module, "layer_name", None)
            if old_name is None or any(p.isdigit() for p in old_name.split(".")):
                continue
            new_name = f"layers.0.{old_name}"
            if new_name in static_forward_context:
                raise ValueError(
                    f"Cannot re-key draft attention layer {old_name} to "
                    f"{new_name}; name already registered."
                )
            static_forward_context[new_name] = static_forward_context.pop(old_name)
            module.layer_name = new_name
            logger.info(
                f"Re-keyed EAGLE3 draft attention layer {old_name} to "
                f"{new_name} for vLLM to extract a layer index"
            )

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

        # pass positions from vLLM to OOT execution path via vLLM's per-forward context
        if is_forward_context_available():
            forward_context = get_vllm_forward_context()
            forward_context.additional_kwargs["atom_positions"] = positions
            # set atom_config into vLLM forward_context in order to
            # make sure main model and draft model can get their specific
            # static_forward_context from their own atom_config
            forward_context.additional_kwargs["atom_config"] = self.atom_config
        elif "positions" in self.atom_config.compilation_config.static_forward_context:
            buf = self.atom_config.compilation_config.static_forward_context[
                "positions"
            ]
            buf[: positions.numel()].copy_(positions)

        if self.is_eagle3_draft_model:
            if inputs_embeds is not None:
                raise NotImplementedError(
                    "ATOM EAGLE3 draft wrappers do not support multimodal "
                    "inputs_embeds in vLLM plugin mode yet."
                )
            if "hidden_states" not in model_kwargs:
                raise ValueError("EAGLE3 draft forward requires hidden_states.")
            hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=model_kwargs["hidden_states"],
            )
            if not isinstance(hidden_states, tuple):
                hidden_states = (hidden_states, hidden_states)
        elif self._is_deepseek_v4:
            # DeepSeek-V4 is a native ATOM model: it reads ATOM's own forward
            # context and takes a native (input_ids, positions) forward — vLLM's
            # generic call contract (intermediate_tensors/inputs_embeds) does not
            # apply (V4 is TP-only, text-only). Bind the proxy cache views and
            # enter `atom_deepseek_v4_forward_context` so ATOM's Context (the
            # input_ids hash-MoE routing key) and chunk-aware attention metadata
            # are populated before the (possibly graph-captured) forward runs.
            from atom.plugin.vllm.deepseek_v4_bridge import (
                atom_deepseek_v4_forward_context,
                bind_deepseek_v4_proxy_cache_views,
            )

            ready = bind_deepseek_v4_proxy_cache_views(self.model, self.vllm_config)
            # Per-request stable state slots + chunk-aware metadata + selective
            # reset are driven from the allocator/params stashed at bind time.
            # Only engage them once the proxy cache is bound (real forwards);
            # dummy/profile forwards fall back to arange slots with no reset.
            slot_allocator = (
                getattr(self.model, "_atom_v4_slot_allocator", None) if ready else None
            )
            meta_params = (
                getattr(self.model, "_atom_v4_meta_params", None) if ready else None
            )
            with atom_deepseek_v4_forward_context(
                atom_config=self.atom_config,
                input_ids=input_ids,
                positions=positions,
                force_dummy=not ready,
                state_model=self.model if ready else None,
                meta_params=meta_params,
                slot_allocator=slot_allocator,
            ):
                hidden_states = self.model(input_ids=input_ids, positions=positions)
        else:
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
        # prevent circular import
        from atom.model_loader.loader import load_model_in_plugin_mode

        is_mtp_draft_model = self.model_arch in {
            "DeepSeekMTPModel",
            "Qwen3NextMTP",
            "Glm4MoeMTPModel",
        }
        draft_hf_config = None
        draft_model_path = None
        if is_mtp_draft_model:
            draft_model_config = getattr(
                getattr(self.atom_config, "speculative_config", None),
                "draft_model_config",
                None,
            )
            if draft_model_config is not None:
                draft_hf_config = getattr(
                    draft_model_config, "hf_config", draft_model_config
                )
        if self.is_eagle3_draft_model:
            # EAGLE3 drafts are standalone checkpoints, so we need both the draft
            # hf_config and the draft checkpoint path
            spec_config = getattr(self.vllm_config, "speculative_config", None)
            draft_model_config = getattr(spec_config, "draft_model_config", None)
            if draft_model_config is not None:
                draft_hf_config = getattr(
                    draft_model_config, "hf_config", draft_model_config
                )
                draft_model_path = getattr(
                    draft_model_config, "model", None
                ) or getattr(spec_config, "model", None)
            if not draft_model_path:
                raise ValueError("EAGLE3 draft model path is missing.")

        loaded_weights_record = load_model_in_plugin_mode(
            model=self.model,
            config=self.atom_config,
            prefix="model.",
            spec_decode=is_mtp_draft_model,
            hf_config_override=draft_hf_config,
            model_name_or_path_override=draft_model_path,
        )
        if self.is_eagle3_draft_model:
            self.has_own_embed_tokens = any(
                "embed_tokens" in name for name in loaded_weights_record
            )
            self.has_own_lm_head = any(
                "lm_head" in name for name in loaded_weights_record
            )
            self.model.has_own_embed_tokens = self.has_own_embed_tokens
            self.model.has_own_lm_head = self.has_own_lm_head
        return loaded_weights_record

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.model.compute_logits(hidden_states)
        return logits


class ATOMForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMMoEForCausalLM(ATOMModelBase, VllmModelForTextGeneration): ...


class ATOMForConditionalGeneration(
    ATOMModelBase, VllmModelForTextGeneration, SupportsMultiModal, SupportsMRoPE
):
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        """
        Get the placeholder text for the `i`th `modality` item in the prompt.
        """
        raise NotImplementedError

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        return self.model.embed_multimodal(**kwargs)

    def configure_mm_token_handling(self, vocab_size, mm_token_ids):
        return self.model.configure_mm_token_handling(vocab_size, mm_token_ids)

    def get_language_model(self):
        return self.model.get_language_model()

    def get_num_mm_encoder_tokens(self, num_image_tokens):
        return self.model.get_num_mm_encoder_tokens(num_image_tokens)

    def get_num_mm_connector_tokens(self, num_vision_tokens):
        return self.model.get_num_mm_connector_tokens(num_vision_tokens)

    def embed_input_ids(
        self, input_ids, multimodal_embeddings=None, *, is_multimodal=None
    ):
        return self.model.embed_input_ids(
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

    def _embed_text_input_ids(self, input_ids, embed_input_ids, *, is_multimodal):
        return self.model._embed_text_input_ids(
            input_ids, embed_input_ids, is_multimodal=is_multimodal
        )

    def get_mrope_input_positions(self, input_tokens, mm_features):
        return self.model.get_mrope_input_positions(input_tokens, mm_features)

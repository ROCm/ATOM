"""Model-level GLM-5.2 DSA adaptation for SGLang plugin mode."""

from __future__ import annotations

from typing import Any

import torch

from atom.plugin.sglang.models.deepseek_mla import (
    _align_qknorm_fusion_for_sglang,
    _patch_indexer_layernorm_for_sglang,
)
from atom.plugin.sglang.models.deepseek_mla_forward import (
    _patch_attention_projs_for_sglang_mxfp4,
)
from atom.plugin.sglang.models.glm52_dsa_attention import (
    SGLangATOMGLM52MLAAttention,
)


def _in_cuda_graph_capture() -> bool:
    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_is_capture_mode,
        )
    except ImportError:
        return False
    return bool(get_is_capture_mode())


def _capture_attention_variant() -> str | None:
    """SGLang's dual-graph label, when this forward is inside CUDA-graph capture."""

    try:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            get_capture_attention_variant,
        )
    except ImportError:
        return None
    if not _in_cuda_graph_capture():
        return None
    return get_capture_attention_variant()


def glm52_short_context_is_dense(forward_batch: Any, topk_tokens: int) -> bool:
    """True when GLM should read the dense KV table instead of top-k indices.

    Kimi does not set the opt-in flag, so its plugin indexer is unchanged.
    Single-token decode stays on the indexer. A dense CUDA-graph variant of
    verify or extend is dense; the sparse variant is not.
    """

    if forward_batch is None or int(topk_tokens) <= 0:
        return False
    from atom.plugin.sglang.attention_backend.sparse_mla_indexer import (
        sparse_attn_indexer_sglang_plugin_mode,
    )

    if not getattr(
        sparse_attn_indexer_sglang_plugin_mode, "_glm52_skip_short_topk", False
    ):
        return False
    mode = getattr(forward_batch, "forward_mode", None)
    is_verify = bool(getattr(mode, "is_target_verify", lambda: False)())
    is_decode = bool(getattr(mode, "is_decode_or_idle", lambda: False)())
    # FP8 GSM8K is one query token. Replacing its indexer with a dense fill
    # leaves sparse MLA empty and the score falls to chance.
    if is_decode and not is_verify:
        return False
    variant = _capture_attention_variant()
    if variant == "sparse":
        return False
    if variant == "dense":
        return True
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    cpu = getattr(forward_batch, "seq_lens_cpu", None)
    if cpu is None or bs <= 0:
        return False
    try:
        seq_max = int(max(cpu[:bs]))
    except Exception as exc:
        if "capturing" not in str(exc):
            raise
        return False
    if seq_max <= int(topk_tokens):
        return True
    from atom.plugin.sglang.glm52_dsa_bridge import GLM52_GRAPH_SEQ_LEN_CAPACITY

    extend = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if extend is None:
        return False
    try:
        extend_max = int(max(extend[:bs]))
    except Exception as exc:
        if "capturing" not in str(exc):
            raise
        return False
    fill = int(GLM52_GRAPH_SEQ_LEN_CAPACITY)
    return seq_max == fill and 0 < extend_max < fill


def short_verify_skips_indexer(
    metadata: Any,
    *,
    is_prefill: bool,
    dcp_world_size: int,
    topk_tokens: int,
) -> bool:
    """Skip top-k only for a multi-token verify whose context is inside top-k.

    GLM-FP8 decode is one query token. Skipping the indexer there leaves sparse
    MLA with an empty index buffer, and GSM8K falls to chance.
    """

    if is_prefill or int(dcp_world_size) != 1 or int(topk_tokens) <= 0:
        return False
    if metadata is None:
        return False
    max_seqlen_q = int(getattr(metadata, "max_seqlen_q", 1) or 1)
    max_seqlen_k = int(getattr(metadata, "max_seqlen_k", 0) or 0)
    return max_seqlen_q > 1 and max_seqlen_k <= int(topk_tokens)


def _install_short_verify_indexer_skip(module: Any) -> None:
    """Skip paged MQA top-k when the verify context is already inside top-k.

    The native op writes that score on the decode path. Below ``index_topk``
    every token is selected, so attention reads the dense KV table instead.
    Installed on the plugin indexer object; ``deepseek_v2.py`` stays unchanged.
    """

    indexer = getattr(module, "indexer", None)
    if indexer is None or getattr(indexer, "_atom_glm52_short_verify_skip", False):
        return
    if getattr(module, "skip_topk", False):
        return

    original = indexer.sparse_attn_indexer_impl

    def _sparse_attn_indexer(*args: Any, **kwargs: Any):
        from atom.distributed.dcp_utils import get_dcp_world_size
        from atom.plugin.sglang.glm52_dsa_bridge import (
            clamp_published_indexer_context,
            mask_oob_indexer_pages,
        )
        from atom.plugin.sglang.runtime.attention_backend_resolver import (
            resolve_sglang_runtime,
        )
        from atom.plugin.sglang.runtime.context import get_current_forward_batch
        from atom.utils.forward_context import get_forward_context

        context = get_forward_context()
        metadata = getattr(context, "attn_metadata", None)
        forward_batch = get_current_forward_batch()
        is_prefill = bool(
            getattr(getattr(context, "context", None), "is_prefill", False)
        )
        topk_tokens = int(getattr(indexer, "topk_tokens", 0) or 0)
        # Multi-token verify inside index_topk has no selection to score.
        # Single-token FP8 decode must still call the indexer: skipping it
        # leaves sparse MLA empty. The clamp's host copy is illegal while
        # the decode graph is capturing, so capture skips only that copy.
        if short_verify_skips_indexer(
            metadata,
            is_prefill=is_prefill,
            dcp_world_size=get_dcp_world_size(),
            topk_tokens=topk_tokens,
        ):
            return None
        if (
            not _in_cuda_graph_capture()
            and metadata is not None
            and forward_batch is not None
        ):
            runtime = resolve_sglang_runtime(forward_batch)
            try:
                clamped = clamp_published_indexer_context(
                    metadata, forward_batch, runtime.req_to_token_pool
                )
            except Exception as exc:
                if "capturing" not in str(exc):
                    raise
                clamped = False
            if clamped:
                try:
                    mask_oob_indexer_pages(metadata, runtime.token_to_kv_pool)
                except Exception as exc:
                    if "capturing" not in str(exc):
                        raise
            if short_verify_skips_indexer(
                metadata,
                is_prefill=is_prefill,
                dcp_world_size=get_dcp_world_size(),
                topk_tokens=topk_tokens,
            ):
                return None
            if not is_prefill:
                # top_k_per_row_decode uses seq - next_n + (row % next_n) + 1.
                # A padded row shorter than next_n wraps that to a huge length.
                next_n = int(getattr(metadata, "max_seqlen_q", 1) or 1)
                context_lens = getattr(metadata, "context_lens", None)
                if (
                    next_n > 1
                    and torch.is_tensor(context_lens)
                    and int(context_lens.numel())
                ):
                    context_lens.clamp_(min=next_n)
        return original(*args, **kwargs)

    indexer.sparse_attn_indexer_impl = _sparse_attn_indexer
    indexer._atom_glm52_short_verify_skip = True
    from atom.plugin.sglang.attention_backend.sparse_mla_indexer import (
        sparse_attn_indexer_sglang_plugin_mode,
    )

    sparse_attn_indexer_sglang_plugin_mode._glm52_skip_short_topk = True


def setup_glm52_dsa_for_sglang(model: Any) -> None:
    """Patch GLM-5.2 for native ATOM sparse MLA under SGLang.

    This deliberately does not install ``SGLangDeepseekMLAAttention``.  GLM-5.2
    should keep ATOM's native ``MLAAttention`` frontend so full-index layers run
    the ATOM indexer into a shared physical-index buffer and shared-index layers
    reuse that buffer.
    """

    if not hasattr(model, "atom_config"):
        from atom.config import get_current_atom_config

        model.atom_config = get_current_atom_config()

    from atom.models.deepseek_v2 import DeepseekV2MLAAttention

    try:
        from sglang.srt.configs.model_config import is_deepseek_dsa
    except ImportError:
        from sglang.srt.configs.model_config import is_deepseek_nsa as is_deepseek_dsa

    from sglang.srt.layers.communicator import get_attn_tp_context

    config = model.config
    get_attn_tp_context().init_context(config.q_lora_rank, is_deepseek_dsa(config))

    last_full_index_seen = False
    for module in model.modules():
        if not isinstance(module, DeepseekV2MLAAttention):
            continue

        _align_qknorm_fusion_for_sglang(module)
        _patch_attention_projs_for_sglang_mxfp4(module)
        _patch_indexer_layernorm_for_sglang(module)

        if not isinstance(module.mla_attn, SGLangATOMGLM52MLAAttention):
            raise TypeError(
                "GLM-5.2 SGLang native DSA setup expected "
                "SGLangATOMGLM52MLAAttention. Ensure the GLM construction "
                "context is installed before model initialization."
            )

        if getattr(module, "is_v32", False):
            _install_short_verify_indexer_skip(module)
            owns_active_indexer = getattr(
                module, "indexer", None
            ) is not None and not getattr(module, "skip_topk", False)
            if owns_active_indexer:
                last_full_index_seen = True
            elif not last_full_index_seen:
                raise RuntimeError(
                    "GLM-5.2 IndexShare cannot start with a shared-index layer; "
                    f"layer={getattr(module, 'prefix', '<unknown>')!r}"
                )

    model._atom_sglang_uses_glm52_native_dsa = True

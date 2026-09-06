"""ATOM DeepSeek-V4 patches for vLLM's throwaway cudagraph-profiling KV cache.

vLLM sizes it at one block per sequence, which is short for the V4 proxy page.
Both patches below no-op for non-V4 models. Nothing here concerns prefix
caching; see ``deepseek_v4_prefix_patch`` for that.
"""

import functools
import logging

logger = logging.getLogger("atom")


def _mark_v4_proxy_cache_mode(static_forward_context, is_profiling: bool) -> None:
    for layer in static_forward_context.values():
        if getattr(layer, "_atom_v4_proxy_layer", False):
            layer._atom_v4_profiling_kv_cache = is_profiling


def apply_vllm_v4_profile_cache_patch() -> None:
    """Mark vLLM 0.26's temporary CUDA-graph profiling KV cache.

    The temporary cache intentionally contains only one block per captured
    request and cannot hold V4's fixed per-request SWA arena. The V4 forward
    must therefore stay on its existing dummy-attention path until vLLM
    installs the real cache.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.initialize_kv_cache
    if getattr(original, "_atom_v4_profile_cache_patched", False):
        return

    @functools.wraps(original)
    def wrapped_initialize_kv_cache(
        self,
        kv_cache_config,
        is_profiling: bool = False,
    ):
        result = original(
            self,
            kv_cache_config,
            is_profiling=is_profiling,
        )
        _mark_v4_proxy_cache_mode(
            self.compilation_config.static_forward_context,
            is_profiling,
        )
        return result

    wrapped_initialize_kv_cache._atom_v4_profile_cache_patched = True
    GPUModelRunner.initialize_kv_cache = wrapped_initialize_kv_cache


def apply_vllm_v4_profiling_min_blocks_patch(vllm_config=None) -> None:
    """Make vLLM's cudagraph-profiling KV cache big enough for the V4 proxy.

    ``_init_minimal_kv_cache_for_profiling`` allocates one block per sequence,
    which only works for a backend whose per-page bytes are self-contained. The
    V4 proxy page is not -- ``_proxy_page_bytes`` amortizes the SWA ring across
    ``ceil(max_model_len / 128)`` pages -- so less raises "proxy cache too
    small". Raise the floor to that count; the blocks are transient.

    ``vllm_config`` only seeds the log line; the floor is re-derived at call
    time so this stays installable from ``register_model()``.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from atom.plugin.vllm.deepseek_v4_bridge import _v4_proxy_min_blocks

    original = getattr(GPUModelRunner, "_init_minimal_kv_cache_for_profiling", None)
    if original is None or getattr(original, "_atom_v4_min_blocks_patched", False):
        return

    def _proxy_min_blocks_for(config) -> int:
        mc = getattr(config, "model_config", None)
        if mc is None or not any(
            "DeepseekV4" in str(a) for a in (getattr(mc, "architectures", None) or [])
        ):
            return 0
        return _v4_proxy_min_blocks(config)

    # The count is computed inside the original and the inputs that would steer
    # it also size the metadata builders it constructs, so intercept the one
    # thing carrying the count: the KVCacheConfig it builds.
    from vllm.v1.core import kv_cache_utils

    @functools.wraps(original)
    def wrapped_init_minimal_kv_cache_for_profiling(self):
        floor = _proxy_min_blocks_for(getattr(self, "vllm_config", None))
        if floor <= 0:
            return original(self)

        inner = kv_cache_utils.get_kv_cache_config_from_groups

        @functools.wraps(inner)
        def floored(*args, **kwargs):
            config = inner(*args, **kwargs)
            old = int(config.num_blocks)
            if old <= 0 or old >= floor:
                return config
            for tensor in config.kv_cache_tensors:
                tensor.size = (tensor.size // old) * floor
            config.num_blocks = floor
            return config

        kv_cache_utils.get_kv_cache_config_from_groups = floored
        try:
            return original(self)
        finally:
            kv_cache_utils.get_kv_cache_config_from_groups = inner

    wrapped_init_minimal_kv_cache_for_profiling._atom_v4_min_blocks_patched = True
    GPUModelRunner._init_minimal_kv_cache_for_profiling = (
        wrapped_init_minimal_kv_cache_for_profiling
    )
    logger.info(
        "ATOM DeepSeek-V4: cudagraph-profiling KV cache floor installed "
        "(%s; the proxy page amortizes the SWA ring over exactly that many).",
        (
            f"{_proxy_min_blocks_for(vllm_config)} blocks"
            if vllm_config is not None
            else "resolved per model runner"
        ),
    )

import functools
import logging

logger = logging.getLogger("atom")


def _patch_eagle3_model_type_checks() -> None:
    # vLLM's V1 EAGLE proposer SpecDecodeBaseProposer.propose() has an explicit
    # isinstance() check for native vLLM EAGLE3 model classes before calling
    # `combine_hidden_states()`. ATOM's vLLM plugin mode provides the same behavior
    # through the ATOMModelBase wrapper, so patch the type checks to accept the
    # ATOMModelBase wrapper
    try:
        from atom.plugin.vllm.model_wrapper import ATOMModelBase
        import vllm.v1.spec_decode.llm_base_proposer as llm_base_proposer
    except Exception:
        logger.warning(
            "vLLM plugin: failed to patch vLLM V1 EAGLE3 proposer type checks. "
            "This can happen if you are using an in-compatible vLLM version. "
            "Please make sure that the correct vLLM version is installed."
        )
        return

    if getattr(llm_base_proposer, "_atom_eagle3_model_types_patched", False):
        return

    # Supported archs in vLLM's `llm_base_proposer.py`
    for name in ("Eagle3LlamaForCausalLM", "Eagle3DeepseekV2ForCausalLM"):
        original = getattr(llm_base_proposer, name, None)
        if original is None:
            continue
        if isinstance(original, tuple):
            widened = (*original, ATOMModelBase)
        else:
            widened = (original, ATOMModelBase)
        setattr(llm_base_proposer, name, widened)

    setattr(llm_base_proposer, "_atom_eagle3_model_types_patched", True)
    logger.info("ATOM plugin: patched vLLM EAGLE3 proposer type checks.")


def _get_attn_backend_block_size(backend) -> int:
    supported = backend.get_supported_kernel_block_sizes()
    get_preferred = getattr(backend, "get_preferred_block_size", None)
    if get_preferred is None:
        return supported[0]
    return get_preferred(supported[0])


@functools.cache
def _get_mla_block_size() -> int:
    from atom.plugin.vllm.attention.backend import AiterMlaBackendForVllm

    return _get_attn_backend_block_size(AiterMlaBackendForVllm)


@functools.cache
def _get_mha_block_size() -> int:
    from atom.plugin.vllm.attention.backend import AiterMhaBackendForVllm

    return _get_attn_backend_block_size(AiterMhaBackendForVllm)


def _spec_has_heterogeneous_mla_mha_backend(kv_cache_spec) -> bool:
    try:
        from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec
    except Exception:
        return False

    has_mla = False
    has_non_mla_attn = False
    for spec in kv_cache_spec.values():
        if isinstance(spec, MLAAttentionSpec):
            has_mla = True
        elif isinstance(spec, AttentionSpec):
            has_non_mla_attn = True
    return has_mla and has_non_mla_attn


def _split_mla_and_mha_layers(kv_cache_spec):
    from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

    mla_layers = {}
    mha_layers = {}
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, MLAAttentionSpec):
            mla_layers[name] = spec
        elif isinstance(spec, AttentionSpec):
            mha_layers[name] = spec
        else:
            raise NotImplementedError(
                "The heterogeneous EAGLE3 KV pool only supports MLA target with "
                f"MHA draft, but got unexpected spec {type(spec).__name__} for "
                f"layer {name}."
            )
    return mla_layers, mha_layers


def _build_heterogeneous_kv_cache_groups(kv_cache_spec):
    # Build separate groups for MLA and MHA with distinct block sizes and page sizes
    # to bypass page size unification.
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, UniformTypeKVCacheSpecs

    mla_layers, mha_layers = _split_mla_and_mha_layers(kv_cache_spec)
    assert mla_layers, "Heterogeneous EAGLE3 requires at least 1 MLA layer"
    assert mha_layers, "Heterogeneous EAGLE3 requires at least 1 MHA layer"

    # Use UniformTypeKVCacheSpecs so per-layer page sizes are preserved even
    # if individual MLA layers differ, though they should be identical.
    mla_specs = {
        name: spec.copy_with_new_block_size(_get_mla_block_size())
        for name, spec in mla_layers.items()
    }
    mla_uniform = UniformTypeKVCacheSpecs.from_specs(mla_specs)
    assert mla_uniform is not None, (
        "Failed to build UniformTypeKVCacheSpecs for MLA target layers"
    )
    mla_group = KVCacheGroupSpec(
        layer_names=list(mla_specs.keys()),
        kv_cache_spec=mla_uniform,
    )

    mha_specs = [
        spec.copy_with_new_block_size(_get_mha_block_size())
        for spec in mha_layers.values()
    ]
    merged_mha = mha_specs[0].merge(mha_specs)
    mha_group = KVCacheGroupSpec(
        layer_names=list(mha_layers.keys()),
        kv_cache_spec=merged_mha,
    )

    return [mla_group, mha_group]


def _groups_are_heterogeneous_mla_mha(kv_cache_groups) -> bool:
    try:
        from vllm.v1.kv_cache_interface import (
            AttentionSpec,
            MLAAttentionSpec,
            UniformTypeKVCacheSpecs,
        )
    except Exception:
        logger.warning(
            "vLLM plugin: failed to recognize ATOM heterogeneous EAGLE3 KV pool. "
            "This can happen if you are using an in-compatible vLLM version. "
            "Please make sure that the correct vLLM version is installed."
        )
        return False

    if len(kv_cache_groups) != 2:
        return False

    def _is_mla(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            specs = list(spec.kv_cache_specs.values())
            return bool(specs) and all(isinstance(s, MLAAttentionSpec) for s in specs)
        return isinstance(spec, MLAAttentionSpec)

    def _is_mha(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            specs = list(spec.kv_cache_specs.values())
            return bool(specs) and all(
                isinstance(s, AttentionSpec) and not isinstance(s, MLAAttentionSpec)
                for s in specs
            )
        return isinstance(spec, AttentionSpec) and not isinstance(
            spec, MLAAttentionSpec
        )

    g0, g1 = kv_cache_groups
    return (_is_mla(g0) and _is_mha(g1)) or (
        _is_mla(g1) and _is_mha(g0)
    )


def _build_heterogeneous_kv_cache_config_from_groups(
    vllm_config, kv_cache_groups, available_memory
):
    # Custom kv cache allocator for mixed mla/mha target/draft layout.
    # Allocates a single number of blocks for all layers of both groups
    from vllm.v1.core.kv_cache_utils import may_override_num_blocks
    from vllm.v1.kv_cache_interface import (
        KVCacheConfig,
        KVCacheTensor,
        UniformTypeKVCacheSpecs,
    )

    def _iter_layer_specs(group):
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            for layer_name, layer_spec in spec.kv_cache_specs.items():
                yield layer_name, layer_spec
        else:
            for layer_name in group.layer_names:
                yield layer_name, spec

    bytes_per_block_all_layers = 0
    for group in kv_cache_groups:
        for _layer_name, layer_spec in _iter_layer_specs(group):
            bytes_per_block_all_layers += layer_spec.page_size_bytes

    assert bytes_per_block_all_layers > 0, "Zero per-block bytes"
    num_blocks = available_memory // bytes_per_block_all_layers
    num_blocks = max(num_blocks, 0)
    num_blocks = may_override_num_blocks(vllm_config, num_blocks)

    kv_cache_tensors = []
    for group in kv_cache_groups:
        for layer_name, layer_spec in _iter_layer_specs(group):
            kv_cache_tensors.append(
                KVCacheTensor(
                    size=layer_spec.page_size_bytes * num_blocks,
                    shared_by=[layer_name],
                )
            )

    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=kv_cache_tensors,
        kv_cache_groups=kv_cache_groups,
    )


def _heterogeneous_max_memory_usage_bytes(vllm_config, kv_cache_groups):
    # Max bytes needed for both groups to hold max_model_len tokens
    from vllm.utils.math_utils import cdiv
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs

    max_model_len = vllm_config.model_config.max_model_len
    total = 0
    for group in kv_cache_groups:
        spec = group.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            block_size = spec.block_size
            per_block_bytes = sum(
                s.page_size_bytes for s in spec.kv_cache_specs.values()
            )
        else:
            block_size = spec.block_size
            per_block_bytes = spec.page_size_bytes * len(group.layer_names)
        num_blocks_for_len = cdiv(max_model_len, block_size)
        total += num_blocks_for_len * per_block_bytes
    return total


def _patch_heterogeneous_eagle3_kv_cache() -> None:
    """Patch vLLM KV-cache grouping/allocation for heterogeneous KV cache so
    MLA target can coexist with an MHA EAGLE3 draft.
    Only MLA target and MHA draft combination is supported for now.
    """
    try:
        import vllm.v1.core.kv_cache_utils as vllm_kv_cache_utils
    except Exception:
        logger.warning(
            "ATOM plugin: failed to import vLLM kv_cache_utils; cannot enable "
            "MLA target with MHA EAGLE3 draft. This can happen with "
            "incompatible vLLM version."
        )
        return

    if getattr(vllm_kv_cache_utils, "_atom_heterogeneous_eagle3_patched", False):
        return

    orig_get_groups = vllm_kv_cache_utils.get_kv_cache_groups
    orig_config_from_groups = vllm_kv_cache_utils.get_kv_cache_config_from_groups
    orig_max_mem = vllm_kv_cache_utils._max_memory_usage_bytes_from_groups

    @functools.wraps(orig_get_groups)
    def patched_get_kv_cache_groups(vllm_config, kv_cache_spec):
        if getattr(
            vllm_config.model_config, "use_mla", False
        ) and _spec_has_heterogeneous_mla_mha_backend(kv_cache_spec):
            logger.info(
                "ATOM plugin: using heterogeneous KV cache layout - MLA target "
                "and MHA EAGLE3 draft - with separate per-group pools."
            )
            return _build_heterogeneous_kv_cache_groups(kv_cache_spec)
        return orig_get_groups(vllm_config, kv_cache_spec)

    @functools.wraps(orig_config_from_groups)
    def patched_get_kv_cache_config_from_groups(
        vllm_config, kv_cache_groups, available_memory
    ):
        if _groups_are_heterogeneous_mla_mha(kv_cache_groups):
            return _build_heterogeneous_kv_cache_config_from_groups(
                vllm_config, kv_cache_groups, available_memory
            )
        return orig_config_from_groups(vllm_config, kv_cache_groups, available_memory)

    @functools.wraps(orig_max_mem)
    def patched_max_memory_usage_bytes_from_groups(vllm_config, kv_cache_groups):
        if _groups_are_heterogeneous_mla_mha(kv_cache_groups):
            return _heterogeneous_max_memory_usage_bytes(vllm_config, kv_cache_groups)
        return orig_max_mem(vllm_config, kv_cache_groups)

    vllm_kv_cache_utils.get_kv_cache_groups = patched_get_kv_cache_groups
    vllm_kv_cache_utils.get_kv_cache_config_from_groups = patched_get_kv_cache_config_from_groups
    vllm_kv_cache_utils._max_memory_usage_bytes_from_groups = (
        patched_max_memory_usage_bytes_from_groups
    )
    vllm_kv_cache_utils._atom_heterogeneous_eagle3_patched = True
    logger.info(
        "ATOM plugin: patched vLLM KV-cache grouping/allocation for "
        "MLA target with MHA EAGLE3 speculative decoding."
    )


def apply_vllm_spec_decode_patch() -> None:
    """Patch vLLM speculative decoding for ATOM metadata compatibility."""
    from atom.plugin.vllm.attention.metadata import (
        AiterMhaMetadataForVllm,
        AiterMlaMetadataForVllm,
        AiterMlaSparseIndexerMetadataForVllm,
        AiterMlaSparseMetadataForVllm,
    )
    from atom.utils.forward_context import (
        AttentionMetaData as AtomAttentionMetaData,
    )
    from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer

    _patch_eagle3_model_type_checks()
    _patch_heterogeneous_eagle3_kv_cache()

    original_init = SpecDecodeBaseProposer.__init__
    if getattr(original_init, "_atom_allowed_attn_types_patched", False):
        return

    atom_allowed_attn_types = (
        AtomAttentionMetaData,
        AiterMhaMetadataForVllm,
        AiterMlaMetadataForVllm,
        AiterMlaSparseMetadataForVllm,
        AiterMlaSparseIndexerMetadataForVllm,
    )

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        allowed = getattr(self, "allowed_attn_types", None)
        if allowed is not None:
            self.allowed_attn_types = tuple(
                dict.fromkeys((*allowed, *atom_allowed_attn_types))
            )

    setattr(wrapped_init, "_atom_allowed_attn_types_patched", True)
    SpecDecodeBaseProposer.__init__ = wrapped_init

    logger.info(
        "ATOM plugin: patched vLLM speculative decoder for "
        "ATOM attention-metadata compatibility."
    )

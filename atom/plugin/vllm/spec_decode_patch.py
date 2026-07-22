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


def _share_atom_draft_with_target(draft_wrapper, target_model) -> None:
    draft_base = getattr(draft_wrapper, "model", draft_wrapper)
    share = getattr(draft_base, "share_with_target", None)
    if share is None:
        return
    target_base = getattr(target_model, "model", target_model)
    share(target_base, set())
    logger.info(
        "ATOM plugin: shared target weights with MTP draft via "
        "%s.share_with_target().",
        draft_base.__class__.__name__,
    )


def _patch_vllm_llm_base_model_sharing() -> None:
    """Run ATOM draft sharing after vLLM's generic MTP sharing path, and widen
    the proposer's ``allowed_attn_types`` with ``CommonAttentionMetadata`` for
    the ATOM DeepSeek-V4 MTP draft only.

    V4's proxy metadata builder returns the vLLM ``CommonAttentionMetadata``
    itself (with ATOM's V4 metadata attached), so the propose-loop
    ``isinstance(group_md, allowed_attn_types)`` gate must accept that base
    type. Done here -- after the draft model is loaded and its type is known --
    so the (over-broad) base type stays out of the whitelist for every other
    MTP/eagle model, whose draft emits a concrete backend metadata type and
    whose type check therefore stays strict.
    """
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    original_load = SpecDecodeBaseProposer.load_model
    if getattr(original_load, "_atom_share_with_target_patched", False):
        return

    @functools.wraps(original_load)
    def wrapped_load_model(self, target_model):
        original_load(self, target_model)
        _share_atom_draft_with_target(getattr(self, "model", None), target_model)
        if getattr(getattr(self, "model", None), "_is_deepseek_v4_mtp", False):
            from vllm.v1.attention.backend import CommonAttentionMetadata

            allowed = getattr(self, "allowed_attn_types", None)
            if allowed is not None and CommonAttentionMetadata not in allowed:
                self.allowed_attn_types = (*allowed, CommonAttentionMetadata)
                logger.info(
                    "ATOM plugin: allowed CommonAttentionMetadata for the "
                    "DeepSeek-V4 MTP draft attention type check."
                )

    setattr(wrapped_load_model, "_atom_share_with_target_patched", True)
    SpecDecodeBaseProposer.load_model = wrapped_load_model


def _patch_vllm_draft_kv_group_validation() -> None:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    original_validate = SpecDecodeBaseProposer.validate_same_kv_cache_group
    original_initialize = SpecDecodeBaseProposer.initialize_attn_backend
    if getattr(original_validate, "_atom_kv_group_validation_patched", False):
        return

    def _first_kv_block_size(kv_cache_config) -> int:
        group = kv_cache_config.kv_cache_groups[0]
        spec = group.kv_cache_spec
        block_size = getattr(spec, "block_size", None)
        if block_size is not None:
            return int(block_size)
        specs = getattr(spec, "kv_cache_specs", None)
        if specs:
            return int(next(iter(specs.values())).block_size)
        raise ValueError("Cannot determine KV cache block_size for ATOM draft")

    @functools.wraps(original_validate)
    def wrapped_validate_same_kv_cache_group(self, kv_cache_config):
        # ATOM DeepSeek-V4 MTP uses native ATOM attention behind the V4 proxy
        # bridge, so vLLM sees no draft AttentionLayerBase/KV layers. The
        # upstream assertion only handles one-or-more draft layers.
        if not getattr(self, "_draft_attn_layer_names", None):
            logger.info(
                "ATOM plugin: no vLLM draft attention layers detected; "
                "skipping draft KV group validation."
            )
            return
        try:
            return original_validate(self, kv_cache_config)
        except AssertionError:
            groups = []
            for idx, group in enumerate(kv_cache_config.kv_cache_groups):
                names = list(getattr(group, "layer_names", ()))
                draft_names = sorted(set(names) & self._draft_attn_layer_names)
                if draft_names:
                    groups.append((idx, draft_names))
            logger.error(
                "ATOM plugin: draft KV group validation failed; "
                "draft_attn_layer_names=%s grouped_as=%s",
                sorted(self._draft_attn_layer_names),
                groups,
            )
            raise

    @functools.wraps(original_initialize)
    def wrapped_initialize_attn_backend(
        self,
        kv_cache_config,
        kernel_block_sizes=None,
    ):
        if not getattr(self, "_draft_attn_layer_names", None):
            self.draft_attn_groups = []
            self.kv_cache_gid = 0
            self.block_size = _first_kv_block_size(kv_cache_config)
            logger.info(
                "ATOM plugin: no vLLM draft attention layers detected; "
                "using target KV block_size=%d for drafting slot mapping.",
                self.block_size,
            )
            return
        return original_initialize(self, kv_cache_config, kernel_block_sizes)

    setattr(
        wrapped_validate_same_kv_cache_group,
        "_atom_kv_group_validation_patched",
        True,
    )
    setattr(
        wrapped_initialize_attn_backend,
        "_atom_kv_group_validation_patched",
        True,
    )
    SpecDecodeBaseProposer.validate_same_kv_cache_group = (
        wrapped_validate_same_kv_cache_group
    )
    SpecDecodeBaseProposer.initialize_attn_backend = wrapped_initialize_attn_backend


def _patch_vllm_draft_positions_on_metadata() -> None:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    original_build = SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata
    if getattr(original_build, "_atom_positions_patched", False):
        return

    @functools.wraps(original_build)
    def wrapped_build_per_group_and_layer_attn_metadata(
        self,
        common_attn_metadata,
        draft_index: int = 0,
    ):
        # Only ATOM DeepSeek-V4 MTP needs the draft's ``common_attn_metadata``
        # augmented so the V4 sparse-attention bridge can pre-compute its topk
        # (C128A) metadata: the padded token count / slot mapping (draft_index>0)
        # and the per-token ``positions`` field. For every other MTP/eagle model
        # defer entirely to vLLM's original builder so their behavior is
        # byte-identical to upstream (the shared ``positions`` field in
        # particular must not be overwritten for them).
        if not getattr(getattr(self, "model", None), "_is_deepseek_v4_mtp", False):
            return original_build(self, common_attn_metadata, draft_index)
        num_tokens = int(
            getattr(common_attn_metadata, "num_actual_tokens", 0)
            or getattr(common_attn_metadata, "num_tokens", 0)
            or 0
        )
        if draft_index > 0 and hasattr(common_attn_metadata, "batch_size"):
            _mode, num_tokens, _num_tokens_across_dp = (
                self._determine_batch_execution_and_padding(
                    common_attn_metadata.batch_size()
                )
            )
            common_attn_metadata.num_actual_tokens = num_tokens
            common_attn_metadata.slot_mapping = self._get_slot_mapping(num_tokens)
        if num_tokens > 0 and hasattr(self, "_get_positions"):
            common_attn_metadata.positions = self._get_positions(num_tokens)
        return original_build(self, common_attn_metadata, draft_index)

    setattr(
        wrapped_build_per_group_and_layer_attn_metadata, "_atom_positions_patched", True
    )
    SpecDecodeBaseProposer.build_per_group_and_layer_attn_metadata = (
        wrapped_build_per_group_and_layer_attn_metadata
    )


def _patch_vllm_deepseek_v4_mtp_first_pass_inputs() -> None:
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    original_set_inputs = SpecDecodeBaseProposer.set_inputs_first_pass
    if getattr(original_set_inputs, "_atom_v4_mtp_inputs_patched", False):
        return

    @functools.wraps(original_set_inputs)
    def wrapped_set_inputs_first_pass(
        self,
        target_token_ids,
        next_token_ids,
        target_positions,
        target_hidden_states,
        token_indices_to_sample,
        cad,
        num_rejected_tokens_gpu,
    ):
        if (
            getattr(getattr(self, "model", None), "_is_deepseek_v4_mtp", False)
            and not self.needs_extra_input_slots
        ):
            if token_indices_to_sample is None:
                token_indices_to_sample = cad.query_start_loc[1:] - 1
            num_tokens = target_token_ids.shape[0]
            self.input_ids[:num_tokens] = target_token_ids
            self.input_ids[token_indices_to_sample] = next_token_ids
            if self.uses_xdrope_dim > 0 and self.draft_uses_xdrope_dim == 0:
                target_positions = target_positions[0]
            self._set_positions(num_tokens, target_positions)
            self.hidden_states[:num_tokens] = target_hidden_states
            return num_tokens, token_indices_to_sample, cad
        return original_set_inputs(
            self,
            target_token_ids,
            next_token_ids,
            target_positions,
            target_hidden_states,
            token_indices_to_sample,
            cad,
            num_rejected_tokens_gpu,
        )

    setattr(wrapped_set_inputs_first_pass, "_atom_v4_mtp_inputs_patched", True)
    SpecDecodeBaseProposer.set_inputs_first_pass = wrapped_set_inputs_first_pass


def apply_vllm_spec_decode_patch() -> None:
    """Patch vLLM speculative decoding for ATOM metadata compatibility."""
    _patch_vllm_llm_base_model_sharing()
    _patch_vllm_draft_kv_group_validation()
    _patch_vllm_draft_positions_on_metadata()
    _patch_vllm_deepseek_v4_mtp_first_pass_inputs()

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
        logger.info(
            "ATOM plugin: patched vLLM speculative decoder for "
            "ATOM MTP target sharing."
        )
        return

    # Concrete ATOM backend metadata types emitted by ATOM draft models. Adding
    # these to any ATOM proposer's whitelist is safe (they are ATOM's own types;
    # non-ATOM metadata never matches them). The over-broad base
    # ``CommonAttentionMetadata`` is deliberately NOT added here -- it is added
    # only for the V4 MTP draft in ``_patch_vllm_llm_base_model_sharing``.
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

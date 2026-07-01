import functools
import logging

logger = logging.getLogger("atom")


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

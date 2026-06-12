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

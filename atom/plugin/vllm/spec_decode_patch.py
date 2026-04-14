import functools
import logging

logger = logging.getLogger("atom")


def apply_vllm_spec_decode_patch() -> None:
    """Allow ATOM attention metadata in vLLM speculative decoding checks."""
    try:
        from atom.utils.forward_context import AttentionMetaData as AtomAttentionMetaData
        from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
    except Exception as exc:
        logger.debug("ATOM spec-decode patch skipped: %s", exc)
        return

    original_init = SpecDecodeBaseProposer.__init__
    if getattr(original_init, "_atom_allowed_attn_types_patched", False):
        return

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        allowed = getattr(self, "allowed_attn_types", None)
        if isinstance(allowed, tuple) and AtomAttentionMetaData not in allowed:
            self.allowed_attn_types = (*allowed, AtomAttentionMetaData)

    setattr(wrapped_init, "_atom_allowed_attn_types_patched", True)
    SpecDecodeBaseProposer.__init__ = wrapped_init
    logger.info(
        "ATOM plugin: patched vLLM speculative decoder to accept "
        "atom.utils.forward_context.AttentionMetaData."
    )

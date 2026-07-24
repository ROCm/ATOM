"""Forward-mode router: target_verify / draft_extend / draft_decode / prefill."""

from __future__ import annotations

from atom.plugin.sglang.glm52_mtp.draft_decode import build_decode_metadata
from atom.plugin.sglang.glm52_mtp.draft_extend import (
    build_mtp_draft_extend_decode_metadata,
    should_use_mtp_draft_extend_decode_path,
)
from atom.plugin.sglang.glm52_mtp.prefill import (
    build_mtp_draft_extend_prefill_metadata,
    build_prefill_metadata,
    is_draft_extend_prefill,
)
from atom.plugin.sglang.glm52_mtp.target_verify import (
    build_mtp_verify_decode_metadata,
    should_use_mtp_verify_prefill_path,
)


def build_atom_glm52_attention_metadata_from_sglang(
    forward_batch,
    positions,
    *,
    token_to_kv_pool,
    req_to_token_pool,
    atom_config,
):
    if getattr(forward_batch.forward_mode, "is_target_verify", lambda: False)():
        if should_use_mtp_verify_prefill_path(forward_batch, positions, atom_config):
            raise RuntimeError(
                "GLM-5.2 DSA target_verify prefill metadata is not ported yet; "
                "use decode path (default) or unset ATOM_GLM52_TV_VERIFY_PATH"
            )
        return build_mtp_verify_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    if forward_batch.forward_mode.is_decode_or_idle():
        return build_decode_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    if is_draft_extend_prefill(forward_batch):
        return build_mtp_draft_extend_prefill_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    if getattr(forward_batch.forward_mode, "is_draft_extend", lambda **kwargs: False)(
        include_v2=True
    ):
        if should_use_mtp_draft_extend_decode_path(forward_batch):
            return build_mtp_draft_extend_decode_metadata(
                forward_batch,
                positions,
                token_to_kv_pool=token_to_kv_pool,
                req_to_token_pool=req_to_token_pool,
                atom_config=atom_config,
            )
        return build_prefill_metadata(
            forward_batch,
            positions,
            token_to_kv_pool=token_to_kv_pool,
            req_to_token_pool=req_to_token_pool,
            atom_config=atom_config,
        )
    return build_prefill_metadata(
        forward_batch,
        positions,
        token_to_kv_pool=token_to_kv_pool,
        req_to_token_pool=req_to_token_pool,
        atom_config=atom_config,
    )

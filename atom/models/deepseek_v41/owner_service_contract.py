# SPDX-License-Identifier: MIT
"""Explicit admission for the combined request-owner and retained prefill path."""


def owner_prefill_enabled(
    config, *, owner_service, retained_prefill, ep4_verified=False
):
    """Admit the TP4 DSpark5 owner deployment, optionally with verified EP4.

    Prefill uses the existing retained-tail arithmetic and native aux capture.
    The owner target replaces decode before this path's large-row eligibility.
    The runtime must validate actual expert modules before setting ep4_verified.
    """
    if not owner_service or not retained_prefill:
        return False
    spec = config.speculative_config
    return (
        spec is not None
        and spec.method == "dspark"
        and spec.num_speculative_tokens == 5
        and config.tensor_parallel_size == 4
        and config.parallel_config.data_parallel_size == 1
        and not config.enable_dp_attention
        and not config.enable_tbo
        and config.prefill_context_parallel_size == 1
        and config.decode_context_parallel_size == 1
        and (not config.enable_expert_parallel or ep4_verified is True)
        and config.index_cache_dtype == "fp4"
        and config.kv_cache_dtype == "bf16"
        and config.enable_decoder_swa_bounded_replay
    )

# SPDX-License-Identifier: MIT
"""Token identity shared by forced-acceptance target and draft forwards."""

from atom.utils import envs


def resolve_synthetic_token_id(config) -> int | None:
    """Pick a shared fake ID only when synthetic forward is explicitly enabled.

    Avoid configured special/stop tokens so a constant stream does not end
    immediately. None preserves rejection-only behavior. Resolution is host-only
    and deterministic across ranks; the environment is read at initialization.
    """
    spec = config.speculative_config
    if spec is None or spec.synthetic_acceptance_rates is None:
        return None
    if not envs.ATOM_SPEC_DECODE_SYNTHETIC_FORWARD:
        return None

    excluded = set()
    configs = (
        config,
        config.hf_config,
        getattr(config, "generation_config", None),
        spec.draft_model_hf_config,
    )
    for source in configs:
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "stop_token_ids"):
            ids = getattr(source, name, None)
            if ids is not None:
                excluded.update(ids if isinstance(ids, (list, tuple, set)) else [ids])

    vocab_size = min(
        config.hf_config.vocab_size,
        getattr(spec.draft_model_hf_config, "vocab_size", config.hf_config.vocab_size),
    )
    for token_id in range(vocab_size):
        if token_id not in excluded:
            return token_id
    raise ValueError("Forced speculative acceptance needs a non-special token ID.")

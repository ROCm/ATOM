# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 bounded replay: how far below a hit the window state rebuilds.

DeepSeek-V4's per-request state is each layer's sliding window plus the
compressors' partial groups. Neither is in the cached blocks, but both are
local: layer l's window at H depends only on layer l-1 over the last `W`
positions, so with every layer's compressed KV and indexer K cached
(read-only), recomputing `W + (L-1)(W-1)` tokens below H rebuilds all L windows
exactly. The compressor groups need at most `2*ratio - 1` trailing positions of
a layer's input, which that span already covers (and H is a multiple of 128, so
no C128 group is open there). An MTP layer reads the trunk's output and carries
a window of its own, so it counts as one more layer. So does a DSpark draft:
each of its layers writes its window from the same fused target hidden states
(`project_context` of `dspark_target_layer_ids`, position by position), none
from another draft layer, and it reads that window only at a verified anchor,
which the replay always lies W or more below.

The length is `Config.state_replay_tokens`, set from here the way V4.1's
`min_final_prefill_rows` is; `BlockManager` decides per admission whether
replaying beats resuming from the nearest checkpoint, and the forward runs as
V4.1's do (`Replay`). Kept apart from the builder so the CPU tests and `Config`
can reach it without the AITER build the builder's module imports.
"""

import logging

from atom.utils import envs

logger = logging.getLogger("atom")


def _connector_replays(kv_transfer_config) -> bool:
    """Whether a bounded replay can run beside this KV connector.

    A connector load starts at `num_cached_tokens` and raises it when the load
    lands, and a replay moves exactly that field below blocks it has claimed:
    a load that knows nothing of the replay would write those shared blocks
    and resume past it. LMCache MP offload's native-state scheduler is the one
    that knows (`Sequence.replay_kv_load`); under `multi` (P/D plus offload)
    the P/D connector's loads do not.
    """
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    name = KVConnectorFactory.connector_name(kv_transfer_config)
    if name is None:
        return True
    # Named as the factory resolves it, the role defaulting as the connector's
    # does (`validated_kv_role`).
    return (
        name == "lmcache_mp"
        and kv_transfer_config.get("kv_role", "offload") == "offload"
    )


def state_replay_tokens(config) -> int:
    """Tokens a bounded replay recomputes below a hit, or 0 when it is off.

    Off unless ATOM_DSV4_STATE_REPLAY is set, and only where the receptive
    field argument above holds and the V4 prefill can express it: the plain V4
    trunk, compress ratios 4 / 128, no speculative method but MTP or the
    model's own DSpark layers, and a single-rank prefill (context parallel splits the query axis the
    floors are indexed by). It replays a prefix-cache hit, so it also needs the
    prefix cache, and no KV connector but LMCache MP offload (`_connector_replays`).
    """
    if not envs.ATOM_DSV4_STATE_REPLAY:
        return 0
    reason = None
    hf = config.hf_config
    arches = getattr(hf, "architectures", None) or []
    spec = config.speculative_config
    method = (spec.method or "") if spec is not None else ""
    if not any(str(a).startswith("DeepseekV4For") for a in arches):
        reason = f"architecture {arches} is not the DeepSeek-V4 trunk"
    elif not set(getattr(hf, "compress_ratios", ()) or ()) <= {0, 4, 128}:
        reason = "compress ratios other than 4 / 128"
    elif method not in ("", "mtp", "dspark"):
        reason = f"speculative method {method!r}"
    elif getattr(spec, "use_dspark_with_draft", lambda: False)():
        # A separate draft model keeps its own paged KV under the target's
        # block ids, which a replay would rewrite inside the hit; and the one
        # hop argument above is about DSpark layers reading the trunk.
        reason = "a separate DSpark draft model (--draft-model)"
    elif (
        config.prefill_context_parallel_size > 1
        or config.decode_context_parallel_size > 1
    ):
        reason = "context parallelism"
    elif not config.enable_prefix_caching:
        reason = "prefix caching off"
    elif not _connector_replays(config.kv_transfer_config):
        reason = "a KV connector other than LMCache MP offload alone"
    if reason is not None:
        logger.warning("[State Cache] bounded replay off: %s.", reason)
        return 0
    window = int(hf.sliding_window)
    layers = int(hf.num_hidden_layers)
    if method == "mtp":
        layers += int(getattr(hf, "num_nextn_predict_layers", 1) or 1)
    elif method == "dspark":
        layers += 1  # one hop: every draft layer reads the trunk directly
    tokens = window + (layers - 1) * (window - 1)
    logger.info(
        "[State Cache] bounded replay on: a hit past the nearest checkpoint by "
        "more than %d tokens replays that many (window %d, %d layers).",
        tokens,
        window,
        layers,
    )
    return tokens

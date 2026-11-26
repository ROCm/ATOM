# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""The `EngineSpec` atomesh serves the EngineCores directly from
(``--engine-dispatch direct``).

Its shape is `atom/mesh/src/atom_engine/spec.rs`. Python spawns the Cores and
hands this over once; Rust (`routers/atom_bridge/direct`) is their only client
from then on, and renders, tokenizes and parses as it does over the token
bridge. What only Python can decide is decided here, as `TokenBridge` decides
it: the deployment's output cap, the required tool-call channel and the ids
that end an output. What direct dispatch cannot serve is refused before the
weights load: `check_direct_args` on the args alone, `check_direct_tokenizer`
once the tokenizer is in. CoreManager refuses the deployments it cannot hand
an external client (pipeline parallelism, multi-node, RapidServe).
"""

from __future__ import annotations

import json
from typing import Any

from atom.entrypoints.openai.model_capabilities import deployment_metadata
from atom.entrypoints.openai.protocol import DEFAULT_MAX_TOKENS
from atom.entrypoints.openai.reasoning import VALID_TEMPLATE_EFFORTS
from atom.entrypoints.openai.tool_parser.registry import required_call_channel
from atom.kv_transfer.disaggregation.factory import parse_kv_transfer_config
from atom.kv_transfer.disaggregation.pd_producer import pd_transfer_entries
from atom.utils import envs

# The DP placements Rust has (`policies/dp_load.rs`, round robin); the prefix
# routers (`share_cache`, `prefix_spread`) are CoreManager's alone.
DIRECT_DP_STRATEGIES = ("round_robin", "least_requests", "least_tokens")

_SERVE_OVER_THE_BRIDGE = "; serve it with --engine-dispatch bridge."


def check_direct_args(engine_args: Any) -> None:
    """Refuse what the args alone show direct dispatch cannot serve: a P/D
    (prefill/decode) worker, and a DP placement Rust does not have."""
    pd = pd_transfer_entries(parse_kv_transfer_config(engine_args.kv_transfer_config))
    if pd:
        connector = pd[0].get("kv_connector", "moriio")
        role = pd[0].get("kv_role", "kv_producer")
        raise ValueError(
            f"kv_connector {connector!r} (kv_role {role!r}) makes a P/D worker, "
            "which direct dispatch does not serve yet" + _SERVE_OVER_THE_BRIDGE
        )
    strategy = engine_args.dp_load_balance
    if strategy not in DIRECT_DP_STRATEGIES:
        raise ValueError(
            f"--dp-load-balance {strategy!r} is not one direct dispatch places by "
            f"(it has {', '.join(DIRECT_DP_STRATEGIES)})" + _SERVE_OVER_THE_BRIDGE
        )


def check_direct_tokenizer(renderer: Any, model: str) -> None:
    """Refuse a model whose chat template Rust cannot render (`renderer` is
    the token bridge's `ChatRenderer`): direct dispatch has no Python to
    render it in."""
    if renderer.needed:
        raise ValueError(
            f"{model}'s chat prompt is rendered in Python (a message encoder, or "
            "a template that is code), which direct dispatch cannot do"
            + _SERVE_OVER_THE_BRIDGE
        )


def required_channel(renderer: Any, tokenizer: Any) -> dict | None:
    """The channel a `tool_choice="required"` request is held to, as
    `TokenBridge` names it; the Core holds it to its end ids. None where the
    tool-call format has none."""
    channel = required_call_channel(renderer.tool_parser_cls, tokenizer)
    if channel is None:
        return None
    return {
        "after": list(channel.after),
        "opening": list(channel.opening),
        "opens_at": channel.opens_at,
        "end_ids": list(channel.end_ids),
    }


def build_engine_spec(
    engine: Any,
    tokenizer: Any,
    renderer: Any,
    served_name: str,
    default_chat_template_kwargs: dict[str, Any],
) -> str:
    """The spec of `engine`'s Cores, spawned for an external client."""
    config = engine.config
    metadata = deployment_metadata(config, renderer.encoder)
    return json.dumps(
        {
            "model": {
                "served_name": served_name,
                "tokenizer_path": config.model,
                "default_chat_template_kwargs": default_chat_template_kwargs,
                "model_card": metadata,
                "max_model_len": config.max_model_len,
                "default_max_tokens": DEFAULT_MAX_TOKENS,
                "max_output_tokens": metadata.get("max_output_tokens"),
                "required_channel": required_channel(renderer, tokenizer),
                "template_efforts": sorted(VALID_TEMPLATE_EFFORTS),
                "end_token_ids": list(config.end_token_ids or ()),
            },
            "balance": {
                "strategy": config.dp_load_balance,
                "req_equiv": envs.ATOM_DP_LB_REQ_EQUIV,
                "session_affinity": envs.ATOM_DP_SESSION_AFFINITY,
            },
            "engines": engine.core_mgr.engine_endpoints,
        }
    )

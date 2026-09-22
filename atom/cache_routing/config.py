# SPDX-License-Identifier: MIT
"""Opt-in configuration, shared by the frontend, scheduler and PP/TP workers."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from atom.cache_routing.keys import namespace_digest


@dataclass(frozen=True)
class CacheRoutingConfig:
    execution_id: str
    catalog_url: str
    content_namespace: str
    canonical_block_size: int = 16
    storage_domain_id: str = ""
    max_entries: int = 200000
    max_log_bytes: int = 16777216
    stale_seconds: float = 3.0
    cpu_poll_interval_seconds: float = 0.5

    @classmethod
    def from_env(cls) -> CacheRoutingConfig | None:
        """Read ATOM_CACHE_ROUTING_CONFIG (JSON); absence disables observation."""
        raw = os.environ.get("ATOM_CACHE_ROUTING_CONFIG")
        if not raw:
            return None
        data = json.loads(raw)
        data["content_namespace"] = namespace_digest(data.pop("namespace_manifest"))
        cfg = cls(**data)
        endpoint = urlparse(cfg.catalog_url)
        if (
            not cfg.execution_id
            or endpoint.scheme != "http"
            or not endpoint.hostname
            or not endpoint.port
        ):
            raise ValueError(
                "cache routing requires execution_id and an HTTP catalog host:port"
            )
        if endpoint.path not in ("", "/") or endpoint.query or endpoint.username:
            raise ValueError("catalog_url must be an HTTP origin")
        if (
            min(
                cfg.canonical_block_size,
                cfg.max_entries,
                cfg.max_log_bytes,
                cfg.stale_seconds,
                cfg.cpu_poll_interval_seconds,
            )
            <= 0
        ):
            raise ValueError("cache routing geometry and budgets must be positive")
        if (
            not math.isfinite(cfg.cpu_poll_interval_seconds)
            or not math.isfinite(cfg.stale_seconds)
            or cfg.cpu_poll_interval_seconds >= cfg.stale_seconds
        ):
            raise ValueError("CPU poll interval must be finite and below stale_seconds")
        return cfg

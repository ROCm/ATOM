# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

DEFAULT_E2E_FUSION_PROFILE = "default"


@dataclass(frozen=True)
class E2EFusionProfile:
    name: str
    batch: int
    seq: int
    hidden: int
    ffn: int
    num_chunks: int

    @property
    def tokens(self) -> int:
        return self.batch * self.seq


_BASE_PROFILES = {
    DEFAULT_E2E_FUSION_PROFILE: E2EFusionProfile(
        name=DEFAULT_E2E_FUSION_PROFILE,
        batch=2,
        seq=2048,
        hidden=4096,
        ffn=14336,
        num_chunks=4,
    ),
    # Double token count to make communication more visible while keeping
    # the GEMM width unchanged.
    "backend_gap": E2EFusionProfile(
        name="backend_gap",
        batch=4,
        seq=2048,
        hidden=4096,
        ffn=14336,
        num_chunks=4,
    ),
    # Increase both token count and FFN width so the pipeline has more
    # compute to overlap and a better chance to amortize chunk overhead.
    "pipeline_gain": E2EFusionProfile(
        name="pipeline_gain",
        batch=4,
        seq=2048,
        hidden=4096,
        ffn=28672,
        num_chunks=4,
    ),
}

_SHAPE_SWEEP_PROFILES = (
    E2EFusionProfile(
        "comm_bound_small", batch=2, seq=2048, hidden=4096, ffn=8192, num_chunks=4
    ),
    _BASE_PROFILES[DEFAULT_E2E_FUSION_PROFILE],
    E2EFusionProfile(
        "compute_bound_small", batch=2, seq=2048, hidden=4096, ffn=28672, num_chunks=4
    ),
    E2EFusionProfile(
        "comm_bound_large", batch=4, seq=2048, hidden=4096, ffn=8192, num_chunks=4
    ),
    _BASE_PROFILES["backend_gap"],
    _BASE_PROFILES["pipeline_gain"],
)


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    value = env.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _require_positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def get_e2e_fusion_profile(
    name: str | None = None,
    env: Mapping[str, str] | None = None,
) -> E2EFusionProfile:
    source_env = os.environ if env is None else env
    profile_name = name or source_env.get(
        "LUMEN_E2E_PROFILE", DEFAULT_E2E_FUSION_PROFILE
    )

    if profile_name not in _BASE_PROFILES:
        valid = ", ".join(sorted(_BASE_PROFILES))
        raise ValueError(
            f"Unknown E2E fusion profile '{profile_name}'. Valid profiles: {valid}"
        )

    base = _BASE_PROFILES[profile_name]
    return E2EFusionProfile(
        name=base.name,
        batch=_require_positive(
            "LUMEN_E2E_BATCH", _env_int(source_env, "LUMEN_E2E_BATCH", base.batch)
        ),
        seq=_require_positive(
            "LUMEN_E2E_SEQ", _env_int(source_env, "LUMEN_E2E_SEQ", base.seq)
        ),
        hidden=_require_positive(
            "LUMEN_E2E_HIDDEN", _env_int(source_env, "LUMEN_E2E_HIDDEN", base.hidden)
        ),
        ffn=_require_positive(
            "LUMEN_E2E_FFN", _env_int(source_env, "LUMEN_E2E_FFN", base.ffn)
        ),
        num_chunks=_require_positive(
            "LUMEN_E2E_NUM_CHUNKS",
            _env_int(source_env, "LUMEN_E2E_NUM_CHUNKS", base.num_chunks),
        ),
    )


def get_e2e_fusion_shape_sweep(
    env: Mapping[str, str] | None = None,
) -> list[E2EFusionProfile]:
    source_env = os.environ if env is None else env
    return [
        E2EFusionProfile(
            name=profile.name,
            batch=profile.batch,
            seq=profile.seq,
            hidden=profile.hidden,
            ffn=profile.ffn,
            num_chunks=_require_positive(
                "LUMEN_E2E_NUM_CHUNKS",
                _env_int(source_env, "LUMEN_E2E_NUM_CHUNKS", profile.num_chunks),
            ),
        )
        for profile in _SHAPE_SWEEP_PROFILES
    ]


def format_e2e_shape_tag(profile: E2EFusionProfile) -> str:
    return f"T{profile.tokens}_H{profile.hidden}_F{profile.ffn}_C{profile.num_chunks}"


def wgrad_delay_dims(profile: E2EFusionProfile) -> tuple[int, int, int]:
    return profile.tokens, profile.hidden, profile.ffn


def fp8_param_e2e_weight_shape(profile: E2EFusionProfile) -> tuple[int, int]:
    return profile.ffn, profile.hidden


def fp8_param_e2e_input_shape(
    profile: E2EFusionProfile,
    in_features: int | None = None,
) -> tuple[int, int, int]:
    features = profile.hidden if in_features is None else in_features
    return profile.batch, profile.seq, features


def fp8_param_pipeline_layer_shapes(profile: E2EFusionProfile) -> list[tuple[int, int]]:
    return [
        (profile.ffn, profile.hidden),
        (profile.ffn, profile.hidden),
        (profile.hidden, profile.ffn),
        (profile.hidden, profile.hidden),
    ]


def format_e2e_fusion_profile(profile: E2EFusionProfile) -> str:
    return (
        f"profile={profile.name}, B={profile.batch}, S={profile.seq}, "
        f"H={profile.hidden}, FFN={profile.ffn}, chunks={profile.num_chunks}"
    )

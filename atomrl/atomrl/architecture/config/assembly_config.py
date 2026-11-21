"""Runtime assembly configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeAssemblyConfig:
    """Top-level runtime composition options."""

    training_backend: str = "fsdp2"
    inference_backend: str = "atom"
    topology: str = "decoupled"
    strict_policy: bool = True
    use_new_assembler: bool = True
    role_impl_keys: dict[str, str] = field(default_factory=dict)

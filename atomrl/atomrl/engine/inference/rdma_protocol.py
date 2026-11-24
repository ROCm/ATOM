# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Versioned capability contract for RDMA weight reloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass

RDMA_PROTOCOL_VERSION = 2


@dataclass(frozen=True, slots=True)
class RDMACapability:
    """Serializable capabilities exposed by each vLLM worker."""

    protocol_version: int
    module_path: str
    online_quant_reload: bool
    prequantized_stream: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

from __future__ import annotations

from typing import Any

import torch


def dump_tensor(
    *,
    tag: str,
    tensor: torch.Tensor | None,
    layer: int | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    del tag, tensor, layer, meta
    return

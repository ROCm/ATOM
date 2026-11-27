# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""BF16 tail for the open NVFP4 MLA tile.

One ring per request per layer holds the original BF16 latent of the tile
that is still accepting tokens. The ring is 16 tokens, one tile. It is not a
page and does not grow with context length.

A tile with fewer than 16 tokens is still quantized into the FP4 cache, using
only the valid rows. The ring keeps the BF16 originals so the next token
recomputes that tile's scale from those values. Dequantizing the FP4 already
written would round the same numbers a second time.
"""

from __future__ import annotations

import torch

# One NoPE block scale covers this many tokens. Matches the AITER producer
# and TRT-LLM's HP_BLOCK_SIZE.
HP_BLOCK_SIZE = 16


class Nvfp4HpRing:
    """``[num_reqs, layers, ring_size, head_dim]`` BF16, allocated once."""

    def __init__(
        self,
        num_reqs: int,
        layers: int,
        head_dim: int,
        *,
        device: torch.device | str = "cpu",
    ):
        if num_reqs < 1 or layers < 1 or head_dim < 1:
            raise ValueError(
                "NVFP4 HP ring needs a positive request, layer, and head count, "
                f"got reqs={num_reqs} layers={layers} head_dim={head_dim}."
            )
        self.num_reqs = num_reqs
        self.layers = layers
        self.head_dim = head_dim
        self.ring_size = HP_BLOCK_SIZE
        self.buf = torch.zeros(
            (num_reqs, layers, self.ring_size, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )

    def nbytes(self) -> int:
        """Bytes of the whole ring. Page size does not enter this number."""
        return self.buf.numel() * self.buf.element_size()

    def append_tokens(
        self,
        layer: int,
        req_ids: torch.Tensor,
        positions: torch.Tensor,
        values: torch.Tensor,
        page_size: int,
        *,
        check: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write one new BF16 token per row and return that token's open tile.

        Returns ``tiles [N, 16, head_dim]``, ``valid [N]``, ``page_ids [N]``,
        ``tile_ids [N]``. Rows at and after ``valid`` are zero, so a partial
        tile can be quantized immediately. ``page_size`` only selects the FP4
        page and the tile inside it; it does not change the ring.

        ``check=False`` skips the value checks, which read the tensors back to
        the host and so cannot run inside a CUDA graph capture.
        """
        self._check_page_size(page_size)
        self._check_layer(layer)
        req_ids = req_ids.to(device=self.buf.device, dtype=torch.long)
        positions = positions.to(device=self.buf.device, dtype=torch.long)
        if req_ids.ndim != 1 or positions.shape != req_ids.shape:
            raise ValueError("req_ids and positions must be 1-D and the same length.")
        if values.shape != (req_ids.shape[0], self.head_dim):
            raise ValueError(
                f"values must be [{req_ids.shape[0]}, {self.head_dim}], "
                f"got {tuple(values.shape)}."
            )
        if (
            check
            and req_ids.numel()
            and (int(req_ids.min()) < 0 or int(req_ids.max()) >= self.num_reqs)
        ):
            raise ValueError(
                f"req id outside 0..{self.num_reqs - 1}: {req_ids.tolist()}."
            )
        if check and positions.numel() and int(positions.min()) < 0:
            raise ValueError("NVFP4 token position must be >= 0.")

        slots = torch.remainder(positions, self.ring_size)
        self.buf[req_ids, layer, slots] = values.to(dtype=self.buf.dtype)

        tile_start = (positions // HP_BLOCK_SIZE) * HP_BLOCK_SIZE
        valid = (positions - tile_start + 1).to(torch.int32)
        offsets = torch.arange(HP_BLOCK_SIZE, device=self.buf.device)
        gathered = tile_start[:, None] + offsets[None, :]
        tiles = self.buf[
            req_ids[:, None], layer, torch.remainder(gathered, self.ring_size)
        ]
        # masked_fill, not boolean-index assignment: the latter calls nonzero
        # and syncs with the host.
        tiles = tiles.masked_fill((offsets[None, :] >= valid[:, None])[:, :, None], 0)

        page_ids = torch.div(positions, page_size, rounding_mode="floor").to(
            torch.int32
        )
        tile_ids = torch.div(
            torch.remainder(positions, page_size),
            HP_BLOCK_SIZE,
            rounding_mode="floor",
        ).to(torch.int32)
        return tiles, valid, page_ids, tile_ids

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.layers:
            raise ValueError(f"layer {layer} outside 0..{self.layers - 1}.")

    @staticmethod
    def _check_page_size(page_size: int) -> None:
        if page_size < HP_BLOCK_SIZE or page_size % HP_BLOCK_SIZE != 0:
            raise ValueError(
                f"NVFP4 page_size must be a positive multiple of {HP_BLOCK_SIZE}, "
                f"got {page_size}."
            )

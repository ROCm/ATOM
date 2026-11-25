# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""DataProto: tensor-dict batch protocol for cross-worker communication."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)


class DataProto:
    """A tensor-dict batch that flows between controller and workers.

    All cross-worker data exchange uses this format. Tensors are stored on
    CPU for serialization; workers move them to GPU as needed.

    Supports variable-length (remove-padding) sequences via optional
    ``cu_seqlens`` and ``seq_lens`` metadata tensors.

    ``ragged`` carries per-row payloads that cannot be a rectangular tensor, as a
    list with one entry per row. It is reindexed by every row operation exactly
    like ``tensors``, which ``meta`` is not: ``meta`` is shallow-copied unchanged
    into each chunk, so a per-row payload placed there would both survive
    ``select_idxs`` with stale ordering and be replicated to every worker.
    MoE rollout routing is the motivating case -- ``[seq_len, layers, top_k]``
    per row, which is ~350MB packed for a long-run batch but 12.7GB if padded to
    a common length.
    """

    def __init__(
        self,
        tensors: dict[str, torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        ragged: dict[str, list[Any]] | None = None,
    ) -> None:
        self.tensors: dict[str, torch.Tensor] = tensors or {}
        self.meta: dict[str, Any] = meta or {}
        self.ragged: dict[str, list[Any]] = ragged or {}

    def _ragged_at(self, index) -> dict[str, list[Any]]:
        """Reindex every ragged column by a sequence of row positions."""
        if not self.ragged:
            return {}
        rows = list(index)
        return {k: [v[i] for i in rows] for k, v in self.ragged.items()}

    @staticmethod
    def _ragged_concat(protos: list[DataProto]) -> dict[str, list[Any]]:
        keys = [p for p in protos if p.ragged]
        if not keys:
            return {}
        out: dict[str, list[Any]] = {}
        for k in keys[0].ragged:
            merged: list[Any] = []
            for p in protos:
                merged.extend(p.ragged.get(k, []))
            out[k] = merged
        return out

    # ------------------------------------------------------------------
    # Variable-length / remove-padding helpers
    # ------------------------------------------------------------------

    @property
    def cu_seqlens(self) -> torch.Tensor | None:
        """Cumulative sequence lengths ``(B+1,)`` for packed/remove-padding input."""
        return self.tensors.get("cu_seqlens")

    @cu_seqlens.setter
    def cu_seqlens(self, value: torch.Tensor) -> None:
        self.tensors["cu_seqlens"] = value

    @property
    def seq_lens(self) -> torch.Tensor | None:
        """Per-sequence lengths ``(B,)``."""
        return self.tensors.get("seq_lens")

    @seq_lens.setter
    def seq_lens(self, value: torch.Tensor) -> None:
        self.tensors["seq_lens"] = value

    @property
    def max_seqlen(self) -> int:
        """Maximum sequence length in the batch (0 if not set)."""
        if self.seq_lens is not None:
            return int(self.seq_lens.max().item())
        return self.meta.get("max_seqlen", 0)

    @property
    def is_packed(self) -> bool:
        """Whether this batch uses packed/remove-padding format."""
        return self.cu_seqlens is not None

    def __getitem__(
        self, item: str | int | slice | list[int] | torch.Tensor
    ) -> torch.Tensor | DataProto:
        if isinstance(item, str):
            return self.tensors[item]
        return self.select_idxs(item)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.tensors[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def __len__(self) -> int:
        if not self.tensors:
            if self.ragged:
                return len(next(iter(self.ragged.values())))
            return 0
        first_key = next(iter(self.tensors))
        return self.tensors[first_key].shape[0]

    @property
    def batch_size(self) -> int:
        return len(self)

    def keys(self) -> list[str]:
        return list(self.tensors.keys())

    def check_consistency(self) -> None:
        """Validate that all tensors and ragged columns share the same row count."""
        if not self.tensors:
            return
        first_key = next(iter(self.tensors))
        first_size = self.tensors[first_key].shape[0]
        for key, tensor in self.tensors.items():
            if tensor.shape[0] != first_size:
                raise ValueError(
                    "DataProto has inconsistent batch dimensions: "
                    f"{first_key} has {first_size}, {key} has {tensor.shape[0]}"
                )
        for key, rows in self.ragged.items():
            if len(rows) != first_size:
                raise ValueError(
                    "DataProto has inconsistent batch dimensions: "
                    f"{first_key} has {first_size}, ragged {key} has {len(rows)}"
                )

    def to(self, device: str | torch.device) -> DataProto:
        """Move all tensors to the given device (ragged payloads stay put)."""
        return DataProto(
            tensors={k: v.to(device) for k, v in self.tensors.items()},
            meta=self.meta.copy(),
            ragged={k: list(v) for k, v in self.ragged.items()},
        )

    def cpu(self) -> DataProto:
        return self.to("cpu")

    def cuda(self, device: int = 0) -> DataProto:
        return self.to(f"cuda:{device}")

    def _normalize_indices(
        self, idxs: int | slice | list[int] | torch.Tensor
    ) -> torch.Tensor:
        """Normalize user indices to a 1D tensor of bool or long."""
        if isinstance(idxs, int):
            if idxs < 0:
                idxs += self.batch_size
            idx_tensor = torch.tensor([idxs], dtype=torch.long)
        elif isinstance(idxs, slice):
            idx_tensor = torch.arange(self.batch_size)[idxs]
        elif isinstance(idxs, list):
            idx_tensor = torch.as_tensor(idxs)
        elif isinstance(idxs, torch.Tensor):
            idx_tensor = idxs
        else:
            raise TypeError(f"Unsupported index type for DataProto: {type(idxs)}")

        if idx_tensor.dtype == torch.bool:
            if idx_tensor.ndim != 1:
                raise ValueError(
                    "Boolean index tensor for DataProto must be 1-dimensional"
                )
            if idx_tensor.numel() != self.batch_size:
                raise ValueError(
                    "Boolean index tensor length mismatch: "
                    f"expected {self.batch_size}, got {idx_tensor.numel()}"
                )
            return idx_tensor

        idx_tensor = idx_tensor.to(dtype=torch.long).reshape(-1)
        idx_tensor = torch.where(
            idx_tensor < 0, idx_tensor + self.batch_size, idx_tensor
        )
        return idx_tensor

    def select_idxs(self, idxs: int | slice | list[int] | torch.Tensor) -> DataProto:
        """Select rows by integer/slice/list/tensor indices."""
        if self.batch_size == 0:
            return DataProto(meta=self.meta.copy())
        norm = self._normalize_indices(idxs)
        out: dict[str, torch.Tensor] = {}
        for key, tensor in self.tensors.items():
            index = norm.to(device=tensor.device)
            out[key] = tensor[index]
        rows = norm.nonzero(as_tuple=True)[0] if norm.dtype == torch.bool else norm
        return DataProto(
            tensors=out,
            meta=self.meta.copy(),
            ragged=self._ragged_at(int(i) for i in rows.tolist()),
        )

    def split(self, num_chunks: int) -> list[DataProto]:
        """Split batch into num_chunks equal parts along dim 0."""
        if self.batch_size == 0:
            return [DataProto(meta=self.meta.copy()) for _ in range(num_chunks)]

        result = []
        for i in range(num_chunks):
            start = i * self.batch_size // num_chunks
            end = (i + 1) * self.batch_size // num_chunks
            chunk_tensors = {k: v[start:end] for k, v in self.tensors.items()}
            result.append(
                DataProto(
                    tensors=chunk_tensors,
                    meta=self.meta.copy(),
                    ragged={k: v[start:end] for k, v in self.ragged.items()},
                )
            )
        return result

    @staticmethod
    def concat(protos: list[DataProto]) -> DataProto:
        """Concatenate DataProtos along dim-0 without sequence auto-padding."""
        if not protos:
            return DataProto()
        non_empty = [p for p in protos if p.batch_size > 0]
        if not non_empty:
            return DataProto(meta=protos[0].meta.copy())

        ref_keys = non_empty[0].keys()
        merged_tensors = {
            key: torch.cat([p.tensors[key] for p in non_empty], dim=0)
            for key in ref_keys
        }
        return DataProto(
            tensors=merged_tensors,
            meta=non_empty[0].meta.copy(),
            ragged=DataProto._ragged_concat(non_empty),
        )

    @staticmethod
    def merge(protos: list[DataProto]) -> DataProto:
        """Merge a list of DataProtos by concatenating tensors along dim 0.

        All non-empty chunks must share the same set of tensor keys. A
        ``ValueError`` is raised if keys are inconsistent across chunks.
        """
        if not protos:
            return DataProto()

        non_empty = [p for p in protos if p.batch_size > 0]
        if not non_empty:
            return DataProto(meta=protos[0].meta.copy())

        ref_keys = set(non_empty[0].keys())
        for idx, p in enumerate(non_empty[1:], start=1):
            chunk_keys = set(p.keys())
            if chunk_keys != ref_keys:
                missing = ref_keys - chunk_keys
                extra = chunk_keys - ref_keys
                raise ValueError(
                    f"DataProto.merge: chunk {idx} keys mismatch. "
                    f"missing={missing or '{}'}, extra={extra or '{}'}"
                )

        merged_tensors = {}
        for key in ref_keys:
            parts = [p[key] for p in non_empty]
            if parts[0].ndim >= 2:
                max_seq = max(t.shape[1] for t in parts)
                if any(t.shape[1] != max_seq for t in parts):
                    padded = []
                    for t in parts:
                        diff = max_seq - t.shape[1]
                        if diff > 0:
                            trailing_dims = t.ndim - 2
                            pad_spec = [0, 0] * trailing_dims + [0, diff]
                            t = torch.nn.functional.pad(t, pad_spec, value=0)
                        padded.append(t)
                    parts = padded
            merged_tensors[key] = torch.cat(parts, dim=0)

        return DataProto(
            tensors=merged_tensors,
            meta=non_empty[0].meta.copy(),
            ragged=DataProto._ragged_concat(non_empty),
        )

    def select(self, keys: list[str]) -> DataProto:
        """Return a new DataProto with only the specified tensor keys."""
        return DataProto(
            tensors={k: self.tensors[k] for k in keys if k in self.tensors},
            meta=self.meta.copy(),
            ragged={k: list(v) for k, v in self.ragged.items()},
        )

    def update(self, other: DataProto) -> None:
        """Update tensors, ragged columns and meta from another DataProto."""
        self.tensors.update(other.tensors)
        self.ragged.update(other.ragged)
        self.meta.update(other.meta)

    def reorder(self, idxs: int | slice | list[int] | torch.Tensor) -> None:
        """Reorder rows in-place according to indices."""
        if self.batch_size == 0:
            return
        norm = self._normalize_indices(idxs)
        for key, tensor in self.tensors.items():
            index = norm.to(device=tensor.device)
            self.tensors[key] = tensor[index]
        rows = norm.nonzero(as_tuple=True)[0] if norm.dtype == torch.bool else norm
        self.ragged = self._ragged_at(int(i) for i in rows.tolist())

    def pad_to_divisor(self, size_divisor: int) -> tuple[DataProto, int]:
        """Pad rows so batch size becomes divisible by ``size_divisor``."""
        if size_divisor <= 0:
            raise ValueError(f"size_divisor must be positive, got {size_divisor}")
        if self.batch_size == 0:
            return DataProto(meta=self.meta.copy()), 0

        remainder = self.batch_size % size_divisor
        if remainder == 0:
            return (
                DataProto(
                    tensors={k: v.clone() for k, v in self.tensors.items()},
                    meta=self.meta.copy(),
                    ragged={k: list(v) for k, v in self.ragged.items()},
                ),
                0,
            )

        pad_size = size_divisor - remainder
        idxs = torch.arange(pad_size, dtype=torch.long) % self.batch_size
        padding_part = self.select_idxs(idxs)
        return DataProto.concat([self, padding_part]), pad_size

    def unpad(self, pad_size: int) -> DataProto:
        """Remove trailing padded rows from a DataProto."""
        if pad_size < 0:
            raise ValueError(f"pad_size must be non-negative, got {pad_size}")
        if pad_size == 0 or self.batch_size == 0:
            return DataProto(
                tensors={k: v.clone() for k, v in self.tensors.items()},
                meta=self.meta.copy(),
                ragged={k: list(v) for k, v in self.ragged.items()},
            )
        keep = max(self.batch_size - pad_size, 0)
        return self.select_idxs(slice(0, keep))

    def repeat(self, repeat_times: int = 2, interleave: bool = True) -> DataProto:
        """Repeat all rows by a scalar repeat factor."""
        if repeat_times <= 0:
            raise ValueError(f"repeat_times must be positive, got {repeat_times}")
        repeated: dict[str, torch.Tensor] = {}
        for key, tensor in self.tensors.items():
            if interleave:
                repeated[key] = tensor.repeat_interleave(repeat_times, dim=0)
            else:
                repeated[key] = (
                    tensor.unsqueeze(0)
                    .expand(repeat_times, *tensor.shape)
                    .reshape(-1, *tensor.shape[1:])
                )
        n = self.batch_size
        order = (
            [i for i in range(n) for _ in range(repeat_times)]
            if interleave
            else [i for _ in range(repeat_times) for i in range(n)]
        )
        return DataProto(
            tensors=repeated,
            meta=self.meta.copy(),
            ragged=self._ragged_at(order),
        )

    def sample_level_repeat(
        self, repeat_times: list[int] | tuple[int, ...] | torch.Tensor
    ) -> DataProto:
        """Repeat each row with per-sample repeat factors."""
        if isinstance(repeat_times, tuple):
            repeat_times = list(repeat_times)
        if isinstance(repeat_times, list):
            repeats = torch.as_tensor(repeat_times, dtype=torch.long)
        elif isinstance(repeat_times, torch.Tensor):
            repeats = repeat_times.to(dtype=torch.long)
        else:
            raise TypeError(f"Unsupported repeat_times type: {type(repeat_times)}")

        if repeats.ndim != 1:
            raise ValueError("repeat_times must be 1-dimensional")
        if repeats.numel() != self.batch_size:
            raise ValueError(
                f"repeat_times length mismatch: expected {self.batch_size}, got {repeats.numel()}"
            )
        if (repeats < 0).any():
            raise ValueError("repeat_times values must be non-negative")

        repeated = {
            key: tensor.repeat_interleave(repeats.to(device=tensor.device), dim=0)
            for key, tensor in self.tensors.items()
        }
        order = [i for i, r in enumerate(repeats.tolist()) for _ in range(r)]
        return DataProto(
            tensors=repeated,
            meta=self.meta.copy(),
            ragged=self._ragged_at(order),
        )

    def mini_batches(self, batch_size: int) -> Iterator[DataProto]:
        """Yield mini-batches of the given size."""
        for start in range(0, self.batch_size, batch_size):
            end = min(start + batch_size, self.batch_size)
            chunk_tensors = {k: v[start:end] for k, v in self.tensors.items()}
            yield DataProto(
                tensors=chunk_tensors,
                meta=self.meta.copy(),
                ragged={k: v[start:end] for k, v in self.ragged.items()},
            )

    def add_router_distributions(self, layer_idx: int, logits: torch.Tensor) -> None:
        """Add MoE router distribution for a given layer (R3 support)."""
        key = f"router_dist_layer_{layer_idx}"
        self.tensors[key] = logits.cpu()

    def get_router_distributions(self) -> dict[int, torch.Tensor]:
        """Retrieve all stored router distributions keyed by layer index."""
        result: dict[int, torch.Tensor] = {}
        prefix = "router_dist_layer_"
        for key, tensor in self.tensors.items():
            if key.startswith(prefix):
                layer_idx = int(key[len(prefix) :])
                result[layer_idx] = tensor
        return result

    def has_router_distributions(self) -> bool:
        return any(k.startswith("router_dist_layer_") for k in self.tensors)

    # ------------------------------------------------------------------
    # Packing convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_packed(
        cls,
        packed_input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        extra_tensors: dict[str, torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> DataProto:
        """Create a DataProto from already-packed (remove-padding) tensors.

        Args:
            packed_input_ids: ``(1, total_tokens)`` or ``(total_tokens,)``
            cu_seqlens: ``(B+1,)`` cumulative sequence lengths
            position_ids: ``(1, total_tokens)`` or ``(total_tokens,)`` optional
            extra_tensors: additional tensors to include
            meta: metadata dict
        """
        cu_seqlens.shape[0] - 1
        seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

        tensors: dict[str, torch.Tensor] = {
            "input_ids": packed_input_ids,
            "cu_seqlens": cu_seqlens,
            "seq_lens": seq_lens,
        }
        if position_ids is not None:
            tensors["position_ids"] = position_ids
        if extra_tensors:
            tensors.update(extra_tensors)

        m = dict(meta or {})
        m["max_seqlen"] = int(seq_lens.max().item())
        return cls(tensors=tensors, meta=m)

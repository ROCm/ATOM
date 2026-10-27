# SPDX-License-Identifier: MIT
"""Request leases for vision outputs and scatter into arbitrary prefill chunks.

The engine releases leases after the last in-flight consumer completes. The
scheduler retains CPU payloads for retry, but sends them only until a successful
prefill acknowledges the worker's lease. Decode carries no image payload.
"""

import hashlib
from dataclasses import dataclass, field

import numpy as np
import torch


def multimodal_cache_seed(data):
    """Hash processed media once; token-only prefix identity is insufficient.

    Include layout as well as values: an identical patch buffer arranged into
    different image grids need not produce the same language embeddings.
    """
    digest = hashlib.blake2b(digest_size=8, person=b"ATOM-media-v1")
    for name in ("pixel_values", "image_grid_thw", "token_types"):
        value = data.get(name)
        if value is None:
            continue
        if hasattr(value, "detach"):
            value = value.detach().cpu().contiguous()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        else:
            value = np.ascontiguousarray(value)
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(value.tobytes())
    digest.update(repr(data.get("embedding_spans", ())).encode())
    return int.from_bytes(digest.digest(), "little")


def prefill_media_payload(data, cache_seed, *, cache_ready):
    """Keep span metadata; send media tensors only until the worker owns a lease."""
    omitted = {"token_types"}
    if cache_ready:
        omitted.update(("pixel_values", "image_grid_thw"))
    return {
        **{key: value for key, value in data.items() if key not in omitted},
        "cache_seed": cache_seed,
    }


def embedding_indices(spans, position, length):
    """Paired query/embedding indices for intersections with an input slice."""
    query, source, offset = [], [], 0
    for start, count in spans:
        first, end = max(start, position), min(start + count, position + length)
        if first < end:
            query.extend(range(first - position, end - position))
            source.extend(range(offset + first - start, offset + end - start))
        offset += count
    return query, source


@dataclass
class VisionEntry:
    values: torch.Tensor
    users: set[int] = field(default_factory=set)


class VisionEmbeddingCache:
    def __init__(self):
        self.entries: dict[int, VisionEntry] = {}
        self.leases: dict[int, int] = {}
        self.encodes = 0

    def acquire(self, request_id, data, encode):
        seed = data["cache_seed"]
        if request_id in self.leases and self.leases[request_id] != seed:
            raise ValueError("A live vision request cannot change its image identity")
        entry = self.entries.get(seed)
        if entry is None:
            if "pixel_values" not in data:
                raise RuntimeError("Vision lease missing for an acknowledged request")
            values = encode(data)
            expected = sum(count for _, count in data["embedding_spans"])
            if values.shape[0] != expected:
                raise ValueError(
                    "Vision output rows disagree with explicit image spans"
                )
            entry = VisionEntry(values)
            self.entries[seed] = entry
            self.encodes += len(data["embedding_spans"])
        entry.users.add(request_id)
        self.leases[request_id] = seed
        if entry.values.is_cuda:
            entry.values.record_stream(torch.cuda.current_stream(entry.values.device))
        return entry.values

    def release(self, request_ids):
        for request_id in request_ids:
            seed = self.leases.pop(request_id, None)
            if seed is None:
                continue
            entry = self.entries[seed]
            entry.users.remove(request_id)
            if not entry.users:
                del self.entries[seed]

    def clear(self):
        self.entries.clear()
        self.leases.clear()


def embed_multimodal_batch(model, cache, input_ids, batch, device, dtype):
    """Consume request leases and scatter only the current chunk's intersections."""
    hidden = model.embed_input_ids(input_ids)
    offset = 0
    for request_id, length, end in zip(
        batch.req_ids, batch.num_scheduled_tokens, batch.context_lens
    ):
        data = batch.multimodal_data.get(request_id)
        if data is not None:
            # Acquiring before the first image intersection lets later chunks
            # send descriptors only, even if the first chunk contains just text.
            values = cache.acquire(
                request_id,
                data,
                lambda payload: model.get_vision_embeddings(
                    payload["pixel_values"].to(device=device, dtype=dtype),
                    payload["image_grid_thw"],
                ),
            )
            query, source = embedding_indices(
                data["embedding_spans"], int(end) - int(length), int(length)
            )
            if query:
                query = torch.tensor(query, device=device) + offset
                source = torch.tensor(source, device=device)
                hidden[query] = values[source]
        offset += int(length)
    return hidden

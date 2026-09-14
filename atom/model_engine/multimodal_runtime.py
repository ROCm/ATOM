# SPDX-License-Identifier: MIT
"""Request leases for vision outputs and scatter into arbitrary prefill chunks.

The engine releases leases after the last in-flight consumer completes. The
scheduler retains CPU payloads for retry, but sends them only until a successful
prefill acknowledges the worker's lease. Decode carries no image payload.
"""

from dataclasses import dataclass, field

import torch

from .multimodal import embedding_indices


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

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
            # Hash the buffer in place; a video's pixels can be gigabytes.
            digest.update(memoryview(value.view(torch.uint8).numpy()))
        else:
            value = np.ascontiguousarray(value)
            digest.update(str((value.dtype, value.shape)).encode())
            digest.update(memoryview(value).cast("B"))
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


def vision_pass_groups(patch_counts: list[int], max_patches: int) -> list[int]:
    """Greedy group sizes, in grids, of at most ``max_patches`` patches each.

    A grid is never split: one larger than ``max_patches`` takes a pass alone.
    """
    groups: list[int] = []
    patches = 0
    for count in patch_counts:
        if groups and patches + count <= max_patches:
            groups[-1] += 1
            patches += count
        else:
            groups.append(1)
            patches = count
    return groups


def encode_grids_in_passes(
    encode, pixel_values, grid_thws, max_patches, device, dtype
) -> torch.Tensor:
    """``encode(pixel_values, grid_thws)`` over whole grids in bounded passes.

    A model sets ``vision_pass_patches`` when every op of its vision encoder
    works per grid or per token, so encoding consecutive groups of grids gives
    the rows of one encode, in order. Each pass moves only its own pixels to
    the device. Without it (None) the request's grids are encoded at once.
    """
    if max_patches is None:
        return encode(pixel_values.to(device=device, dtype=dtype), grid_thws)
    patch_counts = grid_thws.prod(dim=-1).tolist()
    if sum(patch_counts) != pixel_values.shape[0]:
        raise ValueError(
            f"pixel_values has {pixel_values.shape[0]} patches but the grids "
            f"describe {sum(patch_counts)}"
        )
    groups = vision_pass_groups(patch_counts, max_patches)
    grid_groups = torch.split(grid_thws, groups)
    pixel_groups = torch.split(
        pixel_values, [int(grids.prod(dim=-1).sum()) for grids in grid_groups]
    )
    outputs = [
        encode(pixels.to(device=device, dtype=dtype), grids)
        for pixels, grids in zip(pixel_groups, grid_groups)
    ]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


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
                lambda payload: encode_grids_in_passes(
                    model.get_vision_embeddings,
                    payload["pixel_values"],
                    payload["image_grid_thw"],
                    getattr(model, "vision_pass_patches", None),
                    device,
                    dtype,
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

# SPDX-License-Identifier: MIT
"""CPU references for metadata/state tests; GPU cases call production kernels."""

import torch

from atom.model_ops.attentions.deepseek_v41.planes import split_source
from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE


def prefill_window(history, step, geometry, *, stage=True):
    batches = step.cu_seqlens_q.numel() - 1
    ring = geometry.ring_slots
    size = geometry.prefill_staging(batches, step.running_tokens).nbytes
    source = step.prefill_window_staging
    if source is None:
        raise ValueError("Prefill window staging must be reserved before preparation")
    if source.numel() < size:
        raise ValueError("Prefill window staging is smaller than the live step")
    planes = split_source(source[:size], geometry.head_dim)
    if stage:
        stage_window(history, planes, step, ring)
    return planes, tuple(p[batches * ring :] for p in planes)


def stage_window(history, planes, step, ring):
    for batch in range(step.cu_seqlens_q.numel() - 1):
        if step.cu_seqlens_q[batch + 1] == step.cu_seqlens_q[batch]:
            continue
        slot = int(step.slots[batch])
        if slot < 0:
            continue
        for dst, old in zip(planes, history):
            dst[batch * ring : (batch + 1) * ring].copy_(
                old[slot * ring : (slot + 1) * ring]
            )


def reset_windows(sources, ptrs, slots, ring, dim):
    for source in sources:
        for plane in split_source(source, dim):
            plane.view(plane.shape[0] // ring, -1).index_fill_(0, slots, 0)


def gather_windows(sources, ptrs, layer_ids, layers, slots, ring, dim):
    single = isinstance(layers, int)
    layers = layer_ids[layers : layers + 1] if single else layers.flatten()
    output = torch.empty(
        (layers.numel(), slots.numel(), ring, dim), dtype=torch.bfloat16
    )
    rows = slots.long()[:, None] * ring + torch.arange(ring)
    for at, layer in enumerate(layers.tolist()):
        data, scale = split_source(sources[layer], dim)
        raw, codes = data[rows], scale[rows]
        steps = torch.exp2(codes.float() - 127)
        steps[codes == 255] = float("nan")
        output[at].copy_(raw.view(FP8_DTYPE).float() * steps.repeat_interleave(32, -1))
    return output[0] if single else output


def write_window_planes(values, scales, planes, step, window, dim, cached_until=None):
    for t in range(step.running_tokens):
        batch = int(step.batch_ids[t])
        if batch < 0 or t < int(step.cu_seqlens_q[batch + 1]) - window.ring_slots:
            continue
        pos = int(step.positions[t])
        if cached_until is not None and pos < int(cached_until[batch]):
            continue
        row = window.index(int(step.slots[batch]), pos)
        for plane, src in zip(planes, (values, scales)):
            plane[row].copy_(src.view(torch.uint8).view(-1, plane.shape[-1])[t])

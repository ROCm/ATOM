# SPDX-License-Identifier: MIT
"""ATOM's V4.1 metadata builder, with the one thing vLLM's capture needs.

Core allocates a dummy batch's scratch cache per call and frees it with the
call, which is right for ATOM's own engine: its capture entry fabricates a
batch with ``is_dummy_run=False`` against the serving cache, so no recorded
kernel ever holds a scratch address, and its only dummy batches come from
`warmup_model`, outside any capture.

This path does capture dummy batches. `v41_stage_step` falls back to a
fabricated dummy batch whenever a step arrives without a snapshot, without a
slot allocator, or before the proxy pool is bound -- which includes vLLM's
memory-profiling forward -- and under breakable capture a graph can be
recording while that happens. A scratch cache freed when the capture returns
leaves the recorded kernels pointing at freed memory, read back on the first
replay as an illegal access reported asynchronously somewhere else entirely.

So while capture is possible, hand out one cache, allocated once at the
capture ceiling and kept alive on the builder. Every bucket's dummy batch fits
inside it -- its page, slot and token counts are caps and the batch indexes
from zero -- so one allocation serves them all rather than one per bucket (the
STATE side alone is ~5 MiB per slot).
"""

from atom.model_ops.attentions.deepseek_v41.backend import (
    DeepseekV41MetadataBuilder,
)
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.plugin.vllm.breakable_capture import breakable_capture_enabled


class VllmDeepseekV41MetadataBuilder(DeepseekV41MetadataBuilder):
    """The builder ATOM ships, with this path's two answers."""

    # The deferred stale-state verdict is wrong under concurrency here, and
    # wrong in the direction that takes the engine down -- it refuses correct
    # state. `DeepseekV41MetadataBuilder._state_probe_blocking` carries the
    # measurement; this is where this path says it wants the blocking one.
    _state_probe_blocking_default = True

    def _dummy_cache(self, pages, slots, running_tokens):
        if not breakable_capture_enabled():
            return super()._dummy_cache(pages, slots, running_tokens)
        cached = getattr(self, "_capture_dummy_cache", None)
        if cached is None:
            ceiling_pages = max(
                pages, -(-self.max_num_batched_tokens // self.block_size)
            )
            cached = PagedAttentionCache(
                self.geometry,
                ceiling_pages,
                max(slots, self.max_bs),
                self.device,
                max_tokens=max(running_tokens, self.max_num_batched_tokens),
                workspace=self.score_workspace,
            )
            self._capture_dummy_cache = cached
        return cached

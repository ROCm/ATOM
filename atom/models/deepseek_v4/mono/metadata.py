# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4 metadata builder: native pools with model-owned Mono lifetimes."""

from atom.model_ops.attentions.deepseek_v4_attn import (
    DeepseekV4AttentionMetadataBuilder,
)


class MonoMetadataBuilder(DeepseekV4AttentionMetadataBuilder):
    """Native metadata and pool lifecycle, with model-owned Mono resources."""

    def __init__(self, model_runner):
        super().__init__(model_runner)
        self._mono_session = None
        self._mono_batch = None
        self._mono_cache_ready = False
        from atom.models.deepseek_v4.mono.dispatch import install_mono_decode

        model_runner.model = install_mono_decode(
            model_runner.model,
            model_runner.config,
            getattr(model_runner, "drafter", None),
            self,
        )

    def invalidate_pool_rows(self):
        # The existing pool hook runs before a new allocation replaces any
        # native address. Retire captured readers while the old pools exist.
        if self._mono_session is not None:
            self._mono_session.engine.cache_rebinding()
        self._mono_cache_ready = False
        super().invalidate_pool_rows()

    def allocate_kv_cache_tensors(self, *, blocks, buf):
        result = super().allocate_kv_cache_tensors(blocks=blocks, buf=buf)
        self._mono_cache_ready = True
        return result

    def prepare_mono(self):
        # Binding waits for the first model/capture call, after the native
        # module walk and KV accounting have both finished.
        if (
            self._mono_session is not None
            and self._mono_cache_ready
            and self._mono_session.adapter is None
        ):
            self._mono_cache_ready = False
            self._mono_session.engine.cache_bound()

    def close(self):
        if self._mono_session is not None:
            self._mono_session.engine.shutdown()
        super().close()

    def build(self, batch, running_bs, running_tokens, max_seqlen_q):
        self._mono_batch = batch
        if (
            self._mono_session is None
            or not self._mono_session.profile.name.startswith("dspark_")
        ):
            return super().build(batch, running_bs, running_tokens, max_seqlen_q)
        operations = batch.state_maintenance_ops

        def maintain():
            if operations.relocations:
                self.relocate_state_slots(operations.relocations)
            if operations.checkpoint_stores or operations.checkpoint_restores:
                self.execute_paged_state_copies(
                    operations.checkpoint_stores, operations.checkpoint_restores
                )

        self._mono_session.run_state_maintenance(maintain, empty=operations.empty)
        if batch.total_tokens_num_prefill > 0:
            return self.prepare_prefill(batch, running_bs)
        return self.prepare_decode(batch, running_bs, running_tokens, max_seqlen_q)

    def build_for_cudagraph_capture(self, bs, max_q_len=None):
        self._mono_batch = None
        self.prepare_mono()
        return super().build_for_cudagraph_capture(bs, max_q_len)

    def blank_cache_write_targets(self):
        if self._mono_session is not None:
            self._mono_session.engine.bind_graphs()
        super().blank_cache_write_targets()

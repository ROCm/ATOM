# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wrap a native V4 metadata builder with Mono cache/capture lifetimes.

Installed explicitly by ``dispatch.install_mono_decode`` after admission. The
native builder keeps its buffers, pool geometry and metadata implementation;
this adapter neither constructs a second builder nor installs a model.
"""


class MonoMetadataBuilder:
    """Delegate native metadata while ordering Mono readers of its pools."""

    __slots__ = ("_mono_batch", "_mono_cache_ready", "_mono_session", "_native")

    def __init__(self, native):
        self._native = native
        self._mono_session = None
        self._mono_batch = None
        self._mono_cache_ready = False

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_native"), name)

    def __setattr__(self, name, value):
        if name in self.__slots__:
            object.__setattr__(self, name, value)
        else:
            # Native methods and external pool binding must see the same state.
            setattr(self._native, name, value)

    def invalidate_pool_rows(self):
        # The existing pool hook runs before a new allocation replaces any
        # native address. Retire captured readers while the old pools exist.
        if self._mono_session is not None:
            self._mono_session.engine.cache_rebinding()
        self._mono_cache_ready = False
        self._native.invalidate_pool_rows()

    def allocate_kv_cache_tensors(self, *, blocks, buf):
        result = self._native.allocate_kv_cache_tensors(blocks=blocks, buf=buf)
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
        self._native.close()

    def build(self, batch, *args, **kwargs):
        """Preserve the native build contract while ordering joint state writes."""
        self._mono_batch = batch
        if self._mono_session is None or not self._mono_session._has_draft_resources():
            return self._native.build(batch, *args, **kwargs)
        # Native owns the copy order, padding checks and both phase signatures.
        # A step that copies prefix/COW state runs that whole build after the
        # joint draft readers retire; empty maintenance still votes on all ranks.
        native_build = self._native.build
        result = None

        def build_native():
            nonlocal result
            result = native_build(batch, *args, **kwargs)

        empty = batch.state_maintenance_ops.empty
        self._mono_session.run_state_maintenance(build_native, empty=empty)
        if empty:
            return native_build(batch, *args, **kwargs)
        return result

    def build_for_cudagraph_capture(self, bs, max_q_len=None):
        self._mono_batch = None
        self.prepare_mono()
        return self._native.build_for_cudagraph_capture(bs, max_q_len)

    def blank_cache_write_targets(self):
        if self._mono_session is not None:
            self._mono_session.engine.bind_graphs()
        self._native.blank_cache_write_targets()

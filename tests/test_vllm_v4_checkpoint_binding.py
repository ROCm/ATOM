# SPDX-License-Identifier: MIT
"""Execute the complete bridge import/binder without a vLLM or GPU runtime."""

import os
import subprocess
import sys
from pathlib import Path


def _exercise_binding():
    import importlib.machinery
    import sys
    from types import ModuleType, SimpleNamespace

    from aiter_stub import atom_under_stubs

    paths = (
        "vllm",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention_layer_base",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backend",
        "vllm.v1.core",
        "vllm.v1.core.kv_cache_utils",
    )
    for name in paths:
        module = ModuleType(name)
        module.__path__ = []
        module.__spec__ = importlib.machinery.ModuleSpec(
            name, loader=None, is_package=True
        )
        sys.modules[name] = module

    class AttentionLayerBase:
        pass

    class AttentionBackend:
        pass

    class AttentionMetadataBuilder:
        pass

    sys.modules[
        "vllm.model_executor.layers.attention_layer_base"
    ].AttentionLayerBase = AttentionLayerBase
    backend = sys.modules["vllm.v1.attention.backend"]
    backend.AttentionBackend = AttentionBackend
    backend.AttentionMetadataBuilder = AttentionMetadataBuilder
    backend.AttentionCGSupport = SimpleNamespace(UNIFORM_BATCH=object())
    sys.modules["vllm.v1.core.kv_cache_utils"].resolve_block_hashes = lambda *args: None

    with atom_under_stubs() as load:
        bridge = load("atom.plugin.vllm.deepseek_v4_state_checkpoint")
        bridge._attach_native()
        copier = bridge._PluginV4StateCopier(
            geometry=None,
            arena_planes=[],
            row_widths=[],
            ratios=[],
            num_slots=0,
            max_num_seqs=1,
            spec=None,
        )
        # This is the inherited production base warmup, on its valid no-copy path.
        assert copier.warmup_per_req_cache() is None

        class Parent:
            def warmup_per_req_cache(self):
                return self.checkpoint_spec

            @staticmethod
            def _checkpoint_segment_sizes(value):
                return ("static", value)

            @classmethod
            def checkpoint_image_bytes(cls):
                return cls

            @property
            def _slot_views(self):
                return self._planes

        class Native(Parent):
            pass

        bridge._native_builder_cls = lambda: Native
        bridge._native_staging_fn = lambda: (lambda self: self)
        bridge._NATIVE_COPY_METHODS = (
            "warmup_per_req_cache",
            "_checkpoint_segment_sizes",
            "checkpoint_image_bytes",
            "_slot_views",
        )
        bridge._PluginV4StateCopier._native_attached = False
        bridge._attach_native()
        marker = object()
        copier = bridge._PluginV4StateCopier(
            geometry=None,
            arena_planes=[],
            row_widths=[],
            ratios=[],
            num_slots=0,
            max_num_seqs=1,
            spec=marker,
            kv_planes=marker,
        )
        assert copier.warmup_per_req_cache() is marker
        assert copier._checkpoint_segment_sizes(7) == ("static", 7)
        assert bridge._PluginV4StateCopier._checkpoint_segment_sizes(8) == ("static", 8)
        assert copier.checkpoint_image_bytes() is bridge._PluginV4StateCopier
        assert copier._slot_views is marker
    print("Inherited warmup and descriptor binding passed")


def test_native_checkpoint_binding_inherits_methods_and_preserves_descriptors():
    # Isolate external framework stand-ins; import all ATOM code normally.
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(root / "tests"))))
    result = subprocess.run(
        [sys.executable, __file__],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise_binding()

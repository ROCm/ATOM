# SPDX-License-Identifier: MIT
"""Close V4.1's builder after runner retirement, before vLLM's TP teardown."""

import functools
import logging
from dataclasses import dataclass
from typing import Any

import torch

from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

logger = logging.getLogger("atom")


@dataclass
class _V41Shutdown:
    builders: list[Any]
    runner_done: bool = False
    runner_result: Any = None


def _owned_builders(runner) -> list[Any]:
    # Keep only builders, not models/graph wrappers that the runner is retiring.
    builders = []
    model = getattr(runner, "model", None)
    builder = getattr(model, "_deepseek_v41_builder", None)
    if builder is not None:
        builders.append(builder)
    config = getattr(runner, "vllm_config", None)
    compilation = getattr(config, "compilation_config", None)
    for layer in getattr(compilation, "static_forward_context", {}).values():
        builder = getattr(layer, "_atom_v41_builder", None)
        if builder is not None and not any(builder is old for old in builders):
            builders.append(builder)
    return builders


def _wrap_shutdown(original):
    @functools.wraps(original)
    def shutdown(self, *args, **kwargs):
        state = getattr(self, "_atom_v41_shutdown", None)
        if state is None:
            builders = _owned_builders(self)
            if not builders:
                return original(self, *args, **kwargs)
            # V2 deletes model and clears static_forward_context. Preserve the
            # owner before either happens, also if the original shutdown raises.
            state = self._atom_v41_shutdown = _V41Shutdown(builders)
        if not state.runner_done:
            state.runner_result = original(self, *args, **kwargs)
            state.runner_done = True
        if state.builders:
            # The worker has stopped forward/replay before calling shutdown.
            # Retiring its graph owners does not imply that every external graph
            # reference vanished; no further replay is allowed after this point.
            with torch.cuda.device(self.device):
                torch.cuda.synchronize(self.device)
                while state.builders:
                    state.builders[0].close()
                    # An exception retains this owner and prevents WorkerProc
                    # from reaching TP teardown. Completed owners are not retried.
                    del state.builders[0]
            logger.info(
                "ATOM plugin: closed V4.1 metadata builders after runner shutdown"
            )
        return state.runner_result

    shutdown._atom_v41_shutdown_patched = True
    return shutdown


def apply_vllm_v41_shutdown_patch() -> int:
    """Cover both unrelated runner classes; other models pass through unchanged."""
    covered = 0
    for runner_cls in gpu_model_runner_classes():
        original = getattr(runner_cls, "shutdown", None)
        if original is None or getattr(original, "_atom_v41_shutdown_patched", False):
            continue
        runner_cls.shutdown = _wrap_shutdown(original)
        covered += 1
    return covered

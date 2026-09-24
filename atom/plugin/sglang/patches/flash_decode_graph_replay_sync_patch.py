# SPDX-License-Identifier: MIT
"""Synchronize after long Flash decode CUDA-graph replay to pin async HSA faults."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def apply_flash_decode_graph_replay_sync_patch() -> None:
    runners = []
    for mod_name, cls_name in (
        (
            "sglang.srt.model_executor.runner.decode_cuda_graph_runner",
            "DecodeCudaGraphRunner",
        ),
        ("sglang.srt.model_executor.cuda_graph_runner", "CudaGraphRunner"),
    ):
        try:
            import importlib

            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name, None)
            if cls is not None:
                runners.append(cls)
        except Exception:
            continue
    if not runners:
        logger.debug("No CUDA graph runner for Flash long-replay sync patch")
        return
    for cls in runners:
        if getattr(cls, "_atom_flash_long_replay_sync", False):
            continue
        for method_name in ("replay", "execute", "run"):
            if not hasattr(cls, method_name):
                continue
            original = getattr(cls, method_name)

            def _make(orig):
                def wrapped(self, *args, **kwargs):
                    result = orig(self, *args, **kwargs)
                    try:
                        import os

                        import torch

                        if os.environ.get("ATOM_FLASH_GRAPH_REPLAY_SYNC", "") != "1":
                            return result
                        seq = -1
                        fb = args[0] if args else kwargs.get("forward_batch")
                        if fb is not None:
                            sl = getattr(fb, "seq_lens", None)
                            if torch.is_tensor(sl) and sl.numel():
                                seq = int(sl.max().item())
                        if seq >= 512:
                            torch.cuda.synchronize()
                            logger.info(
                                "Flash long decode graph sync after %s (seq_max=%s)",
                                orig.__name__,
                                seq,
                            )
                    except Exception as exc:
                        logger.warning("Flash long decode graph sync failed: %s", exc)
                    return result

                return wrapped

            setattr(cls, method_name, _make(original))
        cls._atom_flash_long_replay_sync = True
        logger.info("Patched %s for Flash long decode graph sync", cls.__name__)

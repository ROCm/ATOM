"""Keep vLLM's worker rendezvous on a TCPStore for aiter custom all-reduce.

vLLM 0.28 switched the multiproc/uniproc executors to a ``file://`` rendezvous
(a ``torch.distributed.FileStore``) and only falls back to the previous
``tcp://`` rendezvous when ``aiter_requires_tcp_store()`` is true. That helper
inspects vLLM's *own* AITER switches (``VLLM_ROCM_USE_AITER`` plus
``VLLM_ROCM_USE_AITER_CUSTOM_AR``), which the ATOM plugin never sets: ATOM
brings up aiter's distributed environment itself, and aiter's
``CustomAllreduce`` IPC-metadata exchange asserts the default store is a
``TCPStore``. Without this patch every TP>1 ATOM run dies during model load:

    AssertionError: IPC metadata exchange requires a pure-TCP KV store
    (torch.distributed.TCPStore), got FileStore.

Forcing the TCP rendezvous restores exactly the vLLM 0.27.1 behaviour. Drop
this patch once aiter's IPC exchange accepts a FileStore.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger("atom")

_EXECUTOR_MODULES = (
    "vllm.v1.executor.multiproc_executor",
    "vllm.v1.executor.uniproc_executor",
)


def _atom_requires_tcp_store() -> bool:
    return True


def apply_vllm_tcp_store_patch() -> None:
    from vllm.utils import network_utils

    if getattr(network_utils, "_atom_tcp_store_patch", False):
        return

    if not hasattr(network_utils, "aiter_requires_tcp_store"):
        # vLLM < 0.28 always used the TCP rendezvous; nothing to force.
        return

    network_utils.aiter_requires_tcp_store = _atom_requires_tcp_store
    # The executors bind the helper at import time; patch any that already ran.
    for module_name in _EXECUTOR_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "aiter_requires_tcp_store"):
            module.aiter_requires_tcp_store = _atom_requires_tcp_store

    network_utils._atom_tcp_store_patch = True
    logger.info(
        "ATOM plugin: pinned the vLLM worker rendezvous to a TCPStore for "
        "aiter custom all-reduce."
    )

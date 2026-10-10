# SPDX-License-Identifier: MIT
"""Every GPU model runner class a worker might instantiate.

vLLM ships two unrelated `GPUModelRunner` classes -- the original in
`vllm.v1.worker.gpu_model_runner` and the V2 rewrite in
`vllm.v1.worker.gpu.model_runner` -- and `GPUWorker` picks between them on
`use_v2_model_runner`. They do not share a base class and do not share method
objects, so patching one leaves the other untouched. A patch written against
the first name alone is therefore not "applied unless vLLM moved it": it is
silently inert for every deployment that selects the other, and inert is
indistinguishable from applied in the logs.

Measured on 0.31.1 with the V2 runner selected: `issubclass` is False in both
directions and `A.initialize_kv_cache is B.initialize_kv_cache` is False.
"""

import logging

logger = logging.getLogger("atom")

_MODULES = (
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.worker.gpu.model_runner",
)


def gpu_model_runner_classes() -> list[type]:
    """The runner classes present in this vLLM, newest name last.

    Missing modules are skipped rather than raised on: which names exist is a
    property of the installed version, and a patch that must cover both should
    not refuse to cover one.
    """
    classes: list[type] = []
    for module_name in _MODULES:
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
            cls = module.GPUModelRunner
        except (ImportError, AttributeError):
            continue
        if cls not in classes:
            classes.append(cls)
    if not classes:
        logger.debug("ATOM plugin: no vLLM GPUModelRunner class found to patch")
    return classes

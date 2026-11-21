"""rocprofiler-register helper (AMD ROCm).

Ported from sgl-project/sglang PR #44292.

rocprofiler-register records the library that claimed registration in the
environment variable ``ROCPROFILER_REGISTER_LIBRARY``, which a process then
passes to every child it spawns. A child that inherits the claim believes a
profiler is already registered and never registers its own, so
``torch.profiler`` in the child records CPU rows only: it reports no GPU events
and no error.

ATOM spawns its GPU workers (engine-core / TP ranks) with the ``spawn`` start
method, and each worker imports torch at the top of its entry module, before
any ``atom`` code runs. The inherited claim therefore has to be dropped in the
PARENT, before the spawn -- see ``drop_inherited_registration_for_children``.
"""

import importlib.util
import os
import pathlib

__all__ = [
    "drop_inherited_registration_for_children",
    "torch_built_with_hip",
]


def torch_built_with_hip() -> bool:
    """Whether torch is a ROCm build, without importing torch.

    torch/version.py is nothing but a few constant assignments, so reading only
    that file is enough and avoids importing torch.
    """
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None or torch_spec.origin is None:
        return False
    version_path = pathlib.Path(torch_spec.origin).with_name("version.py")
    version_spec = importlib.util.spec_from_file_location("torch_version", version_path)
    if version_spec is None or version_spec.loader is None:
        return False
    version = importlib.util.module_from_spec(version_spec)
    try:
        version_spec.loader.exec_module(version)
    except OSError:
        return False
    return getattr(version, "hip", None) is not None


def _keep_requested() -> bool:
    # Imported lazily: this helper is only reached in the parent, after atom is
    # already loaded, so there is no torch-before-import concern here.
    from atom.utils import envs

    return envs.ATOM_KEEP_INHERITED_ROCPROFILER_REGISTER


def drop_inherited_registration_for_children() -> None:
    """Drop ROCPROFILER_REGISTER_LIBRARY so spawned children do not inherit it.

    Call this in the PARENT immediately before spawning GPU worker processes
    (the engine-core / TP ranks). Those workers import torch at the top of their
    entry module, before any ``atom`` import, so dropping the variable inside the
    worker would be too late: rocprofiler-sdk is already loaded there and the
    pop would have no effect. Clearing the variable in the parent first makes
    every child inherit a clean environment and register its own kineto client,
    so ``torch.profiler`` captures GPU events in each worker.

    Popping the variable here does NOT unregister the parent's own already-loaded
    library; it only removes the breadcrumb that would otherwise pass to
    children. Set ``ATOM_KEEP_INHERITED_ROCPROFILER_REGISTER=1`` to opt out and
    restore the old inherit-the-claim behaviour.
    """
    if not torch_built_with_hip():
        return
    if _keep_requested():
        return
    os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)

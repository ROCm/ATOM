"""rocprofiler-register helpers (AMD ROCm).

Ported from sgl-project/sglang PR #44292.

rocprofiler-register records the library that claimed registration in the
environment variable ``ROCPROFILER_REGISTER_LIBRARY``, which the parent process
then passes to every spawned child. A child that inherits the claim believes a
profiler is already registered and never registers its own, so ``torch.profiler``
in the child records CPU rows only: it reports no GPU events and no error.

This module drops such an inherited claim. It must run before torch is imported,
so nothing it imports may pull torch in (stdlib only).
"""

import importlib.util
import os
import pathlib

__all__ = [
    "clear_inherited_rocprofiler_registration",
    "drop_inherited_registration_for_children",
    "torch_built_with_hip",
]


def torch_built_with_hip() -> bool:
    """Whether torch is a ROCm build, without importing torch.

    torch/version.py is nothing but a few constant assignments, so reading only
    that file is enough and avoids importing torch (which would load the
    rocprofiler-sdk library and make the clear below a no-op).
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


def _rocprofiler_sdk_already_loaded() -> bool:
    """Whether this process has already loaded a rocprofiler-sdk library.

    A profiler that is really running has loaded its library into this process,
    so it shows up in the process memory map. If the map cannot be read, return
    True so the caller treats the process as one where a profiler may be running
    and changes nothing.
    """
    try:
        with open("/proc/self/maps") as f:
            return "librocprofiler-sdk" in f.read()
    except OSError:
        return True


def _keep_requested() -> bool:
    val = os.environ.get("ATOM_KEEP_INHERITED_ROCPROFILER_REGISTER", "0")
    return val.strip().lower() not in ("0", "", "false", "no")


def clear_inherited_rocprofiler_registration() -> None:
    """Delete ROCPROFILER_REGISTER_LIBRARY if this process inherited it.

    When kineto registers (PyTorch 2.12+ uses rocprofiler-sdk), HIP writes the
    library path into ROCPROFILER_REGISTER_LIBRARY. A spawned child inherits it,
    believes a profiler is already registered, refuses to register its own, and
    torch.profiler then records host ops only -- no GPU events, no error.

    Only ROCm sets this variable, and only PyTorch 2.12+ reads it (2.11 and older
    use roctracer). The clear is skipped when a profiler is genuinely loaded in
    this process, so it never breaks a profiler that is really collecting data.
    The process must call this before it imports torch.
    """
    if os.environ.get("ROCPROFILER_REGISTER_LIBRARY") is None:
        return
    if not torch_built_with_hip():
        return
    if _keep_requested():
        return
    if _rocprofiler_sdk_already_loaded():
        return
    os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)


def drop_inherited_registration_for_children() -> None:
    """Drop ROCPROFILER_REGISTER_LIBRARY so spawned children do not inherit it.

    Call this in the PARENT immediately before spawning GPU worker processes
    (the engine-core / TP workers). Those workers import torch at the top of
    their entry module, before any ``atom`` import, so the in-process clear in
    ``atom/__init__`` runs too late there (the rocprofiler-sdk library is
    already loaded and the guard skips). Clearing the variable in the parent
    first makes every child inherit a clean environment and register its own
    kineto client, so ``torch.profiler`` captures GPU events in each worker.

    Unlike ``clear_inherited_rocprofiler_registration`` this does NOT check
    ``/proc/self/maps``: the parent has the library loaded for its own reasons,
    but the environment breadcrumb must not pass to children. Popping it here
    does not unregister the parent's already-loaded library.
    """
    if not torch_built_with_hip():
        return
    if _keep_requested():
        return
    os.environ.pop("ROCPROFILER_REGISTER_LIBRARY", None)

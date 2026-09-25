"""Isolate HIP IPC event import from the LMCache server's Python GIL."""

import json
from pathlib import Path
from types import SimpleNamespace


def install():
    import torch
    from lmcache import torch_dev
    from lmcache.v1.platform.base.event_ipc import (
        DefaultEventIPCBackend,
        get_event_ipc_backend,
    )
    from torch.cuda._utils import _get_device_index
    from torch.utils.cpp_extension import ROCM_HOME, load

    if not torch.version.hip or not ROCM_HOME:
        raise RuntimeError("The IPC import experiment requires ROCm")
    backend = get_event_ipc_backend("cuda")
    if type(backend) is not DefaultEventIPCBackend or backend._event_module is not torch_dev:
        raise RuntimeError("Unexpected LMCache event backend")
    build = Path("/tmp/k3-hip-event-import")
    build.mkdir(exist_ok=True)
    native = load(
        name="k3_hip_event_import",
        sources=[str(Path(__file__).with_suffix(".cpp"))],
        build_directory=str(build),
        extra_cflags=["-D__HIP_PLATFORM_AMD__", "-DUSE_ROCM=1"],
        extra_include_paths=[str(Path(ROCM_HOME) / "include")],
        extra_ldflags=[
            "-lc10_hip",
            "-ltorch_hip",
            f"-L{ROCM_HOME}/lib",
            "-lamdhip64",
        ],
        with_cuda=False,
        verbose=True,
    )

    class EventFactory:
        def __new__(cls, interprocess=False):
            return torch_dev.Event(interprocess=interprocess)

        @staticmethod
        def from_ipc_handle(device, handle):
            index = _get_device_index(device, optional=True)
            return native.from_ipc_handle(index, handle)

    backend._event_module = SimpleNamespace(Event=EventFactory)
    backend.check_event_support("cuda")
    print(
        "K3_IMPORT_GIL "
        + json.dumps(
            {
                "torch": torch.__version__,
                "torch_git": torch.version.git_version,
                "hip": torch.version.hip,
                "python_gil_released_during_import": True,
                "scope": "LMCache server only",
            }
        ),
        flush=True,
    )
    return backend

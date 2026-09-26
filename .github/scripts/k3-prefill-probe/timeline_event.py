"""Bounded process-lifetime counter backend for the single-P experiment only.

Mappings stay registered until process exit, with a hard limit of 64 per process.
There is no production context-recycling policy here. A memfd has no persistent
filesystem name; its creator keeps the FD open while remote imports are possible.
"""

import argparse
import ast
import ctypes
import fcntl
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

BUILD = Path("/tmp/k3-prefill-timeline")
LIMIT = (1 << 63) - 1
MAX_CHANNELS = 64
_registry = None
_factory = None


def emit(kind, **fields):
    print(
        "K3_TIMELINE " + json.dumps({"kind": kind, "pid": os.getpid(), **fields}),
        flush=True,
    )


def build():
    BUILD.mkdir(exist_ok=True)
    source = Path(__file__).with_name("timeline.cpp")
    rocm = Path(os.environ.get("ROCM_HOME", "/opt/rocm"))
    temporary = BUILD / "timeline.so.tmp"
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-shared",
        "-fPIC",
        "-pthread",
        "-D__HIP_PLATFORM_AMD__",
        f"-I{rocm}/include",
        str(source),
        f"-L{rocm}/lib",
        "-lamdhip64",
        f"-Wl,-rpath,{rocm}/lib",
        "-o",
        str(temporary),
    ]
    subprocess.run(command, check=True, timeout=120)
    temporary.replace(BUILD / "timeline.so")
    manifest = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(
            (BUILD / "timeline.so").read_bytes()
        ).hexdigest(),
        "command": command,
    }
    (BUILD / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    emit("build", **manifest)


class Native:
    def __init__(self):
        if os.environ.get("GPU_STREAMOPS_CP_WAIT", "0") != "0":
            raise RuntimeError(
                "Counter probe requires GPU_STREAMOPS_CP_WAIT unset or 0"
            )
        manifest = json.loads((BUILD / "build.json").read_text())
        if (
            manifest["source_sha256"]
            != hashlib.sha256(
                Path(__file__).with_name("timeline.cpp").read_bytes()
            ).hexdigest()
        ):
            raise RuntimeError("Counter native source differs from compiled source")
        if (
            manifest["library_sha256"]
            != hashlib.sha256((BUILD / "timeline.so").read_bytes()).hexdigest()
        ):
            raise RuntimeError("Counter library differs from build manifest")
        self.lib = ctypes.CDLL(str(BUILD / "timeline.so"))
        self.poisoned = False
        ptr, u64 = ctypes.c_void_p, ctypes.c_uint64
        self.lib.k3_error.restype = ctypes.c_char_p
        for name, signature in {
            "k3_open_fd": [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_size_t,
                ctypes.POINTER(ptr),
            ],
            "k3_record": [ptr, ctypes.c_size_t, ctypes.POINTER(u64)],
            "k3_wait": [ptr, ctypes.c_size_t, u64],
            "k3_query": [ptr, ctypes.POINTER(u64)],
            "k3_close_after_drain": [ptr, u64],
        }.items():
            method = getattr(self.lib, name)
            method.argtypes, method.restype = signature, ctypes.c_int

    def call(self, name, *args):
        if self.poisoned:
            raise RuntimeError(
                "Counter backend is poisoned by an earlier native failure"
            )
        if getattr(self.lib, name)(*args):
            self.poisoned = True
            raise RuntimeError(self.lib.k3_error().decode())


class Channel:
    def __init__(self, registry, identity, device, descriptor, owner, stream=None):
        self.registry, self.identity, self.device = registry, identity, device
        self.owner, self.stream = owner, stream  # Keep the writer stream alive.
        self.fd = descriptor if owner else None
        self.pointer = ctypes.c_void_p()
        self.lock = threading.Lock()
        self.records = self.waits = self.queries = self.synchronizes = 0
        self.closed = False
        with registry.torch.cuda.device(device):
            registry.native.call(
                "k3_open_fd",
                descriptor,
                hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode())
                .hexdigest()[:32]
                .encode(),
                identity[0].encode(),
                identity[2].encode(),
                device,
                int(owner),
                stream.cuda_stream if owner else 0,
                ctypes.byref(self.pointer),
            )

    def check_open(self):
        self.registry.check_process()
        if self.closed:
            raise RuntimeError("Counter channel closed")

    def record(self, stream):
        self.check_open()
        with self.lock, self.registry.torch.cuda.device(self.device):
            self.check_open()
            value = ctypes.c_uint64()
            self.registry.native.call(
                "k3_record", self.pointer, stream.cuda_stream, ctypes.byref(value)
            )
            self.records += 1
            return value.value

    def wait(self, stream, generation):
        self.check_open()
        if stream.device.index != self.device:
            raise ValueError("Counter wait stream device mismatch")
        with self.lock, self.registry.torch.cuda.device(self.device):
            self.check_open()
            self.registry.native.call(
                "k3_wait", self.pointer, stream.cuda_stream, generation
            )
            self.waits += 1

    def value(self):
        self.check_open()
        with self.lock:
            self.check_open()
            value = ctypes.c_uint64()
            self.registry.native.call("k3_query", self.pointer, ctypes.byref(value))
            self.queries += 1
            return value.value


class Registry:
    def __init__(self, torch):
        scope = os.environ.get("ATOMESH_RUN_TOKEN", "")
        if not re.fullmatch(r"[0-9a-f]{32}", scope):
            raise RuntimeError("Counter probe requires the experiment run token")
        self.scope, self.torch = scope, torch
        self.pid = os.getpid()
        self.native = Native()
        self.lock = threading.RLock()
        self.channels, self.writers, self.device_uuids = {}, {}, {}

    def _budget(self):
        self.check_process()
        if len(self.channels) >= MAX_CHANNELS:
            raise RuntimeError(
                "Single-P counter experiment reached its 64-channel limit"
            )

    def check_process(self):
        if os.getpid() != self.pid:
            raise RuntimeError("Counter registry cannot be inherited across fork")
        if self.native.poisoned:
            raise RuntimeError("Counter registry is poisoned by a native failure")

    def device_uuid(self, device):
        if device not in self.device_uuids:
            self.device_uuids[device] = str(
                self.torch.cuda.get_device_properties(device).uuid
            )
        return self.device_uuids[device]

    def writer(self, stream):
        self.check_process()
        device = stream.device.index
        if type(device) is not int or not 0 <= device < 8:
            raise ValueError("The experiment requires a local device in 0..7")
        if isinstance(stream, self.torch.cuda.ExternalStream):
            raise TypeError("External streams are outside this experiment")
        key = (device, stream.cuda_stream)
        with self.lock:
            if key in self.writers:
                return self.writers[key]
            self._budget()
            epoch = uuid.uuid4().hex
            fd = os.memfd_create(
                f"k3-counter-{epoch}", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
            )
            try:
                os.ftruncate(fd, 4096)
                fcntl.fcntl(
                    fd,
                    fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL,
                )
                identity = (
                    self.scope,
                    epoch,
                    self.device_uuid(device),
                    os.getpid(),
                    fd,
                )
                channel = Channel(self, identity, device, fd, True, stream)
            except BaseException:
                self.native.poisoned = True
                os.close(fd)
                raise
            self.channels[identity] = self.writers[key] = channel
            emit(
                "map",
                owner=True,
                identity=identity,
                device=device,
                stream=stream.cuda_stream,
                registered_channels=len(self.channels),
                limit=MAX_CHANNELS,
            )
            return channel

    def imported(self, handle, device):
        self.check_process()
        if (
            type(handle) is not bytes
            or not 0 < len(handle) <= 512
            or not handle.startswith(b"K3TL1:")
        ):
            raise ValueError("Invalid counter wire header")
        fields = json.loads(handle[6:])
        if type(fields) is not list or len(fields) != 6:
            raise ValueError("Invalid counter wire fields")
        scope, epoch, gpu_uuid, pid, fd, generation = fields
        if (
            scope != self.scope
            or type(epoch) is not str
            or not re.fullmatch(r"[0-9a-f]{32}", epoch)
            or type(gpu_uuid) is not str
            or not 0 < len(gpu_uuid) < 128
            or type(pid) is not int
            or pid <= 0
            or type(fd) is not int
            or not 0 <= fd < 1_048_576
            or type(generation) is not int
            or not 0 < generation <= LIMIT
        ):
            raise ValueError("Invalid counter identity/generation")
        identity = tuple(fields[:5])
        with self.lock:
            if gpu_uuid != self.device_uuid(device):
                raise ValueError("Counter physical device UUID mismatch")
            if identity not in self.channels:
                self._budget()
                descriptor = os.open(f"/proc/{pid}/fd/{fd}", os.O_RDWR | os.O_CLOEXEC)
                try:
                    channel = Channel(self, identity, device, descriptor, False)
                except BaseException:
                    self.native.poisoned = True
                    raise
                finally:
                    os.close(descriptor)
                self.channels[identity] = channel
                emit(
                    "map",
                    owner=False,
                    identity=identity,
                    device=device,
                    registered_channels=len(self.channels),
                    limit=MAX_CHANNELS,
                )
            return CounterEvent(
                self, self.channels[identity], generation, imported=True
            )

    def close_after_test_drain(self, expected):
        """Only the startup smoke calls this after its explicit two-sided drain."""
        self.check_process()
        with self.lock:
            self.check_process()
            if set(expected) != set(self.channels):
                raise ValueError("Drain ACK must cover every registered channel")
            for identity, channel in self.channels.items():
                generation = expected[identity]
                if type(generation) is not int or not 0 < generation <= LIMIT:
                    raise ValueError("Invalid expected generation at drain")
                if channel.value() != generation:
                    raise RuntimeError("Drain did not reach the last issued generation")
            for channel in self.channels.values():
                with channel.lock, self.torch.cuda.device(channel.device):
                    self.native.call(
                        "k3_close_after_drain",
                        channel.pointer,
                        expected[channel.identity],
                    )
                    channel.closed = True
                    if channel.fd is not None:
                        os.close(channel.fd)
            emit("test_close", channels=len(self.channels))
            self.channels.clear()
            self.writers.clear()


@dataclass(frozen=True, slots=True, init=False)
class CounterEvent:
    registry: object
    _state: object
    imported: bool
    _lock: object = field(compare=False, repr=False)

    def __init__(self, registry, channel=None, generation=None, imported=False):
        object.__setattr__(self, "registry", registry)
        object.__setattr__(self, "_state", (channel, generation) if channel else None)
        object.__setattr__(self, "imported", imported)
        object.__setattr__(self, "_lock", threading.Lock())

    @property
    def channel(self):
        return self._state[0] if self._state else None

    @property
    def generation(self):
        return self._state[1] if self._state else None

    def record(self, stream=None):
        self.registry.check_process()
        with self._lock:
            if self.imported or self._state is not None:
                raise RuntimeError("Counter event supports exactly one record")
            stream = stream or self.registry.torch.cuda.current_stream()
            channel = self.registry.writer(stream)
            generation = channel.record(stream)
            object.__setattr__(self, "_state", (channel, generation))

    def _recorded(self):
        if self.channel is None or self.generation is None:
            raise RuntimeError(
                "Counter event must be recorded before export/wait/query"
            )
        self.channel.check_open()

    def ipc_handle(self):
        self._recorded()
        return (
            b"K3TL1:"
            + json.dumps(
                [*self.channel.identity, self.generation], separators=(",", ":")
            ).encode()
        )

    def wait(self, stream=None):
        self._recorded()
        stream = stream or self.registry.torch.cuda.current_stream(self.channel.device)
        self.channel.wait(stream, self.generation)

    def query(self):
        self._recorded()
        return self.channel.value() >= self.generation

    def synchronize(self):
        # Preserves LMCache's existing synchronous future API. Hot query/wait
        # never calls this or a device/stream synchronize operation.
        self._recorded()
        self.channel.synchronizes += 1
        while not self.query():
            time.sleep(0.001)


def install():
    global _registry, _factory
    if _registry is not None:
        _registry.check_process()
    import importlib.metadata
    import inspect

    import lmcache
    import torch
    from lmcache import torch_dev
    from lmcache.v1.platform import current_device_spec, get_device_spec
    from lmcache.v1.platform.base.event_ipc import (
        DefaultEventIPCBackend,
        get_event_ipc_backend,
    )
    from lmcache.v1.platform.isolated_ipc import is_isolated_ipc
    from torch.cuda._utils import _get_device_index

    if not torch.version.hip:
        raise RuntimeError("Counter experiment requires ROCm")
    backend = get_event_ipc_backend("cuda")
    if _registry is not None:
        _registry.check_process()
        if backend._event_module.Event is not _factory:
            raise RuntimeError("Counter event factory changed after installation")
        return backend
    if (
        importlib.metadata.version("lmcache") != "0.5.5.dev114+rocm7.2"
        or torch_dev is not torch.cuda
        or current_device_spec.backend_name != "rocm"
        or get_device_spec("cuda") is not current_device_spec
        or get_event_ipc_backend(0) is not backend
        or is_isolated_ipc()
        or type(backend) is not DefaultEventIPCBackend
        or backend._event_module is not torch_dev
    ):
        raise RuntimeError("Unexpected LMCache event backend")
    root = Path(lmcache.__file__).resolve().parent
    distribution_root = Path(
        importlib.metadata.distribution("lmcache").locate_file("")
    ).resolve()
    if root.parent != distribution_root:
        raise RuntimeError("LMCache metadata and imported source have different roots")
    if not Path(inspect.getfile(DefaultEventIPCBackend)).resolve().is_relative_to(root):
        raise RuntimeError("Mixed LMCache package roots")
    worker_source = root / "v1/multiprocess/transfer_context/worker_transfer.py"
    worker_tree = ast.parse(worker_source.read_text())
    expected = ast.parse(
        "event = self._event_backend.create_event(self._device)\n"
        "self._event_backend.record_event(event, torch_dev.current_stream())\n"
        "return cast(IPCEvent, event)"
    ).body
    factories = [
        node
        for node in ast.walk(worker_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "create_recorded_event"
        and ast.dump(ast.Module(body=node.body[-3:], type_ignores=[]))
        == ast.dump(ast.Module(body=expected, type_ignores=[]))
    ]
    if len(factories) != 1:
        raise RuntimeError("LMCache worker does not use the expected IPC event factory")
    registry = Registry(torch)

    class Factory:
        def __new__(cls, interprocess=False):
            if not interprocess:
                raise ValueError("Only IPC events are supported by this experiment")
            return CounterEvent(registry)

        @staticmethod
        def from_ipc_handle(device, handle):
            registry.check_process()
            return registry.imported(handle, _get_device_index(device, optional=True))

    backend._event_module = SimpleNamespace(Event=Factory)
    backend.check_event_support("cuda")
    _registry, _factory = registry, Factory
    emit(
        "install",
        scope=registry.scope,
        max_channels=MAX_CHANNELS,
        lifetime="process; only startup smoke explicitly drains and closes",
        torch=torch.__version__,
        hip=torch.version.hip,
    )
    return backend


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    build()

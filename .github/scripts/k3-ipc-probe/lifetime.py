"""Exercise the actual dev114 and upstream5280 backends with HIP IPC handles."""

import gc
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import torch


def shm_snapshot(handle):
    # ROCm 7.2 HIP IPC handles begin with the NUL-terminated shared name.
    name = handle.split(b"\0", 1)[0].decode("ascii")
    if not re.fullmatch(r"/hip_[A-Za-z0-9_]+", name):
        raise ValueError(f"Unexpected HIP IPC handle name: {name!r}")
    path = Path("/dev/shm") / name[1:]
    try:
        stat = path.stat()
        return {
            "name": name,
            "exists": True,
            "inode": stat.st_ino,
            "device": stat.st_dev,
            "size": stat.st_size,
        }
    except FileNotFoundError:
        return {"name": name, "exists": False}


if len(sys.argv) == 3 and sys.argv[1] == "--import":
    try:
        handle = bytes.fromhex(sys.argv[2])
        before = shm_snapshot(handle)
        event = torch.cuda.Event.from_ipc_handle(0, handle)
        event.synchronize()
        print(
            json.dumps(
                {
                    "imported": True,
                    "ready": event.query(),
                    "before": before,
                    "imported_shm": shm_snapshot(handle),
                }
            ),
            flush=True,
        )
        del event
        gc.collect()
        print(json.dumps({"after_delete": shm_snapshot(handle)}), flush=True)
    except (RuntimeError, OSError, ValueError) as exc:
        print(json.dumps({"imported": False, "error": str(exc)}), flush=True)
        raise SystemExit(1)
    raise SystemExit(0)

torch.cuda.init()
package = types.ModuleType("lmcache")
package.torch_dev = torch.cuda
package.torch_device_type = "cuda"
sys.modules["lmcache"] = package
results = {
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "device": torch.cuda.get_device_name(0),
    "cases": {},
}


def save():
    Path("/results/lifetime.json").write_text(json.dumps(results, indent=2) + "\n")


for label in ["stock", "patched"]:
    path = Path(__file__).with_name("event_" + label + ".py")
    spec = importlib.util.spec_from_file_location("backend_" + label, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = module.DefaultEventIPCBackend(event_module=torch.cuda, device_type="cuda")
    event = backend.create_event(torch.device("cuda:0"))
    backend.record_event(event, torch.cuda.current_stream())
    event.synchronize()
    handle = backend.export_event(event, torch.device("cuda:0"))
    case = {"after_export": shm_snapshot(handle), "imports": []}
    results["cases"][label] = case
    save()
    del event
    gc.collect()
    case["after_producer_delete"] = shm_snapshot(handle)
    save()
    imports = case["imports"]
    for attempt in range(2 if label == "patched" else 1):
        try:
            child = subprocess.run(
                [sys.executable, __file__, "--import", handle.hex()],
                capture_output=True,
                text=True,
                timeout=25,
                check=False,
            )
            imports.append(
                {
                    "attempt": attempt + 1,
                    "returncode": child.returncode,
                    "stdout": child.stdout[-4000:],
                    "stderr": child.stderr[-4000:],
                }
            )
        except subprocess.TimeoutExpired:
            imports.append({"attempt": attempt + 1, "timeout": True})
        imports[-1]["after_child_exit"] = shm_snapshot(handle)
        save()
    del backend
    gc.collect()
    case["after_backend_delete"] = shm_snapshot(handle)
    save()
print(json.dumps(results, indent=2), flush=True)
patched = results["cases"]["patched"]
assert patched["after_export"] == patched["after_producer_delete"], (
    "retained export lost its shared memory"
)
assert patched["imports"][0].get("returncode") == 0, (
    "retained export was not importable"
)

"""Capture the binary passed to HIP during Mooncake experiment startup."""
import functools
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import traceback


def trace_warmup(function):
    @functools.wraps(function)
    def traced(device):
        directory = os.environ.get("MOONCAKE_KERNEL_EVIDENCE_DIR")
        if not directory:
            return function(device)
        import torch
        import triton

        root = Path(directory) / f"{socket.gethostname()}-{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        record = {
            "function": function.__name__, "device": str(device),
            "hostname": socket.gethostname(), "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "kernel": platform.release(), "python": platform.python_version(),
            "torch": torch.__version__, "hip": torch.version.hip,
            "triton": triton.__version__, "triton_file": triton.__file__,
            "target": str(triton.runtime.driver.active.get_current_target()),
            "gpu": str(torch.cuda.get_device_properties(device)),
            "gpu_arch": torch.cuda.get_device_properties(device).gcnArchName,
            "memory_before": list(torch.cuda.mem_get_info(device)),
            "environment": {
                key: os.environ[key] for key in (
                    "TRITON_CACHE_DIR", "TRITON_LIBHIP_PATH", "TRITON_OVERRIDE_ARCH",
                    "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                    "HSA_OVERRIDE_GFX_VERSION", "HSA_NO_SCRATCH_RECLAIM",
                    "HSA_ENABLE_IPC_MODE_LEGACY", "HIP_FORCE_DEV_KERNARG",
                ) if key in os.environ
            },
            "loads": [],
        }
        utils = triton.runtime.driver.active.utils
        original = utils.load_binary

        def capture(name, binary, shared, index):
            if name != "batch_memcpy_kernel":
                return original(name, binary, shared, index)
            number = len(record["loads"])
            path = root / f"load-{number}.hsaco"
            path.write_bytes(binary)
            item = {
                "name": name, "bytes": len(binary), "device_index": index,
                "shared_bytes": shared,
                "sha256": hashlib.sha256(binary).hexdigest(),
            }
            record["loads"].append(item)
            try:
                result = original(name, binary, shared, index)
            except Exception as error:
                item["error"] = repr(error)
                raise
            item["loaded"] = True
            return result

        utils.load_binary = capture
        try:
            result = function(device)
            torch.cuda.synchronize(device)
            record["status"] = "passed"
            return result
        except Exception as error:
            record.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            utils.load_binary = original
            try:
                record["memory_after"] = list(torch.cuda.mem_get_info(device))
                paths = sorted({
                    line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if any(name in line for name in (
                        "libamdhip64", "libhsa-runtime64", "libtriton", "hip_utils",
                    ))
                })
                record["libraries"] = []
                for name in paths:
                    path = Path(name)
                    item = {"path": name, "bytes": path.stat().st_size}
                    if "libtriton" not in name:
                        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                    record["libraries"].append(item)
                cache = os.environ.get("TRITON_CACHE_DIR")
                if cache:
                    for path in Path(cache).rglob("batch_memcpy_kernel.*"):
                        target = root / "cache" / path.parent.name / path.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
            except Exception as error:
                record["collection_error"] = repr(error)
            try:
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                (root / "result.json").write_text(json.dumps(record, indent=2) + "\n")
                print("[mooncake-kernel-evidence] " + json.dumps({
                    "path": str(root), "status": record.get("status"),
                    "loads": record["loads"],
                }), flush=True)
            except Exception as error:
                print(f"[mooncake-kernel-evidence] Cannot save report: {error}", flush=True)
    return traced


@trace_warmup
def check_copy(device):
    import torch
    from vllm.v1.worker.mamba_utils import batch_memcpy

    source = torch.arange(1024, dtype=torch.int32, device=device).to(torch.uint8)
    destination = torch.zeros_like(source)
    batch_memcpy(
        torch.tensor([source.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([destination.data_ptr()], dtype=torch.uint64, device=device),
        torch.tensor([source.numel()], dtype=torch.int32, device=device),
    )
    torch.cuda.synchronize(device)
    assert torch.equal(source, destination), "batch_memcpy returned different bytes"


if __name__ == "__main__":
    import sys
    import torch

    device = torch.device("cuda", int(sys.argv[1]))
    torch.cuda.set_device(device)
    check_copy(device)

"""Verify this experiment's exact Python overlay and unmodified compute kernel."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path("/usr/local/lib/python3.12/dist-packages")
manifest = json.loads(Path("/opt/atomesh-mori/manifest.json").read_text())
actual = {}
for name, expected in manifest["installed_files"].items():
    actual[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    assert actual[name] == expected, (name, actual[name], expected)

from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_connector import (
    MoRIIOConnector,
)
from vllm.distributed.kv_transfer.kv_connector.v1.moriio.moriio_common import (
    MoRIIOConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector
from vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector import (
    SimpleCPUOffloadConnector,
)

assert hasattr(MoRIIOConfig, "later_stage_notify_ports")
print(
    json.dumps(
        {
            "hashes": actual,
            "vllm": importlib.metadata.version("vllm"),
            "amd_mori": importlib.metadata.version("amd_mori"),
            "imports": [
                c.__name__
                for c in (MoRIIOConnector, MultiConnector, SimpleCPUOffloadConnector)
            ],
        }
    )
)

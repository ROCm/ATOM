"""Verify the installed rebuilt TE without inspecting its C++ object layout."""

import hashlib
import json
from pathlib import Path

import mooncake.engine

expected = json.loads(Path("/opt/atomesh-timeout/expected.json").read_text())
site = Path("/usr/local/lib/python3.12/dist-packages")
engine = Path(mooncake.engine.__file__)
connector = (
    site / "vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
)
native = site / "vllm/model_executor/kernels/linear/mxfp8/rocm_native.py"
actual = {
    "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
    "connector_sha256": hashlib.sha256(connector.read_bytes()).hexdigest(),
    "module_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
}
assert actual["engine_sha256"] == expected["engine_sha256"]
assert actual["connector_sha256"] == expected["instrumented_connector_sha256"]
assert actual["module_sha256"] == expected["base_module_sha256"]
assert hasattr(mooncake.engine, "TransferEngine")
assert not Path("/opt/atomesh-timeout/libmooncake_timeout_probe.so").exists()
print(json.dumps(dict(actual, engine_path=str(engine), native_layout_probe=False)))

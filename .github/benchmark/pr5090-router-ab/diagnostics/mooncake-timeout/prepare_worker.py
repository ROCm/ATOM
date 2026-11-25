"""Build-time exact-image guard and narrowly scoped connector instrumentation."""

import ast
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ENGINE_SHA = "540428ce937a5be29726195730246f053f88d3ddfabaa2af33ad79c7a92198a8"
CONNECTOR_SHA = "56a614c7d4416cd921937354782431f9ccc7df11511adb910b2ddf044d1d8c09"
MODULE = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"
)


def instrument(original):
    if hashlib.sha256(original.encode()).hexdigest() != CONNECTOR_SHA:
        raise ValueError("Unexpected connector source; do not patch a different worker")
    replacements = {
        "            self.xfer_stats.record_failed_transfer()\n": "            from atomesh_timeout_probe import record_failure\n"
        "            record_failure(self, remote_session, src_ptrs, dst_ptrs, lengths, ret_value, duration)\n"
        "            self.xfer_stats.record_failed_transfer()\n",
        "        self.device_kv_caches = kv_caches\n": "        from atomesh_timeout_probe import record_registration\n"
        "        record_registration(self, 'register')\n"
        "        self.device_kv_caches = kv_caches\n",
        "        self._kv_unregistered = True\n": "        self._kv_unregistered = True\n"
        "        from atomesh_timeout_probe import record_registration\n"
        "        record_registration(self, 'unregister')\n",
        "        self._kv_unregistered = False\n\n    async def fetch_finished_recving_reqs": "        self._kv_unregistered = False\n"
        "        from atomesh_timeout_probe import record_registration\n"
        "        record_registration(self, 'restore')\n\n    async def fetch_finished_recving_reqs",
    }
    for old, new in replacements.items():
        if original.count(old) != 1:
            raise ValueError(f"Expected exactly one instrumentation seam: {old!r}")
        original = original.replace(old, new)
    ast.parse(original)
    return original


def engine_file():
    dist = importlib.metadata.distribution("mooncake-transfer-engine-rocm")
    found = [
        dist.locate_file(f)
        for f in dist.files
        if str(f).endswith(".so") and Path(f).name.startswith("engine")
    ]
    if len(found) != 1:
        raise ValueError(f"Expected one installed engine, found {found}")
    return found[0]


def main():
    engine = engine_file()
    actual = hashlib.sha256(engine.read_bytes()).hexdigest()
    if actual != ENGINE_SHA:
        raise ValueError(
            "Unexpected native engine ELF; diagnostic offsets do not apply"
        )
    if "--verify" not in sys.argv:
        MODULE.write_text(instrument(MODULE.read_text()), encoding="utf-8")
    print(
        json.dumps(
            {
                "engine_path": str(engine),
                "engine_sha256": actual,
                "connector_sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
                "native_engine_unchanged": True,
            }
        )
    )


if __name__ == "__main__":
    main()

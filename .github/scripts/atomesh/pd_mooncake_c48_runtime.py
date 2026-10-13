"""Prepare a source-verified Mooncake overlay on the pinned serving image."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    vendor = Path(__file__).parent / "mooncake-c48"
    provenance = json.loads((vendor / "provenance.json").read_text())
    version = importlib.metadata.version("vllm")
    assert "81198e97b" in version, version
    package = Path(importlib.util.find_spec("vllm").origin).parent
    before = {}
    for relative, expected in provenance["expected_image_source_hashes"].items():
        before[relative] = digest(package / relative)
        assert before[relative] == expected, (relative, before[relative], expected)
    site = Path("/tmp/mooncake-c48-site")
    site.mkdir(exist_ok=True)
    target = site / "vllm"
    # Reuse every image file, including compiled extensions. Replace one Python
    # file with the already validated stats-only child compatibility fix.
    shutil.copytree(package, target, copy_function=os.symlink)
    multi = target / "distributed/kv_transfer/kv_connector/v1/multi_connector.py"
    multi.unlink()
    shutil.copyfile(vendor / "multi_connector.py", multi)
    assert digest(multi) == provenance["multi_connector_after_sha256"]
    warmup = target / "model_executor/warmup/mamba_triton_warmup.py"
    assert digest(warmup) == "63cc5af797ecc41384e7a60a1f387a1bbae400f7128c71c27be6b29adbac021c"
    original = warmup.read_text()
    warmup.unlink()
    warmup.write_text(original + "\nfrom vllm._mooncake_kernel_evidence import trace_warmup\n"
                      "_warm_batch_memcpy_kernel = trace_warmup(_warm_batch_memcpy_kernel)\n")
    shutil.copyfile(vendor / "kernel_evidence.py", target / "_mooncake_kernel_evidence.py")
    shutil.copyfile(vendor / "startup_state.py", target / "_mooncake_startup_state.py")
    stages = {
        "v1/worker/gpu_worker.py": (
            "7cb04b35cfaeff7cb2fa8dfc716a8d02dcf5b345af0b438ea8e41f9a8f9cee22",
            [f"Worker.{name}" for name in (
                "load_model", "determine_available_memory", "initialize_from_config",
                "compile_or_warm_up_model",
            )],
        ),
        "distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py": (
            "c0d13a4cb6aebc9b558cacf624c731e6f692775ce05a73a6d3a93c6a2665b380",
            ["MooncakeConnectorWorker.__init__", "MooncakeConnectorWorker.register_kv_caches"],
        ),
        "distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py": (
            "a8fdf8179b8518510d6be7be12c0ba066aed7d6aa0e11a371d02880ce772a4f0",
            ["MooncakeStoreWorker.__init__", "MooncakeStoreWorker.register_kv_caches"],
        ),
        "model_executor/warmup/kernel_warmup.py": (
            "c99b79e748aa01e9b8d84caaab71cfd53955c0510dda1bacaaa75fcd22ea1cfc",
            ["kernel_warmup"],
        ),
    }
    stage_sources = {}
    for relative, (expected, names) in stages.items():
        path = target / relative
        assert digest(path) == expected, relative
        source = path.read_text()
        if relative.endswith("kernel_warmup.py"):
            assert source.count("            registry.warmup()") == 1
            source = source.replace(
                "            registry.warmup()",
                '            _trace_stage("jit_registry")(registry.warmup)()',
            )
        source += "\nfrom vllm._mooncake_startup_state import trace_stage as _trace_stage\n"
        for name in names:
            source += f"{name} = _trace_stage({name!r})({name})\n"
        path.unlink()
        path.write_text(source)
        stage_sources[relative] = {"before": expected, "after": digest(path)}
    versions = {}
    for name in ("vllm", "torch", "triton", "amd-aiter", "amd_mori", "mooncake-transfer-engine-rocm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "python": sys.executable, "image_package": str(package),
        "overlay_package": str(target), "versions": versions,
        "verified_image_source_hashes": before,
        "multi_connector_after_sha256": digest(multi),
        "warmup_observer_sha256": digest(warmup),
        "kernel_evidence_sha256": digest(target / "_mooncake_kernel_evidence.py"),
        "startup_state_sha256": digest(target / "_mooncake_startup_state.py"),
        "startup_stage_sources": stage_sources,
        "provenance": provenance,
        "compiled_extensions": [
            {"path": str(p), "resolved": str(p.resolve()), "bytes": p.stat().st_size}
            for p in target.glob("*.so")
        ],
    }
    Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()

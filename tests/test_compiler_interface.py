# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

from atom.utils import compilation_counter
from atom.utils.compiler_inferface import _save_standalone_compiled_graph


def test_non_saveable_torch_213_artifact_is_skipped():
    class NonSaveableArtifact:
        def is_saveable(self):
            return False

        def save(self, **kwargs):
            pytest.fail("save must not run when is_saveable() is false")

    assert (
        _save_standalone_compiled_graph(
            NonSaveableArtifact(), "/tmp/not-used", "subgraph"
        )
        is None
    )


def test_legacy_no_aot_runtime_error_is_skipped():
    class LegacyArtifact:
        def save(self, **kwargs):
            raise RuntimeError(
                "CompiledArtifact.save failed to save due to no "
                "aot_autograd artifacts"
            )

    assert (
        _save_standalone_compiled_graph(LegacyArtifact(), "/tmp/not-used", "subgraph")
        is None
    )


def test_unexpected_save_runtime_error_is_not_swallowed():
    class BrokenArtifact:
        def save(self, **kwargs):
            raise RuntimeError("permission denied")

    with pytest.raises(RuntimeError, match="permission denied"):
        _save_standalone_compiled_graph(BrokenArtifact(), "/tmp/not-used", "subgraph")


def test_saveable_artifact_returns_cache_handle(monkeypatch):
    saved = {}

    class SaveableArtifact:
        def is_saveable(self):
            return True

        def save(self, **kwargs):
            saved.update(kwargs)

    monkeypatch.setattr(compilation_counter, "num_compiled_artifacts_saved", 0)

    handle = _save_standalone_compiled_graph(
        SaveableArtifact(), "/tmp/artifact", "subgraph"
    )

    assert handle == ("subgraph", "/tmp/artifact")
    assert saved == {"path": "/tmp/artifact", "format": "unpacked"}
    assert compilation_counter.num_compiled_artifacts_saved == 1


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
@pytest.mark.parametrize("prefix", ["", "backbone", "draft-head", "backbone-abc123"])
def test_lower_level_cache_dirs_persist(tmp_path, monkeypatch, adapter_name, prefix):
    import os
    from pathlib import Path

    from atom.utils import compiler_inferface as ci

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/previous/inductor")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/previous/triton")
    rank_dir = tmp_path / "rank_0"
    cache_dir = rank_dir / prefix if prefix else rank_dir
    adaptor = getattr(ci, adapter_name)()
    adaptor.initialize_cache(str(cache_dir), prefix=prefix)
    assert adaptor.cache_dir == str(cache_dir)
    assert adaptor.prefix == prefix
    assert Path(adaptor.base_cache_dir) == rank_dir
    assert Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]) == rank_dir / "inductor_cache"
    assert Path(os.environ["TRITON_CACHE_DIR"]) == rank_dir / "triton_cache"
    assert (rank_dir / "inductor_cache").is_dir()
    assert (rank_dir / "triton_cache").is_dir()

    sentinel = rank_dir / "triton_cache" / "existing-artifact"
    sentinel.write_text("keep")
    adaptor.initialize_cache(str(cache_dir), prefix=prefix)
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
@pytest.mark.parametrize("preset", [False, True])
def test_disabled_cache_keeps_environment(tmp_path, monkeypatch, adapter_name, preset):
    import os

    from atom.utils import compiler_inferface as ci

    variables = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    for name in variables:
        if preset:
            monkeypatch.setenv(name, "/caller-owned/" + name)
        else:
            monkeypatch.delenv(name, raising=False)
    before = {name: os.environ.get(name) for name in variables}
    path = tmp_path / "unused" / "backbone"
    adaptor = getattr(ci, adapter_name)()
    adaptor.initialize_cache(str(path), disable_cache=True, prefix="backbone")
    assert adaptor.cache_dir == str(path)
    assert adaptor.prefix == "backbone"
    assert adaptor.base_cache_dir.rstrip("/") == str(path.parent)
    assert {name: os.environ.get(name) for name in variables} == before
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
def test_rank_isolation_and_graph_prefix_sharing(tmp_path, monkeypatch, adapter_name):
    import os
    from pathlib import Path

    from atom.utils import compiler_inferface as ci

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    adaptor = getattr(ci, adapter_name)()
    paths = []
    for rank in range(2):
        for prefix in ("backbone", "eagle_head"):
            adaptor.initialize_cache(
                str(tmp_path / f"rank_{rank}" / prefix), prefix=prefix
            )
            paths.append(Path(os.environ["TRITON_CACHE_DIR"]))
    assert paths[0] == paths[1]
    assert paths[2] == paths[3]
    assert paths[0] != paths[2]


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
def test_cache_directory_error_is_not_silenced(tmp_path, monkeypatch, adapter_name):
    from atom.utils import compiler_inferface as ci

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    (tmp_path / "inductor_cache").write_text("not a directory")
    with pytest.raises(FileExistsError):
        getattr(ci, adapter_name)().initialize_cache(str(tmp_path))


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
def test_triton_cache_reused_across_processes(tmp_path, adapter_name):
    """Exercise the real cache manager after adaptor initialization; no GPU."""
    import importlib.util
    import json
    import os
    import subprocess
    import sys

    if importlib.util.find_spec("triton") is None:
        pytest.skip("requires Triton cache manager")
    script = """
import json, sys
from pathlib import Path
from atom.utils import compiler_inferface as ci
from triton.runtime.cache import get_cache_manager
root = Path(sys.argv[1])
getattr(ci, sys.argv[2])().initialize_cache(str(root / "backbone"), prefix="backbone")
cache = get_cache_manager("a" * 64)
assert Path(cache.cache_dir).is_relative_to(root / "triton_cache")
path = cache.get_file("adaptor-persistence.txt")
if sys.argv[3] == "write":
    assert path is None
    path = cache.put("persist across processes", "adaptor-persistence.txt", binary=False)
else:
    assert path is not None
    assert Path(path).read_text() == "persist across processes"
print(json.dumps({"path": path, "mtime": Path(path).stat().st_mtime_ns}))
"""
    env = os.environ.copy()
    for name in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        env.pop(name, None)
    outputs = []
    for mode in ("write", "read"):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path / "rank_0"),
                adapter_name,
                mode,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
def test_compiled_triton_kernel_reused_in_new_process(tmp_path, adapter_name):
    """Real kernel artifact reuse; the second process must not recompile."""
    import importlib.util
    import json
    import os
    import subprocess
    import sys

    import torch

    if not torch.cuda.is_available() or importlib.util.find_spec("triton") is None:
        pytest.skip("requires a GPU and Triton")
    from triton import knobs

    if not hasattr(knobs.compilation, "listener"):
        pytest.skip("requires Triton's compilation listener")

    script = tmp_path / "cache_worker.py"
    script.write_text("""
import json, os, sys
from pathlib import Path
import torch
import triton
import triton.language as tl
from triton import knobs
from atom.utils import compiler_inferface as ci

@triton.jit
def add_one(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(X + offsets, offsets < N, 0)
    tl.store(Y + offsets, values + 1, offsets < N)

root = Path(sys.argv[1])
getattr(ci, sys.argv[2])().initialize_cache(str(root / "backbone"), prefix="backbone")
events = []
def listener(**kw):
    events.append({
        "name": kw["src"].name,
        "cache_hit": kw["cache_hit"],
        "files": kw["metadata_group"],
    })
knobs.compilation.listener = listener
x = torch.arange(513, device="cuda", dtype=torch.float32) + int(sys.argv[3])
y = torch.empty_like(x)
add_one[(triton.cdiv(x.numel(), 256),)](x, y, x.numel(), 256)
torch.testing.assert_close(y, x + 1, rtol=0, atol=0)
assert len(events) == 1
assert events[0]["cache_hit"] == (sys.argv[3] == "1")
for value in events[0]["files"].values():
    assert Path(value).is_relative_to(root / "triton_cache")
print(json.dumps(events[0]))
""")
    env = os.environ.copy()
    for name in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_ALWAYS_COMPILE",
    ):
        env.pop(name, None)
    outputs = []
    for process in range(2):
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                str(tmp_path / "rank_0"),
                adapter_name,
                str(process),
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
            check=True,
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert outputs[0]["files"] == outputs[1]["files"]


def test_standalone_inductor_artifacts_reused_in_new_process(tmp_path):
    """Real CPU standalone codegen and disk-cache reuse, not env-write mocking."""
    import hashlib
    import json
    import os
    import shutil
    import subprocess
    import sys

    import torch._inductor

    if not hasattr(torch._inductor, "standalone_compile"):
        pytest.skip("requires torch._inductor.standalone_compile (Torch 2.8+)")
    if shutil.which("c++") is None and shutil.which("g++") is None:
        pytest.skip("requires a C++ compiler for CPU Inductor codegen")

    script = tmp_path / "standalone_worker.py"
    script.write_text("""
import hashlib, json, sys
from pathlib import Path
import torch
from torch import fx
from torch._dynamo.utils import counters
from torch._inductor import standalone_compile
from torch._inductor.runtime.runtime_utils import cache_dir
from atom.utils.compiler_inferface import InductorStandaloneAdaptor

root = Path(sys.argv[1])
InductorStandaloneAdaptor().initialize_cache(str(root / "backbone"), prefix="backbone")
assert Path(cache_dir()) == root / "inductor_cache"
def function(x, y):
    return (torch.sin(x) + y) * 2
graph = fx.symbolic_trace(function)
x = torch.arange(4096, dtype=torch.float32) / 1024
y = torch.linspace(-1, 1, 4096)
compiled = standalone_compile(
    graph, [x, y], dynamic_shapes="from_example_inputs",
    options={"config_patches": {"compile_threads": 1, "fx_graph_cache": True}},
)
# Changing runtime values must not require another specialization.
torch.testing.assert_close(compiled(x, y), function(x, y))
torch.testing.assert_close(compiled(x.neg(), y), function(x.neg(), y))
files = {}
for p in Path(cache_dir()).rglob("*"):
    if p.is_file() and p.suffix in (".py", ".cpp", ".so"):
        files[str(p.relative_to(cache_dir()))] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "mtime_ns": p.stat().st_mtime_ns,
        }
print(json.dumps({
    "files": files,
    "inductor": dict(counters["inductor"]),
    "aot_autograd": dict(counters["aot_autograd"]),
}))
""")
    env = os.environ.copy()
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES",
    ):
        env.pop(name, None)
    outputs = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(script), str(tmp_path / "rank_0")],
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
            check=True,
        )
        outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
    first, second = outputs
    assert first["inductor"].get("fxgraph_cache_miss", 0) >= 1
    assert second["inductor"].get("fxgraph_cache_hit", 0) >= 1
    assert second["inductor"].get("fxgraph_cache_miss", 0) == 0
    assert second["aot_autograd"].get("autograd_cache_hit", 0) >= 1
    assert first["files"]
    assert first["files"] == second["files"]
    # Verify the serialized manifest itself has stable content.
    assert hashlib.sha256(
        json.dumps(first["files"], sort_keys=True).encode()
    ).digest() == (
        hashlib.sha256(json.dumps(second["files"], sort_keys=True).encode()).digest()
    )


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
def test_initialization_redirects_only_future_cache_work(
    tmp_path, monkeypatch, caplog, adapter_name
):
    import importlib.util
    import logging
    import os
    from pathlib import Path

    if importlib.util.find_spec("triton") is None:
        pytest.skip("requires Triton cache manager")
    from triton.runtime.cache import get_cache_manager

    from atom.utils import compiler_inferface as ci

    early = tmp_path / "early_cache"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(early))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "early_inductor"))
    before = get_cache_manager("b" * 64)
    path = Path(before.put("early artifact", "early.txt", binary=False))
    first_stat = path.stat().st_mtime_ns
    rank = tmp_path / "rank_0"
    with caplog.at_level(logging.INFO, logger="atom"):
        getattr(ci, adapter_name)().initialize_cache(
            str(rank / "backbone"), prefix="backbone"
        )
    assert str(rank / "inductor_cache") in caplog.text
    assert str(rank / "triton_cache") in caplog.text
    assert os.environ["TRITON_CACHE_DIR"] == str(rank / "triton_cache")
    after = get_cache_manager("b" * 64)
    assert Path(after.cache_dir).is_relative_to(rank / "triton_cache")
    assert after.get_file("early.txt") is None
    # Initialization redirects subsequent work; it does not migrate/delete
    # kernels compiled before the first backend initialization.
    assert path.read_text() == "early artifact"
    assert path.stat().st_mtime_ns == first_stat

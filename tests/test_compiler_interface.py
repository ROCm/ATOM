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

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
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
    "adapter_name,external_inductor",
    [
        ("InductorAdaptor", False),
        ("InductorStandaloneAdaptor", False),
        ("InductorStandaloneAdaptor", True),
    ],
)
def test_compiled_triton_kernel_reused_in_new_process(
    tmp_path, adapter_name, external_inductor
):
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
external_inductor = sys.argv[4] == "1"
if external_inductor:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(root / "operator_inductor")
expected_triton = root / "operator_inductor/triton" if external_inductor else root / "triton_cache"
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
    assert Path(value).is_relative_to(expected_triton)
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
                str(int(external_inductor)),
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
    from torch._inductor.runtime import runtime_utils
    from triton.runtime.cache import get_cache_manager

    from atom.utils import compiler_inferface as ci

    temporary_root = tmp_path / "torchinductor_test"
    monkeypatch.setattr(runtime_utils, "default_cache_dir", lambda: str(temporary_root))
    early = temporary_root / "triton" / "0"
    monkeypatch.setenv("TRITON_CACHE_DIR", str(early))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(temporary_root))
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


@pytest.fixture(autouse=True)
def _isolate_compiler_cache_ownership(monkeypatch):
    import os

    from atom.utils import compiler_inferface as ci

    monkeypatch.setattr(ci, "_managed_compiler_cache_dirs", {})
    for name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        # Register even an absent key for restoration. delenv(raising=False)
        # alone would not record anything if the variable was initially absent,
        # so direct assignments by the adaptor would otherwise leak.
        previous = os.environ.get(name)
        monkeypatch.setenv(name, previous if previous is not None else "")
        if previous is None:
            monkeypatch.delenv(name)


@pytest.mark.parametrize(
    "adapter_name", ["InductorAdaptor", "InductorStandaloneAdaptor"]
)
def test_external_paths_and_regular_compatibility(
    tmp_path, monkeypatch, caplog, adapter_name
):
    import logging
    import os

    from atom.utils import compiler_inferface as ci

    external = {
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "operator-inductor"),
        "TRITON_CACHE_DIR": str(tmp_path / "operator-triton"),
    }
    for name, path in external.items():
        monkeypatch.setenv(name, path)
    root = tmp_path / "atom" / "rank_0"
    with caplog.at_level(logging.INFO, logger="atom"):
        getattr(ci, adapter_name)().initialize_cache(
            str(root / "backbone"), prefix="backbone"
        )
    if adapter_name == "InductorStandaloneAdaptor":
        assert {k: os.environ[k] for k in external} == external
        assert not root.exists()
        assert all(path in caplog.text for path in external.values())
    else:
        assert os.environ["TRITON_CACHE_DIR"] == str(root / "triton_cache")
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor_cache")


@pytest.mark.parametrize("default_suffix", ["", "0", "2/child"])
def test_standalone_redirects_auto_defaults(tmp_path, monkeypatch, default_suffix):
    import os

    from torch._inductor.runtime import runtime_utils

    from atom.utils import compiler_inferface as ci

    default = tmp_path / "torchinductor_default"
    monkeypatch.setattr(runtime_utils, "default_cache_dir", lambda: str(default))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(default))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(default / "triton" / default_suffix))
    root = tmp_path / "atom"
    ci.InductorStandaloneAdaptor().initialize_cache(str(root))
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(root / "inductor_cache")
    assert os.environ["TRITON_CACHE_DIR"] == str(root / "triton_cache")


def test_standalone_managed_paths_follow_new_owner_but_not_user_changes(
    tmp_path, monkeypatch
):
    import os

    from atom.utils import compiler_inferface as ci

    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        monkeypatch.delenv(key, raising=False)
    adaptor = ci.InductorStandaloneAdaptor()
    adaptor.initialize_cache(str(tmp_path / "rank_0"))
    adaptor.initialize_cache(str(tmp_path / "rank_1"))
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "rank_1/triton_cache")
    user = str(tmp_path / "explicit-triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", user)
    adaptor.initialize_cache(str(tmp_path / "rank_2"))
    assert os.environ["TRITON_CACHE_DIR"] == user
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "rank_2/inductor_cache"
    )


def test_standalone_does_not_match_similarly_named_temp_path(tmp_path, monkeypatch):
    import os

    from torch._inductor.runtime import runtime_utils

    from atom.utils import compiler_inferface as ci

    default = tmp_path / "torchinductor_default"
    monkeypatch.setattr(runtime_utils, "default_cache_dir", lambda: str(default))
    custom = str(default / "triton-custom")
    monkeypatch.setenv("TRITON_CACHE_DIR", custom)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
    ci.InductorStandaloneAdaptor().initialize_cache(str(tmp_path / "atom"))
    assert os.environ["TRITON_CACHE_DIR"] == custom


@pytest.mark.parametrize("initial", [None, "", "/caller-cache"])
@pytest.mark.parametrize("mutation", ["assign", "delete"])
def test_cache_fixture_restores_environment(monkeypatch, initial, mutation):
    import os

    names = ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    for name in names:
        if initial is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, initial)
    before = {name: os.environ.get(name) for name in names}
    with pytest.MonkeyPatch.context() as isolated:
        _isolate_compiler_cache_ownership.__wrapped__(isolated)
        assert {name: os.environ.get(name) for name in names} == before
        for name in names:
            if mutation == "assign":
                os.environ[name] = "/set-directly-by-adaptor"
            else:
                os.environ.pop(name, None)
    assert {name: os.environ.get(name) for name in names} == before


@pytest.mark.parametrize("triton_origin", ["unset", "default", "managed", "explicit"])
def test_external_inductor_owns_automatic_triton(tmp_path, monkeypatch, triton_origin):
    import os

    from torch._inductor.runtime import runtime_utils

    from atom.utils import compiler_inferface as ci

    default = tmp_path / "torchinductor_default"
    monkeypatch.setattr(runtime_utils, "default_cache_dir", lambda: str(default))
    external = tmp_path / "mounted_inductor"
    explicit = tmp_path / "explicit_triton"
    old_managed = tmp_path / "prior_atom/triton_cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(external))
    if triton_origin == "unset":
        monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    elif triton_origin == "default":
        monkeypatch.setenv("TRITON_CACHE_DIR", str(default / "triton/0"))
    elif triton_origin == "managed":
        monkeypatch.setenv("TRITON_CACHE_DIR", str(old_managed))
        ci._managed_compiler_cache_dirs["TRITON_CACHE_DIR"] = str(old_managed)
    else:
        monkeypatch.setenv("TRITON_CACHE_DIR", str(explicit))
    expected = explicit if triton_origin == "explicit" else external / "triton"
    adaptor = ci.InductorStandaloneAdaptor()
    for rank in ("rank_0", "rank_1"):
        adaptor.initialize_cache(str(tmp_path / "atom" / rank))
        assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(external)
        assert os.environ["TRITON_CACHE_DIR"] == str(expected)
    if triton_origin != "explicit":
        assert expected.is_dir()
        assert not (tmp_path / "atom").exists()


def test_default_root_snapshot_precedes_environment_mutation(tmp_path, monkeypatch):
    import os

    from torch._inductor.runtime import runtime_utils

    from atom.utils import compiler_inferface as ci

    default = tmp_path / "torchinductor_default"
    original = {
        "TORCHINDUCTOR_CACHE_DIR": str(default),
        "TRITON_CACHE_DIR": str(default / "triton/0"),
    }
    for name, value in original.items():
        monkeypatch.setenv(name, value)
    observed = []

    def default_root():
        observed.append({k: os.environ.get(k) for k in original})
        return str(default)

    monkeypatch.setattr(runtime_utils, "default_cache_dir", default_root)
    ci.InductorStandaloneAdaptor().initialize_cache(str(tmp_path / "atom"))
    assert observed == [original]


def test_inherited_triton_follows_external_owner_change(tmp_path, monkeypatch):
    import os

    from atom.utils import compiler_inferface as ci

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "external_a"))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    adaptor = ci.InductorStandaloneAdaptor()
    adaptor.initialize_cache(str(tmp_path / "atom_a"))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "external_b"))
    adaptor.initialize_cache(str(tmp_path / "atom_b"))
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "external_b/triton")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "operator_override"))
    adaptor.initialize_cache(str(tmp_path / "atom_c"))
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "operator_override")


def test_empty_external_inductor_is_not_a_triton_parent(tmp_path, monkeypatch):
    import os

    from atom.utils import compiler_inferface as ci

    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "")
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    ci.InductorStandaloneAdaptor().initialize_cache(str(tmp_path / "atom"))
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == ""
    assert os.environ["TRITON_CACHE_DIR"] == str(tmp_path / "atom/triton_cache")


def test_external_triton_creation_error_is_not_silenced(tmp_path, monkeypatch):
    from atom.utils import compiler_inferface as ci

    external = tmp_path / "external"
    external.mkdir()
    (external / "triton").write_text("not a directory")
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(external))
    monkeypatch.delenv("TRITON_CACHE_DIR", raising=False)
    with pytest.raises(FileExistsError):
        ci.InductorStandaloneAdaptor().initialize_cache(str(tmp_path / "atom"))

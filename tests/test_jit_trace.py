"""Kernel-compile accounting: hooks the cold paths, never the hot one."""

import json
import sys
import types

import pytest

from atom.utils import jit_trace


@pytest.fixture(autouse=True)
def _uninstalled():
    """Each test installs its own tracer into its own fake packages."""
    jit_trace.close()
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in ("flydsl", "aiter")}
    for key in saved:
        del sys.modules[key]
    yield
    jit_trace.close()
    for key in [k for k in sys.modules if k.split(".")[0] in ("flydsl", "aiter")]:
        del sys.modules[key]
    sys.modules.update(saved)


def fake_flydsl(compile_result="lowered", cache_value=None, raises=None):
    jit_function = types.ModuleType("flydsl.compiler.jit_function")

    class MlirCompiler:
        calls = []

        @classmethod
        def compile(cls, module, *, arch="", func_name="", link_libs=None):
            cls.calls.append((module, arch, func_name))
            if raises is not None:
                raise raises
            return compile_result

    class JitCacheManager:
        cache_dir = "/cache/flydsl"

        def get(self, cache_key):
            return cache_value

    jit_function.MlirCompiler = MlirCompiler
    jit_function.JitCacheManager = JitCacheManager

    compiler_pkg = types.ModuleType("flydsl.compiler")
    compiler_pkg.jit_function = jit_function
    root = types.ModuleType("flydsl")
    root.compiler = compiler_pkg
    for name, mod in (
        ("flydsl", root),
        ("flydsl.compiler", compiler_pkg),
        ("flydsl.compiler.jit_function", jit_function),
    ):
        sys.modules[name] = mod
    return jit_function


def fake_aiter(result="module"):
    core = types.ModuleType("aiter.jit.core")
    core.build_module = lambda md_name, *a, **k: result
    jit_pkg = types.ModuleType("aiter.jit")
    jit_pkg.core = core
    root = types.ModuleType("aiter")
    root.jit = jit_pkg
    for name, mod in (("aiter", root), ("aiter.jit", jit_pkg), ("aiter.jit.core", core)):
        sys.modules[name] = mod
    return core


def events(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_compile_is_timed_and_the_result_passes_through(tmp_path):
    jit_function = fake_flydsl()
    fake_aiter()
    path = tmp_path / "jit.jsonl"
    assert set(jit_trace.install(str(path))) == {"flydsl", "aiter"}

    out = jit_function.MlirCompiler.compile(
        "mod", arch="gfx950", func_name="flydsl_bpreshuffle_8w"
    )
    assert out == "lowered"
    # The hook must not eat the arguments the compiler was called with.
    assert jit_function.MlirCompiler.calls == [("mod", "gfx950", "flydsl_bpreshuffle_8w")]

    (record,) = [e for e in events(path) if e["event"] == "flydsl_compile"]
    assert record["name"] == "flydsl_bpreshuffle_8w"
    assert record["arch"] == "gfx950"
    assert record["failed"] is False
    assert record["ms"] >= 0


def test_cache_probe_reports_hit_and_miss(tmp_path):
    jit_function = fake_flydsl(cache_value=None)
    path = tmp_path / "jit.jsonl"
    jit_trace.install(str(path))
    assert jit_function.JitCacheManager().get("k" * 200) is None
    (probe,) = [e for e in events(path) if e["event"] == "flydsl_cache_probe"]
    assert probe["hit"] is False
    assert probe["cache_dir"] == "/cache/flydsl"
    # The key is a content hash; only enough of it to pair with a compile.
    assert probe["name"] == "k" * 64


def test_cache_hit_is_recorded_and_the_value_is_returned(tmp_path):
    jit_function = fake_flydsl(cache_value="artifact")
    path = tmp_path / "jit.jsonl"
    jit_trace.install(str(path))
    assert jit_function.JitCacheManager().get("key") == "artifact"
    (probe,) = [e for e in events(path) if e["event"] == "flydsl_cache_probe"]
    assert probe["hit"] is True


def test_aiter_build_is_recorded_by_module_name(tmp_path):
    core = fake_aiter()
    path = tmp_path / "jit.jsonl"
    jit_trace.install(str(path))
    assert core.build_module("module_gemm_a8w8", ["a.cu"], verbose=False) == "module"
    (record,) = [e for e in events(path) if e["event"] == "aiter_build_module"]
    assert record["name"] == "module_gemm_a8w8"
    assert record["failed"] is False


def test_a_failed_compile_is_recorded_and_still_raises(tmp_path):
    jit_function = fake_flydsl(raises=RuntimeError("lowering failed"))
    path = tmp_path / "jit.jsonl"
    jit_trace.install(str(path))
    with pytest.raises(RuntimeError, match="lowering failed"):
        jit_function.MlirCompiler.compile("mod", func_name="k")
    (record,) = [e for e in events(path) if e["event"] == "flydsl_compile"]
    assert record["failed"] is True


def test_a_backend_that_cannot_be_hooked_does_not_take_the_other_down(tmp_path):
    """Telemetry that reaches into another package's internals must degrade.

    A FlyDSL or AITER release that moved these attributes is a missing
    measurement. Refusing to serve over it would not be.
    """
    fake_flydsl()  # aiter deliberately absent
    path = tmp_path / "jit.jsonl"
    assert jit_trace.install(str(path)) == ("flydsl",)


def test_nothing_hooked_leaves_no_file_claiming_an_empty_trace(tmp_path):
    """An empty trace means "nothing compiled"; it must not also mean "no hook"."""
    path = tmp_path / "jit.jsonl"
    assert jit_trace.install(str(path)) == ()
    assert jit_trace._tracer is None


def test_install_is_idempotent(tmp_path):
    jit_function = fake_flydsl()
    path = tmp_path / "jit.jsonl"
    jit_trace.install(str(path))
    assert jit_trace.install(str(tmp_path / "other.jsonl")) == ()
    jit_function.MlirCompiler.compile("mod", func_name="k")
    # One hook, so one record -- not a stack of wrappers each writing its own.
    assert len([e for e in events(path) if e["event"] == "flydsl_compile"]) == 1
    assert not (tmp_path / "other.jsonl").exists()

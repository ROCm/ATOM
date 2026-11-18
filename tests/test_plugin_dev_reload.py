# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dev-mode plugin code reload (``--plugin-dev``): re-import, prepare, rollback.

CPU only, no AITER and no GPU: the plugin is a fake package written under a
temporary ``--plugin-dev-root`` whose entry point goes through the real import
machinery (``importlib.import_module`` of the module the entry point names), so
"reload" here means an actual re-import from disk -- editing the package's
``atom_hook.py`` between calls changes the code the host serves.

What is covered:

- ``plugin_host.prepare_plan`` runs each planned provider's module-level
  ``prepare(device)`` at boot, and a prepare failure refuses the start;
- ``plugin_host.reload_plugins`` re-imports the modules under the dev roots,
  prepares the providers again, installs the new plan and reports
  ``code_digest``/``reloaded_modules``/``prepared``/``native``;
- on an import, registration or prepare failure the previous ``sys.modules``
  objects and host state are restored and the old implementation keeps serving;
- ``plugin_code_digest`` is the documented stable layout (and the payload
  ``code_digest`` is normalized to it);
- dev mode is required (and requires a hot-plug boot plus allowlisted roots);
- the hot-plug payload/aggregation carries ``reload``/``code_digest`` and lifts
  the reload fields to the reply;
- the plugin-round guard refuses a reload before it pauses generation, reloads
  a module or installs a plan;
- the wiring (Config/CLI/env, runner, utility handler, API route) is present.
"""

import ast
import importlib
import json
import os
import queue
import re
import sys
import time
import types
import uuid
from pathlib import Path

import pytest

from atom import plugin_host as host
from atom import plugin_recipe as recipes
from atom.model_engine import plugin_hotplug as hotplug
from atom.model_engine.engine_utility import EngineUtilityHandler

REPO_ROOT = Path(__file__).resolve().parent.parent
if not (REPO_ROOT / "atom").is_dir():
    # Running the file from outside the checkout (a CPU box running this one
    # test on its own): find the tree the imported atom package came from.
    REPO_ROOT = next(
        Path(entry)
        for entry in sys.path
        if (Path(entry) / "atom" / "config.py").is_file()
    )
OUT = "v41.attn.out"
OMIT = object()


# -- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_host(monkeypatch):
    """Default deny, no entry points, AITER left alone, eager graph state."""
    aiter = []
    monkeypatch.setattr(host, "_entry_points", dict)
    monkeypatch.setattr(host, "_configure_aiter", lambda names: aiter.append(names))
    monkeypatch.setattr(host, "_graph_state", lambda: (False, None))
    monkeypatch.setattr(host, "_valid_rows", lambda rows: None)
    host._reset_for_testing()
    yield aiter
    host._reset_for_testing()


class _EntryPoint:
    """The slice of ``importlib.metadata.EntryPoint`` the host uses.

    ``load`` is what the real one does: import the module named by the value and
    get the attribute. That is what makes a re-import read the new file, so the
    fixture must not hand the host a pre-loaded callable.
    """

    def __init__(self, value, module):
        self.value = value
        self.module = module

    def load(self):
        module_name, _, attribute = self.value.partition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attribute)


_DEFAULT_REGISTER = (
    "host.register(\n"
    f'    seam="{OUT}",\n'
    "    contract=1,\n"
    "    impl=_pair,\n"
    "    owner=OWNER,\n"
    ")"
)


def _indent(source: str) -> str:
    return "\n".join("    " + line for line in source.splitlines())


class DevPlugin:
    """A fake plugin package in a temp directory, reachable by its entry point."""

    def __init__(self, base: Path, monkeypatch):
        self.suffix = uuid.uuid4().hex[:10]
        self.package = f"atom_dev_fixture_{self.suffix}"
        self.provider = f"dev-fixture-{self.suffix}"
        self.root = base
        self.package_dir = base / self.package
        self.package_dir.mkdir(parents=True)
        (self.package_dir / "__init__.py").write_text(
            '"""dev fixture package"""\n', encoding="utf-8"
        )
        self.write("v1")
        self.entry_points = {
            self.provider: _EntryPoint(
                f"{self.package}.atom_hook:register", f"{self.package}.atom_hook"
            )
        }
        monkeypatch.setattr(host, "_entry_points", lambda: dict(self.entry_points))
        monkeypatch.syspath_prepend(str(base))

    @property
    def module_name(self) -> str:
        return f"{self.package}.atom_hook"

    def write(
        self,
        version: str,
        *,
        extra: str = "",
        register_source: str | None = None,
        prepare_source: str | None = None,
        close_source: str = "",
        source: str | None = None,
    ) -> None:
        """(Re)write the plugin module on disk; the next import reads it."""
        if source is None:
            parts = [
                f"VERSION = {version!r}",
                f"OWNER = {self.provider!r}",
            ]
            if extra:
                parts.append(extra)
            parts += [
                "",
                "",
                "def _pair(*args, **kwargs):",
                '    return "y-" + VERSION, "scale-" + VERSION',
                "",
                "",
                "def register(host):",
                _indent(register_source or _DEFAULT_REGISTER),
            ]
            if prepare_source is not OMIT:
                if prepare_source is None:
                    prepare_source = 'return {"build": VERSION, "device": str(device)}'
                parts += ["", "", "def prepare(device):", _indent(prepare_source)]
            if close_source:
                parts += ["", "", close_source.rstrip("\n")]
            source = "\n".join(parts) + "\n"
        (self.package_dir / "atom_hook.py").write_text(source, encoding="utf-8")

    def module(self):
        return sys.modules.get(self.module_name)


@pytest.fixture
def dev(tmp_path, monkeypatch):
    """A plugin in ``tmp_path/devroot`` plus its entry point."""
    root = tmp_path / "devroot"
    root.mkdir()
    (root / "plugin_pkg_marker.txt").write_text("not code", encoding="utf-8")
    plugin = DevPlugin(root, monkeypatch)
    try:
        yield plugin
    finally:
        for name in [
            n
            for n in sys.modules
            if n == plugin.package or n.startswith(plugin.package + ".")
        ]:
            sys.modules.pop(name, None)


def _native(*args, **kwargs):
    return "native-y", "native-scale"


def _boot(plugin, *, dev_mode=True, roots=None, bind=True):
    """Configure for a dev hot-plug boot and bind one seam (the fixture wrote v1)."""
    if not dev_mode:
        dev_roots = ()
    elif roots is None:
        dev_roots = [str(plugin.root)]
    else:
        dev_roots = roots
    host.configure(
        [plugin.provider],
        hotplug=True,
        dev=dev_mode,
        dev_roots=dev_roots,
        kill_switch=False,
    )
    return host.bind(OUT, _native) if bind else None


def _code_digest(plugin):
    return host.plugin_code_digest([str(plugin.root)])


# -- boot prepare ------------------------------------------------------------


def test_prepare_plan_calls_prepare_once_per_planned_provider(dev):
    _boot(dev)
    state = host.status()
    assert state["dev"] is True
    assert state["dev_roots"] == [str(dev.root.resolve())]
    assert state["code_digest"] == _code_digest(dev)
    assert host.dev_enabled() is True

    prepared = host.prepare_plan("cpu")
    assert prepared == {
        "prepared": [dev.provider],
        "native": {dev.provider: {"build": "v1", "device": "cpu"}},
    }


def test_prepare_plan_is_empty_without_a_plan():
    assert host.prepare_plan("cpu") == {"prepared": [], "native": {}}
    assert host.dev_enabled() is False
    assert host.status()["dev"] is False and host.status()["code_digest"] is None


def test_a_boot_prepare_failure_refuses_the_start_with_the_provider_named(dev):
    dev.write("v1", prepare_source="raise RuntimeError('native lib missing')")
    host.configure(
        [dev.provider],
        hotplug=True,
        dev=True,
        dev_roots=[str(dev.root)],
        kill_switch=False,
    )
    with pytest.raises(RuntimeError) as refused:
        host.prepare_plan("cuda:0")
    assert dev.provider in str(refused.value)
    assert "prepare" in str(refused.value)
    assert "native lib missing" in str(refused.value)


def test_a_provider_without_prepare_is_skipped(dev):
    dev.write("v1", prepare_source=OMIT)
    source = (dev.package_dir / "atom_hook.py").read_text(encoding="utf-8")
    assert "def prepare(" not in source
    _boot(dev)
    assert host.prepare_plan("cpu") == {"prepared": [], "native": {}}
    assert host.owners(OUT, "any") == (dev.provider,)


# -- reload: the new code serves --------------------------------------------


def test_reload_reimports_the_plugin_and_serves_the_new_implementation(dev):
    slot = _boot(dev)
    assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")
    boot_plan = host.status()["plan_id"]
    digest_before = _code_digest(dev)
    old_module = dev.module()
    assert old_module is not None and old_module.VERSION == "v1"

    dev.write("v2")
    digest_after = _code_digest(dev)
    assert digest_after != digest_before

    started = time.perf_counter()
    result = host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    elapsed = time.perf_counter() - started

    assert result["hotplug"] is True and result["dev"] is True
    assert result["reloaded"] is True
    assert result["plan_id"] == boot_plan == host.status()["plan_id"]
    assert result["code_digest"] == digest_after == _code_digest(dev)
    assert result["reloaded_modules"] == sorted({dev.package, dev.module_name})
    assert result["prepared"] == [dev.provider]
    assert result["native"] == {dev.provider: {"build": "v2", "device": "cpu"}}
    assert 0.0 <= result["reload_seconds"] <= elapsed + 0.01

    # The bound seam routes to the re-imported implementation, and the module
    # in sys.modules is a new object (the old one was replaced, not mutated).
    assert slot(1, 2, 3, 4) == ("y-v2", "scale-v2")
    assert dev.module() is not old_module and dev.module().VERSION == "v2"
    assert host.owners(OUT, "any") == (dev.provider,)
    assert host.status()["dev"] is True


def test_reload_never_serves_a_stale_bytecode_cache(dev):
    """An edit that keeps the size and the mtime second must still reload.

    CPython's default ``.pyc`` validation is (source mtime, size), so writing
    ``v1`` -> ``v2`` in the same second with the same file size would import
    the old bytecode even after the module is dropped. The reload drops each
    module's ``__cached__`` file, so this pins that guarantee.
    """
    slot = _boot(dev)
    assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")
    module = dev.module()
    cached = getattr(module, "__cached__", None)
    assert cached, "the fixture must have produced a bytecode cache to drop"
    source = dev.package_dir / "atom_hook.py"
    stat = source.stat()
    dev.write("v2")
    os.utime(source, (stat.st_atime, stat.st_mtime))  # same mtime, same size
    assert source.stat().st_size == stat.st_size
    installed = host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    assert installed["native"] == {dev.provider: {"build": "v2", "device": "cpu"}}
    assert slot(1, 2, 3, 4) == ("y-v2", "scale-v2")


def test_reload_takes_the_plan_object_or_the_enable_names(dev):
    _boot(dev)
    plan = recipes.shorthand([dev.provider])
    dev.write("v2")
    installed = host.reload_plugins(plan, roots=[str(dev.root)], device="cpu")
    assert installed["plan_id"] == plan.plan_id == host.status()["plan_id"]
    dev.write("v3")
    installed = host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    assert installed["native"][dev.provider]["build"] == "v3"


def test_reload_of_a_python_only_edit_keeps_the_native_build(dev):
    """The digest covers Python code; a code edit alone changes only it."""
    slot = _boot(dev)
    first = host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    assert first["native"] == {dev.provider: {"build": "v1", "device": "cpu"}}
    dev.write("v2")
    second = host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    assert first["code_digest"] != second["code_digest"]
    assert second["native"] == {dev.provider: {"build": "v2", "device": "cpu"}}
    assert slot(1, 2, 3, 4) == ("y-v2", "scale-v2")


# -- reload rollback ---------------------------------------------------------


@pytest.mark.parametrize("failure", ["import", "register", "prepare"])
def test_a_failed_reload_restores_the_modules_the_plan_and_the_bindings(dev, failure):
    slot = _boot(dev)
    assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")
    plan_id = host.status()["plan_id"]
    refusals = len(host.status()["refusals"])
    old_module = dev.module()

    if failure == "import":
        dev.write("v2", source="def broken(:\n")
    elif failure == "register":
        dev.write(
            "v2",
            register_source="raise RuntimeError('register boom')",
        )
    else:
        dev.write("v2", prepare_source="raise RuntimeError('prepare boom')")

    with pytest.raises(Exception) as failed:
        host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")

    if failure == "prepare":
        assert "prepare boom" in str(failed.value)
    assert dev.module() is old_module and old_module.VERSION == "v1"
    assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")
    state = host.status()
    assert state["plan_id"] == plan_id
    assert state["dev"] is True and state["dev_roots"] == [str(dev.root.resolve())]
    assert len(state["refusals"]) == refusals
    assert host.owners(OUT, "any") == (dev.provider,)


def test_a_failed_reload_closes_the_fresh_modules_best_effort(dev):
    """The half-imported new code is closed; the old modules stay untouched."""
    slot = _boot(dev)
    old_module = dev.module()
    marker = dev.root / "closed.txt"
    dev.write(
        "v2",
        extra="from pathlib import Path",
        close_source=(
            "def close():\n" f"    Path({str(marker)!r}).write_text(VERSION)\n"
        ),
        prepare_source="raise RuntimeError('prepare boom')",
    )
    with pytest.raises(RuntimeError):
        host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    # The fresh module object (same name, new code) was closed on the way out.
    assert marker.read_text(encoding="utf-8") == "v2"
    assert dev.module() is old_module and old_module.VERSION == "v1"
    assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")


def test_reload_refuses_to_reimport_a_compiled_python_extension(dev):
    """A compiler extension cannot be re-initialized: refuse, change nothing."""
    slot = _boot(dev)
    plan_id = host.status()["plan_id"]
    extension = dev.root / "native_thing.so"
    extension.write_bytes(b"\x7fELF fake")
    name = f"{dev.package}.native_thing"
    fake = types.ModuleType(name)
    fake.__file__ = str(extension)
    sys.modules[name] = fake
    try:
        with pytest.raises(RuntimeError, match="compiled Python extension"):
            host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    finally:
        sys.modules.pop(name, None)
    assert host.status()["plan_id"] == plan_id
    assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")


# -- dev mode and root validation -------------------------------------------


def test_reload_needs_dev_mode(dev):
    _boot(dev, dev_mode=False)
    with pytest.raises(RuntimeError, match="dev mode"):
        host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")
    assert host.dev_enabled() is False


def test_reload_waits_for_the_model_build(dev):
    _boot(dev, bind=False)
    with pytest.raises(RuntimeError, match="after the model build"):
        host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")


def test_reload_roots_must_be_the_configured_allowlist(dev, tmp_path):
    _boot(dev)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="not allowlisted"):
        host.reload_plugins([dev.provider], roots=[str(elsewhere)], device="cpu")
    assert host.status()["plan_id"] is not None  # nothing changed


def test_reload_roots_cannot_be_narrowed_to_leave_a_tree_stale(dev, tmp_path):
    other = tmp_path / "other-root"
    other.mkdir()
    _boot(dev, roots=[str(dev.root), str(other)])
    with pytest.raises(ValueError, match="every configured dev root"):
        host.reload_plugins([dev.provider], roots=[str(dev.root)], device="cpu")


def test_dev_configuration_is_validated(dev):
    with pytest.raises(ValueError, match="hot-plug boot"):
        host.configure(
            None, hotplug=False, dev=True, dev_roots=[str(dev.root)], kill_switch=False
        )
    with pytest.raises(ValueError, match="at least one"):
        host.configure(None, hotplug=True, dev=True, dev_roots=(), kill_switch=False)
    with pytest.raises(ValueError, match="existing directory"):
        host.configure(
            None,
            hotplug=True,
            dev=True,
            dev_roots=[str(dev.root / "missing")],
            kill_switch=False,
        )
    with pytest.raises(ValueError, match="without plugin"):
        host.configure(
            None,
            hotplug=True,
            dev=False,
            dev_roots=[str(dev.root)],
            kill_switch=False,
        )


def test_dev_mode_cannot_run_with_the_plugin_host_killed(dev):
    with pytest.raises(ValueError, match="plugin host enabled"):
        host.configure(
            None,
            hotplug=True,
            dev=True,
            dev_roots=[str(dev.root)],
            kill_switch=True,
        )


def test_a_provider_imported_outside_the_dev_roots_is_refused(
    dev, tmp_path, monkeypatch
):
    """A reload that cannot drop the module must refuse, not silently no-op."""
    elsewhere = tmp_path / "installed-copy"
    elsewhere.mkdir()
    installed = DevPlugin(elsewhere, monkeypatch)
    empty_root = tmp_path / "empty-root"
    empty_root.mkdir()
    try:
        slot = _boot(installed, roots=[str(empty_root)])
        with pytest.raises(RuntimeError, match="not under a --plugin-dev-root"):
            host.reload_plugins(
                [installed.provider], roots=[str(empty_root)], device="cpu"
            )
        assert slot(1, 2, 3, 4) == ("y-v1", "scale-v1")
        assert host.status()["plan_id"] is not None
    finally:
        for name in [
            n
            for n in sys.modules
            if n == installed.package or n.startswith(installed.package + ".")
        ]:
            sys.modules.pop(name, None)


# -- the code digest ---------------------------------------------------------


def _digest(plugin):
    return host.plugin_code_digest([str(plugin.root)])


def test_the_code_digest_is_stable_and_follows_the_documented_layout(dev):
    digest = _digest(dev)
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert digest == _digest(dev)
    # Files that are not plugin code do not change it.
    (dev.root / "README.md").write_text("docs", encoding="utf-8")
    cache = dev.root / "__pycache__"
    cache.mkdir()
    (cache / "atom_hook.cpython-314.pyc").write_bytes(b"pyc")
    assert _digest(dev) == digest
    # Python, native libraries and headers do.
    (dev.root / "kernels").mkdir()
    (dev.root / "kernels" / "fp6_quant.h").write_text("#pragma once", encoding="utf-8")
    after_header = _digest(dev)
    assert after_header != digest
    (dev.root / "libfp6quant_v2.so").write_bytes(b"\x7fELF v2")
    after_so = _digest(dev)
    assert after_so != after_header
    dev.write("v2")
    assert _digest(dev) not in (digest, after_header, after_so)


def test_the_code_digest_accepts_nested_roots_and_dedupes(dev):
    assert host.plugin_code_digest([str(dev.root), str(dev.package_dir)]) == _digest(
        dev
    )
    # Roots are resolved to absolute paths.
    assert host.plugin_code_digest([str(dev.root.resolve())]) == _digest(dev)


def test_the_code_digest_refuses_to_hash_code_it_cannot_read(dev):
    with pytest.raises(ValueError, match="existing directory"):
        host.plugin_code_digest([str(dev.root / "missing")])
    with pytest.raises(ValueError, match="at least one"):
        host.plugin_code_digest([])
    # The configured allowlist is the default.
    _boot(dev)
    assert host.plugin_code_digest() == _code_digest(dev)


# -- the hot-plug payload ----------------------------------------------------


def _recipe(seams):
    owners = sorted({owner for names in seams.values() for owner in names})
    document = {
        "schema": recipes.RECIPE_SCHEMA,
        "model_arch": "DeepseekV41ForCausalLM",
        "plugins": [
            {
                "name": owner,
                "serves": sorted(
                    seam for seam, names in seams.items() if owner in names
                ),
            }
            for owner in owners
        ],
        "plan": {"seams": {seam: list(names) for seam, names in seams.items()}},
        "graph": {"mode": "any"},
    }
    return json.dumps(document)


def test_a_reload_payload_parses_and_normalizes_the_digest():
    digest = "a" * 64
    request = hotplug.parse_hotplug_payload(
        {"recipe": [_recipe({OUT: ["bmm"]})], "reload": True, "code_digest": digest}
    )
    assert request.reload is True
    assert request.code_digest == "sha256:" + digest
    request = hotplug.parse_hotplug_payload(
        {
            "recipe": [_recipe({OUT: ["bmm"]})],
            "reload": True,
            "code_digest": "sha256:" + "B" * 64,
        }
    )
    assert request.code_digest == "sha256:" + "b" * 64
    # A plain hot-plug keeps its old defaults.
    request = hotplug.parse_hotplug_payload({"recipe": [_recipe({OUT: ["bmm"]})]})
    assert request.reload is False and request.code_digest is None


@pytest.mark.parametrize(
    "extra, reason",
    [
        ({"reload": "yes"}, "payload-schema"),
        ({"reload": 1}, "payload-schema"),
        ({"reload": True, "code_digest": "nope"}, "code-digest-invalid"),
        ({"reload": True, "code_digest": "sha256:" + "z" * 64}, "code-digest-invalid"),
        ({"reload": True, "code_digest": 7}, "payload-schema"),
    ],
)
def test_a_bad_reload_payload_is_a_400_with_its_reason(extra, reason):
    with pytest.raises(hotplug.HotplugPayloadError) as refused:
        hotplug.parse_hotplug_payload({"recipe": [_recipe({OUT: ["bmm"]})], **extra})
    assert refused.value.reason == reason
    assert hotplug.refusal_status(refused.value) == 400


def _report(**overrides):
    report = {
        "rank": 0,
        "tp_world_size": 4,
        "status": "ok",
        "error": None,
        "plan_id": "sha256:plan",
        "rows": [8],
        "sizes": [2],
        "capture_seconds": 0.5,
        "reloaded": False,
        "code_digest": None,
        "reload_seconds": 0.0,
        "prepared": [],
        "native": None,
    }
    report.update(overrides)
    return report


def test_aggregate_reports_lifts_the_reload_fields_to_the_reply():
    merged = hotplug.aggregate_reports(
        [
            _report(
                reloaded=True,
                code_digest="sha256:" + "a" * 64,
                reload_seconds=1.5,
                prepared=["fp6"],
                native={"fp6": {"build": "v2"}},
            ),
            _report(
                rank=1,
                reloaded=True,
                code_digest="sha256:" + "a" * 64,
                reload_seconds=2.5,
                prepared=["fp6"],
                native={"fp6": {"build": "v2"}},
            ),
        ]
    )
    assert merged["reloaded"] is True
    assert merged["code_digest"] == "sha256:" + "a" * 64
    assert merged["reload_seconds"] == 2.5
    assert merged["prepared"] == ["fp6"]
    assert merged["native"] == {"fp6": {"build": "v2"}}
    assert len(merged["ranks"]) == 2

    plain = hotplug.aggregate_reports([_report(), _report(rank=1)])
    assert plain["reloaded"] is False
    assert plain["code_digest"] is None and plain["native"] is None


def test_ranks_that_reloaded_different_code_are_refused():
    with pytest.raises(hotplug.HotplugPayloadError) as refused:
        hotplug.aggregate_reports(
            [
                _report(reloaded=True, code_digest="sha256:" + "a" * 64),
                _report(rank=1, reloaded=True, code_digest="sha256:" + "b" * 64),
            ]
        )
    assert refused.value.reason == "code-digest-mismatch"


# -- the engine handler and the round guard ----------------------------------


class FakeEngine:
    """The slice of EngineCore the hot-plug utility handler touches."""

    def __init__(self, *, dev_mode=True):
        self.config = types.SimpleNamespace(
            plugin_hotplug=True,
            plugin_dev=dev_mode,
            plugin_dev_roots=("/plugin/root",),
            pipeline_parallel_size=1,
            enable_dp_attention=False,
            enable_rapidserve=False,
            parallel_config=types.SimpleNamespace(data_parallel_size=1),
        )
        self.plugin_round_id = None
        self.plugin_round_started_at = None
        self._generation_paused = False
        self.scheduler = None
        self.paused = 0
        self.resumed = 0

    def pause_generation(self, deadline=None):
        self.paused += 1
        self._generation_paused = True

    def resume_generation(self):
        self.resumed += 1
        self._generation_paused = False

    def has_pending_kv_work(self):
        return False


class FakeRunner:
    """The model-runner side: record the hot-plug call, report one plan."""

    def __init__(self):
        self.calls = []
        self.report = {
            "rank": 0,
            "status": "ok",
            "plan_id": "sha256:plan",
            "rows": [8],
            "sizes": [2],
            "capture_seconds": 0.1,
            "reloaded": True,
            "code_digest": "sha256:" + "a" * 64,
            "native": {"dev-fixture": {"build": "v2"}},
        }

    def call_func(self, name, payload=None, wait_out=False):
        self.calls.append((name, payload))
        return dict(self.report)


class Wired:
    def __init__(self, *, dev_mode=True):
        self.engine = FakeEngine(dev_mode=dev_mode)
        self.runner = FakeRunner()
        self.handler = EngineUtilityHandler(self.runner, queue.Queue(), label="test")
        self.handler.engine = self.engine

    def reply(self):
        kind, reply = self.handler.output_queue.get_nowait()
        assert kind == "UTILITY_RESPONSE"
        return reply


@pytest.fixture
def wired():
    return Wired()


def test_a_reload_during_a_plugin_round_is_refused_before_anything(wired):
    began = wired.handler._handle_begin_plugin_round({"round_id": "run-1"})
    reply = wired.handler._handle_hotplug_plugin(
        {"payload": {"reload": True, "capture": False}}
    )
    assert reply["reason"] == hotplug.ROUND_IN_PROGRESS_REASON
    assert reply["active_round"]["round_id"] == "run-1"
    # No pause, no reload, no install: the round keeps the plan.
    assert wired.engine.paused == 0 and wired.engine.resumed == 0
    assert wired.runner.calls == []
    assert wired.engine.plugin_round_id == "run-1"
    assert wired.engine.plugin_round_started_at == began["started_at"]


def test_a_reload_on_a_non_dev_server_is_refused_before_the_pause():
    wired = Wired(dev_mode=False)
    reply = wired.handler._handle_hotplug_plugin(
        {"payload": {"reload": True, "capture": False}}
    )
    assert reply["reason"] == hotplug.RELOAD_UNSUPPORTED_REASON
    # The API maps that coded refusal to a 409 with the reason.
    coded = hotplug.PluginRefusal(hotplug.RELOAD_UNSUPPORTED_REASON, "not a dev box")
    assert hotplug.refusal_status(coded) == 409
    assert hotplug.refusal_reason(coded) == hotplug.RELOAD_UNSUPPORTED_REASON
    assert wired.engine.paused == 0 and wired.engine.resumed == 0
    assert wired.runner.calls == []


def test_a_runner_reload_refusal_keeps_its_reason(wired):
    wired.runner.report = {
        "rank": 0,
        "status": "error",
        "error": "RuntimeError: reload code digest mismatch",
        "reason": hotplug.RELOAD_DIGEST_MISMATCH_REASON,
    }
    payload = {"reload": True, "code_digest": "sha256:" + "a" * 64, "capture": False}
    reply = wired.handler._handle_hotplug_plugin({"payload": payload})
    assert reply["reason"] == hotplug.RELOAD_DIGEST_MISMATCH_REASON
    # The handler still resumes the generation it paused and reports one call.
    assert wired.engine.paused == 1 and wired.engine.resumed == 1
    assert wired.runner.calls == [(hotplug.HOTPLUG_CMD, payload)]


def test_a_successful_reload_reply_carries_the_reload_result(wired):
    payload = {"reload": True, "code_digest": "sha256:" + "a" * 64, "capture": False}
    reply = wired.handler._handle_hotplug_plugin({"payload": payload})
    result = reply["result"]
    assert result["reloaded"] is True
    assert result["code_digest"] == "sha256:" + "a" * 64
    assert result["native"] == {"dev-fixture": {"build": "v2"}}
    assert wired.engine.paused == 1 and wired.engine.resumed == 1


# -- wiring: config, CLI, env, runner, API ----------------------------------

TEXT = {
    path: (REPO_ROOT / path).read_text(encoding="utf-8")
    for path in (
        "atom/config.py",
        "atom/model_engine/arg_utils.py",
        "atom/model_engine/model_runner.py",
        "atom/model_engine/engine_utility.py",
        "atom/model_engine/engine_core_mgr.py",
        "atom/entrypoints/openai/api_server.py",
        "atom/utils/envs.py",
    )
}


def _flat(source):
    return re.sub(r"\s+", " ", source)


def test_the_config_and_environment_carry_dev_mode():
    config = _flat(TEXT["atom/config.py"])
    assert "plugin_dev: bool = False" in config
    assert "plugin_dev_roots: tuple[str, ...] = ()" in config
    assert "plugin-dev" in config  # the compile-cache marker
    envs = TEXT["atom/utils/envs.py"]
    assert '"ATOM_PLUGIN_DEV": lambda:' in envs
    assert '"ATOM_PLUGIN_DEV_ROOTS": lambda:' in envs
    assert "_path_list_env" in envs


def test_the_cli_flags_reach_the_config(tmp_path):
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    root = tmp_path / "plugin-root"
    root.mkdir()
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(
        [
            "--plugin-hotplug",
            "--plugin-dev",
            "--plugin-dev-root",
            str(root),
            "--plugin-dev-root",
            str(tmp_path),
        ]
    )
    kwargs = EngineArgs.from_cli_args(args)._get_engine_kwargs()
    assert kwargs["plugin_dev"] is True
    assert kwargs["plugin_dev_roots"] == (str(root), str(tmp_path))
    assert {"plugin_dev", "plugin_dev_roots"} <= set(Config.__dataclass_fields__)


def test_the_cli_refuses_dev_mode_without_its_prerequisites(tmp_path):
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    def kwargs(argv):
        return EngineArgs.from_cli_args(parser.parse_args(argv))._get_engine_kwargs()

    with pytest.raises(ValueError, match="--plugin-hotplug"):
        kwargs(["--plugin-dev", "--plugin-dev-root", str(tmp_path)])
    with pytest.raises(ValueError, match="--plugin-dev-root"):
        kwargs(["--plugin-hotplug", "--plugin-dev"])


def test_the_env_is_the_fallback_for_the_dev_flags(monkeypatch, tmp_path):
    from atom.model_engine.arg_utils import EngineArgs
    from atom.utils.arg_parser import FlexibleArgumentParser

    monkeypatch.setenv("ATOM_PLUGIN_DEV", "1")
    monkeypatch.setenv("ATOM_PLUGIN_DEV_ROOTS", str(tmp_path))
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = parser.parse_args(["--plugin-hotplug"])
    kwargs = EngineArgs.from_cli_args(args)._get_engine_kwargs()
    assert kwargs["plugin_dev"] is True
    assert kwargs["plugin_dev_roots"] == (str(tmp_path),)


def test_the_runner_configures_dev_mode_and_prepares_the_boot_plan():
    flat = _flat(TEXT["atom/model_engine/model_runner.py"])
    assert 'dev=getattr(config, "plugin_dev", False)' in flat
    assert 'dev_roots=getattr(config, "plugin_dev_roots", ())' in flat
    assert "plugin_host.prepare_plan(self.device)" in flat
    # Prepare runs after the model is built and before warmup/capture.
    source = TEXT["atom/model_engine/model_runner.py"]
    assert source.index("self._build_and_load_model(model_class)") < source.index(
        "plugin_host.prepare_plan(self.device)"
    )
    assert source.index("plugin_host.prepare_plan(self.device)") < source.index(
        "self._maybe_warmup()"
    )


def test_the_runner_reload_path_checks_dev_mode_and_digests():
    source = TEXT["atom/model_engine/model_runner.py"]
    flat = _flat(source)
    assert "plugin_host.dev_enabled()" in flat
    assert "plugin_host.plugin_code_digest(roots)" in flat
    assert 'checked_code_digest(payload.get("code_digest"))' in flat
    assert "self._tp_agreed_code_digest(local_digest)" in flat
    assert "plugin_host.reload_plugins(" in flat
    assert "RELOAD_UNSUPPORTED_REASON" in source
    assert "RELOAD_DIGEST_MISMATCH_REASON" in source
    runner = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "ModelRunner"
    )
    methods = {node.name for node in runner.body if isinstance(node, ast.FunctionDef)}
    assert "_tp_agreed_code_digest" in methods
    # A reload failure is reported before the capture: the recapture call comes
    # after the two _tp_agree gates, so a refused reload releases no graph.
    assert flat.index("plugin_host.reload_plugins(") < flat.index(
        "self.recapture_cudagraph(rows)"
    )
    assert flat.index("_tp_agree(0 if failure is None else 1)") < flat.index(
        "plugin_host.reload_plugins("
    )


def test_the_utility_handler_refuses_a_reload_before_pausing_and_keeps_reasons():
    flat = _flat(TEXT["atom/model_engine/engine_utility.py"])
    assert "RELOAD_UNSUPPORTED_REASON" in flat
    assert 'result.get("reason")' in flat
    assert 'reply["reason"] = reason' in flat
    # The round guard is first; the dev-mode pre-check is before the pause.
    assert flat.index("round_in_progress_reply(HOTPLUG_CMD, engine)") < flat.index(
        'payload.get("reload") is True'
    )
    assert flat.index('payload.get("reload") is True') < flat.index(
        "engine.pause_generation(deadline=deadline)"
    )


def test_the_manager_forwards_the_reload_payload():
    flat = _flat(TEXT["atom/model_engine/engine_core_mgr.py"])
    assert "payload=payload" in flat
    assert "reload" in flat  # documented in hotplug_plugin's docstring


def test_the_api_route_maps_reload_fields_and_refuses_a_non_dev_server():
    tree = ast.parse(TEXT["atom/entrypoints/openai/api_server.py"])
    route = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "plugin_hotplug"
    )
    doc = ast.get_docstring(route) or ""
    assert '"reload"' in doc and "code_digest" in doc and "prepare(device)" in doc
    flat = _flat(
        ast.get_source_segment(TEXT["atom/entrypoints/openai/api_server.py"], route)
    )
    assert '"reload": hotplug.reload' in flat
    assert '"code_digest": hotplug.code_digest' in flat
    assert "RELOAD_UNSUPPORTED_REASON" in flat
    assert 'getattr(config, "plugin_dev", False)' in flat
    # The dev check is before the manager call (so no pause happens for it).
    assert flat.index("RELOAD_UNSUPPORTED_REASON") < flat.index(
        'call = getattr(manager, "hotplug_plugin", None)'
    )
    assert "asyncio.to_thread(call, payload, hotplug.timeout)" in flat

    status_route = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "plugin_status"
    )
    status_flat = _flat(
        ast.get_source_segment(
            TEXT["atom/entrypoints/openai/api_server.py"], status_route
        )
    )
    assert '"dev_mode"' in status_flat and '"dev_roots"' in status_flat

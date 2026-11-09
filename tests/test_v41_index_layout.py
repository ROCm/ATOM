# SPDX-License-Identifier: MIT
"""CPU checks for V4.1 FP4 layout selection and cache compatibility."""

import ast
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import atom.config as config_module
from atom.config import CompilationConfig, Config
from atom.kv_transfer.offload import config as offcfg
from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    PagedStateCheckpointSpec,
)
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.pool_layout.entry_arena import EntryMajorArena
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.model_ops.deepseek_v41.index_plane import FP4_MFMA, IndexUnits
from atom.utils import CpuGpuBuffer
from tests.test_deepseek_v4_transfer_regions import mooncake_page_writer  # noqa: F401

ROOT = Path(__file__).parents[1]


def _geometry(layout, *, fp4=True):
    return V41PoolGeometry(
        2,
        ((0, 2), (1, 1)),
        128,
        128,
        512,
        128,
        index_fp4=fp4,
        index_block_rows=8,
        index_layout=layout,
    )


@pytest.mark.parametrize(
    "arch,backend,expected",
    [
        ("gfx950", "opus", "mfma"),
        ("gfx950", "flydsl", "preshuffle"),
        ("gfx1250", "opus", "natural"),
    ],
)
def test_startup_snapshots_runtime_layout_for_scheduler_and_worker(
    monkeypatch, arch, backend, expected
):
    # A multi-arch build override disagrees with the actual device. Neither
    # startup nor later namespace construction may consult the build target.
    monkeypatch.setenv("GPU_ARCHS", "gfx950;gfx1250")
    monkeypatch.setenv("ATOM_V4_UNIFIED_MQA", backend)
    chip = ModuleType("aiter.jit.utils.chip_info")
    chip.get_gfx = Mock(side_effect=AssertionError("build target queried"))
    chip.get_gfx_runtime = Mock(return_value=arch)
    monkeypatch.setitem(sys.modules, chip.__name__, chip)
    monkeypatch.setattr(config_module, "get_generation_config", lambda _: None)
    monkeypatch.setattr(
        config_module,
        "QuantizationConfig",
        lambda *a, **kw: SimpleNamespace(exclude_layers=[]),
    )
    config = Config(
        model=str(ROOT / "tests/models/deepseek_v41/fixtures"),
        tensor_parallel_size=4,
        enable_expert_parallel=True,
        enforce_eager=True,
        index_cache_dtype="fp4",
        kv_transfer_config={"kv_connector": "lmcache_mp"},
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
    )
    config.quant_config = None
    assert config.index_cache_layout == expected
    chip.get_gfx_runtime.assert_called_once()
    chip.get_gfx.assert_not_called()
    cfg = SimpleNamespace(chunk_size=8192)
    scheduler_namespace = offcfg.build_page_namespace(config, cfg, 4)
    compile_hash = config.compute_hash()

    # Spawned workers receive the resolved value; subsequent environment
    # changes cannot silently reinterpret persistent PAGE objects.
    worker = pickle.loads(pickle.dumps(config))
    monkeypatch.setenv("ATOM_V4_UNIFIED_MQA", "flydsl" if backend == "opus" else "opus")
    chip.get_gfx_runtime.side_effect = AssertionError("namespace queried a device")
    assert worker.index_cache_layout == expected
    assert offcfg.build_page_namespace(worker, cfg, 4) == scheduler_namespace
    assert worker.compute_hash() == compile_hash
    worker.index_cache_layout = "natural" if expected != "natural" else "mfma"
    assert offcfg.build_page_namespace(worker, cfg, 4) != scheduler_namespace
    assert worker.compute_hash() != compile_hash


def test_equal_size_fp4_layouts_have_distinct_checkpoint_identities():
    layouts = [_geometry(name) for name in ("preshuffle", "mfma", "natural")]
    assert len({geo.paged_bytes for geo in layouts}) == 1
    assert len({geo.checkpoint_image_bytes for geo in layouts}) == 1
    assert len({geo.layout_id for geo in layouts}) == 3
    assert all(replace(geo).layout_id == geo.layout_id for geo in layouts)
    assert _geometry(None).layout_id == layouts[0].layout_id
    assert _geometry(None, fp4=False).layout_id not in {
        geo.layout_id for geo in layouts
    }


@pytest.fixture
def cache_cls():
    # Load the actual allocation class independently of the module's GPU
    # scorers/writers. All storage and view arithmetic below runs on CPU.
    path = ROOT / "atom/model_ops/attentions/deepseek_v41/cache.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "PagedAttentionCache"
    )
    namespace = {
        "torch": torch,
        "EntryMajorArena": EntryMajorArena,
        "CpuGpuBuffer": CpuGpuBuffer,
        "FP4_MFMA": FP4_MFMA,
        "IndexUnits": IndexUnits,
    }
    exec(  # noqa: S102 -- executes the production allocation class
        compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["PagedAttentionCache"]


@pytest.mark.parametrize("layout", ["preshuffle", "mfma", "natural"])
def test_cache_uses_declared_layout_and_rejects_other_checkpoint(cache_cls, layout):
    geo = _geometry(layout)
    cache = cache_cls(geo, 2, 1, "cpu")
    assert {units.layout for units in cache.index_units.values()} == {layout}
    spec = PagedStateCheckpointSpec(
        geo.paged_bytes,
        geo.state_bytes,
        geo.layout_id,
        geo.checkpoint_image_bytes,
    )
    copies = StateCopies(cache, spec, 1)
    other = _geometry("mfma" if layout != "mfma" else "preshuffle")
    op = CheckpointRestoreOp(
        0,
        tuple(range(spec.units_per_checkpoint)),
        geo.checkpoint_image_bytes,
        other.layout_id,
    )
    before = cache.backing.clone()
    with pytest.raises(ValueError, match="Checkpoint layout or size mismatch"):
        copies.execute([], [op])
    torch.testing.assert_close(cache.backing, before)


@pytest.fixture
def v41_export():
    # As in test_lmcache_mp_native_layout: run the shipped exporter with CPU
    # tensors, leaving model construction and GPU kernel imports outside it.
    path = ROOT / "atom/model_ops/attentions/deepseek_v41/backend.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41MetadataBuilder"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_kv_transfer_tensors"
    )
    namespace = {}
    exec(  # noqa: S102 -- execute production exporter, not a copy of its logic
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["get_kv_transfer_tensors"]


def _export(layout, exporter, cache_cls):
    geo = _geometry(layout)
    builder = SimpleNamespace(
        cache=cache_cls(geo, 2, 1, "cpu"),
        model_runner=SimpleNamespace(
            config=SimpleNamespace(kv_transfer_config={"kv_connector": "mooncake"}),
            state_runtime=SimpleNamespace(checkpoint_spec=None),
        ),
    )
    transfer = exporter(builder)
    transfer.set_block_count(2)
    return transfer


@pytest.mark.parametrize("consumer_layout", ["preshuffle", "mfma", "natural"])
def test_pd_checks_physical_layout_before_writing(
    v41_export, cache_cls, mooncake_page_writer, consumer_layout  # noqa: F811
):
    producer = _export("mfma", v41_export, cache_cls)
    consumer = _export(consumer_layout, v41_export, cache_cls)
    src, dst = producer.block_regions, consumer.block_regions
    assert [r.unit_bytes for r in src] == [r.unit_bytes for r in dst]
    writes = Mock(return_value=True)
    connector = SimpleNamespace(
        _slot_regions=[],
        _block_regions=[(r.base_addr, r.unit_bytes) for r in src],
        _block_region_roles=[r.semantic_role for r in src],
        _block_region_consumer_indices=None,
        _fp4_index_layout=False,
        pp_size=1,
        _consumer_region_map=lambda n, m, *args: list(range(n)),
        _rdma_write=writes,
    )
    request = {
        "consumer_block_base_addrs": [r.base_addr for r in dst],
        "consumer_block_bpb": [r.unit_bytes for r in dst],
        "consumer_region_roles": [r.semantic_role for r in dst],
        "consumer_slot_base_addrs": [],
        "consumer_slot_bps": [],
        "dst_slot_index": 0,
    }
    run = mooncake_page_writer["_execute_block_slot_transfer"]
    args = (connector, request, "cpu-test", [0], [1], {"slot_index": 0}, "request")
    if consumer_layout == "mfma":
        assert run(*args, blocks_only=True)
        writes.assert_called_once()
    else:
        before = [p.view.clone() for p in consumer.pages]
        with pytest.raises(RuntimeError, match="role mismatch"):
            run(*args, blocks_only=True)
        writes.assert_not_called()
        for page, saved in zip(consumer.pages, before):
            torch.testing.assert_close(page.view, saved)

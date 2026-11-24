# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

ATOMRL_ROOT = Path(__file__).parents[4]
REPO_ROOT = ATOMRL_ROOT / "tests"
SOURCE_ROOT = ATOMRL_ROOT / "atomrl" / "lumen"
OPS_INIT_PATH = SOURCE_ROOT / "models/dsv4/ops/__init__.py"
CANONICAL_OPS_INIT_PATH = SOURCE_ROOT / "ops/dsv4/__init__.py"
COMPRESSOR_PATH = SOURCE_ROOT / "models/dsv4/ops/compressor.py"
PRETRAIN_PATH = SOURCE_ROOT / "models/dsv4/megatron/pretrain.py"
PIPELINE_PATH = SOURCE_ROOT / "models/dsv4/megatron/pipeline.py"
DSV4_EXAMPLES = ATOMRL_ROOT / "lumen" / "examples" / "dsv4"

CANONICAL_OPS = {
    "sparse_mla.py",
    "indexer.py",
    "indexer_utils.py",
    "cp_utils.py",
    "dsa_topk.py",
    "rope.py",
    "qat.py",
    "utils.py",
}
CANONICAL_KERNELS = {
    "sparse_mla": {"triton_sparse_mla.py"},
    "indexer": {
        "aiter_dsv4_indexer.py",
    },
    "quant": {"precision_aligned_ops.py", "act_quant.py"},
}


def test_dsv4_sources_live_in_canonical_ops_and_kernel_trees():
    canonical_ops = SOURCE_ROOT / "ops/dsv4"
    for filename in CANONICAL_OPS:
        assert (canonical_ops / filename).is_file(), filename

    canonical_kernels = SOURCE_ROOT / "kernels/dsv4"
    for group, filenames in CANONICAL_KERNELS.items():
        assert (canonical_kernels / group / "__init__.py").is_file(), group
        for filename in filenames:
            assert (canonical_kernels / group / filename).is_file(), filename
    assert not tuple(canonical_kernels.rglob("tilelang_*.py"))

    assert not (SOURCE_ROOT / "models/dsv4/ops/kernel").exists()
    for legacy_name in (
        "sparse_mla_backend.py",
        "indexer_backend.py",
        "indexer_utils.py",
        "cp_utils.py",
        "dsa_topk.py",
        "rope.py",
        "qat.py",
        "utils.py",
    ):
        assert not (SOURCE_ROOT / "models/dsv4/ops" / legacy_name).exists()


def test_internal_python_imports_use_canonical_dsv4_paths():
    forbidden = (
        "atomrl.lumen.models.dsv4.ops.kernel",
        "atomrl.lumen.models.dsv4.ops.sparse_mla_backend",
        "atomrl.lumen.models.dsv4.ops.indexer_backend",
        "atomrl.lumen.models.dsv4.ops.indexer_utils",
        "atomrl.lumen.models.dsv4.ops.cp_utils",
        "atomrl.lumen.models.dsv4.ops.dsa_topk",
        "atomrl.lumen.models.dsv4.ops.rope",
        "atomrl.lumen.models.dsv4.ops.qat",
        "atomrl.lumen.models.dsv4.ops.utils",
    )
    for root in (SOURCE_ROOT, ATOMRL_ROOT / "lumen" / "examples"):
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert not any(old_path in source for old_path in forbidden), path


def test_canonical_and_model_facades_do_not_eagerly_import_qat_or_kernels():
    for path in (CANONICAL_OPS_INIT_PATH, OPS_INIT_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(
            not isinstance(node, ast.ImportFrom)
            or not (node.module or "").startswith(
                ("atomrl.lumen.ops.dsv4.qat", "atomrl.lumen.kernels.dsv4")
            )
            for node in imports
        ), path


def test_canonical_dsv4_modules_import_without_model_or_tilekernels(monkeypatch):
    lumen_package = types.ModuleType("atomrl.lumen")
    lumen_package.__path__ = [str(SOURCE_ROOT)]
    ops_package = types.ModuleType("atomrl.lumen.ops")
    ops_package.__path__ = [str(SOURCE_ROOT / "ops")]
    monkeypatch.setitem(sys.modules, "atomrl.lumen", lumen_package)
    monkeypatch.setitem(sys.modules, "atomrl.lumen.ops", ops_package)

    transformer = types.ModuleType("megatron.core.transformer")
    transformer.TransformerConfig = object
    for name in ("megatron", "megatron.core"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, transformer.__name__, transformer)

    for name in tuple(sys.modules):
        if name.startswith(
            (
                "atomrl.lumen.ops.dsv4",
                "atomrl.lumen.models.dsv4",
                "atomrl.lumen.kernels.dsv4",
                "tile_kernels",
            )
        ):
            monkeypatch.delitem(sys.modules, name)

    module_names = (
        "cp_utils",
        "dsa_topk",
        "indexer",
        "indexer_utils",
        "qat",
        "rope",
        "sparse_mla",
        "utils",
    )
    for module_name in module_names:
        module = importlib.import_module(f"atomrl.lumen.ops.dsv4.{module_name}")
        assert Path(module.__file__).is_relative_to(SOURCE_ROOT / "ops/dsv4")

    assert not any(name.startswith("atomrl.lumen.models.dsv4") for name in sys.modules)
    assert not any(name.startswith("atomrl.lumen.kernels.dsv4") for name in sys.modules)
    assert not any(name.startswith("tile_kernels") for name in sys.modules)


def test_indexer_is_fixed_to_aiter(monkeypatch):
    indexer = importlib.import_module("atomrl.lumen.ops.dsv4.indexer")
    aiter_marker = object()

    aiter_module = types.ModuleType(
        "atomrl.lumen.kernels.dsv4.indexer.aiter_dsv4_indexer"
    )
    aiter_module.batched_indexer_fwd = aiter_marker
    aiter_module.batched_indexer_bwd = aiter_marker
    monkeypatch.setitem(sys.modules, aiter_module.__name__, aiter_module)

    monkeypatch.setenv("V4_INDEXER_IMPL", "tilelang")
    assert indexer.get_indexer_impl() == "aiter"
    assert indexer.get_batched_indexer_fwd() is aiter_marker
    assert indexer.get_batched_indexer_bwd() is aiter_marker


def test_sparse_mla_is_fixed_to_triton(monkeypatch):
    sparse_mla = importlib.import_module("atomrl.lumen.ops.dsv4.sparse_mla")
    triton_marker = object()

    triton_module = types.ModuleType(
        "atomrl.lumen.kernels.dsv4.sparse_mla.triton_sparse_mla"
    )
    triton_module.sparse_attn_triton = triton_marker
    monkeypatch.setitem(sys.modules, triton_module.__name__, triton_module)

    monkeypatch.setenv("V4_SPARSE_MLA_BACKEND", "tilelang")
    assert sparse_mla.get_sparse_mla_backend() == "triton"
    assert sparse_mla.get_sparse_attn_fn() is triton_marker


def test_standalone_pretrain_enables_variable_pipeline_shapes_on_config():
    source = (SOURCE_ROOT / "patches/builders/dsv4.py").read_text(encoding="utf-8")
    assert "config.variable_seq_lengths = True" in source
    assert "config.batch_p2p_comm = False" in source
    assert "pipeline_model_parallel_size" in source


def test_dsv4_pipeline_shape_exchange_preserves_hc_dimension():
    assert PIPELINE_PATH.is_file()
    source = PIPELINE_PATH.read_text(encoding="utf-8")
    assert "shape_rank = 4" in source
    assert "_p2p_ops(" in source
    assert "torch.distributed.batch_isend_irecv(" not in source


def test_dsv4_ops_lazily_exports_qat_without_importing_tilekernels(monkeypatch):
    for name in tuple(sys.modules):
        if name == "tile_kernels" or name.startswith("tile_kernels."):
            monkeypatch.delitem(sys.modules, name)

    spec = importlib.util.spec_from_file_location(
        "_test_canonical_dsv4_ops_init", CANONICAL_OPS_INIT_PATH
    )
    assert spec is not None and spec.loader is not None
    canonical_ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(canonical_ops)
    assert not any(name.startswith("tile_kernels") for name in sys.modules)

    qat_path = SOURCE_ROOT / "ops/dsv4/qat.py"
    spec = importlib.util.spec_from_file_location("_test_dsv4_qat", qat_path)
    assert spec is not None and spec.loader is not None
    qat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qat)
    assert callable(qat.fp8_simulate_qat)
    assert not any(name.startswith("tile_kernels") for name in sys.modules)

    marker = object()
    canonical_module = types.ModuleType("atomrl.lumen.ops.dsv4")
    canonical_module.fp8_simulate_qat = marker
    monkeypatch.setitem(sys.modules, "atomrl.lumen.ops.dsv4", canonical_module)
    spec = importlib.util.spec_from_file_location("_test_dsv4_ops_init", OPS_INIT_PATH)
    assert spec is not None and spec.loader is not None
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)
    assert ops.fp8_simulate_qat is marker
    assert ops.fp8_simulate_qat is marker


def test_compressor_has_no_module_level_qat_import():
    tree = ast.parse(COMPRESSOR_PATH.read_text(encoding="utf-8"))
    imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert all(
        not isinstance(node, ast.ImportFrom)
        or node.module != "atomrl.lumen.ops.dsv4.qat"
        for node in imports
    )


def test_dsv4_launchers_mount_explicit_aiter_tree():
    paths_source = (DSV4_EXAMPLES / "dsv4_paths.sh").read_text(encoding="utf-8")
    assert 'AITER_DIR="${AITER_DIR:-${WORKSPACE_ROOT}/aiter}"' in paths_source

    common = (DSV4_EXAMPLES / "dsv4_docker_common.sh").read_text(encoding="utf-8")
    assert "${AITER_DIR}:/workspace/aiter" in common
    assert "AITER_DIR=/workspace/aiter" in common

    launchers = (
        "run_dsv4_4layer_finetune.sh",
        "run_dsv4_4layer_pretrain.sh",
        "run_dsv4_4layer_profile.sh",
        "run_dsv4_flash_finetune.sh",
        "run_dsv4_flash_pretrain.sh",
    )
    for launcher in launchers:
        source = (DSV4_EXAMPLES / launcher).read_text(encoding="utf-8")
        assert "${AITER_DIR}:/workspace/aiter" in source, launcher
        if "AITER_DIR=/workspace/aiter" not in source:
            assert "dsv4_docker_common.sh" in source, launcher
            assert "dsv4_docker_append_kernel_env" in source, launcher

    two_node_launchers = (
        "launch_dsv4_2node.sh",
        "launch_dsv4_flash_finetune_2node.sh",
        "launch_dsv4_flash_pretrain_2node.sh",
    )
    for launcher in two_node_launchers:
        source = (DSV4_EXAMPLES / launcher).read_text(encoding="utf-8")
        assert '"AITER_DIR=${AITER_DIR}"' in source, launcher


def test_model_facade_preserves_triton_mhc_backend_compatibility():
    spec = importlib.util.spec_from_file_location(
        "_test_dsv4_ops_compat", OPS_INIT_PATH
    )
    assert spec is not None and spec.loader is not None
    ops = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ops)

    assert ops.get_mhc_backend() == "triton"
    assert ops.configure_mhc_backend() == "triton"
    assert ops.configure_mhc_backend("triton") == "triton"
    assert ops.log_mhc_backend() == "triton"

    try:
        ops.configure_mhc_backend("tilelang")
    except ValueError as error:
        assert "AIter Triton" in str(error)
    else:
        raise AssertionError("TileLang MHC compatibility request should fail")


def test_container_preflight_requires_all_dsv4_mhc_apis():
    bootstrap = (DSV4_EXAMPLES / "bootstrap_env.sh").read_text(encoding="utf-8")
    setup = (DSV4_EXAMPLES / "setup_container_env.sh").read_text(encoding="utf-8")
    assert 'AITER_DIR="${AITER_DIR:-/workspace/aiter}"' in bootstrap
    assert "${AITER_DIR}" in bootstrap
    for symbol in ("mhc_pre_dsv4", "mhc_post_dsv4", "mhc_head_dsv4"):
        assert symbol in setup
    assert "missing required AIter DSV4 MHC APIs" in setup

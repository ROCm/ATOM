"""CPU-only checks for DeepSeek-V4 2-buffer FP8 PD transfer regions."""

import ast
from pathlib import Path
from types import SimpleNamespace


class _FakeTensor:
    def __init__(self, addr: int, numel: int, element_size: int):
        self._addr = addr
        self._numel = numel
        self._element_size = element_size

    def data_ptr(self) -> int:
        return self._addr

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


def _load_region_builder_class():
    """Load only the method under test, avoiding the module's GPU import chain."""
    source_path = (
        Path(__file__).parents[1]
        / "atom/model_ops/attentions/deepseek_v4_attn.py"
    )
    tree = ast.parse(source_path.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "DeepseekV4AttentionMetadataBuilder"
    )
    method = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_kv_transfer_tensors"
    )
    test_class = ast.ClassDef(
        name="_RegionBuilder",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))
    namespace = {}
    exec(compile(module, source_path, "exec"), namespace)
    return namespace["_RegionBuilder"]


_RegionBuilder = _load_region_builder_class()


def _make_builder(*, fp8: bool):
    block_size = 128
    head_dim = 16
    rope_head_dim = 4
    num_swa_blocks = 2
    num_blocks = 3
    k1_csa = block_size // 4
    k2_hca = block_size // 128
    swa_pages = num_swa_blocks * block_size
    ratios = [0, 4, 128]
    elem_classical = 1 if fp8 else 2

    unified = []
    rope = []
    for layer_id, ratio in enumerate(ratios):
        entries_per_block = k1_csa if ratio == 4 else k2_hca if ratio == 128 else 0
        pages = swa_pages + num_blocks * entries_per_block
        unified.append(
            _FakeTensor(10_000 * (layer_id + 1), pages * head_dim, elem_classical)
        )
        rope.append(
            _FakeTensor(100_000 + 10_000 * layer_id, pages * rope_head_dim, 2)
            if fp8
            else None
        )

    runner = SimpleNamespace(
        max_per_req_cache_slots=8,
        num_swa_blocks=num_swa_blocks,
        num_physical_kvcache_blocks=num_blocks,
        v4_unified_kv=unified,
        v4_unified_kv_rope=rope,
        v4_csa_idx_kv=[_FakeTensor(90_000, num_blocks * k1_csa * 8, 1)],
    )

    builder = object.__new__(_RegionBuilder)
    builder.model_runner = runner
    builder._kv_fp8 = fp8
    builder._classical_dtype = SimpleNamespace(itemsize=elem_classical)
    builder._swa_dtype = builder._classical_dtype
    builder._rope_dtype = SimpleNamespace(itemsize=2)
    builder.block_size = block_size
    builder.head_dim = head_dim
    builder.rope_head_dim = rope_head_dim
    builder.num_layers = len(ratios)
    builder.compress_ratios = ratios
    builder.k1_csa = k1_csa
    builder.k2_hca = k2_hca
    builder.csa_layers = [1]
    builder._aligned_index_dim = 8
    return builder


def test_fp8_pd_regions_cover_nope_and_rope_pools():
    builder = _make_builder(fp8=True)
    transfer = builder.get_kv_transfer_tensors()

    # CSA/HCA each contribute FP8-nope + BF16-rope regions; the CSA indexer
    # contributes one additional region.
    assert len(transfer.block_regions) == 5
    assert [region.unit_bytes for region in transfer.block_regions] == [
        32 * 16,
        32 * 4 * 2,
        1 * 16,
        1 * 4 * 2,
        32 * 8,
    ]

    swa_pages = 2 * 128
    csa_nope, csa_rope = transfer.block_regions[:2]
    assert csa_nope.base_addr == 20_000 + swa_pages * 16
    assert csa_nope.total_bytes == 3 * 32 * 16
    assert csa_rope.base_addr == 110_000 + swa_pages * 4 * 2
    assert csa_rope.total_bytes == 3 * 32 * 4 * 2

    # Every layer contributes two independently addressable SWA regions.
    assert len(transfer.swa_block_regions) == 6
    assert [region.unit_bytes for region in transfer.swa_block_regions] == [
        128 * 16,
        128 * 4 * 2,
    ] * 3


def test_bf16_pd_region_layout_is_unchanged():
    builder = _make_builder(fp8=False)
    transfer = builder.get_kv_transfer_tensors()

    assert len(transfer.block_regions) == 3
    assert [region.unit_bytes for region in transfer.block_regions] == [
        32 * 16 * 2,
        1 * 16 * 2,
        32 * 8,
    ]
    assert len(transfer.swa_block_regions) == 3
    assert all(
        region.unit_bytes == 128 * 16 * 2
        for region in transfer.swa_block_regions
    )

# SPDX-License-Identifier: MIT
import itertools
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from atom.config import get_hf_config
from atom.models.deepseek_v41.weights import (
    CheckpointReader,
    WeightSpec,
    build_weight_manifest,
    checkpoint_schema,
)

from .reference import FIXTURES


@pytest.fixture(scope="module")
def schema():
    return checkpoint_schema(get_hf_config(str(FIXTURES)))


def test_source_schema_and_explicit_owner_slices(schema):
    assert len(schema) == 96085
    assert "layers.3.attn.compressor.wkv.weight" not in schema
    assert "layers.24.attn.indexer.wk.weight" not in schema
    assert "layers.24.attn.indexer.wq_b.weight" in schema
    assert "mtp.0.markov_head.embed.weight" not in schema
    assert "mtp.2.markov_head.embed.weight" in schema
    manifest = build_weight_manifest(schema, tp_rank=7, tp_size=8, ep_rank=7, ep_size=8)
    entries = {e.source.name: e for e in manifest}
    assert len(entries) == len(schema)
    assert entries["layers.0.ffn.experts.0.w1.weight"].action == "skip"
    assert entries["layers.0.ffn.experts.383.w1.weight"].shape == (2304, 2560)
    assert entries["layers.0.ffn.shared_experts.w1.scale"].shape == (9, 160)
    assert entries["layers.0.ffn.shared_experts.w2.scale"].shape == (160, 9)
    assert entries["layers.0.attn.wq_b.scale"].start == 896
    assert entries["layers.2.attn.indexer.wq_b.weight"].axis is None
    assert entries["layers.1.engram.embed.weight"].action == "host"
    assert entries["layers.0.attn.wo_a.scale"].action == "dequant_scale"
    assert entries["mtp.2.norm.weight"].reason == "draft explicitly excluded"
    assert entries["vision.norm.weight"].reason == "vision explicitly excluded"
    tp_only = {
        e.source.name: e for e in build_weight_manifest(schema, tp_rank=7, tp_size=8)
    }
    assert tp_only["layers.0.ffn.experts.383.w2.weight"].shape == (5120, 144)
    assert tp_only["layers.0.ffn.experts.383.w2.scale"].shape == (5120, 9)
    with pytest.raises(ValueError, match="TP partition"):
        build_weight_manifest(schema, tp_size=16)


def test_published_headers_match_generated_schema(schema):
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for complete checkpoint validation")
    with CheckpointReader(directory, schema) as reader:
        assert len(reader.weight_map) == 96085
        assert not reader._handles  # Header validation never maps table payloads.


def _checkpoint(tmp_path, tensors):
    weight_map = {}
    for i, (name, tensor) in enumerate(tensors.items()):
        shard = f"shard{i}.safetensors"
        save_file({name: tensor}, tmp_path / shard)
        weight_map[name] = shard
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_split_shard_reads_native_bytes_unconverted(tmp_path):
    torch.manual_seed(57)
    weights = {
        "linear.weight": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "linear.scale": torch.tensor([[0.5, 0.25], [2.0, 4.0]]).to(
            torch.float8_e8m0fnu
        ),
        "wo_a.weight": torch.randn(64, 64).to(torch.float8_e4m3fn),
        "wo_a.scale": torch.tensor([[8.0, 0.25], [2.0, 1.0]]).to(torch.float8_e8m0fnu),
        "expert.weight": torch.randint(-128, 127, (64, 32), dtype=torch.int8),
        "expert.scale": torch.ones(64, 2).to(torch.float8_e8m0fnu),
    }
    specs = {}
    for name, tensor in weights.items():
        block = (1, 32) if name.startswith("expert") else (32, 32)
        dtype = (
            "F8_E8M0"
            if name.endswith("scale")
            else ("I8" if name.startswith("expert") else "F8_E4M3")
        )
        specs[name] = WeightSpec(
            name,
            tuple(tensor.shape),
            dtype,
            tp_axis=0,
            block=block,
            dequantize=name.startswith("wo_a"),
        )
    _checkpoint(tmp_path, weights)
    manifest = {
        entry.source.name: entry
        for entry in build_weight_manifest(specs, tp_rank=1, tp_size=2)
    }
    # Compared through uint8: the point is that a shard reaches the caller as
    # the checkpoint's own bytes, packed FP4 and e8m0 scales included, with no
    # dtype conversion on the way -- `copy_` between mismatched dtypes zeros
    # silently, so a conversion here would not announce itself later.
    with CheckpointReader(tmp_path, specs) as reader:
        for name in ("linear.weight", "wo_a.weight", "expert.weight", "wo_a.scale"):
            rows = 1 if name.endswith(".scale") else 32
            assert torch.equal(
                reader.read(manifest[name]).view(torch.uint8),
                weights[name][rows:].view(torch.uint8),
            ), name
    corrupted = {**specs, "absent.weight": WeightSpec("absent.weight", (1,), "BF16")}
    with pytest.raises(ValueError, match="Checkpoint schema mismatch"):
        CheckpointReader(tmp_path, corrupted)


def test_engram_provider_maps_independent_shards_and_only_gathers_rows(tmp_path):
    prefix = "layers.1.engram.embed"
    values = (
        torch.arange(256, dtype=torch.float32).reshape(4, 64).to(torch.float8_e4m3fn)
    )
    scales = torch.tensor([[1.0, 2.0], [0.25, 4.0], [2.0, 0.5], [0.25, 0.125]]).to(
        torch.float8_e8m0fnu
    )
    _checkpoint(tmp_path, {prefix + ".weight": values, prefix + ".scale": scales})
    specs = {
        prefix
        + ".weight": WeightSpec(
            prefix + ".weight", (4, 64), "F8_E4M3", scope="engram_table"
        ),
        prefix
        + ".scale": WeightSpec(
            prefix + ".scale", (4, 2), "F8_E8M0", scope="engram_table"
        ),
    }
    config = SimpleNamespace(
        engram_layer_ids=[1], engram_num_embeddings=[4], engram_head_dim=64
    )
    with CheckpointReader(tmp_path, specs) as reader:
        table = reader.engram_tables(config)[1]
        actual = table.gather(np.array([[3, 0]], dtype=np.int64))
        expected = values.float()[[3, 0]] * scales.float()[[3, 0]].repeat_interleave(
            32, 1
        )
        assert torch.equal(actual[0], expected)
        assert table._tensor.device.type == "cpu" and table.dtype == torch.float8_e4m3fn
        assert len(reader._handles) == 2
        with pytest.raises(ValueError, match="owned by host"):
            reader.read(build_weight_manifest(specs)[0])


def test_real_engram_native_rows(schema):
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for real Engram table probes")
    config = get_hf_config(directory)
    with CheckpointReader(directory, schema) as reader:
        tables = reader.engram_tables(config)
        for layer, table in tables.items():
            indices = np.array([0, 101, table.num_rows - 1], dtype=np.int64)
            actual = table.gather(indices)
            expected = []
            for row in indices:
                data = table._tensor[int(row) : int(row) + 1].float()
                scale = table._scale[int(row) : int(row) + 1].float()
                expected.append(
                    (data.reshape(1, 8, 32) * scale[:, :, None]).reshape(256)
                )
            assert torch.equal(actual, torch.stack(expected)), layer
            assert table._tensor.device.type == "cpu"


def test_packed_rules_claim_exactly_their_own_checkpoint_tensors(schema):
    """Every fusion rule is a substring match, so state which names it takes.

    `attn.wkv` and `attn.compressor.wkv` differ only in what sits between
    `attn.` and `.wkv`, and `attn.wq_a` differs from `attn.wq_b` in one
    character. The loader takes the first rule whose key is a substring and
    stops looking, so a rule that reaches one tensor too far does not fail --
    it loads that tensor into the wrong half of a fused parameter.
    """
    pytest.importorskip("aiter", reason="the model module builds AITER-backed layers")
    from atom.models.deepseek_v41.model import DeepseekV41ForCausalLM

    rules = DeepseekV41ForCausalLM.packed_modules_mapping
    claims = {key: {name for name in schema if key in name} for key in rules}
    for left, right in itertools.combinations(rules, 2):
        assert not claims[left] & claims[right], (left, right)
    for key, names in claims.items():
        assert names, key
        # The key must land on a whole module path, not inside one: every name
        # it claims has to end with the key plus one tensor suffix.
        for name in names:
            assert name.rsplit(".", 1)[0].endswith(key), (key, name)
    # `layers.{i}.attn.wq_a` and `attn.wkv` for the 40 backbone layers and the
    # 3 draft stages, weight and scale each.
    assert len(claims["attn.wq_a"]) == len(claims["attn.wkv"]) == 86
    # The two near misses, named rather than left to a set that came out empty:
    # the compressor's own BF16 `wkv` (no scale), and the query's second half.
    assert "layers.2.attn.compressor.wkv.weight" in schema
    assert "layers.2.attn.compressor.wkv.weight" not in claims["attn.wkv"]
    assert "layers.2.attn.wq_b.weight" in schema
    assert "layers.2.attn.wq_b.weight" not in claims["attn.wq_a"]

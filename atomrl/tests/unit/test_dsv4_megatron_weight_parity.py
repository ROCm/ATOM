import json

import torch
from safetensors.torch import save_file

from tests.integration.run_dsv4_megatron_weight_parity import compare_exports


def _write_checkpoint(path, tensors):
    path.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(path / shard))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {name: shard for name in tensors},
            }
        )
    )


def test_compare_exports_identifies_first_mismatched_weight(tmp_path) -> None:
    source = tmp_path / "source"
    exported = tmp_path / "exported"
    _write_checkpoint(
        source,
        {
            "layers.0.attn_norm.weight": torch.ones(4),
            "layers.0.ffn.gate.weight": torch.ones(2, 4),
        },
    )
    _write_checkpoint(
        exported,
        {
            "layers.0.attn_norm.weight": torch.ones(4),
            "layers.0.ffn.gate.weight": torch.full((2, 4), 2.0),
        },
    )

    report = compare_exports(source, exported)

    assert report["matched"] == 1
    assert report["mismatched"] == 1
    assert report["first_mismatch"]["name"] == "layers.0.ffn.gate.weight"
    assert report["first_mismatch"]["max_abs_diff"] == 1.0

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
import torch


def _load_module(name: str, relative_path: str):
    path = Path(__file__).parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_receiver = _load_module(
    "_atom_rdma_weight_receiver_test", "atom/rollout/rdma_weight_receiver.py"
)
_updater = _load_module("_atom_weight_updater_test", "atom/rollout/weight_updater.py")
_CMD_BUCKET = _receiver._CMD_BUCKET
_CMD_END = _receiver._CMD_END
_HEADER_WORDS = _receiver._HEADER_WORDS
RDMAWeightReceiverMixin = _receiver.RDMAWeightReceiverMixin
_decode_bucket = _receiver._decode_bucket
WeightUpdaterMixin = _updater.WeightUpdaterMixin


class _PackedProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 2))
        self.weight_scale = torch.nn.Parameter(torch.ones(1))

    @staticmethod
    def weight_loader(param, tensor, shard_id):
        row = {"q": 0, "k": 1}[shard_id]
        param.data[row].copy_(tensor.reshape(-1))


class _PackedModel(torch.nn.Module):
    packed_modules_mapping: ClassVar[dict] = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
    }

    def __init__(self):
        super().__init__()
        self.qkv_proj = _PackedProjection()


class _Runner(WeightUpdaterMixin):
    def __init__(self):
        self.model = _PackedModel()
        self.device = torch.device("cpu")
        self.rank = 0
        self.world_size = 1
        self.label = "test-runner"
        self.clear_count = 0

    def clear_kv_cache(self):
        self.clear_count += 1

    @staticmethod
    def _is_fp8_param(module, param):
        return isinstance(module, _PackedProjection)

    def _requantize_fp8_weight(self, module, param_name, param, tensor):
        param.data.copy_(tensor)
        module.weight_scale.data.fill_(2.0)
        return True

    def _invalidate_cudagraphs_after_weight_update(self):
        pass


def test_transaction_preserves_packed_shards_across_buckets():
    runner = _Runner()
    runner.begin_weight_update(7)
    with pytest.raises(RuntimeError, match="is in progress"):
        runner.assert_weight_update_ready()

    first = runner.apply_weight_bucket(
        [("q_proj.weight", torch.tensor([1.0, 2.0], dtype=torch.bfloat16))],
        payload_bytes=4,
    )
    assert first["loaded_internal"] == 0
    assert runner.clear_count == 0
    assert list(runner._packed_weight_accum) == ["qkv_proj.weight"]

    second = runner.apply_weight_bucket(
        [("k_proj.weight", torch.tensor([3.0, 4.0], dtype=torch.bfloat16))],
        payload_bytes=4,
    )
    assert second["loaded_internal"] == 2
    manifest = runner.commit_weight_update(7, verify_full_load=True)

    assert manifest["buckets"] == 2
    assert manifest["bytes"] == 8
    assert manifest["missing"] == []
    assert manifest["loaded_internal_names"] == [
        "qkv_proj.weight",
        "qkv_proj.weight_scale",
    ]
    assert runner.clear_count == 1
    torch.testing.assert_close(
        runner.model.qkv_proj.weight,
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_failed_reload_fences_serving_until_new_full_commit():
    runner = _Runner()
    runner.begin_weight_update(1)
    runner.apply_weight_bucket([("unknown.weight", torch.ones(1))])

    with pytest.raises(RuntimeError, match="incomplete ATOM RDMA weight reload"):
        runner.commit_weight_update(1, verify_full_load=True)
    with pytest.raises(RuntimeError, match="serving is fenced"):
        runner.assert_weight_update_ready()

    runner.begin_weight_update(2)
    runner.apply_weight_bucket(
        [
            ("q_proj.weight", torch.tensor([5.0, 6.0], dtype=torch.bfloat16)),
            ("k_proj.weight", torch.tensor([7.0, 8.0], dtype=torch.bfloat16)),
        ]
    )
    runner.commit_weight_update(2, verify_full_load=True)
    runner.assert_weight_update_ready()

    with pytest.raises(RuntimeError, match="must increase"):
        runner.begin_weight_update(2)


def test_bf16_packed_parameter_requires_every_declared_shard():
    runner = _Runner()
    runner._is_fp8_param = lambda module, param: False
    runner.begin_weight_update(1)
    runner.apply_weight_bucket([("q_proj.weight", torch.tensor([1.0, 2.0]))])

    with pytest.raises(RuntimeError, match="incomplete packed shards"):
        runner.commit_weight_update(1, verify_full_load=True)
    with pytest.raises(RuntimeError, match="serving is fenced"):
        runner.assert_weight_update_ready()


def test_vllm_v1_bucket_metadata_decodes_without_translation():
    assert (_CMD_END, _CMD_BUCKET, _HEADER_WORDS) == (0, 1, 4)
    values = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
    payload = values.contiguous().view(torch.uint8).reshape(-1)
    entries = [
        {
            "name": "model.embed_tokens.weight",
            "shape": [1, 2],
            "dtype": "bfloat16",
            "offset": 0,
            "nbytes": payload.numel(),
        }
    ]
    metadata = torch.tensor(
        list(json.dumps(entries, separators=(",", ":")).encode("utf-8")),
        dtype=torch.uint8,
    )

    decoded = _decode_bucket(metadata, payload)
    assert decoded[0][0] == "model.embed_tokens.weight"
    torch.testing.assert_close(decoded[0][1], values)


def test_dp_tp_rank_is_flattened_inside_server_base_rank(monkeypatch):
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return object()

    lumenrl = types.ModuleType("lumenrl")
    utils = types.ModuleType("lumenrl.utils")
    independent = types.ModuleType("lumenrl.utils.independent_process_group")
    independent.init_independent_process_group = fake_init
    monkeypatch.setitem(sys.modules, "lumenrl", lumenrl)
    monkeypatch.setitem(sys.modules, "lumenrl.utils", utils)
    monkeypatch.setitem(
        sys.modules, "lumenrl.utils.independent_process_group", independent
    )

    runner = type("RankRunner", (RDMAWeightReceiverMixin,), {})()
    runner.config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(data_parallel_rank_local=1)
    )
    runner.world_size = 2
    runner.rank = 1
    runner.label = "rank-runner"

    assert runner.init_rdma_weight_group("10.0.0.1", 1234, 1, 5, "weights")
    assert calls[0]["rank"] == 4
    assert calls[0]["world_size"] == 5

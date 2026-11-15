# SPDX-License-Identifier: MIT
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from atom.spec_decode.calibration import calibration_identity, validate_calibration


def profile():
    return {
        "version": 1,
        "identity": {"gpu": "test", "model": "test"},
        "draft_width": 5,
        "max_num_seqs": 4,
        "sts_temperatures": [1.0] * 5,
        "sps_table": [100 / (i + 1) for i in range(25)],
    }


def test_profile_covers_smaller_runtime_batches():
    data = profile()
    validate_calibration(data, data["identity"], width=5, max_batch=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("identity", {"gpu": "different"}),
        ("draft_width", 4),
        ("max_num_seqs", 2),
        ("sts_temperatures", [float("nan")] * 5),
        ("sts_temperatures", [-1.0] * 5),
        ("sps_table", [0.0] * 25),
        ("sps_table", [1.0] * 12),
        ("sps_table", list(range(1, 26))),
    ],
)
def test_incompatible_or_invalid_profile_is_rejected(field, value):
    data = profile()
    identity = deepcopy(data["identity"])
    data[field] = value
    with pytest.raises(ValueError):
        validate_calibration(data, identity, width=5, max_batch=4)


def identity_of(tmp_path, monkeypatch, **kw):
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "gpu")
    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(model_type="m"),
        tensor_parallel_size=4,
        enable_dp_attention=False,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        enforce_eager=False,
        # as `Config` resolves it: the target's, unless the draft owns a pool
        speculative_config=SimpleNamespace(draft_kv_dtype="fp8"),
    )
    for key, value in kw.items():
        setattr(config, key, value)
    return calibration_identity(config, "cpu")


def test_mxfp4_kv_does_not_share_an_fp8_identity(tmp_path, monkeypatch):
    fp8 = identity_of(tmp_path, monkeypatch)
    assert "draft_attention_dtype" not in fp8
    kv_mx = identity_of(tmp_path, monkeypatch, kv_cache_dtype="fp4")
    index_mx = identity_of(tmp_path, monkeypatch, index_cache_dtype="fp4")
    assert len({str(fp8), str(kv_mx), str(index_mx)}) == 3

    data = {**profile(), "identity": fp8}
    with pytest.raises(ValueError):
        validate_calibration(data, kv_mx, width=5, max_batch=4)


@pytest.fixture
def a8w4_runtime(monkeypatch):
    """Import the real boundary, stubbing only the external AITER capability API."""
    query = Mock(return_value="opus")
    api = ModuleType("aiter.ops.mla_sparse_prefill")
    api.get_mla_sparse_prefill_v41_backend = query
    name = "atom.model_ops.attentions.deepseek_v41.a8w4_attention"
    path = Path(__file__).resolve().parents[1] / (name.replace(".", "/") + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, api.__name__, api)
        spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", "off")
    return query


@pytest.mark.parametrize("target_dtype", ["bf16", "fp8", "fp4"])
def test_v41_calibration_requires_matching_draft_precision(
    tmp_path, monkeypatch, target_dtype, a8w4_runtime
):
    def identity(draft_dtype):
        return identity_of(
            tmp_path,
            monkeypatch,
            hf_config=SimpleNamespace(model_type="deepseek_v41_text"),
            kv_cache_dtype=target_dtype,
            speculative_config=SimpleNamespace(draft_kv_dtype=target_dtype),
            dspark=SimpleNamespace(draft_kv_dtype=draft_dtype),
        )

    bf16, fp8 = identity("auto"), identity("fp8")
    assert "draft_attention_dtype" not in bf16
    assert fp8["draft_attention_dtype"] == "fp8"
    for current, other in ((bf16, fp8), (fp8, bf16)):
        validate_calibration(
            {**profile(), "identity": current}, current, width=5, max_batch=4
        )
        with pytest.raises(ValueError, match="model/hardware/runtime"):
            validate_calibration(
                {**profile(), "identity": other}, current, width=5, max_batch=4
            )
    a8w4_runtime.assert_not_called()


@pytest.mark.parametrize("target_dtype", ["bf16", "fp8", "fp4"])
def test_pre_pr_bf16_draft_profile_still_matches(
    tmp_path, monkeypatch, target_dtype, a8w4_runtime
):
    # This is the pre-PR schema, with no new optional identity fields.
    file_hash = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
    legacy = {
        "model_files": dict.fromkeys(
            ("config.json", "model.safetensors.index.json"), file_hash
        ),
        "model_type": "deepseek_v41_text",
        "tensor_parallel_size": 4,
        "kv_cache_dtype": target_dtype,
        "index_cache_dtype": "fp8",
        "graph": True,
        "gpu_name": "gpu",
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
    }
    current = identity_of(
        tmp_path,
        monkeypatch,
        hf_config=SimpleNamespace(model_type="deepseek_v41_text"),
        kv_cache_dtype=target_dtype,
        speculative_config=SimpleNamespace(draft_kv_dtype=target_dtype),
        dspark=SimpleNamespace(draft_kv_dtype="auto"),
    )
    assert current == legacy
    validate_calibration(
        {**profile(), "identity": legacy}, current, width=5, max_batch=4
    )
    a8w4_runtime.assert_not_called()


@pytest.mark.parametrize("dp", [False, True])
def test_fp4_calibration_distinguishes_actual_target_backend(
    tmp_path, monkeypatch, a8w4_runtime, dp
):
    def identity():
        return identity_of(
            tmp_path,
            monkeypatch,
            hf_config=SimpleNamespace(
                model_type="deepseek_v41_text", num_attention_heads=64, head_dim=512
            ),
            enable_dp_attention=dp,
            kv_cache_dtype="fp4",
            speculative_config=SimpleNamespace(draft_kv_dtype="fp4"),
            dspark=SimpleNamespace(draft_kv_dtype="auto"),
        )

    adapter = identity()
    a8w4_runtime.assert_not_called()
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", "auto")
    native = identity()
    assert native == {**adapter, "target_attention_dtype": "a8w4"}
    args, kwargs = a8w4_runtime.call_args
    assert args == (torch.device("cpu"), (1, 64 if dp else 16, 512))
    assert kwargs["unified_kv_dtype"] == torch.uint8
    assert kwargs["layout"] == "contiguous"
    for measured, deployed in ((adapter, native), (native, adapter)):
        with pytest.raises(ValueError, match="model/hardware/runtime"):
            validate_calibration(
                {**profile(), "identity": measured}, deployed, width=5, max_batch=4
            )
    # An AITER build with no eligible A8W4 backend still uses the adapter.
    a8w4_runtime.return_value = None
    fallback = identity()
    assert fallback == adapter
    validate_calibration(
        {**profile(), "identity": adapter}, fallback, width=5, max_batch=4
    )

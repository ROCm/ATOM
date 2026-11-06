# SPDX-License-Identifier: MIT
"""Check fused-chain dispatch and launch bounds without Triton or a GPU."""

from dataclasses import dataclass, replace

import pytest

torch = pytest.importorskip("torch")

from atom.model_ops import dspark_prob_markov_chain as chain


@dataclass
class _Metadata:
    """GPU tensor metadata for a gate that never reads tensor contents."""

    shape: tuple
    dtype: object = torch.bfloat16
    device: object = torch.device("cuda:0")
    strides: tuple | None = None

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    def stride(self, dim=None):
        strides = self.strides
        if strides is None:
            strides = torch.empty(self.shape, device="meta").stride()
        return strides if dim is None else strides[dim]


@pytest.fixture
def metadata(monkeypatch):
    monkeypatch.setattr(chain, "_is_gfx1250", lambda _: True)
    return {
        "base": _Metadata((24, 5, 129280)),
        "w1": _Metadata((129280, 256)),
        "w2": _Metadata((129280, 256)),
        "keys": _Metadata((7, 64), torch.uint64),
    }


def _can_use(inputs):
    return chain.can_use_prob_markov_chain(
        inputs["base"], inputs["w1"], inputs["w2"], inputs["keys"]
    )


@pytest.mark.parametrize("b", [1, 3, 16, 24, 32])
def test_supported_metadata_including_padded_workspace(metadata, b):
    metadata["base"] = replace(metadata["base"], shape=(b, 5, 129280))
    assert _can_use(metadata)


@pytest.mark.parametrize(
    "name,changes",
    [
        ("base", {"shape": (0, 5, 129280)}),
        ("base", {"shape": (33, 5, 129280)}),
        ("base", {"shape": (24, 2, 129280)}),
        ("base", {"shape": (24, 5, 128)}),
        ("base", {"shape": (24, 129280)}),
        ("base", {"dtype": torch.float32}),
        ("base", {"device": torch.device("cpu")}),
        ("base", {"strides": (129280 * 10, 129280 * 2, 2)}),
        ("base", {"strides": (2**31, 129280, 1)}),
        ("w1", {"shape": (129280, 128)}),
        ("w2", {"dtype": torch.float16}),
        ("w1", {"device": torch.device("cuda:1")}),
        ("w2", {"device": torch.device("cpu")}),
        ("w1", {"strides": (512, 2)}),
        ("w2", {"strides": (2**31, 1)}),
        ("keys", {"shape": (64,)}),
        ("keys", {"shape": (4, 64)}),
        ("keys", {"shape": (7, 23)}),
        ("keys", {"dtype": torch.int64}),
        ("keys", {"device": torch.device("cuda:1")}),
        ("keys", {"strides": (128, 2)}),
        ("keys", {"strides": (0, 1)}),
        ("keys", {"strides": (23, 1)}),
        ("keys", {"strides": (2**31, 1)}),
    ],
)
def test_unsupported_metadata_falls_back_before_device_probe(
    metadata, monkeypatch, name, changes
):
    metadata[name] = replace(metadata[name], **changes)

    def unexpected_probe(_):
        pytest.fail("invalid metadata should not query the GPU")

    monkeypatch.setattr(chain, "_is_gfx1250", unexpected_probe)
    assert not _can_use(metadata)


def test_missing_workspace_and_other_arch_fall_back(metadata, monkeypatch):
    monkeypatch.setattr(chain, "_is_gfx1250", lambda _: False)
    assert not _can_use(metadata)
    metadata["keys"] = None
    assert not _can_use(metadata)


@pytest.mark.parametrize("last_offset,expected", [(2**31 - 1, True), (2**31, False)])
def test_int32_element_offset_boundary(last_offset, expected):
    tensor = torch.empty_strided((2, 2), (last_offset - 1, 1), device="meta")
    assert chain._fits_int32_offsets(tensor) is expected


@pytest.mark.parametrize(
    "name,changes",
    [
        ("base_logits", {"dtype": torch.float32}),
        ("prev_ids", {"device": "cpu"}),
        ("temps", {"stride": (2,)}),
        ("keys", {"stride": (1, 1)}),
        ("q_out", {"stride": (2**31, 32, 1)}),
        ("saved_ids", {"device": "cpu"}),
    ],
)
def test_invalid_launch_buffers_are_rejected_before_gpu_import(name, changes):
    shapes = {
        "base_logits": ((2, 5, 32), torch.bfloat16),
        "prev_ids": ((2,), torch.int32),
        "markov_w1": ((32, 16), torch.bfloat16),
        "markov_w2": ((32, 16), torch.bfloat16),
        "out": ((2, 5), torch.int64),
        "q_out": ((2, 5, 32), torch.float32),
        "temps": ((2,), torch.float32),
        "seed": ((1,), torch.int64),
        "embed_out": ((2, 5, 16), torch.bfloat16),
        "keys": ((5, 2), torch.uint64),
        "saved_ids": ((2, 5), torch.int32),
    }
    tensors = {
        key: torch.empty(shape, dtype=dtype, device="meta")
        for key, (shape, dtype) in shapes.items()
    }
    shape, dtype = shapes[name]
    options = {"dtype": dtype, "device": "meta"}
    options.update(changes)
    stride = options.pop("stride", None)
    tensors[name] = (
        torch.empty(shape, **options)
        if stride is None
        else torch.empty_strided(shape, stride, **options)
    )
    with pytest.raises(AssertionError):
        chain.prob_markov_sample_chain(**tensors)

# SPDX-License-Identifier: MIT
# Tests for the shuffled MLA KV layout switch (ATOM_MLA_KV_SHUFFLED).

from types import SimpleNamespace

import pytest

from atom.utils import envs

_FLAGS = (
    "ATOM_MLA_KV_SHUFFLED",
    "ATOM_USE_TRITON_MLA",
    "ATOM_USE_TRITON_MLA_SHUFFLE_KV",
)


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"ATOM_MLA_KV_SHUFFLED": "1"}, True),
        ({"ATOM_USE_TRITON_MLA": "1", "ATOM_USE_TRITON_MLA_SHUFFLE_KV": "1"}, True),
        ({"ATOM_USE_TRITON_MLA_SHUFFLE_KV": "1"}, False),
        ({"ATOM_USE_TRITON_MLA": "1"}, False),
    ],
)
def test_kv_shuffled_flag_mapping(monkeypatch, env, expected):
    for name in _FLAGS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert envs.ATOM_MLA_KV_SHUFFLED is expected


def _config(enable_tbo=False, prefix=False, chunked=False, spec=None):
    return SimpleNamespace(
        enable_tbo=enable_tbo,
        enable_prefix_caching=prefix,
        enable_chunked_prefill=chunked,
        speculative_config=spec,
    )


_SPEC = SimpleNamespace(num_speculative_tokens=1)


def _check(config, triton=False, sparse=False, dcp=1, page=1):
    pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
    pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")
    from atom.model_ops.attention_mla import check_kv_shuffled_supported

    check_kv_shuffled_supported(
        config,
        use_triton_mla=triton,
        is_sparse_mla=sparse,
        dcp_world_size=dcp,
        mla_page_size=page,
    )


@pytest.mark.parametrize(
    "config, triton",
    [
        (_config(), False),
        (_config(enable_tbo=True), False),
        (_config(prefix=True, chunked=True), False),
        (_config(spec=_SPEC), True),
    ],
)
def test_kv_shuffled_supported(config, triton):
    _check(config, triton=triton)


@pytest.mark.parametrize(
    "kwargs, config, item",
    [
        ({"sparse": True}, _config(), "sparse MLA"),
        ({"dcp": 2}, _config(), "DCP>1"),
        ({"page": 64}, _config(), "ATOM_MLA_PAGE_SIZE>1"),
        ({}, _config(enable_tbo=True, prefix=True), "TBO with prefix caching"),
        ({}, _config(enable_tbo=True, chunked=True), "TBO with prefix caching"),
        ({}, _config(spec=_SPEC), "speculative decoding"),
    ],
)
def test_kv_shuffled_rejects(kwargs, config, item):
    with pytest.raises(ValueError, match=item):
        _check(config, **kwargs)


def test_kv_shuffled_rejects_lists_every_item():
    with pytest.raises(ValueError) as err:
        _check(
            _config(enable_tbo=True, prefix=True, spec=_SPEC),
            sparse=True,
            dcp=2,
            page=64,
        )
    for item in (
        "sparse MLA",
        "DCP>1",
        "ATOM_MLA_PAGE_SIZE>1",
        "TBO with prefix caching or chunked prefill",
        "speculative decoding without ATOM_USE_TRITON_MLA",
    ):
        assert item in str(err.value)

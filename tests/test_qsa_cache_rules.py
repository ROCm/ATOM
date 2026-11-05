# SPDX-License-Identifier: MIT

"""Qwen3.8's QSA cache rules: which caches a server keeps, and which KV
transfer may move them. No kernels, so these run without a GPU."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.qwen4_exp.qsa_cache_rules import (
    QSA_DRAFT_ROWS,
    QSA_ROWS,
    check_kv_transfer,
    qsa_cache_formats,
    qsa_page_segments,
    qsa_paged_layout_id,
    qsa_row_space,
)


@pytest.mark.parametrize(
    ("kv", "index", "expected"),
    [
        ("bf16", None, ("bf16", "bf16")),
        ("auto", None, ("bf16", "bf16")),
        ("fp8", None, ("fp8", "fp8")),
        ("fp8", "bf16", ("fp8", "bf16")),
        ("fp8", "fp4", ("fp8", "fp4")),
        ("bf16", "fp8", ("bf16", "fp8")),
    ],
)
def test_cache_formats_follow_the_kv_dtype(kv, index, expected):
    assert (
        qsa_cache_formats(SimpleNamespace(kv_cache_dtype=kv, index_cache_dtype=index))
        == expected
    )


@pytest.mark.parametrize(("kv", "index"), [("int8", None), ("fp4", None)])
def test_cache_formats_reject_unserved(kv, index):
    with pytest.raises(ValueError, match="Qwen QSA serves"):
        qsa_cache_formats(SimpleNamespace(kv_cache_dtype=kv, index_cache_dtype=index))


def _check(kv_transfer_config, kv_cache_dtype="bf16", index_cache_dtype=None):
    check_kv_transfer(
        SimpleNamespace(
            kv_transfer_config=kv_transfer_config,
            kv_cache_dtype=kv_cache_dtype,
            index_cache_dtype=index_cache_dtype,
        )
    )


_PD = "disaggregated.*kv_connector={!r}"
_NO_IMAGE = "kv_connector={!r} does not read"


@pytest.mark.parametrize(
    "kv_transfer_config, backend, error",
    [
        ({"kv_connector": "mooncake", "kv_role": "kv_producer"}, "mooncake", _PD),
        ({"kv_connector": "moriio", "kv_role": "kv_consumer"}, "moriio", _PD),
        # No connector named: the factory defaults to moriio.
        ({"kv_role": "kv_producer"}, "moriio", _PD),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                    {"kv_connector": " MOONCAKE ", "kv_role": "kv_producer"},
                ],
            },
            "mooncake",
            _PD,
        ),
        (
            {
                "kv_connector": "multi",
                "connectors": [
                    {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                    {"kv_connector": "lmcache_offload", "kv_role": "offload"},
                ],
            },
            "mooncake",
            _PD,
        ),
        # Checked by the canonical name the connector is built from.
        ({"kv_connector": "Mooncake", "kv_role": "kv_producer"}, "mooncake", _PD),
        ({"kv_connector": " moriio", "kv_role": "kv_consumer"}, "moriio", _PD),
        ({"kv_connector": "MoRIIO", "kv_role": "kv_consumer"}, "moriio", _PD),
        (
            {"kv_connector": "Mooncake_Store", "kv_role": "offload"},
            "mooncake_store",
            _NO_IMAGE,
        ),
    ],
)
def test_kv_transfers_that_drop_the_state_are_refused(
    kv_transfer_config, backend, error
):
    """A P/D peer would get the QSA pages and continue from a zero GDN/PLE state."""
    with pytest.raises(ValueError, match=error.format(backend)):
        _check(kv_transfer_config)


@pytest.mark.parametrize("reads", [True, False])
def test_qwen38_runs_any_tier_that_declares_it_reads_page_images(monkeypatch, reads):
    """The knowledge lives on the registration, not in a Qwen3.8 list."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    name = "page_image_tier_under_test"
    monkeypatch.setitem(KVConnectorFactory._registry, name, {})
    monkeypatch.setitem(KVConnectorFactory._aliases, name, name)
    monkeypatch.setitem(KVConnectorFactory._offload, name, True)
    monkeypatch.setitem(KVConnectorFactory._reads_page_images, name, reads)
    kv_transfer_config = {"kv_connector": name, "kv_role": "offload"}
    if reads:
        _check(kv_transfer_config)
    else:
        with pytest.raises(ValueError, match=f"kv_connector={name!r} does not"):
            _check(kv_transfer_config)


_IN_PROCESS = {"kv_connector": "LMCacheOffloadConnector", "kv_role": "offload"}
_MP = {"kv_connector": "lmcache_mp", "kv_role": "offload"}


@pytest.mark.parametrize(
    "kv_cache_dtype, index_cache_dtype",
    [
        ("fp8", None),  # the index cache follows the KV cache
        ("fp8", "fp8"),
        ("fp8", "bf16"),
        ("bf16", "fp8"),
        ("bf16", "fp4"),
        ("auto", "fp4"),
    ],
)
@pytest.mark.parametrize(
    "kv_transfer_config",
    [
        _IN_PROCESS,
        {"kv_connector": "multi", "connectors": [_MP, _IN_PROCESS]},
    ],
)
def test_qwen38_runs_in_process_offload_of_quantized_caches(
    kv_transfer_config, kv_cache_dtype, index_cache_dtype
):
    """The in-process codec carries the QSA scale planes (`k_scale`, `v_scale`,
    `extra_planes`)."""
    _check(kv_transfer_config, kv_cache_dtype, index_cache_dtype)


@pytest.mark.parametrize(
    "kv_transfer_config, kv_cache_dtype, index_cache_dtype",
    [
        (_IN_PROCESS, "bf16", "bf16"),
        (_IN_PROCESS, "bf16", None),
        (_IN_PROCESS, "auto", "bfloat16"),
        (_IN_PROCESS, None, None),
        (_MP, "fp8", None),
        (_MP, "fp8", "fp4"),
        (_MP, "bf16", "fp8"),
        (None, "fp8", "fp8"),
    ],
)
def test_qwen38_keeps_bf16_in_process_and_every_lmcache_mp_cache(
    kv_transfer_config, kv_cache_dtype, index_cache_dtype
):
    _check(kv_transfer_config, kv_cache_dtype, index_cache_dtype)


_MTP = SimpleNamespace(method="mtp")


@pytest.mark.parametrize(
    "kv_cache_dtype, index_cache_dtype, draft_env, spec, draft",
    [
        ("fp8", None, "1", _MTP, True),
        ("bf16", "fp4", "1", _MTP, True),  # a quantized index cache alone
        ("fp8", "fp8", "0", _MTP, False),
        ("bf16", None, "1", _MTP, False),  # nothing to keep in BF16
        ("auto", "bfloat16", "1", _MTP, False),
        ("fp8", "fp8", "1", SimpleNamespace(method="eagle3"), False),
        ("fp8", "fp8", "1", None, False),
    ],
)
def test_the_draft_rows_and_the_page_namespace_read_one_rule(
    monkeypatch, kv_cache_dtype, index_cache_dtype, draft_env, spec, draft
):
    monkeypatch.setenv("ATOM_QSA_DRAFT_KV_BF16", draft_env)
    config = SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        index_cache_dtype=index_cache_dtype,
        speculative_config=spec,
    )
    assert qsa_page_segments(config)["draft_kv"] == ("bf16" if draft else "target")
    if spec is _MTP:
        assert qsa_row_space(config, 40, 40) == (QSA_DRAFT_ROWS if draft else QSA_ROWS)
    assert qsa_row_space(config, 39, 40) == QSA_ROWS


_KV = ((4, 2, 8), torch.bfloat16)
_PLE = [("conv", (16, 10), torch.bfloat16), ("ngram", (3,), torch.int64)]


def test_the_layout_id_spells_the_storage_the_page_fields_and_the_ple_planes():
    layout = qsa_paged_layout_id(
        "gdn-paged-state-v1:base", "vk", [("k", *_KV), ("v", *_KV)], _PLE
    )
    kv = "(4,2,8),torch.bfloat16"
    assert layout == (
        f"gdn-paged-state-v1:base:ssm_storage=vk:page=k={kv};v={kv}"
        ":ple_conv=(16,10),torch.bfloat16:ple_ngram=(3,),torch.int64"
    )
    assert qsa_paged_layout_id("base", "kv", [("k", *_KV)], []) == (
        f"base:ssm_storage=kv:page=k={kv}"
    )


@pytest.mark.parametrize(
    "other",
    [
        [("k", (4, 2, 4), torch.bfloat16)],  # same name, other shape
        [("k", (4, 2, 8), torch.uint8)],  # same name, other dtype
        [("v", *_KV)],
    ],
)
def test_the_layout_id_tells_page_fields_apart(other):
    assert qsa_paged_layout_id("base", "kv", [("k", *_KV)], _PLE) != (
        qsa_paged_layout_id("base", "kv", other, _PLE)
    )


def test_the_layout_id_tells_the_ssm_storage_and_the_ple_planes_apart():
    ids = {
        qsa_paged_layout_id("base", storage, [("k", *_KV)], planes)
        for storage in ("kv", "vk")
        for planes in (_PLE, _PLE[:1], _PLE[::-1], [])
    }
    assert len(ids) == 8

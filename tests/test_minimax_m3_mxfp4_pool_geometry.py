# SPDX-License-Identifier: MIT
"""MiniMax-M3 ``--kv-cache-dtype fp4``: every paged attention layer is an MXFP4 pool.

Pinned here, CPU only: the one rule that picks the layout (`mxfp4_kv_layout`),
and the startup refusals of the
FP4 draft flavors whose MXFP4 path awaits GPU validation. fp8 / bf16 and non-M3 keep
the dtype-wide cache. This file needs no aiter; the builder and pool half is
`test_minimax_m3_mxfp4_pool_builder.py`.
"""

import sys
import types

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import Mxfp4KvLayout
from atom.models.minimax_m3.config import mxfp4_kv_layout, validate_runtime_config

M3_HF = types.SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"])


def cfg(kv="fp4", hf=M3_HF, **kw):
    return types.SimpleNamespace(kv_cache_dtype=kv, hf_config=hf, **kw)


# ---- the rule ------------------------------------------------------------------


def test_every_m3_fp4_layer_gets_the_mxfp4_layout():
    assert mxfp4_kv_layout(cfg("fp4")) == Mxfp4KvLayout()


@pytest.mark.parametrize("kv", ["fp8", "bf16"])
def test_fp8_and_bf16_keep_the_dtype_wide_cache(kv):
    assert mxfp4_kv_layout(cfg(kv)) is None


def test_fp4_on_another_model_or_a_bare_config_is_not_ours():
    other = types.SimpleNamespace(architectures=["DeepseekV41ForCausalLM"])
    assert mxfp4_kv_layout(cfg("fp4", hf=other)) is None
    assert mxfp4_kv_layout(types.SimpleNamespace(kv_cache_dtype="fp4")) is None


# ---- startup refusals of FP4 drafts without an MXFP4 path ----------------------


@pytest.fixture
def gfx950(monkeypatch):
    mod = types.ModuleType("aiter.jit.utils.chip_info")
    mod.get_gfx_runtime = lambda: "gfx950"
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", mod)


class _Spec:
    def __init__(self, method="", dspark=False):
        self.method, self._dspark = method, dspark

    def use_dspark(self):
        return self._dspark


def validate(kv="fp4", spec=None, index="fp4"):
    validate_runtime_config(
        cfg(
            kv,
            index_cache_dtype=index,
            kv_cache_block_size=128,
            torch_dtype=torch.bfloat16,
            speculative_config=spec,
        )
    )


def test_a_target_without_a_draft_passes(gfx950):
    validate()


def test_an_eagle3_draft_is_refused_under_fp4(gfx950):
    with pytest.raises(ValueError, match="EAGLE3"):
        validate(spec=_Spec("eagle3"))


def test_a_dspark_draft_is_refused_under_fp4(gfx950):
    with pytest.raises(ValueError, match="DSpark"):
        validate(spec=_Spec("dspark", dspark=True))
    # also the flavor that only the checkpoint's own field names
    with pytest.raises(ValueError, match="DSpark"):
        validate(spec=_Spec("mtp", dspark=True))


def test_other_drafts_and_other_dtypes_are_not_caught(gfx950):
    validate(spec=_Spec("mtp"))
    # fp8 KV with an MXFP4 index never reaches the fp4 draft paths
    validate(kv="fp8", spec=_Spec("eagle3"), index="fp4")

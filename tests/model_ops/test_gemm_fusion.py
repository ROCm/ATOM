# SPDX-License-Identifier: MIT
"""CPU checks of the fused SP QKV runtime against AITER's shipped profile and
its call-id and fail-stop contracts."""

from types import SimpleNamespace

import pytest
import torch

gf = pytest.importorskip("aiter.gemm_fusion")

from aiter.ops.gemm_fusion.sp_qkv import specs

from atom.model_ops.gemm_fusion import runtime as gfr
from atom.model_ops.gemm_fusion.minimax_m3 import _spec

PROFILE = "minimax_m3_sp4_gfx950_v1"
CTX = gf.MatchCtx(arch="gfx950", cu=256, world=4, m=0)


def _runtime(profile=None):
    return gfr.GemmFusionRuntime(gf, profile, CTX, None, 4, 128)


def _module(mode):
    """MiniMax-M3 attention at -tp 1 -sp 4."""
    impl = SimpleNamespace(
        head_dim=128,
        num_heads=16,
        num_kv_heads=1,
        rotary_emb=SimpleNamespace(rotary_dim=64),
        kv_cache_dtype="fp8",
    )
    module = SimpleNamespace(
        head_dim=128,
        num_heads=64,
        num_kv_heads=4,
        qkv_proj=SimpleNamespace(
            output_size=9216 if mode == "dense" else 9856, input_size=6144
        ),
        q_norm=SimpleNamespace(variance_epsilon=1e-6),
        attn=SimpleNamespace(impl=impl),
    )
    if mode != "dense":
        module.idx_head_dim, module.num_idx_heads = 128, 4
    return module


@pytest.mark.parametrize("mode", specs.MODES)
def test_sites_match_the_profile_and_every_bucket_is_prepared(mode):
    """A drifted spec matches no row and leaves the layer silently native; a
    bucket outside the prepared set would never fuse."""
    spec = _spec(gf, _module(mode), mode, 4, 128 if mode == "dense" else 16)
    assert spec.signature() == specs.sp_qkv_spec(mode).signature()

    profile = gf.load_profile(PROFILE)
    rt = _runtime(profile)
    prepared = gfr._bucket_ms(profile, spec, CTX, max_m=2048)
    assert prepared[0] == 1 and prepared[-1] == 2048
    buckets = {m: rt.bucket(spec, m) for m in range(1, 2049)}
    assert {b for b in buckets.values() if b is not None} <= set(prepared)
    # R16 pad-M: below a large bucket's floor the call stays native.
    assert [buckets[m] for m in (3, 256, 300, 385, 386, 513)] == [
        4,
        256,
        None,
        None,
        512,
        768,
    ]


def test_large_plan_tags_are_distinct_in_one_forward():
    """A large plan tags its handshakes with only the low bits of the call
    id; two layers sharing them in one forward read each other's rows."""
    rt = _runtime()
    rt.bucket = lambda spec, m: m
    plan = SimpleNamespace(vid="sp_qkv.large.v1", n_stages=1, src="r18")
    rt.sites = [
        SimpleNamespace(
            name=f"model.layers.{i}.self_attn.qkv", spec=None, plans={512: plan}
        )
        for i in range(62)
    ]
    ids = rt._resolve_call_ids(4 * 512)
    mask = (1 << specs.large_tag_bits("r18")) - 1
    tags = {i & mask for i in ids.values()}
    assert len(ids) == len(tags) == 62 and 0 not in tags
    assert all(0 < i < 1 << 31 for i in ids.values())
    # Every rank derives the same ids, whatever order it found the layers in.
    rt.sites.reverse()
    assert rt._resolve_call_ids(4 * 512) == ids


def test_a_plan_any_rank_cannot_bind_is_dropped_on_every_rank(monkeypatch):
    prepared = []

    def plan():
        return SimpleNamespace(
            n_stages=1, prepare=lambda x, **_: prepared.append(x.shape[0])
        )

    shared = {512: plan(), 768: plan()}
    lin = SimpleNamespace(weight=torch.empty(0), input_size=8, weight_scale=None)
    rt = _runtime()
    rt.sites = [
        SimpleNamespace(name=f"s{i}", linear=lin, plans=dict(shared)) for i in range(2)
    ]
    rt.res = SimpleNamespace(reset=lambda: None, step_tensor=lambda: "step")
    # A peer failed to bind the 768 bucket.
    peer = [0, 1, 0, 1]
    monkeypatch.setattr(
        gfr, "group_max_each", lambda codes, group: list(map(max, codes, peer))
    )

    rt.prepare(kv_cache_data={})
    assert prepared == [512, 768]
    assert [s.plans for s in rt.sites] == [{512: shared[512]}] * 2
    assert rt.ready and rt.step == "step"
    with pytest.raises(RuntimeError, match="once"):
        rt.prepare(kv_cache_data={})


def test_a_failed_launch_stops_serving_and_skips_the_collective_close():
    errs = iter([None, 0, 0x21])
    res = SimpleNamespace(rank=2, enqueued=[], released=False)
    res.readback_enqueue = res.enqueued.append
    res.readback_consume = lambda step: next(errs)
    res.release = lambda: setattr(res, "released", True)
    rt = _runtime()
    rt.res, rt.ready = res, True

    rt.before_step()
    rt.before_step()  # copy still in flight: no new readback
    rt.before_step()
    with pytest.raises(RuntimeError, match="rank 2.*ERR 0x21"):
        rt.before_step()
    assert res.enqueued == [0, 1]
    rt.close()
    assert not res.released

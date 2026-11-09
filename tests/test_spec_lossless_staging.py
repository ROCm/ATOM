# SPDX-License-Identifier: MIT
"""ATOM_SPEC_LOSSLESS_SAMPLING's proposer wiring: arming and its fallbacks, the
publication the staged buffers ride, the draft-graph warmup, and the verify
policy, run through the real `DSparkProposer` methods on CPU buffers.

Imports the proposer, hence aiter, so the CPU unit job skips this file; the
row-by-row staging policy itself is `DraftQStaging`, tested without aiter in
`tests/test_draft_q_staging.py`.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the proposer imports the AITER sampler ops")

from atom.model_ops.sampler import Sampler
from atom.model_ops.spec_sampling import sparse_q_store
from atom.model_ops.spec_sampling.dspark_markov_sample import _sample_scratch
from atom.model_ops.spec_sampling.ratio_verify import RatioVerifyInputs
from atom.model_ops.spec_sampling.sampling_scratch import carve
from atom.spec_decode import dspark_lossless, dspark_proposer
from atom.spec_decode.drafter import Drafter
from atom.spec_decode.dspark_proposer import DSparkProposer
from atom.utils import CpuGpuBuffer
from atom.utils.h2d import PublicationGroup

MAX_BS = 4
T = 3
VOCAB = 16
LOSSLESS = (
    "draft_seed",
    "draft_q_row_base",
    "draft_verify_q_row_base",
)


class FakeDraftModel:
    supports_lossless_draft = True
    vocab_size = VOCAB

    def __init__(self):
        self.markov_head = SimpleNamespace(
            markov_w1=nn.Embedding(VOCAB, 8), markov_w2=nn.Embedding(VOCAB, 8)
        )
        self.draft_sampling = None


def proposer(
    monkeypatch,
    *,
    lossless=True,
    pp=1,
    pcp=1,
    model=None,
    bind=True,
    synthetic=None,
    **config,
):
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1" if lossless else "0")
    monkeypatch.setattr(dspark_lossless, "get_pcp_world_size", lambda: pcp)
    # CPU stand-in weights; the GPU arming check has its own test below.
    monkeypatch.setattr(
        dspark_lossless, "markov_weights_unsupported", lambda w1, w2: None
    )
    prop = DSparkProposer.__new__(DSparkProposer)
    prop.config = SimpleNamespace(
        max_num_seqs=MAX_BS,
        pipeline_parallel_size=pp,
        **{
            "kv_transfer_config": {},
            "enable_rapidserve": False,
            "disagg_is_decode": False,
            **config,
        },
    )
    prop.speculative_config = SimpleNamespace(synthetic_acceptance_rates=synthetic)
    prop.device = torch.device("cpu")
    prop.runner = SimpleNamespace(rank=1)
    prop.metadata_buffers = {}
    prop.mtp_k = T
    prop.model = FakeDraftModel() if model is None else model
    prop.lossless = dspark_lossless.DSparkLossless.arm(prop)
    if bind and prop.lossless is not None:
        prop.lossless.bind_target(target_vars(), prop.model)
    return prop


def target_vars():
    """CPU stand-ins for the runner's `forward_vars` the draft reads."""
    return {
        name: CpuGpuBuffer(MAX_BS, dtype=dtype, device="cpu")
        for name, dtype in (
            ("temperatures", torch.float32),
            ("top_ks", torch.int32),
            ("top_ps", torch.float32),
        )
    }


def group(prop):
    """The runner's publication over the proposer's buffers: the real
    `PublicationGroup`, so staging records counts through `set_counts`."""
    return PublicationGroup(
        None,
        "sampling",
        [
            SimpleNamespace(name=name, buffer=buffer)
            for name, buffer in prop.metadata_buffers.items()
        ],
    )


def batch(
    req_ids,
    temperatures=None,
    *,
    top_ks=None,
    top_ps=None,
    dummy=False,
    final_chunk=None,
):
    n = len(req_ids)
    temps = np.ones(n, np.float32) if temperatures is None else temperatures
    return SimpleNamespace(
        req_ids=list(req_ids),
        temperatures=np.asarray(temps, np.float32),
        top_ks=np.asarray([-1] * n if top_ks is None else top_ks, np.int32),
        top_ps=np.asarray([1.0] * n if top_ps is None else top_ps, np.float32),
        is_dummy_run=dummy,
        is_final_chunk=final_chunk,
    )


def stage(prop, b, *, verifiable=True, verifies=True):
    """One forward's staging."""
    g = group(prop)
    prop.lossless.stage(
        b, len(b.req_ids), g, drafts_verifiable=verifiable, verifies=verifies
    )
    return {name: g.counts[g.indices[name]] for name in LOSSLESS}


def dest(prop):
    return prop.metadata_buffers["draft_q_row_base"].np.tolist()


def verify(prop):
    return prop.metadata_buffers["draft_verify_q_row_base"].np.tolist()


# ---- arming -----------------------------------------------------------------


def test_everything_the_kernels_read_rides_the_sampling_publication(monkeypatch):
    prop = proposer(monkeypatch)
    for name in LOSSLESS:
        assert prop.metadata_buffers[name].publication_group == "sampling", name
    sampling = prop.model.draft_sampling
    assert sampling.q_row_base is prop.metadata_buffers["draft_q_row_base"].gpu
    # Unpublished rows are harmless from the start: no destination.
    assert dest(prop) == [-1] * MAX_BS and verify(prop) == [-1] * MAX_BS


def test_the_draft_reads_the_targets_own_temperatures_and_filters(monkeypatch):
    """No copy to drift: once bound, the draft's per-row temperatures and
    filters are the runner's `forward_vars` buffers themselves. Arming only
    allocates; the model is bound once, complete."""
    prop = proposer(monkeypatch, bind=False)
    assert prop.model.draft_sampling is None
    forward_vars = target_vars()
    prop.lossless.bind_target(forward_vars, prop.model)
    sampling = prop.model.draft_sampling
    assert sampling.temperatures is forward_vars["temperatures"].gpu
    assert sampling.top_k is forward_vars["top_ks"].gpu
    assert sampling.top_p is forward_vars["top_ps"].gpu


@pytest.mark.parametrize(
    "lossless, dp, passes",
    [(True, 1, 2), (True, 2, 1), (False, 1, 1)],
    ids=["armed", "armed-dp2", "flag-off"],
)
def test_an_armed_drafter_declares_a_greedy_block_only_without_dp(
    monkeypatch, lossless, dp, passes
):
    """The greedy block is a second pass over the same backbone, not a term
    of the graph key: replay is decided on the DP-agreed batch alone, and
    which head a step needs is a per-rank fact."""
    model = FakeDraftModel()
    model.supports_block_graph = True
    prop = proposer(
        monkeypatch,
        lossless=lossless,
        model=model,
        parallel_config=SimpleNamespace(data_parallel_size=dp),
    )
    declared = prop._declare_draft_graphs()
    assert len(declared) == passes and declared[0] is prop.block
    assert prop.block.epilogue == prop._block_head and not prop.block.variant
    if passes == 1:
        assert prop.block_greedy is None
        return
    assert declared[1] is prop.block_greedy
    assert prop.block_greedy.forward == prop.block.forward
    assert prop.block_greedy.epilogue == prop._block_head_greedy
    assert prop.block_greedy.name != prop.block.name


@pytest.mark.parametrize("topk, width", [(None, VOCAB), (5, 5), (99999, VOCAB)])
def test_the_q_store_is_k_wide_and_never_wider_than_the_vocab(monkeypatch, topk, width):
    if topk is not None:
        monkeypatch.setattr(sparse_q_store, "Q_WIDTH", topk)
    sampling = proposer(monkeypatch).model.draft_sampling
    store = sampling.q_store
    assert store.ids.shape == store.probs.shape == (2 * MAX_BS * T, width)
    assert store.meta.shape == (2 * MAX_BS * T, 2)
    # The draft's dense scores are per batch row, whatever K is.
    s = carve(sampling.scratch, _sample_scratch(MAX_BS, VOCAB)).s
    assert s.shape == (MAX_BS, VOCAB)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lossless": False},
        {"pp": 2},
        {"pcp": 2},
        {"model": SimpleNamespace(supports_lossless_draft=False)},
        {"kv_transfer_config": {"kv_role": "kv_producer"}},
        {"kv_transfer_config": {"connectors": [{"kv_role": "kv_producer"}]}},
        {"enable_rapidserve": True, "disagg_is_decode": False},
    ],
    ids=[
        "flag-off",
        "pp",
        "pcp",
        "unported-flavor",
        "pd-producer",
        "pd-producer-multi",
        "rapidserve-prefill",
    ],
)
def test_the_greedy_pair_stays_when_not_armed(monkeypatch, kwargs):
    prop = proposer(monkeypatch, **kwargs)
    assert prop.lossless is None
    assert not set(LOSSLESS) & set(prop.metadata_buffers)


def test_a_block_narrower_than_the_verify_keeps_the_greedy_pair(monkeypatch, caplog):
    """A window caps the drafted rows below `mtp_k`, which the verify walks;
    q rows that short would let the verify read into the next slot."""
    model = FakeDraftModel()
    model.window_size = T - 1
    with caplog.at_level("WARNING", logger="atom"):
        prop = proposer(monkeypatch, model=model)
    assert prop.lossless is None
    assert not set(LOSSLESS) & set(prop.metadata_buffers)
    assert f"draft_tokens_per_seq={T - 1}, mtp_k={T}" in caplog.text
    # A window at least as wide as the block arms.
    model = FakeDraftModel()
    model.window_size = T
    assert proposer(monkeypatch, model=model).lossless is not None


@pytest.mark.parametrize(
    "config",
    [
        {"kv_transfer_config": {"kv_role": "kv_consumer"}},
        {"enable_rapidserve": True, "disagg_is_decode": True},
    ],
    ids=["pd-consumer", "rapidserve-decode"],
)
def test_the_verifying_side_of_a_disaggregated_pair_still_arms(monkeypatch, config):
    assert proposer(monkeypatch, **config).lossless is not None


def test_a_head_the_fused_op_cannot_run_keeps_the_greedy_pair(monkeypatch):
    prop = proposer(monkeypatch, lossless=False)
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
    monkeypatch.setattr(
        dspark_lossless,
        "markov_weights_unsupported",
        lambda w1, w2: "markov weights are torch.float32/torch.float32",
    )
    prop.lossless = dspark_lossless.DSparkLossless.arm(prop)
    assert prop.lossless is None and prop.model.draft_sampling is None


def test_a_sampler_without_aiters_filters_keeps_the_greedy_pair(monkeypatch):
    """The verify's p_f is aiter's kept set; `Sampler`'s native fallback
    would draw the bonus from another one."""
    prop = proposer(monkeypatch, lossless=False)
    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
    monkeypatch.setattr(dspark_lossless, "AITER_TOPK_TOPP_AVAILABLE", False)
    prop.lossless = dspark_lossless.DSparkLossless.arm(prop)
    assert prop.lossless is None and prop.model.draft_sampling is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"lossless": False},
        {"pp": 2},
        {"pcp": 2},
        {"model": SimpleNamespace(supports_lossless_draft=False)},
        {"kv_transfer_config": {"kv_role": "kv_producer"}},
        {"enable_rapidserve": True, "disagg_is_decode": False},
    ],
    ids=[
        "armed",
        "flag-off",
        "pp",
        "pcp",
        "unported-flavor",
        "pd-producer",
        "rapidserve-prefill",
    ],
)
def test_only_an_armed_engine_keeps_top_k_ties(monkeypatch, kwargs):
    """A refused or unset flag leaves `Sampler` on aiter's own law: the
    runner keeps ties (`Sampler.keep_ties`) only where the drafter arms."""
    prop = proposer(monkeypatch, **kwargs)
    armed = prop.lossless is not None
    assert armed is (kwargs == {})
    assert not Sampler().keep_top_k_ties


def test_the_ops_scratch_is_allocated_once_when_armed(monkeypatch):
    """Next to the q store, before the KV cache is sized: the profiling run
    sees it and no step allocates."""
    prop = proposer(monkeypatch)
    assert prop.lossless.draft_scratch.nbytes


def test_the_rejection_sampler_owns_the_verify_scratch_and_warmup(monkeypatch):
    """The ratio verify compiles per store width, draft count, vocab and
    logits dtype, so the warmup must use the ones serving will; its scratch,
    warmup and step counter are the rejection sampler's, not the drafter's."""
    from atom.model_ops import rejection_sampler as rs_mod

    sampler = rs_mod.RejectionSampler()
    monkeypatch.setattr(sparse_q_store, "Q_WIDTH", 8)
    calls = []
    monkeypatch.setattr(rs_mod, "warmup_ratio_verify", lambda *a: calls.append(a))
    sampler.arm_ratio(MAX_BS, T, VOCAB, torch.bfloat16, "cpu")
    assert sampler._ratio_scratch.nbytes
    ((steps, vocab, width, dtype, scratch),) = calls
    # The store's width, as the drafter sizes it (`store_width`).
    assert (steps, vocab, width, dtype) == (T, VOCAB, 8, torch.bfloat16)
    assert scratch is sampler._ratio_scratch


def test_the_verify_refuses_logits_it_was_not_warmed_for(monkeypatch):
    """A verify on another logits dtype would JIT every ratio kernel
    mid-serving, so it is refused instead."""
    from atom.model_ops import rejection_sampler as rs_mod

    sampler = rs_mod.RejectionSampler()
    monkeypatch.setattr(rs_mod, "warmup_ratio_verify", lambda *a: None)
    sampler.arm_ratio(MAX_BS, T, VOCAB, torch.bfloat16, "cpu")
    reached = []
    monkeypatch.setattr(rs_mod, "rejection_sample", lambda *a, **k: reached.append(a))
    meta = SimpleNamespace(
        input_ids=None,
        target_logits_indices=None,
        num_spec_steps=1,
        cu_num_draft_tokens=None,
        bonus_logits_indices=None,
    )
    ratio = RatioVerifyInputs(
        None, torch.full((2,), -1, dtype=torch.int32), 0, None, None, None
    )
    sampler(meta, torch.zeros(2, VOCAB, dtype=torch.bfloat16), None, ratio=ratio)
    assert len(reached) == 1
    # The drafter's inputs, verified in the scratch the sampler owns.
    passed = reached[0][4]
    assert passed.scratch is sampler._ratio_scratch
    assert passed._replace(scratch=None) == ratio
    with pytest.raises(RuntimeError, match="warmed up for torch.bfloat16"):
        sampler(meta, torch.zeros(2, VOCAB), None, ratio=ratio)
    # Exact match reads no ratio kernel, so any dtype goes.
    sampler(meta, torch.zeros(2, VOCAB), None)
    assert len(reached) == 2


# ---- staging ----------------------------------------------------------------


def test_every_flavors_propose_claims_its_staging(monkeypatch):
    """`Drafter.propose` claims the staging before any flavor's `_propose`,
    so a flavor that replaces the native one (V4 mono) cannot forget it."""
    from atom.models.deepseek_v4.mono.proposer import MonoDSparkProposer

    prop = proposer(monkeypatch)
    mono = MonoDSparkProposer.__new__(MonoDSparkProposer)
    mono.__dict__.update(vars(prop))
    drafted = []
    monkeypatch.setattr(
        MonoDSparkProposer, "_propose", lambda self, *args: drafted.append(args)
    )
    args = tuple(range(6))
    stage(mono, batch(["a"]))
    mono.propose(*args)
    assert drafted == [args]
    # A second propose without a staging of its own never reaches the flavor.
    with pytest.raises(RuntimeError, match="fresh DSparkLossless.stage"):
        mono.propose(*args)
    assert drafted == [args]


def test_a_draft_graph_warmup_writes_through_no_row_base(monkeypatch):
    """A recapture after a rollout sleep/wake warms the block directly, not
    through propose; whatever bases the last served step staged must not
    reach it, or the synthetic batch overwrites live requests' q."""
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b", "c"]))
    assert any(base >= 0 for base in dest(prop)[:3])
    seen = {}

    def base_warmup(self, build_context, end_trace=None):
        seen["np"] = self.metadata_buffers["draft_q_row_base"].np.tolist()
        seen["gpu"] = self.metadata_buffers["draft_q_row_base"].gpu.tolist()

    monkeypatch.setattr(dspark_proposer.Drafter, "_warmup_draft_graphs", base_warmup)
    prop.warmup_draft_graphs(None)
    assert seen["np"] == [-1] * MAX_BS and seen["gpu"] == [-1] * MAX_BS
    # The slot map is untouched: those requests still own their q rows.
    assert {"a", "b", "c"} <= set(prop.lossless.staging.slots._slot_of)


def test_the_v4_mono_draft_graph_warmup_writes_through_no_row_base(monkeypatch):
    """The V4 mono proposer replaces the native warmup without calling it;
    `Drafter.warmup_draft_graphs` still clears the staged row bases first."""
    from atom.models.deepseek_v4.mono.proposer import MonoDSparkProposer

    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b", "c"]))
    assert any(base >= 0 for base in dest(prop)[:3])
    mono = MonoDSparkProposer.__new__(MonoDSparkProposer)
    mono.__dict__.update(vars(prop))
    seen = {}

    def retire():
        seen["np"] = mono.metadata_buffers["draft_q_row_base"].np.tolist()
        seen["gpu"] = mono.metadata_buffers["draft_q_row_base"].gpu.tolist()

    mono._mono_session = SimpleNamespace(
        retire_draft_captures_for_startup=retire,
        draft_runner=SimpleNamespace(_request_destinations=[]),
    )
    mono.block = SimpleNamespace(will_capture=False, release_graphs=lambda: None)
    monkeypatch.setattr(prop.runner, "capture_sizes", [], raising=False)
    mono.warmup_draft_graphs(None)
    assert seen["np"] == [-1] * MAX_BS and seen["gpu"] == [-1] * MAX_BS


# ---- verify policy ----------------------------------------------------------


def test_the_verify_reads_the_store_and_each_requests_first_q_row(monkeypatch):
    prop = proposer(monkeypatch)
    stage(prop, batch(["a", "b"]))
    stage(prop, batch(["b", "a"]))
    temps, top_ks, top_ps = torch.ones(2), 5, torch.tensor([0.9, 0.7])
    ratio = prop.lossless.verify_inputs(2, temps, top_ks, top_ps)
    # The sparse store itself; the verify borrows nothing of the draft's.
    assert ratio.q is prop.lossless.q_store
    # The step the last stage published: this forward's draft and verify key.
    assert ratio.step == prop.metadata_buffers["draft_seed"].np[0] == 1
    # The published verify bases, read where the H2D lands them.
    gpu = prop.metadata_buffers["draft_verify_q_row_base"].gpu
    q_row_base = ratio.q_row_base
    assert q_row_base.shape == (2,) and q_row_base.data_ptr() == gpu.data_ptr()
    # The batch's own sampling inputs, as `postprocess` took them.
    assert ratio.temperatures is temps
    assert ratio.top_ks is top_ks and ratio.top_ps is top_ps
    # The scratch is the rejection sampler's.
    assert ratio.scratch is None
    assert min(verify(prop)[:2]) >= 0


# ---- exclusion with relaxed MTP ----------------------------------------------


class _Bare(Drafter):
    def _resolve_mtp_k(self):
        return 1

    def _propose(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.parametrize("lossless, relaxed", [(1, 1), (1, 0), (0, 1), (0, 0)])
def test_dspark_refuses_relaxed_mtp_where_lossless_sampling_arms(
    monkeypatch, lossless, relaxed
):
    monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", str(relaxed))
    if lossless and relaxed:
        with pytest.raises(ValueError, match="ATOM_ENABLE_RELAXED_MTP"):
            proposer(monkeypatch)
    else:
        prop = proposer(monkeypatch, lossless=bool(lossless))
        assert (prop.lossless is not None) is bool(lossless)


@pytest.mark.parametrize("lossless", [True, False])
def test_dspark_refuses_synthetic_acceptance_where_lossless_sampling_arms(
    monkeypatch, lossless
):
    """Synthetic acceptance replaces the verify, so the sampled draft's q
    would be stored and never read."""
    if lossless:
        with pytest.raises(ValueError, match="synthetic_acceptance_rates"):
            proposer(monkeypatch, synthetic=[0.5])
    else:
        assert proposer(monkeypatch, lossless=False, synthetic=[0.5]).lossless is None


def test_a_plain_drafter_starts_with_both_flags_set(monkeypatch):
    """Only DSpark reads ATOM_SPEC_LOSSLESS_SAMPLING to arm, so an MTP/EAGLE
    relaxed deployment that inherits it is not refused."""
    from atom.spec_decode import drafter as drafter_mod

    monkeypatch.setenv("ATOM_SPEC_LOSSLESS_SAMPLING", "1")
    monkeypatch.setenv("ATOM_ENABLE_RELAXED_MTP", "1")
    monkeypatch.setitem(drafter_mod.support_draft_model_arch_dict, "Plain", "plain")
    monkeypatch.setattr(drafter_mod, "resolve_obj_by_qualname", lambda name: None)
    monkeypatch.setattr(_Bare, "_build_draft_model", lambda self, cls: None)
    monkeypatch.setattr(_Bare, "_allocate_metadata_buffers", lambda *args: {})
    monkeypatch.setattr(_Bare, "_build_draft_graphs", lambda self: None)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(architectures=["Plain"])
        ),
        torch_dtype=torch.bfloat16,
        max_num_batched_tokens=8,
        max_num_seqs=2,
    )
    drafter = _Bare(config, torch.device("cpu"), None)
    assert drafter.mtp_k == 1
    assert drafter.lossless is None

# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: the fused/non-fused dispatch the runner builds per layer.

Covers, all CPU-only, without a GPU, a real model, or a real TP group:
  - the opt-in defaults to off, so the full/shared split always collapses to
    "every layer uses the existing non-fused path" unless both the env var
    and ``fused_indexer_refusal`` agree;
  - the full/shared predicate (``_is_full_indexer_layer``) is the exact same
    gate ``_refresh_indexer`` already uses, proven behaviorally (not just by
    reading the source) by checking which layers make ``_refresh_indexer``
    take its early-return path;
  - ``_build`` actually populates ``self._ops``/``self._fused_ops`` the way
    the switch and the full/shared split say it should -- not just that the
    boolean expression evaluates a particular way (tests that would still
    pass if the dispatch logic were deleted do not belong here);
  - C2/Important#2: ``index_request_width``/``index_block_table_stride`` are
    computed from the deployment (query length, max_model_len,
    kv_cache_block_size) and forwarded explicitly, at both the
    ``_build`` -> ``_GlmFusedLayerOp`` and ``_GlmFusedLayerOp`` ->
    ``Glm5TP4MonoKernel`` boundaries -- never left to the kernel's own
    defaults, which are only coincidentally right for a batch of one request;
  - Important#3: a fused-only build failure (weight mapping or kernel build)
    disables the fused path alone, TP-uniformly, and ``_build`` still
    succeeds -- the external-indexer mono path keeps serving;
  - ``_publish_selected_indices``'s delegation to
    ``triton_convert_req_index_to_global_index`` (``max_seqlen_q == 1``) and
    ``triton_gather_kv_indices_sparse`` (MTP, ``max_seqlen_q > 1``), neither
    of which has any other test coverage. A real differential/oracle run
    against either shared converter's own ``@triton.jit`` kernel would need a
    GPU (see ``tests/model_ops/test_triton_dynamic_shapes.py`` for how this
    repo gates that). These tests instead prove two separate things CPU-only:
    (1) the exact already-used-in-production function for the step's actual
    ``max_seqlen_q`` is called, in the documented argument order, publishing
    in place into ``shared_indices``; and (2) C-B specifically --
    ``triton_gather_kv_indices_sparse``'s per-token gather semantics
    (``_gather_kv_indices_sparse_kernel``, mirrored here in pure Python/torch
    from its own source, not just invoked) reimplemented and checked against
    hand-computed PER-TOKEN destinations at ``query_length=4``, ``batch=2``,
    with non-identity ``kv_indices`` -- this is the shape the old
    ``triton_convert_req_index_to_global_index``-always call silently
    corrupted (every row of a multi-row request aliasing onto one slot, and
    every request after the first landing at the wrong offset), invisible at
    ``batch=1``/``query_length=1`` and undetected by a mock that only checks
    "some function was called";
  - I-A: the base (non-fused) ops-build failure path in ``_build`` must
    reach the exact same ``tp_uniform_local_validation`` collective
    call-sites, in the same order, as a rank whose base build succeeds --
    not return before reaching one of them. Before the fix this path's
    ``except`` block returned directly with no collective call at all, so a
    rank whose local build failed left early while its peers (reaching a
    *later* collective the fused build phase added) blocked forever on a
    participant that already returned.
"""

from types import SimpleNamespace
from typing import ClassVar

import pytest

# `mono.runner` binds the TP group through `aiter.dist.parallel_state` at
# module load.  Skip on the non-GPU unit gate; runs wherever AITER is present.
pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
import torch

import atom.models.glm52.mono.runner as runner_mod
from atom.models.glm52.mono.config import (
    FUSED_INDEX_MAX_SEQ,
    FUSED_INDEX_PAGE_SIZE,
)
from atom.models.glm52.mono.kernels.geometry_config import glm5_tp_config
from atom.models.glm52.mono.runner import (
    Glm52MonoRunner,
    _GlmFusedLayerOp,
    _is_full_indexer_layer,
    _publish_selected_indices,
)
from atom.mono.runtime.consensus import MonoUnsupported


def _fused_servable_atom_config(**overrides):
    """A deployment that is TP4/fp8 (so ``native_config`` resolves) and also
    fused-indexer-servable (so only the env var decides ``_fused_indexer_on``)."""
    fields = {
        "speculative_config": None,
        "tensor_parallel_size": 4,
        "kv_cache_dtype": "fp8",
        "index_cache_dtype": "fp8",
        "max_model_len": FUSED_INDEX_MAX_SEQ,
        "kv_cache_block_size": FUSED_INDEX_PAGE_SIZE,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _layer(indexer, skip_topk):
    return SimpleNamespace(
        self_attn=SimpleNamespace(indexer=indexer, skip_topk=skip_topk)
    )


def test_opt_in_defaults_to_off(monkeypatch):
    # With the env var unset, Glm52MonoRunner.__init__ must compute
    # _fused_indexer_on == False regardless of how servable the deployment
    # otherwise is -- this is the property that keeps a default deployment's
    # built binaries, routing, and numerics exactly what they are today.
    monkeypatch.delenv("ATOM_GLM52_FUSED_INDEXER", raising=False)
    runner = Glm52MonoRunner(causal_lm=None, atom_config=_fused_servable_atom_config())
    assert runner._fused_indexer_on is False


def test_opt_in_on_and_servable_turns_fused_indexer_on(monkeypatch):
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    runner = Glm52MonoRunner(causal_lm=None, atom_config=_fused_servable_atom_config())
    assert runner._fused_indexer_on is True


def test_opt_in_on_but_refused_stays_off(monkeypatch):
    # The switch alone is not enough: a refusal (here, FP4 index cache) must
    # still force the existing path, same as if the switch were off.
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    runner = Glm52MonoRunner(
        causal_lm=None,
        atom_config=_fused_servable_atom_config(index_cache_dtype="fp4"),
    )
    assert runner._fused_indexer_on is False


@pytest.mark.parametrize("tp, fused_on", [(4, True), (8, False)])
def test_fused_indexer_and_its_cp_turn_on_at_tp4_only(monkeypatch, tp, fused_on):
    # TP8 is a served shard, but the fused indexer's only build
    # (`Glm5TP4MonoKernel`) refuses any TP but 4: with both switches on and
    # every other condition fusion-compatible, TP8 must keep the
    # external-indexer path instead of attempting that build.
    monkeypatch.setenv("ATOM_GLM52_FUSED_INDEXER", "1")
    monkeypatch.setenv("ATOM_GLM52_INDEXER_CP", "1")
    runner = Glm52MonoRunner(
        causal_lm=None,
        atom_config=_fused_servable_atom_config(
            tensor_parallel_size=tp, decode_context_parallel_size=1
        ),
    )
    assert runner._shard == glm5_tp_config(tp)
    assert runner._fused_indexer_on is fused_on
    assert runner._indexer_cp is fused_on


@pytest.mark.parametrize(
    "indexer, skip_topk, expected_full",
    [
        (None, False, False),
        (object(), True, False),
        (object(), False, True),
    ],
)
def test_is_full_indexer_layer_matches_refresh_indexer_guard(
    indexer, skip_topk, expected_full
):
    # Behavioral proof, not just a read of the source: _is_full_indexer_layer
    # must agree with whether _refresh_indexer takes its early-return path.
    # A "full" layer's bare SimpleNamespace lacks input_layernorm, so
    # _refresh_indexer raises AttributeError once it proceeds past the guard
    # instead of silently doing nothing.
    layer = _layer(indexer, skip_topk)
    assert _is_full_indexer_layer(layer) is expected_full
    if expected_full:
        with pytest.raises(AttributeError):
            Glm52MonoRunner._refresh_indexer(layer, state=None, positions=None)
    else:
        assert (
            Glm52MonoRunner._refresh_indexer(layer, state=None, positions=None) is None
        )


# ---------------------------------------------------------------------------
# `_build`-level dispatch: a bare runner, with every collaborator `_build`
# reads faked out so it runs CPU-only (no real model, no GPU, no TP group).
# ---------------------------------------------------------------------------


def _fake_mono_layer(layer_idx, *, full):
    """A layer `mono_layers` accepts (``mlp.experts`` present): "full" when it
    carries an indexer and is not skip_topk, "shared" otherwise."""
    return SimpleNamespace(
        layer_idx=layer_idx,
        mlp=SimpleNamespace(experts=object()),
        self_attn=SimpleNamespace(
            indexer=(object() if full else None), skip_topk=False
        ),
    )


class _RecordingOp:
    """Stands in for `_GlmLayerOp`/`_GlmFusedLayerOp`: records every
    construction (positionally, as `_build` actually calls them) and the
    order of `close()` calls, without touching a kernel, a device, or a TP
    group."""

    instances: ClassVar[list] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.closed = False
        self.op = SimpleNamespace()  # stands in for the kernel's own runtime
        type(self).instances.append(self)

    def close(self) -> None:
        self.closed = True


class _FailOnNthOp(_RecordingOp):
    """Like `_RecordingOp`, but raises on the Nth construction -- for testing
    that a later failure rolls back earlier-in-this-call successes."""

    fail_on_call: int = 0  # 1-based; 0 means never fail
    call_count: int = 0

    def __init__(self, *args, **kwargs):
        type(self).call_count += 1
        if type(self).call_count == type(self).fail_on_call:
            raise MonoUnsupported("simulated fused build failure")
        super().__init__(*args, **kwargs)


def _build_ready_runner(
    monkeypatch, *, fused_indexer_on, full_layers=1, shared_layers=1, query_length=1
):
    layers = [_fake_mono_layer(i, full=True) for i in range(full_layers)] + [
        _fake_mono_layer(full_layers + i, full=False) for i in range(shared_layers)
    ]
    model = SimpleNamespace(
        layers=layers,
        start_layer=0,
        end_layer=len(layers),
        norm=SimpleNamespace(weight=torch.zeros(1)),
    )
    runner = object.__new__(Glm52MonoRunner)
    runner._lm = SimpleNamespace(model=model)
    runner._atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(index_topk=128),
        kv_cache_dtype="fp8",
        speculative_config=None,
        max_model_len=32768,
        kv_cache_block_size=16,
    )
    runner._query_length = query_length
    runner._shard = SimpleNamespace(hidden=5120)
    runner._ops = {}
    runner._fused_ops = {}
    runner._weights = {}
    runner._prepared = {}
    runner._fused_weights = {}
    runner._fused_prepared = {}
    runner._runtimes = {}
    runner._base_launches = {}
    runner._fused_launches = {}
    runner._fused_runtimes = {}
    runner._refused = set()
    runner._announced = set()
    runner._fused_indexer_on = fused_indexer_on
    # The CP switch is off here, which is the pre-switch build: these tests
    # pin fused-indexer dispatch, not CP. tests/models/glm52/
    # test_indexer_cp_switch.py covers the switch itself.
    runner._indexer_cp = False

    monkeypatch.setattr(runner_mod, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(runner_mod, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        runner_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )
    monkeypatch.setattr(runner_mod, "native_config", lambda **kwargs: object())
    monkeypatch.setattr(runner_mod, "glm5_kernel_samples", lambda rows, ql: rows)
    monkeypatch.setattr(runner_mod, "bf16_vector", lambda *a, **k: None)

    def _fake_layer_weights(layer, rank, npes, with_indexer=False):
        return SimpleNamespace(layer_idx=layer.layer_idx, with_indexer=with_indexer)

    monkeypatch.setattr(runner_mod, "layer_weights", _fake_layer_weights)
    monkeypatch.setattr(
        "atom.models.glm52.mono.kernels.op.prepare_glm5_weights",
        lambda weights, attention_weight: {"weights": weights},
    )

    def _fake_tp_uniform(error, *, group, world_size, context):
        if error is not None:
            raise MonoUnsupported(f"{context}: {error}")

    monkeypatch.setattr(runner_mod, "tp_uniform_local_validation", _fake_tp_uniform)
    return runner


def test_switched_off_builds_only_the_non_fused_op_for_every_layer(monkeypatch):
    # M1 replacement: a real _build() call, not a re-evaluation of the gate
    # expression. If the `if self._fused_indexer_on` guard around the fused
    # weight-mapping/build phases were ever deleted or inverted, this would
    # catch it (self._fused_ops would stop being empty).
    _RecordingOp.instances = []
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _RecordingOp)
    monkeypatch.setattr(runner_mod, "_GlmFusedLayerOp", _RecordingOp)
    runner = _build_ready_runner(
        monkeypatch, fused_indexer_on=False, full_layers=2, shared_layers=1
    )
    assert runner._build(rows=1) is True
    assert len(runner._ops) == 3  # every layer, full and shared alike
    assert runner._fused_ops == {}


def test_switched_on_builds_fused_only_for_full_layers(monkeypatch):
    # M1 replacement: with the switch on, _build must populate _fused_ops for
    # exactly the full layers, alongside (not instead of) their _ops entry --
    # the base op stays the fallback a failed fused build drops back to.
    _RecordingOp.instances = []
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _RecordingOp)
    monkeypatch.setattr(runner_mod, "_GlmFusedLayerOp", _RecordingOp)
    runner = _build_ready_runner(
        monkeypatch, fused_indexer_on=True, full_layers=2, shared_layers=1
    )
    assert runner._build(rows=1) is True
    assert len(runner._ops) == 3
    assert len(runner._fused_ops) == 2
    full_layer_indices = {0, 1}
    assert {layer_idx for layer_idx, _ in runner._fused_ops} == full_layer_indices


def test_build_forwards_query_length_and_block_table_stride_to_fused_op(monkeypatch):
    # C2/Important#2, at the _build -> _GlmFusedLayerOp boundary: the runner's
    # own query length and max_model_len // kv_cache_block_size must reach
    # _GlmFusedLayerOp, not be silently dropped (which previously left
    # Glm5TP4MonoKernel on its own, batch-of-one-only defaults).
    _RecordingOp.instances = []
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _RecordingOp)
    monkeypatch.setattr(runner_mod, "_GlmFusedLayerOp", _RecordingOp)
    runner = _build_ready_runner(
        monkeypatch,
        fused_indexer_on=True,
        full_layers=1,
        # At least one shared layer, because the two runtimes each number their
        # own launches 0 .. N-1 and a base runtime with zero launches has no
        # valid `launches_per_step`. GLM-5.2 always has 57 of them, so an
        # all-full model is not a shape the runner has to serve -- it is now
        # refused by name rather than mis-numbered.
        shared_layers=1,
        query_length=5,
    )
    # max_model_len=32768, kv_cache_block_size=16 (set in _build_ready_runner)
    # -> the correct stride is 2048, not the kernel's own
    # index_max_seq // index_page_size default (16384 // 16 == 1024).
    assert runner._build(rows=5) is True
    (fused_op,) = runner._fused_ops.values()
    # _GlmFusedLayerOp's positional signature: weights, samples, topk,
    # kv_cache_dtype, index_page_size, index_request_width,
    # index_block_table_stride, prepared_weights, runtime.
    assert fused_op.args[5] == 5  # index_request_width == self._query_length
    assert fused_op.args[6] == 2048  # index_block_table_stride


def test_fused_layer_op_forwards_width_and_stride_to_the_kernel_not_its_defaults(
    monkeypatch,
):
    # C2/Important#2, at the _GlmFusedLayerOp -> Glm5TP4MonoKernel boundary.
    # samples=8 is deliberately not in (5, 10), so Glm5TP4MonoKernel's own
    # unsafe default (`5 if samples in (5, 10) else samples`) would silently
    # equal 8 -- treating an 8-row step as ONE request of width 8 and
    # collapsing every other request's rows into request 0. Passing
    # index_request_width=4 explicitly (two 4-row requests) must survive
    # unchanged. Likewise index_block_table_stride=2048 must survive instead
    # of the kernel's own index_max_seq // index_page_size == 1024.
    from atom.models.glm52.mono.kernels import tp4_op

    captured = {}

    class _FakeKernel:
        def __init__(self, weights, samples, **kwargs):
            captured["weights"] = weights
            captured["samples"] = samples
            captured.update(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(tp4_op, "Glm5TP4MonoKernel", _FakeKernel)
    monkeypatch.setattr(runner_mod, "get_tensor_model_parallel_rank", lambda: 2)
    monkeypatch.setattr(runner_mod, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        runner_mod, "get_tp_group", lambda: SimpleNamespace(cpu_group=None)
    )

    weights = object()
    prepared = object()
    runtime = object()
    _GlmFusedLayerOp(
        weights,
        8,  # samples
        128,  # topk
        "fp8",  # kv_cache_dtype
        16,  # index_page_size
        4,  # index_request_width
        2048,  # index_block_table_stride
        False,  # indexer_cp
        32768,  # index_max_seq
        prepared,
        runtime,
        18,  # launches_per_step -- the fused runtime's own count (full layers)
    )
    assert captured["weights"] is weights
    assert captured["samples"] == 8
    assert captured["index_request_width"] == 4
    assert captured["index_block_table_stride"] == 2048
    assert captured["index_page_size"] == 16
    assert captured["with_indexer"] is True
    assert captured["indexer_cp"] is False
    # The build capacity is the caller's, NOT ``FUSED_INDEX_MAX_SEQ``. The
    # scoring work map is enumerated from this number and not from the
    # request (``layout.py:448``), so compiling a small deployment against the
    # 1 M ceiling costs 64x the tasks for identical work -- measured at
    # +11.14 % against the external indexer. If this assertion ever reads
    # 1048576, that regression is back.
    assert captured["index_max_seq"] == 32768
    assert captured["index_max_seq"] != FUSED_INDEX_MAX_SEQ
    assert captured["rank"] == 2
    assert captured["npes"] == 4


def test_fused_weight_mapping_failure_falls_back_without_refusing_rows(monkeypatch):
    # Important#3: a fused-only (with_indexer=True) weight-mapping failure
    # must disable only the fused path -- never reach self._refused, never
    # make _build return False -- leaving the base ops (the external-indexer
    # mono path) fully built and the deployment still served.
    _RecordingOp.instances = []
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _RecordingOp)
    monkeypatch.setattr(runner_mod, "_GlmFusedLayerOp", _RecordingOp)
    runner = _build_ready_runner(
        monkeypatch, fused_indexer_on=True, full_layers=2, shared_layers=1
    )

    def _failing_fused_layer_weights(layer, rank, npes, with_indexer=False):
        if with_indexer:
            raise MonoUnsupported("indexer weights missing")
        return SimpleNamespace(layer_idx=layer.layer_idx, with_indexer=with_indexer)

    monkeypatch.setattr(runner_mod, "layer_weights", _failing_fused_layer_weights)

    assert runner._build(rows=1) is True
    assert runner._fused_indexer_on is False
    assert 1 not in runner._refused
    assert len(runner._ops) == 3  # base path: unaffected, fully usable
    assert runner._fused_ops == {}


def test_fused_build_failure_rolls_back_this_calls_fused_ops_and_falls_back(
    monkeypatch,
):
    # Important#3, the other failure site: the fused KERNEL build itself (not
    # the weight mapping) fails on the second of two full layers. The first
    # full layer's fused op, already constructed in this same _build() call,
    # must be closed and removed -- not left dangling half-built -- and the
    # base path must still come up.
    _RecordingOp.instances = []
    _FailOnNthOp.instances = []
    _FailOnNthOp.call_count = 0
    _FailOnNthOp.fail_on_call = 2
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _RecordingOp)
    monkeypatch.setattr(runner_mod, "_GlmFusedLayerOp", _FailOnNthOp)
    runner = _build_ready_runner(
        monkeypatch, fused_indexer_on=True, full_layers=2, shared_layers=1
    )

    assert runner._build(rows=1) is True
    assert runner._fused_indexer_on is False
    assert 1 not in runner._refused
    assert len(runner._ops) == 3
    assert runner._fused_ops == {}
    # The first fused op DID get built (call 1 succeeded) and must have been
    # closed during rollback, not merely forgotten.
    first_built = [op for op in _FailOnNthOp.instances if isinstance(op, _FailOnNthOp)]
    assert len(first_built) == 1
    assert first_built[0].closed is True


# ---------------------------------------------------------------------------
# `_publish_selected_indices`: wiring/coordinate-space delegation.
# ---------------------------------------------------------------------------


def test_publish_selected_indices_delegates_translation_to_the_shared_converter(
    monkeypatch,
):
    # C1/C3, at max_seqlen_q == 1 (the only width this converter is correct
    # for -- see C-B's test below for query_length > 1): this function must
    # no longer scatter the kernel's own LOGICAL scratch directly into
    # shared_indices (the old index_copy_/boolean-mask bug). It must instead
    # call triton_convert_req_index_to_global_index -- the same function the
    # non-fused/external indexer path already uses at this width
    # (deepseek_v2.py's own `else` branch) -- passing the step's real
    # coordinate-space tensors, and publish strictly through that call's
    # `out=` kwarg.
    samples, topk = 2, 4
    # Logical (pre-translation) scratch: sample 0 selected logical positions
    # [2, 0, 1] (zero-padded sentinel at column 3); sample 1 selected [1]
    # (zero-padded beyond that).
    values = torch.tensor([2, 0, 1, 0, 1, 0, 0, 0], dtype=torch.int32)
    op = SimpleNamespace(
        scr_layout={"indices": 0},
        topk=topk,
        scratch=values.view(torch.uint8).contiguous(),
    )
    sparse_indptr = torch.tensor([0, 3, 5], dtype=torch.int32)  # page_kv_indptr
    cu_seqlens_q = torch.tensor([0, 1, 2], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 10, 20], dtype=torch.int32)
    kv_indices = torch.arange(20, dtype=torch.int32)
    shared_indices = torch.full((5,), -1, dtype=torch.int32)

    captured = {}

    def _fake_convert(
        qo_indptr, kv_indptr_, page_kv_indptr, kv_indices_, token_indices, **kwargs
    ):
        captured.update(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr_,
            page_kv_indptr=page_kv_indptr,
            kv_indices=kv_indices_,
            token_indices=token_indices,
            **kwargs,
        )
        return kwargs.get("out")

    def _fail_if_called_gather(*args, **kwargs):
        raise AssertionError(
            "triton_gather_kv_indices_sparse must not be called at "
            "max_seqlen_q == 1 -- that is the per-request width, handled by "
            "triton_convert_req_index_to_global_index"
        )

    monkeypatch.setattr(
        runner_mod, "triton_convert_req_index_to_global_index", _fake_convert
    )
    monkeypatch.setattr(
        runner_mod, "triton_gather_kv_indices_sparse", _fail_if_called_gather
    )
    _publish_selected_indices(
        op,
        shared_indices,
        sparse_indptr,
        cu_seqlens_q,
        kv_indptr,
        kv_indices,
        samples,
        max_seqlen_q=1,
        batch_id_per_q_token=None,
    )

    assert captured["qo_indptr"] is cu_seqlens_q
    assert captured["kv_indptr"] is kv_indptr
    assert captured["page_kv_indptr"] is sparse_indptr
    assert captured["kv_indices"] is kv_indices
    assert captured["NUM_TOPK_TOKENS"] == topk
    assert captured["out"] is shared_indices  # published in place, not a copy

    # The kernel's raw (samples, topk) scratch, untranslated: proves this
    # function performs NO translation of its own -- that is entirely the
    # shared converter's job now.
    token_indices = captured["token_indices"]
    assert tuple(token_indices.shape) == (samples, topk)
    assert token_indices.tolist() == values.view(samples, topk).tolist()

    # The mock does no real writing; shared_indices staying at its sentinel
    # proves there is no OTHER write path left in this function (e.g. a
    # leftover index_copy_ alongside the new call).
    assert shared_indices.tolist() == [-1, -1, -1, -1, -1]


def _reference_gather_kv_indices_sparse(
    sparse_kv_indptr, batch_id_per_q_token, topk_indices, kv_indices, kv_indptr, out
):
    """Pure Python/torch reimplementation of ``_gather_kv_indices_sparse_kernel``
    (``atom/model_ops/attention_mla.py``), transcribed directly from its
    ``tl.load``/``tl.store`` body: per TOKEN (not per request), gather
    ``kv_indices[kv_indptr[req] + pos]`` for each of that token's selected
    logical positions, into ``out[sparse_kv_indptr[token] : sparse_kv_indptr
    [token + 1]]``. This is the test's oracle for C-B's per-token-destination
    assertion -- not a mock that only records a call, but the kernel's own
    documented semantics executed on CPU.
    """
    num_tokens = topk_indices.shape[0]
    for token_id in range(num_tokens):
        req_id = int(batch_id_per_q_token[token_id])
        out_start = int(sparse_kv_indptr[token_id])
        out_end = int(sparse_kv_indptr[token_id + 1])
        kv_len = out_end - out_start
        if req_id < 0:
            continue
        kv_base = int(kv_indptr[req_id])
        kv_end = int(kv_indptr[req_id + 1])
        req_kv_len = kv_end - kv_base
        for col in range(min(kv_len, topk_indices.shape[1])):
            pos = int(topk_indices[token_id, col])
            if 0 <= pos < req_kv_len:
                out[out_start + col] = kv_indices[kv_base + pos]
    return out


def test_publish_selected_indices_mtp_gathers_real_per_token_destinations(
    monkeypatch,
):
    # C-B: at max_seqlen_q > 1 (MTP -- GLM5_QUERY_LENGTHS' 4/5/6, the NORMAL
    # serving case, not an edge case), sparse_kv_indptr is per-TOKEN while
    # cu_seqlens_q/kv_indptr stay per-REQUEST. Calling
    # triton_convert_req_index_to_global_index here (the bug this test
    # guards against) makes every row of a request collapse onto request 0's
    # single per-request slot (last write wins) and misaligns every request
    # after the first -- invisible at batch=1 or query_length=1, which is
    # exactly why the old mock-based test above passed while this was
    # broken. This test uses query_length=4, batch=2 (8 token rows), and
    # non-identity kv_indices (a physical-row table, not an identity
    # arange), and asserts the actual PER-TOKEN gathered values -- not just
    # that some function was called.
    samples, topk = 8, 2

    # Per-token request id: tokens 0-3 are request 0's 4 query rows
    # (query_length=4), tokens 4-7 are request 1's.
    batch_id_per_q_token = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32)
    # Per-token sparse_kv_indptr: each of the 8 tokens gets its OWN 2-wide
    # slot (topk=2) -- the per-token domain C-B is about, structurally
    # different from the per-request domain triton_convert_req_index_to_
    # global_index assumes (which would give request 0 a single [0, 2) slot
    # for all 4 of its rows).
    sparse_kv_indptr = torch.arange(0, (samples + 1) * topk, topk, dtype=torch.int32)
    # Per-request kv_indptr: request 0 owns 5 logical KV positions, request 1
    # owns 7, back-to-back in kv_indices.
    kv_indptr = torch.tensor([0, 5, 12], dtype=torch.int32)
    # Non-identity physical-row table: kv_indices[i] != i everywhere, so a
    # coordinate-space bug shows up as a wrong VALUE, not an accidentally
    # correct one.
    kv_indices = torch.tensor(
        [100, 101, 102, 103, 104, 200, 201, 202, 203, 204, 205, 206],
        dtype=torch.int32,
    )
    # Each token's own top-k logical positions (within its OWN request's
    # kv_len) -- deliberately distinct per token, so an aliasing bug (every
    # row of a request collapsing onto one slot) is visible as a wrong
    # destination, not merely a wrong value in an already-wrong slot.
    topk_indices = torch.tensor(
        [
            [4, 0],
            [1, 3],
            [2, 2],
            [0, 4],
            [6, 0],
            [3, 5],
            [1, 2],
            [0, 6],
        ],
        dtype=torch.int32,
    )
    off = 0
    op = SimpleNamespace(
        scr_layout={"indices": off},
        topk=topk,
        # Flattened to 1D before the byte-view, matching how `op.scratch` is
        # actually laid out (a flat scratch buffer `_publish_selected_
        # indices` slices by byte offset then reshapes back to
        # (samples, topk) itself) -- viewing a still-2D tensor as uint8
        # appends a trailing bytes-dim instead, which a flat slice can't
        # index the way the production code does.
        scratch=topk_indices.reshape(-1).contiguous().view(torch.uint8).clone(),
    )
    shared_indices = torch.full((samples * topk,), -1, dtype=torch.int32)

    expected = _reference_gather_kv_indices_sparse(
        sparse_kv_indptr,
        batch_id_per_q_token,
        topk_indices,
        kv_indices,
        kv_indptr,
        out=torch.full((samples * topk,), -1, dtype=torch.int32),
    )
    # Sanity on the oracle itself: every token's destination must differ
    # from at least one other token in the SAME request -- otherwise this
    # test could not detect the old "all rows alias onto request 0's slot"
    # bug even in principle.
    per_request_0 = expected[: 4 * topk].tolist()
    assert (
        len(set(map(tuple, [per_request_0[i : i + topk] for i in range(0, 8, topk)])))
        == 4
    )
    assert expected.tolist() == [
        104,
        100,
        101,
        103,
        102,
        102,
        100,
        104,
        206,
        200,
        203,
        205,
        201,
        202,
        200,
        206,
    ]

    captured = {}

    def _fake_gather(
        sparse_kv_indptr_,
        batch_id_per_q_token_,
        topk_indices_,
        kv_indices_,
        kv_indptr_,
        **kwargs,
    ):
        captured.update(
            sparse_kv_indptr=sparse_kv_indptr_,
            batch_id_per_q_token=batch_id_per_q_token_,
            topk_indices=topk_indices_,
            kv_indices=kv_indices_,
            kv_indptr=kv_indptr_,
            **kwargs,
        )
        out = kwargs["out"]
        return _reference_gather_kv_indices_sparse(
            sparse_kv_indptr_,
            batch_id_per_q_token_,
            topk_indices_,
            kv_indices_,
            kv_indptr_,
            out=out,
        )

    def _fail_if_called_convert(*args, **kwargs):
        raise AssertionError(
            "triton_convert_req_index_to_global_index must not be called at "
            "max_seqlen_q > 1 -- that is the per-request width; this is MTP"
        )

    monkeypatch.setattr(runner_mod, "triton_gather_kv_indices_sparse", _fake_gather)
    monkeypatch.setattr(
        runner_mod, "triton_convert_req_index_to_global_index", _fail_if_called_convert
    )

    # cu_seqlens_q is still passed through (same call signature) even though
    # the MTP branch does not consume it -- mirrors deepseek_v2.py's own
    # call, which keeps a uniform argument list across both branches.
    cu_seqlens_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    _publish_selected_indices(
        op,
        shared_indices,
        sparse_kv_indptr,
        cu_seqlens_q,
        kv_indptr,
        kv_indices,
        samples,
        max_seqlen_q=4,
        batch_id_per_q_token=batch_id_per_q_token,
    )

    assert captured["sparse_kv_indptr"] is sparse_kv_indptr
    assert captured["batch_id_per_q_token"] is batch_id_per_q_token
    assert captured["kv_indices"] is kv_indices
    assert captured["kv_indptr"] is kv_indptr
    assert captured["NUM_TOPK_TOKENS"] == topk
    assert captured["out"] is shared_indices

    # The actual point of this test: per-token destinations match the
    # reference gather exactly -- request 1's rows land at request 1's own
    # offsets (not request 0's), and no two tokens in the same request wrote
    # the same slot.
    assert shared_indices.tolist() == expected.tolist()


def test_one_ranks_base_build_failure_still_reaches_every_collective(monkeypatch):
    # I-A: the base (non-fused) ops-build failure path in `_build` must reach
    # the SAME tp_uniform_local_validation call-sites, in the same order, as
    # a rank whose base build succeeds -- not return before one of them.
    # Before the fix, this path's `except` block returned directly with NO
    # collective call, while the fused-phase build added a LATER collective
    # downstream of it: a rank whose local ops build failed would leave
    # early, and its peers (whose local build succeeded) would walk into
    # that later collective and block forever on a participant that already
    # returned -- a silent gloo hang, no error.
    #
    # Simulated here, CPU-only and deterministically (no real TP group, no
    # threads, no timing), by comparing the ordered SEQUENCE of `context`
    # strings two otherwise-identical "ranks" pass to
    # tp_uniform_local_validation: one whose local base build succeeds, one
    # whose local base build raises. If the failing rank's sequence stops
    # short of the succeeding rank's, that rank skipped a collective its
    # peer reached -- exactly I-A's hang, made visible as a sequence
    # mismatch instead of a wedged test run.
    call_sequences: dict[str, list[str]] = {"ok": [], "fail": []}

    def _recording_tp_uniform(label):
        def _tp_uniform(error, *, group, world_size, context):
            call_sequences[label].append(context)
            if error is not None:
                raise MonoUnsupported(f"{context}: {error}")

        return _tp_uniform

    _RecordingOp.instances = []
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _RecordingOp)
    monkeypatch.setattr(runner_mod, "_GlmFusedLayerOp", _RecordingOp)
    runner_ok = _build_ready_runner(
        monkeypatch, fused_indexer_on=False, full_layers=1, shared_layers=1
    )
    monkeypatch.setattr(
        runner_mod, "tp_uniform_local_validation", _recording_tp_uniform("ok")
    )
    assert runner_ok._build(rows=1) is True

    _FailOnNthOp.instances = []
    _FailOnNthOp.call_count = 0
    _FailOnNthOp.fail_on_call = 1
    monkeypatch.setattr(runner_mod, "_GlmLayerOp", _FailOnNthOp)
    runner_fail = _build_ready_runner(
        monkeypatch, fused_indexer_on=False, full_layers=1, shared_layers=1
    )
    monkeypatch.setattr(
        runner_mod, "tp_uniform_local_validation", _recording_tp_uniform("fail")
    )
    assert runner_fail._build(rows=1) is False
    assert 1 in runner_fail._refused

    assert call_sequences["ok"] == [
        "GLM-5.2 weight mapping failed",
        "GLM-5.2 ops build failed",
    ]
    # The regression this guards: before the fix, call_sequences["fail"]
    # would end after "GLM-5.2 weight mapping failed" -- this rank would
    # never call tp_uniform_local_validation for the ops-build phase at all.
    assert call_sequences["fail"] == call_sequences["ok"]

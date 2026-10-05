# SPDX-License-Identifier: MIT
"""Sharing per-request state across a rapidserve pair.

The pair runs two processes on one GPU. Prefill writes a request's compressor
state; decode continues from it. That only works if a slot index names the same
bytes on both sides, so prefill exports the per-request pool over CUDA IPC and
decode carves identical views over it -- which is safe only for a backend that
says so, because two processes carving *almost* the same layout is one request
reading another's state, with nothing downstream able to tell.

Hence the capability: `shares_per_req_cache()` is the single switch the export,
the import, and the checkpoint guard all read.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/model_runner.py"
)
CORE = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/engine_core.py"
)
ATTENTIONS = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_ops/attentions"
)


def _method(path: pathlib.Path, owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


# ── The capability is opt-in ─────────────────────────────────────────────


def test_the_base_backend_does_not_share():
    """Default False. A backend that has not thought about `buf` must not be
    assumed to carve identically in two processes.

    Read statically, for the reason `test_kv_pool_contract` gives: this module
    imports aiter, and the regression would land on CI, which has none."""
    fn = _method(ATTENTIONS / "backends.py", "AttentionMetadataBuilder",
                 "shares_per_req_cache")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    assert returns[0].value.value is False


def test_only_backends_that_honour_buf_opt_in():
    """Every `shares_per_req_cache` returning True must belong to a class whose
    `allocate_per_req_cache` takes `buf` -- the capability is a promise about
    that signature, and a True on a backend without it would have the runner
    pass an argument nobody accepts."""
    offenders = []
    for src in sorted(ATTENTIONS.rglob("*.py")):
        tree = ast.parse(src.read_text(), filename=str(src))
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            methods = {
                fn.name: fn for fn in cls.body if isinstance(fn, ast.FunctionDef)
            }
            shares = methods.get("shares_per_req_cache")
            if shares is None:
                continue
            returns_true = any(
                isinstance(n, ast.Return)
                and isinstance(n.value, ast.Constant)
                and n.value.value is True
                for n in ast.walk(shares)
            )
            if not returns_true:
                continue
            alloc = methods.get("allocate_per_req_cache")
            if alloc is None or "buf" not in {a.arg for a in alloc.args.args}:
                offenders.append(f"{src.name}:{cls.name}")
    assert not offenders, (
        f"{offenders} claim shares_per_req_cache() but their "
        "allocate_per_req_cache does not accept `buf`"
    )


def test_deepseek_v4_shares_and_checks_the_size():
    """V4 is the one backend ported. The size check is its whole safety
    argument: every offset comes from `pool_geometry`, so agreeing on the total
    is agreeing on the split."""
    src = pathlib.Path(
        pathlib.Path(__file__).resolve().parent.parent
        / "atom/model_ops/attentions/deepseek_v4_attn.py"
    ).read_text()
    tree = ast.parse(src)
    cls = next(
        c
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and "shares_per_req_cache" in {
            f.name for f in c.body if isinstance(f, ast.FunctionDef)
        }
    )
    alloc = next(
        f
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name == "allocate_per_req_cache"
    )
    body = ast.get_source_segment(src, alloc)
    assert body is not None
    assert "buf.numel() != total_bytes" in body, (
        "V4 must reject an imported pool whose size disagrees with the "
        "geometry it just derived"
    )


# ── Neither the export nor the guard may bypass the capability ───────────


@pytest.mark.parametrize(
    "method", ["_shared_pool_names", "_bind_kv_cache_to_modules"]
)
def test_the_runner_keeps_both_halves_of_the_handoff(method):
    """Export and bind are one contract read from two sides; a tree with only
    one of them is how the merge left this file."""
    assert _method(RUNNER, "RapidServeModelRunner", method) is not None


def test_binding_refuses_state_it_was_not_given():
    """The refusal `main` carried, narrowed: it fires on an unshared pool
    rather than on per-request state as such."""
    src = ast.get_source_segment(
        RUNNER.read_text(),
        _method(RUNNER, "RapidServeModelRunner", "_bind_kv_cache_to_modules"),
    )
    assert src is not None
    assert "stateful and per_req_buf is None" in src
    assert "NotImplementedError" in src


def test_the_checkpoint_guard_asks_the_capability():
    """Both guard sites -- prefill's in ModelRunner.get_num_blocks and decode's
    in the RapidServe override -- have to agree, or one side starts and the
    other refuses."""
    for owner in ("ModelRunner", "RapidServeModelRunner"):
        src = ast.get_source_segment(
            RUNNER.read_text(), _method(RUNNER, owner, "get_num_blocks")
        )
        assert src is not None
        assert "shares_per_req_cache()" in src, (
            f"{owner}.get_num_blocks no longer consults the capability, so it "
            "refuses (or admits) rapidserve independently of the other side"
        )


# ── The slot indices reach prefill ───────────────────────────────────────


def test_block_assignment_carries_state_slots_not_the_old_swa_table():
    from atom.model_engine.disagg_types import BlockAssignment

    names = {f.name for f in __import__("dataclasses").fields(BlockAssignment)}
    assert "state_slots" in names
    assert "swa_block_table" not in names, (
        "the paged-SWA table went away with #1771; a field nothing fills is a "
        "trap for the next reader"
    )


def test_the_assignment_round_trip_uses_the_same_field():
    """Send and apply are in one file and drifted apart once already."""
    text = CORE.read_text()
    send = ast.get_source_segment(
        text, _method(CORE, "DecodeEngineCore", "_send_block_assignment")
    )
    apply_ = ast.get_source_segment(
        text, _method(CORE, "PrefillEngineCore", "_apply_pending_assignments")
    )
    assert send is not None and apply_ is not None
    assert "state_slots=list(seq.state_slots)" in send
    assert "seq.state_slots = list(assignment.state_slots)" in apply_


# ── A restore must land before prefill is told to go ─────────────────────


def test_restores_are_flushed_before_the_assignment_is_sent():
    """Admission queues the restore; the forward that reads the restored slot
    is prefill's, on a stream this process cannot order against. A ZMQ send is
    not a stream barrier, so the copy has to be finished -- not merely issued --
    before the assignment goes out."""
    src = ast.get_source_segment(
        CORE.read_text(),
        _method(CORE, "DecodeEngineCore", "_process_engine_step"),
    )
    assert src is not None
    flush = src.index("_flush_state_maintenance()")
    send = src.index("_send_block_assignment(seq)")
    assert flush < send, "the flush must precede the sends, not follow them"


def test_the_flush_waits_for_the_copies():
    """Issuing the copy is not enough: the reader is in another process, and
    CUDA IPC shares memory rather than stream order. The host wait is what
    orders them -- it returns, then the assignment is sent, then prefill
    launches."""
    src = ast.get_source_segment(
        RUNNER.read_text(),
        _method(RUNNER, "RapidServeModelRunner", "flush_state_maintenance"),
    )
    assert src is not None
    assert ".synchronize()" in src


def test_the_flush_does_not_wait_on_the_whole_device():
    """`torch.cuda.synchronize()` would also drain `_decode_streams`, where
    this process's in-flight decode batch runs -- the very work rapidserve
    overlaps with prefill. The wait has to cover the copies and no more."""
    fn = _method(RUNNER, "RapidServeModelRunner", "flush_state_maintenance")
    # Structural, not textual: the comment above the wait names the call it is
    # warning against, and a substring check cannot tell prose from code.
    device_wide = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "synchronize"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "cuda"
    ]
    assert not device_wide, "torch.cuda.synchronize() drains every stream"


def test_the_flush_drains_rather_than_peeks():
    """`take_state_maintenance_ops` is a drain, so ops taken here are not taken
    again by schedule(); peeking would run every copy twice."""
    src = ast.get_source_segment(
        CORE.read_text(),
        _method(CORE, "DecodeEngineCore", "_flush_state_maintenance"),
    )
    assert src is not None
    assert "take_state_maintenance_ops()" in src


# ── The IPC payload distinguishes "not shared" from "too old" ────────────


class TestSharedPoolHandles:
    @staticmethod
    def _mod():
        return pytest.importorskip("atom.model_engine.ipc_utils")

    def test_a_producer_with_no_paged_pool_is_not_an_error(self):
        """V4's case: `paged_pool_bytes` is zero, so the carved pool is empty
        and the runner never assigns `kv_cache`. None here means "there was
        none", which the consumer handles by keeping its own empty one."""
        assert self._mod().import_kv_cache({"kv_cache": None}) is None

    def test_a_declared_name_the_producer_omitted_raises(self):
        """The silent failure this exists to prevent: decode keeps the tensor
        it just zeroed while prefill writes another. Shapes match, layout
        matches, and the indexer simply finds nothing."""
        with pytest.raises(KeyError, match="v4_csa_idx_kv"):
            self._mod().import_shared_pools({"kv_cache": None}, ["v4_csa_idx_kv"])

    def test_a_name_the_producer_had_no_tensor_for_is_dropped(self):
        """Declared but None: nothing to bind, and nothing to complain about."""
        assert self._mod().import_shared_pools({"x": None}, ["x"]) == {}


# ── A sharing backend has to name what it shares ─────────────────────────


def test_a_sharing_backend_names_something():
    """`shares_per_req_cache` without `shared_runner_attrs` shares an empty
    set, which for a backend holding its bytes outside `kv_cache` is the same
    as not sharing at all -- and reads as success."""
    offenders = []
    for src in sorted(ATTENTIONS.rglob("*.py")):
        tree = ast.parse(src.read_text(), filename=str(src))
        for cls in ast.walk(tree):
            if not isinstance(cls, ast.ClassDef):
                continue
            methods = {f.name: f for f in cls.body if isinstance(f, ast.FunctionDef)}
            shares = methods.get("shares_per_req_cache")
            if shares is None or not any(
                isinstance(n, ast.Return)
                and isinstance(n.value, ast.Constant)
                and n.value.value is True
                for n in ast.walk(shares)
            ):
                continue
            if "shared_runner_attrs" not in methods:
                offenders.append(f"{src.name}:{cls.name}")
    assert not offenders, (
        f"{offenders} share their per-request cache but declare no "
        "shared_runner_attrs"
    )


def test_v4_shares_only_names_it_publishes():
    """A typo here is a tensor nobody shares and nobody reports."""
    src = (ATTENTIONS / "deepseek_v4_attn.py").read_text()
    tree = ast.parse(src)
    declared = {
        n.value
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
        for f in cls.body
        if isinstance(f, ast.FunctionDef) and f.name == "shared_runner_attrs"
        # Skip the docstring: it is the function's first statement and is a
        # string Constant like every name below it.
        for n in ast.walk(ast.Module(body=f.body[1:], type_ignores=[]))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    }
    published = {
        k.value
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef)
        for f in cls.body
        if isinstance(f, ast.FunctionDef)
        and f.name in ("allocate_kv_cache_tensors", "allocate_per_req_cache")
        for n in ast.walk(f)
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict)
        for k in n.value.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    # PER_REQ_POOL_ATTR reaches the dict as a Name, not a literal; it is
    # covered by test_deepseek_v4_shares_and_checks_the_size.
    assert declared <= published | {"per_req_pool"}, (
        f"{sorted(declared - published)} are declared shared but never "
        "published as runner attributes"
    )

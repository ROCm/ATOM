"""SeqView identity contract.

ATOM's offload scheduler stores the seq object and compares identity to detect
that a request id was reused. Get this wrong and either every step looks like a
new request (load lifecycle reset forever) or a genuinely new request inherits
the previous one's pending load.
"""

from __future__ import annotations

from types import SimpleNamespace

from atom.plugin.vllm.kv_transfer.seq_view import SeqViewRegistry


def _request(rid: str = "r1", prompt=(1, 2, 3)):
    return SimpleNamespace(request_id=rid, prompt_token_ids=list(prompt))


def test_same_request_yields_the_same_view():
    reg = SeqViewRegistry()
    req = _request()

    assert reg.get_or_create(req) is reg.get_or_create(req)


def test_reused_id_with_a_new_request_yields_a_new_view():
    reg = SeqViewRegistry()
    first = reg.get_or_create(_request("r1"))

    second = reg.get_or_create(_request("r1"))  # same id, different request

    assert second is not first, (
        "a recycled request id must present as a new seq, or the new request "
        "inherits the old one's pending load"
    )


def test_mutable_offload_state_survives_across_lookups():
    reg = SeqViewRegistry()
    req = _request()
    view = reg.get_or_create(req)
    view.offload_loaded_tokens = 256
    view.set_block_table([7, 8, 9])

    again = reg.get_or_create(req)

    assert again.offload_loaded_tokens == 256
    assert again.block_table == [7, 8, 9]


def test_prompt_tokens_are_the_key_source_not_decode_output():
    reg = SeqViewRegistry()
    req = _request(prompt=(1, 2, 3))
    req.all_token_ids = [1, 2, 3, 99, 100]  # decode has appended output
    view = reg.get_or_create(req)

    # LMCache keys come from these; letting decode output in would change a
    # prefix's key mid-request and orphan everything already stored.
    assert view.token_ids == [1, 2, 3]
    assert view.num_prompt_tokens == 3


def test_frontier_is_pushed_in_from_vllm():
    reg = SeqViewRegistry()
    view = reg.get_or_create(_request())
    assert view.num_cached_tokens == 0

    view.set_num_cached_tokens(128)

    assert view.num_cached_tokens == 128


def test_drop_forgets_the_request():
    reg = SeqViewRegistry()
    reg.get_or_create(_request("r1"))
    reg.drop("r1")
    assert reg.get("r1") is None and len(reg) == 0


def test_preemption_forgets_placement_but_not_what_was_stored():
    """vLLM reuses the same Request object, so the view has to be reset in place.

    Preemption hands the blocks to somebody else without telling the connector,
    and the save loop sizes its next store from exactly these two fields -- a
    stale block table plus a stale frontier is another request's KV stored under
    this request's token ids.
    """
    reg = SeqViewRegistry()
    req = _request()
    view = reg.get_or_create(req)
    view.set_block_table([7, 8, 9])
    view.set_num_cached_tokens(384)
    view.offload_loaded_tokens = 256
    view.offload_handoff_boundary_tokens = 256
    view.prefix_hashes_published = True

    view.reset_for_preemption()

    assert view.block_table == []
    assert view.num_cached_tokens == 0
    assert view.offload_loaded_tokens == 0
    assert view.offload_handoff_boundary_tokens == 0
    assert view.prefix_hashes_published is False
    # Same view: the request keeps its identity, so ATOM's scheduler must not
    # see this as a recycled request id.
    assert reg.get_or_create(req) is view


def test_view_accepts_the_frozen_placement_the_chunked_scheduler_writes():
    """`__slots__` makes "the scheduler sets an attribute on the seq" a contract.

    ATOM's `ChunkedOffloadSchedulerBase` freezes a finishing request's placement
    onto the seq object so a final save can still be dispatched after vLLM has
    taken the blocks back. ATOM's own `Sequence` has a `__dict__` and absorbs
    that silently; a slotted view raises `AttributeError` instead -- out of
    `request_finished`, which runs on every completed request.

    Calling the real unbound method is the point: a fake scheduler would still
    pass if the base class grew another such attribute tomorrow.
    """
    from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase

    reg = SeqViewRegistry()
    view = reg.get_or_create(_request("r1", prompt=(1, 2, 3, 4)))
    view.set_block_table([5, 6])
    view.set_num_cached_tokens(4)

    scheduler = SimpleNamespace(
        _load_lifecycles={},
        _active_load_operations={},
        _load_failed_seqs={},
        _save_tracker={"r1": [view, 0]},
        _early_release=True,
        should_defer_free=lambda seq: False,
    )
    # Bind the real helper rather than stubbing it: it takes the seq, so a stub
    # would hide exactly the kind of attribute access this test exists to catch.
    scheduler._release_failed_load_attempt = (
        ChunkedOffloadSchedulerBase._release_failed_load_attempt.__get__(scheduler)
    )

    ChunkedOffloadSchedulerBase.request_finished(scheduler, view)

    assert view._offload_finished_cached_tokens == 4
    # Popped, because nothing was still deferring the free.
    assert "r1" not in scheduler._save_tracker


def test_preemption_forgets_the_frozen_placement_too():
    """Frozen placement is placement; a preempted request's is equally stale."""
    reg = SeqViewRegistry()
    view = reg.get_or_create(_request())
    view._offload_finished_block_ids = [7, 8, 9]
    view._offload_finished_cached_tokens = 384

    view.reset_for_preemption()

    assert not hasattr(view, "_offload_finished_block_ids")
    assert not hasattr(view, "_offload_finished_cached_tokens")


def test_view_takes_every_attribute_the_scheduler_the_plugin_builds_writes():
    """The slot list has to track every scheduler module the plugin can reach.

    `ChunkedOffloadSchedulerBase` is shared: the native path hands it an ATOM
    `Sequence`, which has a `__dict__` and absorbs any attribute, while the
    plugin path hands it this slotted view, which raises `AttributeError`. So a
    commit that adds `seq.<something> = ...` to the shared scheduler is a
    working change on one path and a crash on the other, with nothing in that
    commit's own diff to suggest it.

    That is not hypothetical: `offload_load_start_tokens` (#2154, P/D
    disaggregation) was added to the shared scheduler with no slot here, which
    made the *first successful tier load* on the plugin path raise -- the one
    code path the feature exists for, so no boot or smoke test reaches it.

    The scan walks from the class the plugin actually constructs, read out of
    `connector.py` rather than named here, then over that class's whole MRO.
    Scanning one hand-picked file instead would already be wrong today:
    `_offload_common.OffloadSchedulerMixin._maybe_start_unaligned_handoff` is a
    base of `ChunkedOffloadSchedulerBase` and writes two seq attributes from a
    file `chunked_scheduler.py`'s own text never mentions -- both happen to be
    slotted, so a file-scoped scan passes by luck, not by coverage. Deriving
    both the class and its files makes the scan follow the wiring: add a dsv4
    branch to `connector.py` and its five seq writes come into scope on their
    own (three are unslotted today), with no edit here.

    Remaining limitation, stated because it is load-bearing: the scan only sees
    writes whose base name is literally `seq`. A method that binds the view to
    some other local first would be missed, silently. Every such assignment in
    the scanned modules uses `seq` today.
    """
    import ast
    import importlib
    import pathlib

    from atom.plugin.vllm.kv_transfer.seq_view import SeqView

    def _seq_writes(path: str) -> dict[str, int]:
        found: dict[str, int] = {}
        for node in ast.walk(ast.parse(pathlib.Path(path).read_text())):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "seq"
                ):
                    found.setdefault(target.attr, node.lineno)
        return found

    # Which scheduler class does the plugin build? Read it off the source: the
    # connector imports vllm at module scope, so it cannot be imported here,
    # and hardcoding the name is exactly the assumption this test should not
    # be making.
    plugin_src = (
        pathlib.Path(__file__)
        .parents[2]
        .joinpath("atom/plugin/vllm/kv_transfer/connector.py")
    )
    plugin_tree = ast.parse(plugin_src.read_text())
    origin = {
        alias.asname or alias.name: node.module
        for node in ast.walk(plugin_tree)
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    built: set[tuple[str, str]] = set()
    for node in ast.walk(plugin_tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if not isinstance(node.value.func, ast.Name):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "_scheduler"
                and node.value.func.id in origin
            ):
                built.add((origin[node.value.func.id], node.value.func.id))

    assert built, (
        f"found no `self._scheduler = <Imported>(...)` in {plugin_src} -- the "
        "scan lost its entry point, so it is checking nothing"
    )

    written: dict[str, int] = {}
    where: dict[str, str] = {}
    scanned: set[str] = set()
    for module_name, class_name in sorted(built):
        cls = getattr(importlib.import_module(module_name), class_name)
        for base in cls.__mro__:
            source = getattr(importlib.import_module(base.__module__), "__file__", None)
            if not source or "/atom/" not in source:
                continue
            scanned.add(pathlib.Path(source).name)
            for name, line in _seq_writes(source).items():
                written.setdefault(name, line)
                where.setdefault(name, f"{pathlib.Path(source).name}:{line}")

    # Guard the scope derivation, which is the part that can fail as a whole:
    # rewire the connector through a factory or a conditional and the parse
    # above finds no class, so the file set collapses to nothing and every
    # assertion below passes vacuously. Name a file we know writes on a seq
    # rather than only checking the set is non-empty -- the MRO always
    # contributes the scheduler's own module, so non-empty is nearly free.
    assert scanned, "scope derivation produced no files to scan"
    assert "chunked_scheduler.py" in scanned, (
        "scope derivation lost the known writer; scanned "
        f"{sorted(scanned)} -- fix the derivation, not this assertion"
    )

    # Guard the scanner itself: if a refactor renames the loop variable, the
    # scan silently finds nothing and this test passes while checking nothing.
    assert "offload_loaded_tokens" in written, (
        f"found no seq attribute writes across {sorted(built)} -- the scan is "
        "broken, not the scheduler"
    )

    takeable = set(SeqView.__slots__) | {
        name
        for name in dir(SeqView)
        if isinstance(getattr(SeqView, name, None), property)
        and getattr(SeqView, name).fset is not None
    }
    missing = {name: where[name] for name in written if name not in takeable}

    assert not missing, (
        "the scheduler the plugin builds assigns these on a seq, which SeqView "
        f"cannot hold: {missing} (name -> file:line). Add each to "
        "SeqView.__slots__; do not delete the assignment -- the native path "
        "reads it back."
    )

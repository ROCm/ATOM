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

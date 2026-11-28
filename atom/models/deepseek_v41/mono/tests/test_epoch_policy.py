# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU tests of the V4.1 epoch tags (``epoch_policy``): no FlyDSL, no AITER, no GPU."""

import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "v41_epoch_policy", HERE / "epoch_policy.py"
)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)


def _const(path, name):
    m = re.search(rf"^{name}\s*=\s*(\d+)", (HERE / path).read_text(), re.MULTILINE)
    assert m, (path, name)
    return int(m.group(1))


def test_every_launch_tag_fits_below_the_epoch_bits():
    layers = 40
    moe0 = _const("runner.py", "MOE_TAG0")
    post0 = _const("runner.py", "POST_TAG0")
    head = _const("kernels/draft_head.py", "HEAD_TAG")
    gather = _const("kernels/col_gather.py", "GATHER_TAG")
    ctx = _const("kernels/draft_ctx.py", "CTX_TAG")
    tags = [layers, post0 + layers, moe0 + layers, head, gather + 8, ctx]
    assert max(tags) < 1 << ep.TAG_BITS, tags


def test_a_live_tag_is_never_zero_and_changes_every_epoch():
    for e in (0, 1, 2, 255, 256, ep.EPOCH_MASK, ep.EPOCH_MASK + 1, 123456789):
        for tag in (1, 41, 128 + 40, 192, 208, 240):
            t = ep.epoch_tag(e, tag)
            assert t & 0xFF == tag and t != 0
            assert ep.epoch_tag(e + 1, tag) != t


def test_a_previous_steps_pair_is_stale():
    for e in (1, 2, 1000, 1 << 20):
        for tag in (1, 41, 192):
            assert ep.epoch_tag(e - 1, tag) != ep.epoch_tag(e, tag)
            # zeroed memory (a full begin) never matches
            assert ep.epoch_tag(e, tag) != 0


def test_a_coded_pair_keeps_its_code_and_goes_stale_within_the_full_period():
    tag = 41
    for e in (5, 77777):
        live = ep.epoch_tag(e, tag)
        for code in (0, 1, 127, 128, 255):
            w = ep.coded_word(live, code)
            assert ep.coded_live(w, live) and ep.code_of(w) == code
            # every earlier epoch a periodic full begin has not cleared is stale
            for back in (1, 2, 255, 256, ep.FULL_EVERY - 1, ep.FULL_EVERY):
                assert not ep.coded_live(
                    ep.coded_word(ep.epoch_tag(e - back, tag), code), live
                )
    # the wrap the full period stays below
    wrap = 1 << (ep.CODED_TAG_BITS - ep.TAG_BITS)
    assert ep.coded_live(
        ep.coded_word(ep.epoch_tag(10, tag), 3), ep.epoch_tag(10 + wrap, tag)
    )
    assert ep.FULL_EVERY < wrap


def test_the_begin_is_slim_only_after_a_draft_pass_and_off_the_full_period():
    assert ep.slim_allowed(11, 10)
    # the previous step had no draft pass (capture, warm-up)
    assert not ep.slim_allowed(11, 9)
    assert not ep.slim_allowed(11, 11)
    # the policy alone: a new runner's DRAFTED starts at -1, so its first begin is full
    assert ep.slim_allowed(1, 0)
    assert not ep.slim_allowed(ep.FULL_EVERY, ep.FULL_EVERY - 1)
    assert ep.slim_allowed(ep.FULL_EVERY + 1, ep.FULL_EVERY)

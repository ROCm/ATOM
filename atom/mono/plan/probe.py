# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Timing probes of a layer kernel: ablations that keep a build's work and change
what it waits for or reads. For measurement only: a probe build's values are not
the model's (the harness checks it does not hang or fault); an empty probe string
is the production build, unchanged.

A probe string is ``;``-separated terms:

``skip=g+g``   a consumer of these hand-off groups (``GROUPS`` or a region name) loads
               the mailbox once and goes on, whatever tag it reads (stale data, the
               same access pattern): the group's dependency wait is gone, its round trip stays
``exit=stage`` the kernel returns after this stage (``STAGES``), every CTA at once
``loop[=r+r]`` the all-reduce pushes of ``attn`` / ``ffn`` (default both) land in this rank's
               own buffer (slot = the wave's source rank) instead of the peers': no xGMI,
               the pushes' writes and the polls stay
``nb=k+k``     these weight / cache reads (``KINDS``; ``moeu`` / ``moed``: a routed expert's
               up-gate / down, ``shu`` / ``shd`` the shared one's) go through zero-sized buffers: the
               loads issue, no byte comes from HBM
``pin``        the router's (routing key, sigmoid) pairs come from a table (``tl`` points at
               it: the real step's ``scores`` copy-0 image, [layer][token][expert][4] words
               = (key, tag, sigmoid, tag)), and a consumer that skips the ``scores`` wait
               reads the table: every variant routes as the real step did
"""

from dataclasses import dataclass

GROUPS = {
    "k1i": ("k1.qkv", "k1.hdone", "k1.x8s"),  # K1's own GEMV -> head / norm hand-offs
    "iscore": ("iscore", "sel_long", "sel_cp"),  # K1 scores -> selection
    "k1done": ("k1.rdone",),  # K1's residual -> router
    "sp": ("sp_o", "sp_m", "sp_l"),  # split -> merge
    "mg": ("attnq_s",),  # merge -> o_proj
    "ar1": ("attn", "a", "a_ag"),  # o_proj -> all-reduce -> router
    "scr": (
        "scores",
    ),  # router -> every CTA's routing (with ``pin``: read from the table)
    "xn": ("xn8", "xsc", "xndone"),  # router -> up / gate: the normalized input
    "rt": ("scores", "xn8", "xsc", "xndone"),  # both
    "mid": ("mid", "ugdone"),  # up / gate -> down
    "ar2": ("ffn", "ffn_ag"),  # down -> all-reduce -> layer end
}
STAGES = ("start", "k1", "sel", "split", "merge", "o", "router")
KINDS = ("moeu", "moed", "shu", "shd", "qkv", "o", "gate", "kv", "idx")
# a probe may name a group: ``moe`` = both routed halves, ``sh`` = both shared halves
KIND_GROUPS = {"moe": ("moeu", "moed"), "sh": ("shu", "shd")}


@dataclass(frozen=True)
class Probe:
    skip: frozenset = frozenset()
    exit: str = ""
    loop: frozenset = frozenset()
    nb: frozenset = frozenset()
    pin: bool = False

    def __bool__(self):
        return bool(self.skip or self.exit or self.loop or self.nb or self.pin)


def parse(text: str) -> Probe:
    skip, exit_, loop, nb, pin = set(), "", set(), set(), False
    for term in filter(None, (t.strip() for t in text.split(";"))):
        key, _, val = term.partition("=")
        if key == "skip":
            for g in val.split("+"):
                skip.update(GROUPS.get(g, (g,)))
        elif key == "exit":
            assert val in STAGES, val
            exit_ = val
        elif key == "loop":
            loop.update(val.split("+") if val else ("attn", "ffn"))
        elif key == "nb":
            for k in val.split("+"):
                assert k in KINDS or k in KIND_GROUPS, val
                nb.update(KIND_GROUPS.get(k, (k,)))
        elif key == "pin":
            pin = True
        else:
            raise ValueError(f"unknown probe term {term!r}")
    return Probe(frozenset(skip), exit_, frozenset(loop), frozenset(nb), pin)

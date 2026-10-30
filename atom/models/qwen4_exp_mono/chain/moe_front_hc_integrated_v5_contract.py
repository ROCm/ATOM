"""Explicit HC phase ABI: static layer binding, three dynamic arguments."""

from typing import NamedTuple


class HcStatic(NamedTuple):
    weight: int
    raw: int
    router: int
    logits: int
    mixed: int
    payload: int
    ids: int
    weights: int
    sorted_ids: int
    sorted_weights: int
    experts: int
    valid: int
    m_indices: int
    reverse: int
    out: int
    flags: int
    diag: int

    def pack(self, part, normed, stride):
        return (
            part,
            normed,
            self.weight,
            self.raw,
            stride,
            self.router,
            self.logits,
            self.mixed,
            self.payload,
            self.ids,
            self.weights,
            self.sorted_ids,
            self.sorted_weights,
            self.experts,
            self.valid,
            self.m_indices,
            self.reverse,
            self.out,
            self.flags,
            self.diag,
        )

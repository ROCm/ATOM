"""Addressing an image, after measurement corrected what addressing means.

The eight image layers carry identical specs, so vLLM collects them into ONE
kv cache group -- read off a running engine as `kv cache group sizes
[256, 256]`, the proxy's and theirs. One block id therefore names one block
in EACH of the eight layers, 8 x 739,840 B, which is one whole image.

An earlier version of this file read the id list as eight separate blocks.
Nothing downstream distinguishes the two readings -- eight ids and one id are
both "a list of ints" -- and the wrong one scatters an image across eight
unrelated prefixes, so the shape is checked where it enters.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.v41_state_image import (
    V41ImageExporter,
    _as_block_id,
    sizing_from_geometry,
)

UNIT, SLOT = 739840, 5276672


def sizing():
    return sizing_from_geometry(SimpleNamespace(paged_bytes=UNIT, state_bytes=SLOT))


def test_one_block_id_is_one_image():
    assert _as_block_id([17], sizing()) == 17
    assert _as_block_id(17, sizing()) == 17


def test_the_earlier_eight_id_reading_is_refused_by_name():
    """It is the shape this file used to accept, so the message says so."""
    with pytest.raises(ValueError, match="earlier, wrong"):
        _as_block_id(list(range(8)), sizing())


def test_a_negative_block_id_is_passed_through_not_refused():
    """vLLM writes -1 into a checkpoint column that has none this step, and
    `NULL_BLOCK_ID` is 0. Both are "nothing to copy", which the caller
    declines -- raising here would turn the common case into an error."""
    assert _as_block_id([-1], sizing()) == -1
    assert _as_block_id([0], sizing()) == 0


class _Copier:
    def __init__(self):
        self.calls = []
        self.stored = self.restored = 0

    def run(self, stores, restores, descriptor_slot=0):
        self.calls.append((list(stores), list(restores)))


def exporter(c=None):
    return V41ImageExporter(sizing(), c or _Copier())


def test_export_copies_the_rows_vllm_asked_for():
    c = _Copier()
    e = exporter(c)
    e.export(None, rows=[(1, 100), (2, 200)])
    (stores, restores), = c.calls
    assert stores == [(1, 100), (2, 200)] and restores == []
    assert e.stored == 2


def test_a_null_row_is_declined_not_written():
    """vLLM's NULL_BLOCK_ID is 0 and -1 is "no checkpoint this step" -- the
    common case. Writing anyway scribbles on block 0, which belongs to
    somebody."""
    c = _Copier()
    e = exporter(c)
    e.export(None, rows=[(1, 0), (2, -1)])
    assert c.calls == [] and e.declined_null == 2 and e.stored == 0


def test_a_step_with_nothing_to_store_does_not_call_the_copier():
    c = _Copier()
    exporter(c).export(None, rows=[])
    assert c.calls == []


def test_export_without_rows_says_what_is_missing():
    with pytest.raises(TypeError, match="rows="):
        exporter().export(None)


def test_restore_is_the_mirror_and_counts_what_it_read():
    c = _Copier()
    e = exporter(c)
    assert e.restore([(4, 300), (5, 0)]) == 1
    (stores, restores), = c.calls
    assert stores == [] and restores == [(4, 300)]
    assert e.restored == 1 and e.declined_null == 1


# --- the copier itself, on CPU tensors ----------------------------------


def _copier_on_cpu(num_blocks=4, max_batch=4):
    import torch

    from atom.plugin.vllm.v41_state_image import V41ImageCopier

    s = sizing()
    state = torch.zeros((8, s.slot_bytes), dtype=torch.uint8)
    layers = [torch.zeros((num_blocks, s.unit_bytes), dtype=torch.uint8)
              for _ in range(s.units)]
    return V41ImageCopier(s, state, layers, max_batch=max_batch), state, layers


def test_the_copier_plans_against_the_layers_it_was_given():
    c, _, _ = _copier_on_cpu()
    assert c.plan.num_spans >= 1


def test_too_few_bound_layers_is_refused():
    """Eight layers hold an image; seven hold a truncated one and nothing
    downstream would say so."""
    import torch

    from atom.plugin.vllm.v41_state_image import V41ImageCopier

    s = sizing()
    state = torch.zeros((2, s.slot_bytes), dtype=torch.uint8)
    layers = [torch.zeros((2, s.unit_bytes), dtype=torch.uint8) for _ in range(7)]
    with pytest.raises(ValueError, match="image layers bound"):
        V41ImageCopier(s, state, layers)


def test_an_undersized_layer_is_refused():
    import torch

    from atom.plugin.vllm.v41_state_image import V41ImageCopier

    s = sizing()
    state = torch.zeros((2, s.slot_bytes), dtype=torch.uint8)
    layers = [torch.zeros((2, s.unit_bytes), dtype=torch.uint8) for _ in range(8)]
    layers[3] = torch.zeros((2, s.unit_bytes - 1), dtype=torch.uint8)
    with pytest.raises(ValueError, match="short of"):
        V41ImageCopier(s, state, layers)


def test_a_block_outside_what_vllm_allocated_is_refused():
    c, _, _ = _copier_on_cpu(num_blocks=4)
    with pytest.raises(IndexError, match="outside the 4"):
        c.run([(0, 9)], [])


def test_a_slot_outside_the_state_arena_is_refused():
    c, _, _ = _copier_on_cpu()
    with pytest.raises(IndexError, match="slot 99"):
        c.run([(99, 1)], [])


# --- the round trip, on real bytes --------------------------------------


def test_an_image_survives_a_round_trip_byte_for_byte():
    """The claim the whole mechanism rests on, checked on actual bytes.

    Everything else here is shape and bookkeeping. This writes a slot into
    the image layers, scribbles over the slot, reads the image back, and
    compares -- which is the only thing that distinguishes a working
    checkpoint from one that moves plausible-looking garbage.
    """
    import torch

    c, state, layers = _copier_on_cpu(num_blocks=4)
    s = sizing()
    want = torch.randint(0, 256, (s.slot_bytes,), dtype=torch.uint8)
    state[2] = want
    c.run([(2, 1)], [])
    state[2] = 0                      # the next forward would do this
    assert not torch.equal(state[2], want)
    c.run([], [(2, 1)])
    assert torch.equal(state[2], want)


def test_a_restore_reads_the_block_it_was_given_not_another():
    """Two images, two blocks: reading one must not return the other."""
    import torch

    c, state, _ = _copier_on_cpu(num_blocks=4)
    s = sizing()
    a = torch.randint(0, 256, (s.slot_bytes,), dtype=torch.uint8)
    b = torch.randint(0, 256, (s.slot_bytes,), dtype=torch.uint8)
    state[0], state[1] = a, b
    c.run([(0, 1), (1, 2)], [])
    state[0] = state[1] = 0
    c.run([], [(0, 2)])               # slot 0 restored from B's block
    assert torch.equal(state[0], b)
    assert not torch.equal(state[0], a)


def test_an_image_lands_across_all_eight_layers():
    """Not in one: a layer left untouched means a truncated image, and the
    round trip alone would not notice if the copier silently used fewer."""
    import torch

    c, state, layers = _copier_on_cpu(num_blocks=4)
    s = sizing()
    state[0] = torch.randint(1, 256, (s.slot_bytes,), dtype=torch.uint8)
    c.run([(0, 3)], [])
    touched = [int(layers[i][3].max()) for i in range(s.units)]
    assert all(t > 0 for t in touched), f"untouched layers: {touched}"


def test_a_store_does_not_touch_a_neighbouring_block():
    import torch

    c, state, layers = _copier_on_cpu(num_blocks=4)
    s = sizing()
    state[0] = torch.randint(1, 256, (s.slot_bytes,), dtype=torch.uint8)
    c.run([(0, 2)], [])
    for i in range(s.units):
        for b in (0, 1, 3):
            assert int(layers[i][b].max()) == 0, f"layer {i} block {b} was written"

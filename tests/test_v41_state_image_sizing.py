"""The image sizing, and the disagreement it exists to refuse.

An image is cut into PAGE units by native's own copier, which validates
against `PagedStateCheckpointSpec`. The scheduler half has to know the unit
count before a spec exists, to register that many layers, so the number is
computed twice in two processes from two objects. These pin that the two
agree, and that a disagreement is named rather than discovered at the first
store.
"""

from types import SimpleNamespace

import pytest

from atom.plugin.vllm.v41_state_image import (
    V41ImageSizing,
    assert_agrees_with_spec,
    image_layer_name,
    is_image_layer,
    sizing_from_geometry,
    sizing_from_spec,
)

# DeepSeek-V4.1-Flash, read off the registration line rather than config:
# 739,840 B per 256-token block over five planes, 5,276,672 B per state slot.
UNIT, SLOT = 739840, 5276672


def geometry(paged=UNIT, state=SLOT):
    return SimpleNamespace(paged_bytes=paged, state_bytes=state)


def spec(unit=UNIT, slot=SLOT, image=None, layout="dsv41-bf16"):
    image = slot if image is None else image
    return SimpleNamespace(
        page_unit_bytes=unit, slot_bytes=slot, image_bytes=image,
        layout_id=layout,
        units_per_checkpoint=(image + unit - 1) // unit,
    )


def test_v41_takes_eight_units():
    """The number this whole mechanism is sized by, stated once.

    5,276,672 / 739,840 is 7.13, so eight. One unit is exactly one proxy
    block at 256 tokens, which is what makes this simpler than the
    DeepSeek-V4 case it follows.
    """
    s = sizing_from_geometry(geometry())
    assert s.units == 8
    assert s.unit_bytes == UNIT
    assert s.padded_bytes == 8 * UNIT >= SLOT


def test_the_padding_is_reported_not_hidden():
    """642,048 B per image are real pool bytes nothing reads."""
    s = sizing_from_geometry(geometry())
    assert s.padded_bytes - s.image_bytes == 8 * UNIT - SLOT


def test_the_two_halves_agree_on_v41():
    assert_agrees_with_spec(sizing_from_geometry(geometry()), spec())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("page_unit_bytes", UNIT * 2),
        ("slot_bytes", SLOT + UNIT),
        ("image_bytes", SLOT - UNIT),
    ],
)
def test_a_disagreement_is_named(field, bad):
    """Each one is a different claim about what a stored image means.

    Without this they surface as a copier exception in a worker, after the
    pool is allocated -- or, when the sizes happen to stay compatible, as an
    image cut at the wrong ranges and no exception at all.
    """
    kwargs = {"unit": UNIT, "slot": SLOT}
    if field == "page_unit_bytes":
        kwargs["unit"] = bad
    elif field == "slot_bytes":
        kwargs["slot"] = bad
    else:
        kwargs["image"] = bad
    with pytest.raises(ValueError, match="disagrees"):
        assert_agrees_with_spec(sizing_from_geometry(geometry()), spec(**kwargs))


def test_an_image_larger_than_its_slot_is_refused():
    with pytest.raises(ValueError, match="exceeds"):
        V41ImageSizing(UNIT, SLOT + 1, SLOT, 8, "x").check()


def test_too_few_units_is_refused():
    with pytest.raises(ValueError, match="short of"):
        V41ImageSizing(UNIT, SLOT, SLOT, 7, "x").check()


def test_one_unit_too_many_is_refused():
    """Not cosmetic: a spare unit is 739,840 B per image of pool nothing uses,
    and it would make the layer count disagree with the copier's."""
    with pytest.raises(ValueError, match="one too many"):
        V41ImageSizing(UNIT, SLOT, SLOT, 9, "x").check()


def test_an_image_that_divides_exactly_needs_no_slack():
    s = sizing_from_geometry(geometry(state=UNIT * 4))
    assert s.units == 4 and s.padded_bytes == s.image_bytes


def test_layer_names_are_positional_and_recognisable():
    """Unit i is layer i. The order is the key space: renumbering makes every
    stored image address the wrong bytes."""
    names = [image_layer_name(i) for i in range(8)]
    assert len(set(names)) == 8
    assert all(is_image_layer(n) for n in names)
    assert not is_image_layer("atom.dsv41.proxy")
    assert not is_image_layer("model.layers.0.self_attn.attn")


def test_a_narrowed_image_takes_fewer_units():
    """V4.1 does not narrow today -- `checkpoint_image_bytes` returns the whole
    slot -- but the sizing must follow if it ever does."""
    s = sizing_from_geometry(geometry(), image_bytes=UNIT * 3)
    assert s.units == 3

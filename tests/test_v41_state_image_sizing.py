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


# --- the registration vLLM has to end up with ---------------------------


def _spec(nbytes=UNIT, block=256, alignment=4096):
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        shapes=((nbytes,),), dtypes=(torch.uint8,), block_size=block,
        num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=alignment,
        tp_replicated=False,
    )


def test_the_spec_holds_exactly_one_unit():
    from atom.plugin.vllm.v41_state_image import image_kv_cache_spec

    s = sizing_from_geometry(geometry())
    spec = image_kv_cache_spec(s, 256, 4096)
    assert spec.page_size_bytes == UNIT


def test_the_spec_declares_itself_checkpointable():
    """Without this the scheduler does not know these blocks can be
    checkpointed at a prefill boundary, and nothing is ever stored."""
    from atom.plugin.vllm.v41_state_image import image_kv_cache_spec

    spec = image_kv_cache_spec(sizing_from_geometry(geometry()), 256, 4096)
    assert spec.num_prefill_checkpoint_blocks == 1
    assert spec.prefill_checkpoint_alignment == 4096


def test_the_state_is_per_rank_not_replicated():
    """Each TP rank holds its own ring and compressor tails; an image is that
    rank's. Claiming replication would make one rank's image stand for all."""
    from atom.plugin.vllm.v41_state_image import image_kv_cache_spec

    assert image_kv_cache_spec(sizing_from_geometry(geometry()), 256, 4096).tp_replicated is False


def test_a_complete_registration_passes():
    from atom.plugin.vllm.v41_state_image import check_registration, expected_layer_names

    s = sizing_from_geometry(geometry())
    check_registration({n: _spec() for n in expected_layer_names(s)}, s)


def test_too_few_layers_is_refused():
    """The tail of every image would be missing, and nothing else would say so."""
    from atom.plugin.vllm.v41_state_image import check_registration, expected_layer_names

    s = sizing_from_geometry(geometry())
    names = expected_layer_names(s)[:-1]
    with pytest.raises(ValueError, match="missing="):
        check_registration({n: _spec() for n in names}, s)


def test_a_gap_in_the_numbering_is_refused():
    """Unit i goes to layer i, so a hole shifts every unit after it."""
    from atom.plugin.vllm.v41_state_image import check_registration, expected_layer_names, image_layer_name

    s = sizing_from_geometry(geometry())
    names = [n for n in expected_layer_names(s) if n != image_layer_name(3)]
    names.append(image_layer_name(99))
    with pytest.raises(ValueError, match="missing=|unexpected="):
        check_registration({n: _spec() for n in names}, s)


def test_an_undersized_layer_is_refused():
    """Every image would be truncated, silently."""
    from atom.plugin.vllm.v41_state_image import check_registration, expected_layer_names

    s = sizing_from_geometry(geometry())
    specs = {n: _spec() for n in expected_layer_names(s)}
    specs[expected_layer_names(s)[2]] = _spec(nbytes=UNIT - 1)
    with pytest.raises(ValueError, match="short of"):
        check_registration(specs, s)


def test_other_layers_are_ignored_by_the_check():
    """The proxy and the model's own layers share this dict."""
    from atom.plugin.vllm.v41_state_image import check_registration, expected_layer_names

    s = sizing_from_geometry(geometry())
    specs = {n: _spec() for n in expected_layer_names(s)}
    specs["atom.dsv41.proxy"] = _spec(nbytes=1)
    specs["model.layers.0.self_attn.attn"] = _spec(nbytes=1)
    check_registration(specs, s)


# --- the gate the wiring has to open, not the environment ---------------


def test_registration_alone_does_not_admit_a_prefix_hit(monkeypatch):
    """Measured the hard way. vLLM allocates and hashes the image blocks as
    soon as the layers are declared, whether or not anything fills them, so
    opening the prefix-caching gate on registration served a hit from
    unwritten blocks and killed the engine with `needs state at ...` on the
    second send of one prompt."""
    import atom.plugin.vllm.v41_state_image as img

    monkeypatch.setenv("ATOM_V41_STATE_IMAGES", "1")
    monkeypatch.setattr(img, "_IMAGES_WIRED", False)
    assert img.images_registered() is True
    assert img.images_can_serve() is False


def test_wiring_without_registration_does_not_admit_one_either(monkeypatch):
    import atom.plugin.vllm.v41_state_image as img

    monkeypatch.delenv("ATOM_V41_STATE_IMAGES", raising=False)
    monkeypatch.setattr(img, "_IMAGES_WIRED", True)
    assert img.images_can_serve() is False


def test_both_together_admit_one(monkeypatch):
    import atom.plugin.vllm.v41_state_image as img

    monkeypatch.setenv("ATOM_V41_STATE_IMAGES", "1")
    monkeypatch.setattr(img, "_IMAGES_WIRED", True)
    assert img.images_can_serve() is True


def test_the_prefix_caching_gates_ask_the_module_not_the_environment(monkeypatch):
    """Both gates -- the platform's and the runtime config's -- have to read
    the same answer, or one of them admits a hit the other would refuse."""
    import atom.models.deepseek_v41.config as cfg
    import atom.plugin.vllm.platform as plat
    import atom.plugin.vllm.v41_state_image as img

    monkeypatch.setenv("ATOM_V41_STATE_IMAGES", "1")
    monkeypatch.setattr(img, "_IMAGES_WIRED", False)
    assert plat._v41_state_images_enabled() is False
    assert cfg._v41_state_images_enabled() is False
    monkeypatch.setattr(img, "_IMAGES_WIRED", True)
    assert plat._v41_state_images_enabled() is True
    assert cfg._v41_state_images_enabled() is True

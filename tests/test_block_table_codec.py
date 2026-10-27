# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/block_table_codec.py and the BlockTable version
# it is built on.

import array
import copy
import pickle
from types import SimpleNamespace

import pytest

from atom.model_engine.block_table_codec import (
    BlockTableDelta,
    BlockTableDeltaDecoder,
    BlockTableDeltaEncoder,
)
from atom.model_engine.sequence import BlockTable, new_block_table


def make_batch(req_ids, rows):
    """The two attributes the codec reads, plus one it must carry through."""
    return SimpleNamespace(req_ids=list(req_ids), block_tables=list(rows), step=0)


def round_trip(encoder, decoder, req_ids, rows):
    """Encode one step and decode it as a worker would, over a pickle.

    Returns what travelled and what the worker ends up with. Decoding
    overwrites `block_tables` in place, so the wire form is kept first.
    """
    delivered = pickle.loads(pickle.dumps(encoder.encode(make_batch(req_ids, rows))))
    wire = delivered.block_tables
    return wire, decoder.decode(delivered)


# ── BlockTable version ─────────────────────────────────────────────────────


class TestBlockTableVersion:
    def test_appends_keep_the_version(self):
        bt = new_block_table([1, 2, 3])
        before = bt.version
        bt.append(4)
        bt.extend([5, 6])
        bt += array.array("i", [7])
        assert bt.version == before
        assert list(bt) == [1, 2, 3, 4, 5, 6, 7]

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda bt: bt.__setitem__(0, 99), id="setitem"),
            pytest.param(lambda bt: bt.__delitem__(slice(None)), id="clear"),
            pytest.param(lambda bt: bt.__delitem__(0), id="del-one"),
            pytest.param(lambda bt: bt.pop(), id="pop"),
            pytest.param(lambda bt: bt.insert(0, 99), id="insert"),
            pytest.param(lambda bt: bt.remove(2), id="remove"),
            pytest.param(lambda bt: bt.reverse(), id="reverse"),
        ],
    )
    def test_disturbing_ids_redraws_the_version(self, mutate):
        bt = new_block_table([1, 2, 3])
        before = bt.version
        mutate(bt)
        assert bt.version != before

    def test_versions_are_never_reused(self):
        versions = {new_block_table().version for _ in range(100)}
        assert len(versions) == 100

    @pytest.mark.parametrize(
        "duplicate",
        [
            pytest.param(lambda bt: pickle.loads(pickle.dumps(bt)), id="pickle"),
            pytest.param(copy.copy, id="copy"),
            pytest.param(copy.deepcopy, id="deepcopy"),
        ],
    )
    def test_a_duplicate_is_a_new_table_with_a_new_version(self, duplicate):
        """Two tables answering for one version would let a stale prefix stand."""
        bt = new_block_table([1, 2, 3])
        other = duplicate(bt)
        assert isinstance(other, BlockTable)
        assert list(other) == [1, 2, 3]
        assert other.version != bt.version

    def test_is_still_an_int32_array(self):
        bt = new_block_table([1, 2, 3])
        assert isinstance(bt, array.array)
        assert (bt.typecode, bt.itemsize) == ("i", 4)
        assert memoryview(bt).tobytes() == array.array("i", [1, 2, 3]).tobytes()


# ── steady state ───────────────────────────────────────────────────────────


class TestSteadyState:
    def test_first_step_is_whole_then_only_appends_travel(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table([1, 2, 3]), new_block_table([4, 5])]

        wire, decoded = round_trip(encoder, decoder, [7, 8], rows)
        assert list(wire.base_lengths) == [0, 0]
        assert list(wire.tail_values) == [1, 2, 3, 4, 5]
        assert [list(r) for r in decoded.block_tables] == [[1, 2, 3], [4, 5]]

        rows[0].append(6)
        rows[1].append(7)
        wire, decoded = round_trip(encoder, decoder, [7, 8], rows)
        assert list(wire.base_lengths) == [3, 2]
        assert list(wire.tail_values) == [6, 7]
        assert [list(r) for r in decoded.block_tables] == [[1, 2, 3, 6], [4, 5, 7]]

    def test_a_step_that_appends_nothing_sends_nothing(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table([1, 2, 3])]
        round_trip(encoder, decoder, [7], rows)

        wire, decoded = round_trip(encoder, decoder, [7], rows)
        assert wire.tail_values.size == 0
        assert [list(r) for r in decoded.block_tables] == [[1, 2, 3]]

    def test_decoded_rows_are_int32_arrays(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        _, decoded = round_trip(encoder, decoder, [7], [new_block_table([1, 2])])
        (row,) = decoded.block_tables
        assert isinstance(row, array.array)
        assert (row.typecode, row.itemsize) == ("i", 4)

    def test_other_batch_attributes_survive_and_the_source_is_untouched(self):
        encoder = BlockTableDeltaEncoder()
        batch = make_batch([7], [new_block_table([1, 2])])
        wire = encoder.encode(batch)
        assert wire is not batch
        assert wire.step == batch.step
        assert isinstance(batch.block_tables[0], BlockTable)

    def test_the_wire_shrinks_with_context(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table(range(6000)) for _ in range(50)]
        whole = len(pickle.dumps(make_batch(range(50), rows)))
        round_trip(encoder, decoder, range(50), rows)
        for row in rows:
            row.append(6000)
        delta = len(pickle.dumps(encoder.encode(make_batch(range(50), rows))))
        assert delta * 100 < whole


# ── a table that was disturbed ─────────────────────────────────────────────


class TestDisturbedTables:
    def test_a_prefix_rewritten_in_place_is_resent_whole(self):
        """`disown_claimed_prefix` swaps ids in place, keeping the length.

        Nothing about the row's identity, length, or last id changes, so only
        the version distinguishes this from a step that appended nothing.
        """
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table([1, 2, 3])]
        round_trip(encoder, decoder, [7], rows)

        rows[0][0] = 99  # the privatised block
        wire, decoded = round_trip(encoder, decoder, [7], rows)
        assert list(wire.base_lengths) == [0]
        assert [list(r) for r in decoded.block_tables] == [[99, 2, 3]]

    def test_a_cleared_and_refilled_table_is_resent_whole(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table([1, 2, 3])]
        round_trip(encoder, decoder, [7], rows)

        del rows[0][:]
        rows[0].extend([4, 5, 6, 7])
        wire, decoded = round_trip(encoder, decoder, [7], rows)
        assert list(wire.base_lengths) == [0]
        assert [list(r) for r in decoded.block_tables] == [[4, 5, 6, 7]]

    def test_a_replacement_table_for_the_same_request_is_resent_whole(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        round_trip(encoder, decoder, [7], [new_block_table([1, 2, 3])])

        wire, decoded = round_trip(encoder, decoder, [7], [new_block_table([8, 9])])
        assert list(wire.base_lengths) == [0]
        assert [list(r) for r in decoded.block_tables] == [[8, 9]]

    def test_a_request_absent_for_a_step_is_resent_whole(self):
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = {7: new_block_table([1, 2]), 8: new_block_table([3, 4])}
        round_trip(encoder, decoder, [7, 8], list(rows.values()))
        round_trip(encoder, decoder, [8], [rows[8]])  # 7 preempted

        wire, decoded = round_trip(encoder, decoder, [7, 8], list(rows.values()))
        assert list(wire.base_lengths) == [0, 2]
        assert [list(r) for r in decoded.block_tables] == [[1, 2], [3, 4]]


# ── rows the previous batch still holds ────────────────────────────────────


class TestRetainedRows:
    def test_growing_a_row_does_not_edit_the_batch_before_it(self):
        """The token processor can still be reading the previous batch."""
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table([1, 2])]
        _, first = round_trip(encoder, decoder, [7], rows)
        retained = first.block_tables[0]

        rows[0].append(3)
        _, second = round_trip(encoder, decoder, [7], rows)
        assert list(retained) == [1, 2]
        assert list(second.block_tables[0]) == [1, 2, 3]


# ── batches the encoder cannot account for ─────────────────────────────────


class TestFallback:
    def test_a_batch_without_tables_passes_through(self):
        encoder = BlockTableDeltaEncoder()
        batch = make_batch([], [])
        assert encoder.encode(batch) is batch

    def test_hand_built_rows_pass_through(self):
        encoder = BlockTableDeltaEncoder()
        batch = make_batch([7], [array.array("i", [1, 2])])
        assert encoder.encode(batch) is batch

    def test_a_row_count_that_does_not_match_passes_through(self):
        encoder = BlockTableDeltaEncoder()
        batch = make_batch([7, 8], [new_block_table([1, 2])])
        assert encoder.encode(batch) is batch

    def test_a_whole_batch_resets_both_sides(self):
        """A fallback step must not leave the two caches disagreeing."""
        encoder, decoder = BlockTableDeltaEncoder(), BlockTableDeltaDecoder()
        rows = [new_block_table([1, 2])]
        round_trip(encoder, decoder, [7], rows)

        # One step the encoder cannot account for, delivered as it stands.
        plain = make_batch([7], [array.array("i", [1, 2])])
        assert encoder.encode(plain) is plain
        decoder.decode(plain)

        rows[0].append(3)
        wire, decoded = round_trip(encoder, decoder, [7], rows)
        assert list(wire.base_lengths) == [0]
        assert [list(r) for r in decoded.block_tables] == [[1, 2, 3]]


class TestDecoderRefusesNonsense:
    def test_a_missing_prefix_raises_rather_than_guessing(self):
        decoder = BlockTableDeltaDecoder()
        batch = make_batch([7], [])
        batch.block_tables = BlockTableDelta(
            base_lengths=[2], tail_offsets=[0, 1], tail_values=[9]
        )
        with pytest.raises(RuntimeError, match="missing block-table prefix"):
            decoder.decode(batch)

    def test_a_row_count_that_does_not_match_raises(self):
        decoder = BlockTableDeltaDecoder()
        batch = make_batch([7, 8], [])
        batch.block_tables = BlockTableDelta(
            base_lengths=[0], tail_offsets=[0, 1], tail_values=[9]
        )
        with pytest.raises(RuntimeError, match="2 requests"):
            decoder.decode(batch)


# ── the RPC seam ───────────────────────────────────────────────────────────


class TestRpcSeam:
    def test_only_forward_is_encoded(self):
        encoder = BlockTableDeltaEncoder()
        batch = make_batch([7], [new_block_table([1, 2])])
        assert encoder.encode_rpc("get_num_blocks", (batch,)) == (batch,)
        assert encoder.encode_rpc("forward", (batch,))[0] is not batch

    def test_trailing_arguments_are_preserved(self):
        encoder = BlockTableDeltaEncoder()
        batch = make_batch([7], [new_block_table([1, 2])])
        assert encoder.encode_rpc("forward", (batch, "extra"))[1] == "extra"

    def test_an_argumentless_forward_is_left_alone(self):
        assert BlockTableDeltaEncoder().encode_rpc("forward", ()) == ()
        assert BlockTableDeltaDecoder().decode_rpc("forward", []) == []

"""Host-only request ownership and prefix admission for continuous serving."""

import heapq
import math
from dataclasses import dataclass


class RequestOwners:
    """A STATE slot keeps its owner until the slot gets a new request epoch."""

    def __init__(self, requests_per_rank=64):
        self.limit = requests_per_rank
        self.slots = {}

    def bind(self, requests):
        for request_id, slot in requests:
            previous = self.slots.get(slot)
            if previous is not None and previous[0] == request_id:
                continue
            moved = next(
                (old for old, (rid, _) in self.slots.items() if rid == request_id), None
            )
            if moved is not None:
                _, owner = self.slots.pop(moved)
                self.slots[slot] = (request_id, owner)
                continue
            counts = [0] * 4
            for other_slot, (_, owner) in self.slots.items():
                # Prefill arrives in small batches while prior requests remain
                # resident. Account for every registered STATE slot, including
                # slots absent from this batch. Replacing a slot releases its
                # old epoch before choosing the new owner.
                if other_slot != slot:
                    counts[owner] += 1
            owner = min(range(4), key=lambda rank: (counts[rank], rank))
            if counts[owner] >= self.limit:
                raise ValueError("The persistent request-owner capacity is exhausted")
            self.slots[slot] = (request_id, owner)
        result = {rid: self.slots[slot][1] for rid, slot in requests}
        counts = [0] * 4
        for _, owner in self.slots.values():
            counts[owner] += 1
        if max(counts, default=0) > self.limit:
            raise ValueError("Resident STATE owner allocation exceeds capacity")
        return result


def prefix_lengths(request_ids, lengths, owners, confidence, policy, capacity=192):
    """Spend each owner's capacity on nonempty prefixes, using ready old data.

    A missing old confidence vector gets the fixed calibration prior. Admission
    never changes the proposal width, reservation, or status generation.
    """
    result = dict(zip(request_ids, lengths, strict=True))
    table = policy["head_probabilities"]
    for rank in range(4):
        ids = [rid for rid in request_ids if owners[rid] == rank]
        if sum(result[rid] for rid in ids) <= capacity:
            continue
        if len(ids) > capacity:
            raise ValueError("Every owner request must retain its anchor")
        predictions = {}
        for rid in ids:
            head = confidence.get(rid)
            probs = [
                (
                    table[j][min(9, max(0, int(float(head[j]) * 10)))]
                    if head is not None
                    and math.isfinite(float(head[j]))
                    and 0 <= float(head[j]) <= 1
                    else sum(table[j]) / len(table[j])
                )
                for j in range(5)
            ]
            # Calibration describes prefix rewards. Preserve prefix monotonicity.
            predictions[rid] = [min(probs[: j + 1]) for j in range(5)]
        maximum = {rid: result[rid] for rid in ids}
        heap = [(-predictions[rid][0], rid, 1) for rid in ids if maximum[rid] > 1]
        heapq.heapify(heap)
        for rid in ids:
            result[rid] = 1
        for _ in range(capacity - len(ids)):
            if not heap:
                break
            _, rid, column = heapq.heappop(heap)
            result[rid] += 1
            if result[rid] < maximum[rid]:
                heapq.heappush(heap, (-predictions[rid][column], rid, column + 1))
    return tuple(result[rid] for rid in request_ids)


@dataclass(frozen=True)
class ServingRows:
    capacity: int
    rank_indices: tuple[tuple[int, ...], ...]
    packet_rows: tuple[int, ...]
    global_from_packet: tuple[int, ...]

    @classmethod
    def bind(cls, spans, owners):
        ranks = tuple(
            tuple(i for i, span in enumerate(spans) if owners[span.request_id] == rank)
            for rank in range(4)
        )
        maximum = max(sum(spans[i].length for i in indices) for indices in ranks)
        capacity = next(
            (n for n in (16, 32, 64, 96, 128, 192, 384) if n >= maximum), None
        )
        if capacity is None or max(map(len, ranks)) > 64:
            raise ValueError("Owner step exceeds the serving graph capacity")
        packet, inverse = [], [-1] * sum(span.length for span in spans)
        for indices in ranks:
            rows = [
                row
                for i in indices
                for row in range(spans[i].offset, spans[i].offset + spans[i].length)
            ]
            for row in rows:
                if inverse[row] != -1:
                    raise ValueError("An executed row has two request owners")
                inverse[row] = len(packet)
                packet.append(row)
            packet.extend([0] * (capacity - len(rows)))
        if any(row < 0 for row in inverse):
            raise ValueError("Every executed row needs a final consumer")
        return cls(capacity, ranks, tuple(packet), tuple(inverse))


def rebuild_prefix_batch(batch, selected_lengths):
    """Rebuild worker input rows; native verifier still rejects against full T5."""
    import numpy as np

    old = tuple(map(int, batch.num_scheduled_tokens))
    if len(old) != len(selected_lengths) or any(
        not 1 <= new <= prior for new, prior in zip(selected_lengths, old, strict=True)
    ):
        raise ValueError("Admission must keep a nonempty scheduled prefix")
    if tuple(selected_lengths) == old:
        return False
    source = np.asarray(batch.scheduled_tokens)
    offset, chunks = 0, []
    for prior, new in zip(old, selected_lengths, strict=True):
        chunks.append(source[offset : offset + new])
        offset += prior
    batch.scheduled_tokens = np.concatenate(chunks)
    batch.num_scheduled_tokens = np.asarray(
        selected_lengths, dtype=batch.num_scheduled_tokens.dtype
    )
    batch.total_tokens_num_decode = sum(selected_lengths)
    batch.total_tokens_num = batch.total_tokens_num_decode
    return True


def written_page_units(step, block_tables, geometry, *, compress_buffers):
    """PAGE units named by native compressor/index writers for this step.

    Window and compressor history writes live in STATE. Main prefix latents
    and index values/scales use the compression plan's position and PAGE
    table, so previously replicated prompt pages need no new publication.
    Read the current native publication's host buffer, bounded by its actual
    count. CompressPlan owns the GPU view, not a separate CPU plan or a D2H copy.
    """
    result = {span.request_id: set() for span in step.requests}
    for ratio, plan in step.plans.items():
        buffer = compress_buffers[ratio]
        if (
            buffer.gpu.data_ptr() != plan.compress_plan_gpu.data_ptr()
            or not 0 <= plan.num_compress <= plan.compress_plan_gpu.shape[0]
            or plan.num_compress > buffer.np.shape[0]
        ):
            raise ValueError("PAGE accounting needs this step's native compress buffer")
        rows = buffer.np[: plan.num_compress]
        per_page = geometry.rows_per_page(ratio)
        for _, batch_index, position, _ in rows:
            batch_index, position = int(batch_index), int(position)
            if position < 0 or batch_index < 0:
                continue
            span = step.requests[batch_index]
            unit = block_tables[batch_index][(position // ratio) // per_page]
            result[span.request_id].add(int(unit))
    return {rid: tuple(sorted(units)) for rid, units in result.items()}

"""Stable owner inputs and an explicit boundary between warmup and serving."""


def tensor_binding(tensor):
    """Every part of a tensor view that a recorded reader depends on."""
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        str(tensor.dtype),
        str(tensor.device),
    )


def graph_key(capacity, tokens, max_q_len, planned_host, embeddings):
    return (
        capacity,
        tensor_binding(tokens),
        max_q_len,
        tuple(sorted(planned_host.items())),
        tuple(
            (layer, tensor_binding(value))
            for layer, value in sorted(embeddings.items())
        ),
    )


class OwnerGraphInputs:
    """Fixed storage: a changing native publication is copied before replay.

    Token order includes all four owners, as the vocabulary embedding requires.
    Engram rows belong only to this rank. No cursor or cache state is owned here.
    The caller stages/joins the native Engram producer exactly once per forward.
    """

    def __init__(self, input_ids, embeddings, max_owner_rows=384):
        import torch

        if input_ids.ndim != 1 or not input_ids.is_contiguous():
            raise ValueError("Owner graph input must be a contiguous token vector")
        self.max_owner_rows = max_owner_rows
        self.tokens = input_ids.new_empty(4 * max_owner_rows)
        self.embeddings = {}
        for layer, value in embeddings.items():
            if (
                value.ndim != 3
                or value.shape[0] != 1
                or value.device != input_ids.device
            ):
                raise ValueError(
                    "Native Engram rows must be [1, rows, hidden] on the token device"
                )
            self.embeddings[layer] = torch.empty(
                (1, max_owner_rows, value.shape[2]),
                dtype=value.dtype,
                device=value.device,
            )

    def views(self, capacity):
        if (
            capacity not in (16, 32, 64, 96, 128, 192, 384)
            or capacity > self.max_owner_rows
        ):
            raise ValueError("Unknown owner graph capacity")
        return self.tokens[: 4 * capacity], {
            layer: value[:, :capacity] for layer, value in self.embeddings.items()
        }

    def stage(self, input_ids, packet_rows, local_rows, embeddings, capacity):
        import torch

        tokens, owned = self.views(capacity)
        if (
            input_ids.ndim != 1
            or not input_ids.is_contiguous()
            or input_ids.dtype != tokens.dtype
            or input_ids.device != tokens.device
            or packet_rows.shape != tokens.shape
            or local_rows.shape != (capacity,)
            or set(embeddings) != set(owned)
        ):
            raise ValueError("Owner graph publication changed its input contract")
        # Reading .get() is significant: EngramStagedRows waits for the native
        # producer (including its TP gather) before these copies may execute.
        torch.index_select(input_ids, 0, packet_rows, out=tokens)
        for layer, output in owned.items():
            source = embeddings.get(layer)
            if (
                source.ndim != 3
                or source.shape[0] != 1
                or source.shape[2] != output.shape[2]
                or source.dtype != output.dtype
                or source.device != output.device
            ):
                raise ValueError("Owner graph publication changed its Engram contract")
            torch.index_select(source, 1, local_rows, out=output)
        return tokens, owned


class GraphCoverage:
    """A sealed phase never learns a graph on a timed serving request.

    All ranks exchange both their miss and phase before this check. If any
    rank has sealed, none may capture, including one that read the control
    file just before the transition. Sealing is monotonic for a server life.
    """

    def __init__(self):
        self.sealed = False
        self.captures = 0
        self.misses_after_seal = 0

    def request(self, missing, phase):
        if phase not in ("warmup", "sealed"):
            raise ValueError("Owner graph phase must be warmup or sealed")
        self.sealed |= phase == "sealed"
        return bool(missing), self.sealed

    def require_capture(self, rank_requests):
        if len(rank_requests) != 4 or any(
            not isinstance(row, (tuple, list))
            or len(row) != 2
            or any(type(value) is not bool for value in row)
            for row in rank_requests
        ):
            raise ValueError("Incomplete four-rank graph coverage agreement")
        self.sealed |= any(row[1] for row in rank_requests)
        missing = any(row[0] for row in rank_requests)
        if missing and self.sealed:
            self.misses_after_seal += 1
            raise RuntimeError(
                "Unwarmed owner graph after sealing; end this measurement before capture"
            )
        return missing

    def captured(self):
        if self.sealed:
            raise RuntimeError("Cannot record an owner graph in the sealed phase")
        self.captures += 1

    def report(self):
        return {
            "sealed": self.sealed,
            "captures": self.captures,
            "misses_after_seal": self.misses_after_seal,
        }

"""Fixed CPU tensor exchange for complete four-rank graph decisions."""


def validate_request(request):
    if (
        not isinstance(request, tuple)
        or len(request) != 2
        or any(type(v) is not bool for v in request)
    ):
        raise ValueError("Graph agreement requires (missing, sealed) booleans")


class FixedGraphAgreement:
    """One persistent input and four persistent output CPU tensors.

    Preserve the individual four-rank requests consumed by GraphCoverage;
    no object serialization, no change to local miss checking or sealing.
    GPU execution and paired serving benefit remain to be validated.
    """

    def __init__(self, cpu_group):
        import torch
        import torch.distributed as dist

        if dist.get_world_size(cpu_group) != 4:
            raise ValueError("Owner graph agreement requires exactly four ranks")
        self.group = cpu_group
        self.send = torch.empty(2, dtype=torch.int32, device="cpu")
        self.receive = [torch.empty_like(self.send) for _ in range(4)]

    def exchange(self, request):
        import torch.distributed as dist

        validate_request(request)
        self.send[0] = int(request[0])
        self.send[1] = int(request[1])
        dist.all_gather(self.receive, self.send, group=self.group)
        rows = [value.tolist() for value in self.receive]
        if any(v not in (0, 1) for row in rows for v in row):
            raise RuntimeError("Invalid graph agreement tensor values")
        return [tuple(bool(v) for v in row) for row in rows]

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The RDMA weight stream between real processes, over a real process group.

The rest of the receiver's suite stubs the group and replaces dist.broadcast,
so it would not notice the vendored group helper calling torch's private
constructor wrongly, a rendezvous that never completes, or a rank layout that
leaves a broadcast one participant short -- each of which hangs every rank
rather than failing. Here the trainer and the receivers are separate processes
joined by ``init_independent_process_group``: over gloo on any machine, and
over RCCL, driven by LumenRL's own sender, wherever there are GPUs for it.

The receivers stand for the TP ranks of one engine, so they also decide on each
stream together, through ``vote_on_stream`` over a CPU group of their own.
"""

import multiprocessing
import queue
import runpy
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import stubbed_aiter

_SNAPSHOT = Path(__file__).parent / "fixtures" / "lumenrl_rdma_sender.txt"
# Every rank finishes in seconds once started; past this a stream has hung.
_DEADLINE_S = 180

# (name, rank whose first bucket is rejected, rank whose begin is refused).
# Streamed one after another over the same groups, so each later one also
# shows the groups are still in step after a rank failed and drained.
_CLEAN = ("clean", None, None)
_ONE_RANK_FAILS = ("one rank fails", 2, None)
_ONE_RANK_REFUSES = ("one rank refuses", None, 2)


def _checkpoint() -> dict[str, torch.Tensor]:
    return {
        "a": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "b": torch.full((8,), 7.0),
    }


def _expected() -> dict[str, list]:
    return {name: tensor.tolist() for name, tensor in _checkpoint().items()}


def _receiving_runner(device: torch.device, rejects_buckets: bool, refuses: bool):
    """The real transaction, over a model holding exactly the checkpoint."""
    from torch import nn

    from atom.rollout.weight_updater import WeightUpdaterMixin

    class _Runner(WeightUpdaterMixin):
        label = "smoke"
        rank = 0
        world_size = 1

        def __init__(self):
            self.device = device
            self.model = nn.Module()
            for name, tensor in _checkpoint().items():
                zeros = torch.zeros_like(tensor, device=device)
                self.model.register_parameter(
                    name, nn.Parameter(zeros, requires_grad=False)
                )
            if refuses:
                # As if this rank alone had already begun a later version.
                self._last_started_weight_version = 1 << 30

        def clear_kv_cache(self):
            pass

        def _sync_target_model(self):
            return self.model

        def apply_weight_bucket(self, named_tensors, payload_bytes=0):
            if rejects_buckets:
                raise RuntimeError("this rank rejects its first bucket")
            return super().apply_weight_bucket(named_tensors, payload_bytes)

        def weights(self) -> dict[str, list]:
            return {n: p.cpu().tolist() for n, p in self.model.named_parameters()}

    return _Runner()


def _send(group, device: torch.device, backend: str, version: int) -> None:
    import json

    import torch.distributed as dist

    if backend == "nccl":
        # LumenRL's sender itself, as the contract tests pin it.
        sender = runpy.run_path(str(_SNAPSHOT), run_name="lumenrl_rdma_sender")
        sender["send_weight_stream"](
            group, list(_checkpoint().items()), bucket_size_bytes=64, version=version
        )
        return

    # The same framing on CPU, where the sender's own buckets cannot be built:
    # it stages every tensor on a GPU first.
    def header(command, metadata_bytes, payload_bytes):
        frame = [command, metadata_bytes, payload_bytes, version]
        dist.broadcast(torch.tensor(frame, dtype=torch.int64), src=0, group=group)

    for name, tensor in _checkpoint().items():
        payload = tensor.contiguous().view(torch.uint8).reshape(-1)
        entry = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "offset": 0,
            "nbytes": payload.numel(),
        }
        raw = json.dumps([entry], separators=(",", ":")).encode("utf-8")
        metadata = torch.tensor(list(raw), dtype=torch.uint8)
        header(1, metadata.numel(), payload.numel())
        dist.broadcast(metadata, src=0, group=group)
        dist.broadcast(payload, src=0, group=group)
    header(0, 0, 0)


def _rank_main(rank, world_size, ports, backend, scenarios, results):
    """One process: rank 0 is the trainer, the rest are one engine's ranks."""
    import torch.distributed as dist

    with stubbed_aiter():
        from atom.rollout.rdma_weight_receiver import (
            receive_weight_stream,
            vote_on_stream,
        )
    from atom.utils.independent_process_group import init_independent_process_group

    if backend == "nccl":
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
        # A CPU rank has nothing to synchronize, and a CPU-only torch raises.
        torch.cuda.synchronize = lambda *args, **kwargs: None
    groups = []
    try:
        stream_port, vote_port = ports
        group = init_independent_process_group(
            backend=backend,
            init_method=f"tcp://127.0.0.1:{stream_port}",
            timeout=timedelta(seconds=60),
            world_size=world_size,
            rank=rank,
            group_name="rdma-smoke",
        )
        groups.append(group)
        if rank > 0:
            tp = init_independent_process_group(
                backend="gloo",
                init_method=f"tcp://127.0.0.1:{vote_port}",
                timeout=timedelta(seconds=60),
                world_size=world_size - 1,
                rank=rank - 1,
                group_name="rdma-smoke-tp",
            )
            groups.append(tp)
    except Exception as exc:  # noqa: BLE001 - the parent reports it
        for scenario, *_ in scenarios:
            detail = f"{type(exc).__name__}: {exc}"
            results.put((scenario, rank, "raised", detail, None))
        return
    try:
        for version, (scenario, failing, refusing) in enumerate(scenarios, start=1):
            if rank == 0:
                try:
                    _send(group, device, backend, version)
                    results.put((scenario, rank, "sent", None, None))
                except Exception as exc:  # noqa: BLE001 - the parent reports it
                    detail = f"{type(exc).__name__}: {exc}"
                    results.put((scenario, rank, "raised", detail, None))
                continue
            runner = _receiving_runner(device, rank == failing, rank == refusing)
            try:
                receive_weight_stream(
                    group,
                    runner,
                    device=device,
                    expected_version=version,
                    agree=lambda ready, began: vote_on_stream(tp, ready, began),
                )
                state, detail = "committed", runner.weights()
            except Exception as exc:  # noqa: BLE001 - the parent reports it
                state, detail = "raised", f"{type(exc).__name__}: {exc}"
            serving = runner.get_weight_update_status()["healthy"]
            results.put((scenario, rank, state, detail, serving))
    finally:
        for g in groups:
            dist.destroy_process_group(g)


def _free_ports(count: int) -> list[int]:
    probes = [socket.socket() for _ in range(count)]
    try:
        for probe in probes:
            probe.bind(("127.0.0.1", 0))
        return [probe.getsockname()[1] for probe in probes]
    finally:
        for probe in probes:
            probe.close()


def _streams(world_size: int, backend: str, scenarios) -> dict:
    """Run *scenarios* as streams across *world_size* processes, in order;
    what each rank reported for each."""
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    args = (world_size, _free_ports(2), backend, scenarios, results)
    procs = [
        ctx.Process(target=_rank_main, args=(rank, *args), daemon=True)
        for rank in range(world_size)
    ]
    for proc in procs:
        proc.start()
    reports = {scenario: {} for scenario, *_ in scenarios}
    owed = world_size * len(scenarios)
    deadline = time.monotonic() + _DEADLINE_S
    try:
        while owed and time.monotonic() < deadline:
            try:
                scenario, rank, state, detail, serving = results.get(timeout=1.0)
            except queue.Empty:
                if not any(proc.is_alive() for proc in procs):
                    break  # every process gone; any report still owed never comes
                continue
            reports[scenario][rank] = (state, detail, serving)
            owed -= 1
    finally:
        for proc in procs:
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    for scenario, by_rank in reports.items():
        silent = sorted(set(range(world_size)) - set(by_rank))
        assert not silent, (
            f"{scenario}: rank(s) {silent} never finished, so the stream hung: "
            f"{reports}"
        )
    return reports


@pytest.fixture(scope="module")
def gloo_reports():
    scenarios = (_CLEAN, _ONE_RANK_FAILS, _ONE_RANK_REFUSES)
    return _streams(world_size=3, backend="gloo", scenarios=scenarios)


def test_a_stream_lands_on_every_receiver_over_a_real_group(gloo_reports):
    reports = gloo_reports[_CLEAN[0]]

    assert reports[0][0] == "sent", reports
    for rank in (1, 2):
        assert reports[rank] == ("committed", _expected(), True), reports


def test_a_rank_that_fails_mid_stream_strands_nobody_and_commits_nowhere(
    gloo_reports,
):
    """The drain keeps the trainer and the peer from hanging; the vote keeps the
    peer from serving a version the failing rank does not hold."""
    reports = gloo_reports[_ONE_RANK_FAILS[0]]

    assert reports[0][0] == "sent", reports
    state, detail, serving = reports[1]
    assert (state, serving) == ("raised", False), reports
    assert "another rank taking it could not" in detail, reports
    state, detail, serving = reports[2]
    assert (state, serving) == ("raised", False), reports
    assert "rejects its first bucket" in detail, reports


def test_a_rank_that_refuses_a_stream_leaves_no_rank_serving_it(gloo_reports):
    """Refused on one rank only, the stream was committed by its peer, and the
    refusing rank went on serving its old weights beside it."""
    reports = gloo_reports[_ONE_RANK_REFUSES[0]]

    assert reports[0][0] == "sent", reports
    state, detail, serving = reports[1]
    assert (state, serving) == ("raised", False), reports
    assert "another rank taking it could not" in detail, reports
    state, detail, serving = reports[2]
    assert (state, serving) == ("raised", False), reports
    assert "must increase" in detail, reports


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two GPUs: one for the sender, one for a receiver",
)
def test_lumenrls_sender_streams_into_atom_over_rccl():
    world_size = min(3, torch.cuda.device_count())
    reports = _streams(world_size, "nccl", scenarios=(_CLEAN,))[_CLEAN[0]]

    assert reports[0][0] == "sent", reports
    for rank in range(1, world_size):
        assert reports[rank] == ("committed", _expected(), True), reports

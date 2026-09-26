"""Model-free remote GPU check; all GPU contexts live in disposable peers."""

import json
import time


def receive(connection):
    if not connection.poll(60):
        raise TimeoutError("Counter smoke peer did not respond")
    return connection.recv()


def peer(rank, connection):
    import timeline_event
    import torch

    torch.cuda.set_device(0)
    backend = timeline_event.install()
    assert timeline_event.install() is backend
    stream = torch.cuda.Stream()
    tensor = torch.zeros(4096, device="cuda:0", dtype=torch.int32)
    torch.cuda.synchronize()
    connection.send(tensor)
    remote = receive(connection)
    assert remote.sum().item() == 0
    connection.send("mapped")
    assert receive(connection) == "mapped"
    expected = {}
    pending = 0
    checks = []
    for step in range(1, 33):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            tensor.fill_(step * 10 + rank)
            event = backend.create_event(0)
            backend.record_event(event, stream)
        handle = backend.export_event(event, 0)
        expected[event.channel.identity] = event.generation
        connection.send(handle)
        if step % 2:
            del event
        incoming = backend.import_event(receive(connection), 0)
        pending += not backend.query_event(incoming)
        with torch.cuda.stream(stream):
            backend.wait_event(incoming, stream)
            result = remote.clone()
            reply = backend.create_event(0)
            backend.record_event(reply, stream)
        expected[incoming.channel.identity] = incoming.generation
        expected[reply.channel.identity] = reply.generation
        connection.send(backend.export_event(reply, 0))
        acknowledgement = backend.import_event(receive(connection), 0)
        backend.wait_event(acknowledgement, stream)
        expected[acknowledgement.channel.identity] = acknowledgement.generation
        stream.synchronize()
        assert bool(torch.all(result == step * 10 + 1 - rank).item())
        assert backend.query_event(incoming)
        assert backend.query_event(acknowledgement)
        checks.append(step)
        del incoming, reply, acknowledgement, result
    assert pending > 0, "Smoke did not observe an unfinished counter"
    assert len(expected) == 2 and set(expected.values()) == {64}
    torch.cuda.synchronize()
    connection.send(("drained", expected))
    assert receive(connection) == ("drained", expected)
    timeline_event._registry.close_after_test_drain(expected)
    connection.send("closed")
    assert receive(connection) == "closed"
    del remote
    connection.send("released")
    assert receive(connection) == "released"
    connection.close()
    print(
        "K3_TIMELINE_SMOKE "
        + json.dumps(
            {
                "rank": rank,
                "steps": len(checks),
                "pending": pending,
                "channels": 2,
                "final_generation": 64,
                "closed": True,
            }
        ),
        flush=True,
    )


def main():
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    left, right = context.Pipe()
    children = [
        context.Process(target=peer, args=(0, left)),
        context.Process(target=peer, args=(1, right)),
    ]
    try:
        for child in children:
            child.start()
        left.close()
        right.close()
        deadline = time.monotonic() + 180
        while any(child.is_alive() for child in children):
            if any(child.exitcode not in (None, 0) for child in children):
                raise RuntimeError("Counter smoke peer failed")
            if time.monotonic() >= deadline:
                raise TimeoutError("Counter smoke exceeded 180 seconds")
            time.sleep(0.1)
        if any(child.exitcode != 0 for child in children):
            raise RuntimeError("Counter smoke peer failed")
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
        for child in children:
            if child.pid is not None:
                child.join(5)
                if child.is_alive():
                    child.kill()
                    child.join(5)
    print("K3_TIMELINE_SMOKE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

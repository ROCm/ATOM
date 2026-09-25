"""Remote startup check of the original and GIL-releasing IPC import paths."""

import json


def produce(connection):
    import torch

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    for value in range(1, 9):
        tensor = torch.zeros(4096, device="cuda:0", dtype=torch.int32)
        connection.send(tensor)
        if not connection.poll(120) or connection.recv() != "mapped":
            raise RuntimeError("IPC consumer did not acknowledge the tensor mapping")
        with torch.cuda.stream(stream):
            torch.cuda._sleep(100_000_000)
            tensor.fill_(value)
            event = torch.cuda.Event(interprocess=True)
            event.record(stream)
        connection.send((event.ipc_handle(), value))
        if not connection.poll(120) or connection.recv() != "done":
            raise RuntimeError("IPC consumer did not acknowledge completion")
        del tensor, event
    connection.close()


def check(backend):
    import torch
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    connection, other = context.Pipe()
    producer = context.Process(target=produce, args=(other,))
    producer.start()
    other.close()
    results = []
    try:
        torch.cuda.set_device(0)
        stream = torch.cuda.Stream()
        for iteration in range(8):
            if not connection.poll(120):
                raise RuntimeError("IPC producer timed out")
            tensor = connection.recv()
            if tensor.sum().item() != 0:
                raise RuntimeError("Unexpected initial IPC tensor contents")
            connection.send("mapped")
            if not connection.poll(120):
                raise RuntimeError("IPC event producer timed out")
            handle, value = connection.recv()
            label = "original" if iteration % 2 == 0 else "gil_released"
            event = (
                torch.cuda.Event.from_ipc_handle(0, handle)
                if label == "original"
                else backend.import_event(handle, 0)
            )
            if type(event) is not torch.cuda.Event:
                raise RuntimeError("IPC import changed the event type")
            pre_wait_ready = backend.query_event(event)
            with torch.cuda.stream(stream):
                backend.wait_event(event, stream)
                total = tensor.sum().item()
            if total != value * 4096 or not backend.query_event(event):
                raise RuntimeError("IPC import did not preserve stream ordering")
            backend.synchronize_event(event, 0)
            results.append(
                {
                    "path": label,
                    "value": value,
                    "sum": total,
                    "pre_wait_ready": pre_wait_ready,
                }
            )
            del event, tensor
            connection.send("done")
        producer.join(30)
        if producer.exitcode != 0:
            raise RuntimeError(f"IPC producer failed: {producer.exitcode}")
    finally:
        connection.close()
        if producer.is_alive():
            producer.terminate()
            producer.join(5)
        if producer.is_alive():
            producer.kill()
            producer.join(5)
    print("K3_IMPORT_GIL_SMOKE " + json.dumps(results), flush=True)

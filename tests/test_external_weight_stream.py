"""A fast external reader must not retain a whole checkpoint behind H2D copies."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest
import torch

from atom.model_loader.loading_core import load_weights_into_model


@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("limit_pending_futures", [False, True])
def test_external_weight_stream_backpressure_preserves_parameters(
    monkeypatch, num_threads, limit_pending_futures
):
    monkeypatch.setenv("ATOM_LOADER_NUM_THREADS", str(num_threads))
    count = 64
    model = torch.nn.Module()
    model.weights = torch.nn.ParameterList(
        [torch.nn.Parameter(torch.zeros(1), requires_grad=False) for _ in range(count)]
    )
    copy_started, reader_exhausted, release_copies = Event(), Event(), Event()

    def iterator(_path, _disable_mmap, wants=None):
        for index in range(count):
            name = f"weights.{index}"
            assert wants is None or wants(name)
            yield name, torch.tensor([float(index)])
        reader_exhausted.set()

    def copy_weight(param, tensor):
        copy_started.set()
        assert release_copies.wait(timeout=20)
        param.data.copy_(tensor)

    def load():
        return load_weights_into_model(
            model=model,
            model_name_or_path="<external-stream>",
            hf_config=SimpleNamespace(num_hidden_layers=1),
            default_weight_loader=copy_weight,
            fuse_shared_expert=lambda *_: False,
            is_rank0=lambda: False,
            weights_iterator=iterator,
            limit_pending_futures=limit_pending_futures,
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(load)
        try:
            assert copy_started.wait(timeout=10)
            if num_threads == 1 or limit_pending_futures:
                # Slow copies apply backpressure to the reader before it can
                # materialize the entire checkpoint, without pinning the test
                # to the implementation's exact queue size.
                assert not reader_exhausted.wait(timeout=0.1)
            else:
                # Existing callers still use the original unbounded reader.
                assert reader_exhausted.wait(timeout=10)
        finally:
            release_copies.set()
        loaded = result.result(timeout=20)

    assert loaded == {f"weights.{index}" for index in range(count)}
    torch.testing.assert_close(
        torch.cat(list(model.weights)), torch.arange(count, dtype=torch.float32)
    )

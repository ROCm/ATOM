# SPDX-License-Identifier: MIT
import importlib.util

import pytest

from tests.models.deepseek_v41.conftest import single_rank, small_config  # noqa: F401


@pytest.fixture(autouse=True)
def cpu_plane_references(monkeypatch, request):
    """CPU metadata tests use test references; CUDA inputs keep the real kernels."""
    if importlib.util.find_spec("triton") is None:
        return
    from atom.model_ops.attentions.deepseek_v41 import cache, planes
    from tests.attentions.deepseek_v41 import plane_reference

    def dispatch(real, ref, cuda):
        def run(*args, **kwargs):
            return (real if cuda(args) else ref)(*args, **kwargs)

        return run

    for name, cuda in (
        ("prefill_window", lambda a: a[0][0].is_cuda),
        ("reset_windows", lambda a: a[2].is_cuda),
        ("gather_windows", lambda a: a[4].is_cuda),
        ("write_window_planes", lambda a: a[0].is_cuda),
    ):
        real = getattr(planes, name)
        wrapped = dispatch(real, getattr(plane_reference, name), cuda)
        monkeypatch.setattr(planes, name, wrapped)
        monkeypatch.setattr(cache, name, wrapped)
        if getattr(request.module, name, None) is real:
            monkeypatch.setattr(request.module, name, wrapped)

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ModelRunner.exit runs once (`still_running` is cleared first), so a MegaMoE
close error must not cut the rest of the teardown short: it is raised only
after the dist env, the CUDA graphs, the KV cache and the model are released."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.model_engine import model_runner as mr
from atom.model_ops.fused_moe import mori_v2_prepare_finalize as mv2


def test_a_failed_megamoe_close_still_releases_everything(monkeypatch):
    calls = []

    def close_mega_transports():
        calls.append("close_mega_transports")
        raise RuntimeError("close failed")

    monkeypatch.setenv("ATOM_MORI_V2", "1")
    monkeypatch.setattr(mv2, "close_mega_transports", close_mega_transports)
    monkeypatch.setattr(mr, "get_kvconnector", lambda: None)
    monkeypatch.setattr(mr, "destroy_dist_env", lambda: calls.append("destroy"))
    monkeypatch.setattr(mr.torch.cuda, "empty_cache", lambda: calls.append("empty"))
    runner = SimpleNamespace(
        still_running=True,
        vision_embeddings=SimpleNamespace(clear=lambda: None),
        req_graph=object(),
        kv_cache=object(),
        model=object(),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        mr.ModelRunner.exit(runner)
    assert calls == ["close_mega_transports", "destroy", "empty"]
    assert runner.req_graph is None
    assert not hasattr(runner, "kv_cache") and not hasattr(runner, "model")

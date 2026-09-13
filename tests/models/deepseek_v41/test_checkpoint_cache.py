# SPDX-License-Identifier: MIT
"""Interrupted TP evaluation must resume the same response subset on all ranks."""

from types import SimpleNamespace

import pytest

from .checkpoint_cache import seed_worker_caches


def test_interrupted_evaluation_seeds_identical_tp_response_sets(tmp_path):
    harness = pytest.importorskip("lm_eval.api.model")
    sqlite = pytest.importorskip("sqlitedict")

    class InterruptibleLM(harness.LM):
        def __init__(self, fail_after=None):
            super().__init__()
            self.fail_after = fail_after
            self.calls = []

        def loglikelihood(self, requests):
            results = []
            for request in requests:
                if len(self.calls) == self.fail_after:
                    raise RuntimeError("interrupted evaluation")
                self.calls.append(request.args)
                result = (-float(request.args[0]), False)
                self.cache_hook.add_partial("loglikelihood", request.args, result)
                results.append(result)
            return results

        def loglikelihood_rolling(self, requests):
            raise NotImplementedError

        def generate_until(self, requests):
            raise NotImplementedError

    identity = {"source": "fixed", "tp_size": 3}
    seed_worker_caches(tmp_path, identity, 3)
    requests = [SimpleNamespace(args=(str(i), "answer")) for i in range(4)]
    first = InterruptibleLM(fail_after=2)
    wrapper = harness.CachingLM(first, str(tmp_path / "tp0_rank0.db"))
    try:
        with pytest.raises(RuntimeError, match="interrupted"):
            wrapper.loglikelihood(requests)
    finally:
        wrapper.dbdict.close()
    # A worker may commit a different subset just before process termination.
    with sqlite.SqliteDict(tmp_path / "tp1_rank0.db", autocommit=True) as stale:
        stale[harness.hash_args("loglikelihood", requests[2].args)] = (-2.0, False)
        stale["unrelated"] = (123, True)
    seed_worker_caches(tmp_path, identity, 3)
    for rank in range(3):
        model = InterruptibleLM()
        wrapper = harness.CachingLM(model, str(tmp_path / f"tp{rank}_rank0.db"))
        try:
            assert wrapper.loglikelihood(requests) == [
                (-float(i), False) for i in range(4)
            ]
            assert model.calls == [request.args for request in requests[2:]]
            assert "unrelated" not in wrapper.dbdict
        finally:
            wrapper.dbdict.close()


def test_cache_identity_mismatch_does_not_replace_worker_data(tmp_path):
    seed_worker_caches(tmp_path, {"source": "old"}, 2)
    worker = tmp_path / "tp1_rank0.db"
    worker.write_bytes(b"unchanged")
    with pytest.raises(ValueError, match="different evaluation"):
        seed_worker_caches(tmp_path, {"source": "new"}, 2)
    assert worker.read_bytes() == b"unchanged"


def test_existing_responses_without_identity_are_rejected(tmp_path):
    (tmp_path / "tp0_rank0.db").touch()
    with pytest.raises(ValueError, match="no evaluation identity"):
        seed_worker_caches(tmp_path, {"source": "unknown"}, 2)

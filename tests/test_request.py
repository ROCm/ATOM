# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/request.py

from atom.model_engine.request import RequestOutput


class TestRequestOutput:
    def test_basic_creation(self):
        ro = RequestOutput(request_id=1, output_tokens=[10, 20], finished=False)
        assert ro.request_id == 1
        assert ro.output_tokens == [10, 20]
        assert ro.finished is False
        assert ro.finish_reason is None

    def test_finished_with_reason(self):
        ro = RequestOutput(
            request_id=2, output_tokens=[5], finished=True, finish_reason="eos"
        )
        assert ro.finished is True
        assert ro.finish_reason == "eos"

    def test_empty_output_tokens(self):
        ro = RequestOutput(request_id=0, output_tokens=[], finished=False)
        assert ro.output_tokens == []

    def test_max_tokens_reason(self):
        ro = RequestOutput(
            request_id=3,
            output_tokens=[1, 2, 3],
            finished=True,
            finish_reason="max_tokens",
        )
        assert ro.finish_reason == "max_tokens"

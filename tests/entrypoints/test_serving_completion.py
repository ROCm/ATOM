# SPDX-License-Identifier: MIT

from atom.entrypoints.openai.serving_completion import (
    _strip_stop_strings,
    build_completion_response,
    build_completion_response_multi,
)


def _final_output(text: str) -> dict:
    return {
        "text": text,
        "finish_reason": "stop",
        "num_tokens_input": 3,
        "num_tokens_output": 5,
        "ttft": 0.0,
        "tpot": 0.0,
        "latency": 0.0,
    }


def test_strip_stop_strings_removes_earliest_match():
    text = "answer #### 18\nQuestion: next\n</s>"

    assert _strip_stop_strings(text, ["</s>", "Question:"]) == "answer #### 18\n"


def test_build_completion_response_strips_stop_string():
    resp = build_completion_response(
        "cmpl-test",
        "test-model",
        _final_output("18\nQuestion: leaked prompt"),
        ["Question:"],
    )

    assert resp.choices[0]["text"] == "18\n"


def test_build_completion_response_multi_strips_stop_string_per_choice():
    resp = build_completion_response_multi(
        "cmpl-test",
        "test-model",
        [
            _final_output("first\nQuestion: leaked"),
            _final_output("second</s> leaked"),
        ],
        ["Question:", "</s>"],
    )

    assert resp.choices[0]["text"] == "first\n"
    assert resp.choices[1]["text"] == "second"

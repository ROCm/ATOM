import json

from atom.entrypoints.openai.sse import data_frame, event_frame

CHAT_DELTA = {
    "id": "chatcmpl-abc",
    "object": "chat.completion.chunk",
    "created": 1786000000,
    "model": "DeepSeek-V4-Flash",
    "choices": [
        {
            "index": 0,
            "delta": {"content": " throughput"},
            "finish_reason": None,
            "logprobs": None,
        }
    ],
}


def _payload_of(frame: str, prefix: str) -> dict:
    assert frame.startswith(prefix)
    assert frame.endswith("\n\n")
    return json.loads(frame[len(prefix) : -2])


def test_data_frame_matches_the_json_dumps_it_replaced():
    frame = data_frame(CHAT_DELTA)

    assert frame == f"data: {json.dumps(CHAT_DELTA, separators=(',', ':'))}\n\n"
    assert _payload_of(frame, "data: ") == CHAT_DELTA


def test_event_frame_carries_the_event_name():
    frame = event_frame("message_start", {"type": "message_start"})

    assert frame.startswith("event: message_start\ndata: ")
    assert _payload_of(frame, "event: message_start\ndata: ") == {
        "type": "message_start"
    }


def test_frames_round_trip_unicode_and_control_characters():
    payload = {"text": '你好 "🎉"\n\tdone', "n": None, "ok": True, "ratio": 0.5}

    assert _payload_of(data_frame(payload), "data: ") == payload
    # A raw newline inside the payload would split the frame and corrupt the
    # stream, so the encoder has to escape it.
    assert "\n" not in data_frame(payload)[:-2]


def test_encoder_is_shared_across_calls():
    """A per-call encoder would put the allocation back on the hot path."""
    first = data_frame(CHAT_DELTA)
    second = data_frame(CHAT_DELTA)

    assert first == second

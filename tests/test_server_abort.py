# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""A request the server ended itself (a weight cache daemon's `stop` or
`load`): the client is answered an error, not a completion cut short."""

import asyncio
import json

import pytest

from atom.entrypoints.openai.protocol import (
    STREAM_DONE_MESSAGE,
    ServerAborted,
    raise_for_server_abort,
    server_aborted_frames,
)
from atom.entrypoints.openai.serving_completion import stream_completion_response
from atom.entrypoints.openai.streaming_dispatch import StreamOutputCollector
from atom.model_engine.request import SERVER_ABORT_REASON, server_abort

WHY = "the server is loading other model weights, send the request again"


def test_a_finish_reason_says_whether_the_server_ended_the_request():
    assert server_abort(SERVER_ABORT_REASON + WHY) == WHY
    # A client that went away, and the ordinary ends, are not that.
    for other in (None, "", "aborted", "stop", "max_tokens"):
        assert server_abort(other) is None
        raise_for_server_abort(other)
    with pytest.raises(ServerAborted) as raised:
        raise_for_server_abort(SERVER_ABORT_REASON + WHY)
    assert raised.value.status_code == 503 and WHY in str(raised.value)


def test_a_stream_the_server_ended_gets_an_error_and_not_a_stop():
    cleaned = []

    async def frames():
        collector = StreamOutputCollector("r")
        stream = stream_completion_response(
            "id",
            "m",
            collector,
            7,
            3,
            lambda seq_id, aborted=False: cleaned.append((seq_id, aborted)),
            lambda request_id: None,
        )
        collector.put_nowait({"token_ids": [1], "text": "a", "finished": False})
        first = await stream.__anext__()
        collector.put_nowait(
            {
                "token_ids": [],
                "text": "",
                "finished": True,
                "finish_reason": SERVER_ABORT_REASON + WHY,
            }
        )
        return first, [frame async for frame in stream]

    first, rest = asyncio.run(frames())
    assert json.loads(first[len("data: ") :])["choices"][0]["text"] == "a"
    # No finish chunk: what the client got is not passed off as complete.
    (end,) = rest
    assert end.endswith(STREAM_DONE_MESSAGE) and "finish_reason" not in end
    error = json.loads(end.split("\n\n")[0][len("data: ") :])["error"]
    assert error["code"] == "server_abort" and WHY in error["message"]
    assert end == server_aborted_frames(ServerAborted(WHY))
    # The engine ended it: the stream has nothing to abort there.
    assert cleaned == [(7, False)]

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Brain retries/fallbacks, the Anthropic Messages client, session
persistence/resume, history compaction, and image provenance helpers."""

import json
import sys
from pathlib import Path

import pytest

from atom.agent import image
from atom.agent.brain import (
    Brain,
    BrainError,
    Reply,
    ToolCall,
    messages_completion,
    to_anthropic,
)
from atom.agent.config import AgentConfig
from atom.agent.session import INTERRUPTED, ChatSession, Hooks, SessionLog, repair
from atom.agent.tools import ToolBox, ToolContext

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import fake_openai_server as fake  # noqa: E402

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


@pytest.fixture
def server():
    fake.FAULTS.clear()
    fake.REQUESTS.clear()
    port, stop = fake.start()
    yield f"http://127.0.0.1:{port}"
    stop()
    fake.FAULTS.clear()


@pytest.fixture
def cfg(tmp_path):
    return AgentConfig(
        state_dir=tmp_path / "state",
        docs_root=None,
        runtime="local",
        lease_dir=tmp_path / "leases",
    )


def _brain(url, **kw):
    return Brain(url + "/v1", "fake-model", backoff_s=(0.01,), **kw)


@pytest.mark.parametrize("fault", ["503", "truncate", "stall"])
def test_retryable_failures_are_retried(server, fault):
    fake.FAULTS[:] = [fault]
    notes = []
    brain = _brain(server, timeout=1, on_retry=notes.append)
    reply = brain.chat([{"role": "user", "content": "hi"}])
    assert reply.content == "Hello from fake."
    assert len(notes) == 1 and "retry 1/3" in notes[0]


def test_retries_are_bounded(server):
    fake.FAULTS[:] = ["503"] * 5
    with pytest.raises(BrainError) as e:
        _brain(server, max_retries=2).chat([{"role": "user", "content": "hi"}])
    assert e.value.retryable and "503" in str(e.value)
    assert len(fake.REQUESTS) == 3


def test_client_errors_are_not_retried(server):
    with pytest.raises(BrainError) as e:
        Brain(server + "/nope", "m", backoff_s=(0.01,)).chat(
            [{"role": "user", "content": "x"}]
        )
    assert not e.value.retryable
    assert len(fake.REQUESTS) == 0  # 404 from the router, no POST recorded


def test_session_falls_back_to_the_next_brain(server, cfg):
    dead = Brain("http://127.0.0.1:9/v1", "gone", max_retries=0)
    dead.fallbacks = [_brain(server)]
    notices = []
    session = ChatSession(
        dead, ToolBox(ToolContext(cfg)), "sys", Hooks(on_notice=notices.append)
    )
    assert session.send("hi") == "Hello from fake."
    assert session.brain.model == "fake-model"
    assert "continuing on" in notices[0]
    # The model is told on its next turn.
    session.send("again")
    assert fake.REQUESTS[-1]["body"]["messages"][-1]["content"].startswith(
        "[atom-chat moved"
    )


def test_to_anthropic_alternates_and_pairs_tool_results():
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "dropped",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"x": 1}'},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "g", "arguments": ""},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "ra"},
        {"role": "tool", "tool_call_id": "b", "content": "rb"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "next"},
    ]
    system, msgs, tools = to_anthropic(history, [WEATHER])
    assert system == "sys"
    assert [m["role"] for m in msgs] == [
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    uses = msgs[1]["content"]
    assert [u["type"] for u in uses] == ["tool_use", "tool_use"]
    assert uses[0]["input"] == {"x": 1} and uses[1]["input"] == {}
    assert [r["tool_use_id"] for r in msgs[2]["content"]] == ["a", "b"]
    assert tools[0]["input_schema"]["properties"]["city"]["type"] == "string"


def test_messages_completion_text_and_tool_use(server):
    r = messages_completion(server, "fake-model", [{"role": "user", "content": "hi"}])
    assert r.content == "Hello from messages." and r.finish_reason == "stop"
    assert r.usage == {"prompt_tokens": 9, "completion_tokens": 3}
    r = messages_completion(
        server,
        "fake-model",
        [{"role": "user", "content": "weather in Austin?"}],
        tools=[WEATHER],
        api_key="k",
    )
    assert r.finish_reason == "tool_calls"
    assert [(c.id, c.name, c.parsed_arguments()) for c in r.tool_calls] == [
        ("toolu_1", "get_weather", {"city": "Austin"})
    ]
    headers = {k.lower(): v for k, v in fake.REQUESTS[-1]["headers"].items()}
    assert headers["x-api-key"] == "k" and headers["anthropic-version"]
    assert fake.REQUESTS[-1]["body"]["max_tokens"] > 0


def test_session_runs_tools_over_anthropic_api(server, cfg):
    """A tool round trip through ChatSession on the Messages dialect."""
    brain = Brain(server, "fake-model", api="anthropic", backoff_s=(0.01,))
    box = ToolBox(ToolContext(cfg))
    session = ChatSession(brain, box, "sys")
    # get_weather is not an agent tool: the error goes back as a tool_result.
    assert session.send("weather in Austin?") == "Hello from messages."
    sent = fake.REQUESTS[-1]["body"]["messages"]
    assert sent[-1]["content"][0]["type"] == "tool_result"
    assert sent[-1]["content"][0]["tool_use_id"] == "toolu_1"


class Scripted:
    fallbacks: list = []

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None, on_content=None, on_reasoning=None):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    def describe(self):
        return "scripted"


def test_session_log_round_trip_and_resume(cfg):
    log = SessionLog(cfg.state_dir)
    s = ChatSession(
        Scripted(
            [
                Reply(
                    tool_calls=[
                        ToolCall("c1", "explain_flags", '{"flags": ["--level"]}')
                    ]
                ),
                Reply(content="answer"),
            ]
        ),
        ToolBox(ToolContext(cfg)),
        "sys",
        log=log,
    )
    s.send("q1")
    meta, history = SessionLog.load(cfg.state_dir, log.id)
    assert [m["role"] for m in history] == ["user", "assistant", "tool", "assistant"]
    assert SessionLog.resolve(cfg.state_dir, "last") == log.id
    assert SessionLog.list(cfg.state_dir)[0]["first_message"] == "q1"
    resumed = ChatSession(
        Scripted([Reply(content="ok")]),
        ToolBox(ToolContext(cfg)),
        "new sys",
        history=history,
    )
    assert resumed.messages[0]["content"] == "new sys"
    resumed.send("q2")
    assert resumed.messages[-2]["content"].startswith("[atom-chat resumed")


def test_failed_first_step_is_not_persisted(cfg):
    log = SessionLog(cfg.state_dir)
    s = ChatSession(
        Scripted([BrainError("down")]), ToolBox(ToolContext(cfg)), "sys", log=log
    )
    assert s.send("q").startswith("[brain error]")
    assert SessionLog.load(cfg.state_dir, log.id)[1] == []


def test_repair_answers_cut_off_tool_calls():
    calls = [
        {"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}},
        {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{}"}},
    ]
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "a", "content": "ok"},
        # crash before b's result
    ]
    fixed = repair(history)
    assert fixed[-1] == {"role": "tool", "tool_call_id": "b", "content": INTERRUPTED}
    json.loads(INTERRUPTED)  # valid JSON for the model


def test_compaction_drops_whole_turns(cfg, monkeypatch):
    from atom.agent import session as session_mod

    monkeypatch.setattr(session_mod, "HISTORY_CHAR_BUDGET", 3000)
    s = ChatSession(Scripted([Reply(content="fine")]), ToolBox(ToolContext(cfg)), "sys")
    call = {"id": "t", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    for i in range(6):
        s.messages += [
            {"role": "user", "content": f"u{i}" + "x" * 300},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [dict(call, id=f"t{i}")],
            },
            {"role": "tool", "tool_call_id": f"t{i}", "content": "r" * 900},
            {"role": "assistant", "content": "a" * 200},
        ]
    s.send("latest")
    roles = [m["role"] for m in s.messages]
    # Every tool result still follows its call.
    for i, m in enumerate(s.messages):
        if m["role"] == "tool":
            assert s.messages[i - 1].get("tool_calls")
    assert "elided" in json.dumps(s.messages)
    assert roles.count("user") < 7 and s.messages[-2]["content"] == "latest"


def test_image_helpers(tmp_path, monkeypatch):
    assert image.unknown_env(
        {"ATOM_A": "1", "ATOM_TYPO": "1", "OMP_NUM_THREADS": "4"}, {"ATOM_A"}
    ) == ["ATOM_TYPO"]
    assert image.unknown_env({"ATOM_X": "1"}, set()) == []  # unknown -> no claim
    monkeypatch.setattr(image, "_run", lambda *a, **k: None)  # no docker
    assert image.probe("rocm/atom:latest", tmp_path) is None
    info = image.ImageInfo("img", "sha256:x", tmp_path)
    src = tmp_path / "atom" / "atom" / "model_engine"
    src.mkdir(parents=True)
    (src / "arg_utils.py").write_text('p.add_argument("--level", type=int)\n')
    assert "--level" in info.flag_table()

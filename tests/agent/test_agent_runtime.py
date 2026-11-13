# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deploy lifecycle, brain client, chat loop and MCP server against a fake
OpenAI server -- the full path minus the GPU."""

import io
import json
import os
import sys
from pathlib import Path

import pytest

from atom.agent import deploy, hardware
from atom.agent.brain import Brain, Reply, ToolCall, chat_completion
from atom.agent.config import AgentConfig
from atom.agent.mcp_server import handle, serve
from atom.agent.policy import Policy
from atom.agent.session import ChatSession
from atom.agent.tools import ToolBox, ToolContext

TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(TESTS))
import fake_openai_server  # noqa: E402


@pytest.fixture
def fake_server():
    port, stop = fake_openai_server.start()
    yield f"http://127.0.0.1:{port}/v1"
    stop()


@pytest.fixture
def cfg(tmp_path):
    return AgentConfig(
        state_dir=tmp_path / "state",
        docs_root=None,
        runtime="local",
        lease_dir=tmp_path / "leases",
    )


def test_chat_completion_assembles_split_tool_calls(fake_server):
    r = chat_completion(
        fake_server,
        "fake-model",
        [{"role": "user", "content": "weather in Austin?"}],
        tools=[
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        ],
    )
    assert [(c.name, c.parsed_arguments()) for c in r.tool_calls] == [
        ("get_weather", {"city": "Austin"})
    ]
    assert r.reasoning == "need the tool"
    assert r.usage["completion_tokens"] == 4
    plain = chat_completion(
        fake_server, "fake-model", [{"role": "user", "content": "hi"}]
    )
    assert plain.content == "Hello from fake." and plain.ttft_s is not None


def test_store_roundtrip(tmp_path):
    store = deploy.DeploymentStore(tmp_path)
    assert store.new_id() == "d1"
    dep = deploy.Deployment("d1", "m", "local", [0], 8000, {}, [], "log", 0.0)
    store.save(dep)
    assert store.new_id() == "d2"
    assert store.get("8000").id == "d1"
    dep.stopped = True
    store.save(dep)
    assert store.active() == [] and len(store.all()) == 1


def test_docker_command_and_mounts():
    dep = deploy.Deployment(
        "d3", "m", "docker", [4, 5], 8001, {"AITER_LOG_LEVEL": "WARNING"},
        ["--model", "/hub/models--org--m/snapshots/abc", "-tp", "2"], "log", 0.0,
        image="rocm/atom:latest", container="atom-agent-d3",
    )  # fmt: skip
    cmd = deploy.build_docker_command(dep)
    assert "--device=/dev/kfd" in cmd and "--ipc=host" in cmd
    assert "HIP_VISIBLE_DEVICES=4,5" in cmd
    assert "/hub/models--org--m:/hub/models--org--m:ro" in cmd
    assert cmd[cmd.index("rocm/atom:latest") + 1 :][:3] == ["python3", "-u", "-m"]
    assert "atom-agent=d3" in cmd


def test_local_lifecycle_with_fake_server(cfg, monkeypatch):
    """launch -> wait_ready -> smoke_test -> stop, with the fake server
    standing in for atom.entrypoints.openai_server."""
    monkeypatch.setattr(deploy, "SERVER_MODULE", "fake_openai_server")
    monkeypatch.setenv(
        "PYTHONPATH", f"{TESTS}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"
    )
    monkeypatch.setattr(hardware, "list_gpus", lambda with_names=True: [])
    from atom.agent import planner

    port = planner.pick_port(18500, set())
    ctx = ToolContext(cfg)
    dep = deploy.launch(
        cfg,
        ctx.store,
        model="fake",
        gpus=[0],
        port=port,
        args=["--server-port", str(port)],
    )
    try:
        st = deploy.wait_ready(dep, timeout_s=30, poll_s=0.3)
        assert st["state"] == "ready", st
        assert st["phase"] == "serving"
        box = ToolBox(ctx)
        smoke = box.call("smoke_test", {"deployment": dep.id})
        assert smoke["ok"] and smoke["tool_calling"]["works"], smoke
        bench = box.call(
            "benchmark",
            {"deployment": dep.id, "input_len": 8, "output_len": 4, "concurrency": 2},
        )
        assert bench["requests_ok"] == 8 and bench["requests_failed"] == 0
        exported = box.call("export_deployment", {"deployment": dep.id})
        assert all(Path(f).is_file() for f in exported["files"])
    finally:
        out = deploy.stop(ctx.store, dep, grace_s=5)
    assert out["stopped"]
    assert deploy.status(ctx.store.get(dep.id))["state"] == "stopped"


def test_stop_needs_approval(cfg):
    ctx = ToolContext(cfg, approve=lambda *a: False)
    dep = deploy.Deployment(
        "d1", "m", "local", [0], 8000, {}, [], "log", 0.0, pid=999999999
    )
    ctx.store.save(dep)
    assert ToolBox(ctx).call("stop_server", {"deployment": "d1"}) == {
        "declined_by_user": True
    }
    assert not ctx.store.get("d1").stopped


def test_run_command_read_only_vs_approval(cfg):
    asked = []
    ctx = ToolContext(cfg, approve=lambda n, a, s: asked.append(s) or False)
    box = ToolBox(ctx)
    assert box.call("run_command", {"command": "ls /"})["exit_code"] == 0
    assert asked == []
    assert box.call("run_command", {"command": "touch /tmp/x"})["declined_by_user"]
    assert box.call("run_command", {"command": "ls / ; rm -rf /tmp/x"})[
        "declined_by_user"
    ]
    assert box.call("run_command", {"command": "docker rm -f x"})["declined_by_user"]
    assert len(asked) == 3


class ScriptedBrain:
    """Plays back replies and records what the session sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, tools=None, on_content=None, on_reasoning=None):
        self.seen.append(json.loads(json.dumps(messages)))
        reply = self.replies.pop(0)
        if on_content and reply.content:
            on_content(reply.content)
        return reply


def test_session_runs_tools_and_keeps_reasoning(cfg):
    brain = ScriptedBrain(
        [
            Reply(
                reasoning="check docs",
                tool_calls=[ToolCall("c1", "explain_flags", '{"flags": ["--level"]}')],
            ),
            Reply(content="--level picks the compile mode."),
        ]
    )
    session = ChatSession(brain, ToolBox(ToolContext(cfg)), "sys")
    assert session.send("what is --level?") == "--level picks the compile mode."
    second_call = brain.seen[1]
    assistant = second_call[-2]
    assert assistant["reasoning_content"] == "check docs"
    assert assistant["tool_calls"][0]["function"]["name"] == "explain_flags"
    tool_msg = json.loads(second_call[-1]["content"])
    assert "--level" in tool_msg


def test_session_reports_bad_tool_arguments(cfg):
    brain = ScriptedBrain(
        [
            Reply(tool_calls=[ToolCall("c1", "read_doc", "{not json")]),
            Reply(content="sorry"),
        ]
    )
    ChatSession(brain, ToolBox(ToolContext(cfg)), "sys").send("x")
    assert "not JSON" in brain.seen[1][-1]["content"]


def test_switch_brain_takes_effect_next_step(cfg, fake_server, monkeypatch):
    ctx = ToolContext(cfg, policy=Policy("auto-safe"))  # switching brain is "safe"
    dep = deploy.Deployment("d1", "fake", "local", [], 0, {}, [], "log", 0.0)
    monkeypatch.setattr(
        deploy,
        "status",
        lambda d: {"state": "ready", "served_model": "fake-model"},
    )
    monkeypatch.setattr(
        deploy.Deployment, "base_url", property(lambda s: fake_server[:-3])
    )
    ctx.store.save(dep)
    old = ScriptedBrain(
        [Reply(tool_calls=[ToolCall("c1", "switch_brain", '{"deployment": "d1"}')])]
    )
    session = ChatSession(old, ToolBox(ctx), "sys")
    assert session.send("use the new model") == "Hello from fake."
    assert isinstance(session.brain, Brain) and session.brain.model == "fake-model"


def test_mcp_protocol(cfg):
    box = ToolBox(ToolContext(cfg), include_brain_switch=False)
    init = handle(
        box, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    assert init["result"]["capabilities"]["tools"] is not None
    assert (
        handle(box, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    )
    tools = handle(box, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"][
        "tools"
    ]
    by_name = {t["name"]: t for t in tools}
    assert "switch_brain" not in by_name
    assert by_name["launch_server"]["annotations"]["readOnlyHint"] is False
    assert by_name["inspect_model"]["annotations"]["readOnlyHint"] is True
    call = handle(
        box,
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "explain_flags", "arguments": {"flags": ["--level"]}},
        },
    )
    assert call["result"]["isError"] is False
    assert "--level" in call["result"]["content"][0]["text"]
    bad = handle(box, {"jsonrpc": "2.0", "id": 4, "method": "nope"})
    assert bad["error"]["code"] == -32601

    out = io.StringIO()
    serve(cfg, io.StringIO('{"jsonrpc":"2.0","id":9,"method":"ping"}\nnot json\n'), out)
    lines = [json.loads(x) for x in out.getvalue().splitlines()]
    assert lines[0] == {"jsonrpc": "2.0", "id": 9, "result": {}}
    assert lines[1]["error"]["code"] == -32700


def test_slash_brain_switch_is_announced_to_the_model(cfg):
    first = ScriptedBrain([Reply(content="a")])
    session = ChatSession(first, ToolBox(ToolContext(cfg)), "sys")
    session.send("hi")
    second = ScriptedBrain([Reply(content="b"), Reply(content="c")])
    second.describe = lambda: "m2 @ http://x/v1"
    session.switch_brain(second)
    session.send("which model?")
    assert second.seen[0][-1]["content"].startswith("[atom-chat switched")
    session.send("again")
    assert second.seen[1][-1]["content"] == "again"  # only announced once


def test_policy_modes():
    assert Policy("read-only").decide("launch").action == "deny"
    assert Policy("ask").decide("stop_own").action == "ask"
    safe = Policy("auto-safe", ("rocm-smi *",))
    assert safe.decide("launch").action == "allow"
    assert safe.decide("launch_forced").action == "ask"
    assert safe.decide("command", "rm -rf /x").action == "ask"
    assert safe.decide("command", "rocm-smi --showpids").action == "allow"
    assert Policy("auto").decide("launch_forced").action == "allow"
    with pytest.raises(ValueError):
        Policy("yolo")


def test_read_only_mode_refuses_and_audits(cfg):
    asked = []
    ctx = ToolContext(
        cfg,
        approve=lambda *a: asked.append(a) or True,
        policy=Policy("read-only"),
        session="s-test",
    )
    dep = deploy.Deployment("d1", "m", "local", [0], 8000, {}, [], "log", 0.0, pid=1)
    ctx.store.save(dep)
    box = ToolBox(ctx)
    res = box.call("stop_server", {"deployment": "d1"})
    assert "read-only" in res["declined_by_policy"]
    assert box.call("run_command", {"command": "touch /tmp/x"})["declined_by_policy"]
    assert asked == []  # the user is never asked in read-only mode
    events = ctx.audit.read(session="s-test")
    approvals = [e for e in events if e["event"] == "approval"]
    tools = [e for e in events if e["event"] == "tool"]
    assert [a["allowed"] for a in approvals] == [False, False]
    assert {a["by"] for a in approvals} == {"policy"}
    assert [t["tool"] for t in tools] == ["stop_server", "run_command"]
    assert tools[0]["declined"]

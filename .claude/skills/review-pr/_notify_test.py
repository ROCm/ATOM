#!/usr/bin/env python3
"""Drive _notify.py against a fake transport.

The property under test is one thing: `notified=true` reaches GITHUB_OUTPUT only when GitHub
actually created a comment. The report job on hosted infra is silenced by that flag, so a
notifier that sets it without posting turns a visible failure into a silent one -- which is
what the first live run on ATOM did (PR #3088, run 43225144156).

These drive `main()`, not the POST helper. The decision that matters -- whether to claim
"notified" -- lives in main(), so a test that calls the helper and then re-implements that
`if` proves the helper right while leaving the decision uncovered: restore the bug in main()
and such a test stays green. Only the transport is faked; everything above it is shipping
code. selftest.sh covers the two no-network paths by running _notify.py as a subprocess,
and runs this file through its pytest block for the ones that do post.

Usage: python3 -m pytest .claude/skills/review-pr/_notify_test.py
"""

import io
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SRC = HERE / "_notify.py"
REPO = "ROCm/ATOM"
PR = "42"
URL = f"https://api.github.com/repos/{REPO}/issues/{PR}/comments"


def load():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    mod = types.ModuleType("_notify_under_test")
    mod.__file__ = str(SRC)
    src = compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec")
    exec(src, mod.__dict__)  # noqa: S102 - deliberate: exec the shipping file
    return mod


# Once, not per test: isolation comes from monkeypatch-managed mutations (env, argv, urlopen),
# not from re-reading disk; re-exec per test would buy nothing and repeat ~4KB of I/O each call.
MOD = load()


def drive(scn, tmp_path, monkeypatch, capsys):
    """Run main() end to end over a fake transport -> (rc, stdout, notified)."""

    def fake_urlopen(req, timeout=None):
        if "raises" in scn:
            raise scn["raises"]
        payload = scn["payload"]
        resp = io.BytesIO(b"" if payload is None else json.dumps(payload).encode())
        resp.status = scn["status"]
        resp.url = scn.get("landed", URL)
        return resp

    # Substitute only the socket call; Request, headers, URL and the id check stay real.
    monkeypatch.setattr(MOD.urllib.request, "urlopen", fake_urlopen)

    work = tmp_path / "ws"
    work.mkdir(exist_ok=True)
    (work / ".atom-review-status").write_text(
        scn.get("status_line", "glm\tGLM backend unavailable"), encoding="utf-8"
    )
    out_file = tmp_path / "gh_output"
    for k, v in {
        "GITHUB_WORKSPACE": str(work),
        "GITHUB_OUTPUT": str(out_file),
        "ATOM_BOT_TOKEN": "tok",
        "GITHUB_REPOSITORY": REPO,
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(MOD.sys, "argv", ["_notify.py", PR])

    rc = MOD.main()
    notified = out_file.exists() and "notified=true" in out_file.read_text()
    return rc, capsys.readouterr().out, notified


def test_a_created_comment_is_reported_and_claims_notified(
    tmp_path, monkeypatch, capsys
):
    rc, out, notified = drive(
        {"status": 201, "payload": {"id": 999}}, tmp_path, monkeypatch, capsys
    )
    assert rc == 0
    assert notified is True
    assert "reported [glm]" in out
    assert "999" in out, "the comment id is the evidence; log it so a reader can check"


@pytest.mark.parametrize(
    "scn, frags",
    [
        pytest.param(
            {
                "status": 200,
                "payload": [{"id": 1}, {"id": 2}],
                # urllib downgrades POST -> GET across a 301, so a moved repo answers 200
                # with the existing comments and creates nothing. This shape broke production.
                "landed": "https://api.github.com/repos/ROCm/atom/issues/42/comments",
            },
            ["no comment was created", "redirected to"],
            id="redirected-post-returning-the-comment-list",
        ),
        pytest.param(
            {"status": 200, "payload": {}},
            ["no comment was created"],
            id="2xx-without-an-id",
        ),
        pytest.param(
            {"raises": OSError("boom")},
            ["boom"],
            id="raising-transport",
        ),
    ],
)
def test_a_post_that_creates_nothing_claims_nothing(
    scn, frags, tmp_path, monkeypatch, capsys
):
    rc, out, notified = drive(scn, tmp_path, monkeypatch, capsys)
    assert rc == 0, "a failed notice must not fail CI"
    assert (
        notified is False
    ), "a POST that created nothing must not silence the report job"
    assert URL in out, "every failure names the URL so a log reader can act"
    for frag in frags:
        assert frag in out

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
code. selftest.sh covers the two no-network paths by running this file as a subprocess.

Usage: python3 -m pytest .claude/skills/review-pr/_notify_test.py
"""

import io
import json
import sys
import types
import urllib.request
from pathlib import Path

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
    # Executing the shipping file rather than importing a copy is the point: these cannot
    # keep passing after _notify.py changes underneath them.
    src = compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec")
    exec(src, mod.__dict__)  # noqa: S102 - deliberate, see above
    return mod


class FakeResp:
    """What urlopen hands back: a context manager that is also a byte stream."""

    def __init__(self, status, payload, url=URL):
        self.status = status
        self.url = url
        body = b"" if payload is None else json.dumps(payload).encode()
        self._body = io.BytesIO(body)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self._body.read(*a)


def drive(scn, tmp_path, monkeypatch):
    """Run main() end to end over a fake transport -> (rc, stdout, notified)."""
    mod = load()

    def fake_urlopen(req, timeout=None):
        if "raises" in scn:
            raise scn["raises"]
        return FakeResp(scn["status"], scn["payload"], scn.get("landed", URL))

    # Substitute only the socket call; Request, headers, URL and the id check stay real.
    mod.urllib = types.SimpleNamespace(
        request=types.SimpleNamespace(
            Request=urllib.request.Request, urlopen=fake_urlopen
        )
    )

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
    monkeypatch.setattr(mod.sys, "argv", ["_notify.py", PR])

    buf = io.StringIO()
    monkeypatch.setattr(mod.sys, "stdout", buf)
    rc = mod.main()
    notified = out_file.exists() and "notified=true" in out_file.read_text()
    return rc, buf.getvalue(), notified


def test_a_created_comment_is_reported_and_claims_notified(tmp_path, monkeypatch):
    rc, out, notified = drive(
        {"status": 201, "payload": {"id": 999}}, tmp_path, monkeypatch
    )
    assert rc == 0
    assert notified is True
    assert "reported [glm]" in out
    assert "999" in out, "the comment id is the evidence; log it so a reader can check"


def test_a_redirected_post_returning_the_comment_list_claims_nothing(
    tmp_path, monkeypatch
):
    # urllib downgrades POST -> GET across a 301, so a moved repo answers 200 with the
    # existing comments and creates nothing. This is the shape that broke production.
    rc, out, notified = drive(
        {
            "status": 200,
            "payload": [{"id": 1}, {"id": 2}],
            "landed": "https://api.github.com/repos/ROCm/atom/issues/42/comments",
        },
        tmp_path,
        monkeypatch,
    )
    assert rc == 0, "a failed notice must not fail CI"
    assert (
        notified is False
    ), "a POST that created nothing must not silence the report job"
    assert "no comment was created" in out
    assert "redirected to" in out, "name the redirect, not just the status"


def test_a_2xx_without_an_id_claims_nothing(tmp_path, monkeypatch):
    rc, out, notified = drive({"status": 200, "payload": {}}, tmp_path, monkeypatch)
    assert rc == 0
    assert notified is False
    assert "no comment was created" in out


def test_a_raising_transport_claims_nothing(tmp_path, monkeypatch):
    rc, out, notified = drive({"raises": OSError("boom")}, tmp_path, monkeypatch)
    assert rc == 0
    assert notified is False
    assert "boom" in out


def test_every_failure_names_the_url_so_a_log_reader_can_act(tmp_path, monkeypatch):
    for scn in ({"status": 200, "payload": {}}, {"raises": OSError("boom")}):
        _, out, _ = drive(scn, tmp_path, monkeypatch)
        assert URL in out

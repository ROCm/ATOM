#!/usr/bin/env python3
"""Drive _notify.py against a fake GitHub.

Same shape as lost_review_test.py: the fake replaces only the transport, and the source is
read from _notify.py rather than copied, so these cannot keep passing after it changes
underneath.

The property under test is one thing: `notified=true` is written only when GitHub actually
created a comment. The report job on hosted infra is silenced by that flag, so a notifier
that sets it without posting turns a visible failure into a silent one -- which is what the
first live run on ATOM did.

Usage: python3 -m pytest .claude/skills/review-pr/_notify_test.py
"""

import contextlib
import io
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "_notify.py"
URL = "https://api.github.com/repos/ROCm/ATOM/issues/42/comments"


def load():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    mod = types.ModuleType("_notify_under_test")
    mod.__file__ = str(SRC)
    exec(compile(SRC.read_text(encoding="utf-8"), str(SRC), "exec"), mod.__dict__)
    return mod


class FakeResp:
    """What urlopen hands back: a context manager that is also a byte stream."""

    def __init__(self, status, payload, url=URL):
        self.status, self.url = status, url
        self._body = io.BytesIO(json.dumps(payload).encode()) if payload is not None else io.BytesIO(b"")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        return self._body.read(*a)


def drive(scn, tmp_path):
    """-> (comment id or None, detail, whether notified=true was written)."""
    mod = load()
    out_file = tmp_path / "gh_output"
    mod.os.environ["GITHUB_OUTPUT"] = str(out_file)

    def fake_open(req):
        if "raises" in scn:
            raise scn["raises"]
        return FakeResp(scn["status"], scn["payload"], scn.get("landed", URL))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cid, detail = mod.post_comment("ROCm/ATOM", "42", "body", "tok", _open=fake_open)
        if cid is not None:
            mod.mark_notified()
    notified = out_file.exists() and "notified=true" in out_file.read_text()
    return cid, detail, notified


def test_a_created_comment_is_reported_and_marks_notified(tmp_path):
    cid, detail, notified = drive({"status": 201, "payload": {"id": 999}}, tmp_path)
    assert cid == 999
    assert "201" in detail
    assert notified is True


def test_a_redirected_post_that_returns_the_comment_list_does_not_mark_notified(tmp_path):
    # urllib downgrades POST -> GET across a 301, so a moved repo answers 200 with the
    # existing comments and creates nothing. This is the shape that broke production.
    cid, detail, notified = drive(
        {"status": 200, "payload": [{"id": 1}, {"id": 2}],
         "landed": "https://api.github.com/repos/ROCm/atom/issues/42/comments"},
        tmp_path,
    )
    assert cid is None
    assert notified is False, "a POST that created nothing must not silence the report job"
    assert "no comment was created" in detail
    assert "redirected to" in detail, "the detail must name the redirect, not just the status"


def test_a_2xx_without_an_id_does_not_mark_notified(tmp_path):
    cid, detail, notified = drive({"status": 200, "payload": {}}, tmp_path)
    assert cid is None
    assert notified is False
    assert "no comment was created" in detail


def test_a_raising_transport_does_not_mark_notified(tmp_path):
    cid, detail, notified = drive({"raises": OSError("boom")}, tmp_path)
    assert cid is None
    assert notified is False
    assert "boom" in detail


def test_the_failure_detail_names_the_url_so_a_log_reader_can_act(tmp_path):
    for scn in ({"status": 200, "payload": {}}, {"raises": OSError("boom")}):
        _, detail, _ = drive(scn, tmp_path)
        assert URL in detail

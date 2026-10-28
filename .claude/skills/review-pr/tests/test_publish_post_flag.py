"""_publish.py must never POST unless invoked with --post.

That flag is the only thing standing between a postable decision and an actual
comment landing on a real PR (see check_prompts.py's check_publish_needs_post_flag
for the companion workflow-side half of this guarantee: no issues/pull-requests
write is ever granted, so this flag is the sole closure). A regression here would
make every dry-run invocation -- local testing, a future workflow edit that drops
the flag by accident -- post for real.
"""

import pathlib
import sys

SKILL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL))
import _publish


def test_dry_run_never_calls_do_post(monkeypatch, capsys):
    monkeypatch.setattr(_publish, "decide", lambda pr: ("POST", "postable"))

    def _boom(pr):
        raise AssertionError("do_post must not be called without --post")

    monkeypatch.setattr(_publish, "do_post", _boom)
    rc = _publish.main(["2705"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "DRY-RUN" in out
    assert "POST — postable" in out


def test_post_flag_calls_do_post_exactly_once(monkeypatch, tmp_path):
    # main() re-reads the card after a successful post to grade severity, so
    # point REPORTS at a self-contained tmp tree: the test must not depend on
    # a reports/ directory left behind by a local run (untracked -> absent in CI).
    (tmp_path / "PR-2705").mkdir()
    (tmp_path / "PR-2705" / "card.md").write_text(
        "Review: \u2705 LGTM\n", encoding="utf-8"
    )
    monkeypatch.setattr(_publish, "REPORTS", tmp_path)
    monkeypatch.setattr(_publish, "decide", lambda pr: ("POST", "postable"))
    calls = []

    def _fake_post(pr):
        calls.append(pr)
        return True, "https://github.com/ROCm/ATOM/pull/2705#issuecomment-1"

    monkeypatch.setattr(_publish, "do_post", _fake_post)
    rc = _publish.main(["2705", "--post"])
    assert rc == 0
    assert calls == ["2705"], "do_post must be called exactly once, for the one PR"


def test_skip_decision_never_calls_do_post_even_with_flag(monkeypatch):
    monkeypatch.setattr(_publish, "decide", lambda pr: ("SKIP", "already posted"))

    def _boom(pr):
        raise AssertionError("do_post must not be called for a SKIP decision")

    monkeypatch.setattr(_publish, "do_post", _boom)
    rc = _publish.main(["2705", "--post"])
    assert rc == 0

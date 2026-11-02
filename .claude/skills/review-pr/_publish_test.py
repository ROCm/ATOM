"""What must not regress in _publish.py.

Two things, and the first is the important one:

  1. The --post closure. _publish.py falls back to ~/.git-credentials on the self-hosted box,
     a human PAT with repo write access, so GITHUB_TOKEN scoping does not stop it from posting
     at all -- the missing --post flag is the only thing standing between a decision and a real
     post. Dry-run must reach no POST. And the review it posts must be event COMMENT: APPROVE
     or REQUEST_CHANGES would turn an advisory bot into a merge gate.
  2. The anchoring rule. A finding goes inline only if its cited line is inside the diff AND
     check_citations.py vouched for every citation it carries. An inline comment on the wrong
     line lands in the author's working view, where being wrong costs the most.

Nothing here touches the network: _api is replaced in every test that could reach it.
"""

import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _publish

DIFF = """diff --git a/atom/model_ops/moe.py b/atom/model_ops/moe.py
--- a/atom/model_ops/moe.py
+++ b/atom/model_ops/moe.py
@@ -20,6 +20,7 @@
     moe_shuffle_scale,
 )
+from aiter.ops.topk import _FLYDSL_USE_HERD
 from torch import nn
-gone = 1
 trailing = 2
diff --git a/docs/notes.md b/docs/notes.md
--- a/docs/notes.md
+++ b/docs/notes.md
@@ -100,2 +100,3 @@
 context
+added
 after
"""

CARD = """## ATOM PR #9999 — a card

**One-sentence summary that the review body does not need.**

Review: ⚠️ NEEDS WORK
Validation (advisory): nothing ran here.

⚠️ atom/model_ops/moe.py:23 imports `_FLYDSL_USE_HERD` — **Author must** pin it.

📝 [verified] `elsewhere.py:900` is nowhere near this diff; it is context code.

🔴 moe.py:22 and helper.py:7 — one of these citations was flagged.

_Independent refutation: 3 judged, 0 killed, 3 defended — cross-family._
"""


@pytest.fixture
def report(tmp_path, monkeypatch):
    d = tmp_path / "PR-9999"
    d.mkdir()
    (d / "card.md").write_text(CARD, encoding="utf-8")
    (d / "pr.diff").write_text(DIFF, encoding="utf-8")
    (d / "citations.txt").write_text(
        "CITATIONS NOTE: helper.py:7 -- none of ['helper'] appears anywhere in this file\n"
        "CITATIONS: every cited file:line resolves (1 proximity note(s))\n",
        encoding="utf-8",
    )
    (d / "pr_meta.json").write_text('{"headRefOid": "deadbeef"}', encoding="utf-8")
    monkeypatch.setattr(_publish, "REPORTS", tmp_path)
    return d


# ------------------------------------------------------------ the --post closure


def test_dry_run_reaches_no_post(report, monkeypatch, capsys):
    calls = []

    def fake_api(path, method="GET", body=None):
        calls.append((method, path))
        return [] if method == "GET" else {"html_url": "http://x"}

    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999"])
    assert not [c for c in calls if c[0] == "POST"], calls
    assert "DRY-RUN" in capsys.readouterr().out


def test_payload_flag_alone_still_posts_nothing(report, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        _publish,
        "_api",
        lambda path, method="GET", body=None: calls.append((method, path)) or [],
    )
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--payload"])
    assert not [c for c in calls if c[0] == "POST"], calls
    # and it printed something a human can read as the literal request body
    assert '"event": "COMMENT"' in capsys.readouterr().out


def test_post_goes_to_the_reviews_endpoint_as_a_comment(report, monkeypatch):
    sent = {}

    def fake_api(path, method="GET", body=None):
        if method == "POST":
            sent["path"], sent["body"] = path, body
            return {"html_url": "http://x"}
        return []

    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    assert sent["path"] == "pulls/9999/reviews"
    assert sent["body"]["event"] == "COMMENT"
    assert sent["body"]["commit_id"] == "deadbeef"
    assert all(c["side"] == "RIGHT" for c in sent["body"]["comments"])


def test_no_source_line_can_make_it_approve_or_request_changes():
    src = (HERE / "_publish.py").read_text(encoding="utf-8")
    for verdict in ("APPROVE", "REQUEST_CHANGES"):
        assert f'"{verdict}"' not in src and f"'{verdict}'" not in src


# ------------------------------------------------------------ anchoring


def test_commentable_counts_right_side_lines_only():
    c = _publish.commentable(DIFF)
    assert c["atom/model_ops/moe.py"] == {20, 21, 22, 23, 24}
    assert c["docs/notes.md"] == {100, 101, 102}


def test_anchors_only_what_the_diff_contains(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert [(c["path"], c["line"]) for c in pl["comments"]] == [
        ("atom/model_ops/moe.py", 23)
    ]
    # the context-code finding stays in the summary, in full
    assert "elsewhere.py:900" in pl["body"]


def test_a_flagged_citation_forfeits_the_inline_slot(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    # moe.py:22 IS in the diff, but the same finding also cites helper.py:7, which
    # check_citations flagged -- so the finding is not anchored at all.
    assert not any(c["line"] == 22 for c in pl["comments"])
    assert "helper.py:7" in pl["body"]


def test_no_citation_check_means_nothing_is_anchored(report, monkeypatch):
    (report / "citations.txt").unlink()
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert pl["comments"] == []
    assert _publish.flagged_citations("9999") is None


def test_summary_indexes_every_anchored_finding(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    # an invisible finding must not read like a finding that was never made
    assert "`atom/model_ops/moe.py:23`" in pl["body"]
    assert "1 finding(s) posted inline" in pl["body"]


def test_summary_carries_the_verdict_and_validation_lines(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    body = _publish.review_payload("9999")["body"]
    assert "Review: ⚠️ NEEDS WORK" in body
    assert "Validation (advisory): nothing ran here." in body
    assert "rocm-atom-bot review pr=9999 head=deadbeef" in body


def test_card_md_is_not_rewritten(report, monkeypatch):
    before = (report / "card.md").read_text(encoding="utf-8")
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.review_payload("9999")
    assert (report / "card.md").read_text(encoding="utf-8") == before


def test_dedup_marker_is_looked_for_in_reviews_too(report, monkeypatch):
    seen = []

    def fake_api(path, method="GET", body=None):
        seen.append(path)
        if path.startswith("pulls/9999/reviews"):
            return [{"body": "x <!-- rocm-atom-bot review pr=9999 head=abc -->"}]
        return []

    monkeypatch.setattr(_publish, "_api", fake_api)
    assert _publish.already_posted("9999", "abc") is True
    assert any(p.startswith("pulls/9999/reviews") for p in seen)

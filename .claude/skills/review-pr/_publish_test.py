"""What must not regress in _publish.py.

Three things, and the first is the important one:

  1. The --post closure. _publish.py falls back to ~/.git-credentials on the self-hosted box,
     a human PAT with repo write access, so GITHUB_TOKEN scoping does not stop it from posting
     at all -- the missing --post flag is the only thing standing between a decision and a real
     post. Dry-run must reach no POST. And the review it posts must be event COMMENT: APPROVE
     or REQUEST_CHANGES would turn an advisory bot into a merge gate.
  2. The anchoring rule. A finding goes inline only if its cited line is inside the diff AND
     check_citations.py vouched for every citation it carries. An inline comment on the wrong
     line lands in the author's working view, where being wrong costs the most. A finding that
     cites a span is anchored to the span; one whose line the diff never reaches is linked
     instead, under the same gate -- a link to the wrong code is worse than no link.
  3. The degrade-to-summary fallback. A review is all-or-nothing -- one anchor GitHub will not
     take fails the whole call, summary included, which is worse than the plain issue comment
     this replaced. A malformed span is a new way to produce exactly that, so on a refusal the
     same review must go out again with the anchors folded into the body, not quietly shed.

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


@pytest.fixture(autouse=True)
def _configured_token(monkeypatch):
    """Default posture: ATOM_BOT_TOKEN is configured, so the fallback identity check
    does not apply. Without this every post test runs as if someone had hand-run it off
    ~/.git-credentials, which is the one path that is supposed to be refused -- and the
    refusal would be read as the post machinery being broken. The fallback itself is
    covered by its own test below."""
    monkeypatch.setattr(_publish, "TOK_IS_FALLBACK", False)


@pytest.fixture
def report(tmp_path, monkeypatch):
    d = tmp_path / "PR-9999"
    d.mkdir()
    (d / "card.md").write_text(CARD, encoding="utf-8")
    (d / "pr.diff").write_text(DIFF, encoding="utf-8")
    (d / "citations.txt").write_text(
        "CITATIONS BAD: 1 citation(s) do not resolve\n"
        "  helper.py:7 -- no such file in the tree\n",
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


def test_a_hard_citation_failure_forfeits_the_inline_slot(report, monkeypatch):
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    # moe.py:22 IS in the diff, but the same finding also cites helper.py:7, which the
    # checker could not resolve at all -- so the finding is not anchored. Only hard
    # failures do this; see the proximity-note test below for the half that does not.
    assert not any(c["line"] == 22 for c in pl["comments"])
    assert "helper.py:7" in pl["body"]


def test_a_proximity_note_does_not_forfeit_the_inline_slot(report, monkeypatch):
    """A note is a quality hint, not evidence the cited line is wrong.

    Counting notes as disqualifying was measured doing real damage: on aiter#7303 the
    reviewer produced five cited findings and published zero inline comments, four of
    them vetoed by notes. The checker's own docstring puts that half at roughly 50%
    precision, and a heuristic that silently switches off the feature it guards is a
    worse failure than the wrong anchor it was guarding against -- a wrong anchor is
    visible and correctable, an absent one looks like a review with nothing to say.
    """
    (report / "citations.txt").write_text(
        "CITATIONS NOTE: helper.py:7 -- none of ['helper'] appears anywhere in this file\n"
        "CITATIONS: every cited file:line resolves (1 proximity note(s))\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert any(
        c["line"] == 22 for c in pl["comments"]
    ), "a proximity note must not veto an anchor"


def test_no_citation_check_means_nothing_is_anchored(report, monkeypatch):
    (report / "citations.txt").unlink()
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert pl["comments"] == []
    assert _publish.flagged_citations("9999") is None


def test_the_degraded_body_does_not_claim_anchors_it_no_longer_has(
    report, monkeypatch, capsys
):
    """The summary-only retry must drop the inline index, not just append to it.

    build_review writes "N finding(s) posted inline, on the lines they are about".
    That is true of the attempt GitHub refused and false of the one being sent, where
    comments[] is empty. Left in place, the degraded review tells the reader that N
    findings sit on lines that carry nothing, and then lists the same N again below --
    the inverse of the silence this fallback exists to prevent. The older test for this
    path asserts only that a retry happened, so it passes on the stale index alone.
    """
    sent = []

    def _api(path, method="GET", body=None):
        if method == "POST" and path.endswith("/reviews"):
            sent.append(body)
            if body.get("comments"):
                return {"message": "line must be part of the diff"}
            return {"html_url": "https://example/r/1"}
        return {"state": "open", "merged_at": None, "head": {"sha": "f" * 40}}

    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    monkeypatch.setattr(_publish, "_api", _api)
    _publish.main(["9999", "--post"])
    assert len(sent) == 2 and sent[1]["comments"] == []
    anchored, degraded = sent[0]["body"], sent[1]["body"]
    assert "posted inline, on the lines they are about" in anchored
    assert (
        "posted inline, on the lines they are about" not in degraded
    ), "the degraded review still claims findings are inline"
    # and the findings themselves are still in it, exactly once
    assert "could not be attached to their lines" in degraded


def test_a_hand_run_under_the_wrong_identity_refuses_to_post(
    report, monkeypatch, capsys
):
    """Posting off ~/.git-credentials as someone other than ATOM_BOT_LOGIN is refused.

    preflight.sh checks identity only when ATOM_BOT_TOKEN(_FILE) is set, so this path
    has no other guard. The damage is not just a wrong author: the review carries a
    dedup marker naming ATOM_BOT_LOGIN's account, so the bot later SKIPs a head that
    account never posted for -- a review that is silently never published.
    """
    monkeypatch.setattr(_publish, "TOK_IS_FALLBACK", True)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    posted = []

    def _api(path, method="GET", body=None):
        if path == "user":
            return {"login": "someone-else"}
        if method == "POST":
            posted.append(body)
        return {"state": "open", "merged_at": None, "head": {"sha": "f" * 40}}

    monkeypatch.setattr(_publish, "_api", _api)
    rc = _publish.main(["9999", "--post"])
    assert posted == [], "a mismatched identity must not post at all"
    assert rc == 1
    out = capsys.readouterr().out
    assert "someone-else" in out and _publish.BOT in out


def test_a_refused_post_exits_non_zero(report, monkeypatch, capsys):
    """A refusal the caller cannot see is a refusal that happens every run.

    The likeliest refusal is a token that may post issue comments but not create
    reviews: `Issues: write` without `Pull requests: write`. That fails both the
    anchored and the summary-only attempt, so the degrade path cannot save it --
    and if main() still returns 0, the step stays green while every card is lost.
    """
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)

    # Only the review POST is refused. Stubbing every call would make the gate skip on
    # PR metadata instead, and a test that never reaches the code it names proves nothing.
    def _refuse_reviews(path, method="GET", body=None):
        if method == "POST" and path.endswith("/reviews"):
            return {"message": "Resource not accessible by personal access token"}
        return {"state": "open", "merged_at": None, "head": {"sha": "f" * 40}}

    monkeypatch.setattr(_publish, "_api", _refuse_reviews)
    rc = _publish.main(["9999", "--post"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "failed to post" in out
    assert "not accessible" in out  # the API's own words reach the operator


def test_a_marker_from_a_former_identity_still_counts(monkeypatch):
    """A review that was posted does not become un-posted because the account changed.

    The bot's account was suspended by GitHub seven days after it was created, so the
    identity had to move. Dedup keys on a marker carrying the login: if only the current
    one counts, the first run after a switch re-reviews every open PR and posts a second
    card under the new name.
    """
    import _identity

    old = _identity.KNOWN[1]  # an identity this bot used to speak as
    assert old != _publish.BOT
    monkeypatch.setattr(
        _publish,
        "_api",
        lambda *a, **k: [{"body": f"<!-- {old} review pr=9999 head=abc -->"}],
    )
    assert _publish.already_posted("9999", "abc") is True


def test_a_crashed_checker_is_not_a_clean_verdict(report, monkeypatch):
    """A traceback in citations.txt must forbid anchoring, not permit it.

    This is how it actually failed: check_citations.py raised NameError on every
    invocation, run_one's `|| true` swallowed it, and the traceback was tee'd here.
    An empty flagged-set then read as "nothing wrong with any citation", so every
    finding anchored with no checking at all -- the precise failure the checker
    exists to prevent, reached by the checker being broken.
    """
    (report / "citations.txt").write_text(
        "Traceback (most recent call last):\n"
        '  File "check_citations.py", line 205, in <module>\n'
        "    sys.exit(main(sys.argv))\n"
        "NameError: name 'tree_for' is not defined\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    assert _publish.flagged_citations("9999") is None
    assert _publish.review_payload("9999")["comments"] == []


def test_a_degraded_card_still_carries_its_verdict_line():
    """The downgrade must not consume the `Review:` prefix.

    `_publish` pulls the verdict with `_line_starting(card, "Review:")`. An earlier
    downgrade rewrote that prefix into `Review (not independently refuted):`, so the
    posted review carried no verdict line at all — severity() still worked (it matches
    "Review" without the colon), so nothing internal noticed; only the reader lost
    something. Found by @valar-cop in review.
    """
    card = "## T\n\nReview: 🔴 BLOCK\n\n🔴 bad\n"
    degraded = (
        card + "\nReview (not independently refuted): the refuter did not finish\n"
    )
    assert _publish._line_starting(degraded, "Review:") == "Review: 🔴 BLOCK"
    assert (
        "not independently refuted" in degraded
    )  # the gate's admission is still there


def test_a_degraded_card_keeps_its_red_severity():
    """The refuter-timeout notice must not become the line severity() reads.

    severity() takes the FIRST line starting with "Review", so a notice prepended above
    the verdict makes a 🔴 BLOCK card report as 'ok' -- the high-severity path skipped on
    exactly the runs that already degraded. Found by @valar-cop in review.
    """
    red = "## T\n\nReview: 🔴 BLOCK\n\n🔴 something bad\n"
    note = "Review (not independently refuted): the refuter did not finish\n"
    assert _publish.severity(red) == "high"
    assert (
        _publish.severity(red + "\n" + note) == "high"
    ), "appended notice ate the verdict"
    assert (
        _publish.severity(note + red) == "ok"
    ), "a prepended notice is what this guards"


def test_the_summary_says_what_the_pr_does(report, monkeypatch):
    """The card's bold one-sentence summary must reach the posted review.

    SKILL.md requires that sentence and names its audience -- "a reviewer who has not
    opened the diff" -- and the posted review is where that reader is. build_review
    collected the title, the verdict lines, the findings and the index, and dropped it,
    so every published review opened on a verdict with no statement of what was judged.
    """
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    body = _publish.review_payload("9999")["body"]
    what = next(l for l in CARD.split("\n") if l.startswith("**"))
    assert what in body, "the posted review does not say what the PR does"


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
    assert f"{_publish.BOT} review pr=9999 head=deadbeef" in body


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


# ------------------------------------------------------------ degrade to summary


def _refusing_api(reject_anchored=True):
    """A fake API that refuses any review carrying comments[], as GitHub does when one
    anchor names a line it will not accept."""
    sent = []

    def fake_api(path, method="GET", body=None):
        if method != "POST":
            return []
        sent.append(body)
        if reject_anchored and body.get("comments"):
            return {
                "message": "pull_request_review_thread.line must be part of the diff"
            }
        return {"html_url": "http://x/review"}

    return fake_api, sent


def test_a_refused_anchor_retries_the_same_review_without_anchors(
    report, monkeypatch, capsys
):
    fake_api, sent = _refusing_api()
    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    assert len(sent) == 2, "the refusal must be retried, not given up on"
    assert sent[0]["comments"] and sent[1]["comments"] == []
    out = capsys.readouterr().out
    assert "ANCHORS REFUSED -- folded into summary" in out
    assert "✅" in out


def test_the_degraded_review_still_carries_the_findings_it_could_not_anchor(
    report, monkeypatch
):
    """The whole point: a review that silently sheds half its findings reads exactly like a
    review that only had the other half."""
    fake_api, sent = _refusing_api()
    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    folded = sent[1]["body"]
    assert (
        "_FLYDSL_USE_HERD" in folded
    ), "the anchored finding vanished from the summary"
    assert "atom/model_ops/moe.py:23" in folded
    # and the findings that were never anchored are still there too
    assert "elsewhere.py:900" in folded


def test_a_review_with_no_anchors_is_not_retried_twice(report, monkeypatch):
    """Nothing to degrade to: retrying an identical call just doubles a failure."""
    (report / "citations.txt").unlink()  # -> nothing anchors
    sent = []

    def fake_api(path, method="GET", body=None):
        if method != "POST":
            return []
        sent.append(body)
        return {"message": "some other failure"}

    monkeypatch.setattr(_publish, "_api", fake_api)
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    _publish.main(["9999", "--post"])
    assert len(sent) == 1


# ------------------------------------------------------------ range anchors
#
# 11 of 36 findings on a batch of real cards cite a span (`layernorm.py:1002-1012`) rather
# than a line, and every one of them used to be anchored to the span's first line -- so the
# reader saw a comment pinned to one line of a block the finding was about, which reads as a
# claim about that line. Counted over this bot's own 221 archived findings the same shape is
# 25 spans. The diff below has a file with TWO hunks precisely so the gap between them is
# exercised: a span crossing it is the malformed range that fails the whole review.

RANGE_DIFF = """diff --git a/atom/model_ops/layernorm.py b/atom/model_ops/layernorm.py
--- a/atom/model_ops/layernorm.py
+++ b/atom/model_ops/layernorm.py
@@ -100,4 +100,8 @@
 def fused_add_rmsnorm(x, residual, weight, eps):
+    out = torch.empty_like(x)
+    _launch(out, x, residual, weight, eps)
+    return out
 
 
 def plain_rmsnorm(x, weight, eps):
@@ -200,2 +200,3 @@
 context_at_200
+added_at_201
 context_at_202
"""

RANGE_CARD = """## ATOM PR #8888 — spans

Review: ⚠️ NEEDS WORK

⚠️ [verified] `atom/model_ops/layernorm.py:101-104` allocates and launches without a `torch.cuda`
🔴 atom/model_ops/layernorm.py:106-201 straddles the gap between the two hunks.
📝 layernorm.py:5220 is real code this PR does not touch; `fused_add_rmsnorm` is nowhere near.
📝 nosuchfile.py:12-20 names a file this diff never mentions.
📝 [verified] layernorm.py:381 — 6 published tuning candidates, written as prose.

_Independent refutation: 4 judged, 0 killed, 4 defended — cross-family._
"""


@pytest.fixture
def range_report(tmp_path, monkeypatch):
    d = tmp_path / "PR-8888"
    d.mkdir()
    (d / "card.md").write_text(RANGE_CARD, encoding="utf-8")
    (d / "pr.diff").write_text(RANGE_DIFF, encoding="utf-8")
    (d / "citations.txt").write_text(
        "CITATIONS: every cited file:line resolves\n", encoding="utf-8"
    )
    (d / "pr_meta.json").write_text('{"headRefOid": "c0ffee1"}', encoding="utf-8")
    monkeypatch.setattr(_publish, "REPORTS", tmp_path)
    # REPO is read from the environment at import, and a box with GITHUB_REPOSITORY set
    # would otherwise make the permalink assertions below pass or fail on the box, not on
    # the code.
    monkeypatch.setattr(_publish, "REPO", "ROCm/ATOM")
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: RANGE_DIFF)
    return d


def test_a_cited_span_anchors_the_span_not_its_first_line(range_report):
    pl = _publish.review_payload("8888")
    c = next(x for x in pl["comments"] if x.get("start_line") == 101)
    assert (c["path"], c["start_line"], c["line"]) == (
        "atom/model_ops/layernorm.py",
        101,
        104,
    )
    assert c["start_side"] == "RIGHT" and c["side"] == "RIGHT"


def test_every_span_anchor_is_shaped_the_way_github_takes_one(range_report):
    """start_line strictly below line, and a side named for each end. Any other shape is
    refused, and a refusal takes the summary and every other comment down with it."""
    for c in _publish.review_payload("8888")["comments"]:
        if "start_line" in c:
            assert c["start_line"] < c["line"], c
            assert c["start_side"] == c["side"] == "RIGHT", c
        else:
            assert "start_side" not in c, c


def test_a_span_straddling_two_hunks_degrades_to_the_line_that_is_in_the_diff(
    range_report,
):
    """106 is in the first hunk, 201 in the second, and the lines between are in neither.
    GitHub takes a span only inside one hunk -- so this anchors 106 alone rather than
    sending a range that would fail the entire review."""
    c = [x for x in _publish.review_payload("8888")["comments"] if x["line"] == 106]
    assert len(c) == 1 and "start_line" not in c[0]


def test_a_partly_anchorable_span_is_not_dropped(range_report):
    """The rule is fall back, not discard: a finding whose citation is wider than the diff
    is still a finding about code in the diff."""
    bodies = [c["body"] for c in _publish.review_payload("8888")["comments"]]
    assert any("straddles the gap" in b for b in bodies)


def test_the_summary_index_names_the_span(range_report):
    body = _publish.review_payload("8888")["body"]
    assert "`atom/model_ops/layernorm.py:101-104`" in body


def test_a_span_in_prose_is_not_read_as_a_range(range_report):
    """A real card wrote `flash_kda.py:381 — 6 published tuning candidates`. Read loosely
    that is the backwards span 381..6, which GitHub refuses -- and the refusal is not of
    that comment but of the whole review."""
    cites = _publish.citations("📝 layernorm.py:381 — 6 published tuning candidates.")
    assert ("layernorm.py", 381, 381) in cites
    assert not [c for c in cites if c[2] != c[1]]


def test_a_span_is_not_also_counted_as_its_own_first_line(range_report):
    assert _publish.citations("⚠️ a/b/c.py:10-20 and d.py:4") == [
        ("a/b/c.py", 10, 20),
        ("d.py", 4, 4),
    ]


# ------------------------------------------------------------ permalinks


def test_a_line_outside_the_diff_gets_a_permalink_pinned_to_the_head_sha(range_report):
    body = _publish.review_payload("8888")["body"]
    assert (
        "https://github.com/ROCm/ATOM/blob/c0ffee1/atom/model_ops/layernorm.py#L5220"
        in body
    )


def test_a_span_outside_the_diff_links_the_whole_span(range_report):
    """`#L a -L b` -- the same reason the anchor is a span: the finding is about the block."""
    monkeypatch_free = _publish.permalink(
        "atom/model_ops/layernorm.py", 300, 320, "c0ffee1"
    )
    assert monkeypatch_free.endswith("#L300-L320")


def test_the_permalink_is_in_addition_to_the_human_readable_citation(range_report):
    body = _publish.review_payload("8888")["body"]
    assert "layernorm.py:5220 is real code this PR does not touch" in body


def test_the_url_sits_alone_on_its_line_so_github_can_expand_it(range_report):
    """GitHub renders a bare blob permalink as the code it names only when nothing else
    shares the line. Inside the sentence it is just a shortened link."""
    body = _publish.review_payload("8888")["body"]
    url = "https://github.com/ROCm/ATOM/blob/c0ffee1/atom/model_ops/layernorm.py#L5220"
    assert [l for l in body.split("\n") if l.strip() == url]


def test_an_unresolvable_path_gets_no_permalink(range_report):
    """nosuchfile.py is in no hunk, so there is nothing to build a blob URL from but the
    citation's own guess at a path. A link to the wrong code is worse than no link."""
    body = _publish.review_payload("8888")["body"]
    assert "nosuchfile.py:12-20" in body  # the finding's text is kept
    assert "blob/c0ffee1/nosuchfile.py" not in body


def test_a_flagged_citation_earns_no_permalink_either(report, monkeypatch):
    """Same gate as anchoring, same reason: the checker said this finding's pointing is
    unreliable, and a link the reader follows is where that costs the most."""
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    body = _publish.review_payload("9999")["body"]
    assert "helper.py:7" in body
    assert "/blob/deadbeef/" not in body


def test_no_citation_check_means_no_permalinks_either(report, monkeypatch):
    (report / "citations.txt").unlink()
    monkeypatch.setattr(_publish, "_api_raw", lambda *a, **k: DIFF)
    pl = _publish.review_payload("9999")
    assert pl["comments"] == []
    assert "/blob/" not in pl["body"]


# ------------------------------------------------------------ a refused RANGE degrades


def test_a_refused_range_review_degrades_and_keeps_the_findings_text(
    range_report, monkeypatch, capsys
):
    """A malformed span is a new way to fail an all-or-nothing call. The fallback must catch
    it like any other refusal, and the span's finding must survive into the summary -- a
    review that sheds it reads exactly like a review that never made it."""
    fake_api, sent = _refusing_api()
    monkeypatch.setattr(_publish, "_api", fake_api)
    _publish.main(["8888", "--post"])
    assert len(sent) == 2
    assert sent[0]["comments"] and sent[1]["comments"] == []
    folded = sent[1]["body"]
    assert "allocates and launches" in folded, "the span's finding vanished"
    assert (
        "`atom/model_ops/layernorm.py:101-104`" in folded
    ), "folded as a line, not a span"
    assert "ANCHORS REFUSED -- folded into summary" in capsys.readouterr().out


def test_a_collapsed_span_leaves_no_dangling_tail_in_the_index(range_report):
    """The straddling finding cites `:106-201` and anchors 106 alone, so the citation the
    index strips is `:106` -- which used to leave the line opening with a bare "-201". Found
    on a real ATOM card (`prefill_delayer.py:563-569`, anchored at 563) the first time this
    ran over one."""
    body = _publish.review_payload("8888")["body"]
    assert "— -201" not in body and "— -" not in body

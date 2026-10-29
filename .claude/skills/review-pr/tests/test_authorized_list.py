"""authorized.txt is the security boundary: it decides who may spend GPU time on a
shared box. The gate's parser lives in the workflow's JS, so nothing here could catch a
malformed line -- selftest.sh has no allowlist check at all. These tests mirror the
gate's exact parsing rules (strip, skip blank, skip '#', lowercase) so a bad edit fails
in CI instead of silently locking everyone out or letting a stray token through.
"""

import pathlib

SKILL = pathlib.Path(__file__).resolve().parent.parent
ALLOWLIST = SKILL / "authorized.txt"


def parse():
    """Exactly what .github/workflows/atom-review-bot.yml does to the file."""
    out = set()
    for line in ALLOWLIST.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if s and not s.startswith("#"):
            out.add(s.lower())
    return out


def test_header_comments_are_not_parsed_as_logins():
    logins = parse()
    assert not any("#" in x for x in logins)
    assert not any(" " in x for x in logins), "a login with a space is a malformed line"


def test_entries_look_like_github_logins():
    # GitHub logins: alphanumerics and single hyphens, 1-39 chars.
    for x in parse():
        assert 1 <= len(x) <= 39, f"{x!r} is not a plausible login length"
        assert all(c.isalnum() or c == "-" for c in x), f"{x!r} has illegal characters"


def test_the_list_is_not_empty_and_not_a_single_login():
    # An empty list fails closed (nobody can trigger); a one-name list reads as a broken
    # bot to every other maintainer. Both are states this file has actually been in.
    assert len(parse()) > 1


def test_no_duplicate_logins_differing_only_by_case():
    raw = [
        l.strip()
        for l in ALLOWLIST.read_text(encoding="utf-8").split("\n")
        if l.strip() and not l.strip().startswith("#")
    ]
    assert len(raw) == len(
        parse()
    ), "two entries collapse to the same login once lowercased"


def test_bot_accounts_are_excluded():
    # A bot in the allowlist lets an automated comment trigger a review loop.
    for x in parse():
        assert "bot" not in x, f"{x!r} looks like a bot account"
        assert x not in {"github-actions", "github-advanced-security"}

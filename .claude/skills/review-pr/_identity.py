"""Which GitHub account this bot speaks as.

This was a constant in _publish.py marked "fixed identity, not configuration", and
nine files copied the literal. That held until the account it named was suspended by
GitHub seven days after it was created, and the only way to keep reviewing was to speak
as a different one. An identity that can be taken away from you by a third party is
configuration whether or not you modelled it that way.

Two properties matter more than the name itself:

Self-triggering. The bot's own cards, queue notices and failure reports arrive as
issue_comment events, and the workflow must ignore them. If the guard names one account
and the token belongs to another, every card the bot posts starts a run that posts a
card. This is not a tail risk -- a card's body contains the trigger phrase, so it is
certain -- and it has happened here before (eighteen reviews, thirty-six runs competing
with themselves). So the guard excludes every account this bot has EVER spoken as, not
just the current one.

Dedup. "Already reviewed this head" is recorded by a marker carrying the login. A review
that was posted does not become un-posted because the account changed, so a marker from
any former identity still counts, or the first run after a switch re-reviews everything.
"""

import os

DEFAULT = "aiter-bot"

# Every login this bot has published under. Append, never replace: each entry is a claim
# about history, and dropping one silently re-enables self-triggering and re-review for
# whatever is still on an open PR.
KNOWN = ("aiter-bot", "rocm-atom-bot", "atom-bot")


def login():
    """The account to speak as now. ATOM_BOT_LOGIN must match ATOM_BOT_TOKEN's owner."""
    return (os.environ.get("ATOM_BOT_LOGIN") or DEFAULT).strip()


def markers(pr, sha):
    """Every marker meaning 'this head was already reviewed', current identity first."""
    out, seen = [], set()
    for who in (login(),) + KNOWN:
        if who not in seen:
            seen.add(who)
            out.append(f"{who} review pr={pr} head={sha}")
    return out

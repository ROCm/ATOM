"""Route a failed review to the owner responsible for that CLASS of failure and post one PR
comment. Single source of truth for the failure-class -> owner map, so run_one.sh (which tags
the failure with `fail <class> ...`) and the workflow (which reports it) cannot drift:
check_prompts.py asserts every class run_one.sh can emit is known here.

Called by the workflow's notify step as `python3 _notify.py <pr>`. Reads `.atom-review-status`
(written by run_one.sh as "<class>\t<message>"). The class -> owner map is the table below --
the single source of truth, as in aiter's copy of this file; check_prompts.py asserts every
class run_one.sh can emit is known here, and that the standalone notifiers agree with it.
"""

import json
import os
import sys
import urllib.request

import _identity

# class -> (login, what is wrong). Hardcoded, not read from a file: aiter keeps the same
# map inline in its own _notify.py, and a second file that only this module reads buys
# nothing over a literal while adding a way for the two to drift. Changing who gets paged
# stays an auditable commit either way. There is deliberately no repo-variable override:
# an invisible settings edit is not an audit trail.
OWNERS = {
    "flow": ("zufayu", "the review pipeline itself (bot logic / prompts)"),
    "env": ("zufayu", "the runner environment (gh, token, git, disk)"),
    "glm": ("yhl-amd", "the model backend serving the review agent"),
    "atom": ("valarLip", "the ATOM serving framework"),
    "machine": ("gyohuangxin", "the machine (GPU, disk, network, OS)"),
}


def owner_for(cls):
    """An unknown class routes to the bot owner rather than to nobody."""
    row = OWNERS.get(cls) or OWNERS.get("flow")
    return row[0] if row else "zufayu"


def mark_notified():
    """Tell the report job this PR has been told -- only once a post has actually landed.

    Every early return in main() is a path where nothing reached the PR: no status file, no
    token, or a POST that threw. The report job on hosted infra is the last thing that could
    speak in those cases, and it stays quiet when this flag is set, so setting it anywhere else
    would trade a visible failure for a silent one.
    """
    out = os.environ.get("GITHUB_OUTPUT")
    if not out:
        return
    with open(out, "a", encoding="utf-8") as fh:
        fh.write("notified=true\n")


def post_comment(repo, pr, body, tok):
    """POST one comment; -> (created comment id, failure detail or None). `None` id means nothing landed.

    Returning the id rather than the status is the point. A POST that GitHub answers 2xx but
    does not act on is indistinguishable from a successful one by status alone, and that is
    not hypothetical: urllib downgrades a POST to a GET when it follows a 301/302, so a
    renamed or mis-cased repo answers 200 with the comment *list* and creates nothing. The
    first live run on ATOM hit exactly this -- "reported" printed, notified=true written, the
    report job silenced by that flag, and no comment on the PR. Reading back the id GitHub
    assigned is the only answer here that cannot be faked by a redirect or an empty 2xx.
    """
    url = f"https://api.github.com/repos/{repo}/issues/{pr}/comments"
    req = urllib.request.Request(
        url,
        data=json.dumps({"body": body}).encode(),
        headers={
            "Authorization": f"token {tok}",
            "Accept": "application/vnd.github+json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = getattr(resp, "status", None)
            landed = getattr(resp, "url", url)
            try:
                data = json.load(resp)
            except Exception:  # noqa: BLE001 - best effort
                data = None
    except Exception as e:  # noqa: BLE001 - must not fail CI
        return None, f"POST {url} raised: {e}"

    cid = data.get("id") if isinstance(data, dict) else None
    if cid is None:
        # Say which of the two shapes this was, because they need different fixes: a
        # redirect means the repo moved, a 2xx without an id means the API changed.
        moved = "" if landed == url else f"; request was redirected to {landed}"
        kind = type(data).__name__
        return None, (
            f"POST {url} -> HTTP {status} but no comment was created "
            f"(response was {kind}{moved})"
        )
    return cid, None


def main():
    if len(sys.argv) < 2:
        print("usage: _notify.py <pr>")
        return 2
    pr = sys.argv[1]
    path = os.path.join(os.environ.get("GITHUB_WORKSPACE", "."), ".atom-review-status")
    if not os.path.exists(path):
        return 0  # the review succeeded -> nothing to report
    with open(path, encoding="utf-8") as fh:
        raw = fh.read()
    cls, _, msg = raw.strip().partition("\t")
    if cls not in OWNERS:
        cls = "flow"  # an unknown tag routes to the bot owner rather than nobody
    owner = owner_for(cls)
    domain = OWNERS[cls][1]
    body = (
        f"⚠️ **{_identity.login()}** — this review could not complete: a problem with **{domain}**, "
        f"not with this PR.\n\n> {msg}\n\n@{owner} — please take a look."
    )
    if cls == "glm":
        body += (
            f"\n\n<sub>Triage: if GLM serves but its output is malformed it may be the ATOM "
            f"framework (@{owner_for('atom')}); if the box, GPU, disk or network is unhealthy "
            f"it may be the machine (@{owner_for('machine')}).</sub>"
        )
    body += "\n\nRe-comment `atom-bot review` once it is resolved."

    tok = os.environ.get("ATOM_BOT_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not tok or not repo:
        print("no ATOM_BOT_TOKEN / GITHUB_REPOSITORY -- would post:\n" + body)
        return 0

    cid, detail = post_comment(repo, pr, body, tok)
    if cid is None:
        # Stay silent about having notified: the report job on hosted infra is the last
        # thing that can speak, and it only speaks while this flag is unset.
        print(f"notify failed: {detail}")
        return 0
    mark_notified()
    print(f"reported [{cls}] -> @{owner} (comment {cid})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

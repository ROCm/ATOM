"""The red/blue distinction check (tests/accept/atomesh_mm/checks/distinct.py)
against fake replies and fake ATOMesh and ATOM logs: its verdict on the
answers, the routing key hashes and where each prefill starts."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "accept" / "atomesh_mm" / "checks"))
import distinct

PROMPT_TOKENS = 2413


def run(tmp_path, monkeypatch, hits, hashes=None):
    """Run the check with each request's prefix hit from `hits`, keyed by
    request name (None: no prefill line), logging one decision event per
    request and its prefill in chunks as the engine does."""
    for name, value in {
        "ACCEPT_MODEL": "k31",
        "ACCEPT_TOPOLOGY": "single",
        "ACCEPT_CHECK": "distinct",
        "RUN_DIR": str(tmp_path),
        "RESULTS_DIR": str(tmp_path / "results"),
        "LIMIT": "1",
    }.items():
        monkeypatch.setenv(name, value)
    hashes = hashes or {"red": "aa", "blue": "bb"}
    mesh_log, atom_log = tmp_path / "mesh.log", tmp_path / "atom.log"
    mesh_log.write_text("")
    atom_log.write_text("[atom] Server started successfully\n")
    requests = iter(enumerate(name for name, _ in distinct.REQUESTS))

    def ask(color):
        req_id, name = next(requests)
        with mesh_log.open("a") as f:
            f.write(f"routing decision key_hash={hashes[color]} policy=cache_aware\n")
        with atom_log.open("a") as f:
            f.write(f"[atom] Request {req_id} arrived, input tokens: {PROMPT_TOKENS}\n")
            done = hits[name]
            while done is not None and done < PROMPT_TOKENS:
                new = min(2048, PROMPT_TOKENS - done)
                f.write(
                    f"[atom] Scheduled prefill batch: 1 reqs, {new} new tokens "
                    f"(done: [{done}], new: [{new}]), req_ids: ({req_id},)\n"
                )
                done += new
        return f"{color.title()}."

    monkeypatch.setattr(distinct, "ask", ask)
    with pytest.raises(SystemExit) as exit_:
        distinct.main()
    return exit_.value.code, json.loads((tmp_path / "result.json").read_text())


def test_passes_when_only_the_red_repeat_reuses_the_image(tmp_path, monkeypatch):
    code, result = run(tmp_path, monkeypatch, {"red": 0, "blue": 0, "red_again": 2048})
    assert code == 0, result["reason"]
    hits = [result[name]["prefix_hit"] for name in ("red", "blue", "red_again")]
    assert hits == [0, 0, 2048]


@pytest.mark.parametrize("blue", [0, 2048])
def test_fails_when_the_red_repeat_reuses_no_more_than_blue(
    tmp_path, monkeypatch, blue
):
    # 0 / 0: the image fills no whole hash block, so nothing shows the seed
    # mattered; 2048 / 2048: the blue request reused the red image's KV.
    code, result = run(
        tmp_path, monkeypatch, {"red": 0, "blue": blue, "red_again": blue}
    )
    assert code == 1
    assert "second red request took" in result["reason"]


def test_fails_without_a_prefill_line(tmp_path, monkeypatch):
    code, result = run(
        tmp_path, monkeypatch, {"red": 0, "blue": None, "red_again": 2048}
    )
    assert code == 1
    assert "no prefill start" in result["reason"]


def test_fails_when_both_colors_share_a_key_hash(tmp_path, monkeypatch):
    code, result = run(
        tmp_path,
        monkeypatch,
        {"red": 0, "blue": 0, "red_again": 2048},
        hashes={"red": "aa", "blue": "aa"},
    )
    assert code == 1
    assert "both colors routed on key hash aa" in result["reason"]

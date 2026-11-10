import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "scripts"
    / "check_kv_block_agreement.py"
)
spec = importlib.util.spec_from_file_location("check_kv_block_agreement", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

# A TP8 rank's line from a CI server log before the counts were reconciled.
LINE = (
    "[atom 03:10:17] Memory budget: total_gpu=287.69GB, free=252.32GB, "
    "utilization=0.9, budget=258.92GB, peak_torch=21.34GB, non_torch=14.10GB, "
    "cudagraph_est=0.08GB, safety=5.75GB, available_for_kv=217.65GB, "
    "block_bytes=99840, num_kvcache_blocks={}"
)


def write_log(tmp_path, counts, noise=("Rank 0: loading weights",)):
    log = tmp_path / "atom_server.log"
    log.write_text("\n".join([*noise, *(LINE.format(n) for n in counts)]) + "\n")
    return str(log)


def test_agreeing_ranks_pass(tmp_path, capsys):
    assert check.main([write_log(tmp_path, [2315143] * 4), "--ranks", "4"]) == 0
    assert "All 4 ranks built 2315143 KV blocks" in capsys.readouterr().out


def test_ranks_that_each_kept_their_own_estimate_fail(tmp_path, capsys):
    log = write_log(tmp_path, [2315143, 2328796])
    assert check.main([log, "--ranks", "2"]) == 1
    assert "[2315143, 2328796]" in capsys.readouterr().out


def test_a_missing_rank_fails(tmp_path, capsys):
    assert check.main([write_log(tmp_path, [2315143]), "--ranks", "2"]) == 1
    assert "expected 2 'Memory budget' lines, found 1" in capsys.readouterr().out

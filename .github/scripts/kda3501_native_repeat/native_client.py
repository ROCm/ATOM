"""One pinned instance, three fixed requests, public native client calls only."""

import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path


def save(output, name, data):
    data = json.loads(
        json.dumps(data), parse_constant=lambda value: {"nonfinite": value}
    )
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    assert len(payload.encode()) <= 1024 * 1024, "client record exceeds 1 MiB"
    with (output / name).open("x", encoding="utf-8") as stream:
        stream.write(payload)


def receive(sample, records, rejected):
    if not sample.output_tokens:
        assert not sample.logprobs
        return
    assert sample.logprobs is not None
    assert len(sample.output_tokens) == len(sample.logprobs)
    for token, row in zip(sample.output_tokens, sample.logprobs, strict=True):
        assert len(row) == 42
        if not all(math.isfinite(float(value)) for value in row):
            # Keep only the first rejected native row, bounded by its fixed width.
            if not rejected:
                rejected.append({"token": int(token), "record": list(row)})
            raise AssertionError("non-finite native logprob row")
        assert len(set(row[2::2])) == 20
        top = list(zip(map(int, row[2::2]), map(float, row[3::2]), strict=True))
        records.append(
            {
                "token": int(token),
                "selected": float(row[0]),
                "rank": int(row[1]),
                "top": top,
            }
        )
        assert len(records) <= 32, "duplicate or excess streamed output"


def verify_candidate_origin(directory, log_path):
    """Check the completed native log and public counters; launches no model."""
    from kda_stage1_probe import parse_step

    directory, log_path = Path(directory), Path(log_path)
    assert log_path.stat().st_size <= 16 * 1024**2, "native log exceeds 16 MiB"
    order, steps, phase = [], {}, None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PR3501 phase "):
            phase = line.split()[2]
            assert line == f"PR3501 phase {phase} start" and phase not in steps
            order.append(phase)
            steps[phase] = []
        if "Scheduled prefill batch: " in line:
            assert phase is not None, "unlabelled native prefill"
            steps[phase].append(
                parse_step(line[line.index("Scheduled prefill batch: ") :])
            )
    assert order == ["C0", "W0", "W1"]
    counters = {}
    for phase in order:
        expected = [(0, 1024)] if phase == "C0" else [(512, 512)]
        assert [(row["done"], row["new"]) for row in steps[phase]] == expected
        data = json.loads((directory / (phase + ".json")).read_text())
        before, after = data["before"]["cache"], data["after"]["cache"]
        assert before["checkpoints_kept"] == (0 if phase == "C0" else 1)
        assert after["checkpoints_kept"] == 1
        assert after["chunks_cut_for_end"] - before["chunks_cut_for_end"] == 0
        assert after["cached_tokens"] - before["cached_tokens"] == (
            0 if phase == "C0" else 512
        )
        assert after["offload_tokens"] - before["offload_tokens"] == 0
        counters[phase] = {"before": before, "after": after}
    assert len({steps[phase][0]["sequence"] for phase in order}) == 3
    save(
        directory,
        "prefill-origin.json",
        {
            "coverage": "single C0 prefill, one kept checkpoint, two 512-token resumes",
            "prefill_steps": steps,
            "cache_counters": counters,
            "qualification": "numerical comparison remains separate",
        },
    )


def verify_checkpoint_observations(directory, log_path):
    """Validate bounded all-rank coverage; unequal states remain reported data."""
    directory, log_path = Path(directory), Path(log_path)
    assert log_path.stat().st_size <= 16 * 1024**2
    origin = json.loads((directory / "prefill-origin.json").read_text())
    prefix, phase, size = "PR3501_CHECKPOINT ", None, 0
    events = {"C0": "store", "W0": "restore_1", "W1": "restore_2"}
    ranks = {rank: [] for rank in range(8)}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PR3501 phase "):
            phase = line.split()[2]
        if prefix not in line:
            continue
        assert line.startswith(prefix) and phase in events
        count = len((line + "\n").encode())
        size += count
        assert count <= 2048 and size <= 65536, "observer log bound exceeded"
        row = json.loads(line[len(prefix) :])
        rank = row["rank"]
        assert type(rank) is int and rank in ranks
        assert row["event"] == events[phase]
        assert row["request_id"] == origin["prefill_steps"][phase][0]["sequence"]
        assert row["image_bytes"] == row["op_bytes"] == 56_171_520
        assert row["layout_id"].startswith("kda-paged-state-v1:")
        assert row["layout_id"].endswith(":tp=8:spec=0:carry=all")
        assert len(row["unit_ids"]) == len(set(row["unit_ids"])) == 29
        assert all(type(unit) is int and unit >= 0 for unit in row["unit_ids"])
        assert type(row["slot"]) is int and row["slot"] >= 0
        for name in (
            ("page_sha256",) if phase == "C0" else ("page_sha256", "restored_sha256")
        ):
            value = row[name]
            assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        ranks[rank].append(row)
        assert len(ranks[rank]) <= 3, "extra observer record"
    warm = []
    for records in ranks.values():
        assert [row["event"] for row in records] == list(
            events.values()
        ), "missing rank/event coverage"
        stored = records[0]
        for row in records[1:]:
            assert row["prefill"] is True and row["cached"] == row["scheduled"] == 512
            for flag in (
                "identity_matches",
                "page_unchanged",
                "restored_matches_incoming",
                "restored_matches_reference",
                "controls_expected",
            ):
                assert type(row[flag]) is bool
            assert row["identity_matches"] and all(
                row[k] == stored[k] for k in ("unit_ids", "layout_id", "op_bytes")
            ), "checkpoint identity changed"
            controls = {
                "read_slot": row["slot"],
                "write_slot": row["slot"],
                "has_initial_state": True,
                "query_start_loc": [0, 512],
                "replayssm_cursor": -1,
                "position_span": [512, 1023, 512],
                "num_prefills": 1,
                "num_actual_tokens": 512,
            }
            assert row["controls_expected"] == (row["controls"] == controls)
            for flag, left, right in (
                ("page_unchanged", row["page_sha256"], stored["page_sha256"]),
                (
                    "restored_matches_incoming",
                    row["restored_sha256"],
                    row["page_sha256"],
                ),
                (
                    "restored_matches_reference",
                    row["restored_sha256"],
                    stored["page_sha256"],
                ),
            ):
                assert not row[flag] or left == right
            warm.append(row)
    save(
        directory,
        "checkpoint-observations.json",
        {
            "coverage": "one store and two restores on each of eight ranks",
            "logged_bytes": size,
            "records": ranks,
            "all_pages_unchanged": all(row["page_unchanged"] for row in warm),
            "all_restored_match_incoming": all(
                row["restored_matches_incoming"] for row in warm
            ),
            "all_restored_match_reference": all(
                row["restored_matches_reference"] for row in warm
            ),
            "all_controls_expected": all(row["controls_expected"] for row in warm),
            "qualification": "observation coverage only; numerics remain separate; CPU reads synchronize; MLA state is unobserved",
        },
    )


def main():
    data = json.loads(Path(__file__).with_name("inputs.json").read_text())
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    assert len(data["tokens"]) == 1024
    raw_tokens = json.dumps(data["tokens"], separators=(",", ":")).encode()
    assert hashlib.sha256(raw_tokens).hexdigest() == data["token_sha256"]
    proof_path = Path(sys.argv[2])
    proof = json.loads(proof_path.read_text())
    assert proof["tokens"] == data["tokens"] and proof["revisions"] == data["revisions"]
    draft = data["engine_args"]["draft_model"]
    assert proof["model_files"][draft]["resolved_path"] == draft
    assert all(
        os.environ.get(key) == value for key, value in data["environment"].items()
    )
    import torch
    import triton

    import atom
    from atom import SamplingParams
    from atom.model_engine.arg_utils import EngineArgs

    source = Path(os.environ["PR3501_SOURCE"]).resolve()
    assert Path(atom.__file__).resolve() == source / "atom/__init__.py"
    runtime = {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "aiter": importlib.metadata.version("amd-aiter"),
    }
    assert runtime == {key: data["runtime"][key] for key in runtime}
    save(
        output,
        "client-start.json",
        {
            "source": data["source"],
            "atom_path": atom.__file__,
            "runtime": runtime,
            "args": data["engine_args"],
            "environment": data["environment"],
            "preflight_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        },
    )
    engine, phase = None, "initializing"
    records, outputs, rejected = [], {}, []
    try:
        start = time.monotonic()
        engine = EngineArgs(**data["engine_args"]).create_engine()
        save(output, "initialized.json", {"elapsed_seconds": time.monotonic() - start})
        assert not engine.config.kv_transfer_config
        assert engine.config.speculative_config is None
        completions = []
        for phase in data["phases"]:
            before = {
                "cache": engine.get_cache_statistics(timeout=10),
                "mtp": engine.get_mtp_statistics(timeout=10),
            }
            records, outputs, rejected = [], {}, []
            params = SamplingParams(**data["sampling"])
            engine.core_mgr.reset_dp_router()
            start_request = time.monotonic()
            print(f"PR3501 phase {phase} start", flush=True)
            with engine.core_mgr.offline_backpressure_batch():
                engine.add_request(
                    [list(data["tokens"])],
                    params,
                    stream_callback=lambda sample, target=records, failures=rejected: receive(
                        sample, target, failures
                    ),
                )
            while not engine.is_finished() and (
                engine.core_mgr.is_alive() or engine.core_mgr.is_rest()
            ):
                assert (
                    time.monotonic() - start_request < 300
                ), "request exceeded 300 seconds"
                outputs.update(engine.io_processor.postprocess(engine.step()))
            after = {
                "cache": engine.get_cache_statistics(timeout=10),
                "mtp": engine.get_mtp_statistics(timeout=10),
            }
            rows = list(outputs.values())
            save(
                output,
                phase + ".json",
                {
                    "phase": phase,
                    "before": before,
                    "after": after,
                    "prompt_sha256": data["token_sha256"],
                    "outputs": rows,
                    "records": records,
                    "rejected_callbacks": rejected,
                },
            )
            for stats in (before["mtp"], after["mtp"]):
                assert stats["enabled"] is False
                assert (
                    stats["total_draft_tokens"] == stats["total_accepted_tokens"] == 0
                )
                assert not stats["distribution"]
            assert not rejected, "non-finite native callback was rejected"
            assert len(rows) == 1
            row = rows[0]
            assert (
                row["num_tokens_input"] == 1024 and row["finish_reason"] == "max_tokens"
            )
            assert [
                row["num_tokens_output"],
                len(records),
                len(row["token_ids"]),
                len(row["logprobs"]),
            ] == [32] * 4
            assert [record["token"] for record in records] == row["token_ids"]
            assert [record["selected"] for record in records] == row["logprobs"]
            assert (
                after["cache"]["offload_tokens"] - before["cache"]["offload_tokens"]
                == 0
            )
            expected_cache = 0 if phase == "C0" else 512
            assert (
                after["cache"]["cached_tokens"] - before["cache"]["cached_tokens"]
                == expected_cache
            )
            completions.append(row["token_ids"])
        save(
            output,
            "collection.json",
            {
                "collected": True,
                "qualification": "not_assessed",
                "C0_W0_same_ids": completions[0] == completions[1],
                "W0_W1_same_ids": completions[1] == completions[2],
            },
        )
    except BaseException as error:
        save(
            output,
            "failure.json",
            {
                "phase": phase,
                "error": repr(error),
                "outputs": list(outputs.values()),
                "records": records,
                "rejected_callbacks": rejected,
            },
        )
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    main()

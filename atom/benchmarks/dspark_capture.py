"""Opt-in offline confidence collection. Records numbers, never prompt/token IDs.

This collector synchronizes sampled confidence/verdicts and is not suitable for
performance measurement. Use fixed-width verification to avoid censored labels.
"""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

from atom.spec_decode.calibration import calibration_identity


def install(runner, directory):
    if runner.config.dspark.confidence_schedule:
        raise ValueError("DSpark calibration collection requires fixed verification")
    if runner.config.hf_config.model_type != "deepseek_v41_text":
        raise ValueError("This calibration collector supports native V4.1 only")
    if runner.rank != 0:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    identity = calibration_identity(runner.config, runner.device)
    (root / "identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    output = (root / "confidence.jsonl").open("a", encoding="utf-8", buffering=1)
    previous = OrderedDict()
    active = None
    steps = 0
    count = 0
    forward = runner.forward
    block_run = runner.drafter.block.run
    commit = runner.attn_metadata_builder.commit_speculative_state

    def observed_forward(batch):
        nonlocal active, steps
        active = None if batch.is_dummy_run else batch
        if active is not None:
            steps += 1
        try:
            return forward(batch)
        finally:
            active = None

    def observed_block(*args, **kwargs):
        result = block_run(*args, **kwargs)
        if active is not None and steps % 8 == 0 and result[1] is not None:
            confidence = (
                result[1][: active.total_seqs_num].detach().float().cpu().tolist()
            )
            for req_id, row in zip(active.req_ids, confidence):
                previous[req_id] = (steps, row[:5])
                previous.move_to_end(req_id)
            while len(previous) > 2 * runner.config.max_num_seqs:
                previous.popitem(last=False)
        return result

    def observed_commit(metadata, last_indices):
        nonlocal count
        step = metadata.step
        if (
            active is not None
            and step.decode
            and metadata.cache.pending is not None
            and any(span.request_id in previous for span in step.requests)
        ):
            accepts = (last_indices - step.cu_seqlens_q[:-1]).detach().cpu().tolist()
            for span, accepted in zip(step.requests, accepts):
                prior = previous.pop(span.request_id, None)
                if prior is None or span.length != 6 or not 0 <= accepted <= 5:
                    continue
                old_step, confidence = prior
                if steps - old_step != 1:
                    continue
                request = hashlib.sha256(str(span.request_id).encode()).hexdigest()[:24]
                output.write(
                    json.dumps(
                        {
                            "request": request,
                            "confidence": confidence,
                            "accepted": int(accepted),
                        }
                    )
                    + "\n"
                )
                count += 1
                if count % 128 == 0:
                    (root / "capture-status.json").write_text(
                        json.dumps({"blocks": count, "steps": steps})
                    )
        return commit(metadata, last_indices)

    runner.forward = observed_forward
    runner.drafter.block.run = observed_block
    runner.attn_metadata_builder.commit_speculative_state = observed_commit

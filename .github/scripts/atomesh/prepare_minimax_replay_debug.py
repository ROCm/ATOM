#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Apply audited CI-local hooks before launching either model service."""

import argparse
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

ROOT_FILES = (
    "atom/model_engine/engine_core.py",
    "atom/model_engine/model_runner.py",
    "atom/kv_transfer/offload/dense/connector.py",
    "atom/kv_transfer/offload/atom_lmcache_staging.py",
)


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise ValueError(
            f"Expected exactly one debug insertion anchor: {old[:150]!r}; found {text.count(old)}"
        )
    return text.replace(old, new, 1)


def build_patches(root):
    originals = {name: (root / name).read_text() for name in ROOT_FILES}
    patched = {}
    for name, source in originals.items():
        patched[name] = replace_one(
            source,
            "\nimport torch\n",
            "\nfrom atom import _minimax_replay_debug as _m3debug\n\nimport torch\n",
        )
    name = ROOT_FILES[0]
    s = patched[name]
    s = replace_one(
        s,
        "    def _process_engine_step_inner(self):\n        result = self.scheduler.schedule()",
        "    def _process_engine_step_inner(self):\n        if _m3debug.hold_for_replay(self):\n            return False\n        result = self.scheduler.schedule()",
    )
    s = replace_one(
        s,
        "        # Dispatch KV connector metadata to workers (triggers async KV load)",
        "        _m3debug.verify_batch(self, scheduled_batch)\n\n        # Dispatch KV connector metadata to workers (triggers async KV load)",
    )
    patched[name] = s
    name = ROOT_FILES[1]
    s = patched[name]
    for function, stage in [
        ("prepare_model", "prepare_model"),
        ("run_model", "target_forward"),
        ("postprocess", "sampling_and_draft"),
        ("propose_draft_token_ids", "draft_propose"),
        ("prepare_sampled_ids", "result_copy"),
    ]:
        anchor = f"    def {function}("
        s = replace_one(s, anchor, f'    @_m3debug.traced("{stage}")\n' + anchor)
    anchor = "    def forward(self, batch: ScheduledBatch) -> ScheduledBatchOutput:\n        # Make this forward's staging buffers safe"
    s = replace_one(s, anchor, '    @_m3debug.traced("model_forward")\n' + anchor)
    anchor = (
        "            sampled_tokens = get_tp_group().broadcast(sampled_tokens, src=0)"
    )
    s = replace_one(
        s,
        anchor,
        '            _m3debug.mark("sampled_broadcast:enter", batch=batch, gpu=True, rank=self.rank)\n'
        + anchor
        + '\n            _m3debug.mark("sampled_broadcast:enqueued", batch=batch, gpu=True, rank=self.rank)',
    )
    anchor = "        if run_compute_draft_kv:\n            drafter.compute_draft_kv("
    s = replace_one(
        s,
        anchor,
        '        if run_compute_draft_kv:\n            _m3debug.mark("draft_kv:enter", batch=batch, gpu=True, rank=self.rank)\n            drafter.compute_draft_kv(',
    )
    anchor = "                batch.next_token_ids,\n            )\n        if pp_non_last or self._is_pure_middle_chunk(batch):"
    s = replace_one(
        s,
        anchor,
        '                batch.next_token_ids,\n            )\n            _m3debug.mark("draft_kv:enqueued", batch=batch, gpu=True, rank=self.rank)\n        if pp_non_last or self._is_pure_middle_chunk(batch):',
    )
    anchor = "            copy_done.record(self.async_copy_stream)\n        cpu_tensor_handle.append((cpu_tensor, copy_done))"
    s = replace_one(
        s,
        anchor,
        '            copy_done.record(self.async_copy_stream)\n            _m3debug.mark("sampled_copy:enqueued", gpu=True, stream=self.async_copy_stream)\n        cpu_tensor_handle.append((cpu_tensor, copy_done))',
    )
    anchor = "        cpu_tensor, event = cpu_tensor_handle.pop(0)\n        event.synchronize()"
    s = replace_one(
        s,
        anchor,
        '        cpu_tensor, event = cpu_tensor_handle.pop(0)\n        _m3debug.mark("previous_sampled_copy_wait:enter")\n        event.synchronize()\n        _m3debug.mark("previous_sampled_copy_wait:return")',
    )
    patched[name] = s
    name = ROOT_FILES[2]
    s = patched[name]
    anchor = "        if synchronize is not None:\n            synchronize()"
    s = replace_one(
        s,
        anchor,
        '        _m3debug.mark("save_producer_wait:enter", request_id=int(req.req_id))\n'
        + anchor
        + '\n        _m3debug.mark("save_producer_wait:return", request_id=int(req.req_id))',
    )
    anchor = "        with source_context:\n            self._engine.store(tok_tensor, **store_kwargs)"
    s = replace_one(
        s,
        anchor,
        '        _m3debug.mark("save_store:enter", request_id=int(req.req_id))\n'
        + anchor
        + '\n        _m3debug.mark("save_store:return", request_id=int(req.req_id))',
    )
    patched[name] = s
    name = ROOT_FILES[3]
    s = patched[name]
    anchor = "                device_buf = ensure_buffer(staging_buffer, group_nbytes(group))"
    s = replace_one(
        s,
        anchor,
        '                _m3debug.mark("stage_a_allocate:enter", gpu=True, stream=stage_a.stream)\n'
        + anchor
        + '\n                _m3debug.mark("stage_a_allocate:return", gpu=True, stream=stage_a.stream)',
    )
    anchor = "                stage_a.run(group, device_buf)"
    s = replace_one(
        s,
        anchor,
        anchor
        + '\n                _m3debug.mark("stage_a:enqueued", gpu=True, stream=stage_a.stream)',
    )
    anchor = "                stage_b.run(group, device_buf)"
    s = replace_one(
        s,
        anchor,
        anchor
        + '\n                _m3debug.mark("stage_b:enqueued", gpu=True, stream=stage_b.stream)',
    )
    anchor = "        stage_b.stream.synchronize()"
    s = replace_one(
        s,
        anchor,
        '        _m3debug.mark("stage_b_sync:enter")\n'
        + anchor
        + '\n        _m3debug.mark("stage_b_sync:return")',
    )
    patched[name] = s
    for name, text in patched.items():
        ast.parse(text, filename=name)
    return originals, patched


def apply(root, output):
    originals, patched = build_patches(root)  # Validate every anchor before any write.
    helper = Path(__file__).with_name("minimax_replay_debug.py").read_text()
    ast.parse(helper)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {}
    diff = []
    for name, text in patched.items():
        manifest[name] = {
            "before_sha256": hashlib.sha256(originals[name].encode()).hexdigest(),
            "after_sha256": hashlib.sha256(text.encode()).hexdigest(),
        }
        diff.extend(
            difflib.unified_diff(
                originals[name].splitlines(True),
                text.splitlines(True),
                fromfile="a/" + name,
                tofile="b/" + name,
            )
        )
    manifest["atom/_minimax_replay_debug.py"] = {
        "sha256": hashlib.sha256(helper.encode()).hexdigest()
    }
    (output / "runtime-patch.diff").write_text("".join(diff))
    (output / "runtime-patch-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output / "minimax_replay_debug.py").write_text(helper)
    for name, text in patched.items():
        (root / name).write_text(text)
    (root / "atom/_minimax_replay_debug.py").write_text(helper)
    print(
        "[replay-debug] Applied CI-local hooks; production branch files are unchanged."
    )


def prepare_runtime(source, runtime, output):
    """Copy the read-only mounted package before applying any instrumentation."""
    shutil.copytree(
        source / "atom",
        runtime / "atom",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        copy_function=shutil.copyfile,
    )
    apply(runtime, output)
    (output / "runtime-source.json").write_text(
        json.dumps(
            {"mounted_source": str(source), "runtime_source": str(runtime)},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-root", required=True, type=Path)
    args = parser.parse_args()
    prepare_runtime(args.root, args.runtime_root, args.output)

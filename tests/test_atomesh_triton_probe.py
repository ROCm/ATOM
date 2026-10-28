"""Probe dispatch must avoid model startup and retain negative GPU results."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / '.github/scripts/atomesh'
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location('triton_probe', SCRIPTS / 'pd_vllm_triton_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class TritonProbeTest(unittest.TestCase):
    def test_dispatch_selects_no_model_entrypoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / 'python3'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            result = subprocess.run(['bash', str(SCRIPTS / 'pd_server_atom.sh')],
                env=dict(os.environ, BACKEND='vllm', ATOMESH_ENV_ATOMESH_VLLM_MODE='triton_probe',
                         PATH=tmp + os.pathsep + os.environ['PATH']),
                capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.strip(), str(SCRIPTS / 'pd_vllm_triton_probe.py'))

    def test_failed_or_missing_gpu_results_fail_suite_but_keep_other_cases(self):
        # An rc=0 child with no results must never count as a successful probe.
        masks = []
        def run(argv, **kwargs):
            out = Path(argv[-1])
            masks.append(kwargs['env']['HIP_VISIBLE_DEVICES'])
            if out.name != 'fresh-decode':
                rows = [{'status': 'success', 'kernel': str(i)} for i in range(4)]
                if out.name == 'default-decode':
                    rows[0]['status'] = 'failed'
                probe.dump(out / 'results.json', rows)
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {'XDG_CACHE_HOME': tmp}), \
             patch.object(probe.subprocess, 'run', side_effect=run):
            self.assertEqual(probe.run_suite(Path(tmp) / 'output'), 1)
            rows = json.loads((Path(tmp) / 'output/summary.json').read_text())
            self.assertEqual([r['status'] for r in rows], ['success', 'failed', 'success', 'failed'])
            self.assertEqual(masks, ['0,1,2,3', '4,5,6,7'] * 2)


if __name__ == '__main__':
    unittest.main()

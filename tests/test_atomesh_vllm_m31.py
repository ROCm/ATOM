"""CPU checks for the native-image experiment's isolation and correctness gates."""
import importlib.util
import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / '.github/scripts/atomesh/pd_vllm_m31.py'
SPEC = importlib.util.spec_from_file_location('m31_native', PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class NativeExperimentTest(unittest.TestCase):
    def test_host_visibility_masks_cannot_overlap_roles(self):
        inherited = {'CUDA_VISIBLE_DEVICES': '7', 'ROCR_VISIBLE_DEVICES': '2',
                     'HIP_VISIBLE_DEVICES': '1'}
        with patch.dict(os.environ, {}, clear=True):
            p = runner.role_env(inherited, 'prefill', '127.0.0.1')
            d = runner.role_env(inherited, 'decode', '127.0.0.1')
        self.assertEqual(set(p['HIP_VISIBLE_DEVICES'].split(',')), set('0123'))
        self.assertEqual(set(d['HIP_VISIBLE_DEVICES'].split(',')), set('4567'))
        for env in (p, d):
            self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
            self.assertNotIn('CUDA_VISIBLE_DEVICES', env)

    def test_reset_waits_for_async_store_block_references(self):
        with patch.object(runner, 'request', side_effect=[
            '{"success": false}', '{"success": true}'
        ]) as request, patch.object(runner, 'check_services'), patch.object(runner.time, 'sleep'):
            self.assertEqual(runner.reset_gpu_cache('http://decode'), {'success': True})
        self.assertEqual(request.call_count, 2)
        self.assertNotIn('reset_external', request.call_args.args[0])

    def test_failed_store_loads_do_not_count_as_cpu_reuse(self):
        metrics = '\n'.join([
            '# HELP vllm:mooncake_store_operation_bytes_total bytes',
            'vllm:mooncake_store_operation_bytes_total{operation="load_get",status="error"} 1000',
            'vllm:mooncake_store_operation_bytes_total{operation="save_put",status="ok"} 900',
            'vllm:mooncake_store_operation_bytes_total{operation="load_get",status="ok"} 128',
            'vllm:mooncake_store_operation_bytes_total{operation="load_get",status="ok",engine="1"} 256',
        ])
        self.assertEqual(runner.store_load_bytes(metrics), 384)

    def test_decode_disables_dbo_explicitly(self):
        args = runner.flags({'enable-prefix-caching': True, 'enable-dbo': False,
                             'kv-transfer-config': {'kv_role': 'kv_consumer'}})
        self.assertIn('--enable-prefix-caching', args)
        self.assertIn('--no-enable-dbo', args)


class ValidatedSweepTest(unittest.TestCase):
    def test_failed_smoke_never_starts_measurement(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'RUN_DIR': tmp, 'CONC_LIST': '32,48,64',
            'ATOMESH_ENV_ATOMESH_VLLM_MODE': 'smoke_then_sweep'
        }), patch.object(runner, 'cleanup') as cleanup:
            seen = []
            def fail():
                seen.append(os.environ['ATOMESH_ENV_ATOMESH_VLLM_MODE'])
                raise RuntimeError('smoke failed')
            with self.assertRaisesRegex(RuntimeError, 'smoke failed'):
                runner.smoke_then_sweep(fail)
            self.assertEqual(seen, ['smoke'])
            cleanup.assert_called_once()
            self.assertEqual(os.environ['RUN_DIR'], tmp)
            states = json.loads((Path(tmp) / 'suite-status.json').read_text())
            self.assertEqual(states, [{'phase': 'smoke', 'concurrency': 1, 'status': 'failed'}])

    def test_each_measurement_follows_teardown_and_uses_separate_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'RUN_DIR': tmp, 'CONC_LIST': '32,48,64',
            'ATOMESH_ENV_ATOMESH_VLLM_MODE': 'smoke_then_sweep'
        }), patch.object(runner, 'cleanup') as cleanup:
            seen = []
            def phase():
                self.assertEqual(cleanup.call_count, len(seen))
                seen.append((os.environ['ATOMESH_ENV_ATOMESH_VLLM_MODE'],
                             os.environ['CONC_LIST'], os.environ['RUN_DIR']))
            runner.smoke_then_sweep(phase)
            self.assertEqual([(m, c) for m, c, _ in seen],
                             [('smoke', '1'), ('benchmark', '32'), ('benchmark', '48'), ('benchmark', '64')])
            self.assertEqual(len({path for _, _, path in seen}), 4)
            self.assertEqual(cleanup.call_count, 4)
            self.assertEqual(os.environ['CONC_LIST'], '32,48,64')


if __name__ == '__main__':
    unittest.main()

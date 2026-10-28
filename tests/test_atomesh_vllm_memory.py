"""Memory recovery gates without touching host memory, GPUs, sudo, or Slurm."""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / '.github/scripts/atomesh'
sys.path.insert(0, str(SCRIPTS))
import pd_vllm_memory as memory


def marker(root, job='42', token='token-42', node='pit2-p03-g12'):
    memory.write_json(root / memory.MARKER, {
        'schema': 1, 'job_id': job, 'run_token': token, 'node': node,
        'reason': 'thp_shortfall', 'phase': 'smoke', 'retry_rc': 86,
    })
    (root / 'rank-rc-0').write_text('86\n')


class MemoryRecoveryTest(unittest.TestCase):
    def test_marker_requires_exact_allocation_token_and_valid_node(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker(root)
            self.assertIsNotNone(memory.valid_marker(root, '42', 'token-42'))
            self.assertIsNone(memory.valid_marker(root, '41', 'token-42'))
            self.assertIsNone(memory.valid_marker(root, '42', 'stale'))
            marker(root, node='../another-job')
            self.assertIsNone(memory.valid_marker(root, '42', 'token-42'))

    def test_no_write_permission_never_writes_sysctls(self):
        with patch.object(memory.os, 'access', return_value=False), \
             patch.object(memory.os, 'sync') as sync, \
             patch.object(Path, 'write_text') as write:
            self.assertFalse(memory.clean_host())
            sync.assert_not_called()
            write.assert_not_called()

    def test_authorized_cleanup_syncs_then_drops_cache_and_compacts(self):
        events = []
        with patch.object(memory.os, 'access', return_value=True), \
             patch.object(memory.os, 'sync', side_effect=lambda: events.append('sync')), \
             patch.object(Path, 'write_text', autospec=True,
                          side_effect=lambda path, data: events.append((str(path), data))):
            self.assertTrue(memory.clean_host())
        self.assertEqual(events, ['sync', ('/proc/sys/vm/drop_caches', '3\n'),
                                  ('/proc/sys/vm/compact_memory', '1\n')])

    def test_host_without_sudo_returns_retry_and_keeps_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker(root)
            with patch.object(memory.os, 'access', return_value=False), \
                 patch.object(memory.shutil, 'which', return_value=None), \
                 patch.object(memory.subprocess, 'run') as run:
                self.assertEqual(memory.recover_host(root, '42', 'token-42', 'pit2-p03-g12'), 86)
                run.assert_not_called()
            self.assertTrue((root / memory.MARKER).exists())
            self.assertEqual(json.loads((root / 'host-memory-recovery.json').read_text())['status'], 'unavailable')

    def test_successful_host_cleanup_archives_old_measurements_and_runs_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            marker(root)
            (root / 'c32/benchmark_results').mkdir(parents=True)
            (root / 'c32/benchmark_results/partial.json').write_text('{}')
            with patch.object(memory.os, 'access', return_value=False), \
                 patch.object(memory.shutil, 'which', return_value='/usr/bin/sudo'), \
                 patch.object(memory.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as run:
                self.assertEqual(memory.recover_host(root, '42', 'token-42', 'pit2-p03-g12'), 0)
                self.assertEqual(run.call_args.args[0][:2], ['/usr/bin/sudo', '-n'])
                self.assertFalse((root / 'c32').exists())
                self.assertTrue((root / 'memory-before-local-retry.tar.gz').exists())
                marker(root)
                self.assertEqual(memory.recover_host(root, '42', 'token-42', 'pit2-p03-g12'), 86)
                self.assertEqual(run.call_count, 1)

    def test_thp_shortfall_is_fatal_and_records_allocation_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase = root / 'smoke'
            phase.mkdir()
            def run(cmd, **kwargs):
                if 'probe' in cmd:
                    memory.write_json(Path(cmd[cmd.index('--output') + 1]),
                                      {'requested_bytes': memory.GIB, 'huge_bytes': memory.GIB // 2})
                return subprocess.CompletedProcess(cmd, 0)
            with patch.dict(os.environ, {'RUN_DIR': str(phase), 'ATOMESH_ALLOCATION_RUN_DIR': str(root),
                     'SLURM_JOB_ID': '42', 'ATOMESH_RUN_TOKEN': 'token-42',
                     'ATOMESH_ALLOCATED_NODE': 'pit2-p03-g12'}), \
                 patch.object(memory, 'plan_pins', return_value={0: 1}), \
                 patch('numa_memory_budget.check_budget', return_value=0), \
                 patch.object(memory.subprocess, 'run', side_effect=run):
                with self.assertRaises(memory.StoreMemoryUnavailable):
                    memory.prepare({}, '/models/snapshot', phase)
            self.assertEqual(memory.valid_marker(root, '42', 'token-42')['reason'], 'thp_shortfall')
            self.assertEqual(json.loads((phase / 'memory-preparation.json').read_text())['status'], 'thp_shortfall')



class PostLoadPreparationTest(unittest.TestCase):
    def test_both_roles_arrive_before_one_preparation_and_all_continue(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'prepare') as prepare:
            root = Path(tmp)
            participants = [(role, rank) for role in ('prefill', 'decode') for rank in range(4)]
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(memory.prepare_after_load, {}, '/snapshot', root, r, i, 5)
                           for r, i in participants[:-1]]
                deadline = time.monotonic() + 3
                while len(list(root.glob('store-memory-gate/*.ready.json'))) < 7:
                    if time.monotonic() > deadline:
                        self.fail('Workers did not reach preparation gate')
                    time.sleep(0.01)
                prepare.assert_not_called()
                self.assertTrue(all(not future.done() for future in futures))
                futures.append(pool.submit(memory.prepare_after_load, {}, '/snapshot', root, 'decode', 3, 5))
                for future in futures:
                    future.result(timeout=6)
            prepare.assert_called_once_with({}, '/snapshot', root)
            result = json.loads((root / 'store-memory-gate/result.json').read_text())
            self.assertEqual(result['status'], 'prepared')
            self.assertEqual(len(result['workers']), 8)

    def test_missing_worker_times_out_without_cleaning(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(memory, 'prepare') as prepare:
            with self.assertRaises(TimeoutError):
                memory.prepare_after_load({}, '/snapshot', tmp, 'prefill', 0, timeout=0)
            prepare.assert_not_called()

    def test_thp_failure_releases_all_peers_and_preserves_retry_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase = root / 'smoke'
            with patch.dict(os.environ, {'RUN_DIR': str(phase), 'ATOMESH_ALLOCATION_RUN_DIR': str(root),
                    'SLURM_JOB_ID': '42', 'ATOMESH_RUN_TOKEN': 'token-42',
                    'ATOMESH_ALLOCATED_NODE': 'pit2-p03-g12'}), \
                 patch.object(memory, 'prepare', side_effect=lambda *args: memory.memory_failure('thp_timeout', phase)) as prepare:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [pool.submit(memory.prepare_after_load, {}, '/snapshot', phase, r, i, 5)
                               for r in ('prefill', 'decode') for i in range(4)]
                    for future in futures:
                        with self.assertRaisesRegex(RuntimeError, 'thp_timeout'):
                            future.result(timeout=6)
                prepare.assert_called_once()
            self.assertEqual(memory.valid_marker(root, '42', 'token-42')['phase'], 'smoke')
            self.assertEqual(json.loads((phase / 'store-memory-gate/result.json').read_text())['status'], 'failed')

    def test_worker_only_initializes_store_after_preparation_succeeds(self):
        events = []
        class NativeWorker:
            def initialize_from_config(self, config):
                events.append('store-init')
                return config
        spec = importlib.util.spec_from_file_location('m31_test_worker', SCRIPTS / 'pd_vllm_worker.py')
        worker_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'vllm.v1.worker.gpu_worker': SimpleNamespace(Worker=NativeWorker)}):
            spec.loader.exec_module(worker_module)
        worker = worker_module.StoreMemoryWorker()
        worker.local_rank = 0
        worker.parallel_config = SimpleNamespace(tensor_parallel_size=4)
        worker.vllm_config = SimpleNamespace(model_config=SimpleNamespace(model='/snapshot'))
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'store.json'
            config.write_text('{}')
            with patch.dict(os.environ, {'MOONCAKE_CONFIG_PATH': str(config),
                    'ATOMESH_VLLM_MEMORY_DIR': tmp, 'ATOMESH_VLLM_ROLE': 'prefill'}), \
                 patch.object(worker_module, 'prepare_after_load', side_effect=lambda *args, **kw: events.append('prepare')):
                self.assertEqual(worker.initialize_from_config('kv-config'), 'kv-config')
                self.assertEqual(events, ['prepare', 'store-init'])
            events.clear()
            with patch.dict(os.environ, {'MOONCAKE_CONFIG_PATH': str(config),
                    'ATOMESH_VLLM_MEMORY_DIR': tmp, 'ATOMESH_VLLM_ROLE': 'prefill'}), \
                 patch.object(worker_module, 'prepare_after_load', side_effect=memory.StoreMemoryUnavailable('shortfall')):
                with self.assertRaises(memory.StoreMemoryUnavailable):
                    worker.initialize_from_config('kv-config')
                self.assertEqual(events, [])


class MemoryResubmitTest(unittest.TestCase):
    def run_loop(self, failures=1, stale=False, retries=3, state='FAILED', pinned=''):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(failures):
                job = str(42 + index)
                folder = root / f'slurm_job-{job}'
                folder.mkdir()
                marker(folder, job=job, token='old' if stale else 'token-' + job,
                       node=f'pit2-p03-g{12 + index}')
            source = (SCRIPTS / 'pd_submit.sh').read_text()
            loop = source[source.index('GPU_BUSY_RETRIES='):source.index('# Preserve the raw scheduler result')]
            script = '''set -euo pipefail
BACKEND=vllm
NUM_NODES=1
SLURM_EXCLUDE_NODES=pit2-p03-g09
SLURM_SUBMIT_RUNNER=atomesh-cicd
SBATCH_CMD=(sbatch --exclude pit2-p03-g09 /fake/job.sh)
count=0
submit_and_monitor_slurm_job() {
  JOB_ID=$((42 + count))
  ATOMESH_RUN_TOKEN="token-${JOB_ID}"
  SLURM_STATUS_DIR="${TEST_ROOT}/slurm_job-${JOB_ID}"
  SLURM_STATE="${TEST_STATE}"
  ((count < FAILURES)) || SLURM_STATE=COMPLETED
  printf '%s\\n' "${SBATCH_CMD[*]}" >> "${TEST_ROOT}/commands"
  count=$((count + 1))
}
''' + loop + '\nprintf "%s" "$count" > "$TEST_ROOT/count"\n'
            result = subprocess.run(['bash', '-c', script], text=True, capture_output=True,
                                    env=dict(os.environ, TEST_ROOT=str(root), REPO_ROOT=str(SCRIPTS.parents[2]),
                                             FAILURES=str(failures), ATOMESH_VLLM_MEMORY_RETRIES=str(retries),
                                             TEST_STATE=state, NODE_LIST=pinned))
            self.assertEqual(result.returncode, 0, result.stderr)
            return int((root / 'count').read_text()), (root / 'commands').read_text()

    def test_memory_failure_resubmits_and_preserves_prior_exclusions(self):
        count, commands = self.run_loop()
        self.assertEqual(count, 2)
        self.assertIn('--exclude pit2-p03-g09,pit2-p03-g12 /fake/job.sh', commands)

    def test_stale_marker_or_nonterminal_allocation_never_resubmits(self):
        self.assertEqual(self.run_loop(stale=True)[0], 1)
        self.assertEqual(self.run_loop(state='RUNNING')[0], 1)

    def test_retry_budget_and_explicit_node_pin_are_respected(self):
        self.assertEqual(self.run_loop(failures=4, retries=2)[0], 3)
        self.assertEqual(self.run_loop(pinned='pit2-p03-g12')[0], 1)


if __name__ == '__main__':
    unittest.main()

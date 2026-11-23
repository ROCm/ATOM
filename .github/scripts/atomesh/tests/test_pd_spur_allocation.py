"""Replay scheduler states at the real batch script's command boundary.

The fixture exercises our dispatch conditions; it does not reproduce Spur's
GPU allocation backend or claim that the cluster resource mismatch is fixed.
"""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
FAKE_CLI = '''#!/usr/bin/env python3
import json, os, pathlib, sys
p = pathlib.Path(os.environ['PD_STATE_FIXTURE'])
d = json.loads(p.read_text())
name = pathlib.Path(sys.argv[0]).name
d['calls'].append([name, *sys.argv[1:]])
if name == 'scontrol':
    assert sys.argv[1:3] == ['--controller', 'http://fixture']
    assert sys.argv[3:] == ['show', 'job', '8540']
    i = min(d['queries'], len(d['snapshots']) - 1)
    d['current'] = d['snapshots'][i]
    d['queries'] += 1
    p.write_text(json.dumps(d))
    print(' '.join(f'{k}={v}' for k,v in d['current'].items()))
elif name == 'srun':
    d['dispatch_restarts'] = os.environ.get('ATOMESH_SPUR_DISPATCH_RESTARTS')
    p.write_text(json.dumps(d))
    if d['current']['JobState'] != 'RUNNING':
        print('Error: failed to create job step; job 8540 is not running (state: Pending)', file=sys.stderr)
        sys.exit(1)
    print('fixture: worker dispatch accepted')
else:
    raise AssertionError(name)
'''


def snapshot(state, restarts=2, job_id=8540, nodes=2):
    return dict(JobId=str(job_id), JobState=state, Restarts=str(restarts),
                NumNodes=str(nodes), NodeList='pit2-p03-g[10,25]')


class BatchDispatchTest(unittest.TestCase):
    def run_batch(self, snapshots, timeout='1', case='c48'):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fixture = td / 'state.json'
            fixture.write_text(json.dumps(dict(snapshots=snapshots, queries=0,
                                               current=snapshots[0], calls=[])))
            for name in ('scontrol', 'srun'):
                path = td / name
                path.write_text(FAKE_CLI)
                path.chmod(0o755)
            env = os.environ.copy()
            env.update(PATH=str(td) + os.pathsep + env['PATH'],
                       PD_STATE_FIXTURE=str(fixture), GITHUB_WORKSPACE=str(ROOT),
                       LOG_ROOT=str(td / 'logs'), SPUR_JOB_ID='8540',
                       SLURM_JOB_ID='8540', SPUR_CONTROLLER_ADDR='http://fixture',
                       NUM_NODES='2', BACKEND='vllm',
                       ATOMESH_ENV_ATOMESH_C48_CASE='V-Q' if case == 'c48' else '',
                       ATOMESH_ENV_ATOMESH_PR5090_ROUTER_AB='1' if case == 'pr5090' else '0',
                       ATOMESH_RUN_TOKEN='a' * 32,
                       ATOMESH_SPUR_READY_POLL_SECONDS='0.01',
                       ATOMESH_SPUR_READY_TIMEOUT_SECONDS=timeout)
            result = subprocess.run(['bash', str(ROOT / '.github/scripts/atomesh/pd_slurm_job.sh')],
                                    env=env, capture_output=True, text=True, timeout=5)
            state = json.loads(fixture.read_text())
            proof = td / 'logs/slurm_job-8540/allocation-ready.json'
            if proof.exists():
                state['allocation_evidence'] = json.loads(proof.read_text())
            return result, state

    def test_pr5090_waits_for_running_before_dispatch(self):
        result, state = self.run_batch([snapshot('PENDING'), snapshot('RUNNING')], case='pr5090')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(row[0] == 'srun' for row in state['calls']), 1)
        self.assertEqual(state['allocation_evidence']['run_token'], 'a' * 32)
        self.assertEqual(state['allocation_evidence']['controller']['JobState'], 'RUNNING')

    def test_pr5090_pending_timeout_publishes_failure_without_dispatch(self):
        result, state = self.run_batch([snapshot('PENDING')], timeout='0.1', case='pr5090')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == 'srun' for row in state['calls']))
        self.assertEqual(state['allocation_evidence']['phase'], 'allocation-check-failed')
        self.assertEqual(state['allocation_evidence']['job_id'], '8540')
        self.assertEqual(state['allocation_evidence']['run_token'], 'a' * 32)

    def test_pending_then_running_dispatches_once_after_confirmation(self):
        result, state = self.run_batch([snapshot('PENDING'), snapshot('RUNNING')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertGreaterEqual(state['queries'], 2)
        self.assertEqual(sum(row[0] == 'srun' for row in state['calls']), 1)
        self.assertEqual(state['calls'][-1][0], 'srun')

    def test_running_control_dispatches_once(self):
        result, state = self.run_batch([snapshot('RUNNING')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(row[0] == 'srun' for row in state['calls']), 1)
        self.assertEqual(state['dispatch_restarts'], '2')

    def test_restart_change_never_dispatches_old_batch(self):
        result, state = self.run_batch([snapshot('PENDING', 1), snapshot('RUNNING', 2)])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == 'srun' for row in state['calls']))

    def test_wrong_job_never_dispatches(self):
        result, state = self.run_batch([snapshot('RUNNING', job_id=9999)])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == 'srun' for row in state['calls']))

    def test_incomplete_allocation_never_dispatches(self):
        result, state = self.run_batch([snapshot('RUNNING', nodes=1)])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == 'srun' for row in state['calls']))

    def test_terminal_job_never_dispatches(self):
        result, state = self.run_batch([snapshot('CANCELLED')])
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(row[0] == 'srun' for row in state['calls']))

    def test_pending_wait_is_bounded(self):
        result, state = self.run_batch([snapshot('PENDING')], timeout='0.1')
        self.assertNotEqual(result.returncode, 0)
        self.assertGreater(state['queries'], 0)
        self.assertFalse(any(row[0] == 'srun' for row in state['calls']))


class StartupEvidenceTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            'spur_allocation', ROOT / '.github/scripts/atomesh/pd_spur_allocation.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.args = SimpleNamespace(job_id='8540', run_token='a' * 32,
                                    controller='http://fixture', num_ranks=2,
                                    run_dir=self.directory, output=None,
                                    state='RUNNING', reason='', timeout_seconds=600)

    def controller(self, state='RUNNING', restart=2):
        row = snapshot(state, restart)
        row.update(Restarts=restart, NumNodes=2)
        return patch.object(self.module, 'query_job', return_value=row)

    def marker(self, restart=2, token=None):
        self.module.save(self.directory / 'worker-entry-0.json', dict(
            job_id='8540', run_token=token or self.args.run_token,
            rank=0, restarts=restart))

    def test_queue_time_does_not_consume_worker_start_deadline(self):
        self.args.state = 'PENDING'
        with self.controller() as query:
            self.assertEqual(self.module.watch_startup(self.args, lambda: 0), 0)
            query.assert_not_called()
            self.args.state = 'RUNNING'
            self.assertEqual(self.module.watch_startup(self.args, lambda: 20000), 0)
            self.assertEqual(self.module.watch_startup(self.args, lambda: 20599), 0)
            self.assertEqual(self.module.watch_startup(self.args, lambda: 20600), 124)

    def test_failed_batch_gate_is_observed_while_queue_stays_pending(self):
        self.args.state = 'PENDING'
        proof = dict(job_id='8540', run_token=self.args.run_token, phase='allocation-check-failed')
        self.module.save(self.directory / 'allocation-ready.json', proof)
        with self.controller() as query:
            self.assertEqual(self.module.watch_startup(self.args, lambda: 0), 124)
            query.assert_not_called()
        for key, value in [('job_id', '9999'), ('run_token', 'b' * 32)]:
            with self.subTest(key=key), self.controller() as query:
                self.module.save(self.directory / 'allocation-ready.json', proof | {key: value})
                self.assertEqual(self.module.watch_startup(self.args, lambda: 0), 0)
                query.assert_not_called()

    def test_dispatch_failure_is_bounded_but_ordinary_queue_wait_is_retained(self):
        self.args.state = 'PENDING'
        cases = [('Resources', 0), ('QOSGrpNodeLimit', 0),
                 ('JobLaunchFailure (dispatch confirmation failed (1/2 confirmed): 1 gpu/resource allocation mismatch)', 124)]
        for reason, expected in cases:
            with self.subTest(reason=reason), self.controller() as query:
                self.args.reason = reason
                self.assertEqual(self.module.watch_startup(self.args, lambda: 0), expected)
                query.assert_not_called()
        proof = json.loads((self.directory / 'startup-watch.json').read_text())
        self.assertEqual(proof['phase'], 'dispatch-confirmation-failed')
        self.assertEqual(proof['job_id'], '8540')
        self.assertEqual(proof['run_token'], self.args.run_token)

    def test_valid_worker_entry_prevents_startup_cancellation(self):
        with self.controller():
            self.module.watch_startup(self.args, lambda: 100)
            self.marker()
            self.assertEqual(self.module.watch_startup(self.args, lambda: 1000), 0)

    def test_wrong_token_is_not_startup_evidence(self):
        with self.controller():
            self.module.watch_startup(self.args, lambda: 100)
            self.marker(token='b' * 32)
            self.assertEqual(self.module.watch_startup(self.args, lambda: 701), 124)

    def test_restart_resets_deadline_and_rejects_old_worker_marker(self):
        with self.controller(restart=2):
            self.module.watch_startup(self.args, lambda: 100)
            self.marker(restart=2)
        with self.controller(restart=3):
            self.assertEqual(self.module.watch_startup(self.args, lambda: 900), 0)
            self.assertEqual(self.module.watch_startup(self.args, lambda: 1500), 124)

    def test_disagreeing_controller_view_resets_observation(self):
        with self.controller():
            self.module.watch_startup(self.args, lambda: 100)
        with self.controller('PENDING'):
            self.assertEqual(self.module.watch_startup(self.args, lambda: 699), 0)
        with self.controller():
            self.assertEqual(self.module.watch_startup(self.args, lambda: 800), 0)

    def test_worker_marker_uses_current_identity_and_generation(self):
        env = dict(SPUR_JOB_ID='8540', SLURM_JOB_ID='8540', SPUR_TASK_OFFSET='0',
                   ATOMESH_SPUR_DISPATCH_RESTARTS='2')
        with patch.dict(os.environ, env):
            self.assertEqual(self.module.worker_entry(self.args), 0)
        self.assertEqual(self.module.started_workers(self.args, 2), [0])
        self.assertEqual(self.module.started_workers(self.args, 3), [])

    def test_previous_active_job_blocks_submission_without_cancellation(self):
        manifest = self.directory / 'previous.json'
        manifest.write_text(json.dumps({'jobs': [{'job_id': '8540'}, {'job_id': '8541'}]}))
        self.args.previous_manifest = manifest
        self.args.output = self.directory / 'previous-check.json'
        with patch.object(self.module, 'command', return_value='8541\n9999\n') as command:
            with self.assertRaisesRegex(RuntimeError, 'remain active'):
                self.module.check_previous(self.args)
            self.assertEqual(command.call_args[0][0][0], 'squeue')
            self.assertEqual(command.call_count, 1)
        self.assertFalse(json.loads(self.args.output.read_text())['verified_inactive'])
        with patch.object(self.module, 'command', return_value='9999\n'):
            self.assertEqual(self.module.check_previous(self.args), 0)
        self.assertTrue(json.loads(self.args.output.read_text())['verified_inactive'])

    def test_previous_job_name_blocks_retry_when_slurm_id_was_lost(self):
        name = 'kimi-k3-c48-gap-c16-mc-vllm-40962044035-1'
        manifest = self.directory / 'previous.json'
        manifest.write_text(json.dumps({'jobs': [
            {'job_id': '8540'}, {'job_name': name, 'run_id': '40962044035'}]}))
        self.args.previous_manifest = manifest
        self.args.output = self.directory / 'previous-check.json'
        with patch.object(self.module, 'command', return_value=f'11150|{name}\n') as query:
            with self.assertRaisesRegex(RuntimeError, 'remain active'):
                self.module.check_previous(self.args)
            self.assertIn('--format=%i|%.256j', query.call_args[0][0])
            self.assertEqual(query.call_count, 1)
        proof = json.loads(self.args.output.read_text())
        self.assertEqual(proof['still_active'], ['11150'])
        self.assertFalse(proof['verified_inactive'])
        with patch.object(self.module, 'command', return_value=f'11151|{name[:-1]}2\n'):
            self.assertEqual(self.module.check_previous(self.args), 0)
        self.assertTrue(json.loads(self.args.output.read_text())['verified_inactive'])
        with patch.object(self.module, 'command', return_value='8540|unrelated-name\n'):
            with self.assertRaisesRegex(RuntimeError, 'remain active'):
                self.module.check_previous(self.args)

    def test_previous_job_name_rejects_incomplete_scheduler_output(self):
        name = 'kimi-k3-c48-gap-c16-mc-vllm-40962044035-1'
        manifest = self.directory / 'previous.json'
        manifest.write_text(json.dumps({'jobs': [{'job_name': name}]}))
        self.args.previous_manifest = manifest
        self.args.output = self.directory / 'previous-check.json'
        for output in ['11150\n', '11150|\n', '11150|%.256j\n', '11150|Unknown\n',
                       '11150|kimi-k3-c48+\n',
                       '11150|kimi-k3-c48\n', '11150|' + 'x' * 256 + '\n']:
            with self.subTest(output=output), patch.object(self.module, 'command', return_value=output):
                with self.assertRaises(RuntimeError):
                    self.module.check_previous(self.args)
        with patch.object(self.module, 'command', side_effect=RuntimeError('query failed')):
            with self.assertRaisesRegex(RuntimeError, 'query failed'):
                self.module.check_previous(self.args)

    def test_monitor_hook_preserves_failure_and_cancels_only_its_job(self):
        self.run_monitor_hook('c48')

    def test_pr5090_monitor_hook_cancels_only_its_failed_startup(self):
        self.run_monitor_hook('pr5090')

    def run_monitor_hook(self, case):
        logs = self.directory / 'logs'
        run_dir = logs / 'slurm_job-8540'
        self.module.save(run_dir / 'startup-watch.json', dict(
            job_id='8540', run_token=self.args.run_token, restarts=2,
            running_since=time.monotonic() - 700, phase='waiting-workers'))
        fixture = self.directory / 'state.json'
        fixture.write_text(json.dumps(dict(snapshots=[snapshot('RUNNING')], queries=0,
                                          current=snapshot('RUNNING'), calls=[])))
        cli = self.directory / 'scontrol'
        cli.write_text(FAKE_CLI)
        cli.chmod(0o755)
        source = (ROOT / '.github/scripts/atomesh/pd_submit.sh').read_text()
        hook = source.split('check_spur_workload_failure() {', 1)[1].split(
            '\n# Every Spur worker', 1)[0]
        code = ('set -euo pipefail\n'
                'scancel_slurm_job() { printf "%s" "$1" > "$PD_CANCEL_RECORD"; }\n'
                'SLURM_EXTRA_STATUS_CHECKER=check_spur_workload_failure\n'
                'SPUR_WORKLOAD_CHECK_RC=0\n'
                'check_spur_workload_failure() {' + hook + '\n'
                'check_spur_workload_failure 8540 RUNNING\n'
                'printf "result:%s:%s" "$SPUR_WORKLOAD_CHECK_RC" "$SLURM_EXTRA_STATUS_CHECKER"\n')
        env = os.environ.copy()
        env.update(PATH=str(self.directory) + os.pathsep + env['PATH'],
                   PD_STATE_FIXTURE=str(fixture), PD_CANCEL_RECORD=str(self.directory / 'cancel'),
                   REPO_ROOT=str(ROOT), LOG_ROOT=str(logs),
                   ATOMESH_RUN_TOKEN=self.args.run_token, NUM_NODES='2',
                   ATOMESH_ENV_ATOMESH_C48_CASE='V-Q' if case == 'c48' else '',
                   ATOMESH_ENV_ATOMESH_PR5090_ROUTER_AB='1' if case == 'pr5090' else '0',
                   SPUR_CONTROLLER_ADDR='http://fixture')
        result = subprocess.run(['bash', '-c', code], env=env, text=True,
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('result:124:', result.stdout)
        self.assertTrue((self.directory / 'cancel').exists())


class MonitorReasonTest(unittest.TestCase):
    def test_unconfirmed_cancellation_cannot_loop_in_pending(self):
        code = '''set -euo pipefail
source "$PD_HELPERS"
stream_slurm_logs_once() { :; }
query_slurm_job() { printf '%s\\n' '8540|PENDING|0:00|2|JobLaunchFailure'; }
slurm_job_in_queue() { return 0; }
cancel_hook() { SCANCEL_SENT=1; }
SLURM_EXTRA_STATUS_CHECKER=cancel_hook
monitor_slurm_job 8540
'''
        env = os.environ.copy()
        env['PD_HELPERS'] = (ROOT / '.github/scripts/slurm_submit_helpers.sh').as_posix()
        result = subprocess.run(['bash', '-c', code], env=env, text=True,
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 75, result.stderr)
        self.assertEqual(result.stdout.count('[slurm] job=8540'), 1)
        self.assertIn('still unconfirmed', result.stderr)

    def test_reason_is_forwarded_and_cleared_after_queue_query_misses(self):
        code = '''set -euo pipefail
source "$PD_HELPERS"
PHASE=0
stream_slurm_logs_once() { :; }
sleep() { :; }
query_slurm_job() {
  if [[ "$PHASE" == 0 ]]; then
    printf '%s\\n' '8540|PENDING|0:00|2|JobLaunchFailure (gpu/resource allocation mismatch)'
  else
    return 1
  fi
}
query_slurm_result() { printf '%s\\n' 'COMPLETED|0:0'; }
record_hook() {
  printf 'hook:%s:%s:%s\\n' "$1" "$2" "$3"
  PHASE=1
}
SLURM_EXTRA_STATUS_CHECKER=record_hook
monitor_slurm_job 8540
'''
        env = os.environ.copy()
        env['PD_HELPERS'] = (ROOT / '.github/scripts/slurm_submit_helpers.sh').as_posix()
        result = subprocess.run(['bash', '-c', code], env=env, text=True,
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('hook:8540:PENDING:JobLaunchFailure (gpu/resource allocation mismatch)', result.stdout)
        self.assertIn('hook:8540:COMPLETED:\n', result.stdout)


if __name__ == '__main__':
    unittest.main()

"""Four-rank diagnostic synchronization must not accept absent or failed peers."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('registration_probe', ROOT / '.github/scripts/atomesh/pd_vllm_registration_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class RegistrationProbeTest(unittest.TestCase):
    def test_missing_rank_times_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rank in [1, 2]:
                probe.dump(root / 'barriers/ready' / f'{rank}.json', {'rank': rank})
            with self.assertRaises(TimeoutError):
                probe.barrier(root, 'ready', 0, timeout=0)

    def test_failure_unblocks_peers_and_all_ready_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            probe.dump(root / 'rank-3/failure.json', {'status': 'failed'})
            with self.assertRaisesRegex(RuntimeError, 'Peer failed'):
                probe.barrier(root, 'ready', 0)
            (root / 'rank-3/failure.json').unlink()
            for rank in [1, 2, 3]:
                probe.dump(root / 'barriers/ready' / f'{rank}.json', {'rank': rank})
            probe.barrier(root, 'ready', 0, timeout=0)


if __name__ == '__main__':
    unittest.main()

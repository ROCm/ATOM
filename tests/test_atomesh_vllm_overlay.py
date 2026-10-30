"""vLLM source overlay for the M3.1 PD experiment, without network, GPUs or vLLM."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / '.github/scripts/atomesh'
sys.path.insert(0, str(SCRIPTS))
import pd_vllm_m31 as launcher

STORE = 'vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py'
SHA = 'c4068bcba669b25a1511054dfb4cbbb24d695f84'


def load_worker_module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / 'pd_vllm_worker.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'vllm.v1.worker.gpu_worker': SimpleNamespace(Worker=object)}):
        spec.loader.exec_module(module)
    return module


class OverlayFetchTest(unittest.TestCase):
    def test_spec_must_pin_a_commit_and_stay_under_vllm(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(launcher.urllib.request, 'urlopen') as urlopen:
            for spec in (f'MengqingCao/vllm@main:{STORE}',
                         f'MengqingCao/vllm:{STORE}',
                         f'vllm@{SHA}:{STORE}',
                         f'MengqingCao/vllm@{SHA}:../etc/passwd',
                         f'MengqingCao/vllm@{SHA}:vllm/README.md'):
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    launcher.fetch_vllm_overlay(spec, Path(tmp))
            urlopen.assert_not_called()

    def test_download_is_pinned_and_recorded_with_its_hash(self):
        body = b'PATCHED = True\n'
        urls = []
        def urlopen(url, timeout):
            urls.append(url)
            return io.BytesIO(body)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(launcher.urllib.request, 'urlopen', side_effect=urlopen):
            root = launcher.fetch_vllm_overlay(f'MengqingCao/vllm@{SHA}:{STORE}', Path(tmp))
            self.assertEqual(urls, [f'https://raw.githubusercontent.com/MengqingCao/vllm/{SHA}/{STORE}'])
            self.assertEqual((root / STORE).read_bytes(), body)
            record = json.loads((Path(tmp) / 'vllm-overlay.json').read_text())
            self.assertEqual(record, {'repo': 'MengqingCao/vllm', 'commit': SHA,
                                      'files': {STORE: hashlib.sha256(body).hexdigest()}})


class StoreExtraConfigTest(unittest.TestCase):
    def test_merges_into_the_store_connector_only(self):
        cfg = {'kv_connector': 'MultiConnector', 'kv_connector_extra_config': {'connectors': [
            {'kv_connector': 'MooncakeConnector', 'kv_connector_extra_config': {'num_workers': 10}},
            {'kv_connector': 'MooncakeStoreConnector', 'kv_connector_extra_config': {'load_async': True}}]}}
        launcher.merge_store_extra_config(cfg, {'prefer_local_segment': 'true', 'save_device_sync': 'true'})
        transfer, store = cfg['kv_connector_extra_config']['connectors']
        self.assertEqual(transfer['kv_connector_extra_config'], {'num_workers': 10})
        self.assertEqual(store['kv_connector_extra_config'],
                         {'load_async': True, 'prefer_local_segment': 'true', 'save_device_sync': 'true'})

    def test_rejects_a_config_without_store(self):
        with self.assertRaises(ValueError):
            launcher.merge_store_extra_config({'kv_connector': 'MooncakeConnector'}, {'save_device_sync': 'true'})


class ImportHookTest(unittest.TestCase):
    def setUp(self):
        self.saved_path = list(sys.path)
        self.saved_meta_path = list(sys.meta_path)
        self.addCleanup(self.restore)

    def restore(self):
        sys.path[:] = self.saved_path
        sys.meta_path[:] = self.saved_meta_path
        for name in [name for name in sys.modules if name == 'vllm' or name.startswith('vllm.')]:
            del sys.modules[name]

    def test_overlay_replaces_only_its_own_modules(self):
        module = load_worker_module('m31_overlay_hook_worker')
        with tempfile.TemporaryDirectory() as tmp:
            image, overlay = Path(tmp) / 'image', Path(tmp) / 'overlay'
            for base in (image, overlay):
                (base / 'vllm/a').mkdir(parents=True)
            (image / 'vllm/__init__.py').write_text('')
            (image / 'vllm/a/__init__.py').write_text('')
            (image / 'vllm/a/mod.py').write_text("WHO = 'image'\n")
            (image / 'vllm/a/other.py').write_text("WHO = 'image'\n")
            (overlay / 'vllm/a/mod.py').write_text("from vllm.a.other import WHO as OTHER\nWHO = 'overlay'\n")
            sys.path.insert(0, str(image))
            module.install_vllm_overlay(str(overlay))
            mod = importlib.import_module('vllm.a.mod')
            other = importlib.import_module('vllm.a.other')
            self.assertEqual((mod.WHO, mod.OTHER, other.WHO), ('overlay', 'image', 'image'))
            with self.assertRaises(RuntimeError):
                module.install_vllm_overlay(str(overlay))


class StoreOverlayCheckTest(unittest.TestCase):
    def check(self, overlay_dir, module_file, **attrs):
        module = load_worker_module('m31_overlay_check_worker')
        store_module = ModuleType('m31_fake_store_worker')
        store_module.__file__ = str(module_file)
        exec('class MooncakeStoreWorker:\n    pass\n', store_module.__dict__)
        worker = store_module.MooncakeStoreWorker()
        worker.store = SimpleNamespace(get_hostname=lambda: '10.0.0.1:12000')
        worker.store_replicate_config = SimpleNamespace(preferred_segment=attrs.pop('preferred', '10.0.0.1:12000'))
        for key, value in attrs.items():
            setattr(worker, key, value)
        group = SimpleNamespace(_connectors=[SimpleNamespace(connector_worker=None),
                                             SimpleNamespace(connector_worker=worker)])
        fake = {name: ModuleType(name) for name in ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu_worker')}
        fake['vllm'].v1 = fake['vllm.v1']
        fake['vllm.v1'].worker = fake['vllm.v1.worker']
        fake['vllm.v1.worker'].gpu_worker = fake['vllm.v1.worker.gpu_worker']
        fake['vllm.v1.worker.gpu_worker'].get_kv_transfer_group = lambda: group
        fake['m31_fake_store_worker'] = store_module
        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, fake):
            path = Path(tmp) / 'overlay.json'
            try:
                module.check_store_overlay(path, overlay_dir)
            finally:
                rows = json.loads(path.read_text())['store_workers']
        return rows

    def test_passes_only_with_overlay_module_switches_and_own_segment(self):
        overlay = '/run_logs/vllm-overlay'
        good = dict(prefer_local_segment=True, save_device_sync=True)
        rows = self.check(overlay, Path(overlay) / STORE, **good)
        self.assertEqual(rows[0]['preferred_segment'], '10.0.0.1:12000')
        for module_file, attrs in ((Path('/usr/lib/vllm') / STORE, good),
                                   (Path(overlay) / STORE, dict(good, save_device_sync=False)),
                                   (Path(overlay) / STORE, dict(prefer_local_segment=True)),
                                   (Path(overlay) / STORE, dict(good, preferred='10.0.0.2:1'))):
            with self.subTest(module_file=str(module_file), attrs=attrs), self.assertRaises(RuntimeError):
                self.check(overlay, module_file, **attrs)


if __name__ == '__main__':
    unittest.main()

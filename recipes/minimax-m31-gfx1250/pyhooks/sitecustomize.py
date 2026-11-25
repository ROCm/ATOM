# sitecustomize.py: on PYTHONPATH for the server. aiter.plugin builds its config snapshot on the first routed call,
# under an RLock; with the checkpoint's MXFP8 linears that first call lands inside torch.compile's trace, Dynamo cannot
# enter the RLock, and the graph break makes ATOM's backend compile twice ("VllmBackend can only be called once"). Build
# the snapshot as soon as aiter.plugin is imported, outside any trace.
# (A sitecustomize, not a usercustomize: inside a venv Python turns the user site off and never imports usercustomize.
# It shadows Ubuntu's /usr/lib/python3.12/sitecustomize.py, which only installs the apport exception hook.)
import importlib.abc
import importlib.util
import sys


class _PrimeAiterPlugin(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != "aiter.plugin":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return None
        self._loader = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        module.reload_config()


sys.meta_path.insert(0, _PrimeAiterPlugin())

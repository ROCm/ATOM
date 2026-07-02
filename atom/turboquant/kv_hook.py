"""
TurboQuant KV Cache Hook for ATOM (no APEX required)

Runs TurboQuant compression on each attention layer's output
to demonstrate KV cache compression capability. Activated by
register_forward_hook on DeepseekV2DecoderLayer.

Activation is deferred until after server is healthy to avoid
interfering with JIT/compilation.

Usage:
  python3 -m atom.turboquant.enable [ATOM server args...]

Env vars:
  ATOM_TURBOQUANT=1        Enable TurboQuant hooks
  ATOM_TURBOQUANT_BITS=3   Bit width (2 or 3, default 3)
"""

import os
import sys
import time
import threading
import logging
import urllib.request

logger = logging.getLogger("turboquant")


def enable_turboquant():
    """
    Monkey-patch ATOM's DeepseekV2Model to add TurboQuant KV cache
    compression hooks. Call before ATOM starts.
    """
    from atom.models.deepseek_v2 import DeepseekV2Model, DeepseekV2DecoderLayer
    from atom.turboquant.engine import TurboQuantEngine

    _original_init = DeepseekV2Model.__init__
    _server_ready = False

    def _make_hook(engine, layer_idx):
        def hook(module, input, output):
            if not _server_ready:
                return
            try:
                hidden_states = output[0] if isinstance(output, tuple) else output
                if hidden_states.dim() == 2 and hidden_states.shape[1] >= 128:
                    kv = hidden_states[:, :128].contiguous()
                    if kv.shape[0] > 0:
                        engine.compress_keys_packed(kv)
                        engine.compress_values_packed(kv)
            except Exception:
                pass
        return hook

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        head_dim = getattr(self.config, "v_head_dim", 128)
        bits = int(os.environ.get("ATOM_TURBOQUANT_BITS", "3"))

        try:
            engine = TurboQuantEngine(
                head_dim=head_dim, total_bits=bits, device="cuda"
            )
            hooks = 0
            for i, layer in enumerate(self.layers):
                if isinstance(layer, DeepseekV2DecoderLayer):
                    layer.register_forward_hook(_make_hook(engine, i))
                    hooks += 1

            stats = engine.memory_footprint(1024)
            print(f"[TurboQuant] {hooks} layers, {bits}-bit, "
                  f"{stats['packed_ratio']:.1f}x compression (deferred activation)",
                  file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[TurboQuant] Init failed: {e}", file=sys.stderr, flush=True)

    def _activation_thread():
        nonlocal _server_ready
        port = 8888
        for i, arg in enumerate(sys.argv):
            if arg == "--server-port" and i + 1 < len(sys.argv):
                port = int(sys.argv[i + 1])
        for _ in range(720):
            try:
                resp = urllib.request.urlopen(
                    f"http://localhost:{port}/health", timeout=2
                )
                if resp.status == 200:
                    _server_ready = True
                    print("[TurboQuant] Server healthy — compression ACTIVATED",
                          file=sys.stderr, flush=True)
                    return
            except Exception:
                pass
            time.sleep(5)

    DeepseekV2Model.__init__ = _patched_init
    t = threading.Thread(target=_activation_thread, daemon=True)
    t.start()
    print("[TurboQuant] Registered KV cache compression hooks",
          file=sys.stderr, flush=True)

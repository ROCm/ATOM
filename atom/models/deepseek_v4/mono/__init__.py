# SPDX-License-Identifier: MIT
"""DeepSeek-V4 Mono decode backend.

``dispatch`` owns deployment admission, ``profile`` declares validated serving
shapes, ``runner`` owns whole-stack execution, ``session`` owns serving and
graph lifetimes, ``cache`` binds production KV/state and publishes metadata,
``state`` owns variant arenas, and ``layer`` wraps one compiled layer.
``kernels/plan`` declares device layouts/tasks, and ``kernels`` contains the Apache-2.0 code
derived from FlyDSL PR #1234.
Importing this package has no installation or launch side effects.
"""

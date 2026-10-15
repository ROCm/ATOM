# SPDX-License-Identifier: MIT
"""DeepSeek-V4 Mono decode backend.

``dispatch`` owns deployment admission, ``profile`` declares validated serving
shapes, ``runner`` owns the token-to-token executor and serving session,
``layer`` wraps one compiled layer, and ``kernels`` contains the Apache-2.0 code
derived from FlyDSL PR #1234.
Importing this package has no installation or launch side effects.
"""

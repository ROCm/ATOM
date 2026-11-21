/*
 * Stub for hipThreadExchangeStreamCaptureMode.
 *
 * When CUDA/HIP graph capture is NOT in use (LUMEN_HIP_GRAPHS=0), the
 * per-kernel-launch capture-mode exchange is pure overhead.  LD_PRELOAD
 * this library to replace the ~10,000+ calls per 3 training steps with
 * a trivial no-op.
 *
 * Safety:
 *   - Only use when HIP graph capture is disabled.
 *   - LD_PRELOAD intercepts external callers (PyTorch → HIP) but not
 *     internal calls within libamdhip64.so.
 *   - NCCL/RCCL do not use CUDA graphs, so distributed communication
 *     is unaffected.
 *
 * Build:
 *   gcc -shared -fPIC -o hip_no_stream_capture.so hip_no_stream_capture.c
 *
 * Usage:
 *   LD_PRELOAD=/path/to/hip_no_stream_capture.so python train.py ...
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

 typedef int hipError_t;
 typedef enum {
     hipStreamCaptureModeGlobal     = 0,
     hipStreamCaptureModeThreadLocal = 1,
     hipStreamCaptureModeRelaxed    = 2,
 } hipStreamCaptureMode;

 #define hipSuccess 0

 hipError_t hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode *mode) {
     /* Report that the previous mode was "relaxed" and keep it relaxed.
      * This satisfies PyTorch's CUDAStreamGuard save/restore protocol
      * without touching any HIP runtime state. */
     *mode = hipStreamCaptureModeRelaxed;
     return hipSuccess;
 }

ATOM
====

**ATOM** (Accelerated Training and Optimization for Models) is AMD's high-performance LLM serving framework optimized for ROCm platforms.
Find the source code at `<https://github.com/ROCm/ATOM>`__.

Features
--------

* **High Performance**: Optimized kernels for AMD Instinct GPUs
* **Model Support**: Wide range of LLM architectures (Llama, GPT, etc.)
* **Distributed Serving**: Multi-GPU and multi-node deployment
* **Compilation**: CUDAGraph and ROCm optimizations
* **Benchmarking**: Built-in performance measurement tools

Supported GPUs
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 20 30

   * - GPU
     - Architecture
     - Memory
     - Status
   * - AMD Instinct MI355X
     - CDNA 4 (gfx950)
     - 288 GB HBM3e
     - ✅ Fully Supported (primary CI target)
   * - AMD Instinct MI300X
     - CDNA 3 (gfx942)
     - 192 GB HBM3
     - ✅ Fully Supported
   * - AMD Instinct MI250X
     - CDNA 2 (gfx90a)
     - 128 GB HBM2e
     - ✅ Fully Supported

Quick links
-----------

* **GitHub**: https://github.com/ROCm/ATOM
* **ROCm Documentation**: https://rocm.docs.amd.com
* **Issues**: https://github.com/ROCm/ATOM/issues

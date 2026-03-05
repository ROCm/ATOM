ATOM Documentation
==================

**ATOM** (Accelerated Training and Optimization for Models) is AMD's high-performance LLM serving framework optimized for ROCm platforms.

.. image:: atom_logo.png
   :align: center
   :width: 400px

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guides

   architecture_guide
   configuration_guide
   model_support_guide
   model_ops_guide
   scheduling_kv_cache_guide
   distributed_guide
   compilation_cudagraph_guide
   serving_benchmarking_guide

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/serving
   api/models

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
   * - AMD Instinct MI300X
     - CDNA 3 (gfx942)
     - 192 GB HBM3
     - ✅ Fully Supported
   * - AMD Instinct MI250X
     - CDNA 2 (gfx90a)
     - 128 GB HBM2e
     - ✅ Fully Supported
   * - AMD Instinct MI300A
     - CDNA 3 (gfx950)
     - 128 GB HBM3
     - 🧪 Experimental

Quick Links
-----------

* **GitHub**: https://github.com/ROCm/ATOM
* **ROCm Documentation**: https://rocm.docs.amd.com
* **Issues**: https://github.com/ROCm/ATOM/issues

Getting Help
------------

* **Documentation**: https://sunway513.github.io/ATOM/
* **GitHub Issues**: https://github.com/ROCm/ATOM/issues
* **ROCm Community**: https://github.com/ROCm/ROCm/discussions

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

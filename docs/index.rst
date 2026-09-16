ATOM
====

**ATOM** (AiTer Optimized Model) is AMD's high-performance LLM serving framework optimized for ROCm platforms.
Find the source code at `<https://github.com/ROCm/ATOM>`__.

Features
--------

* **High Performance**: Optimized kernels for AMD Instinct GPUs
* **Model Implementations**: See the registry-backed :doc:`model_support_guide`
* **Distributed Serving**: Multi-GPU and multi-node deployment
* **Compilation**: CUDAGraph and ROCm optimizations
* **Benchmarking**: Built-in performance measurement tools

Hardware and validation
-----------------------

ATOM uses ROCm and AITER kernels for AMD Instinct GPUs. MI300X (CDNA 3,
``gfx942``) and MI355X (CDNA 4, ``gfx950``) correspond to the kernel build
targets in the project's `CI configurations <https://github.com/ROCm/ATOM/tree/main/.github>`_.
Model, precision, kernel and parallelism choices determine the supported
combination; a GPU architecture name alone is not a full compatibility claim.
Start with :doc:`installation` and the model-specific :doc:`model_run_guide`.

Documentation revision
----------------------

This documentation was built from source commit |source_revision|. The version
in the page title comes from the checked-out Git tag or development revision.
The ``latest`` site tracks development; use `GitHub Releases
<https://github.com/ROCm/ATOM/releases>`_ for published release artifacts and
release notes.

Quick links
-----------

* **GitHub**: https://github.com/ROCm/ATOM
* **ROCm Documentation**: https://rocm.docs.amd.com
* **Issues**: https://github.com/ROCm/ATOM/issues

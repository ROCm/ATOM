Installation
============

ATOM runs on AMD Instinct GPUs via ROCm. This page covers system requirements,
installation, and environment setup.

Requirements
------------

* Python 3.10 or later (see ``requires-python`` in ``pyproject.toml``)
* A compatible ROCm and ROCm-enabled PyTorch installation
* An AMD Instinct GPU supported by your model and kernel configuration

Use the pre-built image below for a matched development stack. The package
metadata alone does not establish compatibility with every ROCm release or GPU;
consult the `current CI configurations <https://github.com/ROCm/ATOM/tree/main/.github>`_
and :doc:`model_run_guide` for model-specific settings.

If ROCm is not yet installed, follow the `ROCm installation guide
<https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ before continuing.

Verify your ROCm installation before proceeding:

.. code-block:: bash

   amd-smi
   rocminfo | grep gfx

Installation methods
--------------------

Choose the method that fits your workflow:

- **From source** — use this when you need to modify ATOM or track the latest
  development changes.
- **Docker** — use this for a pre-configured environment with ROCm, PyTorch,
  and all dependencies already installed. Recommended for most deployments.

From source
^^^^^^^^^^^

.. code-block:: bash

   # Start in an environment with a compatible ROCm-enabled PyTorch installed.
   python -m pip install amd-aiter
   git clone https://github.com/ROCm/ATOM.git
   cd ATOM
   python -m pip install -e .

AITER is installed separately; it is not an ATOM submodule. ATOM declares its
Python dependencies in ``pyproject.toml``. For a non-editable installation, use
``python -m pip install .`` instead. Select an AITER wheel compatible with the
environment, following the `AITER installation guide
<https://rocm.github.io/aiter/installation.html>`_.

For a base container and its matching ROCm/PyTorch versions, see the maintained
`README installation recipe <https://github.com/ROCm/ATOM#installation>`_.

Docker
^^^^^^

The nightly development image includes ROCm, PyTorch, AITER, and ATOM.
``latest`` is a moving tag; record its digest when reproducing a result and
reuse that digest for subsequent runs.

.. code-block:: bash

   docker pull rocm/atom-dev:latest
   docker image inspect rocm/atom-dev:latest --format '{{json .RepoDigests}}'

   docker run --device=/dev/kfd --device=/dev/dri \
              --group-add video --ipc=host \
              -it rocm/atom-dev:latest

``--device=/dev/kfd`` and ``--device=/dev/dri`` expose the GPU to the
container. ``--ipc=host`` is required for multi-GPU workloads that use shared
memory between processes.

Environment variables
---------------------

Set these variables in your shell before building or starting the server:

.. code-block:: bash

   # ROCm installation path (default if installed via package manager)
   export ROCM_PATH=/opt/rocm

   # Suppress AITER kernel log flooding during server startup
   export AITER_LOG_LEVEL=WARNING

``GPU_ARCHS`` is used by kernel build paths such as AITER source builds. Follow
the relevant dependency's build instructions when setting it; it is not an
ATOM runtime support switch. See :doc:`environment_variables` for
``ATOM_*`` runtime variables.

Verify the installation
-----------------------

Run the following to confirm ATOM imported correctly and ROCm is accessible:

.. code-block:: python

   import importlib.metadata
   import atom
   import torch
   import triton

   print("ATOM modules available:")
   print(f"  - LLMEngine: {hasattr(atom, 'LLMEngine')}")
   print(f"  - SamplingParams: {hasattr(atom, 'SamplingParams')}")

   print(f"\nPyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip}")
   print(f"Triton version: {triton.__version__}")
   for distribution in ("atom", "amd-aiter"):
       print(f"{distribution}: {importlib.metadata.version(distribution)}")

A successful installation prints ``True`` for both ``LLMEngine`` and
``SamplingParams``, reports an available GPU, and shows a nonempty ROCm version
string. Save this output together with the container digest or source commits
when reporting a regression.

Troubleshooting
---------------

**ImportError: No module named 'atom'**
   The ATOM package is not on ``PYTHONPATH``. If you installed from source with
   ``pip install -e .``, confirm you are in the same virtual environment.
   Also ensure ROCm libraries are on the library path:

   .. code-block:: bash

      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**RuntimeError: No AMD GPU found**
   The GPU is not visible to the process. Check that ``amd-smi`` lists your
   device and that the ROCm kernel modules are loaded:

   .. code-block:: bash

      amd-smi
      rocminfo | grep gfx

   In Docker, confirm you passed ``--device=/dev/kfd --device=/dev/dri`` when
   starting the container.

**AITER log flooding on startup**
   AITER prints kernel selection logs by default. Suppress them with:

   .. code-block:: bash

      export AITER_LOG_LEVEL=WARNING

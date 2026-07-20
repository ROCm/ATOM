Installation
============

ATOM runs on AMD Instinct GPUs via ROCm. This page covers system requirements,
installation, and environment setup.

Requirements
------------

* Python 3.10 to 3.12
* `ROCm 6.0 or later <https://rocm.docs.amd.com/en/latest/install/rocm.html>`_
* PyTorch with ROCm support
* AMD Instinct GPU (MI200, MI300, or MI350 series recommended)

If ROCm is not yet installed, follow the `ROCm installation guide
<https://rocm.docs.amd.com/en/latest/install/rocm.html>`_ before continuing.

Verify your ROCm installation before proceeding:

.. code-block:: bash

   rocm-smi
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

   git clone --recursive https://github.com/ROCm/ATOM.git
   cd ATOM
   pip install -r requirements.txt
   pip install -e .

The ``--recursive`` flag is required because ATOM depends on AITER as a
submodule.

Docker
^^^^^^

The pre-built image includes ROCm, PyTorch, and all ATOM dependencies.

.. code-block:: bash

   docker pull rocm/atom:latest

   docker run --device=/dev/kfd --device=/dev/dri \
              --group-add video --ipc=host \
              -it rocm/atom:latest

``--device=/dev/kfd`` and ``--device=/dev/dri`` expose the GPU to the
container. ``--ipc=host`` is required for multi-GPU workloads that use shared
memory between processes.

Environment variables
---------------------

Set these variables in your shell before building or starting the server:

.. code-block:: bash

   # ROCm installation path (default if installed via package manager)
   export ROCM_PATH=/opt/rocm

   # Target GPU architectures — include every architecture you intend to run on
   # gfx90a = MI250X, gfx942 = MI300X, gfx950 = MI355X
   export GPU_ARCHS="gfx90a;gfx942"

   # Suppress AITER kernel log flooding during server startup
   export AITER_LOG_LEVEL=WARNING

``GPU_ARCHS`` controls which GPU targets are compiled. Omitting an architecture
means ATOM will not run on that GPU. See :doc:`environment_variables` for a
full list of ``ATOM_*`` runtime variables.

Verify the installation
-----------------------

Run the following to confirm ATOM imported correctly and ROCm is accessible:

.. code-block:: python

   import atom
   import torch

   print("ATOM modules available:")
   print(f"  - LLMEngine: {hasattr(atom, 'LLMEngine')}")
   print(f"  - SamplingParams: {hasattr(atom, 'SamplingParams')}")

   print(f"\nPyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

A successful installation prints ``True`` for both ``LLMEngine`` and
``SamplingParams``, and shows a ROCm version string rather than ``N/A``.

Troubleshooting
---------------

**ImportError: No module named 'atom'**
   The ATOM package is not on ``PYTHONPATH``. If you installed from source with
   ``pip install -e .``, confirm you are in the same virtual environment.
   Also ensure ROCm libraries are on the library path:

   .. code-block:: bash

      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**RuntimeError: No AMD GPU found**
   The GPU is not visible to the process. Check that ``rocm-smi`` lists your
   device and that the ROCm kernel modules are loaded:

   .. code-block:: bash

      rocm-smi
      rocminfo | grep gfx

   In Docker, confirm you passed ``--device=/dev/kfd --device=/dev/dri`` when
   starting the container.

**AITER log flooding on startup**
   AITER prints kernel selection logs by default. Suppress them with:

   .. code-block:: bash

      export AITER_LOG_LEVEL=WARNING

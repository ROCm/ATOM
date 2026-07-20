Installation
============

Requirements
------------

* Python 3.10 to 3.12
* ROCm 6.0 or later
* PyTorch with ROCm support
* AMD Instinct GPU (MI200 or MI300 series recommended)

Installation methods
--------------------

From source
^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone --recursive https://github.com/ROCm/ATOM.git
   cd ATOM

   # Install dependencies
   pip install -r requirements.txt

   # Build and install
   pip install -e .

Docker
^^^^^^

.. code-block:: bash

   # Pull the pre-built image
   docker pull rocm/atom:latest

   # Run the container
   docker run --device=/dev/kfd --device=/dev/dri \
              --group-add video --ipc=host \
              -it rocm/atom:latest

Environment variables
---------------------

Set these variables before starting the server:

.. code-block:: bash

   # ROCm installation path
   export ROCM_PATH=/opt/rocm

   # Target GPU architectures (semicolon-separated)
   export GPU_ARCHS="gfx90a;gfx942"

   # Suppress AITER kernel log flooding
   export AITER_LOG_LEVEL=WARNING

See :doc:`environment_variables` for a full list of ``ATOM_*`` variables.

Verify the installation
-----------------------

.. code-block:: python

   import atom
   import torch

   print("ATOM modules available:")
   print(f"  - LLMEngine: {hasattr(atom, 'LLMEngine')}")
   print(f"  - SamplingParams: {hasattr(atom, 'SamplingParams')}")

   print(f"\nPyTorch version: {torch.__version__}")
   print(f"ROCm available: {torch.cuda.is_available()}")
   print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")

Troubleshooting
---------------

**ImportError: No module named 'atom'**
   Ensure ROCm libraries are in your library path:

   .. code-block:: bash

      export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH

**RuntimeError: No AMD GPU found**
   Verify the GPU is accessible:

   .. code-block:: bash

      rocm-smi
      rocminfo | grep gfx

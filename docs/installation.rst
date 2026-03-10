Installation
============

Requirements
------------

* Python 3.10 to 3.12
* ROCm 6.0 or later
* PyTorch with ROCm support
* AMD Instinct GPU (MI200 or MI300 series recommended)

Installation Methods
--------------------

From Source
^^^^^^^^^^^

.. code-block:: bash

   # Clone the repository
   git clone --recursive https://github.com/ROCm/ATOM.git
   cd ATOM

   # Install dependencies
   pip install -r requirements.txt

   # Build and install
   python3 setup.py develop

Docker Installation
^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Pull pre-built image
   docker pull rocm/atom:latest

   # Run container
   docker run --device=/dev/kfd --device=/dev/dri \
              --group-add video --ipc=host \
              -it rocm/atom:latest

Environment Variables
---------------------

Required environment variables:

.. code-block:: bash

   # ROCm installation path
   export ROCM_PATH=/opt/rocm

   # GPU architectures
   export GPU_ARCHS="gfx90a;gfx942"

   # ATOM serving configuration
   export ATOM_CACHE_DIR=/tmp/atom_cache
   export ATOM_MAX_BATCH_SIZE=128

Verification
------------

Verify the installation:

.. code-block:: python

   import atom
   import torch

   # Check if ATOM modules loaded successfully
   print("ATOM modules available:")
   print(f"  - LLMEngine: {hasattr(atom, 'LLMEngine')}")
   print(f"  - SamplingParams: {hasattr(atom, 'SamplingParams')}")

   # Check ROCm availability via PyTorch
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
   Verify GPU is accessible:

   .. code-block:: bash

      rocm-smi
      rocminfo | grep gfx

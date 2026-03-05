Supported Models
================

ATOM supports a wide range of LLM architectures optimized for AMD GPUs.

Llama Models
------------

Meta's Llama family:

* Llama 2 (7B, 13B, 70B)
* Llama 3 (8B, 70B)
* CodeLlama
* Llama-2-Chat

**Example:**

.. code-block:: python

   from atom import LLM

   llm = LLM(model="meta-llama/Llama-2-7b-hf")

GPT Models
----------

GPT-style architectures:

* GPT-2
* GPT-J
* GPT-NeoX

**Example:**

.. code-block:: python

   llm = LLM(model="EleutherAI/gpt-j-6b")

Mixtral
-------

Mixture of Experts models:

* Mixtral 8x7B
* Mixtral 8x22B

**Example:**

.. code-block:: python

   llm = LLM(
       model="mistralai/Mixtral-8x7B-v0.1",
       tensor_parallel_size=4
   )

Other Architectures
-------------------

* **Mistral**: Mistral-7B
* **Falcon**: Falcon-7B, Falcon-40B
* **MPT**: MPT-7B, MPT-30B
* **BLOOM**: BLOOM-7B1

Model Configuration
-------------------

Custom model configurations:

.. code-block:: python

   from atom import LLM

   llm = LLM(
       model="/path/to/custom/model",
       trust_remote_code=True,  # For custom architectures
       dtype="bfloat16",
       max_model_len=8192
   )

Performance by Model Size
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Model Size
     - Recommended GPU
     - Tensor Parallel
     - Batch Size
   * - 7B
     - 1x MI250X
     - 1
     - 32-64
   * - 13B
     - 1x MI250X
     - 1
     - 16-32
   * - 30B
     - 2x MI250X
     - 2
     - 8-16
   * - 70B
     - 4x MI300X
     - 4
     - 4-8

Quantization
------------

ATOM supports quantized models for reduced memory:

.. code-block:: python

   llm = LLM(
       model="TheBloke/Llama-2-7B-GPTQ",
       quantization="gptq"
   )

Supported quantization formats:

* GPTQ
* AWQ
* SqueezeLLM

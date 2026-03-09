Serving API
===========

LLMEngine Class
---------------

Main class for loading and serving models.

.. code-block:: python

   from atom import LLMEngine

   llm = LLMEngine(model="meta-llama/Llama-2-7b-hf")

**Parameters:**

* **model** (*str*) - HuggingFace model name or path
* **gpu_memory_utilization** (*float*) - GPU memory usage (0.0-1.0). Default: 0.9
* **max_model_len** (*int*) - Maximum sequence length
* **tensor_parallel_size** (*int*) - Number of GPUs for tensor parallelism. Default: 1
* **dtype** (*str*) - Model dtype ('float16', 'bfloat16', 'float32')

Methods
^^^^^^^

generate()
""""""""""

.. code-block:: python

   sampling_params = SamplingParams(max_tokens=50, temperature=0.8)
   outputs = llm.generate(prompts, sampling_params)

Generate text from prompts.

**Parameters:**

* **prompts** (*list[str]*) - Input prompts (must be a list, even for single prompt)
* **sampling_params** (*SamplingParams | list[SamplingParams]*) - Sampling configuration

**Returns:**

* **outputs** (*list[str]*) - Generated text strings

.. note::
   Unlike some APIs, ``generate()`` requires prompts to be a list and returns
   a list of strings, not RequestOutput objects. Parameters like max_tokens
   must be specified via SamplingParams.

SamplingParams
--------------

.. code-block:: python

   from atom import SamplingParams

   params = SamplingParams(
       temperature=0.8,
       max_tokens=100,
       ignore_eos=False,
       stop_strings=["</s>", "\n\n"]
   )

Configuration for text generation.

**Parameters:**

* **temperature** (*float*) - Controls randomness. Default: 1.0
* **max_tokens** (*int*) - Maximum tokens to generate. Default: 64
* **ignore_eos** (*bool*) - Whether to ignore EOS token. Default: False
* **stop_strings** (*list[str] | None*) - Strings that stop generation. Default: None

.. note::
   The following parameters are NOT currently supported (may be added in future):
   top_p, top_k, presence_penalty, frequency_penalty

Return Values
-------------

The ``generate()`` method returns a list of strings (not RequestOutput objects).

.. code-block:: python

   outputs = llm.generate(["Hello, world!"], sampling_params)
   # outputs is list[str], e.g., ["Hello, world! How are you today?"]

.. note::
   Unlike some LLM serving frameworks, ATOM's generate() method returns
   plain strings, not structured output objects. If you need token IDs
   or other metadata, these are not currently exposed in the API.

Example
-------

Complete example:

.. code-block:: python

   from atom import LLM, SamplingParams

   # Initialize model
   llm = LLM(
       model="meta-llama/Llama-2-7b-hf",
       tensor_parallel_size=2,
       gpu_memory_utilization=0.9
   )

   # Configure sampling
   sampling_params = SamplingParams(
       temperature=0.7,
       top_p=0.9,
       max_tokens=200
   )

   # Generate
   prompts = ["Tell me about AMD GPUs"]
   outputs = llm.generate(prompts, sampling_params=sampling_params)

   for output in outputs:
       print(f"Prompt: {output.prompt}")
       print(f"Generated: {output.text}")

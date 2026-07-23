Serving API
===========

LLMEngine class
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

* **outputs** (*list[dict]*) - One dict per prompt. Each dict contains at
  minimum a ``"text"`` key with the generated string, plus ``"token_ids"``,
  ``"latency"``, ``"finish_reason"``, ``"num_tokens_input"``,
  ``"num_tokens_output"``, ``"ttft"``, and ``"logprobs"``.

.. note::
   ``generate()`` requires prompts to be a list. Access generated text via
   ``outputs[i]["text"]``. Parameters like ``max_tokens`` must be specified
   via ``SamplingParams``.

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
   ``presence_penalty`` and ``frequency_penalty`` are not currently supported.
   ``top_p`` and ``top_k`` are supported (``top_k=-1`` disables it;
   ``top_p=1.0`` disables it).

Return values
-------------

The ``generate()`` method returns a ``list[dict]``. Access the generated
text via the ``"text"`` key:

.. code-block:: python

   outputs = llm.generate(["Hello, world!"], sampling_params)
   print(outputs[0]["text"])  # e.g., "Hello, world! How are you today?"

Example
-------

Complete example:

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   # Initialize model
   llm = LLMEngine(
       model="meta-llama/Llama-2-7b-hf",
       tensor_parallel_size=2,
       gpu_memory_utilization=0.9,
   )

   # Configure sampling
   sampling_params = SamplingParams(
       temperature=0.7,
       top_p=0.9,
       max_tokens=200,
   )

   # Generate
   prompts = ["Tell me about AMD GPUs"]
   outputs = llm.generate(prompts, sampling_params=sampling_params)

   for prompt, output in zip(prompts, outputs):
       print(f"Prompt: {prompt}")
       print(f"Generated: {output['text']}")

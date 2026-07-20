Quick start
===========

This guide will get you started with ATOM in 5 minutes.

Serve a model
-------------

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   # Load model
   llm = LLMEngine(
       model="meta-llama/Llama-2-7b-hf",
       gpu_memory_utilization=0.9,
       max_model_len=4096,
   )

   # Create sampling parameters
   sampling_params = SamplingParams(max_tokens=50, temperature=0.8)

   # Generate text (prompts must be a list)
   outputs = llm.generate(["Hello, my name is"], sampling_params)
   print(outputs[0])

Run batch inference
-------------------

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   llm = LLMEngine(model="meta-llama/Llama-2-7b-hf")

   prompts = [
       "The capital of France is",
       "The largest ocean is",
       "Python is a",
   ]
   sampling_params = SamplingParams(max_tokens=20, temperature=0.7)

   outputs = llm.generate(prompts, sampling_params)
   for prompt, output in zip(prompts, outputs):
       print(f"Prompt: {prompt}")
       print(f"Output: {output}\n")

Use distributed serving
-----------------------

Distribute a large model across multiple GPUs with tensor parallelism:

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   llm = LLMEngine(
       model="meta-llama/Llama-2-70b-hf",
       tensor_parallel_size=4,
       gpu_memory_utilization=0.95,
   )

   sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
   outputs = llm.generate(["Tell me about AMD GPUs"], sampling_params)
   print(outputs[0])

Start the API server
--------------------

Start an OpenAI-compatible REST API server:

.. code-block:: bash

   python -m atom.entrypoints.openai_server \
       --model meta-llama/Llama-2-7b-hf \
       --host 0.0.0.0 \
       --port 8000

Query the server using the ``/v1/completions`` endpoint:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/v1/completions",
       json={
           "model": "meta-llama/Llama-2-7b-hf",
           "prompt": "Hello, world!",
           "max_tokens": 50,
       },
   )
   print(response.json()["choices"][0]["text"])

Performance tips
----------------

1. **GPU memory**: Set ``gpu_memory_utilization`` to 0.9–0.95 for maximum throughput.
2. **Batch size**: Increase ``max_num_batched_tokens`` to improve throughput under load.
3. **KV cache**: Tune ``block_size`` based on your sequence length distribution.
4. **CUDA graphs**: Leave compilation at the default level (3) to enable CUDA graph capture for decode.

Next steps
----------

* :doc:`architecture_guide` — Understand ATOM architecture
* :doc:`configuration_guide` — Configure for your workload
* :doc:`serving_benchmarking_guide` — Measure performance

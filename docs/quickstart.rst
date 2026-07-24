Quick start
===========

This guide shows the most common ways to use ATOM: offline batch inference,
distributed inference across multiple GPUs, and an OpenAI-compatible API server.
All examples assume ATOM is installed. If it isn't, follow the :doc:`installation`
guide first.

Offline inference
-----------------

Use ``LLMEngine`` directly when you want to run inference in a Python script
without a server. Load the model once, then call ``generate()`` with a list of
prompts and a ``SamplingParams`` object.

Single prompt
^^^^^^^^^^^^^

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   llm = LLMEngine(
       model="meta-llama/Llama-2-7b-hf",
       gpu_memory_utilization=0.9,
       max_model_len=4096,
   )

   sampling_params = SamplingParams(max_tokens=50, temperature=0.8)
   outputs = llm.generate(["Hello, my name is"], sampling_params)
   print(outputs[0])

``gpu_memory_utilization=0.9`` reserves 90% of GPU VRAM for the KV cache, leaving
the rest for model weights and activations. ``max_model_len`` caps the combined
prompt-plus-completion length.

Batch inference
^^^^^^^^^^^^^^^

Pass a list of prompts to process them together. ATOM schedules them as a batch
to maximize GPU utilization:

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

Distributed inference
---------------------

For models that exceed the memory of a single GPU, use tensor parallelism to
split the model across multiple GPUs. Set ``tensor_parallel_size`` to the number
of GPUs you want to use:

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

``tensor_parallel_size`` must divide evenly into the model's attention heads.
For most 70B-class models, 4 or 8 GPUs are typical. See :doc:`distributed_guide`
for pipeline parallelism and multi-node configurations.

API server
----------

ATOM provides an OpenAI-compatible REST API server. Start the server with the
model you want to serve:

.. code-block:: bash

   python -m atom.entrypoints.openai_server \
       --model meta-llama/Llama-2-7b-hf \
       --host 0.0.0.0 \
       --port 8000

The server accepts requests at ``/v1/completions`` (single-turn) and
``/v1/chat/completions`` (multi-turn chat). Query it with any OpenAI-compatible
client:

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

* **GPU memory**: Set ``gpu_memory_utilization`` to 0.9–0.95 for maximum KV
  cache size. Lower values leave more room for other processes but reduce
  throughput.
* **Batch size**: Increase ``max_num_batched_tokens`` to improve throughput
  when you expect many concurrent requests.
* **KV cache**: Tune ``block_size`` based on your typical sequence length.
  Shorter sequences benefit from a smaller block size.
* **CUDA graphs**: Leave the default compilation level (3) enabled. This
  captures decode steps as CUDA graphs, which reduces kernel-launch overhead.

Next steps
----------

* :doc:`architecture_guide` — Understand how ATOM processes requests end to end
* :doc:`configuration_guide` — Tune every knob for your workload
* :doc:`serving_benchmarking_guide` — Measure throughput and latency

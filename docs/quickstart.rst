Quickstart
==========

This guide will get you started with ATOM in 5 minutes.

Serving a Model
---------------

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   # Load model
   llm = LLMEngine(
       model="meta-llama/Llama-2-7b-hf",
       gpu_memory_utilization=0.9,
       max_model_len=4096
   )

   # Create sampling parameters
   sampling_params = SamplingParams(max_tokens=50, temperature=0.8)

   # Generate text (note: prompts must be a list)
   outputs = llm.generate(["Hello, my name is"], sampling_params)
   print(outputs[0])

Batch Inference
---------------

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   llm = LLMEngine(model="meta-llama/Llama-2-7b-hf")

   # Batch prompts
   prompts = [
       "The capital of France is",
       "The largest ocean is",
       "Python is a"
   ]

   # Create sampling parameters
   sampling_params = SamplingParams(max_tokens=20, temperature=0.7)

   # Generate in batch
   outputs = llm.generate(prompts, sampling_params)

   # outputs is a list of strings
   for i, output in enumerate(outputs):
       print(f"Prompt: {prompts[i]}")
       print(f"Output: {output}\n")

Distributed Serving
-------------------

Multi-GPU serving:

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   # Use 4 GPUs with tensor parallelism
   llm = LLMEngine(
       model="meta-llama/Llama-2-70b-hf",
       tensor_parallel_size=4,
       gpu_memory_utilization=0.95
   )

   sampling_params = SamplingParams(max_tokens=100, temperature=0.7)
   outputs = llm.generate(["Tell me about AMD GPUs"], sampling_params)
   print(outputs[0])

API Server
----------

Start a RESTful API server:

.. code-block:: bash

   python -m atom.entrypoints.openai_server \
       --model meta-llama/Llama-2-7b-hf \
       --host 0.0.0.0 \
       --port 8000

Query the server:

.. code-block:: python

   import requests

   response = requests.post(
       "http://localhost:8000/generate",
       json={
           "prompt": "Hello, world!",
           "max_tokens": 50
       }
   )

   print(response.json()["text"])

Performance Tips
----------------

1. **GPU Memory**: Set `gpu_memory_utilization` to 0.9-0.95
2. **Batch Size**: Increase `max_num_batched_tokens` for throughput
3. **KV Cache**: Configure `block_size` based on workload
4. **Compilation**: Enable CUDAGraph for repeated inference

Next Steps
----------

* :doc:`architecture_guide` - Understand ATOM architecture
* :doc:`configuration_guide` - Configure for your workload
* :doc:`serving_benchmarking_guide` - Measure performance

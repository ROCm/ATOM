Model reference
===============

The :doc:`../model_support_guide` is the authoritative inventory of ATOM's
native model implementations. Its architecture table is generated from
``support_model_arch_dict`` at documentation build time, without importing GPU
libraries. It also explains model loading and how to add an implementation.

A registry entry identifies an implementation, not a guarantee that every
checkpoint, precision, parallelism configuration, or GPU has been validated.
Use :doc:`../model_run_guide` for launch recipes and
:doc:`../online_quantization_guide` for the implemented online quantization
paths and their restrictions.

Load models with the public ``LLMEngine`` interface:

.. code-block:: python

   from atom import LLMEngine, SamplingParams

   llm = LLMEngine(model="meta-llama/Meta-Llama-3-8B")
   outputs = llm.generate(["What is ROCm?"], SamplingParams(max_tokens=32))
   print(outputs[0]["text"])

See :doc:`serving` for configuration, result dictionaries and multiple
completions per prompt.

FROM vllm/vllm-openai-rocm:minimax-m31-flash-75bc8cf1@sha256:f8e8ab239e5edb7c3070c5802fc7fc0cbb889513bf81183dfc98ce1c77e1dfe1
COPY pp-production.patch prepare_worker.py verify_worker.py atomesh_mori_probe.py test_pp_contract.py test_write_contract.py manifest.json /opt/atomesh-mori/
RUN python3 /opt/atomesh-mori/prepare_worker.py && python3 /opt/atomesh-mori/verify_worker.py

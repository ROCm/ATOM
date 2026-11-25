FROM vllm/vllm-openai-rocm:minimax-m31-flash-75bc8cf1@sha256:f8e8ab239e5edb7c3070c5802fc7fc0cbb889513bf81183dfc98ce1c77e1dfe1
COPY mooncake_timeout_probe.cpp prepare_worker.py /opt/atomesh-timeout/
COPY atomesh_timeout_probe.py /usr/local/lib/python3.12/dist-packages/atomesh_timeout_probe.py
RUN python3 /opt/atomesh-timeout/prepare_worker.py && g++ -std=c++17 -O2 -Wall -Wextra -Werror -fPIC -shared /opt/atomesh-timeout/mooncake_timeout_probe.cpp -ldl -pthread -o /opt/atomesh-timeout/libmooncake_timeout_probe.so

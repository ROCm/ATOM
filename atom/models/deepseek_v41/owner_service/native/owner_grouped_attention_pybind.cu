// SPDX-License-Identifier: MIT
#include "rocm_ops.hpp"
#include "aiter_stream.h"
#include "owner_grouped_attention.h"

PYBIND11_MODULE(AITER_EXTENSION_NAME, m)
{
    AITER_SET_STREAM_PYBIND
    m.def("owner_grouped_attention_fwd", &owner_grouped_attention_fwd,
          py::arg("q"), py::arg("unified_kv"), py::arg("kv_indices_prefix"),
          py::arg("kv_indptr_prefix"), py::arg("kv"), py::arg("kv_indices_extend"),
          py::arg("kv_indptr_extend"), py::arg("attn_sink"), py::arg("partial_o"),
          py::arg("partial_max"), py::arg("partial_sum"), py::arg("out"),
          py::arg("softmax_scale"), py::arg("num_splits"));
}

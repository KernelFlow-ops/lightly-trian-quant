// pybind module for the round12 attention kernels.
#include "attn_api.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("attn_fwd", &a12::attn_fwd, "packed-qkv attention forward with fused RoPE / query bitmask -> (O, lse2, qk_rot)", py::arg("qkv"), py::arg("heads"),
          py::arg("sin") = py::none(), py::arg("cos") = py::none(), py::arg("bits") = py::none(), py::arg("prefix") = -1, py::arg("mask_rows") = 0,
          py::arg("mask_key0") = 0, py::arg("save_rot") = true);
    m.def("attn_bwd", &a12::attn_bwd, "packed-qkv attention backward (dqkv) with fused inverse RoPE / query bitmask", py::arg("qkv"), py::arg("qk_rot"),
          py::arg("o"), py::arg("lse2"), py::arg("do"), py::arg("heads"), py::arg("sin") = py::none(), py::arg("cos") = py::none(), py::arg("bits") = py::none(),
          py::arg("prefix") = -1, py::arg("mask_rows") = 0, py::arg("mask_key0") = 0);
    m.def("build_bits", &a12::build_bits, "mask logits -> QueryBitmask words", py::arg("logits"), py::arg("rnd") = py::none(), py::arg("prob") = py::none());
}

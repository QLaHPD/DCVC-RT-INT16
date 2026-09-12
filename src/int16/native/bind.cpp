#include <torch/extension.h>
torch::Tensor uf_conv(const torch::Tensor&, const torch::Tensor&,
    const c10::optional<torch::Tensor>&, int, int, int, int, int);
torch::Tensor uf_conv_generic(const torch::Tensor&, const torch::Tensor&,
    const c10::optional<torch::Tensor>&, int, int, int, int, int);
torch::Tensor uf_conv_baseline(const torch::Tensor&, const torch::Tensor&,
    const c10::optional<torch::Tensor>&, int, int, int, int, int);
torch::Tensor uf_add(const std::vector<torch::Tensor>&);
torch::Tensor uf_mul(const torch::Tensor&, const torch::Tensor&);
torch::Tensor uf_lut(const torch::Tensor&, const torch::Tensor&);
torch::Tensor uf_bytes(const torch::Tensor&);
torch::Tensor uf_wsilu4(const torch::Tensor&, const torch::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("arithmetic_id")="uf-int16-q9-w13-acc64-rna-sat-v1";
    m.attr("kernel_revision")="tiled-v2";
    m.def("conv2d", &uf_conv);
    m.def("conv2d_generic", &uf_conv_generic);
    m.def("conv2d_baseline", &uf_conv_baseline);
    m.def("add", &uf_add);
    m.def("multiply", &uf_mul);
    m.def("lookup", &uf_lut);
    m.def("from_bytes", &uf_bytes);
    m.def("wsilu4", &uf_wsilu4);
}

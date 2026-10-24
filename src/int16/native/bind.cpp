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
torch::Tensor uf_bytes_unshuffle(const torch::Tensor&);
torch::Tensor uf_wsilu4(const torch::Tensor&, const torch::Tensor&);
torch::Tensor uf_raw_frame(const torch::Tensor&,int,int);
torch::Tensor uf_raw_frames(const std::vector<torch::Tensor>&,int,int);
torch::Tensor uf_symbols(const torch::Tensor&);
torch::Tensor uf_from_symbols(const torch::Tensor&);
torch::Tensor uf_reciprocal(const torch::Tensor&);
torch::Tensor uf_depth_block(const torch::Tensor&,const std::vector<torch::Tensor>&,
    const std::vector<c10::optional<torch::Tensor>>&,const torch::Tensor&,bool);
std::pair<torch::Tensor,c10::optional<torch::Tensor>> uf_depth_block_packed(
    const torch::Tensor&,const std::vector<torch::Tensor>&,
    const std::vector<c10::optional<torch::Tensor>>&,const torch::Tensor&,bool,
    const c10::optional<torch::Tensor>&,bool);
torch::Tensor uf_conv_tiled(const torch::Tensor&, const torch::Tensor&, const c10::optional<torch::Tensor>&, int);
torch::Tensor uf_conv_wsilu4(const torch::Tensor&, const torch::Tensor&, const c10::optional<torch::Tensor>&,
    const torch::Tensor&,int);
torch::Tensor uf_conv_post(const torch::Tensor&,const torch::Tensor&,const c10::optional<torch::Tensor>&,
    const c10::optional<torch::Tensor>&,const c10::optional<torch::Tensor>&,const c10::optional<torch::Tensor>&,int);
torch::Tensor uf_prior_encode(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&,
    torch::Tensor, const torch::Tensor&, const torch::Tensor&, int);
torch::Tensor uf_prior_indexes(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int);
void uf_prior_restore(const torch::Tensor&, const torch::Tensor&, torch::Tensor, const torch::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.attr("arithmetic_id")="uf-int16-q9-w13-acc64-rna-sat-v1";
    m.attr("kernel_revision")="packed-v9";
    m.attr("ffn_transposed")=true;
    m.attr("block_channels_last")=true;
    m.attr("supports_yuv420_unshuffle")=true;
    m.def("conv2d", &uf_conv);
    m.def("conv2d_generic", &uf_conv_generic);
    m.def("conv2d_baseline", &uf_conv_baseline);
    m.def("add", &uf_add);
    m.def("multiply", &uf_mul);
    m.def("lookup", &uf_lut);
    m.def("from_bytes", &uf_bytes);
    m.def("from_bytes_unshuffle", &uf_bytes_unshuffle);
    m.def("wsilu4", &uf_wsilu4);
    m.def("raw_frame", &uf_raw_frame);
    m.def("raw_frames", &uf_raw_frames);
    m.def("symbols", &uf_symbols);
    m.def("from_symbols", &uf_from_symbols);
    m.def("reciprocal", &uf_reciprocal);
    m.def("depth_conv_block", &uf_depth_block,pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("depth_conv_block_packed", &uf_depth_block_packed,pybind11::call_guard<pybind11::gil_scoped_release>());
    m.def("conv2d_tiled", &uf_conv_tiled);
    m.def("conv2d_wsilu4", &uf_conv_wsilu4,pybind11::arg("x"),pybind11::arg("weight"),
          pybind11::arg("bias"),pybind11::arg("lut"),pybind11::arg("tile")=0);
    m.def("conv2d_post", &uf_conv_post,pybind11::arg("x"),pybind11::arg("weight"),
          pybind11::arg("bias"),pybind11::arg("lut"),pybind11::arg("residual"),
          pybind11::arg("shortcut"),pybind11::arg("tile")=0);
    m.def("prior_encode", &uf_prior_encode);
    m.def("prior_indexes", &uf_prior_indexes);
    m.def("prior_restore", &uf_prior_restore);
}

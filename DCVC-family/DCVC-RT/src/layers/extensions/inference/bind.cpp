// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "def.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("process_with_mask_cuda", &process_with_mask_cuda);
    m.def("combine_for_reading_2x_cuda", &combine_for_reading_2x_cuda);
    m.def("restore_y_2x_cuda", &restore_y_2x_cuda);
    m.def("restore_y_4x_cuda", &restore_y_4x_cuda);
    m.def("build_index_dec_cuda", &build_index_dec_cuda);
    m.def("build_index_enc_cuda", &build_index_enc_cuda);
    m.def("bias_quant_cuda", &bias_quant_cuda);
    m.def("round_and_to_int8_cuda", &round_and_to_int8_cuda);
    m.def("clamp_reciprocal_with_quant_cuda", &clamp_reciprocal_with_quant_cuda);
    m.def("add_and_multiply_cuda", &add_and_multiply_cuda);
    m.def("conv2d_int16_cuda", &conv2d_int16_cuda);
    m.def("conv2d_int16_residual_cuda", &conv2d_int16_residual_cuda,
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("residual"), py::arg("tile_n") = 32);
    m.def("round_and_to_int8_int16_cuda", &round_and_to_int8_int16_cuda);
    m.def("add_bias_int16_cuda", &add_bias_int16_cuda);
    m.def("add_tensors_int16_cuda", &add_tensors_int16_cuda);
    m.def("mul_feature_scale_int16_cuda", &mul_feature_scale_int16_cuda);
    m.def("reciprocal_scale_int16_cuda", &reciprocal_scale_int16_cuda);
    m.def("apply_lut_int16_cuda", &apply_lut_int16_cuda);
    m.def("process_with_mask_int16_cuda", &process_with_mask_int16_cuda);
    m.def("combine_for_reading_int16_cuda", &combine_for_reading_int16_cuda);
    m.def("restore_y_parts_int16_cuda", &restore_y_parts_int16_cuda);
    m.def("build_index_dec_int16_cuda", &build_index_dec_int16_cuda);
    m.def("build_index_enc_int16_cuda", &build_index_enc_int16_cuda);
    m.def("add_and_multiply_int16_cuda", &add_and_multiply_int16_cuda);
    m.def("bias_quant_int16_cuda", &bias_quant_int16_cuda);
    m.def("wsilu_chunk_add_int16_cuda", &wsilu_chunk_add_int16_cuda);
    m.def("bias_wsilu_depthwise_conv2d_int16_cuda", &bias_wsilu_depthwise_conv2d_int16_cuda);
    m.def("bias_pixel_shuffle_2_int16_cuda", &bias_pixel_shuffle_2_int16_cuda);
    m.def("bias_pixel_shuffle_8_int16_cuda", &bias_pixel_shuffle_8_int16_cuda);
    m.def("bias_pixel_shuffle_8_cuda", &bias_pixel_shuffle_8_cuda);
    m.def("replicate_pad_cuda", &replicate_pad_cuda);
    m.def("bias_wsilu_depthwise_conv2d_cuda", &bias_wsilu_depthwise_conv2d_cuda);

    py::class_<DepthConvProxy>(m, "DepthConvProxy")
        .def(py::init<>())
        .def("set_param", &DepthConvProxy::set_param)
        .def("set_param_with_adaptor", &DepthConvProxy::set_param_with_adaptor)
        .def("forward", &DepthConvProxy::forward)
        .def("forward_with_quant_step", &DepthConvProxy::forward_with_quant_step)
        .def("forward_with_cat", &DepthConvProxy::forward_with_cat);

    py::class_<SubpelConv2xProxy>(m, "SubpelConv2xProxy")
        .def(py::init<>())
        .def("set_param", &SubpelConv2xProxy::set_param)
        .def("forward", &SubpelConv2xProxy::forward)
        .def("forward_with_cat", &SubpelConv2xProxy::forward_with_cat);
}

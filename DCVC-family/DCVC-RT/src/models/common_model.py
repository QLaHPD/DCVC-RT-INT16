# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn

from ..layers.cuda_inference import combine_for_reading_2x, combine_for_reading_4x, \
    restore_y_2x, restore_y_2x_with_cat_after, add_and_multiply, \
    replicate_pad, restore_y_4x, clamp_reciprocal_with_quant
from ..layers.int16_inference import add_tensors_int16, apply_module_int16, \
    clamp_quant_step_int16, int16_inference_enabled, mul_feature_scale_int16, \
    prior_quant_step_int16, reciprocal_scale_int16, export_int16_lut_state, \
    export_model_int16_state
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder


class CompressionModel(nn.Module):
    def __init__(self, z_channel, extra_qp=0):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(64 + extra_qp, z_channel)
        self.gaussian_encoder = GaussianEncoder()

        self.masks = {}
        self.cuda_streams = {}

    def get_cuda_stream(self, device, idx=0, priority=0):
        key = f"{device}_{priority}_{idx}"
        if key not in self.cuda_streams:
            self.cuda_streams[key] = torch.cuda.Stream(device, priority=priority)
        return self.cuda_streams[key]

    def get_runtime_dtype(self, device):
        if int16_inference_enabled() and device.type == "cuda":
            return torch.int16
        return next(self.parameters()).dtype

    @staticmethod
    def get_qp_num():
        return 64

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    @staticmethod
    def get_downsampled_shape(height, width, p):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        return int(new_h / p + 0.5), int(new_w / p + 0.5)

    def update(self, force_zero_thres=None, prepared_state=None):
        self.entropy_coder = EntropyCoder()
        if prepared_state is not None:
            self.gaussian_encoder.load_prepared(
                self.entropy_coder,
                prepared_state["gaussian_encoder"],
                force_zero_thres=force_zero_thres,
            )
            self.bit_estimator_z.load_prepared(
                self.entropy_coder,
                prepared_state["bit_estimator_z"],
            )
            return
        self.gaussian_encoder.update(self.entropy_coder, force_zero_thres=force_zero_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    def set_use_two_entropy_coders(self, use_two_entropy_coders):
        self.entropy_coder.set_use_two_entropy_coders(use_two_entropy_coders)

    def export_int16_prep(self):
        return {
            "version": 3,
            "bit_estimator_z": self.bit_estimator_z.get_serialized_cdf_info(),
            "gaussian_encoder": self.gaussian_encoder.get_serialized_cdf_info(),
            "int16_model_state": export_model_int16_state(self),
            "int16_luts": export_int16_lut_state((
                (
                    self.gaussian_encoder.scale_min,
                    self.gaussian_encoder.scale_max,
                    self.gaussian_encoder.log_scale_min,
                    self.gaussian_encoder.log_step_recip,
                ),
            )),
        }

    def pad_for_y(self, y):
        _, _, H, W = y.size()
        padding_r, padding_b = self.get_padding_size(H, W, 4)
        y_pad = replicate_pad(y, padding_b, padding_r)
        return y_pad

    def separate_prior(self, params, is_video=False):
        if is_video:
            quant_step, scales, means = params.chunk(3, 1)
            if int16_inference_enabled() and params.dtype == torch.int16 and params.is_cuda:
                quant_step = clamp_quant_step_int16(quant_step)
                q_enc = reciprocal_scale_int16(quant_step, 0.5)[1]
                q_dec = quant_step
                return q_enc, q_dec, scales, means
            quant_step = torch.clamp_min(quant_step, 0.5)
            q_enc = 1. / quant_step
            q_dec = quant_step
        else:
            q = params[:, :2, :, :]
            if int16_inference_enabled() and params.dtype == torch.int16 and params.is_cuda:
                q_enc, q_dec = prior_quant_step_int16(q).chunk(2, 1)
                scales, means = params[:, 2:, :, :].chunk(2, 1)
                return q_enc, q_dec, scales, means
            q_enc, q_dec = (torch.sigmoid(q) * 1.5 + 0.5).chunk(2, 1)
            scales, means = params[:, 2:, :, :].chunk(2, 1)
        return q_enc, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_encoding(params, y):
        q_dec, scales, means = params.chunk(3, 1)
        q_dec, y = clamp_reciprocal_with_quant(q_dec, y, 0.5)
        return y, q_dec, scales, means

    @staticmethod
    def separate_prior_for_video_decoding(params):
        quant_step, scales, means = params.chunk(3, 1)
        if int16_inference_enabled() and params.dtype == torch.int16 and params.is_cuda:
            quant_step = clamp_quant_step_int16(quant_step)
            return quant_step, scales, means
        quant_step = torch.clamp_min(quant_step, 0.5)
        return quant_step, scales, means

    def process_with_mask(self, y, scales, means, mask):
        return self.gaussian_encoder.process_with_mask(y, scales, means, mask)

    @staticmethod
    def apply_module(module, x):
        if int16_inference_enabled() and x.is_cuda and x.dtype == torch.int16:
            return apply_module_int16(module, x)
        return module(x)

    @staticmethod
    def get_one_mask(micro_mask, height, width, dtype, device):
        mask = torch.tensor(micro_mask, dtype=dtype, device=device)
        mask = mask.repeat((height + 1) // 2, (width + 1) // 2)
        mask = mask[:height, :width]
        mask = torch.unsqueeze(mask, 0)
        mask = torch.unsqueeze(mask, 0)
        return mask

    def get_mask_4x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_4x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 4 == 0
                m = torch.ones((batch, channel // 4, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 0)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (0, 0)), height, width, dtype, device)
                m2 = self.get_one_mask(((0, 0), (1, 0)), height, width, dtype, device)
                m3 = self.get_one_mask(((0, 0), (0, 1)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
                mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
                mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
                mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_mask_2x(self, batch, channel, height, width, dtype, device):
        curr_mask_str = f"{batch}_{channel}_{width}_{height}_2x"
        with torch.no_grad():
            if curr_mask_str not in self.masks:
                assert channel % 2 == 0
                m = torch.ones((batch, channel // 2, height, width), dtype=dtype, device=device)
                m0 = self.get_one_mask(((1, 0), (0, 1)), height, width, dtype, device)
                m1 = self.get_one_mask(((0, 1), (1, 0)), height, width, dtype, device)

                mask_0 = torch.cat((m * m0, m * m1), dim=1)
                mask_1 = torch.cat((m * m1, m * m0), dim=1)

                self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @staticmethod
    def single_part_for_writing_4x(x):
        x0, x1, x2, x3 = x.chunk(4, 1)
        return (x0 + x1) + (x2 + x3)

    @staticmethod
    def single_part_for_writing_2x(x):
        x0, x1 = x.chunk(2, 1)
        return x0 + x1

    def compress_prior_2x(self, y, common_params, y_spatial_prior):
        y, q_dec, scales, means = self.separate_prior_for_video_encoding(common_params, y)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        cat_params = torch.cat((y_hat_0, common_params), dim=1)
        scales, means = self.apply_module(y_spatial_prior, cat_params).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = add_and_multiply(y_hat_0, y_hat_1, q_dec)

        y_q_w_0 = self.single_part_for_writing_2x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_2x(y_q_1)
        s_w_0 = self.single_part_for_writing_2x(s_hat_0)
        s_w_1 = self.single_part_for_writing_2x(s_hat_1)
        return y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat

    def decompress_prior_2x(self, common_params, y_spatial_prior):
        infos = self.decompress_prior_2x_part1(common_params)
        y_hat = self.decompress_prior_2x_part2(common_params, y_spatial_prior, infos)
        return y_hat

    def decompress_prior_2x_part1(self, common_params):
        q_dec, scales, means = self.separate_prior_for_video_decoding(common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, dtype, device)

        scales_r = combine_for_reading_2x(scales, mask_0, inplace=False)
        indexes, skip_cond = self.gaussian_encoder.build_indexes_decoder(scales_r)
        self.gaussian_encoder.decode_y(indexes)
        infos = {
            "q_dec": q_dec,
            "mask_0": mask_0,
            "mask_1": mask_1,
            "means": means,
            "scales_r": scales_r,
            "skip_cond": skip_cond,
            "indexes": indexes,
        }
        return infos

    def decompress_prior_2x_part2(self, common_params, y_spatial_prior, infos):
        dtype = common_params.dtype
        device = common_params.device
        y_q_r = self.gaussian_encoder.get_y(infos["scales_r"].shape,
                                            infos["scales_r"].numel(),
                                            dtype, device,
                                            infos["skip_cond"], infos["indexes"])
        y_hat_0, cat_params = restore_y_2x_with_cat_after(y_q_r, infos["means"], infos["mask_0"],
                                                          common_params)
        scales, means = self.apply_module(y_spatial_prior, cat_params).chunk(2, 1)
        scales_r = combine_for_reading_2x(scales, infos["mask_1"], inplace=True)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_1 = restore_y_2x(y_q_r, means, infos["mask_1"])

        y_hat = add_and_multiply(y_hat_0, y_hat_1, infos["q_dec"])
        return y_hat

    def compress_prior_4x(self, y, common_params, y_spatial_prior_reduction,
                          y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                          y_spatial_prior_adaptor_3, y_spatial_prior):
        '''
        y_0 means split in channel, the 0/4 quater
        y_1 means split in channel, the 1/4 quater
        y_2 means split in channel, the 2/4 quater
        y_3 means split in channel, the 3/4 quater
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        y_?_2, means multiply with mask_2
        y_?_3, means multiply with mask_3
        '''
        q_enc, q_dec, scales, means = self.separate_prior(common_params, False)
        common_params = self.apply_module(y_spatial_prior_reduction, common_params)
        dtype = y.dtype
        device = y.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        if int16_inference_enabled() and y.dtype == torch.int16 and y.is_cuda:
            y = mul_feature_scale_int16(y, q_enc)
        else:
            y = y * q_enc

        _, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.apply_module(
            y_spatial_prior, self.apply_module(y_spatial_prior_adaptor_1, params)).chunk(2, 1)
        _, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat_so_far = add_tensors_int16(y_hat_so_far, y_hat_1)
        else:
            y_hat_so_far = y_hat_so_far + y_hat_1
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.apply_module(
            y_spatial_prior, self.apply_module(y_spatial_prior_adaptor_2, params)).chunk(2, 1)
        _, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat_so_far = add_tensors_int16(y_hat_so_far, y_hat_2)
        else:
            y_hat_so_far = y_hat_so_far + y_hat_2
        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.apply_module(
            y_spatial_prior, self.apply_module(y_spatial_prior_adaptor_3, params)).chunk(2, 1)
        _, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat = add_tensors_int16(y_hat_so_far, y_hat_3)
            y_hat = mul_feature_scale_int16(y_hat, q_dec)
        else:
            y_hat = y_hat_so_far + y_hat_3
            y_hat = y_hat * q_dec

        y_q_w_0 = self.single_part_for_writing_4x(y_q_0)
        y_q_w_1 = self.single_part_for_writing_4x(y_q_1)
        y_q_w_2 = self.single_part_for_writing_4x(y_q_2)
        y_q_w_3 = self.single_part_for_writing_4x(y_q_3)
        s_w_0 = self.single_part_for_writing_4x(s_hat_0)
        s_w_1 = self.single_part_for_writing_4x(s_hat_1)
        s_w_2 = self.single_part_for_writing_4x(s_hat_2)
        s_w_3 = self.single_part_for_writing_4x(s_hat_3)
        return y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat

    def decompress_prior_4x(self, common_params, y_spatial_prior_reduction,
                            y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                            y_spatial_prior_adaptor_3, y_spatial_prior):
        _, quant_step, scales, means = self.separate_prior(common_params, False)
        common_params = self.apply_module(y_spatial_prior_reduction, common_params)
        dtype = means.dtype
        device = means.device
        B, C, H, W = means.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, dtype, device)

        if int16_inference_enabled() and scales.dtype == torch.int16 and scales.is_cuda:
            scales_r = combine_for_reading_4x(scales, mask_0)
        else:
            scales_r = self.single_part_for_writing_4x(scales * mask_0)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_0)
        y_hat_so_far = y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.apply_module(
            y_spatial_prior, self.apply_module(y_spatial_prior_adaptor_1, params)).chunk(2, 1)
        if int16_inference_enabled() and scales.dtype == torch.int16 and scales.is_cuda:
            scales_r = combine_for_reading_4x(scales, mask_1)
        else:
            scales_r = self.single_part_for_writing_4x(scales * mask_1)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_1)
        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat_so_far = add_tensors_int16(y_hat_so_far, y_hat_curr_step)
        else:
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.apply_module(
            y_spatial_prior, self.apply_module(y_spatial_prior_adaptor_2, params)).chunk(2, 1)
        if int16_inference_enabled() and scales.dtype == torch.int16 and scales.is_cuda:
            scales_r = combine_for_reading_4x(scales, mask_2)
        else:
            scales_r = self.single_part_for_writing_4x(scales * mask_2)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_2)
        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat_so_far = add_tensors_int16(y_hat_so_far, y_hat_curr_step)
        else:
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        params = torch.cat((y_hat_so_far, common_params), dim=1)
        scales, means = self.apply_module(
            y_spatial_prior, self.apply_module(y_spatial_prior_adaptor_3, params)).chunk(2, 1)
        if int16_inference_enabled() and scales.dtype == torch.int16 and scales.is_cuda:
            scales_r = combine_for_reading_4x(scales, mask_3)
        else:
            scales_r = self.single_part_for_writing_4x(scales * mask_3)
        y_q_r = self.gaussian_encoder.decode_and_get_y(scales_r, dtype, device)
        y_hat_curr_step = restore_y_4x(y_q_r, means, mask_3)
        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat_so_far = add_tensors_int16(y_hat_so_far, y_hat_curr_step)
        else:
            y_hat_so_far = y_hat_so_far + y_hat_curr_step

        if int16_inference_enabled() and y_hat_so_far.dtype == torch.int16 and y_hat_so_far.is_cuda:
            y_hat = mul_feature_scale_int16(y_hat_so_far, quant_step)
        else:
            y_hat = y_hat_so_far * quant_step

        return y_hat

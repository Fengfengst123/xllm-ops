#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Old-chain ATK executor: Add(gamma,1) + AddRmsNorm
# Receives the SAME case JSON as GammaAddRmsNorm (same inputs/golden/threshold),
# but on NPU side executes: gamma_plus_one = gamma + 1.0; npu_add_rms_norm(x1, x2, gamma_plus_one, eps)
# CPU golden is identical to the GammaAddRmsNorm executor (same torch computation).

import torch
import torch_npu
import os
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("old_chain_add_rms_norm_accuracy")
class OldChainAddRmsNormAccuracyApi(BaseApi):
    """Old chain: Add(gamma,1) + AddRmsNorm, using same golden/threshold as GammaAddRmsNorm."""

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["epsilon"] = 0.000001

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        input_x1 = input_data.kwargs["x1"]
        input_x2 = input_data.kwargs["x2"]
        input_gamma = input_data.kwargs["gamma"]
        epsilon = input_data.kwargs["epsilon"]
        addGammaOffset = input_data.kwargs["addGammaOffset"]
        kernelType = input_data.kwargs["kernelType"]

        if self.device == "cpu":
            # CPU golden: identical to GammaAddRmsNorm executor
            return self._cpu_golden(input_x1, input_x2, input_gamma, kernelType, addGammaOffset, epsilon)
        else:
            # NPU: old chain = Add(gamma,1) + AddRmsNorm
            if addGammaOffset:
                gamma_npu = input_gamma + 1.0
            else:
                gamma_npu = input_gamma
            y_out, rstd_out, x_out = torch_npu.npu_add_rms_norm(
                input_x1, input_x2, gamma_npu, float(epsilon))
            return y_out, rstd_out, x_out

    def _cpu_golden(self, input_x1, input_x2, input_gamma, kernelType, addGammaOffset, epsilon=0.000001):
        """Identical CPU golden as GammaAddRmsNorm executor."""
        ori_x_shape = input_x1.shape
        ori_gamma_shape = input_gamma.shape
        xlength = len(ori_x_shape)
        gammaLength = len(ori_gamma_shape)
        torchType32 = torch.float32
        rstdShape = []

        for i in range(xlength):
            if i < (xlength - gammaLength):
                rstdShape.append(ori_x_shape[i])
            else:
                rstdShape.append(1)
        n = xlength - gammaLength
        import numpy as np
        input_gamma = input_gamma.reshape(np.multiply.reduce(np.array(ori_gamma_shape)).item())
        x1_shape = list(ori_x_shape[0:n]) + list(input_gamma.shape)
        input_x1 = input_x1.reshape(x1_shape)
        input_x2 = input_x2.reshape(x1_shape)

        if kernelType == 1:
            oriType = torch.float16
            input_gamma = input_gamma.to(oriType)
            if addGammaOffset:
                input_gamma = (input_gamma + 1).to(oriType)
            xOut = (input_x1.to(oriType) + input_x2.to(oriType))
        elif kernelType == 2:
            oriType = torch.bfloat16
            input_gamma = input_gamma.to(oriType)
            if addGammaOffset:
                input_gamma = (input_gamma + 1).to(oriType)
            x_fp32 = (input_x1.to(torchType32) + input_x2.to(torchType32))
            xOut = x_fp32.to(oriType)
        else:
            oriType = torch.float32
            input_gamma = input_gamma.to(oriType)
            if addGammaOffset:
                input_gamma = input_gamma + 1
            xOut = (input_x1.to(torchType32) + input_x2.to(torchType32))
        x_fp32 = xOut.to(torchType32)
        variance = torch.mean(torch.pow(x_fp32, 2), axis=-1, keepdims=True)
        std = torch.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd
        if kernelType == 1:
            result_mid_ori = result_mid.to(oriType)
            y_array = result_mid_ori * input_gamma.to(oriType)
        elif kernelType == 2:
            result_mid_ori = result_mid.to(oriType)
            y_array = result_mid_ori * input_gamma.to(oriType)
        else:
            y_array = result_mid.to(torchType32) * input_gamma.to(torchType32)
        rstdOut = rstd.reshape(rstdShape).to(torchType32)
        yOut = y_array.reshape(ori_x_shape).to(oriType)
        xOut = x_fp32.reshape(ori_x_shape).to(oriType)
        return yOut, rstdOut, xOut

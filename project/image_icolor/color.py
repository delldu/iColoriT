# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 04日 星期日 16:41:59 CST
# ***
# ************************************************************************************/
#
import pdb
import torch
import torch.nn as nn
from . import data


class Generator(nn.Module):
    """
    # L + ab + mask
    input_nc = 1
    output_nc = 2
    num_in = input_nc + output_nc + 1
    norm_layer = color.get_norm_layer(norm_type="batch")
    """

    def forward(self, rgba):
        # input = torch.cat((input_A, input_B, mask_B), dim=1)
        input = data.rgba2lab(rgba)
        lab_l = input[:, 0:1, :, :]
        # lab_m = input[:, 3:4, :, :]

        out_reg = self.model_out(conv10_2)

        # out_class
        output = data.Lab2rgb(lab_l, out_reg)

        return output.clamp(0.0, 1.0)

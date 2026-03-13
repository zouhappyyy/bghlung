#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 23-8-23
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

from collections import OrderedDict
from typing import Tuple
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
import torch
from nnunet.network_architecture.LetsGo_UNet import LetsGo_UNet
from nnunet.utilities.nd_softmax import softmax_helper


class PHTransTrainerV2(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
                         deterministic, fp16)

        self.max_num_epochs = 400
        self.patch_size = [80, 160, 160]
        self.deep_supervision = True



    def initialize_network(self):
        # self.network = LetsGo_UNet(self.num_input_channels, self.num_classes,
        #                            len(self.net_num_pool_op_kernel_sizes),
        #                            self.arch_list,
        #                            self.net_num_pool_op_kernel_sizes,
        #                            self.net_conv_kernel_sizes)

        from nnunet.network_architecture.phtrans import PHTrans
        self.network = PHTrans(img_size=self.patch_size,
                      base_num_features=24,
                      image_channels=self.num_input_channels,
                      num_classes=self.num_classes,
                      num_pool=len(self.net_num_pool_op_kernel_sizes),
                      pool_op_kernel_sizes=self.net_num_pool_op_kernel_sizes,
                      conv_kernel_sizes=self.net_conv_kernel_sizes,
                      deep_supervision=self.deep_supervision,
                      max_num_features=24*13,
                      depths=[2, 2, 2, 2],
                      num_heads=[3, 6, 12, 24],
                      window_size=[5, 5, 5],
                      drop_path_rate=0)
        from lib.model.layers.utils import count_param
        param = count_param(self.network)
        net_param_str = 'Net totoal parameters: %.2fM (%d)' % (param / 1e6, param)

        print(net_param_str)
        self.print_to_log_file(net_param_str)

        # input_res = (1, self.patch_size[0], self.patch_size[1], self.patch_size[2])
        # input = torch.ones(()).new_empty((1, *input_res), dtype=next(self.network.parameters()).dtype,
        #                                  device=next(self.network.parameters()).device)
        #
        # from fvcore.nn import FlopCountAnalysis
        # flops = FlopCountAnalysis(self.network, input)
        # model_flops = flops.total()
        # self.print_to_log_file(f"MAdds: {round(model_flops * 1e-9, 2)} G")

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper


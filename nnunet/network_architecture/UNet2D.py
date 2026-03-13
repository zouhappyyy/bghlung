#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# @Project: BrainTissueISEG2017
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2022/4/18
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
from __future__ import print_function, division
import torch
import torch.nn as nn
import numpy as np

class ConvBlockND(nn.Module):
    """for 2D and 3D convolutional blocks"""
    def __init__(self, in_channels, out_channels, 
                dim = 2, dropout_p = 0.0):
        """
        dim: should be 2 or 3
        dropout_p: probability to be zeroed
        """
        super(ConvBlockND, self).__init__()
        assert(dim == 2 or dim == 3)
        self.dim = dim 
        if(self.dim == 2):
            self.conv_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.conv_conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm3d(out_channels),
                nn.LeakyReLU()
            )

    def forward(self, x):
        output = self.conv_conv(x)
        return output 

class DownBlock(nn.Module):
    """a convolutional block followed by downsampling"""
    def __init__(self,in_channels, out_channels, 
                dim = 2, dropout_p = 0.0, downsample = True):
        super(DownBlock, self).__init__()
        self.downsample = downsample 
        self.dim = dim
        self.conv = ConvBlockND(in_channels, out_channels, dim, dropout_p)
        if(downsample):
            if(self.dim == 2):
                self.down_layer = nn.MaxPool2d(kernel_size = 2, stride = 2)
            else:
                self.down_layer = nn.MaxPool3d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x_shape = list(x.shape)
        if(self.dim == 2 and len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        output = self.conv(x)
        if(self.downsample):
            output_d = self.down_layer(output)
        else:
            output_d = None 
        if(self.dim == 2 and len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
            if(self.downsample):
                new_shape = [N, D] + list(output_d.shape)[1:]
                output_d = torch.reshape(output_d, new_shape)
                output_d = torch.transpose(output_d, 1, 2)

        return output, output_d

class UpBlock(nn.Module):
    """Upsampling followed by ConConvBlockNDvBlock"""
    def __init__(self, in_channels1, in_channels2, out_channels, 
                 dim = 2, dropout_p = 0.0, bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        self.dim = dim
        if bilinear:
            if(dim == 2):
                self.up = nn.Sequential(
                    nn.Conv2d(in_channels1, in_channels2, kernel_size = 1),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            else:
                self.up = nn.Sequential(
                    nn.Conv3d(in_channels1, in_channels2, kernel_size = 1),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
        else:
            if(dim == 2):
                self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
            else:
                self.up = nn.ConvTranspose3d(in_channels1, in_channels2, kernel_size=2, stride=2)
            
        self.conv = ConvBlockND(in_channels2 * 2, out_channels, dim, dropout_p)

    def forward(self, x1, x2):
        x1_shape = list(x1.shape)
        x2_shape = list(x2.shape)
        if(self.dim == 2 and len(x1_shape) == 5):
            [N, C, D, H, W] = x1_shape
            new_shape = [N*D, C, H, W]
            x1 = torch.transpose(x1, 1, 2)
            x1 = torch.reshape(x1, new_shape)
            [N, C, D, H, W] = x2_shape
            new_shape = [N*D, C, H, W]
            x2 = torch.transpose(x2, 1, 2)
            x2 = torch.reshape(x2, new_shape)

        x1 = self.up(x1)
        output = torch.cat([x2, x1], dim=1)
        output = self.conv(output)
        if(self.dim == 2 and len(x1_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output  

class UNet2D5(nn.Module):
    def __init__(self, params):
        super(UNet2D5, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dims      = self.params['conv_dims']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        self.dropout   = self.params['dropout']
        assert(len(self.ft_chns) == 5)

        self.block0 = DownBlock(self.in_chns, self.ft_chns[0], self.dims[0], self.dropout[0], True)
        self.block1 = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dims[1], self.dropout[1], True)
        self.block2 = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dims[2], self.dropout[2], True)
        self.block3 = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dims[3], self.dropout[3], True)
        self.block4 = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dims[4], self.dropout[4], False)

        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], 
                    self.dims[3], dropout_p = 0.0, bilinear = self.bilinear) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], 
                    self.dims[2], dropout_p = 0.0, bilinear = self.bilinear) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], 
                    self.dims[1], dropout_p = 0.0, bilinear = self.bilinear) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], 
                    self.dims[0], dropout_p = 0.0, bilinear = self.bilinear) 
    
        self.out_conv = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))

    def forward(self, x):
        x0, x0_d = self.block0(x)
        x1, x1_d = self.block1(x0_d)
        x2, x2_d = self.block2(x1_d)
        x3, x3_d = self.block3(x2_d)
        x4, x4_d = self.block4(x3_d)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)
        return output

from lib.model.layers.utils import count_param
#  x  = np.random.rand(4, 4, 32, 96, 96)

if __name__ == "__main__":
    n_modal = 1
    n_classes = 3
    # origin size: (128, 256, 256)
    base_filters = 32
    net = UNet2D5(params={'in_chns':n_modal,
              # channels=(16, 32, 48, 64, 80, 96),
              'feature_chns':[base_filters, base_filters*2, base_filters*4, base_filters*8, base_filters*16],
              'conv_dims': [2, 2, 2, 2, 2],
              'dropout':  [0, 0, 0, 0, 0],
              'class_num': n_classes,
              'bilinear': False})

    param = count_param(net)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))
    ## "brats2020":  {'bg':0, "WT": 1,  'TC': 1, 'ET': 1}
    net.eval()
    with torch.no_grad():
        input_tensor = torch.rand(1, n_modal, 16, 80, 80)

        # Param and FLOPs
        from thop import profile

        flops, params = profile(net, inputs=(input_tensor,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

        seg_output = net(input_tensor)
        print(seg_output.shape)

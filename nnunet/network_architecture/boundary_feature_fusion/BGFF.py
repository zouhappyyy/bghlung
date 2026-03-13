#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: S1_NNBALungTumor3DSeg
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2023/11/10
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv3d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding

#
class ContextBlock3D(nn.Module):

    def __init__(self,
                 inplanes,
                 ratio=8,
                 pooling_type='att',
                 fusion_types=('channel_mul',)):

        super(ContextBlock3D, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes // ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv3d(inplanes, 1, kernel_size=(1,1,1))
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=(1,1,1)),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=(1,1,1)))
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv3d(self.inplanes, self.planes, kernel_size=(1,1,1)),
                nn.LayerNorm([self.planes, 1, 1, 1]),
                nn.ReLU(inplace=True),  # yapf: disable
                nn.Conv3d(self.planes, self.inplanes, kernel_size=(1,1,1)))
        else:
            self.channel_mul_conv = None


    def spatial_pool(self, x):
        batch, channel, depth, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x
            # [N, C, D *  H * W]
            input_x = input_x.view(batch, channel, depth * height * width)
            # [N, 1, C, D * H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, D, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, D * H * W]
            context_mask = context_mask.view(batch, 1, depth * height * width)
            # [N, 1, D * H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1,D * H * W, 1]
            context_mask = context_mask.unsqueeze(-1)
            # print('context_mask',context_mask.shape)
            # [N, 1, C, 1, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1, 1]
            context = context.view(batch, channel, 1, 1, 1)
        else:
            # [N, C, 1, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        context = self.channel_mul_conv(context)

        return context

class AFF3D(nn.Module):
    '''
    Anisotropic Context-aware Feature Fusion Module
    '''

    def __init__(self, channels, r=4):
        super(AFF3D, self).__init__()
        self.global_context = ContextBlock3D(inplanes=channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual

        wei = self.sigmoid(self.global_context(xa))

        xo = x * wei +  residual * (1 - wei)
        return xo



class BGFF_Module(Module):
    """ Boundary-Guided Feature Fusion Module using Position Attention"""

    def __init__(self, in_dim):
        super(BGFF_Module, self).__init__()

        self.global_context = ContextBlock3D(inplanes=in_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(self, seg_tensor, bdr_tensor):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : recalibrated
                attention: B X (DxHxW) X (DxHxW)
        """


        fusion_tensor = seg_tensor + bdr_tensor

        weight = self.sigmoid(self.global_context(fusion_tensor))

        out = torch.cat((seg_tensor*weight, bdr_tensor*(1-weight)), dim=1)

        return out

# class BGFF_Module(Module):
#     """ Boundary-Guided Feature Fusion Module using position-aware attention"""
#
#     def __init__(self, in_dim):
#         super(BGFF_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.shared_conv = nn.Conv3d(2*in_dim, in_dim, kernel_size=1)
#         self.seg_conv = nn.Conv3d(in_dim, 1, kernel_size=1)
#         self.bdr_conv = nn.Conv3d(in_dim, 1, kernel_size=1)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, seg_tensor, bdr_tensor):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : recalibrated features
#                 attention: B X (DxHxW) X (DxHxW)
#         """
#
#
#         fusion_tensor = torch.cat((seg_tensor, bdr_tensor), dim=1)
#
#         # position-aware attention
#         position_att = self.shared_conv(fusion_tensor)
#         seg_weight = self.sigmoid(self.seg_conv(position_att))
#         bdr_weight = self.sigmoid(self.bdr_conv(position_att))
#
#         out = torch.cat((seg_tensor*seg_weight, bdr_tensor*bdr_weight), dim=1)
#
#         return out

# class BGFF_BGPA_Module(Module):
#     """ Boundary-Guided Feature Fusion Module using position-aware attention"""
#
#     def __init__(self, in_dim):
#         super(BGFF_BGPA_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.position_conv = nn.Conv3d(in_dim, 1, kernel_size=1)
#         self.seg_conv = nn.Conv3d(3, 1, kernel_size=5, stride = 1, padding=5//2)  # depthwise conv
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, seg_tensor, bdr_tensor):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : recalibrated features
#                 attention: B X (DxHxW) X (DxHxW)
#         """
#
#
#         # position-aware attention
#
#         position_att_1 = self.position_conv(bdr_tensor)
#         position_att_2 = torch.mean(bdr_tensor, dim=1, keepdim=True)
#         position_att_3, _ = torch.max(bdr_tensor, dim=1, keepdim=True)
#         position_att = torch.cat([position_att_1, position_att_2, position_att_3], dim=1)
#         seg_weight = self.sigmoid(self.seg_conv(position_att))
#
#         out = torch.cat((seg_tensor*seg_weight + seg_tensor, bdr_tensor), dim=1)
#         # out = torch.cat((seg_tensor, bdr_tensor+seg_tensor * seg_weight), dim=1)
#
#         return out
#
# class BGFF_Module(Module):
#     """ Boundary-Guided Feature Fusion Module using position-aware attention"""
#
#     def __init__(self, in_dim):
#         super(BGFF_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.position_conv = nn.Conv3d(in_dim, 1, kernel_size=1)
#         self.seg_conv = nn.Conv3d(2, 1, kernel_size=3, stride = 1, padding=3//2)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, seg_tensor, bdr_tensor):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : recalibrated features
#                 attention: B X (DxHxW) X (DxHxW)
#         """
#
#
#         # position-aware attention
#
#         # position_att_1 = torch.mean(bdr_tensor, dim=1, keepdim=True)
#         # position_att_2, _ = torch.max(bdr_tensor, dim=1, keepdim=True)
#         # position_att = torch.cat([position_att_1, position_att_2], dim=1)
#         # seg_weight = self.sigmoid(self.seg_conv(position_att))
#
#         # out = torch.cat((seg_tensor*seg_weight + seg_tensor, bdr_tensor), dim=1)
#         out = torch.cat((seg_tensor, bdr_tensor), dim=1)
#         return out

# class PAM_Module(Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = Parameter(torch.zeros(1))
#
#         self.softmax = Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out

# class BGFF_Module(Module):
#     """ Boundary-Guided Feature Fusion Module using position-aware attention"""
#
#     def __init__(self, in_dim):
#         super(BGFF_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=1, kernel_size=1)
#         self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=1, kernel_size=1)
#         self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
#         self.softmax = Softmax(dim=-1)
#
#     def forward(self, seg_tensor, bdr_tensor):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : recalibrated features
#                 attention: B X (DxHxW) X (DxHxW)
#         """
#
#
#         fusion_tensor = seg_tensor + bdr_tensor
#
#         # position-aware attention
#         m_batchsize, C, depth, height, width = fusion_tensor.size()
#         proj_query = self.query_conv(fusion_tensor).view(m_batchsize, -1, width * height*depth).permute(0, 2, 1)
#         proj_key = self.key_conv(fusion_tensor).view(m_batchsize, -1, width * height*depth)
#         print(proj_key.shape,proj_query.shape)
#         energy = torch.bmm(proj_query, proj_key)
#         attention = self.softmax(energy)
#         proj_value = self.value_conv(seg_tensor).view(m_batchsize, -1, width * height*depth)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, -1, depth, height, width)
#         print(out.shape)
#         assert False
#
#         out = torch.cat((seg_tensor+out, bdr_tensor), dim=1)
#
#         return out

if __name__ == "__main__":
    dim = 48
    input_bdr_tensor = torch.rand(2, dim, 80, 80, 80)
    input_seg_tensor = torch.rand(2, dim, 80, 80, 80)

    bgff_module = BGFF_Module(in_dim=dim)

    from lib.model.layers.utils import count_param
    param = count_param(bgff_module)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    print(bgff_module(seg_tensor=input_seg_tensor, bdr_tensor=input_bdr_tensor).shape)


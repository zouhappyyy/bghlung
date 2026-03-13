#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# @Project: S1_NNBALungTumor3DSeg
# @IDE: PyCharm
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 2023/9/23
# @Desc: 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from functools import reduce, lru_cache
from operator import mul

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def img2windows(img, H_sp, W_sp):
    """
    img: B C D H W
    """
    B, C, D, H, W = img.shape
    img_reshape = img.view(B, C, D, H // H_sp, H_sp, W // W_sp, W_sp)
    # B, C, D, H // H_sp 3, H_sp, W // W_sp, W_sp ->  B, D, H // H_sp, W // W_sp, H_sp, W_sp, c ->
    img_perm = img_reshape.permute(0, 2, 3, 5, 4, 6, 1).contiguous().reshape(-1, D * H_sp * W_sp, C)

    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, D, H, W):
    """
    img_splits_hw: B' D H W C
    """
    # print('img_splits_hw',img_splits_hw.shape)
    B = int(img_splits_hw.shape[0] / (D * H * W / H_sp / W_sp))
    # print('B',B)
    #
    img = img_splits_hw.view(B, D, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, D, H, W, -1)

    return img



class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        # self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)



    def im2cswin(self, x):
        B, N, C = x.shape

        # print('self.H_sp, self.W_sp',  self.H_sp, self.W_sp)
        D = H = W = self.resolution

        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)

        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape

        D = H = W = self.resolution

        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, D, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 3, 5, 1, 4, 6).contiguous().reshape(-1, C, H_sp, W_sp)
        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print('q, k, v',q.shape, k.shape, v.shape)

        ### Img2Window
        D = H = W = self.resolution
        B, L, C = q.shape

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        # print('v', v.shape)
        # print('attn', attn.shape)

        x = (attn @ v) + lepe

        # print('x', x.shape)
        # print('self.H_sp * self.W_sp', self.H_sp, self.W_sp, self.H_sp * self.W_sp)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, D, H, W).view(B, -1, C)  # B H' W' C

        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class CSWinBlock3D(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        # print('self.patches_resolution, split_size', self.patches_resolution, split_size)
        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 2,
                    resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        #
        H = W = self.patches_resolution
        # print('input', x.shape)
        x = rearrange(x, 'b c d h w -> b d h w c')
        # print('output', x.shape)
        B, D, H, W, C = x.shape

        img = self.norm1(x)
        qkv = self.qkv(img)
        # print('qkv.shape', qkv.shape)
        qkv = qkv.reshape(B, -1, 3, C)
        # print('qkv.shape', qkv.shape)
        qkv = qkv.permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x.reshape(B, -1, C) + self.drop_path(attened_x) # HERE
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.reshape(B, D, H, W, C)

        # rearrange
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


import os
if __name__ == "__main__":
    # split_size = [1, 2, 7, 7]
    split_size = [5, 5, 5, 5]

    # num_stage = 2
    input_resolution = (40,40,40)
    num_heads = 3
    dim = 96

    # # num_stage = 3
    # input_resolution = (20, 20, 20)
    # num_heads = 6
    # dim = 192

    # # num_stage = 4
    # input_resolution = (10, 10, 10)
    # num_heads = 12
    # dim = 312

    mlp_ratio = 4
    qkv_bias = True
    qk_scale = None
    drop =  0.0
    attn_drop =  0.0

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    cswin_block_1 = CSWinBlock3D(
                dim=dim,
                num_heads=num_heads,
                reso=input_resolution[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_size[0],
                drop=drop,
                attn_drop=attn_drop,
               )

    cswin_block_2 = CSWinBlock3D(
                dim=dim,
                num_heads=num_heads,
                reso=input_resolution[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                split_size=split_size[0],
                drop=drop,
                attn_drop=attn_drop,
                )

    input_tensor = torch.rand(1, dim, input_resolution[0], input_resolution[1], input_resolution[2])

    # flop_analyzer = FlopCountAnalysis(cswin_block_1, input_tensor)
    # print('>>'*20, 'swin_block_1', '>>'*20)
    # print(flop_count_table(flop_analyzer))
    #
    # flop_analyzer = FlopCountAnalysis(cswin_block_2, input_tensor)
    # print('>>' * 20, 'swin_block_2', '>>' * 20)
    # print(flop_count_table(flop_analyzer))

    print('input shape', input_tensor.shape)
    print('output shape',cswin_block_2(cswin_block_1(input_tensor)).shape)
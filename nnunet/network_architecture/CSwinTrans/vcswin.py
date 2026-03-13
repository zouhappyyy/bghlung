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


def img2windows(img, D_sp, H_sp, W_sp):
    """
    img: B C D H W
    """
    B, C, D, H, W = img.shape

    # print('B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp', B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)

    # B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp
    img_reshape = img.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)
    # print('img_reshape.shape', img_reshape.shape)

    # D_sp = 10 or 5, H_sp = 10 or 5, W_sp = 10 or 5
    # B 0, C 1, D // D_sp 2, D_sp 3, H // H_sp 4, H_sp 5, W // W_sp 6, W_sp 7-> B 0,D // D_sp 2, H // H_sp 4, W // W_sp 6, D_sp 3, H_sp 5, W_sp 7, C 1
    #  B ,D // D_sp * H // H_sp * W // W_sp, D_sp * H_sp * W_sp , C
    img_perm = img_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, D_sp * H_sp * W_sp, C)
    # print('img_perm.shape', img_perm.shape)

    return img_perm


def windows2img(img_splits_xyz, D_sp, H_sp, W_sp, D, H, W):
    """

    img_splits_hw: B' D H W C
    """

    B = int(img_splits_xyz.shape[0] / (D * H * W / D_sp / H_sp / W_sp))
    # B * D // D_sp * H // H_sp * W // W_sp,  D_sp * H_sp * W_sp, C -> B, D//D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, C
    img = img_splits_xyz.view(B, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, -1)

    # current: B 0, D//D_sp 1, H // H_sp 2, W // W_sp 3, D_sp 4, H_sp 5, W_sp 6, C 7 -> B 0, D//D_sp 1, D_sp 4, H // H_sp 2, H_sp 5, W // W_sp 3, W_sp 6, C 7
    # ->  B , D, H, W, C
    img = img.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        """

        :param dim:
        :param resolution:
        :param idx:
        :param split_size:
        :param dim_out:
        :param num_heads: num_heads // 3
        :param attn_drop:
        :param proj_drop:
        :param qk_scale:
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads

        # dim (96, 192, 312) / (3 or 6 or 12 // 3)
        head_dim = dim // num_heads

        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # strip window
        if idx == -1:
            # b c d h w
            D_sp, H_sp, W_sp = self.resolution, self.resolution, self.resolution
        elif idx == 0:
            # b c d h w*
            D_sp, H_sp, W_sp = self.resolution, self.resolution, self.split_size
        elif idx == 1:
            # b c d h w
            D_sp, H_sp, W_sp = self.resolution, self.split_size, self.resolution
        elif idx == 2:
            # b c d h w
            D_sp, H_sp, W_sp = self.split_size, self.resolution, self.resolution
        else:
            print("ERROR MODE", idx)
            assert False

        self.D_sp = D_sp
        self.H_sp = H_sp
        self.W_sp = W_sp

        stride = 1



        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        # print('im2cswin')
        #
        D = H = W = self.resolution

        # b, d*h*w, c/3 -> b, c/3, d*h*w -> b, c/3, d, h, w
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)

        # b, c/3, d, h, w -> B, D // D_sp * H // H_sp * W // W_sp, D_sp * H_sp * W_sp , C
        x = img2windows(x, self.D_sp, self.H_sp, self.W_sp)

        # D_sp = 10 or 5, H_sp = 10 or 5, W_sp = 10 or 5
        #  B, D // D_sp * H // H_sp * W // W_sp, D_sp * H_sp * W_sp , C -> B * D // D_sp * H // H_sp * W // W_sp,  D_sp * H_sp * W_sp, num_header(3,6,12), C // self.num_heads
        x = x.reshape(-1, self.D_sp * self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)

        # B * D // D_sp * H // H_sp * W // W_sp 0,  D_sp * H_sp * W_sp 1, num_header 2, C // self.num_heads 3->
        # ->  B * D // D_sp * H // H_sp * W // W_sp, num_header, D_sp * H_sp * W_sp, C // self.num_heads
        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def get_lepe(self, x):
        # (b, d*h*w, c/3)
        B, N, C = x.shape

        #
        D = H = W = self.resolution

        # (b, d*h*w, c/3) -> B, C, D, H, W
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)

        D_sp, H_sp, W_sp = self.D_sp, self.H_sp, self.W_sp

        # B, C, D, H, W -> B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp
        x = x.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)

        #  B 0, C 1, D // D_sp 2, D_sp 3, H // H_sp 4, H_sp 5, W // W_sp 6, W_sp 7 -> B 0, D // D_sp 2, H // H_sp 4, W // W_sp 6, C 1, D_sp 3, H_sp 5, W_sp 7
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().reshape(-1, C, D_sp, H_sp, W_sp)

        # B * D // D_sp * H // H_sp * W // W_sp, C, D_sp, H_sp, W_sp
        # locally-enhanced positional encoding
        # lepe = func(x)  ### B', C, D_sp, H_sp, W_sp
        lepe = x

        #  B * D // D_sp * H // H_sp * W // W_sp, C, D_sp, H_sp, W_sp -> B * D // D_sp * H // H_sp * W // W_sp, self.num_heads, C // self.num_heads, D_sp * H_sp * W_sp
        # -> B * D // D_sp * H // H_sp * W // W_sp, self.num_heads,  D_sp * H_sp * W_sp, C // self.num_heads,
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, D_sp * H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        #  B * D // D_sp * H // H_sp * W // W_sp, C, D_sp, H_sp, W_sp -> B * D // D_sp * H // H_sp * W // W_sp, self.num_heads, C // self.num_heads, D_sp * H_sp * W_sp
        # -> B * D // D_sp * H // H_sp * W // W_sp, self.num_heads,  D_sp * H_sp * W_sp, C // self.num_heads,
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.D_sp * self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x

    def forward(self, qkv):
        """
        x:  B, L: d*h*w, C
        """

        # 3, b, d*h*w, c/3 -> (b, d*h*w, c/3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        D = H = W = self.resolution
        B, L, C = q.shape
        #
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)

        # (b, d*h*w, c/3) -> B * D // D_sp * H // H_sp * W // W_sp, num_header, D_sp * H_sp * W_sp, C // self.num_heads
        q = self.im2cswin(q)

        #  (b, d*h*w, c/3) -> B * D // D_sp * H // H_sp * W // W_sp, num_header, D_sp * H_sp * W_sp, C // self.num_heads
        k = self.im2cswin(k)
        # print('q.shape', q.shape)
        # print('k.shape', k.shape)

        # (b, d*h*w, c/3) -> B * D // D_sp * H // H_sp * W // W_sp, self.num_heads,  D_sp * H_sp * W_sp, C // self.num_heads
        v = self.get_lepe(v)
        # print('k.shape', k.shape)

        # (B * D // D_sp * H // H_sp * W // W_sp)-> B, (self.num_heads)->head, (D_sp * H_sp * W_sp)->N, (C // self.num_heads)->C
        q = q * self.scale
        # B head N C @ B head C N --> B head N N
        attn = (q @ k.transpose(-2, -1))
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        # B head N N @ B head N C + B head N C --> B head N C
        # x = (attn @ v) + lepe
        x = attn @ v

        # print('attn.shape',attn.shape)
        # print('v.shape', v.shape)
        # print('lepe.shape', lepe.shape)
        # print('x.shape', x.shape)

        # print('x', x.shape)
        # B, head, N, C ---->
        # B * D // D_sp * H // H_sp * W // W_sp,  D_sp * H_sp * W_sp, self.num_heads, C // self.num_heads  ---->
        # B * D // D_sp * H // H_sp * W // W_sp,  D_sp * H_sp * W_sp, C
        x = x.transpose(1, 2).reshape(-1, self.D_sp * self.H_sp * self.W_sp, C)

        ### Window2Img
        #  B * D // D_sp * H // H_sp * W // W_sp,  D_sp * H_sp * W_sp, C  ->  B , D, H, W, C -> B, D*H*W, C
        x = windows2img(x, self.D_sp, self.H_sp, self.W_sp, D, H, W).view(B, -1, C)  # B H' W' C

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


class VCSWin3DBlock(nn.Module):

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

        # all dim three dimensions
        assert dim % 3 == num_heads % 3 == 0

        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 3

        # self.cpe = nn.Conv3d(dim, dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=dim)
        self.cpe = nn.Identity()

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        if last_stage:
            # if img_resolution == strip width
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim // 3,
                    resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 3, dim_out=dim // 3,
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
        D = H = W = self.patches_resolution

        x = self.cpe(x) + x

        # print('input', x.shape)
        # b c d h w
        x = rearrange(x, 'b c d h w -> b d h w c')

        # b d h w c
        # print('output', x.shape)
        B, D, H, W, C = x.shape

        img = self.norm1(x)

        # b d h w c -> b d h w c*3
        qkv = self.qkv(img)

        # b, d*h*w, 3, c
        qkv = qkv.reshape(B, -1, 3, C)

        # 3, b, d*h*w, c
        qkv = qkv.permute(2, 0, 1, 3)

        if self.branch_num == 3:
            # Dynamic Stripe Window + Parallel Grouping Heads
            # 3, b, d*h*w, c/3
            pinput_1 = qkv[:, :, :, :C // 3]
            # 3, b, d*h*w, c/3
            pinput_2 = qkv[:, :, :, C // 3:C * 2 // 3]
            # 3, b, d*h*w, c/3
            pinput_3 = qkv[:, :, :, C * 2 // 3:]

            # print(pinput_1.shape, pinput_2.shape, pinput_3.shape)

            # B, D, H, W, C/3
            x1 = self.attns[0](pinput_1)
            x2 = self.attns[1](pinput_2)
            x3 = self.attns[2](pinput_3)
            # print('x1.shape', x1.shape)
            # B, D, H, W, C
            attened_x = torch.cat([x1, x2, x3], dim=2)
            # print('attened_x.shape', attened_x.shape)
        else:
            attened_x = self.attns[0](qkv)

        # FFN
        attened_x = self.proj(attened_x)
        # B, D*H*W, C
        x = x.reshape(B, -1, C) + self.drop_path(attened_x)  # HERE
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
    input_resolution = (40, 40, 40)
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
    drop = 0.0
    attn_drop = 0.0

    from fvcore.nn import FlopCountAnalysis, flop_count_table

    cswin_block_1 = VCSWin3DBlock(
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

    cswin_block_2 = VCSWin3DBlock(
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
    print('output shape', cswin_block_2(cswin_block_1(input_tensor)).shape)

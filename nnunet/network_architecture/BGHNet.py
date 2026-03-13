#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 23-8-23
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

from audioop import bias
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from nnunet.network_architecture.neural_network import SegmentationNetwork
from einops import rearrange

class ConvDropoutNormNonlin(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, conv_kwargs=None,
                 norm_op=nn.BatchNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.GELU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        self.conv = conv_op(input_channels, output_channels, **conv_kwargs)

        if dropout_op is not None and dropout_op_kwargs['p'] is not None and dropout_op_kwargs['p'] > 0:
            self.dropout = dropout_op(**dropout_op_kwargs)
        else:
            self.dropout = None

        self.instnorm = norm_op(output_channels, **norm_op_kwargs)
        self.lrelu = nonlin(**nonlin_kwargs) if nonlin_kwargs != None else nonlin()

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))



class Convnext3DBlock(nn.Module):
    """
    ConvNeXt3DBlock.
    """
    def __init__(self, in_channels:int, out_channels:int,  kernel_size=7, skip =True, exp_r=4, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride = 1, padding=kernel_size//2, groups=in_channels)  # depthwise conv
        # nn.InstanceNorm3d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.norm = nn.LayerNorm(in_channels, eps=1e-5)
        self.pwconv1 = nn.Linear(in_channels, exp_r * in_channels)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(exp_r * in_channels, out_channels)
        self.skip = skip
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)),requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, D, H, W) -> (N, D, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            # print(self.gamma.shape, x.shape)
            x = self.gamma * x

        x = x.permute(0, 4, 1, 2, 3)  # (N, D, H, W, C) -> (N, C, D, H, W)

        if self.skip:
            x = input + x
        else:
            x = x
        return x

class DownOrUpSample(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels,
                 conv_op, conv_kwargs,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, nonlinbasic_block=ConvDropoutNormNonlin):
        super(DownOrUpSample, self).__init__()
        self.blocks = nonlinbasic_block(input_feature_channels, output_feature_channels, conv_op, conv_kwargs,
                                        norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                                        nonlin, nonlin_kwargs)

    def forward(self, x):
        return self.blocks(x)

class DeepSupervision(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.proj = nn.Conv3d(
            dim, num_classes, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.proj(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, num_stage, num_only_conv_stage, num_pool, base_num_features, input_resolution, depth, num_heads,
                 window_size, image_channels=1, num_conv_per_stage=2, conv_op=None,
                 norm_op=None, norm_op_kwargs=None,
                 dropout_op=None, dropout_op_kwargs=None,
                 nonlin=None, nonlin_kwargs=None,
                 conv_kernel_sizes=None, conv_pad_sizes=None, pool_op_kernel_sizes=None,
                 basic_block=ConvDropoutNormNonlin, max_num_features=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, down_or_upsample=None, feat_map_mul_on_downscale=2,
                 num_classes=None, is_encoder=True, use_checkpoint=False, is_seg_decoder = False
                 ):

        super().__init__()
        self.num_stage = num_stage
        self.num_only_conv_stage = num_only_conv_stage
        self.num_pool = num_pool
        self.use_checkpoint = use_checkpoint
        self.is_encoder = is_encoder
        self.is_seg_decoder = is_seg_decoder

        dim = min((base_num_features * feat_map_mul_on_downscale **
                   num_stage), max_num_features)

        conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        if num_stage == 0 and is_encoder:
            input_features = image_channels

        elif not is_encoder and num_stage < num_pool:
            # decoder concat
            input_features = 2 * dim
        else:
            input_features = dim

        # self.depth = depth
        conv_kwargs['kernel_size'] = conv_kernel_sizes[num_stage]
        conv_kwargs['padding'] = conv_pad_sizes[num_stage]

        self.input_du_channels = dim
        self.output_du_channels = min(
            int(base_num_features * feat_map_mul_on_downscale ** (num_stage + 1 if is_encoder else num_stage - 1)),
            max_num_features)

        # InstanceNorm
        if not num_stage >= num_only_conv_stage:
            self.conv_blocks = nn.Sequential(
                *([basic_block(input_features, dim, conv_op,
                               conv_kwargs,
                               norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                               nonlin, nonlin_kwargs)] +
                  [basic_block(dim, dim, conv_op,
                               conv_kwargs,
                               norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
                               nonlin, nonlin_kwargs) for _ in range(num_conv_per_stage - 1)]))
        else:
            from nnunet.network_architecture.CSwinTrans.vcswin_cpe import VCSWin3DBlock
            # 5, 5
            split_size = [1, 1, 2, 5, 5]

            self.convnext_blocks = Convnext3DBlock(in_channels=input_features, out_channels=dim, kernel_size=5, exp_r=2,
                                                   layer_scale_init_value=0, skip=True)
            assert input_resolution[0] == input_resolution[1] == input_resolution[2]
            # num_stage: 2,3,4
            self.swin_blocks = nn.ModuleList([
                VCSWin3DBlock(
                    dim=dim,
                    num_heads=num_heads,
                    reso=input_resolution[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    split_size=split_size[num_stage],
                    drop=drop,
                    attn_drop=attn_drop) for i in range(depth)])

        if is_seg_decoder:
            from nnunet.network_architecture.boundary_feature_fusion.BGFF import BGFF_Module
            self.bgff_module = BGFF_Module(in_dim=dim)

        # patch merging layer
        if down_or_upsample is not None:
            dowm_stage = num_stage - 1 if not is_encoder else num_stage
            self.down_or_upsample = nn.Sequential(
                down_or_upsample(self.input_du_channels, self.output_du_channels, pool_op_kernel_sizes[dowm_stage],
                                 pool_op_kernel_sizes[dowm_stage], bias=False),
                norm_op(self.output_du_channels, **norm_op_kwargs)
                )
        else:
            self.down_or_upsample = None
        if not is_encoder:
            self.deep_supervision = DeepSupervision(dim, num_classes)
        else:
            self.deep_supervision = None

    def forward(self, x, skip):
        # if encoder, skip == None

        s = x
        if not self.is_encoder and self.num_stage < self.num_pool:
            # print('decoder, concat x and skip', x.shape, skip.shape)

            if self.is_seg_decoder:
                x = self.bgff_module(seg_tensor=x, bdr_tensor=skip)
            else:
                x = torch.cat((x, skip), dim=1)

        if self.num_stage >= self.num_only_conv_stage:
            x = self.convnext_blocks(x)
        else:
            x = self.conv_blocks(x)

        if self.num_stage >= self.num_only_conv_stage:
            for trans_idx, tblk in enumerate(self.swin_blocks):
                if self.use_checkpoint:
                    s = checkpoint.checkpoint(tblk, s)
                else:
                    s = tblk(s)
            x = x + s

        if self.down_or_upsample is not None:
            du = self.down_or_upsample(x)

        if self.deep_supervision is not None:
            ds = self.deep_supervision(x)

        if self.is_encoder:
            return x, du

        elif self.down_or_upsample is not None:
            return du, ds
        elif self.down_or_upsample is None:
            return None, ds


class BGHNet(SegmentationNetwork):
    def __init__(self,
                 img_size,
                 base_num_features,
                 num_classes,
                 image_channels=1,
                 num_only_conv_stage=2,
                 num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2,
                 pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 deep_supervision=True,
                 max_num_features=None,
                 depths=None,
                 num_heads=None,
                 window_size=None,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 dropout_p=0.0,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op = nn.Dropout3d
        dropout_op_kwargs = {'p': dropout_p, 'inplace': True}
        nonlin = nn.GELU
        nonlin_kwargs = None

        self.do_ds = deep_supervision
        self.num_pool = len(pool_op_kernel_sizes)
        conv_pad_sizes = []
        for krnl in conv_kernel_sizes:
            conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]


        # conv num
        self.encoder_num_conv_per_stage = num_conv_per_stage
        self.decoder_num_conv_per_stage = num_conv_per_stage


        # build layers
        self.down_layers = nn.ModuleList()
        for i_layer in range(self.num_pool):  # 0,1,2,3
            layer = BasicLayer(num_stage=i_layer,
                               num_only_conv_stage=num_only_conv_stage,
                               num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),
                               # 56,28,14,7
                               depth=depths[i_layer - num_only_conv_stage] if (i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer - num_only_conv_stage] if (i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels,
                               num_conv_per_stage=self.encoder_num_conv_per_stage,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                               dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes,
                               pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer - num_only_conv_stage]):sum(
                                   depths[:i_layer - num_only_conv_stage + 1])] if (
                                       i_layer >= num_only_conv_stage) else None,
                               norm_layer=norm_layer,
                               down_or_upsample=nn.Conv3d,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               use_checkpoint=use_checkpoint,
                               is_encoder=True)
            self.down_layers.append(layer)

        # only convs
        self.up_layers = nn.ModuleList()
        for i_layer in range(self.num_pool + 1)[::-1]:
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=6, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                       img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),
                               depth=depths[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels,
                               num_conv_per_stage=self.decoder_num_conv_per_stage ,
                               conv_op=self.conv_op, norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
                               dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes,
                               pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer - num_only_conv_stage]):sum(
                                   depths[:i_layer - num_only_conv_stage + 1])] if (
                                       i_layer >= num_only_conv_stage) else None,
                               norm_layer=norm_layer,
                               down_or_upsample=nn.ConvTranspose3d if (
                                       i_layer > 0) else None,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               num_classes=self.num_classes,
                               use_checkpoint=use_checkpoint,
                               is_encoder=False,
                               is_seg_decoder=True,
                               )
            self.up_layers.append(layer)

        self.up_bdr_layers = nn.ModuleList()
        for i_layer in range(self.num_pool + 1)[::-1]:
            layer = BasicLayer(num_stage=i_layer, num_only_conv_stage=6, num_pool=self.num_pool,
                               base_num_features=base_num_features,
                               input_resolution=(
                                       img_size // np.prod(pool_op_kernel_sizes[:i_layer], 0, dtype=np.int64)),
                               depth=depths[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               num_heads=num_heads[i_layer - num_only_conv_stage] if (
                                       i_layer >= num_only_conv_stage) else None,
                               window_size=window_size,
                               image_channels=image_channels,
                               num_conv_per_stage=self.decoder_num_conv_per_stage,
                               conv_op=self.conv_op,
                               norm_op=norm_op,
                               norm_op_kwargs=norm_op_kwargs,
                               dropout_op=dropout_op,
                               dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                               conv_kernel_sizes=conv_kernel_sizes, conv_pad_sizes=conv_pad_sizes,
                               pool_op_kernel_sizes=pool_op_kernel_sizes,
                               max_num_features=max_num_features,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer - num_only_conv_stage]):sum(
                                   depths[:i_layer - num_only_conv_stage + 1])] if (
                                       i_layer >= num_only_conv_stage) else None,
                               norm_layer=norm_layer,
                               down_or_upsample=nn.ConvTranspose3d if (
                                       i_layer > 0) else None,
                               feat_map_mul_on_downscale=feat_map_mul_on_downscale,
                               num_classes=self.num_classes,
                               use_checkpoint=use_checkpoint,
                               is_encoder=False)
            self.up_bdr_layers.append(layer)

        self.get_bdr_outputs = None
        self.apply(self._InitWeights)

    def _InitWeights(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
                module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=.02)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        x_skip = []
        for layer in self.down_layers:
            s, x = layer(x, None)
            x_skip.append(s)

        out = []
        bdr_out = []

        bdr_x = x
        seg_x = x

        for inx, layer in enumerate(self.up_layers):
            bdr_layer = self.up_bdr_layers[inx]
            # print('idx, bdr_x, seg_x', inx, bdr_x.shape, seg_x.shape)

            if inx > 0:
                # first concat and conv and upsample
                bdr_x_new, bdr_ds = bdr_layer(bdr_x, x_skip[self.num_pool - inx])
                seg_x_new, ds = layer(seg_x, bdr_x)

            else:
                # conv and upsample
                bdr_x_new, bdr_ds = bdr_layer(bdr_x, None)
                seg_x_new, ds = layer(seg_x, None)

            bdr_x = bdr_x_new
            seg_x = seg_x_new


            if inx > 0:
                out.append(ds)
                bdr_out.append(bdr_ds)

        if self.do_ds:
            self.get_bdr_outputs = bdr_out[::-1]
            return out[::-1]
        else:
            self.get_bdr_outputs = bdr_out[-1]
            return out[-1]

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    patch_size = [80, 160, 160]
    window_size = [5, 5, 5]

    # patch_size = [32, 64, 64]
    # window_size = [2, 2, 2]

    num_input_channels = 1
    num_classes = 2
    net_num_pool_op_kernel_sizes = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
    net_conv_kernel_sizes = [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]

    # 43.01M
    network = BGHNet(img_size=patch_size,
                          base_num_features=24,
                          image_channels=num_input_channels,
                          num_classes=num_classes,
                          num_pool=len(net_num_pool_op_kernel_sizes),
                          pool_op_kernel_sizes=net_num_pool_op_kernel_sizes,
                          conv_kernel_sizes=net_conv_kernel_sizes,
                          num_only_conv_stage=3,
                          num_conv_per_stage=2,
                          deep_supervision=True,
                          max_num_features=24 * 13,
                          depths=[2, 2, 2, 2],
                          num_heads=[6, 12, 24, 48],
                          window_size=window_size,
                          drop_path_rate=0)

    print(network)


    from lib.model.layers.utils import count_param
    param = count_param(network)
    # param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    # # net.eval()
    # with torch.no_grad():
    #     input_tensor = torch.rand(1, 1, patch_size[0], patch_size[1], patch_size[2])
    #
    #
    #     # from fvcore.nn import FlopCountAnalysis, flop_count_table
    #     # flop_analyzer = FlopCountAnalysis(network, input_tensor)
    #     # print(flop_count_table(flop_analyzer))
    #
    #     # Param and FLOPs
    #     from thop import profile
    #
    #     flops, params = profile(network, inputs=(input_tensor,))
    #     print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    #     print('Params = ' + str(params / 1000 ** 2) + 'M')
    #
    #     seg_output = network(input_tensor)
    #     for current_out in seg_output:
    #             print('seg_out')
    #             print(current_out.shape)
    #
    #     for current_out in network.get_bdr_outputs:
    #         print('bdr_out')
    #         print(current_out.shape)
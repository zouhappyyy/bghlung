#! /usr/bin/env python
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # 
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Time: 23-8-19
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

from typing import Tuple, Union
import torch
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.network_architecture.lung_tumor.dynunet_block import UnetOutBlock, UnetResBlock
from nnunet.network_architecture.lung_tumor.model_components import UnetrPPEncoder, UnetrUpBlock
from nnunet.network_architecture.neural_network import SegmentationNetwork
from torch import nn


class BADBTrV1(SegmentationNetwork):
    """
    boundary-guided dual-branch transformer
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: Tuple[int, int, int],
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        self.img_size = img_size
        self.final_nonlin = softmax_helper

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (self.img_size[0] // 16, self.img_size[1] // 32, self.img_size[1] // 32,)

        self.hidden_size = hidden_size
        # 80*160*160: [40*40*40, 20*20*20, 10*10*10, 5*5*5]
        self.unetr_pp_encoder = UnetrPPEncoder(
            input_size=[self.img_size[0] // 2 * self.img_size[1] // 4 * self.img_size[2] // 4,
                        self.img_size[0] // 4 * self.img_size[1] // 8 * self.img_size[2] // 8,
                        self.img_size[0] // 8 * self.img_size[1] // 16 * self.img_size[2] // 16,
                        self.img_size[0] // 16 * self.img_size[1] // 32 * self.img_size[2] // 32],
            dims=dims,
            depths=depths,
            num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        # seg results
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.img_size[0] // 8 * self.img_size[1] // 16 * self.img_size[2] // 16,
            conv_decoder=True,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.img_size[0] // 4 * self.img_size[1] // 8 * self.img_size[2] // 8,
            conv_decoder=True,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.img_size[0]//2 * self.img_size[1] // 4 * self.img_size[2] // 4,
            conv_decoder=True,
        )
        #  32*192*192 -> 80*160*160
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=self.img_size[0] * self.img_size[1] * self.img_size[2],
            conv_decoder=True,
        )

        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)


        # boundary branch
        self.bdr_decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.img_size[0] // 8 * self.img_size[1] // 16 * self.img_size[2] // 16,
            conv_decoder=True
        )
        self.bdr_decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.img_size[0] // 4 * self.img_size[1] // 8 * self.img_size[2] // 8,
            conv_decoder=True
        )
        self.bdr_decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=self.img_size[0] // 2 * self.img_size[1] // 4 * self.img_size[2] // 4,
            conv_decoder=True
        )
        #  32*192*192 -> 80*160*160
        self.bdr_decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(2, 4, 4),
            norm_name=norm_name,
            out_size=self.img_size[0] * self.img_size[1] * self.img_size[2],
            conv_decoder=True,
        )

        self.bdr_out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)

        self.get_bdr_outputs = None

        # if self.do_ds:
        #     self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
        #     self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        # print("#####input_shape:", x_in.shape)

        x_output, hidden_states = self.unetr_pp_encoder(x_in)
        # assert False
        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        # print("ENC1:",enc1.shape)
        enc2 = hidden_states[1]
        # print("ENC2:",enc2.shape)
        enc3 = hidden_states[2]
        # print("ENC3:",enc3.shape)
        enc4 = hidden_states[3]
        # print("ENC4:",enc4.shape)

        # Four decoders
        shared_dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)


        bdr_dec3 = self.bdr_decoder5(shared_dec4, enc3)
        bdr_dec2 = self.bdr_decoder4(bdr_dec3, enc2)
        bdr_dec1 = self.bdr_decoder3(bdr_dec2, enc1)
        bdr_out = self.bdr_decoder2(bdr_dec1, convBlock)


        seg_dec3 = self.decoder5(shared_dec4, bdr_dec3)
        seg_dec2 = self.decoder4(seg_dec3, bdr_dec2)
        seg_dec1 = self.decoder3(seg_dec2, bdr_dec1)
        seg_out = self.decoder2(seg_dec1, bdr_out)

        # seg_dec3 = self.decoder5(shared_dec4, enc3)
        # seg_dec2 = self.decoder4(seg_dec3, enc2)
        # seg_dec1 = self.decoder3(seg_dec2, enc1)
        # seg_out = self.decoder2(seg_dec1, convBlock)

        # if self.do_ds:
        #     logits = [self.final_nonlin(self.out1(seg_out)), self.final_nonlin(self.out2(seg_dec1)),self.final_nonlin(self.out3(seg_dec2))]
        # else:
        # if True:
        logits = self.out1(seg_out)

        self.get_bdr_outputs = self.bdr_out1(bdr_out)

        return logits


from lib.model.layers.utils import count_param

if __name__ == '__main__':
    crop_size = [80, 160, 160]
    input_channels = 1
    num_classes = 2
    do_ds = False

    net = BADBTrV1(in_channels=input_channels,
                   out_channels=num_classes,
                   img_size=crop_size,
                   feature_size=16,
                   num_heads=4,
                   depths=[2, 2, 3, 3],
                   dims=[32, 64, 128, 256],
                   do_ds=do_ds,
                   )

    # print(net)
    param = count_param(net)
    # param = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('net totoal parameters: %.2fM (%d)' % (param / 1e6, param))

    # net.eval()
    with torch.no_grad():
        input_tensor = torch.rand(1, 1, crop_size[0], crop_size[1], crop_size[2])

        # Param and FLOPs
        from thop import profile

        flops, params = profile(net, inputs=(input_tensor,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

        seg_output = net(input_tensor)
        print(seg_output.shape)
